from __future__ import annotations

# system imports
import asyncio
import collections
import copy
import dataclasses
import enum
import gettext
import logging
import operator
import typing
import uuid

# local libraries
from nion.instrumentation import Acquisition
from nion.instrumentation import camera_base
from nion.instrumentation import DataChannel
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift.model import DataItem
from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UserInterface
from nion.ui import UserInterface as UserInterfaceModule
from nion.utils import Binding
from nion.utils import Converter
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import ListModel
from nion.utils import Model
from nion.utils.ReferenceCounting import weak_partial
from nion.utils import Registry

from . import HardwareSourceChoice

_ = gettext.gettext


class SequenceState(enum.Enum):
    idle = 0
    scanning = 1


class ScanSpecifier:

    def __init__(self) -> None:
        self.scan_context: typing.Optional[stem_controller.ScanContext] = None
        self.scan_count = 1
        self.size: typing.Optional[typing.Tuple[int, int]] = None
        self.drift_interval_lines = 0
        self.drift_interval_scans = 0


@dataclasses.dataclass
class ScanProcessing:
    include_raw: bool
    include_summed: bool


ProcessingOption = collections.namedtuple("ProcessingOption", ["processing_id", "display_name"])


class ScanAcquisitionProcessing(enum.Enum):
    NONE = ProcessingOption(None,  _("Images"))
    SUM_PROJECT = ProcessingOption("sum_project", _("Spectra"))
    SUM_MASKED = ProcessingOption("sum_masked", _("Virtual Detectors"))


class ScanAcquisitionController:

    def __init__(self, api: typing.Any, document_controller: Facade.DocumentWindow, scan_hardware_source: Facade.HardwareSource, camera_hardware_source: Facade.HardwareSource, scan_specifier: ScanSpecifier) -> None:
        self.__api = api
        self.__document_controller = document_controller
        self.__scan_hardware_source = scan_hardware_source
        self.__camera_hardware_source = camera_hardware_source
        self.__scan_specifier = copy.deepcopy(scan_specifier)
        self.__scan_result_data_stream = typing.cast(Acquisition.FramedDataStream, None)
        self.__scan_acquisition = typing.cast(Acquisition.Acquisition, None)
        self.__scan_drift_logger: typing.Optional[DriftTracker.DriftLogger] = None
        self.acquisition_state_changed_event = Event.Event()

    def start(self, processing: ScanAcquisitionProcessing, scan_processing: ScanProcessing, *, section_height_override: typing.Optional[int] = None) -> None:
        assert scan_processing.include_raw or scan_processing.include_summed

        document_window = self.__document_controller

        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source._hardware_source)

        scan_frame_parameters = scan_hardware_source.get_frame_parameters(2)

        scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters, self.__scan_specifier.size)

        scan_frame_parameters.scan_id = uuid.uuid4()

        # useful code for testing to exit cleanly at this point.
        # self.acquisition_state_changed_event.fire(SequenceState.scanning)
        # self.acquisition_state_changed_event.fire(SequenceState.idle)
        # return

        camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, self.__camera_hardware_source._hardware_source)

        camera_frame_parameters = camera_hardware_source.get_frame_parameters(0)

        camera_frame_parameters.processing = processing.value.processing_id

        channel_names: typing.Dict[Acquisition.Channel, str] = dict()
        for c in scan_hardware_source.get_enabled_channels():
            channel_state = scan_hardware_source.get_channel_state(c)
            channel_index_segment = str(scan_hardware_source.get_channel_index(channel_state.channel_id))
            channel_names[Acquisition.Channel(scan_hardware_source.hardware_source_id, channel_index_segment)] = channel_state.name
        channel_names[Acquisition.Channel(camera_hardware_source.hardware_source_id)] = camera_hardware_source.get_signal_name(camera_frame_parameters)

        document_model = document_window.library._document_model
        event_loop = document_window._document_window.event_loop

        def display_data_item(document_controller: DocumentController.DocumentController, data_item: DataItem.DataItem) -> None:
            Facade.DocumentWindow(document_controller).display_data_item(Facade.DataItem(data_item))

        data_item_data_channel = DataChannel.DataItemDataChannel(document_model, _("Spectrum Image"), channel_names)
        data_item_data_channel.on_display_data_item = weak_partial(display_data_item, self.__document_controller._document_controller)

        drift_correction_functor: typing.Optional[Acquisition.DataStreamFunctor] = None
        section_height = section_height_override
        drift_tracker = scan_hardware_source.drift_tracker
        if drift_tracker and (self.__scan_specifier.drift_interval_lines > 0 or self.__scan_specifier.drift_interval_scans > 0):
            drift_correction_functor = DriftTracker.DriftCorrectionDataStreamFunctor(scan_hardware_source, scan_frame_parameters, drift_tracker, self.__scan_specifier.drift_interval_scans)
            if self.__scan_specifier.drift_interval_lines > 0:
                section_height = self.__scan_specifier.drift_interval_lines
        enable_drift_tracker = False

        synchronized_scan_data_stream = scan_base.make_synchronized_scan_data_stream(
            scan_hardware_source=scan_hardware_source,
            scan_frame_parameters=scan_frame_parameters,
            camera_hardware_source=camera_hardware_source,
            camera_frame_parameters=camera_frame_parameters,
            scan_data_stream_functor=drift_correction_functor,
            section_height=section_height,
            scan_count=self.__scan_specifier.scan_count,
            include_raw=scan_processing.include_raw,
            include_summed=scan_processing.include_summed,
            enable_drift_tracker=enable_drift_tracker
        )
        self.__scan_result_data_stream = Acquisition.FramedDataStream(synchronized_scan_data_stream, data_channel=data_item_data_channel).add_ref()
        self.__scan_acquisition = Acquisition.Acquisition(self.__scan_result_data_stream)
        drift_tracker = scan_hardware_source.drift_tracker
        if drift_tracker:
            self.__scan_drift_logger = DriftTracker.DriftLogger(document_model, drift_tracker, event_loop)

        def finish_grab_async() -> None:
            self.acquisition_state_changed_event.fire(SequenceState.idle)
            self.__scan_acquisition.close()
            self.__scan_acquisition = typing.cast(typing.Any, None)
            self.__scan_result_data_stream.remove_ref()
            self.__scan_result_data_stream = typing.cast(typing.Any, None)
            if self.__scan_drift_logger:
                self.__scan_drift_logger.close()
                self.__scan_drift_logger = None

        self.acquisition_state_changed_event.fire(SequenceState.scanning)

        self.__scan_acquisition.acquire_async(event_loop=event_loop, on_completion=finish_grab_async)

    def cancel(self) -> None:
        logging.debug("abort sequence acquisition")
        self.__scan_acquisition.abort_acquire()

    @property
    def progress(self) -> float:
        return self.__scan_acquisition.progress

    # for running tests
    def _wait(self, timeout: float = 60.0) -> None:
        self.__scan_acquisition.wait_acquire(timeout, on_periodic=self.__document_controller._document_window.periodic)


# see http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)


def calculate_time_size(camera_hardware_source: camera_base.CameraHardwareSource, scan_count: int, scan_pixels: int, camera_width: int, camera_height: int, is_summed: bool, exposure_time: float) -> typing.Tuple[str, str]:
    acquire_pixel_count = scan_pixels
    storage_pixel_count = scan_pixels
    camera_frame_parameters = camera_hardware_source.get_frame_parameters(0).as_dict()
    camera_frame_parameters["acquisition_frame_count"] = acquire_pixel_count
    camera_frame_parameters["storage_frame_count"] = storage_pixel_count
    if is_summed:
        camera_frame_parameters["processing"] = "sum_project"
        storage_memory = storage_pixel_count * camera_width * 4
    else:
        storage_memory = storage_pixel_count * camera_height * camera_width * 4
    acquire_sequence_metrics = camera_hardware_source.get_acquire_sequence_metrics(camera_base.CameraFrameParameters(camera_frame_parameters))
    acquisition_time = acquire_sequence_metrics.get("acquisition_time", exposure_time * acquire_pixel_count) * scan_count  # in seconds
    acquisition_memory = acquire_sequence_metrics.get("acquisition_memory", acquire_pixel_count * camera_width * camera_height * 4)  # in bytes
    storage_memory = acquire_sequence_metrics.get("storage_memory", storage_memory)  # in bytes
    if acquisition_time > 3600:
        time_str = "{0:.1f} hours".format(int(acquisition_time) / 3600)
    elif acquisition_time > 90:
        time_str = "{0:.1f} minutes".format(int(acquisition_time) / 60)
    else:
        time_str = "{} seconds".format(int(acquisition_time))
    if acquisition_memory != 0 and abs(storage_memory / acquisition_memory - 1) > 0.1:
        size_str = "{} ({})".format(sizeof_fmt(acquisition_memory), sizeof_fmt(storage_memory))
    else:
        size_str = sizeof_fmt(storage_memory)
    return time_str, size_str


class PanelDelegate:

    def __init__(self, api: typing.Any) -> None:
        self.__api = api
        self.panel_id = "scan-acquisition-panel"
        self.panel_name = _("Spectrum Imaging / 4d Scan Acquisition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.__scan_acquisition_controller: typing.Optional[ScanAcquisitionController] = None
        self.__acquisition_state = SequenceState.idle
        self.__acquisition_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__line_scan_acquisition_controller = None
        self.__eels_frame_parameters_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__camera_hardware_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__scan_hardware_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__exposure_time_ms_value_model: typing.Optional[Model.PropertyModel[float]] = None
        self.__scan_hardware_source_choice_model: typing.Optional[Model.PropertyModel[str]] = None
        self.__scan_hardware_source_choice: typing.Optional[HardwareSourceChoice.HardwareSourceChoice] = None
        self.__camera_hardware_source_choice_model: typing.Optional[Model.PropertyModel[str]] = None
        self.__camera_hardware_source_choice: typing.Optional[HardwareSourceChoice.HardwareSourceChoice] = None
        self.__styles_list_model: typing.Optional[ListModel.ListModel[ScanAcquisitionProcessing]] = None
        self.__styles_list_property_model: typing.Optional[ListModel.ListPropertyModel] = None
        self.__scan_count = 1
        self.__camera_width = 0
        self.__camera_height = 0
        self.__scan_specifier = ScanSpecifier()
        self.__scan_width = 32  # the width/length of the scan in pixels
        self.__scan_pixels = 0  # the total number of scan pixels
        self.__progress_task: typing.Optional[asyncio.Task[None]] = None
        self.__style_combo_box: typing.Optional[Facade.ComboBoxWidget] = None

    def create_panel_widget(self, ui: Facade.UserInterface, document_controller: Facade.DocumentWindow) -> Facade.ColumnWidget:
        stem_controller_ = typing.cast(stem_controller.STEMController, Registry.get_component("stem_controller"))

        self.__scan_hardware_source_choice_model = ui._ui.create_persistent_string_model("scan_acquisition_hardware_source_id")
        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__scan_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_scanning", False))
        self.__camera_hardware_source_choice_model = ui._ui.create_persistent_string_model("scan_acquisition_camera_hardware_source_id")
        self.__camera_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__camera_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_camera", False))

        self.__scan_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self.__scan_hardware_source_choice).add_ref()
        self.__camera_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self.__camera_hardware_source_choice).add_ref()

        def clear_scan_context_fields() -> None:
            self.__roi_description.text = _("Scan context not active")
            self.__scan_label_widget.text = None
            self.__scan_specifier.scan_context = stem_controller.ScanContext()
            self.__scan_specifier.scan_count = 1
            self.__scan_specifier.size = None
            self.__scan_specifier.drift_interval_lines = 0
            self.__scan_specifier.drift_interval_scans = 0
            self.__acquire_button._widget.enabled = self.__acquisition_state == SequenceState.scanning  # focus will be on the SI data, so enable if scanning
            self.__scan_pixels = 0

        def update_context() -> None:
            assert self.__scan_hardware_source_choice
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source_choice.hardware_source)
            if not scan_hardware_source:
                clear_scan_context_fields()
                return

            scan_context = scan_hardware_source.scan_context

            scan_context_size = scan_context.size
            exposure_ms = self.__exposure_time_ms_value_model.value or 0.0 if self.__exposure_time_ms_value_model else 0.0
            if scan_context.is_valid and scan_hardware_source.line_scan_enabled and scan_hardware_source.line_scan_vector:
                assert scan_context_size
                calibration = scan_context.calibration
                start = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[0])
                end = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[1])
                length = int(Geometry.distance(start, end) * scan_context_size.height)
                max_dim = max(scan_context_size.width, scan_context_size.height)
                length_str = calibration.convert_to_calibrated_size_str(length, value_range=(0, max_dim), samples=max_dim)
                line_str = _("Line Scan")
                self.__roi_description.text = f"{line_str} {length_str} ({length} px)"
                scan_str = _("Scan (1D)")
                scan_length = max(self.__scan_width, 1)
                drift_scans = scan_hardware_source.calculate_drift_scans()
                drift_str = f" / Drift every {drift_scans} scans" if drift_scans > 0 else str()
                self.__scan_label_widget.text = f"{scan_str} {scan_length} px" + drift_str
                self.__scan_pixels = scan_length
                self.__scan_specifier.scan_context = copy.deepcopy(scan_context)
                self.__scan_specifier.scan_count = max(self.__scan_count, 1)
                self.__scan_specifier.size = 1, scan_length
                self.__scan_specifier.drift_interval_lines = 0
                self.__scan_specifier.drift_interval_scans = drift_scans
                self.__acquire_button._widget.enabled = True
            elif scan_context.is_valid and scan_hardware_source.subscan_enabled and scan_hardware_source.subscan_region:
                assert scan_context_size
                calibration = scan_context.calibration
                width = scan_hardware_source.subscan_region.width * scan_context_size.width
                height = scan_hardware_source.subscan_region.height * scan_context_size.height
                width_str = calibration.convert_to_calibrated_size_str(width, value_range=(0, scan_context_size.width), samples=scan_context_size.width)
                height_str = calibration.convert_to_calibrated_size_str(height, value_range=(0, scan_context_size.height), samples=scan_context_size.height)
                rect_str = _("Subscan")
                self.__roi_description.text = f"{rect_str} {width_str} x {height_str} ({int(width)} px x {int(height)} px)"
                scan_str = _("Scan (2D)")
                scan_width = self.__scan_width
                scan_height = int(self.__scan_width * height / width)
                drift_lines = scan_hardware_source.calculate_drift_lines(scan_width, exposure_ms / 1000) if scan_hardware_source else 0
                drift_str = f" / Drift every {drift_lines} lines" if drift_lines > 0 else str()
                drift_scans = scan_hardware_source.calculate_drift_scans()
                drift_str = f" / Drift every {drift_scans} scans" if drift_scans > 0 else drift_str
                self.__scan_label_widget.text = f"{scan_str} {scan_width} x {scan_height} px" + drift_str
                self.__scan_pixels = scan_width * scan_height
                self.__scan_specifier.scan_context = copy.deepcopy(scan_context)
                self.__scan_specifier.scan_count = max(self.__scan_count, 1)
                self.__scan_specifier.size = scan_height, scan_width
                self.__scan_specifier.drift_interval_lines = drift_lines
                self.__scan_specifier.drift_interval_scans = drift_scans
                self.__acquire_button._widget.enabled = True
            elif scan_context.is_valid:
                assert scan_context_size
                calibration = scan_context.calibration
                width = scan_context_size.width
                height = scan_context_size.height
                width_str = calibration.convert_to_calibrated_size_str(width, value_range=(0, scan_context_size.width), samples=scan_context_size.width)
                height_str = calibration.convert_to_calibrated_size_str(height, value_range=(0, scan_context_size.height), samples=scan_context_size.height)
                data_str = _("Context Scan")
                self.__roi_description.text = f"{data_str} {width_str} x {height_str} ({int(width)} x {int(height)})"
                scan_str = _("Scan (2D)")
                scan_width = self.__scan_width
                scan_height = int(self.__scan_width * height / width)
                drift_lines = scan_hardware_source.calculate_drift_lines(scan_width, exposure_ms / 1000) if scan_hardware_source else 0
                drift_str = f" / Drift every {drift_lines} lines" if drift_lines > 0 else str()
                drift_scans = scan_hardware_source.calculate_drift_scans()
                drift_str = f" / Drift every {drift_scans} scans" if drift_scans > 0 else drift_str
                self.__scan_label_widget.text = f"{scan_str} {scan_width} x {scan_height} px" + drift_str
                self.__scan_pixels = scan_width * scan_height
                self.__scan_specifier.scan_context = copy.deepcopy(scan_context)
                self.__scan_specifier.scan_count = max(self.__scan_count, 1)
                self.__scan_specifier.size = scan_height, scan_width
                self.__scan_specifier.drift_interval_lines = drift_lines
                self.__scan_specifier.drift_interval_scans = drift_scans
                self.__acquire_button._widget.enabled = True
            else:
                clear_scan_context_fields()

            self.__scan_count_widget.text = Converter.IntegerToStringConverter().convert(self.__scan_count)

            self.__scan_width_widget.text = Converter.IntegerToStringConverter().convert(self.__scan_width)

            self.__update_estimate()

        def stem_controller_property_changed(key: str) -> None:
            if key in ("subscan_state", "subscan_region", "subscan_rotation", "line_scan_state", "line_scan_vector", "drift_channel_id", "drift_region", "drift_settings"):
                document_controller._document_controller.event_loop.call_soon_threadsafe(update_context)

        def scan_context_changed() -> None:
            # this can be triggered from a thread, so use call soon to transfer it to the UI thread.
            document_controller._document_controller.event_loop.call_soon_threadsafe(update_context)

        self.__stem_controller_property_listener = None
        self.__scan_context_changed_listener = None

        if stem_controller_:
            self.__stem_controller_property_listener = stem_controller_.property_changed_event.listen(stem_controller_property_changed)
            self.__scan_context_changed_listener = stem_controller_.scan_context_changed_event.listen(scan_context_changed)

        column = ui.create_column_widget()

        self.__styles_list_model = ListModel.ListModel[ScanAcquisitionProcessing](items=[ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisitionProcessing.NONE])
        self.__styles_list_property_model = ListModel.ListPropertyModel(self.__styles_list_model)
        self.__style_combo_box = ui.create_combo_box_widget(self.__styles_list_property_model.value, item_text_getter=operator.attrgetter("value.display_name"))
        self.__style_combo_box._widget.set_property("min-width", 100)
        items_binding = Binding.PropertyBinding(self.__styles_list_property_model, "value")
        items_binding.source_setter = None
        typing.cast(UserInterfaceModule.ComboBoxWidget, self.__style_combo_box._widget).bind_items(items_binding)
        self.__style_combo_box.current_index = 0

        self.__acquire_button = ui.create_push_button_widget(_("Acquire"))

        self.__progress_bar = ui.create_progress_bar_widget()
        # self.__progress_bar.enabled = False

        self.__roi_description = ui.create_label_widget()

        self.__scan_count_widget = ui.create_line_edit_widget()
        self.__scan_count_widget._widget.set_property("width", 72)

        self.__scan_processing_widget = ui.create_combo_box_widget(items=["Raw", "Sum", "Raw + Sum"])

        self.__scan_width_widget = ui.create_line_edit_widget()

        self.__exposure_time_widget = ui.create_line_edit_widget()

        self.__estimate_label_widget = ui.create_label_widget()

        self.__scan_label_widget = ui.create_label_widget()

        class ComboBoxWidget:
            def __init__(self, widget: UserInterfaceModule.ComboBoxWidget) -> None:
                self.__combo_box_widget = widget

            @property
            def _widget(self) -> UserInterfaceModule.ComboBoxWidget:
                return self.__combo_box_widget

        camera_row = ui.create_row_widget()
        camera_row.add_spacing(12)
        camera_row.add(ComboBoxWidget(self.__camera_hardware_source_choice.create_combo_box(ui._ui)))
        camera_row.add_spacing(12)
        camera_row.add(self.__style_combo_box)
        camera_row.add_spacing(12)
        camera_row.add_stretch()

        scan_choice_row = ui.create_row_widget()
        scan_choice_row.add_spacing(12)
        scan_choice_row.add(ComboBoxWidget(self.__scan_hardware_source_choice.create_combo_box(ui._ui)))
        scan_choice_row.add_spacing(12)
        scan_choice_row.add_stretch()

        scan_count_row = ui.create_row_widget()
        scan_count_row.add_spacing(12)
        scan_count_row.add(ui.create_label_widget("Scan Count"))
        scan_count_row.add_spacing(12)
        scan_count_row.add(self.__scan_count_widget)
        scan_count_row.add_spacing(12)
        scan_count_row.add(self.__scan_processing_widget)
        scan_count_row.add_spacing(12)
        scan_count_row.add_stretch()

        roi_size_row = ui.create_row_widget()
        roi_size_row.add_spacing(12)
        roi_size_row.add(self.__roi_description)
        roi_size_row.add_spacing(12)
        roi_size_row.add_stretch()

        scan_spacing_pixels_row = ui.create_row_widget()
        scan_spacing_pixels_row.add_spacing(12)
        scan_spacing_pixels_row.add(ui.create_label_widget("Scan Width (pixels)"))
        scan_spacing_pixels_row.add_spacing(12)
        scan_spacing_pixels_row.add(self.__scan_width_widget)
        scan_spacing_pixels_row.add_spacing(12)
        scan_spacing_pixels_row.add_stretch()

        eels_exposure_row = ui.create_row_widget()
        eels_exposure_row.add_spacing(12)
        eels_exposure_row.add(ui.create_label_widget("Camera Exposure Time (ms)"))
        eels_exposure_row.add_spacing(12)
        eels_exposure_row.add(self.__exposure_time_widget)
        eels_exposure_row.add_spacing(12)
        eels_exposure_row.add_stretch()

        scan_row = ui.create_row_widget()
        scan_row.add_spacing(12)
        scan_row.add(self.__scan_label_widget)
        scan_row.add_stretch()

        estimate_row = ui.create_row_widget()
        estimate_row.add_spacing(12)
        estimate_row.add(self.__estimate_label_widget)
        estimate_row.add_stretch()

        acquire_sequence_button_row = ui.create_row_widget()
        acquire_sequence_button_row.add(self.__acquire_button)
        acquire_sequence_button_row.add_spacing(8)
        acquire_sequence_button_row.add(self.__progress_bar)
        acquire_sequence_button_row.add_spacing(8)

        if self.__scan_hardware_source_choice.hardware_source_count > 1:
            column.add_spacing(8)
            column.add(scan_choice_row)
        column.add_spacing(8)
        column.add(camera_row)
        column.add_spacing(8)
        column.add(scan_count_row)
        column.add_spacing(8)
        column.add(roi_size_row)
        column.add_spacing(8)
        column.add(scan_spacing_pixels_row)
        column.add_spacing(8)
        column.add(eels_exposure_row)
        column.add_spacing(8)
        column.add(scan_row)
        column.add_spacing(8)
        column.add(estimate_row)
        column.add_spacing(8)
        column.add(acquire_sequence_button_row)
        column.add_spacing(8)
        column.add_stretch()

        def camera_hardware_source_changed(hardware_source: typing.Optional[HardwareSource.HardwareSource]) -> None:
            styles_list_model = self.__styles_list_model
            self.disconnect_camera_hardware_source()
            if hardware_source and styles_list_model:
                self.connect_camera_hardware_source(hardware_source)
                if hardware_source.features.get("has_masked_sum_option"):
                    styles_list_model.items = [ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisitionProcessing.NONE, ScanAcquisitionProcessing.SUM_MASKED]
                else:
                    styles_list_model.items = [ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisitionProcessing.NONE]

        self.__camera_hardware_changed_event_listener = self.__camera_hardware_source_choice.hardware_source_changed_event.listen(camera_hardware_source_changed)
        camera_hardware_source_changed(self.__camera_hardware_source_choice.hardware_source)

        def style_current_item_changed(current_item: str) -> None:
            self.__update_estimate()

        self.__style_combo_box.on_current_item_changed = style_current_item_changed

        def scan_count_changed(text: str) -> None:
            scan_count = Converter.IntegerToStringConverter().convert_back(text) or 1
            scan_count = max(scan_count, 1)
            if scan_count != self.__scan_count:
                self.__scan_count = scan_count
                update_context()
            self.__scan_count_widget.request_refocus()

        self.__scan_count_widget.on_editing_finished = scan_count_changed

        def scan_width_changed(text: str) -> None:
            scan_width = Converter.IntegerToStringConverter().convert_back(text) or 1
            scan_width = max(scan_width, 1)
            if scan_width != self.__scan_width:
                self.__scan_width = scan_width
                update_context()
            self.__scan_width_widget.request_refocus()

        self.__scan_width_widget.on_editing_finished = scan_width_changed

        def acquisition_state_changed(acquisition_state: SequenceState) -> None:
            self.__acquisition_state = acquisition_state

            async def update_state(is_idle: bool) -> None:
                self.__acquire_button.text = _("Acquire") if is_idle else _("Cancel")
                # self.__progress_bar.enabled = not is_idle
                update_context()  # update the cancel button
                if is_idle and self.__progress_task:
                    self.__progress_task.cancel()
                    self.__progress_task = None
                    self.__progress_bar.value = 100
                if not is_idle and not self.__progress_task:
                    async def update_progress() -> None:
                        while True:
                            if self.__scan_acquisition_controller:
                                self.__progress_bar.value = int(100 * self.__scan_acquisition_controller.progress)
                            await asyncio.sleep(0.25)

                    self.__progress_task = document_controller._document_window.event_loop.create_task(update_progress())

            if acquisition_state == SequenceState.idle:
                self.__scan_acquisition_controller = None
                if self.__acquisition_state_changed_event_listener:
                    self.__acquisition_state_changed_event_listener.close()
                    self.__acquisition_state_changed_event_listener = None
                document_controller._document_window.event_loop.create_task(update_state(True))
            else:
                document_controller._document_window.event_loop.create_task(update_state(False))

        def acquire_sequence() -> None:
            if self.__scan_acquisition_controller:
                if self.__scan_acquisition_controller:
                    self.__scan_acquisition_controller.cancel()
            else:
                scan_hardware_source_choice = self.__scan_hardware_source_choice
                assert scan_hardware_source_choice
                if scan_hardware_source_choice.hardware_source:
                    scan_hardware_source = self.__api.get_hardware_source_by_id(
                        scan_hardware_source_choice.hardware_source.hardware_source_id, version="1.0")
                else:
                    scan_hardware_source = None

                camera_hardware_source_choice = self.__camera_hardware_source_choice
                assert camera_hardware_source_choice
                if camera_hardware_source_choice.hardware_source:
                    camera_hardware_source = self.__api.get_hardware_source_by_id(
                        camera_hardware_source_choice.hardware_source.hardware_source_id, version="1.0")
                else:
                    camera_hardware_source = None

                if scan_hardware_source and camera_hardware_source:
                    self.__scan_acquisition_controller = ScanAcquisitionController(self.__api, document_controller, scan_hardware_source, camera_hardware_source, self.__scan_specifier)
                    self.__acquisition_state_changed_event_listener = self.__scan_acquisition_controller.acquisition_state_changed_event.listen(acquisition_state_changed)
                    scan_processing = ScanProcessing(self.__scan_processing_widget.current_index in (0, 2), self.__scan_processing_widget.current_index in (1, 2))
                    scan_acquisition_processing = self.__style_combo_box.current_item if self.__style_combo_box and self.__style_combo_box.current_item else ScanAcquisitionProcessing.NONE
                    self.__scan_acquisition_controller.start(scan_acquisition_processing, scan_processing)

        self.__acquire_button.on_clicked = acquire_sequence

        self.__update_estimate()

        update_context()

        return column

    def __update_estimate(self) -> None:
        camera_hardware_source = self.__camera_hardware_source_choice.hardware_source if self.__camera_hardware_source_choice else None
        if self.__exposure_time_ms_value_model and isinstance(camera_hardware_source, camera_base.CameraHardwareSource):
            camera_width = self.__camera_width
            camera_height = self.__camera_height
            is_summed = self.__style_combo_box is not None and self.__style_combo_box.current_index == 0
            exposure_time = self.__exposure_time_ms_value_model.value / 1000 if self.__exposure_time_ms_value_model and self.__exposure_time_ms_value_model.value else 0.0
            time_str, size_str = calculate_time_size(camera_hardware_source, self.__scan_count, self.__scan_pixels, camera_width, camera_height, is_summed, exposure_time)
            self.__estimate_label_widget.text = "{0} / {1}".format(time_str, size_str)
        else:
            self.__estimate_label_widget.text = None

    def connect_camera_hardware_source(self, camera_hardware_source: HardwareSource.HardwareSource) -> None:
        assert isinstance(camera_hardware_source, camera_base.CameraHardwareSource)

        self.__exposure_time_ms_value_model = Model.PropertyModel[float]()

        def update_exposure_time_ms(exposure_time_ms: typing.Optional[float]) -> None:
            if exposure_time_ms is not None and exposure_time_ms > 0 and isinstance(camera_hardware_source, camera_base.CameraHardwareSource):
                frame_parameters = camera_hardware_source.get_frame_parameters(0)
                frame_parameters.exposure_ms = exposure_time_ms
                camera_hardware_source.set_frame_parameters(0, frame_parameters)
            self.__update_estimate()

        self.__exposure_time_ms_value_model.on_value_changed = update_exposure_time_ms

        exposure_time_ms_value_binding = Binding.PropertyBinding(self.__exposure_time_ms_value_model, "value", converter=Converter.FloatToStringConverter("{0:.1f}"))

        def eels_profile_parameters_changed(profile_index: int, frame_parameters: camera_base.CameraFrameParameters) -> None:
            if profile_index == 0 and isinstance(camera_hardware_source, camera_base.CameraHardwareSource):
                expected_dimensions = camera_hardware_source.get_expected_dimensions(frame_parameters.binning)
                self.__camera_width = expected_dimensions[1]
                self.__camera_height = expected_dimensions[0]
                if self.__exposure_time_ms_value_model:
                    self.__exposure_time_ms_value_model.value = frame_parameters.exposure_ms
                self.__update_estimate()

        self.__eels_frame_parameters_changed_event_listener = camera_hardware_source.frame_parameters_changed_event.listen(eels_profile_parameters_changed)

        eels_profile_parameters_changed(0, camera_hardware_source.get_frame_parameters(0))

        typing.cast(UserInterfaceModule.LineEditWidget, self.__exposure_time_widget._widget).bind_text(exposure_time_ms_value_binding)  # the widget will close the binding

    def disconnect_camera_hardware_source(self) -> None:
        typing.cast(UserInterfaceModule.LineEditWidget, self.__exposure_time_widget._widget).unbind_text()
        if self.__eels_frame_parameters_changed_event_listener:
            self.__eels_frame_parameters_changed_event_listener.close()
            self.__eels_frame_parameters_changed_event_listener = None
        if self.__exposure_time_ms_value_model:
            self.__exposure_time_ms_value_model.close()
            self.__exposure_time_ms_value_model = None

    def close(self) -> None:
        if self.__eels_frame_parameters_changed_event_listener:
            self.__eels_frame_parameters_changed_event_listener.close()
            self.__eels_frame_parameters_changed_event_listener = None
        if self.__camera_hardware_changed_event_listener:
            self.__camera_hardware_changed_event_listener.close()
            self.__camera_hardware_changed_event_listener = None
        if self.__scan_hardware_changed_event_listener:
            self.__scan_hardware_changed_event_listener.close()
            self.__scan_hardware_changed_event_listener = None
        if self.__scan_hardware_source_choice:
            self.__scan_hardware_source_choice.close()
            self.__scan_hardware_source_choice = None
        if self.__scan_hardware_source_choice_model:
            self.__scan_hardware_source_choice_model.close()
            self.__scan_hardware_source_choice_model = None
        if self.__camera_hardware_source_choice:
            self.__camera_hardware_source_choice.close()
            self.__camera_hardware_source_choice = None
        if self.__camera_hardware_source_choice_model:
            self.__camera_hardware_source_choice_model.close()
            self.__camera_hardware_source_choice_model = None
        if self.__stem_controller_property_listener:
            self.__stem_controller_property_listener.close()
            self.__stem_controller_property_listener = None
        if self.__scan_context_changed_listener:
            self.__scan_context_changed_listener.close()
            self.__scan_context_changed_listener = None
        if self.__scan_hardware_source_stream:
            self.__scan_hardware_source_stream.remove_ref()
        if self.__camera_hardware_source_stream:
            self.__camera_hardware_source_stream.remove_ref()
        if self.__styles_list_model:
            self.__styles_list_model.close()
            self.__styles_list_model = None
        if self.__styles_list_property_model:
            self.__styles_list_property_model.close()
            self.__styles_list_property_model = None


class ScanAcquisitionExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.instrumentation-kit.scan-acquisition"

    def __init__(self, api_broker: typing.Any) -> None:
        # grab the api object.
        api = api_broker.get_api(version=API.version, ui_version=UserInterface.version)
        # be sure to keep a reference or it will be closed immediately.
        self.__panel_ref = api.create_panel(PanelDelegate(api))

    def close(self) -> None:
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        # self.__menu_item_ref.close()
        # self.__menu_item_ref = None
        self.__panel_ref.close()
        self.__panel_ref = None
