from __future__ import annotations

# system imports
import copy
import enum
import functools
import gettext
import logging
import math
import numpy
import threading
import typing
import uuid
import collections
import operator

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.swift import Facade
from nion.swift import HistogramPanel
from nion.swift.model import DataItem
from nion.swift.model import DisplayItem
from nion.swift.model import Graphics
from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UserInterface
from nion.utils import Binding
from nion.utils import Converter
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Registry
from nion.utils import Stream
from nion.utils import ListModel
from . import HardwareSourceChoice

if typing.TYPE_CHECKING:
    from nion.swift.model import DocumentModel

_ = gettext.gettext

title_base = _("Spectrum Image")


def create_and_display_data_item(document_window, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
    # create the data item; large format if it's a collection
    data_item = Facade.DataItem(DataItem.DataItem(large_format=data_and_metadata.is_collection))
    document_window.library._document_model.append_data_item(data_item._data_item)

    # update the session id
    data_item._data_item.session_id = document_window.library._document_model.session_id

    # set the title
    channel_name = data_and_metadata.metadata.get("hardware_source", dict()).get("channel_name", data_and_metadata.metadata.get("hardware_source", dict()).get("hardware_source_name", "Data"))
    dimension_str = (" " + " x ".join([str(d) for d in data_and_metadata.collection_dimension_shape])) if data_and_metadata.is_collection else str()
    data_item.title = f"{title_base}{dimension_str} ({channel_name})"

    # if the last dimension is 1, squeeze the data (1D SI)
    if data_and_metadata.data_shape[0] == 1:
        data_and_metadata = xd.squeeze(data_and_metadata)

    # the data item should not have any other 'clients' at this point; so setting the
    # data and metadata will immediately unload the data (and write to disk). this is important,
    # because the data (up to this point) can be shared data from the DLL.
    data_item.set_data_and_metadata(data_and_metadata)

    # now to display it will reload the data (presumably from an HDF5 or similar on-demand format).
    document_window.display_data_item(data_item)


class SequenceState(enum.Enum):
    idle = 0
    scanning = 1


class ScanSpecifier:

    def __init__(self):
        self.scan_context: typing.Optional[stem_controller.ScanContext] = None
        self.size: typing.Optional[typing.Tuple[int, int]] = None
        self.drift_interval_lines = 0


class CameraDataChannel(scan_base.SynchronizedDataChannelInterface):
    def __init__(self, document_model, channel_name: str, grab_sync_info: scan_base.ScanHardwareSource.GrabSynchronizedInfo):
        self.__document_model = document_model
        self.__data_item = self.__create_data_item(channel_name, grab_sync_info)
        self.__data_item_transaction = None
        self.__data_and_metadata = None
        self.__grab_sync_info = grab_sync_info

    def __create_data_item(self, channel_name: str, grab_sync_info: scan_base.ScanHardwareSource.GrabSynchronizedInfo) -> DataItem.DataItem:
        scan_calibrations = grab_sync_info.scan_calibrations
        data_calibrations = grab_sync_info.data_calibrations
        data_intensity_calibration = grab_sync_info.data_intensity_calibration
        _, data_shape = self.__calculate_axes_order_and_data_shape(grab_sync_info.axes_descriptor, grab_sync_info.scan_size, grab_sync_info.camera_readout_size_squeezed)
        large_format = numpy.prod(data_shape) > 2048**2 * 10
        data_item = DataItem.DataItem(large_format=large_format)
        data_item.title = f"{title_base} ({channel_name})"
        self.__document_model.append_data_item(data_item)
        # Only call "reserve_data" for HDF5 backed data items because it causes problems for ndata backed items.
        if large_format:
            axes_descriptor = grab_sync_info.axes_descriptor
            is_sequence = axes_descriptor.sequence_axes is not None
            collection_dimension_count = len(axes_descriptor.collection_axes) if axes_descriptor.collection_axes is not None else 0
            datum_dimension_count = len(axes_descriptor.data_axes) if axes_descriptor.data_axes is not None else 0
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)
            data_item.reserve_data(data_shape=data_shape, data_dtype=numpy.float32, data_descriptor=data_descriptor)
        data_item.dimensional_calibrations = scan_calibrations + data_calibrations
        data_item.intensity_calibration = data_intensity_calibration
        data_item_metadata = data_item.metadata
        data_item_metadata["instrument"] = copy.deepcopy(grab_sync_info.instrument_metadata)
        data_item_metadata["hardware_source"] = copy.deepcopy(grab_sync_info.camera_metadata)
        data_item_metadata["scan"] = copy.deepcopy(grab_sync_info.scan_metadata)
        data_item.metadata = data_item_metadata
        return data_item

    @property
    def data_item(self) -> DataItem.DataItem:
        return self.__data_item

    def start(self) -> None:
        self.__data_item.increment_data_ref_count()
        self.__data_item_transaction = self.__document_model.item_transaction(self.__data_item)
        self.__document_model.begin_data_item_live(self.__data_item)

    def __calculate_axes_order_and_data_shape(self, axes_descriptor: scan_base.ScanHardwareSource.AxesDescriptor, scan_shape: Geometry.IntSize, camera_readout_size: typing.Tuple[int, ...]) -> typing.Tuple[typing.List[int], typing.Tuple[int, ...]]:
        # axes_descriptor provides the information needed to re-order the axes of the result data approprietly.
        axes_order = []
        if axes_descriptor.sequence_axes:
            axes_order.extend(axes_descriptor.sequence_axes)
        if axes_descriptor.collection_axes:
            axes_order.extend(axes_descriptor.collection_axes)
        if axes_descriptor.data_axes:
            axes_order.extend(axes_descriptor.data_axes)

        data_shape = tuple(scan_shape) + camera_readout_size

        assert len(axes_order) == len(data_shape)

        data_shape = numpy.array(data_shape)[axes_order].tolist()
        is_sequence = axes_descriptor.sequence_axes is not None
        if is_sequence and data_shape[0] == 1:
            data_shape = data_shape[1:]

        return axes_order, data_shape

    def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, scan_shape: Geometry.IntSize, dest_sub_area: Geometry.IntRect, sub_area: Geometry.IntRect, view_id) -> None:
        # there are a few techniques for getting data into a data item. this method prefers directly calling the
        # document model method update_data_item_partial, which is thread safe. if that method is not available, it
        # falls back to the data item method set_data_and_metadata, which must be called from the main thread.
        # the hardware source also supplies a data channel which is thread safe and ends up calling set_data_and_metadata
        # but we skip that so that the updates fit into this class instead.
        update_data_item_partial = getattr(self.__document_model, "update_data_item_partial", None)
        if callable(update_data_item_partial):
            axes_descriptor = self.__grab_sync_info.axes_descriptor
            # The low-level always produces a collection of 1d or 2d data. Re-oder axes and remove length-1-axes below.
            # Calibrations, data descriptor and shape estimates are updated accordingly.
            dimensional_calibrations = data_and_metadata.dimensional_calibrations
            data = data_and_metadata.data
            axes_order, data_shape = self.__calculate_axes_order_and_data_shape(axes_descriptor, scan_shape, data.shape[len(tuple(scan_shape)):])
            assert len(axes_order) == data.ndim
            data = numpy.moveaxis(data, axes_order, list(range(data.ndim)))
            dimensional_calibrations = numpy.array(dimensional_calibrations)[axes_order].tolist()
            is_sequence = axes_descriptor.sequence_axes is not None
            collection_dimension_count = len(axes_descriptor.collection_axes) if axes_descriptor.collection_axes is not None else 0
            datum_dimension_count = len(axes_descriptor.data_axes) if axes_descriptor.data_axes is not None else 0

            src_slice = tuple()
            dst_slice = tuple()
            for index in axes_order:
                if index >= len(sub_area.slice):
                    src_slice += (slice(None),)
                else:
                    src_slice += (sub_area.slice[index],)
                if index >= len(dest_sub_area.slice):
                    dst_slice += (slice(None),)
                else:
                    dst_slice += (dest_sub_area.slice[index],)

            if is_sequence and data.shape[0] == 1:
                data = numpy.squeeze(data, axis=0)
                dimensional_calibrations = dimensional_calibrations[1:]
                src_slice = src_slice[1:]
                dst_slice = dst_slice[1:]
                is_sequence = False
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

            data_and_metadata = DataAndMetadata.new_data_and_metadata(data, data_and_metadata.intensity_calibration,
                                                                      dimensional_calibrations,
                                                                      data_and_metadata.metadata, None,
                                                                      data_descriptor, None,
                                                                      None)
            data_metadata = DataAndMetadata.DataMetadata((tuple(data_shape), data_and_metadata.data_dtype),
                                                         data_and_metadata.intensity_calibration,
                                                         data_and_metadata.dimensional_calibrations,
                                                         metadata=data_and_metadata.metadata,
                                                         data_descriptor=data_descriptor)

            update_data_item_partial(self.__data_item, data_metadata, data_and_metadata, src_slice, dst_slice)
        elif state == "complete":
            # hack for Swift 0.14
            def update_data_item():
                self.__data_item.set_data_and_metadata(data_and_metadata)
            self.__document_model.call_soon_event.fire_any(update_data_item)

    def stop(self) -> None:
        if self.__data_item_transaction:
            self.__data_item_transaction.close()
            self.__data_item_transaction = None
            self.__document_model.end_data_item_live(self.__data_item)
            self.__data_item.decrement_data_ref_count()


class DriftCorrectionBehavior(scan_base.SynchronizedScanBehaviorInterface):
    def __init__(self, document_model: DocumentModel.DocumentModel, scan_hardware_source: scan_base.ScanHardwareSource, scan_frame_parameters: scan_base.ScanFrameParameters):
        # init with the frame parameters from the synchronized grab
        self.__document_model = document_model
        self.__scan_hardware_source = scan_hardware_source
        self.__scan_frame_parameters = copy.deepcopy(scan_frame_parameters)
        # here we convert those frame parameters to the context
        self.__scan_frame_parameters.subscan_pixel_size = None
        self.__scan_frame_parameters.subscan_fractional_size = None
        self.__scan_frame_parameters.subscan_fractional_center = None
        self.__scan_frame_parameters.subscan_rotation = 0.0
        self.__scan_frame_parameters.channel_override = "drift"
        self.__last_xdata = None
        self.__center_nm = Geometry.FloatSize()
        self.__last_offset_nm = Geometry.FloatSize()
        self.__offset_nm_data = numpy.zeros((3, 0), numpy.float)
        data_item = next(iter(data_item for data_item in document_model.data_items if data_item.title == "Drift Log"), None)
        if data_item:
            offset_nm_xdata = DataAndMetadata.new_data_and_metadata(self.__offset_nm_data, intensity_calibration=Calibration.Calibration(units="nm"))
            data_item.set_data_and_metadata(offset_nm_xdata)
        else:
            data_item = DataItem.DataItem(self.__offset_nm_data)
            data_item.title = f"Drift Log"
            self.__document_model.append_data_item(data_item)
            display_item = self.__document_model.get_display_item_for_data_item(data_item)
            display_item.display_type = "line_plot"
            display_item.append_display_data_channel_for_data_item(data_item)
            display_item.append_display_data_channel_for_data_item(data_item)
            display_item._set_display_layer_properties(0, label=_("x"))
            display_item._set_display_layer_properties(1, label=_("y"))
            display_item._set_display_layer_properties(2, label=_("m"))
        self.__data_item = data_item

    def reset(self) -> None:
        self.__last_xdata = None
        self.__center_nm = Geometry.FloatSize()
        self.__last_offset_nm = Geometry.FloatSize()

    def prepare_section(self) -> scan_base.SynchronizedScanBehaviorAdjustments:
        # this method must be thread safe
        # start with the context frame parameters and adjust for the drift region
        adjustments = scan_base.SynchronizedScanBehaviorAdjustments()
        frame_parameters = copy.deepcopy(self.__scan_frame_parameters)
        context_size = Geometry.FloatSize.make(frame_parameters.size)
        drift_channel_id = self.__scan_hardware_source.drift_channel_id
        drift_region = Geometry.FloatRect.make(self.__scan_hardware_source.drift_region)
        drift_rotation = self.__scan_hardware_source.drift_rotation
        if drift_channel_id is not None and drift_region is not None:
            drift_channel_index = self.__scan_hardware_source.get_channel_index(drift_channel_id)
            frame_parameters.subscan_pixel_size = int(context_size.height * drift_region.height * 4), int(context_size.width * drift_region.width * 4)
            if frame_parameters.subscan_pixel_size[0] >= 8 or frame_parameters.subscan_pixel_size[1] >= 8:
                frame_parameters.subscan_fractional_size = drift_region.height, drift_region.width
                frame_parameters.subscan_fractional_center = drift_region.center.y, drift_region.center.x
                frame_parameters.subscan_rotation = drift_rotation
                # attempt to keep drift area in roughly the same position by adding in the accumulated correction.
                frame_parameters.center_nm = tuple(Geometry.FloatPoint.make(frame_parameters.center_nm) + self.__center_nm)
                xdatas = self.__scan_hardware_source.record_immediate(frame_parameters, [drift_channel_index])
                # new_offset = self.__scan_hardware_source.stem_controller.drift_offset_m
                if self.__last_xdata:
                    # calculate offset. if data shifts down/right, offset will be negative (register_translation convention).
                    # offset = Geometry.FloatPoint.make(xd.register_translation(self.__last_xdata, xdatas[0], upsample_factor=10))
                    quality, offset = xd.register_template(self.__last_xdata, xdatas[0])
                    offset = Geometry.FloatPoint.make(offset)
                    offset_nm = Geometry.FloatSize(
                        h=xdatas[0].dimensional_calibrations[0].convert_to_calibrated_size(offset.y),
                        w=xdatas[0].dimensional_calibrations[1].convert_to_calibrated_size(offset.x))
                    # calculate adjustment (center_nm). if center_nm positive, data shifts up/left.
                    # rotate back into context reference frame
                    offset_nm = offset_nm.rotate(-drift_rotation)
                    offset_nm -= self.__center_nm  # adjust for center_nm adjustment above
                    delta_nm = offset_nm - self.__last_offset_nm
                    self.__last_offset_nm = offset_nm
                    offset_nm_xy = math.sqrt(pow(offset_nm.height, 2) + pow(offset_nm.width, 2))
                    self.__offset_nm_data = numpy.hstack([self.__offset_nm_data, numpy.array([offset_nm.height, offset_nm.width, offset_nm_xy]).reshape(3, 1)])
                    offset_nm_xdata = DataAndMetadata.new_data_and_metadata(self.__offset_nm_data, intensity_calibration=Calibration.Calibration(units="nm"))
                    def update_data_item(offset_nm_xdata: DataAndMetadata.DataAndMetadata) -> None:
                        self.__data_item.set_data_and_metadata(offset_nm_xdata)
                    self.__scan_hardware_source._call_soon(functools.partial(update_data_item, offset_nm_xdata))
                    # report the difference from the last time we reported, but negative since center_nm positive shifts up/left
                    adjustments.offset_nm = -delta_nm
                    self.__center_nm -= delta_nm
                    # print(f"{self.__last_offset_nm} {adjustments.offset_nm} [{(new_offset - self.__last_offset) * 1E9}] {offset} {frame_parameters.fov_nm}")
                    if False:  # if offset_nm > drift area / 10?
                        # retake to provide reference at new offset
                        frame_parameters.center_nm = tuple(Geometry.FloatPoint.make(frame_parameters.center_nm) + adjustments.offset_nm)
                        self.__scan_frame_parameters.center_nm = frame_parameters.center_nm
                        xdatas = self.__scan_hardware_source.record_immediate(frame_parameters, [drift_channel_index])
                        new_offset = self.__scan_hardware_source.stem_controller.drift_offset_m
                else:
                    self.__last_xdata = xdatas[0]
        return adjustments


ProcessingOption = collections.namedtuple("ProcessingOption", ["processing_id", "display_name"])


class ScanAcquisitionProcessing(enum.Enum):
    NONE = ProcessingOption(None,  _("Images"))
    SUM_PROJECT = ProcessingOption("sum_project", _("Spectra"))
    SUM_MASKED = ProcessingOption("sum_masked", _("Virtual Detectors"))


class ScanAcquisitionController:

    def __init__(self, api, document_controller: Facade.DocumentWindow, scan_hardware_source, camera_hardware_source, scan_specifier: ScanSpecifier):
        self.__api = api
        self.__document_controller = document_controller
        self.__scan_hardware_source = scan_hardware_source
        self.__camera_hardware_source = camera_hardware_source
        self.__scan_specifier = copy.deepcopy(scan_specifier)
        self.acquisition_state_changed_event = Event.Event()
        self.__thread = None

    def start(self, processing: ScanAcquisitionProcessing) -> None:

        document_window = self.__document_controller

        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source._hardware_source)

        scan_frame_parameters = scan_hardware_source.get_frame_parameters(2)

        scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters, self.__scan_specifier.size)

        scan_frame_parameters["scan_id"] = str(uuid.uuid4())

        # useful code for testing to exit cleanly at this point.
        # self.acquisition_state_changed_event.fire(SequenceState.scanning)
        # self.acquisition_state_changed_event.fire(SequenceState.idle)
        # return

        camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, self.__camera_hardware_source._hardware_source)

        camera_frame_parameters = camera_hardware_source.get_frame_parameters(0)

        camera_frame_parameters["processing"] = processing.value.processing_id

        grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
            scan_frame_parameters=scan_frame_parameters,
            camera=camera_hardware_source,
            camera_frame_parameters=camera_frame_parameters)

        camera_data_channel = CameraDataChannel(self.__document_controller.library._document_model, camera_hardware_source.display_name, grab_sync_info)
        self.__document_controller.display_data_item(Facade.DataItem(camera_data_channel.data_item))

        camera_data_channel.start()

        drift_correction_behavior : typing.Optional[DriftCorrectionBehavior] = None
        section_height = None
        if self.__scan_specifier.drift_interval_lines > 0:
            drift_correction_behavior = DriftCorrectionBehavior(document_window.library._document_model, scan_hardware_source, scan_frame_parameters)
            section_height = self.__scan_specifier.drift_interval_lines

        def grab_synchronized():
            self.acquisition_state_changed_event.fire(SequenceState.scanning)
            try:
                combined_data = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters,
                                                                       camera=camera_hardware_source,
                                                                       camera_frame_parameters=camera_frame_parameters,
                                                                       camera_data_channel=camera_data_channel,
                                                                       scan_behavior=drift_correction_behavior,
                                                                       section_height=section_height)
                if combined_data is not None:
                    scan_data_list, camera_data_list = combined_data

                    def create_and_display_data_item_task():
                        # this will be executed in UI thread
                        for data_and_metadata in scan_data_list:
                            create_and_display_data_item(document_window, data_and_metadata)

                    # queue the task to be executed in UI thread
                    document_window.queue_task(create_and_display_data_item_task)
            finally:
                def stop_channel():
                    camera_data_channel.stop()

                document_window.queue_task(stop_channel)
                self.acquisition_state_changed_event.fire(SequenceState.idle)

        self.__thread = threading.Thread(target=grab_synchronized)
        self.__thread.start()

    def cancel(self) -> None:
        logging.debug("abort sequence acquisition")
        self.__scan_hardware_source._hardware_source.grab_synchronized_abort()

    # for running tests
    def _wait(self, timeout: float = 60.0) -> None:
        if self.__thread:
            self.__thread.join(timeout)


# see http://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Y', suffix)


def calculate_time_size(camera_hardware_source, scan_pixels, camera_width, camera_height, is_summed, exposure_time):
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
    acquire_sequence_metrics = camera_hardware_source.get_acquire_sequence_metrics(camera_frame_parameters)
    acquisition_time = acquire_sequence_metrics.get("acquisition_time", exposure_time * acquire_pixel_count)  # in seconds
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

    def __init__(self, api):
        self.__api = api
        self.panel_id = "scan-acquisition-panel"
        self.panel_name = _("Spectrum Imaging / 4d Scan Acquisition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.__scan_acquisition_controller = None
        self.__acquisition_state = SequenceState.idle
        self.__acquisition_state_changed_event_listener = None
        self.__line_scan_acquisition_controller = None
        self.__eels_frame_parameters_changed_event_listener = None
        self.__camera_hardware_changed_event_listener = None
        self.__scan_hardware_changed_event_listener = None
        self.__exposure_time_ms_value_model = None
        self.__scan_hardware_source_choice = None
        self.__camera_hardware_source_choice = None
        self.__styles_list_model = None
        self.__styles_list_property_model = None
        self.__camera_width = 0
        self.__camera_height = 0
        self.__scan_specifier = ScanSpecifier()
        self.__scan_width = 32  # the width/length of the scan in pixels
        self.__scan_pixels = 0  # the total number of scan pixels

        self.__scan_acquisition_preference_panel = None

    def create_panel_widget(self, ui: Facade.UserInterface, document_controller: Facade.DocumentWindow) -> Facade.ColumnWidget:
        stem_controller_ = typing.cast(stem_controller.STEMController, Registry.get_component("stem_controller"))

        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(ui._ui, "scan_acquisition_hardware_source_id", lambda hardware_source: hardware_source.features.get("is_scanning"))
        self.__camera_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(ui._ui, "scan_acquisition_camera_hardware_source_id", lambda hardware_source: hardware_source.features.get("is_camera"))
        self.__scan_acquisition_preference_panel = ScanAcquisitionPreferencePanel(self.__scan_hardware_source_choice, self.__camera_hardware_source_choice)

        self.__scan_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self.__scan_hardware_source_choice).add_ref()
        self.__camera_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self.__camera_hardware_source_choice).add_ref()

        def update_context() -> None:
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source_choice.hardware_source)
            scan_context = scan_hardware_source.scan_context

            if scan_context.is_valid and scan_hardware_source.line_scan_enabled and scan_hardware_source.line_scan_vector:
                calibration = scan_context.calibration
                start = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[0])
                end = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[1])
                length = int(Geometry.distance(start, end) * scan_context.size.height)
                max_dim = max(scan_context.size.width, scan_context.size.height)
                length_str = calibration.convert_to_calibrated_size_str(length, value_range=(0, max_dim), samples=max_dim)
                line_str = _("Line Scan")
                self.__roi_description.text = f"{line_str} {length_str} ({length} px)"
                scan_str = _("Scan (1D)")
                scan_length = max(self.__scan_width, 1)
                self.__scan_label_widget.text = f"{scan_str} {scan_length} px"
                self.__scan_pixels = scan_length
                self.__scan_specifier.scan_context = copy.deepcopy(scan_context)
                self.__scan_specifier.size = 1, scan_length
                self.__scan_specifier.drift_interval_lines = 0
                self.__acquire_button._widget.enabled = True
            elif scan_context.is_valid and scan_hardware_source.subscan_enabled and scan_hardware_source.subscan_region:
                calibration = scan_context.calibration
                width = scan_hardware_source.subscan_region.width * scan_context.size.width
                height = scan_hardware_source.subscan_region.height * scan_context.size.height
                width_str = calibration.convert_to_calibrated_size_str(width, value_range=(0, scan_context.size.width), samples=scan_context.size.width)
                height_str = calibration.convert_to_calibrated_size_str(height, value_range=(0, scan_context.size.height), samples=scan_context.size.height)
                rect_str = _("Subscan")
                self.__roi_description.text = f"{rect_str} {width_str} x {height_str} ({int(width)} px x {int(height)} px)"
                scan_str = _("Scan (2D)")
                scan_width = self.__scan_width
                scan_height = int(self.__scan_width * height / width)
                drift_lines = scan_hardware_source.calculate_drift_lines(scan_width, self.__exposure_time_ms_value_model.value / 1000) if scan_hardware_source else 0
                drift_str = f" / Drift {drift_lines} lines" if drift_lines > 0 else str()
                self.__scan_label_widget.text = f"{scan_str} {scan_width} x {scan_height} px" + drift_str
                self.__scan_pixels = scan_width * scan_height
                self.__scan_specifier.scan_context = copy.deepcopy(scan_context)
                self.__scan_specifier.size = scan_height, scan_width
                self.__scan_specifier.drift_interval_lines = drift_lines
                self.__acquire_button._widget.enabled = True
            elif scan_context.is_valid:
                calibration = scan_context.calibration
                width = scan_context.size.width
                height = scan_context.size.height
                width_str = calibration.convert_to_calibrated_size_str(width, value_range=(0, scan_context.size.width), samples=scan_context.size.width)
                height_str = calibration.convert_to_calibrated_size_str(height, value_range=(0, scan_context.size.height), samples=scan_context.size.height)
                data_str = _("Context Scan")
                self.__roi_description.text = f"{data_str} {width_str} x {height_str} ({int(width)} x {int(height)})"
                scan_str = _("Scan (2D)")
                scan_width = self.__scan_width
                scan_height = int(self.__scan_width * height / width)
                drift_lines = scan_hardware_source.calculate_drift_lines(scan_width, self.__exposure_time_ms_value_model.value / 1000) if scan_hardware_source else 0
                drift_str = f" / Drift {drift_lines} lines" if drift_lines > 0 else str()
                self.__scan_label_widget.text = f"{scan_str} {scan_width} x {scan_height} px" + drift_str
                self.__scan_pixels = scan_width * scan_height
                self.__scan_specifier.scan_context = copy.deepcopy(scan_context)
                self.__scan_specifier.size = scan_height, scan_width
                self.__scan_specifier.drift_interval_lines = drift_lines
                self.__acquire_button._widget.enabled = True
            else:
                self.__roi_description.text = _("Scan context not active")
                self.__scan_label_widget.text = None
                self.__scan_specifier.scan_context = stem_controller.ScanContext()
                self.__scan_specifier.size = None
                self.__scan_specifier.drift_interval_lines = 0
                self.__acquire_button._widget.enabled = self.__acquisition_state == SequenceState.scanning  # focus will be on the SI data, so enable if scanning
                self.__scan_pixels = 0

            self.__scan_width_widget.text = Converter.IntegerToStringConverter().convert(self.__scan_width)

            self.__update_estimate()

        def stem_controller_property_changed(key: str) -> None:
            if key in ("subscan_state", "subscan_region", "subscan_rotation", "line_scan_state", "line_scan_vector", "drift_channel_id", "drift_region", "drift_settings"):
                document_controller._document_controller.event_loop.call_soon_threadsafe(update_context)

        def scan_context_changed() -> None:
            update_context()

        self.__stem_controller_property_listener = None
        self.__scan_context_changed_listener = None

        if stem_controller_:
            self.__stem_controller_property_listener = stem_controller_.property_changed_event.listen(stem_controller_property_changed)
            self.__scan_context_changed_listener = stem_controller_.scan_context_changed_event.listen(scan_context_changed)

        column = ui.create_column_widget()

        self.__styles_list_model = ListModel.ListModel(items=[ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisitionProcessing.NONE])
        self.__styles_list_property_model = ListModel.ListPropertyModel(self.__styles_list_model)
        self.__style_combo_box = ui.create_combo_box_widget(self.__styles_list_property_model.value, item_text_getter=operator.attrgetter("value.display_name"))
        self.__style_combo_box._widget.set_property("min-width", 100)
        items_binding = Binding.PropertyBinding(self.__styles_list_property_model, "value")
        items_binding.source_setter = None
        self.__style_combo_box._widget.bind_items(items_binding)
        self.__style_combo_box.current_index = 0

        self.__acquire_button = ui.create_push_button_widget(_("Acquire"))

        self.__roi_description = ui.create_label_widget()

        self.__scan_width_widget = ui.create_line_edit_widget()

        self.__exposure_time_widget = ui.create_line_edit_widget()

        self.__estimate_label_widget = ui.create_label_widget()

        self.__scan_label_widget = ui.create_label_widget()

        class ComboBoxWidget:
            def __init__(self, widget):
                self.__combo_box_widget = widget

            @property
            def _widget(self):
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
        acquire_sequence_button_row.add_stretch()

        if self.__scan_hardware_source_choice.hardware_source_count > 1:
            column.add_spacing(8)
            column.add(scan_choice_row)
        column.add_spacing(8)
        column.add(camera_row)
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

        def camera_hardware_source_changed(hardware_source):
            self.disconnect_camera_hardware_source()
            if hardware_source:
                self.connect_camera_hardware_source(hardware_source)
                if hardware_source.features.get("has_masked_sum_option"):
                    self.__styles_list_model.items = [ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisitionProcessing.NONE, ScanAcquisitionProcessing.SUM_MASKED]
                else:
                    self.__styles_list_model.items = [ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisitionProcessing.NONE]

        self.__camera_hardware_changed_event_listener = self.__camera_hardware_source_choice.hardware_source_changed_event.listen(camera_hardware_source_changed)
        camera_hardware_source_changed(self.__camera_hardware_source_choice.hardware_source)

        def style_current_item_changed(current_item):
            self.__update_estimate()

        self.__style_combo_box.on_current_item_changed = style_current_item_changed

        def scan_width_changed(text: str) -> None:
            scan_width = Converter.IntegerToStringConverter().convert_back(text) if text else 1
            scan_width = max(scan_width, 1)
            if scan_width != self.__scan_width:
                self.__scan_width = scan_width
                update_context()
            self.__scan_width_widget.request_refocus()

        self.__scan_width_widget.on_editing_finished = scan_width_changed

        def acquisition_state_changed(acquisition_state: SequenceState) -> None:
            self.__acquisition_state = acquisition_state

            async def update_button_text(text: str) -> None:
                self.__acquire_button.text = text
                update_context()  # update the cancel button

            if acquisition_state == SequenceState.idle:
                self.__scan_acquisition_controller = None
                self.__acquisition_state_changed_event_listener.close()
                self.__acquisition_state_changed_event_listener = None
                document_controller._document_window.event_loop.create_task(update_button_text(_("Acquire")))
            else:
                document_controller._document_window.event_loop.create_task(update_button_text(_("Cancel")))

        def acquire_sequence() -> None:
            if self.__scan_acquisition_controller:
                if self.__scan_acquisition_controller:
                    self.__scan_acquisition_controller.cancel()
            else:
                if self.__scan_hardware_source_choice.hardware_source:
                    scan_hardware_source = self.__api.get_hardware_source_by_id(self.__scan_hardware_source_choice.hardware_source.hardware_source_id, version="1.0")
                else:
                    scan_hardware_source = None

                if self.__camera_hardware_source_choice.hardware_source:
                    camera_hardware_source = self.__api.get_hardware_source_by_id(self.__camera_hardware_source_choice.hardware_source.hardware_source_id, version="1.0")
                else:
                    camera_hardware_source = None

                if scan_hardware_source and camera_hardware_source:
                    self.__scan_acquisition_controller = ScanAcquisitionController(self.__api, document_controller, scan_hardware_source, camera_hardware_source, self.__scan_specifier)
                    self.__acquisition_state_changed_event_listener = self.__scan_acquisition_controller.acquisition_state_changed_event.listen(acquisition_state_changed)
                    self.__scan_acquisition_controller.start(self.__style_combo_box.current_item)

        self.__acquire_button.on_clicked = acquire_sequence

        self.__update_estimate()

        update_context()

        return column

    def __update_estimate(self):
        if self.__exposure_time_ms_value_model:
            camera_hardware_source = self.__camera_hardware_source_choice.hardware_source
            camera_width = self.__camera_width
            camera_height = self.__camera_height
            is_summed = self.__style_combo_box.current_index == 0
            exposure_time = self.__exposure_time_ms_value_model.value / 1000
            time_str, size_str = calculate_time_size(camera_hardware_source, self.__scan_pixels, camera_width, camera_height, is_summed, exposure_time)
            self.__estimate_label_widget.text = "{0} / {1}".format(time_str, size_str)
        else:
            self.__estimate_label_widget.text = None

    def connect_camera_hardware_source(self, camera_hardware_source):

        self.__exposure_time_ms_value_model = Model.PropertyModel()

        def update_exposure_time_ms(exposure_time_ms):
            if exposure_time_ms > 0:
                frame_parameters = camera_hardware_source.get_frame_parameters(0)
                frame_parameters.exposure_ms = exposure_time_ms
                camera_hardware_source.set_frame_parameters(0, frame_parameters)
            self.__update_estimate()

        self.__exposure_time_ms_value_model.on_value_changed = update_exposure_time_ms

        exposure_time_ms_value_binding = Binding.PropertyBinding(self.__exposure_time_ms_value_model, "value", converter=Converter.FloatToStringConverter("{0:.1f}"))

        def eels_profile_parameters_changed(profile_index, frame_parameters):
            if profile_index == 0:
                expected_dimensions = camera_hardware_source.get_expected_dimensions(frame_parameters.binning)
                self.__camera_width = expected_dimensions[1]
                self.__camera_height = expected_dimensions[0]
                self.__exposure_time_ms_value_model.value = frame_parameters.exposure_ms
                self.__update_estimate()

        self.__eels_frame_parameters_changed_event_listener = camera_hardware_source.frame_parameters_changed_event.listen(eels_profile_parameters_changed)

        eels_profile_parameters_changed(0, camera_hardware_source.get_frame_parameters(0))

        self.__exposure_time_widget._widget.bind_text(exposure_time_ms_value_binding)  # the widget will close the binding

    def disconnect_camera_hardware_source(self):
        self.__exposure_time_widget._widget.unbind_text()
        if self.__eels_frame_parameters_changed_event_listener:
            self.__eels_frame_parameters_changed_event_listener.close()
            self.__eels_frame_parameters_changed_event_listener = None
        if self.__exposure_time_ms_value_model:
            self.__exposure_time_ms_value_model.close()
            self.__exposure_time_ms_value_model = None

    def close(self):
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
        if self.__camera_hardware_source_choice:
            self.__camera_hardware_source_choice.close()
            self.__camera_hardware_source_choice = None
        if self.__scan_acquisition_preference_panel:
            # PreferencesDialog.PreferencesManager().unregister_preference_pane(self.__scan_acquisition_preference_panel)
            self.__scan_acquisition_preference_panel = None
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


class ScanAcquisitionPreferencePanel:
    def __init__(self, scan_hardware_source_choice, other_hardware_source_choice):
        self.identifier = "scan_acquisition"
        self.label = _("Spectrum Imaging / 4d Scan Acquisition")
        self.__scan_hardware_source_choice = scan_hardware_source_choice
        self.__camera_hardware_source_choice = other_hardware_source_choice

    def build(self, ui, **kwargs):
        scan_hardware_source_combo_box = self.__scan_hardware_source_choice.create_combo_box(ui)
        other_hardware_source_combo_box = self.__camera_hardware_source_choice.create_combo_box(ui)
        row = ui.create_row_widget()
        row.add(ui.create_label_widget(_("Scan Device")))
        row.add_spacing(12)
        row.add(scan_hardware_source_combo_box)
        row.add_spacing(12)
        row.add(other_hardware_source_combo_box)
        return row


class ScanAcquisitionExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.instrumentation-kit.scan-acquisition"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version=API.version, ui_version=UserInterface.version)
        # be sure to keep a reference or it will be closed immediately.
        self.__panel_ref = api.create_panel(PanelDelegate(api))

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        # self.__menu_item_ref.close()
        # self.__menu_item_ref = None
        self.__panel_ref.close()
        self.__panel_ref = None
