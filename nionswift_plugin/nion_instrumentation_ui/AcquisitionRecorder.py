from __future__ import annotations

# system imports
import datetime
import gettext
import threading
import time
import typing
import uuid

# third part imports
import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.swift.model import DataItem
from nion.swift.model import ImportExportManager
from nion.ui import UserInterface as UserInterfaceModule
from nion.utils import Binding
from nion.utils import Converter
from nion.utils import Model
from nion.utils import Validator
from . import HardwareSourceChoice

if typing.TYPE_CHECKING:
    from nion.swift import DocumentController
    from nion.swift import Facade
    from nion.utils import Event

_ = gettext.gettext


T = typing.TypeVar('T', bound=UserInterfaceModule.Widget)


class WidgetWrapper(typing.Generic[T]):
    def __init__(self, widget: T) -> None:
        self.__widget = widget

    @property
    def _widget(self) -> T:
        return self.__widget


class Controller:

    def __init__(self) -> None:
        self.state = Model.PropertyModel("idle")
        self.frame_count_model = Model.PropertyModel(20)
        self.progress_model = Model.PropertyModel(0)
        self.cancel_event = threading.Event()
        self.__grab_thread = None
        self.__record_thread = None

    def close(self) -> None:
        self.cancel_event.set()
        if self.__grab_thread:
            self.__grab_thread.join()
            self.__grab_thread = None
        if self.__record_thread:
            self.__record_thread.join()
            self.__record_thread = None
        self.state.close()
        self.state = typing.cast(typing.Any, None)
        self.frame_count_model.close()
        self.frame_count_model = typing.cast(typing.Any, None)
        self.progress_model.close()
        self.progress_model = typing.cast(typing.Any, None)

    async def grab(self, document_controller: DocumentController.DocumentController, hardware_source: scan_base.ScanHardwareSource, do_acquire: bool) -> None:
        # this is an async method meaning that it will execute until it calls await, at which time
        # it will let other parts of the software run until the awaited function finishes. in this
        # case, waiting for acquired data and grabbing the last frames are run in a thread.

        assert document_controller
        assert hardware_source

        event_loop = document_controller.event_loop

        self.cancel_event.clear()

        self.state.value = "running"
        self.progress_model.value = 0
        frame_count = self.frame_count_model.value or 0
        was_playing = hardware_source.is_playing

        document_model = document_controller.document_model

        Acquisition.session_manager.begin_acquisition(document_model)  # bump the index

        xdata_group_list: typing.List[typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]] = list()

        def exec_acquire() -> typing.Tuple[bool, int]:
            # this will execute in a thread; the enclosing async routine will continue when it finishes
            try:
                start_time = time.time()
                max_wait_time = max(hardware_source.get_current_frame_time() * 1.5, 3)
                while not hardware_source.is_playing:
                    if time.time() - start_time > max_wait_time:
                        return False, 0
                    time.sleep(0.01)
                while (i := hardware_source.get_sequence_buffer_count()) < frame_count:
                    time.sleep(0.2)
                    if self.cancel_event.is_set():
                        return False, max(i - 1, 0)
                    self.progress_model.value = min(50, int(i / frame_count * 50))
                return True, frame_count
            except Exception as e:
                import traceback
                traceback.print_exc()
            return False, 0

        if do_acquire:
            print(f"AR: start playing {datetime.datetime.now()}")
            hardware_source.prepare_sequence_mode(hardware_source.get_current_frame_parameters(), frame_count)
            hardware_source.start_sequence_mode(hardware_source.get_current_frame_parameters(), frame_count)
            print("AR: wait for acquire")
            success, actual_count = await event_loop.run_in_executor(None, exec_acquire)
            if not was_playing:
                print("AR: restopping")
                hardware_source.stop_playing()
            print(f"AR: acquire finished {datetime.datetime.now()}")
        else:
            success = True
            actual_count = frame_count

        def exec_grab(actual_count: int) -> bool:
            # this will execute in a thread; the enclosing async routine will continue when it finishes
            try:
                scan_id = uuid.uuid4()
                for i in range(actual_count):
                    if self.cancel_event.is_set():
                        return False
                    self.progress_model.value = min(100, 50 + int(i / actual_count * 50))
                    xdata_group_list.append(list(hardware_source.pop_sequence_buffer_data(scan_id).values()))
                self.progress_model.value = 100
                return True
            except Exception as e:
                import traceback
                traceback.print_exc()
                return False
            finally:
                hardware_source.finish_sequence_mode()

        if actual_count > 0:
            print(f"AR: grabbing data {datetime.datetime.now()}")
            self.cancel_event.clear()
            success = await event_loop.run_in_executor(None, exec_grab, actual_count)
            print(f"AR: grab finished {'success' if success else 'failure'} {datetime.datetime.now()}")

        xdata_group = None

        if success:
            if len(xdata_group_list) > 1:
                print(f"AR: making xdata {datetime.datetime.now()}")
                valid_count = 0
                examplar_xdata_group = xdata_group_list[-1]
                shapes = [xdata._data_ex.shape for xdata in examplar_xdata_group if xdata]
                dtypes = [xdata._data_ex.dtype for xdata in examplar_xdata_group if xdata]
                for xdata_group in reversed(xdata_group_list):
                    shapes_i = [xdata._data_ex.shape for xdata in xdata_group if xdata]
                    dtypes_i = [xdata._data_ex.dtype for xdata in xdata_group if xdata]
                    if shapes_i == shapes and dtypes_i == dtypes:
                        valid_count += 1
                xdata_group = list()
                for i, xdata in enumerate(examplar_xdata_group):
                    if xdata:
                        intensity_calibration = xdata.intensity_calibration
                        dimensional_calibrations = [Calibration.Calibration()] + list(xdata.dimensional_calibrations)
                        data_descriptor = DataAndMetadata.DataDescriptor(True,
                                                                         xdata.data_descriptor.collection_dimension_count,
                                                                         xdata.data_descriptor.datum_dimension_count)
                        # TODO: ugly typing.
                        data: numpy.typing.NDArray[typing.Any] = numpy.vstack(list(typing.cast(DataAndMetadata.DataAndMetadata, xdata_group[i])._data_ex for xdata_group in xdata_group_list[-valid_count:])).reshape(valid_count, *shapes[i])
                        xdata = DataAndMetadata.new_data_and_metadata(data,
                                                                      intensity_calibration=intensity_calibration,
                                                                      dimensional_calibrations=dimensional_calibrations,
                                                                      data_descriptor=data_descriptor,
                                                                      metadata=xdata.metadata)
                        xdata_group.append(xdata)
            elif len(xdata_group_list) == 1:
                xdata_group = xdata_group_list[0]

        if xdata_group:
            print(f"AR: making data item {datetime.datetime.now()}")
            for xdata in xdata_group:
                if xdata:
                    data_item = DataItem.DataItem(large_format=True)
                    data_item.set_xdata(xdata)
                    channel_name = xdata.metadata.get("hardware_source", dict()).get("channel_name")
                    channel_ext = (" (" + channel_name + ")") if channel_name else ""

                    acquisition_number = Acquisition.session_manager.get_project_acquisition_index(document_model)
                    data_item_title = f"{hardware_source.display_name} Recording ({channel_ext})"
                    if acquisition_number:
                        data_item_title += f" {acquisition_number}"
                    data_item.title = data_item_title

                    document_model.append_data_item(data_item)
                    display_item = document_model.get_display_item_for_data_item(data_item)
                    if display_item:
                        document_controller.show_display_item(display_item)

        if was_playing:
            print("AR: restarting")
            hardware_source.start_playing()
        self.state.value = "idle"
        self.progress_model.value = 0
        print(f"AR: done {datetime.datetime.now()}")

    def cancel(self) -> None:
        self.cancel_event.set()


class PanelDelegate:

    def __init__(self, api: typing.Any) -> None:
        self.__api = api
        self.panel_id = "acquisition-recorder-panel"
        self.panel_name = _("Scan Acquisition Recorder")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.__controller: typing.Optional[Controller] = None
        self.__scan_hardware_source_choice_model: typing.Optional[Model.PropertyModel[str]] = None
        self.__scan_hardware_source_choice: typing.Optional[HardwareSourceChoice.HardwareSourceChoice] = None
        self.__scan_hardware_source: typing.Optional[scan_base.ScanHardwareSource] = None
        self.__state_changed_listener: typing.Optional[Event.EventListener] = None
        self.__scan_hardware_changed_event_listener: typing.Optional[Event.EventListener] = None

    def create_panel_widget(self, ui: Facade.UserInterface, document_controller: Facade.DocumentWindow) -> Facade.ColumnWidget:
        # note: anything created here should be disposed in close.
        # this method may be called more than once.
        self.__controller = Controller()

        self.__scan_hardware_source_choice_model = ui._ui.create_persistent_string_model("scan_acquisition_hardware_source_id")
        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__scan_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_scanning", False))

        column = ui.create_column_widget()

        # define the main controls
        source_combo_box = self.__scan_hardware_source_choice.create_combo_box(ui._ui)
        count_line_edit = ui.create_line_edit_widget()
        grab_button = ui.create_push_button_widget(_("Grab Previous"))
        record_button = ui.create_push_button_widget(_("Record Next"))
        cancel_button = ui.create_push_button_widget(_("Stop"))
        progress_bar = ui._ui.create_progress_bar_widget(properties={"height": 18, "min-width": 200})
        progress_bar.minimum = 0
        progress_bar.maximum = 100

        # define the layout
        source_row = ui.create_row_widget()
        source_row.add_spacing(12)
        source_row.add(WidgetWrapper[UserInterfaceModule.ComboBoxWidget](source_combo_box))
        source_row.add_spacing(12)
        source_row.add_stretch()
        count_row = ui.create_row_widget()
        count_row.add_spacing(12)
        count_row.add(ui.create_label_widget(_("Frames")))
        count_row.add_spacing(12)
        count_row.add(count_line_edit)
        count_row.add_spacing(12)
        count_row.add_stretch()
        button_row = ui.create_row_widget()
        button_row.add_spacing(12)
        button_row.add(grab_button)
        button_row.add_spacing(12)
        button_row.add(record_button)
        button_row.add_spacing(12)
        button_row.add(cancel_button)
        button_row.add_spacing(12)
        button_row.add_stretch()
        progress_bar_row = ui.create_row_widget()
        progress_bar_row.add_stretch()
        progress_bar_row.add_spacing(12)
        progress_bar_row.add(WidgetWrapper[UserInterfaceModule.ProgressBarWidget](progress_bar))
        progress_bar_row.add_spacing(12)
        progress_bar_row.add_stretch()
        column.add_spacing(8)
        column.add(source_row)
        column.add(count_row)
        column.add(button_row)
        column.add(progress_bar_row)
        column.add_spacing(8)
        column.add_stretch()

        # connect the pieces

        def state_property_changed(property: str) -> None:
            if not self.__scan_hardware_source:
                source_combo_box.enabled = True
                count_line_edit._widget.enabled = False
                grab_button._widget.enabled = False
                record_button._widget.enabled = False
                cancel_button._widget.enabled = False
            elif self.__controller and self.__controller.state.value == "idle":
                source_combo_box.enabled = True
                count_line_edit._widget.enabled = True
                grab_button._widget.enabled = True
                record_button._widget.enabled = True
                cancel_button._widget.enabled = False
            elif self.__controller and self.__controller.state.value == "running":
                source_combo_box.enabled = False
                count_line_edit._widget.enabled = False
                grab_button._widget.enabled = False
                record_button._widget.enabled = False
                cancel_button._widget.enabled = True

        def scan_hardware_source_changed(hardware_source: typing.Optional[HardwareSource.HardwareSource]) -> None:
            if isinstance(hardware_source, scan_base.ScanHardwareSource):
                self.__scan_hardware_source = hardware_source
                state_property_changed("value")

        self.__scan_hardware_changed_event_listener = self.__scan_hardware_source_choice.hardware_source_changed_event.listen(scan_hardware_source_changed)

        typing.cast(UserInterfaceModule.LineEditWidget, count_line_edit._widget).bind_text(Binding.PropertyBinding(self.__controller.frame_count_model, "value", converter=Converter.IntegerToStringConverter(), validator=Validator.IntegerRangeValidator(1, 10000000)))

        self.__state_changed_listener = self.__controller.state.property_changed_event.listen(state_property_changed)

        progress_bar.bind_value(Binding.PropertyBinding(self.__controller.progress_model, "value"))

        event_loop = document_controller._document_controller.event_loop

        def grab() -> None:
            if self.__controller and self.__scan_hardware_source:
                event_loop.create_task(self.__controller.grab(document_controller._document_controller, self.__scan_hardware_source, False))

        def record() -> None:
            if self.__controller and self.__scan_hardware_source:
                event_loop.create_task(self.__controller.grab(document_controller._document_controller, self.__scan_hardware_source, True))

        grab_button.on_clicked = grab
        record_button.on_clicked = record
        cancel_button.on_clicked = self.__controller.cancel

        scan_hardware_source_changed(self.__scan_hardware_source_choice.hardware_source)

        return column

    def close(self) -> None:
        # close anything created in `create_panel_widget`.
        # called when the panel closes, not when the delegate closes.
        self.__scan_hardware_changed_event_listener = None
        self.__state_changed_listener = None
        self.__scan_hardware_source_choice = None
        self.__scan_hardware_source_choice_model = None
        if self.__controller:
            self.__controller.close()
            self.__controller = None


class AcquisitionRecorderExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.instrumentation-kit.acquisition-recorder"

    def __init__(self, api_broker: typing.Any) -> None:
        # grab the api object.
        api = api_broker.get_api(version='1', ui_version='1')
        # be sure to keep a reference or it will be closed immediately.
        self.__panel_ref = api.create_panel(PanelDelegate(api))

    def close(self) -> None:
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        # self.__menu_item_ref.close()
        # self.__menu_item_ref = None
        self.__panel_ref.close()
        self.__panel_ref = None
