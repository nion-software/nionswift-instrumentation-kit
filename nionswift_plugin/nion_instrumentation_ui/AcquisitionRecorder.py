# system imports
import functools
import gettext
import threading
import time

# third part imports
import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift.model import DataItem
from nion.swift.model import ImportExportManager
from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UserInterface
from nion.utils import Binding
from nion.utils import Converter
from nion.utils import Model
from nion.utils import Validator
from . import HardwareSourceChoice

_ = gettext.gettext


class WidgetWrapper:
    def __init__(self, widget):
        self.__widget = widget

    @property
    def _widget(self):
        return self.__widget


class Controller:

    def __init__(self):
        self.state = Model.PropertyModel("idle")
        self.frame_count_model = Model.PropertyModel(20)
        self.progress_model = Model.PropertyModel(0)
        self.cancel_event = threading.Event()
        self.__grab_thread = None
        self.__record_thread = None

    def close(self):
        self.cancel_event.set()
        if self.__grab_thread:
            self.__grab_thread.join()
            self.__grab_thread = None
        if self.__record_thread:
            self.__record_thread.join()
            self.__record_thread = None
        self.state.close()
        self.state = None
        self.frame_count_model.close()
        self.frame_count_model = None
        self.progress_model.close()
        self.progress_model = None

    async def grab(self, document_controller, hardware_source, do_acquire):
        # this is an async method meaning that it will execute until it calls await, at which time
        # it will let other parts of the software run until the awaited function finishes. in this
        # case, waiting for acquired data and grabbing the last frames are run in a thread.

        assert document_controller
        assert hardware_source

        event_loop = document_controller.event_loop

        self.cancel_event.clear()

        self.state.value = "running"
        self.progress_model.value = 0
        frame_count = self.frame_count_model.value
        was_playing = hardware_source.is_playing

        success_ref = [True]

        xdata_group_list = list()

        def exec_acquire():
            # this will execute in a thread; the enclosing async routine will continue when it finishes
            try:
                start_time = time.time()
                max_wait_time = max(hardware_source.get_current_frame_time() * 1.5, 3)
                while not hardware_source.is_playing:
                    if time.time() - start_time > max_wait_time:
                        success_ref[0] = False
                        return
                    time.sleep(0.01)
                hardware_source.get_next_xdatas_to_start(max_wait_time * 2)  # wait for frame + next frame
                for i in range(frame_count):
                    self.progress_model.value = int(100 * i / frame_count)
                    if self.cancel_event.is_set():
                        success_ref[0] = False
                        break
                    hardware_source.get_next_xdatas_to_finish(max_wait_time * 2)
            except Exception as e:
                import traceback
                traceback.print_exc()
                success_ref[0] = False

        if do_acquire:
            print("AR: start playing")
            hardware_source.start_playing()
            print("AR: wait for acquire")
            await event_loop.run_in_executor(None, exec_acquire)
            print("AR: acquire finished")

        def exec_grab():
            # this will execute in a thread; the enclosing async routine will continue when it finishes
            try:
                start_time = time.time()
                max_wait_time = max(hardware_source.get_current_frame_time() * 1.5, 3)
                while hardware_source.is_playing:
                    if time.time() - start_time > max_wait_time:
                        success_ref[0] = False
                        return
                    time.sleep(0.01)
                data_element_groups = hardware_source.get_buffer_data(-frame_count, frame_count)
                for data_element_group in data_element_groups:
                    if self.cancel_event.is_set():
                        success_ref[0] = False
                        break
                    xdata_group = list()
                    for data_element in data_element_group:
                        xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
                        xdata_group.append(xdata)
                    xdata_group_list.append(xdata_group)
                self.progress_model.value = 100
            except Exception as e:
                import traceback
                traceback.print_exc()
                success_ref[0] = False

        if success_ref[0]:
            print("AR: stop playing")
            hardware_source.stop_playing()
            print("AR: grabbing data")
            await event_loop.run_in_executor(None, exec_grab)
            print("AR: grab finished")

        xdata_group = None

        if success_ref[0]:
            if len(xdata_group_list) > 1:
                print("AR: making xdata")
                valid_count = 0
                examplar_xdata_group = xdata_group_list[-1]
                shapes = [xdata.data.shape for xdata in examplar_xdata_group]
                dtypes = [xdata.data.dtype for xdata in examplar_xdata_group]
                for xdata_group in reversed(xdata_group_list):
                    shapes_i = [xdata.data.shape for xdata in xdata_group]
                    dtypes_i = [xdata.data.dtype for xdata in xdata_group]
                    if shapes_i == shapes and dtypes_i == dtypes:
                        valid_count += 1
                xdata_group = list()
                for i, xdata in enumerate(examplar_xdata_group):
                    intensity_calibration = xdata.intensity_calibration
                    dimensional_calibrations = [Calibration.Calibration()] + list(xdata.dimensional_calibrations)
                    data_descriptor = DataAndMetadata.DataDescriptor(True,
                                                                     xdata.data_descriptor.collection_dimension_count,
                                                                     xdata.data_descriptor.datum_dimension_count)
                    data = numpy.vstack(xdata_group[i].data for xdata_group in xdata_group_list[-valid_count:]).reshape(valid_count, *shapes[i])
                    xdata = DataAndMetadata.new_data_and_metadata(data,
                                                                  intensity_calibration=intensity_calibration,
                                                                  dimensional_calibrations=dimensional_calibrations,
                                                                  data_descriptor=data_descriptor)
                    xdata_group.append(xdata)
            elif len(xdata_group_list) == 1:
                xdata_group = xdata_group_list[0]

        if xdata_group:
            print("AR: making data item")
            for xdata in xdata_group:
                data_item = DataItem.DataItem(large_format=True)
                data_item.ensure_data_source()
                data_item.set_xdata(xdata)
                data_item.title = _("Recording of ") + hardware_source.display_name
                document_controller.document_model.append_data_item(data_item)
                document_controller.display_data_item(DataItem.DisplaySpecifier.from_data_item(data_item))

        if was_playing:
            print("AR: restarting")
            hardware_source.start_playing()
        self.state.value = "idle"
        self.progress_model.value = 0
        print("AR: done")

    def cancel(self):
        self.cancel_event.set()


class PanelDelegate:

    def __init__(self, api):
        self.__api = api
        self.panel_id = "acquisition-recorder-panel"
        self.panel_name = _("Scan Acquisition Recorder")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.__controller = Controller()
        self.__scan_hardware_source_choice = None
        self.__scan_hardware_source = None
        self.__state_changed_listener = None
        self.__scan_hardware_changed_event_listener = None

    def create_panel_widget(self, ui, document_controller):
        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(ui._ui, "scan_acquisition_hardware_source_id", lambda hardware_source: hardware_source.features.get("is_scanning"))

        column = ui.create_column_widget()

        # define the main controls
        source_combo_box = self.__scan_hardware_source_choice.create_combo_box(ui._ui)
        count_line_edit = ui.create_line_edit_widget()
        grab_button = ui.create_push_button_widget(_("Grab Previous"))
        record_button = ui.create_push_button_widget(_("Record Next"))
        cancel_button = ui.create_push_button_widget(_("Cancel"))
        progress_bar = ui._ui.create_progress_bar_widget(properties={"height": 18, "min-width": 200})
        progress_bar.minimum = 0
        progress_bar.maximum = 100

        # define the layout
        source_row = ui.create_row_widget()
        source_row.add_spacing(12)
        source_row.add(WidgetWrapper(source_combo_box))
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
        progress_bar_row.add(WidgetWrapper(progress_bar))
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

        def state_property_changed(property):
            if not self.__scan_hardware_source:
                source_combo_box.enabled = True
                count_line_edit._widget.enabled = False
                grab_button._widget.enabled = False
                record_button._widget.enabled = False
                cancel_button._widget.enabled = False
            elif self.__controller.state.value == "idle":
                source_combo_box.enabled = True
                count_line_edit._widget.enabled = True
                grab_button._widget.enabled = True
                record_button._widget.enabled = True
                cancel_button._widget.enabled = False
            elif self.__controller.state.value == "running":
                source_combo_box.enabled = False
                count_line_edit._widget.enabled = False
                grab_button._widget.enabled = False
                record_button._widget.enabled = False
                cancel_button._widget.enabled = True

        def scan_hardware_source_changed(hardware_source):
            self.__scan_hardware_source = hardware_source
            state_property_changed("value")

        self.__scan_hardware_changed_event_listener = self.__scan_hardware_source_choice.hardware_source_changed_event.listen(scan_hardware_source_changed)

        count_line_edit._widget.bind_text(Binding.PropertyBinding(self.__controller.frame_count_model, "value", converter=Converter.IntegerToStringConverter(), validator=Validator.IntegerRangeValidator(1, 100)))

        self.__state_changed_listener = self.__controller.state.property_changed_event.listen(state_property_changed)

        progress_bar.bind_value(Binding.PropertyBinding(self.__controller.progress_model, "value"))

        event_loop = document_controller._document_controller.event_loop

        def grab():
            event_loop.create_task(self.__controller.grab(document_controller._document_controller, self.__scan_hardware_source, False))

        def record():
            event_loop.create_task(self.__controller.grab(document_controller._document_controller, self.__scan_hardware_source, True))

        grab_button.on_clicked = grab
        record_button.on_clicked = record
        cancel_button.on_clicked = self.__controller.cancel

        scan_hardware_source_changed(self.__scan_hardware_source_choice.hardware_source)

        return column

    def close(self):
        if self.__scan_hardware_changed_event_listener:
            self.__scan_hardware_changed_event_listener.close()
            self.__scan_hardware_changed_event_listener = None
        if self.__state_changed_listener:
            self.__state_changed_listener.close()
            self.__state_changed_listener = None
        if self.__scan_hardware_source_choice:
            self.__scan_hardware_source_choice.close()
            self.__scan_hardware_source_choice = None
        self.__controller.close()
        self.__controller = None


class AcquisitionRecorderExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.instrumentation-kit.acquisition-recorder"

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
