# system imports
import contextlib
import copy
import gettext
import logging
import threading
import time
import typing

# third part imports
import numpy

# local libraries
from nion.typeshed import API_1_0 as API
from nion.typeshed import HardwareSource_1_0 as HardwareSource
from nion.typeshed import UI_1_0 as UserInterface
from nion.swift.model import ImportExportManager
from nion.swift.model import HardwareSource as HardwareSourceModule
from nion.utils import Binding
from nion.utils import Converter
from nion.utils import Event
from nion.utils import Model

_ = gettext.gettext


class ScanAcquisitionController(object):

    def __init__(self, api):
        self.__api = api
        self.__aborted = False
        self.acquisition_state_changed_event = Event.Event()

    def start_spectrum_image(self, document_window: API.DocumentWindow) -> None:

        def acquire_spectrum_image(api: API.API, document_window: API.DocumentWindow) -> None:
            try:
                logging.debug("start")
                self.acquisition_state_changed_event.fire({"message": "start"})
                try:
                    eels_camera = api.get_hardware_source_by_id("orca_camera", version="1.0")
                    eels_camera_parameters = eels_camera.get_frame_parameters_for_profile_by_index(0)

                    scan_controller = api.get_hardware_source_by_id("scan_controller", version="1.0")
                    scan_parameters = scan_controller.get_frame_parameters_for_profile_by_index(2)
                    scan_max_size = 256
                    scan_parameters["size"] = min(scan_max_size, scan_parameters["size"][0]), min(scan_max_size, scan_parameters["size"][1])
                    scan_parameters["pixel_time_us"] = int(1000 * eels_camera_parameters["exposure_ms"] * 0.75)
                    scan_parameters["external_clock_wait_time_ms"] = int(eels_camera_parameters["exposure_ms"] * 1.5)
                    scan_parameters["external_clock_mode"] = 1

                    library = document_window.library
                    data_item = library.create_data_item(_("Spectrum Image"))
                    document_window.display_data_item(data_item)

                    # force the data to be held in memory and write delayed by grabbing a data_ref.
                    with library.data_ref_for_data_item(data_item) as data_ref:
                        flyback_pixels = 2
                        with contextlib.closing(eels_camera.create_view_task(frame_parameters=eels_camera_parameters, buffer_size=16)) as eels_view_task:
                            # wait for a frame, then create the record task during the next frame, then wait for that
                            # frame to finish. that will position the scan at the first position. proceed with acquisition.
                            eels_view_task.grab_next_to_finish()
                            eels_view_task.grab_earliest()  # wait for current frame to finish
                            with contextlib.closing(scan_controller.create_record_task(scan_parameters)) as scan_task:
                                try:
                                    scan_height = scan_parameters["size"][0]
                                    scan_width = scan_parameters["size"][1] + flyback_pixels
                                    data_and_metadata_list = eels_view_task.grab_earliest()
                                    eels_data_and_metadata = data_and_metadata_list[1]
                                    eels_data = eels_data_and_metadata.data
                                    frame_index_base = eels_data_and_metadata.metadata["hardware_source"]["frame_index"]
                                    frame_index = eels_data_and_metadata.metadata["hardware_source"]["frame_index"] - frame_index_base
                                    while True:
                                        if self.__aborted:
                                            scan_task.cancel()
                                            break
                                        column = frame_index % scan_width
                                        row = frame_index // scan_width
                                        if data_ref.data is None:
                                            data_ref.data = numpy.zeros(scan_parameters["size"] + (eels_data.shape[0],), numpy.float)
                                        if row >= scan_height:
                                            break
                                        if column < data_ref.data.shape[1]:
                                            data_ref[row, column, :] = eels_data
                                            self.acquisition_state_changed_event.fire({"message": "update", "position": (row, column + flyback_pixels)})
                                        data_and_metadata_list = eels_view_task.grab_earliest()
                                        eels_data_and_metadata = data_and_metadata_list[1]
                                        eels_data = eels_data_and_metadata.data
                                        frame_index = eels_data_and_metadata.metadata["hardware_source"]["frame_index"] - frame_index_base
                                except:
                                    scan_task.cancel()
                                    raise
                finally:
                    self.acquisition_state_changed_event.fire({"message": "end"})
                    logging.debug("end")
            except Exception as e:
                import traceback
                traceback.print_exc()

        self.__thread = threading.Thread(target=acquire_spectrum_image, args=(self.__api, document_window))
        self.__thread.start()

    def start_sequence(self, document_window: API.DocumentWindow, sum_frames: bool) -> None:

        def acquire_sequence(api: API.API, document_window: API.DocumentWindow) -> None:
            try:
                logging.debug("start")
                self.acquisition_state_changed_event.fire({"message": "start"})
                try:
                    eels_camera_id = "orca_camera"
                    eels_camera = api.get_hardware_source_by_id(eels_camera_id, version="1.0")
                    eels_camera_parameters = eels_camera.get_frame_parameters_for_profile_by_index(0)
                    if sum_frames:
                        eels_camera_parameters["processing"] = "sum_project"

                    scan_controller = api.get_hardware_source_by_id("scan_controller", version="1.0")
                    scan_parameters = scan_controller.get_frame_parameters_for_profile_by_index(2)
                    scan_max_size = 256
                    scan_parameters["size"] = min(scan_max_size, scan_parameters["size"][0]), min(scan_max_size, scan_parameters["size"][1])
                    scan_parameters["pixel_time_us"] = int(1000 * eels_camera_parameters["exposure_ms"] * 0.75)
                    # long timeout is needed until memory allocation is outside of the acquire_sequence call.
                    scan_parameters["external_clock_wait_time_ms"] = 20000 # int(eels_camera_parameters["exposure_ms"] * 1.5)
                    scan_parameters["external_clock_mode"] = 1
                    scan_parameters["ac_line_sync"] = False
                    scan_parameters["ac_frame_sync"] = False

                    library = document_window.library

                    eels_camera.set_frame_parameters(eels_camera_parameters)
                    eels_camera._hardware_source.acquire_sequence_prepare()

                    flyback_pixels = 2
                    with contextlib.closing(scan_controller.create_record_task(scan_parameters)) as scan_task:
                        scan_height = scan_parameters["size"][0]
                        scan_width = scan_parameters["size"][1] + flyback_pixels
                        data_elements = eels_camera._hardware_source.acquire_sequence(scan_width * scan_height)
                        data_element = data_elements[0]
                        scan_data_list = scan_task.grab()
                        data_shape = data_element["data"].shape
                        data_element["data"] = data_element["data"].reshape(scan_height, scan_width, *data_shape[1:])[:, flyback_pixels:scan_width, :]
                        if len(scan_data_list) > 0:
                            collection_calibrations = [calibration.write_dict() for calibration in scan_data_list[0].dimensional_calibrations]
                        else:
                            collection_calibrations = [{}, {}]
                        if "spatial_calibrations" in data_element:
                            datum_calibrations = [copy.deepcopy(spatial_calibration) for spatial_calibration in data_element["spatial_calibrations"][1:]]
                        else:
                            datum_calibrations = [{} for i in range(len(data_element["data"].shape) - 2)]
                        # combine the dimensional calibrations from the scan data with the datum dimensions calibration from the sequence
                        data_element["collection_dimension_count"] = 2
                        data_element["spatial_calibrations"] = collection_calibrations + datum_calibrations
                        data_and_metadata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
                        def create_and_display_data_item():
                            data_item = library.get_data_item_for_hardware_source(scan_controller, channel_id=eels_camera_id, processor_id="summed", create_if_needed=True, large_format=True)
                            data_item.title = _("Spectrum Image {}".format(" x ".join([str(d) for d in data_and_metadata.dimensional_shape])))
                            data_item.set_data_and_metadata(data_and_metadata)
                            document_window.display_data_item(data_item)
                            for scan_data_and_metadata in scan_data_list:
                                scan_channel_id = scan_data_and_metadata.metadata["hardware_source"]["channel_id"]
                                scan_channel_name = scan_data_and_metadata.metadata["hardware_source"]["channel_name"]
                                channel_id = eels_camera_id + "_" + scan_channel_id
                                data_item = library.get_data_item_for_hardware_source(scan_controller, channel_id=channel_id, create_if_needed=True)
                                data_item.title = "{} ({})".format(_("Spectrum Image"), scan_channel_name)
                                data_item.set_data_and_metadata(scan_data_and_metadata)
                                document_window.display_data_item(data_item)
                        document_window.queue_task(create_and_display_data_item)  # must occur on UI thread
                finally:
                    self.acquisition_state_changed_event.fire({"message": "end"})
                    logging.debug("end")
            except Exception as e:
                import traceback
                traceback.print_exc()

        self.__thread = threading.Thread(target=acquire_sequence, args=(self.__api, document_window))
        self.__thread.start()

    def start_line_scan(self, document_controller, start, end, sample_count):

        def acquire_line_scan(api, document_controller):
            try:
                logging.debug("start line scan")
                self.acquisition_state_changed_event.fire({"message": "start"})
                try:
                    eels_camera = api.get_hardware_source_by_id("eels_camera", version="1.0")
                    eels_camera_parameters = eels_camera.get_frame_parameters_for_profile_by_index(0)

                    library = document_controller.library
                    data_item = library.create_data_item(_("Spectrum Scan"))

                    scan_controller = api.get_hardware_source_by_id("scan_controller", version="1.0")  # type: HardwareSource.HardwareSource
                    old_probe_state = scan_controller.get_property_as_str("static_probe_state")
                    old_probe_position = scan_controller.get_property_as_float_point("probe_position")

                    # force the data to be held in memory and write delayed by grabbing a data_ref.
                    with library.data_ref_for_data_item(data_item) as data_ref:
                        data = None
                        with contextlib.closing(eels_camera.create_view_task(frame_parameters=eels_camera_parameters)) as eels_view_task:
                            eels_view_task.grab_next_to_finish()
                            scan_controller.set_property_as_str("static_probe_state", "parked")
                            try:
                                for i in range(sample_count):
                                    if self.__aborted:
                                        break
                                    param = float(i) / sample_count
                                    y = start[0] + param * (end[0] - start[0])
                                    x = start[1] + param * (end[1] - start[1])
                                    logging.debug("position %s", (y, x))
                                    scan_controller.set_property_as_float_point("probe_position", (y, x))
                                    data_and_metadata = eels_view_task.grab_next_to_start()[0]
                                    if data is None:
                                        data = numpy.zeros((sample_count,) + data_and_metadata.data_shape, numpy.float)
                                        data_ref.data = data
                                    logging.debug("copying data %s %s %s", data_ref.data.shape, i, data_and_metadata.data.shape)
                                    data_ref[i] = data_and_metadata.data
                            finally:
                                scan_controller.set_property_as_str("static_probe_state", old_probe_state)
                                scan_controller.set_property_as_float_point("probe_position", old_probe_position)
                finally:
                    self.acquisition_state_changed_event.fire({"message": "end"})
                    logging.debug("end line scan")
            except Exception as e:
                import traceback
                traceback.print_exc()

        self.__thread = threading.Thread(target=acquire_line_scan, args=(self.__api, document_controller))
        self.__thread.start()

    def abort(self):
        self.__aborted = True


class PanelDelegate(object):

    def __init__(self, api):
        self.__api = api
        self.panel_id = "scan-acquisition-panel"
        self.panel_name = _("Spectrum Imaging / 4d Scan Acquisition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.__scan_acquisition_controller = None  # type: typing.Optional[ScanAcquisitionController]
        self.__line_scan_acquisition_controller = None
        self.__eels_frame_parameters_changed_event_listener = None
        self.__scan_frame_parameters_changed_event_listener = None
        self.__exposure_time_ms_value_model = None
        self.__scan_width_model = None
        self.__scan_height_model = None

    def create_panel_widget(self, ui, document_controller):
        column = ui.create_column_widget()

        old_start_button_widget = ui.create_push_button_widget(_("Start Spectrum Image"))
        old_status_label = ui.create_label_widget()
        def old_button_clicked():
            if self.__scan_acquisition_controller:
                self.__scan_acquisition_controller.abort()
            else:
                def update_button(state):
                    def update_ui():
                        if state["message"] == "start":
                            old_start_button_widget.text = _("Abort Spectrum Image")
                        elif state["message"] == "end":
                            old_start_button_widget.text = _("Start Spectrum Image")
                            old_status_label.text = _("Using parameters from Record mode.")
                        elif state["message"] == "update":
                            old_status_label.text = "{}: {}".format(_("Position"), state["position"])
                    document_controller.queue_task(update_ui)
                    if state["message"] == "end":
                        self.__acquisition_state_changed_event.close()
                        self.__acquisition_state_changed_event = None
                        self.__scan_acquisition_controller = None
                self.__scan_acquisition_controller = ScanAcquisitionController(self.__api)
                self.__acquisition_state_changed_event = self.__scan_acquisition_controller.acquisition_state_changed_event.listen(update_button)
                self.__scan_acquisition_controller.start_spectrum_image(document_controller)
        old_start_button_widget.on_clicked = old_button_clicked

        old_button_row = ui.create_row_widget()
        old_button_row.add(old_start_button_widget)
        old_button_row.add_stretch()

        old_status_row = ui.create_row_widget()
        old_status_row.add(old_status_label)
        old_status_row.add_stretch()

        old_status_label.text = _("Using parameters from Record mode.")

        line_samples = [16]

        line_button_widget = ui.create_push_button_widget(_("Start Line Scan"))
        line_samples_label = ui.create_label_widget(_("Samples"))
        line_samples_edit_widget = ui.create_line_edit_widget(str(line_samples[0]))
        line_samples_edit_widget.select_all()

        def change_line_samples(text):
            line_samples[0] = max(min(int(text), 1024), 1)
            line_samples_edit_widget.text = str(line_samples[0])
            line_samples_edit_widget.select_all()
        line_samples_edit_widget.on_editing_finished = change_line_samples

        def scan_button_clicked():
            if self.__line_scan_acquisition_controller:
                self.__line_scan_acquisition_controller.abort()
            else:
                def update_button(state):
                    def update_ui():
                        if state["message"] == "start":
                            line_button_widget.text = _("Abort Line Scan")
                        elif state["message"] == "end":
                            line_button_widget.text = _("Start Line Scan")
                    document_controller.queue_task(update_ui)
                    if state["message"] == "end":
                        self.__line_acquisition_state_changed_event.close()
                        self.__line_acquisition_state_changed_event = None
                        self.__line_scan_acquisition_controller = None
                display = document_controller.target_display
                graphics = display.selected_graphics if display else list()
                if len(graphics) == 1:
                    region = graphics[0].region
                    if region and region.type == "line-region":
                        start = region.get_property("start")
                        end = region.get_property("end")
                        # data_shape = data_item.data_and_metadata.data_shape
                        self.__line_scan_acquisition_controller = ScanAcquisitionController(self.__api)
                        self.__line_acquisition_state_changed_event = self.__line_scan_acquisition_controller.acquisition_state_changed_event.listen(update_button)
                        self.__line_scan_acquisition_controller.start_line_scan(document_controller, start, end, line_samples[0])
        line_button_widget.on_clicked = scan_button_clicked

        line_button_row = ui.create_row_widget()
        line_button_row.add(line_button_widget)
        line_button_row.add_stretch()

        line_samples_row = ui.create_row_widget()
        line_samples_row.add(line_samples_label)
        line_samples_row.add(line_samples_edit_widget)
        line_samples_row.add_stretch()

        sum_project_frames_check_box_widget = ui.create_check_box_widget(_("Sum Project Frames"))
        sum_project_frames_check_box_widget.checked = True

        sum_project_frames_row = ui.create_row_widget()
        sum_project_frames_row.add(sum_project_frames_check_box_widget)
        sum_project_frames_row.add_stretch()

        def acquire_sequence():
            self.__scan_acquisition_controller = ScanAcquisitionController(self.__api)
            self.__scan_acquisition_controller.start_sequence(document_controller, sum_project_frames_check_box_widget.checked)

        acquire_sequence_button_widget = ui.create_push_button_widget(_("Acquire"))
        acquire_sequence_button_widget.on_clicked = acquire_sequence

        self.__scan_width_widget = ui.create_line_edit_widget()
        self.__scan_height_widget = ui.create_line_edit_widget()

        self.__exposure_time_widget = ui.create_line_edit_widget()

        self.__estimate_label_widget = ui.create_label_widget()

        scan_size_row = ui.create_row_widget()
        scan_size_row.add_spacing(12)
        scan_size_row.add(ui.create_label_widget("Scan Size (pixels)"))
        scan_size_row.add_spacing(12)
        scan_size_row.add(self.__scan_width_widget)
        scan_size_row.add_spacing(12)
        scan_size_row.add(self.__scan_height_widget)
        scan_size_row.add_stretch()

        eels_exposure_row = ui.create_row_widget()
        eels_exposure_row.add_stretch()
        eels_exposure_row.add(ui.create_label_widget("Camera Exposure Time (ms)"))
        eels_exposure_row.add_spacing(12)
        eels_exposure_row.add(self.__exposure_time_widget)
        eels_exposure_row.add_spacing(12)

        estimate_row = ui.create_row_widget()
        estimate_row.add_spacing(12)
        estimate_row.add(self.__estimate_label_widget)
        estimate_row.add_stretch()

        acquire_sequence_button_row = ui.create_row_widget()
        acquire_sequence_button_row.add(acquire_sequence_button_widget)
        acquire_sequence_button_row.add_stretch()

        # column.add_spacing(8)
        # column.add(old_button_row)
        # column.add(old_status_row)
        # column.add_spacing(8)
        # column.add(line_button_row)
        # column.add(line_samples_row)
        column.add_spacing(8)
        column.add(scan_size_row)
        column.add_spacing(8)
        column.add(eels_exposure_row)
        column.add_spacing(8)
        column.add(estimate_row)
        column.add_spacing(8)
        column.add(acquire_sequence_button_row)
        column.add_spacing(8)
        column.add(sum_project_frames_row)
        column.add_spacing(8)
        column.add_stretch()

        def hardware_source_added(hardware_source):
            if hardware_source == self.__api.get_hardware_source_by_id("orca_camera", version="1.0")._hardware_source:
                self.connect_eels_controller()
            if hardware_source == self.__api.get_hardware_source_by_id("scan_controller", version="1.0")._hardware_source:
                self.connect_scan_controller()

        self.__hardware_source_added_event_listener = HardwareSourceModule.HardwareSourceManager().hardware_source_added_event.listen(hardware_source_added)

        for hardware_source in HardwareSourceModule.HardwareSourceManager().hardware_sources:
            hardware_source_added(hardware_source)

        return column

    def __update_estimate(self):
        if self.__exposure_time_ms_value_model and self.__scan_width_model and self.__scan_height_model:
            pixels = (self.__scan_width_model.value + 2) * self.__scan_height_model.value
            time_s = 2.0 * self.__exposure_time_ms_value_model.value * pixels / 1000
            if time_s > 3600:
                time_str = "{0:.1f} hours".format((int(time_s) + 3599) / 3600)
            elif time_s > 90:
                time_str = "{0:.1f} minutes".format((int(time_s) + 59) / 60)
            else:
                time_str = "{} seconds".format(int(time_s))
            memory = pixels * 2048 * 128 * 4
            if memory > 1024 * 1024 * 1024:
                size_str = "{0:.1f}GB".format(memory / (1024 * 1024 * 1024))
            elif memory > 1024 * 1024:
                size_str = "{0:.1f}MB".format(memory / (1024 * 1024))
            else:
                size_str = "{0:.1f}KB".format(memory / 1024)
            self.__estimate_label_widget.text = "Estimated Time: {0}  Size: {1}".format(time_str, size_str)
        else:
            self.__estimate_label_widget.text = None

    def connect_eels_controller(self):
        eels_controller = self.__api.get_hardware_source_by_id("orca_camera", version="1.0")._hardware_source

        self.__exposure_time_ms_value_model = Model.PropertyModel()

        def update_exposure_time_ms(exposure_time_ms):
            if exposure_time_ms > 0:
                frame_parameters = eels_controller.get_frame_parameters(0)
                frame_parameters.exposure_ms = exposure_time_ms
                eels_controller.set_frame_parameters(0, frame_parameters)
            self.__update_estimate()

        self.__exposure_time_ms_value_model.on_value_changed = update_exposure_time_ms

        exposure_time_ms_value_binding = Binding.PropertyBinding(self.__exposure_time_ms_value_model, "value", converter=Converter.FloatToStringConverter("{0:.1f}"))

        def eels_profile_parameters_changed(profile_index, frame_parameters):
            if profile_index == 0:
                self.__exposure_time_ms_value_model.value = frame_parameters.exposure_ms

        self.__eels_frame_parameters_changed_event_listener = eels_controller.frame_parameters_changed_event.listen(eels_profile_parameters_changed)

        eels_profile_parameters_changed(0, eels_controller.get_frame_parameters(0))

        self.__exposure_time_widget._widget.bind_text(exposure_time_ms_value_binding)  # the widget will close the binding

    def connect_scan_controller(self):
        scan_controller = self.__api.get_hardware_source_by_id("scan_controller", version="1.0")._hardware_source

        self.__scan_width_model = Model.PropertyModel()
        self.__scan_height_model = Model.PropertyModel()

        def update_scan_width(scan_width):
            if scan_width > 0:
                frame_parameters = scan_controller.get_frame_parameters(2)
                frame_parameters.size = frame_parameters.size[0], scan_width
                scan_controller.set_frame_parameters(2, frame_parameters)
            self.__update_estimate()

        def update_scan_height(scan_height):
            if scan_height > 0:
                frame_parameters = scan_controller.get_frame_parameters(2)
                frame_parameters.size = scan_height, frame_parameters.size[1]
                scan_controller.set_frame_parameters(2, frame_parameters)
            self.__update_estimate()

        self.__scan_width_model.on_value_changed = update_scan_width
        self.__scan_height_model.on_value_changed = update_scan_height

        scan_width_binding = Binding.PropertyBinding(self.__scan_width_model, "value", converter=Converter.IntegerToStringConverter())
        scan_height_binding = Binding.PropertyBinding(self.__scan_height_model, "value", converter=Converter.IntegerToStringConverter())

        def scan_profile_parameters_changed(profile_index, frame_parameters):
            if profile_index == 2:
                self.__scan_width_model.value = frame_parameters.size[1]
                self.__scan_height_model.value = frame_parameters.size[0]

        self.__scan_frame_parameters_changed_event_listener = scan_controller.frame_parameters_changed_event.listen(scan_profile_parameters_changed)

        scan_profile_parameters_changed(2, scan_controller.get_frame_parameters(2))

        self.__scan_width_widget._widget.bind_text(scan_width_binding)  # the widget will close the binding
        self.__scan_height_widget._widget.bind_text(scan_height_binding)  # the widget will close the binding

    def close(self):
        if self.__scan_frame_parameters_changed_event_listener:
            self.__scan_frame_parameters_changed_event_listener.close()
            self.__scan_frame_parameters_changed_event_listener = None
        if self.__eels_frame_parameters_changed_event_listener:
            self.__eels_frame_parameters_changed_event_listener.close()
            self.__eels_frame_parameters_changed_event_listener = None
        self.__hardware_source_added_event_listener.close()
        self.__hardware_source_added_event_listener = None


class ScanAcquisitionExtension(object):

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.superscan.scan-acquisition"

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
