# system imports
import contextlib
import gettext
import logging
import threading
import time

# typing
from typing import Optional

# third part imports
import numpy

# local libraries
from nion.typeshed import API_1_0 as API
from nion.typeshed import HardwareSource_1_0 as HardwareSource
from nion.typeshed import UI_1_0 as UserInterface
from nion.swift.model import HardwareSource
from nion.utils import Event

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

    def start_sequence(self, document_window: API.DocumentWindow) -> None:

        def acquire_sequence(api: API.API, document_window: API.DocumentWindow) -> None:
            try:
                logging.debug("start")
                self.acquisition_state_changed_event.fire({"message": "start"})
                try:
                    eels_camera_id = "orca_camera"
                    eels_camera = api.get_hardware_source_by_id(eels_camera_id, version="1.0")
                    eels_camera_parameters = eels_camera.get_frame_parameters_for_profile_by_index(0)

                    scan_controller = api.get_hardware_source_by_id("scan_controller", version="1.0")
                    scan_parameters = scan_controller.get_frame_parameters_for_profile_by_index(2)
                    scan_max_size = 256
                    scan_parameters["size"] = min(scan_max_size, scan_parameters["size"][0]), min(scan_max_size, scan_parameters["size"][1])
                    scan_parameters["pixel_time_us"] = int(1000 * eels_camera_parameters["exposure_ms"] * 0.75)
                    scan_parameters["external_clock_wait_time_ms"] = int(eels_camera_parameters["exposure_ms"] * 1.5)
                    scan_parameters["external_clock_mode"] = 1

                    library = document_window.library

                    flyback_pixels = 2
                    with contextlib.closing(scan_controller.create_record_task(scan_parameters)) as scan_task:
                        time.sleep(0.2)  # give the superscan time to get into first position. 200ms.
                        scan_height = scan_parameters["size"][0]
                        scan_width = scan_parameters["size"][1] + flyback_pixels
                        data_element = eels_camera._hardware_source.acquire_sequence(scan_width * scan_height)
                        data_shape = data_element["data"].shape
                        data_element["data"] = data_element["data"].reshape(scan_height, scan_width, data_shape[1])[:, 0:scan_width-flyback_pixels, :]
                        data_and_metadata = HardwareSource.convert_data_element_to_data_and_metadata(data_element)
                        def create_and_display_data_item():
                            data_item = library.get_data_item_for_hardware_source(scan_controller, channel_id=eels_camera_id, processor_id="summed", create_if_needed=True)
                            data_item.set_data_and_metadata(data_and_metadata)
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

                    scan_controller = api.get_hardware_source_by_id("scan_controller", version="1.0")  # type: HardwareSource
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
        self.panel_name = _("Scan Acquisition")
        self.panel_positions = ["left", "right"]
        self.panel_position = "right"
        self.__scan_acquisition_controller = None  # type: Optional[ScanAcquisitionController]
        self.__line_scan_acquisition_controller = None

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

        def acquire_sequence():
            self.__scan_acquisition_controller = ScanAcquisitionController(self.__api)
            self.__scan_acquisition_controller.start_sequence(document_controller)

        acquire_sequence_button_widget = ui.create_push_button_widget(_("Acquire Sequence"))
        acquire_sequence_button_widget.on_clicked = acquire_sequence

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
        column.add(acquire_sequence_button_row)
        column.add_spacing(8)
        column.add_stretch()

        return column


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
