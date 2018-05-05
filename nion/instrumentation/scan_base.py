# standard libraries
import collections
import copy
import gettext
import logging
import queue
import threading
import time
import typing
import uuid
import weakref

# third party libraries
from nion.instrumentation import stem_controller

# local libraries
from nion.swift.model import DataItem
from nion.swift.model import HardwareSource
from nion.swift.model import Utility
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Registry


_ = gettext.gettext


class ScanFrameParameters:

    def __init__(self, d=None):
        d = d or dict()
        self.size = d.get("size", (512, 512))
        self.center_nm = d.get("center_nm", (0, 0))
        self.fov_size_nm = d.get("fov_size_nm", (8, 8))
        self.pixel_time_us = d.get("pixel_time_us", 10)
        self.fov_nm = d.get("fov_nm", 8)
        self.rotation_rad = d.get("rotation_rad", 0)
        self.subscan_pixel_size = None
        self.subscan_fractional_size = None
        self.subscan_fractional_center = None
        self.external_clock_wait_time_ms = d.get("external_clock_wait_time_ms", 0)
        self.external_clock_mode = d.get("external_clock_mode", 0)  # 0=off, 1=on:rising, 2=on:falling
        self.ac_line_sync = d.get("ac_line_sync", False)
        self.ac_frame_sync = d.get("ac_frame_sync", True)
        self.flyback_time_us = d.get("flyback_time_us", 30.0)

    def as_dict(self):
        d = {
            "size": self.size,
            "center_nm": self.center_nm,
            "fov_size_nm": self.fov_size_nm,
            "pixel_time_us": self.pixel_time_us,
            "fov_nm": self.fov_nm,
            "rotation_rad": self.rotation_rad,
            "external_clock_wait_time_ms": self.external_clock_wait_time_ms,
            "external_clock_mode": self.external_clock_mode,
            "ac_line_sync": self.ac_line_sync,
            "ac_frame_sync": self.ac_frame_sync,
            "flyback_time_us": self.flyback_time_us,
        }
        if self.subscan_pixel_size is not None:
            d["subscan_pixel_size"] = self.subscan_pixel_size
        if self.subscan_fractional_size is not None:
            d["subscan_fractional_size"] = self.subscan_fractional_size
        if self.subscan_fractional_center is not None:
            d["subscan_fractional_center"] = self.subscan_fractional_center
        return d

    def __repr__(self):
        return "size pixels: " + str(self.size) +\
               "\ncenter nm: " + str(self.center_nm) +\
               "\nfov size nm: " + str(self.fov_size_nm) +\
               "\npixel time: " + str(self.pixel_time_us) +\
               "\nfield of view: " + str(self.fov_nm) +\
               "\nrotation: " + str(self.rotation_rad) +\
               "\nexternal clock wait time: " + str(self.external_clock_wait_time_ms) +\
               "\nexternal clock mode: " + str(self.external_clock_mode) +\
               "\nac line sync: " + str(self.ac_line_sync) +\
               "\nac frame sync: " + str(self.ac_frame_sync) +\
               "\nflyback time: " + str(self.flyback_time_us)


class ScanAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, stem_controller, scan_hardware_source, device, hardware_source_id: str, is_continuous: bool, subscan_enabled: bool, subscan_region, frame_parameters: ScanFrameParameters, channel_states: typing.List[typing.Any], display_name: str):
        super().__init__(is_continuous)
        self.__stem_controller = stem_controller
        self.hardware_source_id = hardware_source_id
        self.__device = device
        self.__weak_scan_hardware_source = weakref.ref(scan_hardware_source)
        self.__is_continuous = is_continuous
        self.__display_name = display_name
        self.__hardware_source_id = hardware_source_id
        self.__frame_parameters = copy.deepcopy(frame_parameters)
        self.__frame_number = None
        self.__scan_id = None
        self.__last_scan_id = None
        self.__pixels_to_skip = 0
        self.__channel_states = channel_states
        self.__last_read_time = 0
        self.__subscan_enabled = subscan_enabled
        self.__subscan_region = subscan_region

    def set_frame_parameters(self, frame_parameters):
        self.__frame_parameters = copy.deepcopy(frame_parameters)
        self.__activate_frame_parameters()

    @property
    def frame_parameters(self):
        return self.__frame_parameters

    @property
    def subscan_enabled(self):
        return self.__subscan_enabled

    @subscan_enabled.setter
    def subscan_enabled(self, value):
        self.__subscan_enabled = value
        self.__activate_frame_parameters()

    @property
    def subscan_region(self):
        return self.__subscan_region

    @subscan_region.setter
    def subscan_region(self, value):
        self.__subscan_region = value
        self.__activate_frame_parameters()

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self.__weak_scan_hardware_source()._enter_scanning_state()

        if not any(self.__device.channels_enabled):
            return False
        self._resume_acquisition()
        self.__frame_number = None
        self.__scan_id = None
        return True

    def _suspend_acquisition(self) -> None:
        super()._suspend_acquisition()
        self.__device.cancel()
        self.__device.stop()
        start_time = time.time()
        while self.__device.is_scanning and time.time() - start_time < 1.0:
            time.sleep(0.01)
        self.__last_scan_id = self.__scan_id

    def _resume_acquisition(self) -> None:
        super()._resume_acquisition()
        self.__activate_frame_parameters()
        self.__frame_number = self.__device.start_frame(self.__is_continuous)
        self.__scan_id = self.__last_scan_id
        self.__pixels_to_skip = 0

    def _abort_acquisition(self) -> None:
        super()._abort_acquisition()
        self._suspend_acquisition()

    def _mark_acquisition(self) -> None:
        super()._mark_acquisition()
        self.__device.stop()

    def _stop_acquisition(self) -> None:
        super()._stop_acquisition()
        self.__device.stop()
        start_time = time.time()
        while self.__device.is_scanning and time.time() - start_time < 1.0:
            time.sleep(0.01)
        self.__frame_number = None
        self.__scan_id = None
        self.__weak_scan_hardware_source()._exit_scanning_state()

    def _acquire_data_elements(self):

        # and now we set the calibrations for this image
        def update_calibration_metadata(data_element, data_shape, scan_id, frame_number, channel_name, channel_id, image_metadata):
            if "properties" not in data_element:
                pixel_time_us = float(image_metadata["pixel_time_us"])
                center_x_nm = float(image_metadata.get("center_x_nm", 0.0))
                center_y_nm = float(image_metadata.get("center_y_nm", 0.0))
                fov_nm = float(image_metadata["fov_nm"])
                pixel_size_nm = fov_nm / max(data_shape)
                data_element["title"] = channel_name
                data_element["version"] = 1
                data_element["channel_id"] = channel_id  # needed to match to the channel
                data_element["channel_name"] = channel_name  # needed to match to the channel
                data_element["spatial_calibrations"] = (
                    {"offset": -center_y_nm - pixel_size_nm * data_shape[0] * 0.5, "scale": pixel_size_nm, "units": "nm"},
                    {"offset": -center_x_nm - pixel_size_nm * data_shape[1] * 0.5, "scale": pixel_size_nm, "units": "nm"}
                )
                properties = dict()
                if image_metadata is not None:
                    properties["autostem"] = copy.deepcopy(image_metadata)
                exposure_s = data_shape[0] * data_shape[1] * pixel_time_us / 1000000
                properties["hardware_source_name"] = self.__display_name
                properties["hardware_source_id"] = self.__hardware_source_id
                properties["exposure"] = exposure_s
                properties["frame_index"] = frame_number
                properties["channel_id"] = channel_id  # needed for info after acquisition
                properties["channel_name"] = channel_name  # needed for info after acquisition
                properties["scan_id"] = str(scan_id)
                properties["center_x_nm"] = center_x_nm
                properties["center_y_nm"] = center_y_nm
                properties["pixel_time_us"] = pixel_time_us
                properties["fov_nm"] = fov_nm
                properties["rotation_deg"] = float(image_metadata["rotation_deg"])
                properties["ac_line_sync"] = int(image_metadata["ac_line_sync"])
                data_element["properties"] = properties

        def update_data_element(data_element, channel_index, complete, sub_area, npdata, autostem_properties, frame_number, scan_id):
            channel_name = self.__device.get_channel_name(channel_index)
            channel_id = self.__channel_states[channel_index].channel_id
            if self.__subscan_enabled:
                channel_id += ".subscan"
            update_calibration_metadata(data_element, npdata.shape, scan_id, frame_number, channel_name, channel_id, autostem_properties)
            data_element["data"] = npdata
            data_element["sub_area"] = sub_area
            data_element["state"] = "complete" if complete else "partial"
            data_element["properties"]["valid_rows"] = sub_area[0][0] + sub_area[1][0]

        def get_autostem_properties():
            autostem_properties = None
            try:
                autostem_properties = self.__stem_controller.get_autostem_properties()
            except Exception as e:
                logging.info("autostem.get_autostem_properties has failed")
            return autostem_properties

        _data_elements, complete, bad_frame, sub_area, self.__frame_number, self.__pixels_to_skip = self.__device.read_partial(self.__frame_number, self.__pixels_to_skip)

        min_period = 0.05
        current_time = time.time()
        if current_time - self.__last_read_time < min_period:
            time.sleep(min_period - (current_time - self.__last_read_time))
        self.__last_read_time = time.time()

        if not self.__scan_id:
            self.__scan_id = uuid.uuid4()

        autostem_properties = get_autostem_properties()

        # merge the _data_elements into data_elements
        data_elements = []
        for _data_element in _data_elements:
            if autostem_properties is not None:
                _data_element["properties"].update(autostem_properties)
            # calculate the valid sub area for this iteration
            channel_index = int(_data_element["properties"]["channel_id"])
            _data = _data_element["data"]
            _properties = _data_element["properties"]
            # create the 'data_element' in the format that must be returned from this method
            # '_data_element' is the format returned from the Device.
            data_element = dict()
            update_data_element(data_element, channel_index, complete, sub_area, _data, _properties, self.__frame_number, self.__scan_id)
            data_elements.append(data_element)

        if complete or bad_frame:
            # proceed to next frame
            self.__frame_number = None
            self.__scan_id = None
            self.__pixels_to_skip = 0

        return data_elements

    def __activate_frame_parameters(self):
        device_frame_parameters = copy.copy(self.__frame_parameters)
        context_size = Geometry.FloatSize.make(device_frame_parameters.size)
        device_frame_parameters.fov_size_nm = device_frame_parameters.fov_nm * context_size.aspect_ratio, device_frame_parameters.fov_nm
        if self.subscan_enabled and self.subscan_region:
            subscan_region = Geometry.FloatRect.make(self.subscan_region)
            device_frame_parameters.subscan_pixel_size = int(context_size.height * subscan_region.height), int(context_size.width * subscan_region.width)
            device_frame_parameters.subscan_fractional_size = subscan_region.height, subscan_region.width
            device_frame_parameters.subscan_fractional_center = subscan_region.center.y, subscan_region.center.x
        self.__device.set_frame_parameters(device_frame_parameters)


class ScanHardwareSource(HardwareSource.HardwareSource):

    def __init__(self, stem_controller, device, hardware_source_id: str, display_name: str):
        super().__init__(hardware_source_id, display_name)

        self.features["is_scanning"] = True

        # define events
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()
        self.probe_state_changed_event = Event.Event()
        self.channel_state_changed_event = Event.Event()

        self.__stem_controller = stem_controller

        self.__probe_state_changed_event_listener = self.__stem_controller.probe_state_changed_event.listen(self.__probe_state_changed)
        self.__subscan_state_changed_event_listener = self.__stem_controller._subscan_state_value.property_changed_event.listen(self.__subscan_state_changed)
        self.__subscan_region_changed_event_listener = self.__stem_controller._subscan_region_value.property_changed_event.listen(self.__subscan_region_changed)

        ChannelInfo = collections.namedtuple("ChannelInfo", ["channel_id", "name"])
        self.__device = device
        self.__device.on_device_state_changed = self.__device_state_changed

        # add data channel for each device channel
        channel_info_list = [ChannelInfo(self.__make_channel_id(channel_index), self.__device.get_channel_name(channel_index)) for channel_index in range(self.__device.channel_count)]
        for channel_info in channel_info_list:
            self.add_data_channel(channel_info.channel_id, channel_info.name)
        # add an associated sub-scan channel for each device channel
        for channel_index, channel_info in enumerate(channel_info_list):
            subscan_channel_index, subscan_channel_id, subscan_channel_name = self.get_subscan_channel_info(channel_index, channel_info.channel_id , channel_info.name)
            self.add_data_channel(subscan_channel_id, subscan_channel_name)

        self.__last_idle_position = None  # used for testing

        # configure the initial profiles from the device
        self.__profiles = list()
        self.__profiles.extend(self.__get_initial_profiles())
        self.__current_profile_index = self.__get_initial_profile_index()
        self.__frame_parameters = self.__profiles[0]
        self.__record_parameters = self.__profiles[2]

        self.__acquisition_task = None
        # the task queue is a list of tasks that must be executed on the UI thread. items are added to the queue
        # and executed at a later time in the __handle_executing_task_queue method.
        self.__task_queue = queue.Queue()
        self.__latest_values_lock = threading.RLock()
        self.__latest_values = dict()
        self.record_index = 1  # use to give unique name to recorded images

    def close(self):
        # thread needs to close before closing the stem controller. so use this method to
        # do it slightly out of order for this class.
        self.close_thread()
        # when overriding hardware source close, the acquisition loop may still be running
        # so nothing can be changed here that will make the acquisition loop fail.
        self.__stem_controller.disconnect_probe_connections()
        if self.__probe_state_changed_event_listener:
            self.__probe_state_changed_event_listener.close()
            self.__probe_state_changed_event_listener = None
        if self.__subscan_region_changed_event_listener:
            self.__subscan_region_changed_event_listener.close()
            self.__subscan_region_changed_event_listener = None
        super().close()

        # keep the device around until super close is called, since super
        # may do something that requires the device.
        self.__device.save_frame_parameters()
        self.__device.close()
        self.__device = None

    def periodic(self):
        self.__handle_executing_task_queue()

    def __handle_executing_task_queue(self):
        # gather the pending tasks, then execute them.
        # doing it this way prevents tasks from triggering more tasks in an endless loop.
        tasks = list()
        while not self.__task_queue.empty():
            task = self.__task_queue.get(False)
            tasks.append(task)
            self.__task_queue.task_done()
        for task in tasks:
            try:
                task()
            except Exception as e:
                import traceback
                traceback.print_exc()
                traceback.print_stack()

    @property
    def scan_device(self):
        return self.__device

    def __get_initial_profiles(self) -> typing.List[typing.Any]:
        profiles = list()
        profiles.append(self.__get_frame_parameters(0))
        profiles.append(self.__get_frame_parameters(1))
        profiles.append(self.__get_frame_parameters(2))
        return profiles

    def __get_frame_parameters(self, profile_index: int) -> ScanFrameParameters:
        return self.__device.get_profile_frame_parameters(profile_index)

    def __get_initial_profile_index(self) -> int:
        return 0

    @property
    def flyback_pixels(self):
        return self.__device.flyback_pixels

    @property
    def subscan_state(self) -> stem_controller.SubscanState:
        return self.__stem_controller._subscan_state_value.value

    @property
    def subscan_state_model(self) -> Model.PropertyModel:
        return self.__stem_controller._subscan_state_value

    @property
    def subscan_enabled(self) -> bool:
        return self.__stem_controller._subscan_state_value.value == stem_controller.SubscanState.ENABLED

    @subscan_enabled.setter
    def subscan_enabled(self, value: bool) -> None:
        self.__stem_controller._subscan_state_value.value = stem_controller.SubscanState.ENABLED if value else stem_controller.SubscanState.DISABLED

    @property
    def subscan_region(self):
        return self.__stem_controller._subscan_region_value.value

    @subscan_region.setter
    def subscan_region(self, value):
        self.__stem_controller._subscan_region_value.value = value

    def __subscan_state_changed(self, name):
        subscan_state = self.__stem_controller._subscan_state_value.value
        subscan_enabled = subscan_state == stem_controller.SubscanState.ENABLED
        subscan_region_value = self.__stem_controller._subscan_region_value
        if subscan_enabled and not subscan_region_value.value:
            subscan_region_value.value = ((0.25, 0.25), (0.5, 0.5))
        if self.__acquisition_task:
            self.__acquisition_task.subscan_enabled = subscan_enabled

    def __subscan_region_changed(self, name):
        if self.__acquisition_task:
            subscan_region = self.subscan_region
            if not subscan_region:
                self.subscan_enabled = False
            self.__acquisition_task.subscan_region = subscan_region

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        channel_count = self.__device.channel_count
        channel_states = [self.get_channel_state(i) for i in range(channel_count)]
        return ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, True, self.subscan_enabled, self.subscan_region, self.__frame_parameters, channel_states, self.display_name)

    def _view_task_updated(self, view_task):
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__record_parameters is not None
        channel_count = self.__device.channel_count
        channel_states = [self.get_channel_state(i) for i in range(channel_count)]
        return ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, False, self.subscan_enabled, self.subscan_region, self.__record_parameters, channel_states, self.display_name)

    def __update_frame_parameters(self, profile_index, frame_parameters):
        # update the frame parameters as they are changed from the low level. no need to set them.
        frame_parameters = copy.copy(frame_parameters)
        self.__profiles[profile_index] = frame_parameters
        if profile_index == self.__current_profile_index:
            self.__frame_parameters = copy.copy(frame_parameters)
        if profile_index == 2:
            self.__record_parameters = copy.copy(frame_parameters)
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def set_frame_parameters(self, profile_index, frame_parameters):
        frame_parameters = copy.copy(frame_parameters)
        self.__profiles[profile_index] = frame_parameters
        self.__device.set_profile_frame_parameters(profile_index, frame_parameters)
        if profile_index == self.__current_profile_index:
            self.set_current_frame_parameters(frame_parameters)
        if profile_index == 2:
            self.set_record_frame_parameters(frame_parameters)
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def get_frame_parameters(self, profile):
        return copy.copy(self.__profiles[profile])

    def set_current_frame_parameters(self, frame_parameters):
        if self.__acquisition_task:
            self.__acquisition_task.set_frame_parameters(frame_parameters)
        self.__frame_parameters = copy.copy(frame_parameters)

    def get_current_frame_parameters(self):
        return self.__frame_parameters

    def set_record_frame_parameters(self, frame_parameters):
        self.__record_parameters = copy.copy(frame_parameters)

    def get_record_frame_parameters(self):
        return self.__record_parameters

    @property
    def channel_count(self) -> int:
        return len(self.__device.channels_enabled)

    def get_channel_state(self, channel_index):
        channels_enabled = self.__device.channels_enabled
        assert 0 <= channel_index < len(channels_enabled)
        name = self.__device.get_channel_name(channel_index)
        return self.__make_channel_state(channel_index, name, channels_enabled[channel_index])

    def set_channel_enabled(self, channel_index, enabled):
        changed = self.__device.set_channel_enabled(channel_index, enabled)
        if changed:
            self.__channel_states_changed([self.get_channel_state(i_channel_index) for i_channel_index in range(self.channel_count)])

    def get_subscan_channel_info(self, channel_index: int, channel_id: str, channel_name: str) -> typing.Tuple[int, str, str]:
        return channel_index + self.channel_count, channel_id + ".subscan", " ".join((channel_name, _("SubScan")))

    def get_data_channel_state(self, channel_index):
        # channel indexes larger than then the channel count will be subscan channels
        if channel_index < self.channel_count:
            channel_id, name, enabled = self.get_channel_state(channel_index)
            return channel_id, name, enabled if not self.subscan_enabled else False
        else:
            channel_id, name, enabled = self.get_channel_state(channel_index - self.channel_count)
            subscan_channel_index, subscan_channel_id, subscan_channel_name = self.get_subscan_channel_info(channel_index, channel_id, name)
            return subscan_channel_id, subscan_channel_name, enabled if self.subscan_enabled else False

    def get_channel_index_for_data_channel_index(self, data_channel_index: int) -> int:
        return data_channel_index % self.channel_count

    def convert_data_channel_id_to_channel_id(self, data_channel_id: int) -> int:
        channel_count = self.channel_count
        for channel_index in range(channel_count):
            if data_channel_id == self.get_data_channel_state(channel_index)[0]:
                return channel_index
            if data_channel_id == self.get_data_channel_state(channel_index + channel_count)[0]:
                return channel_index
        assert False

    def record_async(self, callback_fn):
        """ Call this when the user clicks the record button. """
        assert callable(callback_fn)

        def record_thread():
            current_frame_time = self.get_current_frame_time()

            def handle_finished(xdatas):
                callback_fn(xdatas)

            self.start_recording(current_frame_time, finished_callback_fn=handle_finished)

        self.__thread = threading.Thread(target=record_thread)
        self.__thread.start()

    def set_selected_profile_index(self, profile_index):
        self.__current_profile_index = profile_index
        self.set_current_frame_parameters(self.__profiles[self.__current_profile_index])
        self.profile_changed_event.fire(profile_index)

    @property
    def selected_profile_index(self):
        return self.__current_profile_index

    def __profile_frame_parameters_changed(self, profile_index, frame_parameters):
        # this method will be called when the device changes parameters (via a dialog or something similar).
        # it calls __update_frame_parameters instead of set_frame_parameters so that we do _not_ update the
        # current acquisition (which can cause a cycle in that it would again set the low level values, which
        # itself wouldn't be an issue unless the user makes multiple changes in quick succession). not setting
        # current values is different semantics than the scan control panel, which _does_ set current values if
        # the current profile is selected. Hrrmmm.
        with self.__latest_values_lock:
            self.__latest_values[profile_index] = frame_parameters
        def do_update_parameters():
            with self.__latest_values_lock:
                for profile_index in self.__latest_values.keys():
                    self.__update_frame_parameters(profile_index, self.__latest_values[profile_index])
                self.__latest_values = dict()
        self.__task_queue.put(do_update_parameters)

    def __channel_states_changed(self, channel_states):
        # this method will be called when the device changes channels enabled (via dialog or script).
        # it updates the channels internally but does not send out a message to set the channels to the
        # hardware, since they're already set, and doing so can cause strange change loops.
        channel_count = self.channel_count
        assert len(channel_states) == channel_count
        def channel_states_changed():
            for channel_index, channel_state in enumerate(channel_states):
                self.channel_state_changed_event.fire(channel_index, channel_state.channel_id, channel_state.name, channel_state.enabled)
            at_least_one_enabled = False
            for channel_index in range(channel_count):
                if self.get_channel_state(channel_index).enabled:
                    at_least_one_enabled = True
                    break
            if not at_least_one_enabled:
                self.stop_playing()
        self.__task_queue.put(channel_states_changed)

    def __make_channel_id(self, channel_index) -> str:
        return "abcdefgh"[channel_index]

    def __make_channel_state(self, channel_index, channel_name, channel_enabled):
        ChannelState = collections.namedtuple("ChannelState", ["channel_id", "name", "enabled"])
        return ChannelState(self.__make_channel_id(channel_index), channel_name, channel_enabled)

    def __device_state_changed(self, profile_frame_parameters_list, device_channel_states) -> None:
        for profile_index, profile_frame_parameters in enumerate(profile_frame_parameters_list):
            self.__profile_frame_parameters_changed(profile_index, profile_frame_parameters)
        channel_states = list()
        for channel_index, (channel_name, channel_enabled) in enumerate(device_channel_states):
            channel_states.append(self.__make_channel_state(channel_index, channel_name, channel_enabled))
        self.__channel_states_changed(channel_states)

    def get_frame_parameters_from_dict(self, d):
        return ScanFrameParameters(d)

    def get_current_frame_time(self):
        frame_parameters = self.get_current_frame_parameters()
        return frame_parameters.size[0] * frame_parameters.size[1] * frame_parameters.pixel_time_us / 1000000.0

    def get_record_frame_time(self):
        frame_parameters = self.get_record_frame_parameters()
        return frame_parameters.size[0] * frame_parameters.size[1] * frame_parameters.pixel_time_us / 1000000.0

    def clean_data_item(self, data_item: DataItem.DataItem, data_channel: HardwareSource.DataChannel) -> None:
        display = data_item.displays[0]
        for graphic in copy.copy(display.graphics):
            if graphic.graphic_id == "probe":
                display.remove_graphic(graphic)
            if graphic.graphic_id == "subscan":
                display.remove_graphic(graphic)

    def get_buffer_data(self, start: int, count: int) -> typing.Optional[typing.List[typing.List[typing.Dict]]]:
        """Get recently acquired (buffered) data.

        The start parameter can be negative to index backwards from the end.

        If start refers to a buffer item that doesn't exist or if count requests too many buffer items given
        the start value, the returned list may have fewer elements than count.

        Returns None if buffering is not enabled.
        """
        if hasattr(self.__device, "get_buffer_data"):
            return self.__device.get_buffer_data(start, count)
        return None

    def __probe_state_changed(self, probe_state, probe_position):
        # subclasses will override _set_probe_position
        # probe_state can be 'parked', or 'scanning'
        self._set_probe_position(probe_position)
        # update the probe position for listeners and also explicitly update for probe_graphic_connections.
        self.probe_state_changed_event.fire(probe_state, probe_position)

    def _enter_scanning_state(self):
        """Enter scanning state. Acquisition task will call this. Tell the STEM controller."""
        self.__stem_controller._enter_scanning_state()

    def _exit_scanning_state(self):
        """Exit scanning state. Acquisition task will call this. Tell the STEM controller."""
        self.__stem_controller._exit_scanning_state()

    @property
    def probe_state(self):
        return self.__stem_controller.probe_state

    # override from BaseScanHardwareSource
    def _set_probe_position(self, probe_position):
        if probe_position is not None:
            self.__device.set_idle_position_by_percentage(probe_position.x, probe_position.y)
            self.__last_idle_position = probe_position
        else:
            # pass magic value to position to default position which may be top left or center depending on configuration.
            self.__device.set_idle_position_by_percentage(-1.0, -1.0)
            self.__last_idle_position = -1.0, -1.0

    def _get_last_idle_position_for_test(self):
        return self.__last_idle_position

    @property
    def probe_position(self):
        return self.__stem_controller.probe_position

    @probe_position.setter
    def probe_position(self, probe_position):
        self.__stem_controller.set_probe_position(probe_position)

    def validate_probe_position(self):
        self.__stem_controller.validate_probe_position()

    # override from the HardwareSource parent class.
    def data_item_states_changed(self, data_item_states):
        self.__stem_controller._data_item_states_changed(data_item_states)

    @property
    def use_hardware_simulator(self):
        return False

    def get_property(self, name):
        return getattr(self, name)

    def set_property(self, name, value):
        setattr(self, name, value)

    def open_configuration_interface(self, api_broker):
        if hasattr(self.__device, "open_configuration_interface"):
            self.__device.open_configuration_interface()
        if hasattr(self.__device, "show_configuration_dialog"):
            self.__device.show_configuration_dialog(api_broker)

    def shift_click(self, mouse_position, camera_shape):
        frame_parameters = self.__device.current_frame_parameters
        width, height = frame_parameters.size
        fov_nm = frame_parameters.fov_nm
        pixel_size_nm = fov_nm / max(width, height)
        # calculate dx, dy in meters
        dx = 1e-9 * pixel_size_nm * (mouse_position[1] - (camera_shape[1] / 2))
        dy = 1e-9 * pixel_size_nm * (mouse_position[0] - (camera_shape[0] / 2))
        logging.info("Shifting (%s,%s) um.\n", dx * 1e6, dy * 1e6)
        self.__stem_controller.change_stage_position(dy=dy, dx=dx)

    def increase_pmt(self, channel_index):
        if channel_index in (0, 1):  # df, bf
            self.__stem_controller.change_pmt_gain(channel_index, factor=2.0)

    def decrease_pmt(self, channel_index):
        if channel_index in (0, 1):  # df, bf
            self.__stem_controller.change_pmt_gain(channel_index, factor=0.5)

    def get_api(self, version):
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade:

            def __init__(self):
                pass

        return CameraFacade()


_component_registered_listener = None
_component_unregistered_listener = None

def run():
    def component_registered(component, component_types):
        if "scan_device" in component_types:
            stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id(component.stem_controller_id)
            if not stem_controller:
                print("STEM Controller (" + component.stem_controller_id + ") for (" + component.scan_device_id + ") not found. Using proxy.")
                from nion.instrumentation import stem_controller
                stem_controller = stem_controller.STEMController()
            scan_hardware_source = ScanHardwareSource(stem_controller, component, component.scan_device_id, component.scan_device_name)
            HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)

    def component_unregistered(component, component_types):
        if "scan_device" in component_types:
            HardwareSource.HardwareSourceManager().unregister_hardware_source(component)

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    for component in Registry.get_components_by_type("scan_device"):
        component_registered(component, {"scan_device"})
