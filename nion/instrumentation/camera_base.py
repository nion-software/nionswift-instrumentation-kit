# standard libraries
import abc
import copy
import datetime
import gettext
import logging
import queue
import threading
import typing

# typing
# None

# third party libraries
import numpy

# local libraries
from nion.swift.model import HardwareSource
from nion.swift.model import Utility
from nion.utils import Event
from nion.utils import Registry


_ = gettext.gettext

AUTOSTEM_CONTROLLER_ID = "autostem_controller"


class CameraAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, delegate):
        super().__init__(delegate.is_continuous)
        self.__delegate = delegate

    def set_frame_parameters(self, frame_parameters):
        self.__delegate.set_frame_parameters(frame_parameters)

    @property
    def frame_parameters(self):
        return self.__delegate.frame_parameters

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        return self.__delegate.start_acquisition()

    def _suspend_acquisition(self) -> None:
        super()._suspend_acquisition()
        return self.__delegate.suspend_acquisition()

    def _resume_acquisition(self) -> None:
        super()._resume_acquisition()
        self.__delegate.resume_acquisition()

    def _mark_acquisition(self) -> None:
        super()._mark_acquisition()
        self.__delegate.mark_acquisition()

    def _stop_acquisition(self) -> None:
        super()._stop_acquisition()
        self.__delegate.stop_acquisition()

    def _acquire_data_elements(self):
        return self.__delegate.acquire_data_elements()


class CameraHardwareSource(HardwareSource.HardwareSource):

    def __init__(self, camera_adapter, periodic_logger_fn=None):
        super().__init__(camera_adapter.hardware_source_id, camera_adapter.display_name)
        self.__camera_adapter = camera_adapter
        self.__camera_adapter.on_selected_profile_index_changed = self.__selected_profile_index_changed
        self.__camera_adapter.on_profile_frame_parameter_changed = self.__profile_frame_parameter_changed
        self.features["is_camera"] = True
        self.features.update(self.__camera_adapter.features)
        self.add_data_channel()
        if self.__camera_adapter.processor:
            self.add_channel_processor(0, self.__camera_adapter.processor)
        self.log_timing = False
        self.__profiles = list()
        self.__profiles.extend(self.__camera_adapter.get_initial_profiles())
        self.__current_profile_index = self.__camera_adapter.get_initial_profile_index()
        self.__frame_parameters = self.__profiles[0]
        self.__record_parameters = self.__profiles[2]
        self.__acquisition_task = None
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()
        # the periodic logger function retrieves any log messages from the camera. it is called during
        # __handle_log_messages_event. any messages are sent out on the log_messages_event.
        self.__periodic_logger_fn = periodic_logger_fn
        self.log_messages_event = Event.Event()
        self.modes = ["Run", "Tune", "Snap"]
        # the task queue is a list of tasks that must be executed on the UI thread. items are added to the queue
        # and executed at a later time in the __handle_executing_task_queue method.
        self.__task_queue = queue.Queue()
        self.__latest_values_lock = threading.RLock()
        self.__latest_values = list()
        self.__latest_profile_index = None

    def close(self):
        self.__camera_adapter.on_selected_profile_index_changed = None
        self.__camera_adapter.on_profile_frame_parameter_changed = None
        self.__periodic_logger_fn = None
        super().close()
        # keep the camera adapter around until super close is called, since super
        # may do something that requires the camera adapter.
        self.__camera_adapter.close()
        self.__camera_adapter = None

    def periodic(self):
        self.__handle_executing_task_queue()
        self.__handle_log_messages_event()

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

    def __handle_log_messages_event(self):
        if callable(self.__periodic_logger_fn):
            messages, data_elements = self.__periodic_logger_fn()
            if len(messages) > 0 or len(data_elements) > 0:
                self.log_messages_event.fire(messages, data_elements)

    @property
    def camera_adapter(self) -> "CameraAdapter":
        return self.__camera_adapter

    @property
    def camera(self) -> "Camera":
        return self.__camera_adapter.camera

    @property
    def sensor_dimensions(self):
        return self.__camera_adapter.sensor_dimensions

    @property
    def binning_values(self) -> typing.Sequence[int]:
        return self.__camera_adapter.binning_values

    @property
    def readout_area(self):
        return self.__camera_adapter.readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR):
        self.__camera_adapter.readout_area = readout_area_TLBR

    def get_expected_dimensions(self, binning):
        return self.__camera_adapter.get_expected_dimensions(binning)

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        return CameraAcquisitionTask(self.__camera_adapter.create_acquisition_task(self.__frame_parameters))

    def _view_task_updated(self, view_task):
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__record_parameters is not None
        return CameraAcquisitionTask(self.__camera_adapter.create_record_task(self.__record_parameters))

    def acquire_sequence_prepare(self):
        self.__camera_adapter.acquire_sequence_prepare()

    def acquire_sequence(self, n: int) -> typing.Sequence[typing.Dict]:
        return self.__camera_adapter.acquire_sequence(self.get_current_frame_parameters(), n)

    def set_frame_parameters(self, profile_index, frame_parameters):
        frame_parameters = copy.copy(frame_parameters)
        self.__profiles[profile_index] = frame_parameters
        self.__camera_adapter.set_profile_frame_parameters(profile_index, frame_parameters)
        if profile_index == self.__current_profile_index:
            self.set_current_frame_parameters(frame_parameters)
        if profile_index == 2:
            self.set_record_frame_parameters(frame_parameters)
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def get_frame_parameters(self, profile_index):
        return copy.copy(self.__profiles[profile_index])

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

    def set_selected_profile_index(self, profile_index):
        if self.__current_profile_index != profile_index:
            self.__current_profile_index = profile_index
            self.__camera_adapter.set_selected_profile_index(profile_index)
            self.set_current_frame_parameters(self.__profiles[self.__current_profile_index])
            self.profile_changed_event.fire(profile_index)

    @property
    def selected_profile_index(self):
        return self.__current_profile_index

    def __do_update_parameters(self):
        with self.__latest_values_lock:
            if self.__latest_profile_index is not None:
                self.set_selected_profile_index(self.__latest_profile_index)
            self.__latest_profile_index = None
            for profile_index, parameter, value in self.__latest_values:
                frame_parameters = self.get_frame_parameters(profile_index)
                if getattr(frame_parameters, parameter) != value:
                    setattr(frame_parameters, parameter, value)
                    self.set_frame_parameters(profile_index, frame_parameters)
            self.__latest_values = list()

    def __selected_profile_index_changed(self, profile_index):
        with self.__latest_values_lock:
            self.__latest_profile_index = profile_index
        self.__task_queue.put(self.__do_update_parameters)

    def __profile_frame_parameter_changed(self, profile_index, frame_parameter, value):
        # this is the preferred technique for camera adapters to indicate changes
        with self.__latest_values_lock:
            # rebuild the list to remove anything matching our new profile_index/frame_parameter combo.
            latest_values = list()
            for profile_index_, parameter_, value_ in self.__latest_values:
                if profile_index != profile_index_ or frame_parameter != parameter_:
                    latest_values.append((profile_index_, parameter_, value_))
            self.__latest_values = latest_values
            self.__latest_values.append((profile_index, frame_parameter, value))
        self.__task_queue.put(self.__do_update_parameters)

    def get_frame_parameters_from_dict(self, d):
        return self.__camera_adapter.get_frame_parameters_from_dict(d)

    # mode property. thread safe.
    def get_mode(self):
        return self.modes[self.__current_profile_index]

    # translate the mode identifier to the mode enum if necessary.
    # set mode settings. thread safe.
    def set_mode(self, mode):
        self.set_selected_profile_index(self.modes.index(mode))

    def open_configuration_interface(self, api_broker):
        self.__camera_adapter.open_configuration_interface(api_broker)

    def open_monitor(self):
        self.__camera_adapter.open_monitor()

    def shift_click(self, mouse_position, camera_shape):
        if hasattr(self.__camera_adapter, "shift_click") and callable(self.__camera_adapter.shift_click):
            self.__camera_adapter.shift_click(mouse_position, camera_shape)

    def tilt_click(self, mouse_position, camera_shape):
        if hasattr(self.__camera_adapter, "tilt_click") and callable(self.__camera_adapter.tilt_click):
            self.__camera_adapter.tilt_click(mouse_position, camera_shape)

    def get_property(self, name):
        return self.__camera_adapter.get_property(name)

    def set_property(self, name, value):
        self.__camera_adapter.set_property(name, value)

    def get_api(self, version):
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade(object):

            def __init__(self):
                pass

        return CameraFacade()


class CameraFrameParameters(object):

    def __init__(self, d=None):
        d = d or dict()
        self.exposure_ms = d.get("exposure_ms", 125)
        self.binning = d.get("binning", 1)
        self.processing = d.get("processing")
        self.integration_count = d.get("integration_count")

    def as_dict(self):
        return {
            "exposure_ms": self.exposure_ms,
            "binning": self.binning,
            "processing": self.processing,
            "integration_count": self.integration_count,
        }


class Camera(abc.ABC):

    # TODO: dimensional and intensity calibrations should be returned at top level of data element
    # TODO: camera hardware source should query the camera for list of possible modes

    @abc.abstractmethod
    def close(self) -> None:
        """Close the camera."""
        ...

    @property
    @abc.abstractmethod
    def sensor_dimensions(self) -> (int, int):
        """Read-only property for the native sensor size (no binning).

        Returns (height, width) in pixels.

        This is a global property, meaning it affects all profiles, and is assumed to be constant.
        """
        ...

    @property
    @abc.abstractmethod
    def readout_area(self) -> (int, int, int, int):
        """Return the detector readout area.

        Accepts tuple of (top, left, bottom, right) readout rectangle, specified in sensor coordinates.

        There are restrictions on the valid values, depending on camera. This property should use the closest
        appropriate values, rounding up when necessary.

        This is a global property, meaning it affects all profiles.
        """
        ...

    @readout_area.setter
    @abc.abstractmethod
    def readout_area(self, readout_area_TLBR: (int, int, int, int)) -> None:
        """Set the detector readout area.

        The coordinates, top, left, bottom, right, are specified in sensor coordinates.

        There are restrictions on the valid values, depending on camera. This property should always return
        valid values.

        This is a global property, meaning it affects all profiles.
        """
        ...

    @property
    @abc.abstractmethod
    def flip(self):
        """Return whether data is flipped left-right (last dimension).

        This is a global property, meaning it affects all profiles.
        """
        ...

    @flip.setter
    @abc.abstractmethod
    def flip(self, do_flip):
        """Set whether data is flipped left-right (last dimension).

        This is a global property, meaning it affects all profiles.
        """
        return self._controller.SetFlip(do_flip)

    @property
    @abc.abstractmethod
    def binning_values(self) -> typing.List[int]:
        """Return a list of valid binning values (int's).

        This is a global property, meaning it affects all profiles, and is assumed to be constant.
        """
        ...

    @abc.abstractmethod
    def get_expected_dimensions(self, binning: int) -> (int, int):
        """Read-only property for the expected image size (binning and readout area included).

        Returns (height, width).

        Cameras are allowed to bin in one dimension and not the other.
        """
        ...

    @property
    @abc.abstractmethod
    def mode(self) -> str:
        """Return the current mode of the camera, as a case-insensitive string identifier.

        Cameras must currently handle the 'Run', 'Tune', and 'Snap' modes.
        """
        ...

    @mode.setter
    @abc.abstractmethod
    def mode(self, mode: str) -> None:
        """Set the current mode of the camera, using a case-insensitive string identifier.

        Cameras must currently handle the 'Run', 'Tune', and 'Snap' modes.
        """
        ...

    @property
    @abc.abstractmethod
    def mode_as_index(self) -> int:
        """Return the index of the current mode of the camera.

        Cameras must currently handle the 'Run', 'Tune', and 'Snap' modes.
        """
        ...

    @abc.abstractmethod
    def get_exposure_ms(self, mode_id: str) -> float:
        """Return the exposure (in milliseconds) for the mode."""
        ...

    @abc.abstractmethod
    def set_exposure_ms(self, exposure_ms: float, mode_id: str) -> None:
        """Set the exposure (in milliseconds) for the mode.

        Setting the exposure for the currently live mode (if there is one) should change the exposure as soon
        as possible, which may be immediately or may be the next exposed frame.
        """
        ...

    @abc.abstractmethod
    def get_binning(self, mode_id: str) -> int:
        """Return the binning for the mode."""
        ...

    @abc.abstractmethod
    def set_binning(self, binning: int, mode_id: str) -> None:
        """Set the binning for the mode.

        Binning should be one of the values returned from `binning_values`.

        Setting the binning for the currently live mode (if there is one) should change the binning as soon
        as possible, which may be immediately or may be the next frame.
        """
        ...

    def set_integration_count(self, integration_count: int, mode_id: str) -> None:
        """Set the integration code for the mode.

        Integration count can be ignored, in which case integration is performed by higher level code.

        Setting the integration count for the currently live mode (if there is one) should update acquisition as soon
        as possible, which may be immediately or may be the next frame.
        """
        pass

    @property
    @abc.abstractmethod
    def exposure_ms(self) -> float:
        """Return the exposure (in milliseconds) for the current mode."""
        ...

    @exposure_ms.setter
    @abc.abstractmethod
    def exposure_ms(self, value: float) -> None:
        """Set the exposure (in milliseconds) for the current mode."""
        ...

    @property
    @abc.abstractmethod
    def binning(self) -> int:
        """Return the binning for the current mode."""
        ...

    @binning.setter
    @abc.abstractmethod
    def binning(self, value: int) -> None:
        """Set the binning for the current mode."""
        ...

    @property
    @abc.abstractmethod
    def processing(self) -> str:
        """Return processing actions for the current mode.

        Processing may be 'sum_project' or None.
        """
        ...

    @processing.setter
    @abc.abstractmethod
    def processing(self, value: str) -> None:
        """Set processing actions for the current mode.

        Processing may be 'sum_project' or None.
        """
        ...

    @property
    @abc.abstractmethod
    def calibration(self) -> typing.List[dict]:
        """Return list of calibration for each dimension.

        Each calibration is a dict and can include 'scale', 'offset', and 'units' keys.

        This method is deprecated but must be implemented.
        """
        ...

    @abc.abstractmethod
    def start_live(self) -> None:
        """Start live acquisition. Required before using acquire_image."""
        ...

    @abc.abstractmethod
    def stop_live(self) -> None:
        """Stop live acquisition."""
        ...

    @abc.abstractmethod
    def acquire_image(self) -> dict:
        """Acquire the most recent image and return a data element dict.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict. Inside the 'properties' dict you must include 'frame_number' as an int.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).

        If integration_count is non-zero and is handled directly in this method, the 'properties' should also contain
        a 'integration_count' value to indicate how many frames were integrated. If the value is missing, a default
        value of 1 is assumed.
        """
        ...

    def acquire_sequence_prepare(self) -> None:
        """Prepare for acquire_sequence."""
        pass

    def acquire_sequence(self, n: int) -> dict:
        """Acquire a sequence of n images. Return a single data element with two dimensions n x h, w.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).
        """
        return None

    def show_config_window(self) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def show_configuration_dialog(self, api_broker) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def start_monitor(self) -> None:
        """Show a monitor dialog, if needed. Dialog can be modal or modeless."""
        pass



class CameraAdapterAcquisitionTask:

    def __init__(self, hardware_source_id, is_continuous: bool, camera: Camera, frame_parameters, display_name):
        self.hardware_source_id = hardware_source_id
        self.is_continuous = is_continuous
        self.__camera = camera
        self.__display_name = display_name
        self.__frame_parameters = None
        self.__pending_frame_parameters = copy.copy(frame_parameters)

    def set_frame_parameters(self, frame_parameters):
        self.__pending_frame_parameters = copy.copy(frame_parameters)

    @property
    def frame_parameters(self):
        return self.__pending_frame_parameters or self.__frame_parameters

    def start_acquisition(self) -> bool:
        self.resume_acquisition()
        return True

    def suspend_acquisition(self) -> None:
        pass

    def resume_acquisition(self) -> None:
        self.__activate_frame_parameters()
        self.__stop_after_acquire = False
        self.__camera.start_live()

    def mark_acquisition(self) -> None:
        self.__stop_after_acquire = True

    def stop_acquisition(self) -> None:
        self.__camera.stop_live()

    def acquire_data_elements(self):
        if self.__pending_frame_parameters:
            self.__activate_frame_parameters()
        assert self.__frame_parameters is not None
        frame_parameters = self.__frame_parameters
        exposure_ms = frame_parameters.exposure_ms
        binning = frame_parameters.binning
        integration_count = frame_parameters.integration_count if frame_parameters.integration_count else 1
        cumulative_frame_count = 0  # used for integration_count
        cumulative_data = None
        data_element = None  # avoid use-before-set warning
        while cumulative_frame_count < integration_count:
            data_element = self.__camera.acquire_image()
            frames_acquired = data_element["properties"].get("integration_count", 1)
            if cumulative_data is None:
                cumulative_data = data_element["data"]
            else:
                cumulative_data += data_element["data"]
            cumulative_frame_count += frames_acquired
            assert cumulative_frame_count <= integration_count
        if self.__stop_after_acquire:
            self.__camera.stop_live()
        sub_area = (0, 0), cumulative_data.shape
        data_element["data"] = cumulative_data
        data_element["version"] = 1
        data_element["sub_area"] = sub_area
        data_element["state"] = "complete"
        data_element["timestamp"] = data_element.get("timestamp", datetime.datetime.utcnow())
        # add optional calibration properties
        if "spatial_calibrations" in data_element["properties"]:
            data_element["spatial_calibrations"] = data_element["properties"]["spatial_calibrations"]
        else:  # handle backwards compatible case for nionccd1010
            data_element["spatial_calibrations"] = self.__camera.calibration
        if "intensity_calibration" in data_element["properties"]:
            data_element["intensity_calibration"] = data_element["properties"]["intensity_calibration"]
        # grab metadata from the autostem
        autostem = HardwareSource.HardwareSourceManager().get_instrument_by_id(AUTOSTEM_CONTROLLER_ID)
        if autostem:
            try:
                autostem_properties = autostem.get_autostem_properties()
                data_element["properties"].setdefault("autostem", dict()).update(copy.deepcopy(autostem_properties))
                # TODO: file format: remove extra_high_tension
                high_tension_v = autostem_properties.get("high_tension_v")
                if high_tension_v:
                    data_element["properties"]["extra_high_tension"] = high_tension_v
            except Exception as e:
                pass
        data_element["properties"]["hardware_source_name"] = self.__display_name
        data_element["properties"]["hardware_source_id"] = self.hardware_source_id
        data_element["properties"]["exposure"] = exposure_ms / 1000.0
        data_element["properties"]["binning"] = binning
        data_element["properties"]["valid_rows"] = sub_area[0][0] + sub_area[1][0]
        data_element["properties"]["frame_index"] = data_element["properties"]["frame_number"]
        data_element["properties"]["integration_count"] = cumulative_frame_count
        return [data_element]

    def __activate_frame_parameters(self):
        self.__frame_parameters = self.frame_parameters
        self.__pending_frame_parameters = None
        mode_id = self.__camera.mode
        self.__camera.set_exposure_ms(self.__frame_parameters.exposure_ms, mode_id)
        self.__camera.set_binning(self.__frame_parameters.binning, mode_id)
        if hasattr(self.__camera, "set_integration_count"):
            self.__camera.set_integration_count(self.__frame_parameters.integration_count, mode_id)


class CameraAdapter:

    def __init__(self, hardware_source_id, camera_category, display_name, camera: Camera):
        self.hardware_source_id = hardware_source_id
        self.display_name = display_name
        self.camera = camera
        self.modes = ["Run", "Tune", "Snap"]
        self.binning_values = self.camera.binning_values
        self.features = dict()
        self.features["is_nion_camera"] = True
        self.features["has_monitor"] = True
        self.processor = None
        if camera_category.lower() == "ronchigram":
            pass  # put ronchi-specific features here
        if camera_category.lower() == "eels":
            self.features["is_eels_camera"] = True
            self.processor = HardwareSource.SumProcessor(((0.25, 0.0), (0.5, 1.0)))
        self.on_selected_profile_index_changed = None
        self.on_profile_frame_parameter_changed = None

        # on_low_level_parameter_changed is handled for backwards compatibility (old DLLs with new hardware source).
        # new DLLs should call on_mode_changed and on_mode_parameter_changed (handled below) and should NOT call
        # on_low_level_parameter_changed.
        # handling of on_low_level_parameter_changed can be removed once all users have updated to new DLLs (2017-06-23)

        def low_level_parameter_changed(parameter_name):
            # updates all profiles with new exposure/binning values (if changed)
            # parameter_name is ignored
            profile_index = self.camera.mode_as_index
            if parameter_name == "exposureTimems" or parameter_name == "binning":
                if callable(self.on_profile_frame_parameter_changed):
                    for i, mode_id in enumerate(self.modes):
                        exposure_ms = self.camera.get_exposure_ms(mode_id)
                        binning = self.camera.get_binning(mode_id)
                        self.on_profile_frame_parameter_changed(i, "exposure_ms", exposure_ms)
                        self.on_profile_frame_parameter_changed(i, "binning", binning)
            elif parameter_name == "mode":
                if callable(self.on_selected_profile_index_changed):
                    self.on_selected_profile_index_changed(profile_index)

        self.camera.on_low_level_parameter_changed = low_level_parameter_changed

        def mode_changed(mode: str) -> None:
            for index, i_mode in enumerate(self.modes):
                if mode == i_mode:
                    if callable(self.on_selected_profile_index_changed):
                        self.on_selected_profile_index_changed(index)
                    break

        def mode_parameter_changed(mode: str, parameter_name: str, value) -> None:
            for index, i_mode in enumerate(self.modes):
                if mode == i_mode:
                    if callable(self.on_profile_frame_parameter_changed):
                        self.on_profile_frame_parameter_changed(index, parameter_name, value)
                    break

        self.camera.on_mode_changed = mode_changed
        self.camera.on_mode_parameter_changed = mode_parameter_changed

    def close(self):
        # unlisten for events from the image panel
        self.camera.on_low_level_parameter_changed = None
        self.camera.on_mode_changed = None
        self.camera.on_mode_parameter_changed = None
        close_method = getattr(self.camera, "close", None)
        if callable(close_method):
            close_method()

    def get_initial_profiles(self) -> typing.List[typing.Any]:
        # copy the frame parameters from the camera object to self.__profiles
        def get_frame_parameters(profile_index):
            mode_id = self.modes[profile_index]
            exposure_ms = self.camera.get_exposure_ms(mode_id)
            binning = self.camera.get_binning(mode_id)
            return CameraFrameParameters({"exposure_ms": exposure_ms, "binning": binning})
        return [get_frame_parameters(i) for i in range(3)]

    def get_initial_profile_index(self) -> int:
        return self.camera.mode_as_index

    def set_selected_profile_index(self, profile_index: int) -> None:
        mode_id = self.modes[profile_index]
        self.camera.mode = mode_id

    def set_profile_frame_parameters(self, profile_index: int, frame_parameters: CameraFrameParameters) -> None:
        mode_id = self.modes[profile_index]
        self.camera.set_exposure_ms(frame_parameters.exposure_ms, mode_id)
        self.camera.set_binning(frame_parameters.binning, mode_id)

    @property
    def sensor_dimensions(self):
        return self.camera.sensor_dimensions

    @property
    def readout_area(self):
        return self.camera.readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR):
        self.camera.readout_area = readout_area_TLBR

    def get_expected_dimensions(self, binning):
        return self.camera.get_expected_dimensions(binning)

    def create_acquisition_task(self, frame_parameters):
        acquisition_task = CameraAdapterAcquisitionTask(self.hardware_source_id, True, self.camera, frame_parameters, self.display_name)
        return acquisition_task

    def create_record_task(self, frame_parameters):
        record_task = CameraAdapterAcquisitionTask(self.hardware_source_id, False, self.camera, frame_parameters, self.display_name)
        return record_task

    def acquire_sequence_prepare(self):
        if callable(getattr(self.camera, "acquire_sequence_prepare", None)):
            self.camera.acquire_sequence_prepare()

    def __acquire_sequence(self, n: int):
        if callable(getattr(self.camera, "acquire_sequence", None)):
            data_element = self.camera.acquire_sequence(n)
            if data_element is not None:
                return data_element
        self.camera.start_live()
        properties = None
        data = None
        for index in range(n):
            frame_data_element = self.camera.acquire_image()
            frame_data = frame_data_element["data"]
            if data is None:
                data = numpy.empty((n,) + frame_data.shape, frame_data.dtype)
            data[index] = frame_data
            properties = frame_data_element["properties"]
        data_element = dict()
        data_element["data"] = data
        data_element["properties"] = properties
        return data_element

    def acquire_sequence(self, frame_parameters, n: int):
        self.camera.exposure_ms = frame_parameters.exposure_ms
        self.camera.binning = frame_parameters.binning
        self.camera.processing = frame_parameters.processing
        data_element = self.__acquire_sequence(n)
        data_element["version"] = 1
        data_element["state"] = "complete"
        # add optional calibration properties
        if "spatial_calibrations" in data_element["properties"]:
            data_element["spatial_calibrations"] = data_element["properties"]["spatial_calibrations"]
        else:  # handle backwards compatible case for nionccd1010
            data_element["spatial_calibrations"] = self.camera.calibration
        if "intensity_calibration" in data_element["properties"]:
            data_element["intensity_calibration"] = data_element["properties"]["intensity_calibration"]
        # grab metadata from the autostem
        autostem = HardwareSource.HardwareSourceManager().get_instrument_by_id(AUTOSTEM_CONTROLLER_ID)
        if autostem:
            try:
                autostem_properties = autostem.get_autostem_properties()
                data_element["properties"].setdefault("autostem", dict()).update(copy.deepcopy(autostem_properties))
                # TODO: file format: remove extra_high_tension
                high_tension_v = autostem_properties.get("high_tension_v")
                if high_tension_v:
                    data_element["properties"]["extra_high_tension"] = high_tension_v
            except Exception as e:
                pass
        data_element["properties"]["hardware_source_name"] = self.display_name
        data_element["properties"]["hardware_source_id"] = self.hardware_source_id
        data_element["properties"]["exposure"] = frame_parameters.exposure_ms / 1000.0
        return [data_element]

    def open_configuration_interface(self, api_broker):
        if hasattr(self.camera, "show_config_window"):
            self.camera.show_config_window()
        if hasattr(self.camera, "show_configuration_dialog"):
            self.camera.show_configuration_dialog(api_broker)

    def open_monitor(self):
        self.camera.start_monitor()

    def get_frame_parameters_from_dict(self, d):
        return CameraFrameParameters(d)

    def get_property(self, name):
        return getattr(self.camera, name)

    def set_property(self, name, value):
        setattr(self.camera, name, value)

    def shift_click(self, mouse_position, camera_shape):
        autostem = HardwareSource.HardwareSourceManager().get_instrument_by_id(AUTOSTEM_CONTROLLER_ID)
        if autostem:
            radians_per_pixel = autostem.get_value("TVPixelAngle")
            defocus_value = autostem.get_value("C10")  # get the defocus
            dx = radians_per_pixel * defocus_value * (mouse_position[1] - (camera_shape[1] / 2))
            dy = radians_per_pixel * defocus_value * (mouse_position[0] - (camera_shape[0] / 2))
            logging.info("Shifting (%s,%s) um.\n", dx * 1e6, dy * 1e6)
            autostem.set_value("SShft.x", autostem.get_value("SShft.x") - dx)
            autostem.set_value("SShft.y", autostem.get_value("SShft.y") - dy)

    def tilt_click(self, mouse_position, camera_shape):
        autostem = HardwareSource.HardwareSourceManager().get_instrument_by_id(AUTOSTEM_CONTROLLER_ID)
        if autostem:
            radians_per_pixel = autostem.get_value("TVPixelAngle")
            da = radians_per_pixel * (mouse_position[1] - (camera_shape[1] / 2))
            db = radians_per_pixel * (mouse_position[0] - (camera_shape[0] / 2))
            logging.info("Tilting (%s,%s) rad.\n", da, db)
            autostem.set_value("STilt.x", autostem.get_value("STilt.x") - da)
            autostem.set_value("STilt.y", autostem.get_value("STilt.y") - db)


_component_registered_listener = None
_component_unregistered_listener = None

def run():
    def component_registered(component, component_types):
        if "camera_device" in component_types:
            camera_adapter = CameraAdapter(component.camera_id, component.camera_type, component.camera_name, component)
            camera_hardware_source = CameraHardwareSource(camera_adapter)
            HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)

    def component_unregistered(component, component_types):
        if "camera_device" in component_types:
            HardwareSource.HardwareSourceManager().unregister_hardware_source(component)

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    for component in Registry.get_components_by_type("camera_device"):
        component_registered(component, {"camera_device"})
