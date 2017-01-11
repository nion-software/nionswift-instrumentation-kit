# standard libraries
import copy
import queue
import threading
import typing

# typing
# None

# third party libraries
# None

# local libraries
from nion.swift.model import HardwareSource
from nion.swift.model import Utility
from nion.utils import Event


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
        self.__camera_adapter.on_profile_frame_parameters_changed = self.__profile_frame_parameters_changed
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
        self.__latest_values = dict()
        self.__latest_profile_index = None

    def close(self):
        self.__camera_adapter.on_selected_profile_index_changed = None
        self.__camera_adapter.on_profile_frame_parameters_changed = None
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
    def sensor_dimensions(self):
        return self.__camera_adapter.sensor_dimensions

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
            for profile_index in self.__latest_values.keys():
                self.set_frame_parameters(profile_index, self.__latest_values[profile_index])
            self.__latest_values = dict()

    def __selected_profile_index_changed(self, profile_index):
        with self.__latest_values_lock:
            self.__latest_profile_index = profile_index
        self.__task_queue.put(self.__do_update_parameters)

    def __profile_frame_parameters_changed(self, profile_index, frame_parameters):
        with self.__latest_values_lock:
            self.__latest_values[profile_index] = frame_parameters
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

    def open_configuration_interface(self):
        self.__camera_adapter.open_configuration_interface()

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
