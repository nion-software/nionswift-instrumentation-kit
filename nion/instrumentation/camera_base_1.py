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
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata
from nion.swift.model import HardwareSource
from nion.swift.model import Utility
from nion.utils import Event
from nion.utils import Registry


_ = gettext.gettext


class CameraAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, stem_controller, hardware_source_id, is_continuous: bool, camera, camera_category: str, frame_parameters, display_name):
        super().__init__(is_continuous)
        self.__stem_controller = stem_controller
        self.hardware_source_id = hardware_source_id
        self.is_continuous = is_continuous
        self.__camera = camera
        self.__camera_category = camera_category
        self.__display_name = display_name
        self.__frame_parameters = None
        self.__pending_frame_parameters = CameraFrameParameters(frame_parameters)

    def set_frame_parameters(self, frame_parameters):
        self.__pending_frame_parameters = CameraFrameParameters(frame_parameters)

    @property
    def frame_parameters(self):
        return self.__pending_frame_parameters or self.__frame_parameters

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self._resume_acquisition()
        return True

    def _resume_acquisition(self) -> None:
        super()._resume_acquisition()
        self.__activate_frame_parameters()
        self.__stop_after_acquire = False
        self.__camera.start_live()

    def _mark_acquisition(self) -> None:
        super()._mark_acquisition()
        self.__stop_after_acquire = True

    def _stop_acquisition(self) -> None:
        super()._stop_acquisition()
        self.__camera.stop_live()

    def _acquire_data_elements(self):
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
        # camera data is always assumed to be full frame, otherwise deal with subarea 1d and 2d
        data_element["data"] = cumulative_data
        data_element["version"] = 1
        data_element["state"] = "complete"
        data_element["timestamp"] = data_element.get("timestamp", datetime.datetime.utcnow())
        update_spatial_calibrations(data_element, self.__stem_controller, self.__camera, self.__camera_category, cumulative_data.shape, binning, binning)
        update_intensity_calibration(data_element, self.__stem_controller, self.__camera)
        update_autostem_properties(data_element, self.__stem_controller, self.__camera)
        # grab metadata from the autostem
        data_element["properties"]["hardware_source_name"] = self.__display_name
        data_element["properties"]["hardware_source_id"] = self.hardware_source_id
        data_element["properties"]["exposure"] = exposure_ms / 1000.0
        data_element["properties"]["binning"] = binning
        data_element["properties"]["valid_rows"] = cumulative_data.shape[0]
        data_element["properties"]["frame_index"] = data_element["properties"]["frame_number"]
        data_element["properties"]["integration_count"] = cumulative_frame_count
        if self.__camera_category in ("eels", "ronchigram"):
            data_element["properties"]["signal_type"] = self.__camera_category
        return [data_element]

    def __activate_frame_parameters(self):
        self.__frame_parameters = self.frame_parameters
        self.__pending_frame_parameters = None
        mode_id = self.__camera.mode
        self.__camera.set_exposure_ms(self.__frame_parameters.exposure_ms, mode_id)
        self.__camera.set_binning(self.__frame_parameters.binning, mode_id)
        self.__camera.processing = self.__frame_parameters.processing
        if hasattr(self.__camera, "set_integration_count"):
            self.__camera.set_integration_count(self.__frame_parameters.integration_count, mode_id)


class CameraHardwareSource(HardwareSource.HardwareSource):

    def __init__(self, stem_controller_id: str, camera):
        super().__init__(camera.camera_id, camera.camera_name)

        self.__stem_controller_id = stem_controller_id
        self.__stem_controller = None

        self.__camera = camera
        self.__camera_category = camera.camera_type
        self.modes = ["Run", "Tune", "Snap"]
        self.processor = None

        # configure the features
        self.features = dict()
        self.features["is_camera"] = True
        self.features["has_monitor"] = True
        if hasattr(camera, "camera_panel_type"):
            self.features["camera_panel_type"] = camera.camera_panel_type
        if self.__camera_category.lower() == "ronchigram":
            self.features["is_ronchigram_camera"] = True
        if self.__camera_category.lower() == "eels":
            self.features["is_eels_camera"] = True
            self.processor = HardwareSource.SumProcessor(((0.25, 0.0), (0.5, 1.0)))

        # on_low_level_parameter_changed is handled for backwards compatibility (old DLLs with new hardware source).
        # new DLLs should call on_mode_changed and on_mode_parameter_changed (handled below) and should NOT call
        # on_low_level_parameter_changed.
        # handling of on_low_level_parameter_changed can be removed once all users have updated to new DLLs (2017-06-23)

        def low_level_parameter_changed(parameter_name):
            # updates all profiles with new exposure/binning values (if changed)
            # parameter_name is ignored
            profile_index = self.__camera.mode_as_index
            if parameter_name == "exposureTimems" or parameter_name == "binning":
                for i, mode_id in enumerate(self.modes):
                    exposure_ms = self.__camera.get_exposure_ms(mode_id)
                    binning = self.__camera.get_binning(mode_id)
                    self.__profile_frame_parameter_changed(i, "exposure_ms", exposure_ms)
                    self.__profile_frame_parameter_changed(i, "binning", binning)
            elif parameter_name == "mode":
                self.__selected_profile_index_changed(profile_index)

        self.__camera.on_low_level_parameter_changed = low_level_parameter_changed

        def mode_changed(mode: str) -> None:
            for index, i_mode in enumerate(self.modes):
                if mode == i_mode:
                    self.__selected_profile_index_changed(index)
                    break

        def mode_parameter_changed(mode: str, parameter_name: str, value) -> None:
            for index, i_mode in enumerate(self.modes):
                if mode == i_mode:
                    self.__profile_frame_parameter_changed(index, parameter_name, value)
                    break

        self.__camera.on_mode_changed = mode_changed
        self.__camera.on_mode_parameter_changed = mode_parameter_changed

        # self.__camera_adapter.on_selected_profile_index_changed = self.__selected_profile_index_changed
        # self.__camera_adapter.on_profile_frame_parameter_changed = self.__profile_frame_parameter_changed

        # add channels
        self.add_data_channel()
        if self.processor:
            self.add_channel_processor(0, self.processor)

        # define events
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()
        self.log_messages_event = Event.Event()

        # configure profiles
        self.__profiles = list()
        self.__profiles.extend(self.__get_initial_profiles())
        self.__current_profile_index = self.__get_initial_profile_index()
        self.__frame_parameters = CameraFrameParameters(self.__profiles[0])
        self.__record_parameters = CameraFrameParameters(self.__profiles[2])

        self.__acquisition_task = None

        # the periodic logger function retrieves any log messages from the camera. it is called during
        # __handle_log_messages_event. any messages are sent out on the log_messages_event.
        periodic_logger_fn = getattr(self.__camera, "periodic_logger_fn", None)
        self.__periodic_logger_fn = periodic_logger_fn if callable(periodic_logger_fn) else None

        # the task queue is a list of tasks that must be executed on the UI thread. items are added to the queue
        # and executed at a later time in the __handle_executing_task_queue method.
        self.__task_queue = queue.Queue()
        self.__latest_values_lock = threading.RLock()
        self.__latest_values = list()
        self.__latest_profile_index = None

    def close(self):
        self.__periodic_logger_fn = None
        super().close()
        # keep the camera adapter around until super close is called, since super
        # may do something that requires the camera adapter.
        self.__camera.on_low_level_parameter_changed = None
        self.__camera.on_mode_changed = None
        self.__camera.on_mode_parameter_changed = None
        camera_close_method = getattr(self.__camera, "close", None)
        if callable(camera_close_method):
            camera_close_method()
        self.__camera = None

    def periodic(self):
        self.__handle_executing_task_queue()
        self.__handle_log_messages_event()

    def __get_stem_controller(self):
        if not self.__stem_controller:
            self.__stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id(self.__stem_controller_id)
            if not self.__stem_controller:
                print("STEM Controller (" + self.__stem_controller_id + ") for (" + self.hardware_source_id + ") not found. Using proxy.")
                from nion.instrumentation import stem_controller
                self.__stem_controller = self.__stem_controller or stem_controller.STEMController()
        return self.__stem_controller

    def __get_initial_profiles(self) -> typing.List[typing.Any]:
        # copy the frame parameters from the camera object to self.__profiles
        def get_frame_parameters(profile_index):
            mode_id = self.modes[profile_index]
            exposure_ms = self.__camera.get_exposure_ms(mode_id)
            binning = self.__camera.get_binning(mode_id)
            return CameraFrameParameters({"exposure_ms": exposure_ms, "binning": binning})
        return [get_frame_parameters(i) for i in range(3)]

    def __get_initial_profile_index(self) -> int:
        return self.__camera.mode_as_index

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

    def start_playing(self, *args, **kwargs):
        if "frame_parameters" in kwargs:
            self.set_current_frame_parameters(kwargs["frame_parameters"])
        elif len(args) == 1 and isinstance(args[0], dict):
            self.set_current_frame_parameters(args[0])
        super().start_playing(*args, **kwargs)

    def grab_next_to_start(self, *, timeout: float=None, **kwargs) -> typing.List[DataAndMetadata.DataAndMetadata]:
        self.start_playing()
        return self.get_next_xdatas_to_start(timeout)

    def grab_next_to_finish(self, *, timeout: float=None, **kwargs) -> typing.List[DataAndMetadata.DataAndMetadata]:
        self.start_playing()
        return self.get_next_xdatas_to_finish(timeout)

    def grab_sequence_prepare(self, count: int, **kwargs) -> bool:
        return False

    def grab_sequence(self, count: int, **kwargs) -> typing.Optional[typing.List[DataAndMetadata.DataAndMetadata]]:
        return None

    def grab_sequence_abort(self) -> None:
        return None

    def grab_sequence_get_progress(self) -> typing.Optional[float]:
        return None

    @property
    def camera(self):
        return self.__camera

    @property
    def sensor_dimensions(self):
        return self.__camera.sensor_dimensions

    @property
    def binning_values(self) -> typing.Sequence[int]:
        return self.__camera.binning_values

    @property
    def readout_area(self):
        return self.__camera.readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR):
        self.__camera.readout_area = readout_area_TLBR

    def get_expected_dimensions(self, binning):
        return self.__camera.get_expected_dimensions(binning)

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        return CameraAcquisitionTask(self.__get_stem_controller(), self.hardware_source_id, True, self.__camera, self.__camera_category, self.__frame_parameters, self.display_name)

    def _view_task_updated(self, view_task):
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__record_parameters is not None
        return CameraAcquisitionTask(self.__get_stem_controller(), self.hardware_source_id, False, self.__camera, self.__camera_category, self.__record_parameters, self.display_name)

    def acquire_sequence_prepare(self, n: int) -> None:
        frame_parameters = self.get_current_frame_parameters()
        self.__camera.exposure_ms = frame_parameters.exposure_ms
        self.__camera.binning = frame_parameters.binning
        self.__camera.processing = frame_parameters.processing
        if callable(getattr(self.__camera, "acquire_sequence_prepare", None)):
            self.__camera.acquire_sequence_prepare(n)

    def __acquire_sequence(self, n: int, frame_parameters) -> dict:
        if callable(getattr(self.__camera, "acquire_sequence", None)):
            data_element = self.__camera.acquire_sequence(n)
            if data_element is not None:
                return data_element
        # if the device does not implement acquire_sequence, fall back to looping acquisition.
        processing = frame_parameters.processing
        acquisition_task = CameraAcquisitionTask(self.__get_stem_controller(), self.hardware_source_id, True, self.__camera, self.__camera_category, frame_parameters, self.display_name)
        acquisition_task._start_acquisition()
        try:
            properties = None
            data = None
            for index in range(n):
                frame_data_element = acquisition_task._acquire_data_elements()[0]
                frame_data = frame_data_element["data"]
                if data is None:
                    if processing == "sum_project" and len(frame_data.shape) > 1:
                        data = numpy.empty((n,) + frame_data.shape[1:], frame_data.dtype)
                    else:
                        data = numpy.empty((n,) + frame_data.shape, frame_data.dtype)
                if processing == "sum_project" and len(frame_data.shape) > 1:
                    data[index] = Core.function_sum(DataAndMetadata.new_data_and_metadata(frame_data), 0).data
                else:
                    data[index] = frame_data
                properties = copy.deepcopy(frame_data_element["properties"])
                if processing == "sum_project":
                    properties["valid_rows"] = 1
                    spatial_properties = properties.get("spatial_calibrations")
                    if spatial_properties is not None:
                        properties["spatial_calibrations"] = spatial_properties[1:]
        finally:
            acquisition_task._stop_acquisition()
        data_element = dict()
        data_element["data"] = data
        data_element["properties"] = properties
        return data_element

    def acquire_sequence(self, n: int) -> typing.Sequence[typing.Dict]:
        frame_parameters = self.get_current_frame_parameters()
        binning = frame_parameters.binning
        data_element = self.__acquire_sequence(n, frame_parameters)
        data_element["version"] = 1
        data_element["state"] = "complete"
        stem_controller = self.__get_stem_controller()
        if "spatial_calibrations" not in data_element:
            update_spatial_calibrations(data_element, stem_controller, self.__camera, self.__camera_category, data_element["data"].shape[1:], binning, binning)
            if "spatial_calibrations" in data_element:
                data_element["spatial_calibrations"] = [dict(), ] + data_element["spatial_calibrations"]
        update_intensity_calibration(data_element, stem_controller, self.__camera)
        update_autostem_properties(data_element, stem_controller, self.__camera)
        data_element["properties"]["hardware_source_name"] = self.display_name
        data_element["properties"]["hardware_source_id"] = self.hardware_source_id
        data_element["properties"]["exposure"] = frame_parameters.exposure_ms / 1000.0
        return [data_element]

    def get_acquire_sequence_metrics(self, frame_parameters: typing.Dict) -> typing.Dict:
        if hasattr(self.__camera, "get_acquire_sequence_metrics"):
            return self.__camera.get_acquire_sequence_metrics(frame_parameters)
        return dict()

    def set_frame_parameters(self, profile_index, frame_parameters):
        frame_parameters = CameraFrameParameters(frame_parameters)
        self.__profiles[profile_index] = frame_parameters
        # update the frame parameters on the device
        mode_id = self.modes[profile_index]
        self.__camera.set_exposure_ms(frame_parameters.exposure_ms, mode_id)
        self.__camera.set_binning(frame_parameters.binning, mode_id)
        # update the local frame parameters
        if profile_index == self.__current_profile_index:
            self.set_current_frame_parameters(frame_parameters)
        if profile_index == 2:
            self.set_record_frame_parameters(frame_parameters)
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def get_frame_parameters(self, profile_index):
        return CameraFrameParameters(self.__profiles[profile_index])

    def set_current_frame_parameters(self, frame_parameters):
        if self.__acquisition_task:
            self.__acquisition_task.set_frame_parameters(frame_parameters)
        self.__frame_parameters = CameraFrameParameters(frame_parameters)

    def get_current_frame_parameters(self):
        return CameraFrameParameters(self.__frame_parameters)

    def set_record_frame_parameters(self, frame_parameters):
        self.__record_parameters = CameraFrameParameters(frame_parameters)

    def get_record_frame_parameters(self):
        return self.__record_parameters

    def set_selected_profile_index(self, profile_index):
        if self.__current_profile_index != profile_index:
            self.__current_profile_index = profile_index
            # set the camera mode on the camera device
            mode_id = self.modes[profile_index]
            self.__camera.mode = mode_id
            # set current frame parameters
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
        return CameraFrameParameters(d)

    # mode property. thread safe.
    def get_mode(self):
        return self.modes[self.__current_profile_index]

    # translate the mode identifier to the mode enum if necessary.
    # set mode settings. thread safe.
    def set_mode(self, mode):
        self.set_selected_profile_index(self.modes.index(mode))

    def open_configuration_interface(self, api_broker):
        if hasattr(self.__camera, "show_config_window"):
            self.__camera.show_config_window()
        if hasattr(self.__camera, "show_configuration_dialog"):
            self.__camera.show_configuration_dialog(api_broker)

    def open_monitor(self):
        self.__camera.start_monitor()

    def shift_click(self, mouse_position, camera_shape):
        if self.__camera_category.lower() == "ronchigram":
            stem_controller = self.__get_stem_controller()
            radians_per_pixel = stem_controller.get_value("TVPixelAngle")
            defocus_value = stem_controller.get_value("C10")  # get the defocus
            dx = radians_per_pixel * defocus_value * (mouse_position[1] - (camera_shape[1] / 2))
            dy = radians_per_pixel * defocus_value * (mouse_position[0] - (camera_shape[0] / 2))
            logging.info("Shifting (%s,%s) um.\n", dx * 1e6, dy * 1e6)
            stem_controller.set_value("SShft.x", stem_controller.get_value("SShft.x") - dx)
            stem_controller.set_value("SShft.y", stem_controller.get_value("SShft.y") - dy)

    def tilt_click(self, mouse_position, camera_shape):
        if self.__camera_category.lower() == "ronchigram":
            stem_controller = self.__get_stem_controller()
            radians_per_pixel = stem_controller.get_value("TVPixelAngle")
            da = radians_per_pixel * (mouse_position[1] - (camera_shape[1] / 2))
            db = radians_per_pixel * (mouse_position[0] - (camera_shape[0] / 2))
            logging.info("Tilting (%s,%s) rad.\n", da, db)
            stem_controller.set_value("STilt.x", stem_controller.get_value("STilt.x") - da)
            stem_controller.set_value("STilt.y", stem_controller.get_value("STilt.y") - db)

    def get_property(self, name):
        return getattr(self.__camera, name)

    def set_property(self, name, value):
        setattr(self.__camera, name, value)

    def get_api(self, version):
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade:

            def __init__(self):
                pass

        return CameraFacade()


class CameraFrameParameters(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.exposure_ms = self.get("exposure_ms", 125)
        self.binning = self.get("binning", 1)
        self.processing = self.get("processing")
        self.integration_count = self.get("integration_count")

    def __copy__(self):
        return self.__class__(copy.copy(dict(self)))

    def __deepcopy__(self, memo):
        deepcopy = self.__class__(copy.deepcopy(dict(self)))
        memo[id(self)] = deepcopy
        return deepcopy

    def as_dict(self):
        return {
            "exposure_ms": self.exposure_ms,
            "binning": self.binning,
            "processing": self.processing,
            "integration_count": self.integration_count,
        }


def get_stem_control(stem_controller, calibration_controls, key):
    if key + "_control" in calibration_controls:
        valid, value = stem_controller.TryGetVal(calibration_controls[key + "_control"])
        if valid:
            return value
    if key + "_value" in calibration_controls:
        return calibration_controls.get(key + "_value")
    return None


def build_calibration_dict(stem_controller, calibration_controls, prefix, relative_scale=1):
    scale = get_stem_control(stem_controller, calibration_controls, prefix + "_" + "scale")
    scale = scale * relative_scale if scale is not None else scale
    offset = get_stem_control(stem_controller, calibration_controls, prefix + "_" + "offset")
    units = get_stem_control(stem_controller, calibration_controls, prefix + "_" + "units")
    return Calibration.Calibration(offset, scale, units).rpc_dict


def update_spatial_calibrations(data_element, stem_controller, camera, camera_category, data_shape, scaling_x, scaling_y):
    if "spatial_calibrations" not in data_element:
        if "spatial_calibrations" in data_element["properties"]:
            data_element["spatial_calibrations"] = data_element["properties"]["spatial_calibrations"]
        elif hasattr(camera, "calibration"):  # used in nionccd1010
            data_element["spatial_calibrations"] = camera.calibration
        elif stem_controller and hasattr(camera, "calibration_controls"):
            calibration_controls = camera.calibration_controls
            x_calibration_dict = build_calibration_dict(stem_controller, calibration_controls, "x", scaling_x)
            y_calibration_dict = build_calibration_dict(stem_controller, calibration_controls, "y", scaling_y)
            if camera_category.lower() != "eels" and len(data_shape) == 2:
                y_calibration_dict["offset"] = -y_calibration_dict.get("scale", 1) * data_shape[0] * 0.5
                x_calibration_dict["offset"] = -x_calibration_dict.get("scale", 1) * data_shape[1] * 0.5
                data_element["spatial_calibrations"] = [y_calibration_dict, x_calibration_dict]
            else:
                # cover the possibility that EELS data is returned as 1D
                if len(data_shape) == 2:
                    data_element["spatial_calibrations"] = [y_calibration_dict, x_calibration_dict]
                else:
                    data_element["spatial_calibrations"] = [x_calibration_dict]


def update_intensity_calibration(data_element, stem_controller, camera):
    if "intensity_calibration" not in data_element:
        if "intensity_calibration" in data_element["properties"]:
            data_element["intensity_calibration"] = data_element["properties"]["intensity_calibration"]
        elif stem_controller and hasattr(camera, "calibration_controls"):
            calibration_controls = camera.calibration_controls
            data_element["intensity_calibration"] = build_calibration_dict(stem_controller, calibration_controls, "intensity")
    if "counts_per_electron" not in data_element:
        if stem_controller and hasattr(camera, "calibration_controls"):
            calibration_controls = camera.calibration_controls
            counts_per_electron = get_stem_control(stem_controller, calibration_controls, "counts_per_electron")
            if counts_per_electron:
                data_element["properties"]["counts_per_electron"] = counts_per_electron


def update_autostem_properties(data_element, stem_controller, camera):
    if stem_controller:
        try:
            autostem_properties = stem_controller.get_autostem_properties()
            data_element["properties"].setdefault("autostem", dict()).update(autostem_properties)
        except Exception as e:
            pass
        # give camera a chance to add additional properties not already supplied. this also gives
        # the camera a place to add properties outside of the 'autostem' dict.
        camera_update_properties = getattr(camera, "update_acquisition_properties", None)
        if callable(camera_update_properties):
            camera.update_acquisition_properties(data_element["properties"])
        if hasattr(camera, "acquisition_metatdata_groups"):
            acquisition_metatdata_groups = camera.acquisition_metatdata_groups
            stem_controller.apply_metadata_groups(data_element["properties"], acquisition_metatdata_groups)


_component_registered_listener = None
_component_unregistered_listener = None

def run():
    def component_registered(component, component_types):
        if "camera_device" in component_types:
            stem_controller_id = getattr(component, "stem_controller_id", "autostem_controller")
            camera_hardware_source = CameraHardwareSource(stem_controller_id, component)
            if hasattr(component, "priority"):
                camera_hardware_source.priority = component.priority
            component_types = {"hardware_source", "camera_hardware_source"}.union({component.camera_type + "_camera_hardware_source"})
            Registry.register_component(camera_hardware_source, component_types)
            HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)
            component.hardware_source = camera_hardware_source

    def component_unregistered(component, component_types):
        if "camera_device" in component_types:
            camera_hardware_source = component.hardware_source
            Registry.unregister_component(camera_hardware_source)
            HardwareSource.HardwareSourceManager().unregister_hardware_source(camera_hardware_source)

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    for component in Registry.get_components_by_type("camera_device"):
        component_registered(component, {"camera_device"})
