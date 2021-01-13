from __future__ import annotations

# standard libraries
import abc
import collections
import contextlib
import copy
import gettext
import logging
import math
import queue
import threading
import time
import typing
import uuid
import weakref

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import Core
from nion.instrumentation import camera_base
from nion.instrumentation import stem_controller
from nion.swift.model import HardwareSource
from nion.swift.model import ImportExportManager
from nion.swift.model import Utility
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry


_ = gettext.gettext

SubscanState = stem_controller.SubscanState
LineScanState = stem_controller.LineScanState
DriftCorrectionSettings = stem_controller.DriftCorrectionSettings

class ScanFrameParameters(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.size = self.get("size", (512, 512))
        self.center_nm = self.get("center_nm", (0, 0))
        self.fov_size_nm = self.get("fov_size_nm")  # this is a device level parameter; not used at the user level
        self.pixel_time_us = self.get("pixel_time_us", 10)
        self.fov_nm = self.get("fov_nm", 8)
        self.rotation_rad = self.get("rotation_rad", 0)
        self.subscan_pixel_size = self.get("subscan_pixel_size", None)
        self.subscan_fractional_size = self.get("subscan_fractional_size", None)
        self.subscan_fractional_center = self.get("subscan_fractional_center", None)
        self.subscan_rotation = self.get("subscan_rotation", 0.0)
        self.channel_modifier = self.get("channel_modifier")
        self.channel_override = self.get("channel_override")
        self.external_clock_wait_time_ms = self.get("external_clock_wait_time_ms", 0)
        self.external_clock_mode = self.get("external_clock_mode", 0)  # 0=off, 1=on:rising, 2=on:falling
        self.external_scan_mode = self.get("external_scan_mode", 0)  # 0=off, 1=on:rising, 2=on:falling
        self.external_scan_ratio = self.get("external_scan_ratio", 1.0)
        self.ac_line_sync = self.get("ac_line_sync", False)
        self.ac_frame_sync = self.get("ac_frame_sync", True)
        self.flyback_time_us = self.get("flyback_time_us", 30.0)

    def __copy__(self):
        return self.__class__(copy.copy(dict(self)))

    def __deepcopy__(self, memo):
        deepcopy = self.__class__(copy.deepcopy(dict(self)))
        memo[id(self)] = deepcopy
        return deepcopy

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
            "external_scan_mode": self.external_scan_mode,
            "external_scan_ratio": self.external_scan_ratio,
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
        if self.subscan_rotation:  # don't store None or 0.0
            d["subscan_rotation"] = self.subscan_rotation
        if self.channel_modifier:  # don't store None or 0.0
            d["channel_modifier"] = self.channel_modifier
        if self.channel_override:  # don't store None or 0.0
            d["channel_override"] = self.channel_override
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
               "\nexternal scan mode: " + str(self.external_scan_mode) +\
               "\nexternal scan ratio: " + str(self.external_scan_ratio) +\
               "\nac line sync: " + str(self.ac_line_sync) +\
               "\nac frame sync: " + str(self.ac_frame_sync) +\
               "\nflyback time: " + str(self.flyback_time_us) +\
               ("\nsubscan pixel size: " + str(self.subscan_pixel_size) if self.subscan_pixel_size is not None else "") +\
               ("\nsubscan fractional size: " + str(self.subscan_fractional_size) if self.subscan_fractional_size is not None else "") +\
               ("\nsubscan fractional center: " + str(self.subscan_fractional_center) if self.subscan_fractional_center is not None else "") +\
               ("\nsubscan rotation: " + str(self.subscan_rotation) if self.subscan_rotation is not None else "") +\
               ("\nchannel modifier: " + str(self.channel_modifier) if self.channel_modifier is not None else "") +\
               ("\nchannel override: " + str(self.channel_override) if self.channel_override is not None else "")


def update_scan_properties(properties: typing.MutableMapping, scan_frame_parameters: ScanFrameParameters, scan_id_str: str) -> None:
    if scan_id_str:
        properties["scan_id"] = scan_id_str
    properties["center_x_nm"] = float(scan_frame_parameters.get("center_x_nm", 0.0))
    properties["center_y_nm"] = float(scan_frame_parameters.get("center_y_nm", 0.0))
    properties["fov_nm"] = float(scan_frame_parameters["fov_nm"])
    properties["rotation"] = float(scan_frame_parameters.get("rotation_rad", math.radians(scan_frame_parameters.get("rotation_deg", 0.0))))
    properties["rotation_deg"] = math.degrees(properties["rotation"])
    properties["scan_context_size"] = tuple(scan_frame_parameters["size"])
    if scan_frame_parameters.subscan_fractional_size is not None:
        properties["subscan_fractional_size"] = tuple(scan_frame_parameters.subscan_fractional_size)
    if scan_frame_parameters.subscan_pixel_size is not None:
        properties["scan_size"] = (int(scan_frame_parameters.subscan_pixel_size[0]), int(scan_frame_parameters.subscan_pixel_size[1]))
    elif scan_frame_parameters.subscan_fractional_size is not None:
        properties["scan_size"] = (int(properties["scan_context_size"][0] * properties["subscan_fractional_size"][0]),
                                   int(properties["scan_context_size"][0] * properties["subscan_fractional_size"][1]))
    else:
        properties["scan_size"] = properties["scan_context_size"]
    if scan_frame_parameters.subscan_fractional_center is not None:
        properties["subscan_fractional_center"] = tuple(scan_frame_parameters.subscan_fractional_center)
    if scan_frame_parameters.subscan_rotation:
        properties["subscan_rotation"] = scan_frame_parameters.subscan_rotation


# set the calibrations for this image. does not touch metadata.
def update_scan_data_element(data_element, scan_frame_parameters, data_shape, channel_name, channel_id, scan_properties):
    scan_properties = copy.deepcopy(scan_properties)
    pixel_time_us = float(scan_properties["pixel_time_us"])
    line_time_us = float(scan_properties["line_time_us"]) if "line_time_us" in scan_properties else pixel_time_us * data_shape[1]
    center_x_nm = float(scan_properties.get("center_x_nm", 0.0))
    center_y_nm = float(scan_properties.get("center_y_nm", 0.0))
    fov_nm = float(scan_frame_parameters["fov_nm"])  # context fov_nm, not actual fov_nm returned from low level
    if scan_frame_parameters.size[0] > scan_frame_parameters.size[1]:
        fractional_size = scan_frame_parameters.subscan_fractional_size[0] if scan_frame_parameters.subscan_fractional_size else 1.0
        pixel_size = scan_frame_parameters.subscan_pixel_size[0] if scan_frame_parameters.subscan_pixel_size else scan_frame_parameters.size[0]
        pixel_size_nm = fov_nm * fractional_size / pixel_size
    else:
        fractional_size = scan_frame_parameters.subscan_fractional_size[1] if scan_frame_parameters.subscan_fractional_size else 1.0
        pixel_size = scan_frame_parameters.subscan_pixel_size[1] if scan_frame_parameters.subscan_pixel_size else scan_frame_parameters.size[1]
        pixel_size_nm = fov_nm * fractional_size / pixel_size
    data_element["title"] = channel_name
    data_element["version"] = 1
    data_element["channel_id"] = channel_id  # needed to match to the channel
    data_element["channel_name"] = channel_name  # needed to match to the channel
    if scan_properties.get("calibration_style") == "time":
        data_element["spatial_calibrations"] = (
            {"offset": 0.0, "scale": line_time_us / 1E6, "units": "s"},
            {"offset": 0.0, "scale": pixel_time_us / 1E6, "units": "s"}
        )
    else:
        data_element["spatial_calibrations"] = (
            {"offset": -center_y_nm - pixel_size_nm * data_shape[0] * 0.5, "scale": pixel_size_nm, "units": "nm"},
            {"offset": -center_x_nm - pixel_size_nm * data_shape[1] * 0.5, "scale": pixel_size_nm, "units": "nm"}
        )


def update_scan_metadata(scan_metadata: typing.MutableMapping, hardware_source_id: str, display_name: str, scan_frame_parameters, scan_id: typing.Optional[uuid.UUID], scan_properties: typing.Mapping) -> None:
    scan_metadata["hardware_source_id"] = hardware_source_id
    scan_metadata["hardware_source_name"] = display_name
    update_scan_properties(scan_metadata, scan_frame_parameters, str(scan_id) if scan_id else None)
    if scan_frame_parameters:
        scan_metadata["scan_device_parameters"] = dict(scan_frame_parameters)
    if scan_properties:
        scan_properties = dict(scan_properties)
        scan_properties.pop("channel_id", None)  # not part of scan description
        scan_metadata["scan_device_properties"] = scan_properties


def update_detector_metadata(detector_metadata: typing.MutableMapping, hardware_source_id: str, display_name: str, data_shape, frame_number: typing.Optional[int], channel_name: str, channel_id: str, scan_properties: typing.Mapping) -> None:
    detector_metadata["hardware_source_id"] = hardware_source_id
    detector_metadata["hardware_source_name"] = display_name
    pixel_time_us = float(scan_properties["pixel_time_us"])
    line_time_us = float(scan_properties["line_time_us"]) if "line_time_us" in scan_properties else pixel_time_us * data_shape[1]
    exposure_s = data_shape[0] * data_shape[1] * pixel_time_us / 1000000
    detector_metadata["exposure"] = exposure_s
    detector_metadata["frame_index"] = frame_number
    detector_metadata["channel_id"] = channel_id  # needed for info after acquisition
    detector_metadata["channel_name"] = channel_name  # needed for info after acquisition
    detector_metadata["pixel_time_us"] = pixel_time_us
    detector_metadata["line_time_us"] = line_time_us


class SynchronizedDataChannelInterface:
    def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, data_shape: Geometry.IntSize, dest_sub_area: Geometry.IntRect, sub_area: Geometry.IntRect, view_id) -> None: ...


class SynchronizedScanBehaviorAdjustments:
    def __init__(self):
        self.offset_nm : typing.Optional[Geometry.FloatSize] = None


class SynchronizedScanBehaviorInterface:
    def prepare_section(self) -> SynchronizedScanBehaviorAdjustments: ...


class ScanAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, stem_controller_: stem_controller.STEMController, scan_hardware_source, device,
                 hardware_source_id: str, is_continuous: bool, frame_parameters: ScanFrameParameters,
                 channel_ids: typing.List[str], display_name: str):
        # channel_ids is the channel id for each acquired channel
        # for instance, there may be 4 possible channels (0-3, a-d) and acquisition from channels 1,2
        # in that case channel_ids would be [b, c]
        super().__init__(is_continuous)
        self.__stem_controller = stem_controller_
        self.hardware_source_id = hardware_source_id
        self.__device = device
        self.__weak_scan_hardware_source = weakref.ref(scan_hardware_source)
        self.__is_continuous = is_continuous
        self.__display_name = display_name
        self.__hardware_source_id = hardware_source_id
        self.__frame_parameters = ScanFrameParameters(frame_parameters)
        self.__frame_number = None
        self.__scan_id = None
        self.__last_scan_id = None
        self.__fixed_scan_id = uuid.UUID(frame_parameters["scan_id"]) if "scan_id" in frame_parameters else None
        self.__pixels_to_skip = 0
        self.__channel_ids = channel_ids
        self.__last_read_time = 0
        self.__subscan_enabled = False

    def set_frame_parameters(self, frame_parameters):
        self.__frame_parameters = ScanFrameParameters(frame_parameters)
        self.__activate_frame_parameters()

    @property
    def frame_parameters(self):
        return self.__frame_parameters

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self.__weak_scan_hardware_source()._enter_scanning_state()
        if not any(self.__device.channels_enabled):
            return False
        self._resume_acquisition()
        self.__frame_number = None
        self.__scan_id = self.__fixed_scan_id
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

    def _request_abort_acquisition(self) -> None:
        super()._request_abort_acquisition()
        self.__device.cancel()

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
        self.__scan_id = self.__fixed_scan_id
        self.__weak_scan_hardware_source()._exit_scanning_state()

    def _acquire_data_elements(self):

        def update_data_element(data_element, complete, sub_area, npdata):
            data_element["data"] = npdata
            data_element["data_shape"] = self.__frame_parameters.get("data_shape_override")
            data_element["sub_area"] = sub_area
            data_element["dest_sub_area"] = Geometry.IntRect.make(sub_area) + Geometry.IntPoint.make(self.__frame_parameters.get("top_left_override", (0, 0)))
            data_element["state"] = self.__frame_parameters.get("state_override", "complete") if complete else "partial"
            data_element["section_state"] = "complete" if complete else "partial"
            data_element["metadata"].setdefault("hardware_source", dict())["valid_rows"] = sub_area[0][0] + sub_area[1][0]
            data_element["metadata"].setdefault("scan", dict())["valid_rows"] = sub_area[0][0] + sub_area[1][0]

        _data_elements, complete, bad_frame, sub_area, self.__frame_number, self.__pixels_to_skip = self.__device.read_partial(self.__frame_number, self.__pixels_to_skip)

        min_period = 0.05
        current_time = time.time()
        if current_time - self.__last_read_time < min_period:
            time.sleep(min_period - (current_time - self.__last_read_time))
        self.__last_read_time = time.time()

        if not self.__scan_id:
            self.__scan_id = uuid.uuid4()

        # merge the _data_elements into data_elements
        data_elements = []
        for _data_element in _data_elements:
            # calculate the valid sub area for this iteration
            channel_index = int(_data_element["properties"]["channel_id"])
            channel_id = self.__channel_ids[channel_index]
            _data = _data_element["data"]
            _scan_properties = _data_element["properties"]
            scan_id = self.__scan_id
            channel_name = self.__device.get_channel_name(channel_index)
            channel_override = self.__frame_parameters.channel_override
            channel_modifier = self.__frame_parameters.channel_modifier
            channel_id = channel_override or (channel_id + (("_" + channel_modifier) if channel_modifier else ""))

            # create the 'data_element' in the format that must be returned from this method
            # '_data_element' is the format returned from the Device.
            data_element = {"metadata": dict()}
            instrument_metadata = dict()
            update_instrument_properties(instrument_metadata, self.__stem_controller, self.__device)
            if instrument_metadata:
                data_element["metadata"].setdefault("instrument", dict()).update(instrument_metadata)
            update_scan_data_element(data_element, self.__frame_parameters, _data.shape, channel_name, channel_id, _scan_properties)
            update_scan_metadata(data_element["metadata"].setdefault("scan", dict()), self.hardware_source_id, self.__display_name, self.__frame_parameters, scan_id, _scan_properties)
            update_detector_metadata(data_element["metadata"].setdefault("hardware_source", dict()), self.hardware_source_id, self.__display_name, _data.shape, self.__frame_number, channel_name, channel_id, _scan_properties)
            update_data_element(data_element, complete, sub_area, _data)
            data_elements.append(data_element)

        if complete or bad_frame:
            # proceed to next frame
            self.__frame_number = None
            self.__scan_id = self.__fixed_scan_id
            self.__pixels_to_skip = 0

        return data_elements

    def __activate_frame_parameters(self):
        device_frame_parameters = ScanFrameParameters(self.__frame_parameters)
        context_size = Geometry.FloatSize.make(device_frame_parameters.size)
        device_frame_parameters.fov_size_nm = device_frame_parameters.fov_nm * context_size.aspect_ratio, device_frame_parameters.fov_nm
        self.__device.set_frame_parameters(device_frame_parameters)


class RecordTask:
    """Run acquisition in a thread and record the result."""

    def __init__(self, hardware_source, frame_parameters):
        self.__hardware_source = hardware_source
        self.__thread = None

        assert not self.__hardware_source.is_recording

        if frame_parameters:
            self.__hardware_source.set_record_frame_parameters(frame_parameters)

        self.__data_and_metadata_list = None
        # synchronize start of thread; if this sync doesn't occur, the task can be closed before the acquisition
        # is started. in that case a deadlock occurs because the abort doesn't apply and the thread is waiting
        # for the acquisition.
        self.__recording_started = threading.Event()

        def record_thread():
            self.__hardware_source.start_recording()
            self.__recording_started.set()
            self.__data_and_metadata_list = self.__hardware_source.get_next_xdatas_to_finish()
            self.__hardware_source.stop_recording(sync_timeout=3.0)

        self.__thread = threading.Thread(target=record_thread)
        self.__thread.start()
        self.__recording_started.wait()

    def close(self) -> None:
        if self.__thread.is_alive():
            self.__hardware_source.abort_recording()
            self.__thread.join()
        self.__data_and_metadata_list = None
        self.__recording_started = None

    @property
    def is_finished(self) -> bool:
        return not self.__thread.is_alive()

    def grab(self) -> typing.List[DataAndMetadata.DataAndMetadata]:
        self.__thread.join()
        return self.__data_and_metadata_list

    def cancel(self) -> None:
        self.__hardware_source.abort_recording()


def apply_section_rect(scan_frame_parameters: typing.MutableMapping, section_rect: Geometry.IntRect, scan_size: Geometry.IntSize, fractional_area: Geometry.FloatRect, channel_modifier: str) -> typing.MutableMapping:
    section_rect = Geometry.IntRect.make(section_rect)
    section_rect_f = section_rect.to_float_rect()
    subscan_rotation = scan_frame_parameters.get("subscan_rotation", None) or 0.0
    subscan_fractional_center0 = Geometry.FloatPoint.make(scan_frame_parameters.get("subscan_fractional_center", None) or (0.5, 0.5))
    subscan_fractional_size = Geometry.FloatSize(h=fractional_area.height * section_rect_f.height / scan_size.height,
                                                 w=fractional_area.width * section_rect_f.width / scan_size.width)
    subscan_fractional_center = Geometry.FloatPoint(y=fractional_area.top + fractional_area.height * section_rect_f.center.y / scan_size.height,
                                                    x=fractional_area.left + fractional_area.width * section_rect_f.center.x / scan_size.width)
    subscan_fractional_center = subscan_fractional_center.rotate(-subscan_rotation, subscan_fractional_center0)
    section_frame_parameters = copy.deepcopy(scan_frame_parameters)
    section_frame_parameters["section_rect"] = tuple(section_rect)
    section_frame_parameters["subscan_pixel_size"] = tuple(section_rect.size)
    section_frame_parameters["subscan_fractional_size"] = tuple(subscan_fractional_size)
    section_frame_parameters["subscan_fractional_center"] = tuple(subscan_fractional_center)
    section_frame_parameters["channel_modifier"] = channel_modifier
    section_frame_parameters["data_shape_override"] = tuple(scan_size)  # no flyback addition since this is data from scan device
    section_frame_parameters["state_override"] = "complete" if section_rect.bottom == scan_size[0] and section_rect.right == scan_size[1] else "partial"
    section_frame_parameters["top_left_override"] = tuple(section_rect.top_left)
    return section_frame_parameters


def crop_and_calibrate(uncropped_xdata: DataAndMetadata.DataAndMetadata, flyback_pixels: int,
                       scan_calibrations: typing.Optional[DataAndMetadata.CalibrationListType],
                       data_calibrations: typing.Optional[DataAndMetadata.CalibrationListType],
                       data_intensity_calibration: typing.Optional[Calibration.Calibration],
                       metadata: typing.Mapping) -> DataAndMetadata.DataAndMetadata:
    data_shape = uncropped_xdata.data_shape
    scan_shape = uncropped_xdata.collection_dimension_shape
    scan_calibrations = scan_calibrations or uncropped_xdata.collection_dimensional_calibrations
    if flyback_pixels > 0:
        data = uncropped_xdata.data.reshape(*scan_shape, *data_shape[len(scan_shape):])[:, flyback_pixels:scan_shape[1], :]
    else:
        data = uncropped_xdata.data.reshape(*scan_shape, *data_shape[len(scan_shape):])
    dimensional_calibrations = tuple(scan_calibrations) + tuple(data_calibrations)
    return DataAndMetadata.new_data_and_metadata(data, data_intensity_calibration,
                                                 dimensional_calibrations,
                                                 metadata, None,
                                                 uncropped_xdata.data_descriptor, None,
                                                 None)


class ScanHardwareSource(HardwareSource.HardwareSource):

    def __init__(self, stem_controller_: stem_controller.STEMController, device, hardware_source_id: str, display_name: str):
        super().__init__(hardware_source_id, display_name)

        self.features["is_scanning"] = True

        # define events
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()
        self.probe_state_changed_event = Event.Event()
        self.channel_state_changed_event = Event.Event()

        self.__stem_controller = stem_controller_

        self.__probe_state_changed_event_listener = self.__stem_controller.probe_state_changed_event.listen(self.__probe_state_changed)

        self.__subscan_state_changed_event_listener = self.__stem_controller.property_changed_event.listen(self.__subscan_state_changed)
        self.__subscan_region_changed_event_listener = self.__stem_controller.property_changed_event.listen(self.__subscan_region_changed)
        self.__subscan_rotation_changed_event_listener = self.__stem_controller.property_changed_event.listen(self.__subscan_rotation_changed)

        self.__line_scan_state_changed_event_listener = self.__stem_controller.property_changed_event.listen(self.__line_scan_state_changed)
        self.__line_scan_vector_changed_event_listener = self.__stem_controller.property_changed_event.listen(self.__line_scan_vector_changed)

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
        self.add_data_channel("drift", _("Drift"))

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

        # synchronized acquisition
        self.__camera_hardware_source = None
        self.__grab_synchronized_is_scanning = False
        self.__grab_synchronized_aborted = False  # set this flag when abort requested in case low level doesn't follow rules
        self.acquisition_state_changed_event = Event.Event()

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
        if self.__subscan_rotation_changed_event_listener:
            self.__subscan_rotation_changed_event_listener.close()
            self.__subscan_rotation_changed_event_listener = None
        if self.__line_scan_vector_changed_event_listener:
            self.__line_scan_vector_changed_event_listener.close()
            self.__line_scan_vector_changed_event_listener = None
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
    def scan_context(self) -> stem_controller.ScanContext:
        return self.__stem_controller.scan_context

    @property
    def stem_controller(self) -> stem_controller.STEMController:
        return self.__stem_controller

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

    def start_playing(self, *args, **kwargs):
        if "frame_parameters" in kwargs:
            self.set_current_frame_parameters(kwargs["frame_parameters"])
        elif len(args) == 1 and isinstance(args[0], dict):
            self.set_current_frame_parameters(args[0])
        super().start_playing(*args, **kwargs)

    def get_enabled_channels(self) -> typing.Sequence[int]:
        indexes = list()
        for index, enabled in enumerate(self.__device.channels_enabled):
            if enabled:
                indexes.append(index)
        return indexes

    def set_enabled_channels(self, channel_indexes: typing.Sequence[int]) -> None:
        for index in range(self.channel_count):
            self.set_channel_enabled(index, index in channel_indexes)

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
        pass

    def grab_sequence_get_progress(self) -> typing.Optional[float]:
        return None

    AxesDescriptor = collections.namedtuple("AxesDescriptor", ["sequence_axes", "collection_axes", "data_axes"])

    GrabSynchronizedInfo = collections.namedtuple("GrabSynchronizedInfo",
                                                  ["scan_size",
                                                   "fractional_area",
                                                   "is_subscan",
                                                   "camera_readout_size",
                                                   "camera_readout_size_squeezed",
                                                   "channel_modifier",
                                                   "scan_calibrations",
                                                   "data_calibrations",
                                                   "data_intensity_calibration",
                                                   "instrument_metadata",
                                                   "camera_metadata",
                                                   "scan_metadata",
                                                   "axes_descriptor"])

    def grab_synchronized_get_info(self, *, scan_frame_parameters: dict, camera: camera_base.CameraHardwareSource, camera_frame_parameters: camera_base.CameraFrameParameters) -> GrabSynchronizedInfo:

        scan_max_area = 2048 * 2048
        if scan_frame_parameters.get("subscan_pixel_size"):
            scan_param_height = int(scan_frame_parameters["subscan_pixel_size"][0])
            scan_param_width = int(scan_frame_parameters["subscan_pixel_size"][1])
            if scan_param_height * scan_param_width > scan_max_area:
                scan_param_height = scan_max_area // scan_param_width
            fractional_size = Geometry.FloatSize.make(scan_frame_parameters["subscan_fractional_size"])
            fractional_area = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint.make(scan_frame_parameters["subscan_fractional_center"]), fractional_size)
            is_subscan = True
            channel_modifier = "subscan"
        else:
            scan_param_height = int(scan_frame_parameters["size"][0])
            scan_param_width = int(scan_frame_parameters["size"][1])
            if scan_param_height * scan_param_width > scan_max_area:
                scan_param_height = scan_max_area // scan_param_width
            fractional_area = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.5, x=0.5), Geometry.FloatSize(h=1.0, w=1.0))
            is_subscan = False
            channel_modifier = None

        camera_readout_size = Geometry.IntSize.make(camera.get_expected_dimensions(camera_frame_parameters.get("binning", 1)))

        scan_size = Geometry.IntSize(h=scan_param_height, w=scan_param_width)

        if camera_frame_parameters.get("processing") == "sum_project":
            camera_readout_size_squeezed = (camera_readout_size.width,)
            axes_descriptor = ScanHardwareSource.AxesDescriptor(None, [0, 1], [2])
        elif camera_frame_parameters.get("processing") == "sum_masked":
            camera_readout_size_squeezed = (len(camera_frame_parameters.get("active_masks", [0])),)
            axes_descriptor = ScanHardwareSource.AxesDescriptor([2], None, [0, 1])
        else:
            camera_readout_size_squeezed = tuple(camera_readout_size)
            axes_descriptor = ScanHardwareSource.AxesDescriptor(None, [0, 1], [2, 3])

        scan_shape = Geometry.IntSize.make(scan_frame_parameters["size"])
        center_x_nm = float(scan_frame_parameters.get("center_x_nm", 0.0))
        center_y_nm = float(scan_frame_parameters.get("center_y_nm", 0.0))
        fov_nm = float(scan_frame_parameters["fov_nm"])
        pixel_size_nm = fov_nm / max(scan_shape)

        scan_calibrations = (
            Calibration.Calibration(-center_y_nm - pixel_size_nm * scan_shape[0] * 0.5, pixel_size_nm, "nm"),
            Calibration.Calibration(-center_x_nm - pixel_size_nm * scan_shape[1] * 0.5, pixel_size_nm, "nm")
        )

        data_calibrations = camera.get_camera_calibrations(camera_frame_parameters)
        data_intensity_calibration = camera.get_camera_intensity_calibration(camera_frame_parameters)

        camera_metadata = dict()
        camera.update_camera_properties(camera_metadata, camera_frame_parameters)

        scan_metadata = dict()
        scan_metadata["hardware_source_name"] = self.display_name
        scan_metadata["hardware_source_id"] = self.hardware_source_id
        update_scan_metadata(scan_metadata, self.hardware_source_id, self.display_name, ScanFrameParameters(scan_frame_parameters), uuid.UUID(scan_frame_parameters["scan_id"]), dict())

        instrument_metadata = dict()
        update_instrument_properties(instrument_metadata, self.__stem_controller, self.__device)

        return ScanHardwareSource.GrabSynchronizedInfo(scan_size, fractional_area, is_subscan, camera_readout_size,
                                                       camera_readout_size_squeezed, channel_modifier,
                                                       scan_calibrations, data_calibrations, data_intensity_calibration,
                                                       instrument_metadata, camera_metadata, scan_metadata,
                                                       axes_descriptor)

    GrabSynchronizedResult = typing.Optional[typing.Tuple[typing.List[DataAndMetadata.DataAndMetadata], typing.List[DataAndMetadata.DataAndMetadata]]]

    def grab_synchronized(self, *, scan_frame_parameters: dict, camera: camera_base.CameraHardwareSource,
                          camera_frame_parameters: dict, camera_data_channel: SynchronizedDataChannelInterface = None,
                          section_height: int = None, scan_behavior: SynchronizedScanBehaviorInterface = None) -> GrabSynchronizedResult:
        camera_frame_parameters = camera_base.CameraFrameParameters(camera_frame_parameters)
        camera_hardware_source = camera
        self.__camera_hardware_source = camera_hardware_source
        try:
            self.__stem_controller._enter_synchronized_state(self, camera=camera_hardware_source)
            self.__grab_synchronized_is_scanning = True
            self.acquisition_state_changed_event.fire(self.__grab_synchronized_is_scanning)
            old_record_parameters = self.get_record_frame_parameters()
            scan_frame_parameters = ScanFrameParameters(scan_frame_parameters)
            scan_frame_parameters.setdefault("scan_id", str(uuid.uuid4()))
            try:
                scan_info = self.grab_synchronized_get_info(scan_frame_parameters=scan_frame_parameters, camera=camera_hardware_source, camera_frame_parameters=camera_frame_parameters)
                if scan_info.is_subscan:
                    scan_frame_parameters["subscan_pixel_size"] = tuple(scan_info.scan_size)
                else:
                    scan_frame_parameters["size"] = tuple(scan_info.scan_size)
                self.__device.prepare_synchronized_scan(scan_frame_parameters, camera_exposure_ms=camera_frame_parameters["exposure_ms"])
                flyback_pixels = self.__device.flyback_pixels
                scan_size = scan_info.scan_size
                scan_param_height, scan_param_width = tuple(scan_size)
                scan_height = scan_param_height
                scan_width = scan_param_width + flyback_pixels
                scan_calibrations = scan_info.scan_calibrations
                data_calibrations = scan_info.data_calibrations
                data_intensity_calibration = scan_info.data_intensity_calibration

                # abort the scan to not interfere with setup; and clear the aborted flag
                self.abort_playing()
                self.__grab_synchronized_aborted = False

                aborted = False
                data_and_metadata_list = list()  # only used (for return value) if camera_data_channel is None
                scan_data_list_list = list()
                section_height = section_height or scan_height
                section_count = (scan_height + section_height - 1) // section_height
                for section in range(section_count):
                    # print(f"************ {section}/{section_count}")
                    section_rect = Geometry.IntRect.from_tlhw(section * section_height, 0, min(section_height, scan_height - section * section_height), scan_param_width)
                    if scan_behavior:
                        adjustments = scan_behavior.prepare_section()
                        if adjustments.offset_nm:
                            # offset_nm will be in the context reference frame.
                            scan_frame_parameters.center_nm = tuple(Geometry.FloatPoint.make(scan_frame_parameters.center_nm) + adjustments.offset_nm)
                    scan_shape = (section_rect.height, scan_width)  # includes flyback pixels
                    camera_hardware_source.set_current_frame_parameters(camera_frame_parameters)
                    camera_hardware_source.acquire_synchronized_prepare(scan_shape)

                    # generate frame parameters for the section
                    section_frame_parameters = apply_section_rect(scan_frame_parameters, section_rect, scan_size, scan_info.fractional_area, scan_info.channel_modifier)

                    with contextlib.closing(RecordTask(self, section_frame_parameters)) as scan_task:
                        is_last_section = section_rect.bottom == scan_size[0] and section_rect.right == scan_size[1]
                        partial_data_info = camera_hardware_source.acquire_synchronized_begin(camera_frame_parameters, scan_shape)
                        try:
                            uncropped_xdata = partial_data_info.xdata
                            is_complete = partial_data_info.is_complete
                            is_canceled = partial_data_info.is_canceled
                            last_valid_rows = 0  # valid rows within section
                            # this loop is awkward because to make it easy to implement synchronized begin in a backwards
                            # compatible manner, it must return all of its data on the first call. this means that we need
                            # to handle the data by sending it to the channel. and this leads to the awkward implementation
                            # below.
                            while not is_canceled and not self.__grab_synchronized_aborted:
                                # xdata is the full data and includes flyback pixels. crop the flyback pixels in the
                                # next line, but retain other metadata.
                                valid_rows = partial_data_info.valid_rows
                                src_top_row = last_valid_rows
                                metadata = copy.deepcopy(uncropped_xdata.metadata)
                                # this is a hack to prevent some of the potentially misleading metadata
                                # from getting saved into the synchronized data. while it is acceptable to
                                # assume that the hardware_source properties will get copied to the final
                                # metadata for now, camera implementers should be aware that this is likely
                                # to change behavior in the future. please write tests if you make this
                                # assumption so that they fail when this behavior is changed.
                                metadata.setdefault("hardware_source", dict()).pop("frame_number", None)
                                metadata.setdefault("hardware_source", dict()).pop("integration_count", None)
                                metadata.setdefault("hardware_source", dict()).pop("valid_rows", None)
                                metadata.setdefault("hardware_source", dict()).update(copy.deepcopy(scan_info.camera_metadata))
                                metadata["scan"] = copy.deepcopy(scan_info.scan_metadata)
                                metadata["instrument"] = copy.deepcopy(scan_info.instrument_metadata)
                                metadata["scan"]["valid_rows"] = section_rect[0][0] + valid_rows
                                partial_xdata = crop_and_calibrate(uncropped_xdata, flyback_pixels, scan_calibrations, data_calibrations, data_intensity_calibration, metadata)
                                if camera_data_channel and valid_rows > 0:
                                    data_channel_state = "complete" if is_complete and is_last_section else "partial"
                                    data_channel_data_and_metadata = partial_xdata
                                    # src rect is the area of the section data that will be copied
                                    partial_size = Geometry.IntSize(height=valid_rows - last_valid_rows, width=section_rect.width)
                                    src_rect = Geometry.IntRect(Geometry.IntPoint(y=src_top_row), partial_size)
                                    # dst rect is the destination in the final layout.
                                    dst_rect = Geometry.IntRect(Geometry.IntPoint(y=last_valid_rows), partial_size) + section_rect.origin
                                    data_channel_view_id = None
                                    camera_data_channel.update(data_channel_data_and_metadata, data_channel_state,
                                                               Geometry.IntSize(h=scan_param_height, w=scan_param_width),
                                                               dst_rect, src_rect, data_channel_view_id)
                                    last_valid_rows = valid_rows
                                # break out if we're complete
                                if is_complete:
                                    break
                                # otherwise, acquire the next section and continue
                                update_period = camera_data_channel._update_period if hasattr(camera_data_channel, "_update_period") else 1.0
                                partial_data_info = camera_hardware_source.acquire_synchronized_continue(update_period=update_period)
                                is_complete = partial_data_info.is_complete
                                is_canceled = partial_data_info.is_canceled
                                uncropped_xdata = partial_data_info.xdata
                                # unless it's cancelled or aborted, of course.
                                if not uncropped_xdata or is_canceled or self.__grab_synchronized_aborted:
                                    break
                        finally:
                            camera_hardware_source.acquire_synchronized_end()
                        if uncropped_xdata and not is_canceled and not self.__grab_synchronized_aborted:
                            # not aborted
                            # the data_element['data'] ndarray may point to low level memory; we need to get it to disk
                            # quickly. see note below.
                            scan_data_list = scan_task.grab()
                            metadata = copy.deepcopy(uncropped_xdata.metadata)
                            metadata["hardware_source"] = copy.deepcopy(scan_info.camera_metadata)
                            metadata["scan"] = copy.deepcopy(scan_info.scan_metadata)
                            metadata["instrument"] = copy.deepcopy(scan_info.instrument_metadata)
                            metadata["scan"]["valid_rows"] = scan_frame_parameters["size"][1]
                            section_xdata = crop_and_calibrate(uncropped_xdata, flyback_pixels, scan_calibrations, data_calibrations, data_intensity_calibration, metadata)
                            data_and_metadata_list.append(section_xdata)
                            scan_data_list_list.append([scan_data[section_rect.slice] for scan_data in scan_data_list])
                        else:
                            # aborted
                            scan_task.cancel()
                            aborted = True
                            break
                if not aborted:
                    new_scan_data_list = list()
                    if len(scan_data_list_list) > 0:
                        for i in range(len(scan_data_list_list[0])):
                            data_list = [scan_data_list[i] for scan_data_list in scan_data_list_list]
                            scan_data = Core.function_vstack(data_list) if len(data_list) > 1 else data_list[0]
                            scan_data._set_metadata(scan_data_list[i].metadata)
                            new_scan_data_list.append(scan_data)
                    """
                    [s1a, s1b, s2c]
                    [s2a, s2b, s2c]
                    [s3a, s3b, s3c]

                    [s1a, s2a, s3a], etc.
                    """
                    # only return the camera data if camera data channel was not passed in
                    if not camera_data_channel:
                        camera_data_and_metadata = Core.function_vstack(data_and_metadata_list) if len(data_and_metadata_list) > 1 else data_and_metadata_list[0]
                        camera_metadata = data_and_metadata_list[0].metadata
                        camera_data_and_metadata._set_metadata(camera_metadata)
                        return new_scan_data_list, [camera_data_and_metadata]
                    else:
                        return new_scan_data_list, []
                return None
            finally:
                self.set_record_frame_parameters(old_record_parameters)
                self.__stem_controller._exit_synchronized_state(self, camera=camera_hardware_source)
                self.__grab_synchronized_is_scanning = False
                self.acquisition_state_changed_event.fire(self.__grab_synchronized_is_scanning)
                logging.debug("end sequence acquisition")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise

    def grab_synchronized_abort(self) -> None:
        if self.__grab_synchronized_is_scanning:
            # if the state is scanning, the thread could be stuck on acquire sequence or
            # stuck on scan.grab. cancel both here.
            self.__camera_hardware_source.acquire_sequence_cancel()
            self.abort_recording()
        # and set the flag for misbehaving acquire_sequence return values.
        self.__grab_synchronized_aborted = True

    def grab_synchronized_get_progress(self) -> typing.Optional[float]:
        return None

    def grab_buffer(self, count: int, *, start: int=None, **kwargs) -> typing.Optional[typing.List[typing.List[DataAndMetadata.DataAndMetadata]]]:
        if start is None and count is not None:
            assert count > 0
            start = -count
        if start is not None and count is None:
            assert start < 0
            count = -start
        data_element_groups = self.get_buffer_data(start, count)
        if data_element_groups is None:
            return None
        xdata_group_list = list()
        for data_element_group in data_element_groups:
            xdata_group = list()
            for data_element in data_element_group:
                xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
                xdata_group.append(xdata)
            xdata_group_list.append(xdata_group)
        return xdata_group_list

    @property
    def flyback_pixels(self):
        return self.__device.flyback_pixels

    @property
    def subscan_state(self) -> SubscanState:
        return self.__stem_controller.subscan_state

    @property
    def subscan_enabled(self) -> bool:
        return self.__stem_controller.subscan_state == stem_controller.SubscanState.ENABLED

    @subscan_enabled.setter
    def subscan_enabled(self, enabled: bool) -> None:
        if enabled:
            self.__stem_controller.subscan_state = stem_controller.SubscanState.ENABLED
        else:
            self.__stem_controller.subscan_state = stem_controller.SubscanState.DISABLED
            self.__stem_controller._update_scan_context(self.__frame_parameters.size, self.__frame_parameters.center_nm, self.__frame_parameters.fov_nm, self.__frame_parameters.rotation_rad)

    @property
    def subscan_region(self) -> typing.Optional[Geometry.FloatRect]:
        return self.__stem_controller.subscan_region

    @subscan_region.setter
    def subscan_region(self, value: typing.Optional[Geometry.FloatRect]) -> None:
        self.__stem_controller.subscan_region = value

    @property
    def subscan_rotation(self) -> float:
        return self.__stem_controller.subscan_rotation

    @subscan_rotation.setter
    def subscan_rotation(self, value: float) -> None:
        self.__stem_controller.subscan_rotation = value

    @property
    def line_scan_state(self) -> LineScanState:
        return self.__stem_controller.line_scan_state

    @property
    def line_scan_enabled(self) -> bool:
        return self.__stem_controller.line_scan_state == stem_controller.LineScanState.ENABLED

    @line_scan_enabled.setter
    def line_scan_enabled(self, enabled: bool) -> None:
        if enabled:
            self.__stem_controller.line_scan_state = stem_controller.LineScanState.ENABLED
        else:
            self.__stem_controller.line_scan_state = stem_controller.LineScanState.DISABLED
            self.__stem_controller._update_scan_context(self.__frame_parameters.size, self.__frame_parameters.center_nm, self.__frame_parameters.fov_nm, self.__frame_parameters.rotation_rad)

    @property
    def line_scan_vector(self) -> typing.Optional[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]:
        return self.__stem_controller.line_scan_vector

    @line_scan_vector.setter
    def line_scan_vector(self, value: typing.Optional[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]) -> None:
        self.__stem_controller.line_scan_region = value

    def apply_scan_context_subscan(self, frame_parameters: ScanFrameParameters, size: typing.Tuple[int, int]) -> None:
        scan_context = self.scan_context
        if scan_context.is_valid:
            frame_parameters.size = tuple(scan_context.size)
            frame_parameters.center_nm = tuple(scan_context.center_nm)
            frame_parameters.fov_nm = scan_context.fov_nm
            frame_parameters.rotation_rad = scan_context.rotation_rad
        self.__apply_subscan_parameters(frame_parameters, size)

    def __apply_subscan_parameters(self, frame_parameters: ScanFrameParameters, size: typing.Optional[typing.Tuple[int, int]] = None) -> None:
        context_size = Geometry.FloatSize.make(frame_parameters.size)
        if self.subscan_enabled and self.subscan_region:
            subscan_region = self.subscan_region
            subscan_pixel_size = size or (max(int(context_size.height * subscan_region.height), 1), max(int(context_size.width * subscan_region.width), 1))
            frame_parameters.subscan_pixel_size = subscan_pixel_size
            frame_parameters.subscan_fractional_size = max(subscan_region.height, 1 / context_size.height), max(subscan_region.width, 1 / context_size.width)
            frame_parameters.subscan_fractional_center = subscan_region.center.y, subscan_region.center.x
            frame_parameters.subscan_rotation = self.subscan_rotation
        elif self.line_scan_enabled and self.line_scan_vector:
            line_scan_vector = self.line_scan_vector
            start = Geometry.FloatPoint.make(line_scan_vector[0])
            end = Geometry.FloatPoint.make(line_scan_vector[1])
            length = Geometry.distance(start, end)
            length = max(length, max(1 / context_size.width, 1 / context_size.height))
            subscan_pixel_length = max(int(context_size.width * length), 1)
            subscan_pixel_size = size or (1, subscan_pixel_length)
            frame_parameters.subscan_pixel_size = subscan_pixel_size
            frame_parameters.subscan_fractional_size = 1 / context_size.height, length
            frame_parameters.subscan_fractional_center = tuple(Geometry.midpoint(start, end))
            frame_parameters.subscan_rotation = -math.atan2(end.y - start.y, end.x - start.x)
        else:
            size = size or Geometry.IntSize.make(tuple(context_size))
            frame_parameters.size = tuple(size)
            frame_parameters.subscan_pixel_size = None
            frame_parameters.subscan_fractional_size = None
            frame_parameters.subscan_fractional_center = None
            frame_parameters.subscan_rotation = 0.0

    def apply_subscan(self, frame_parameters: ScanFrameParameters) -> None:
        if frame_parameters.get("subscan_fractional_size") and frame_parameters.get("subscan_fractional_center"):
            pass  # let the parameters speak for themselves
        else:
            self.__apply_subscan_parameters(frame_parameters)

    def __subscan_state_changed(self, name: str) -> None:
        if name == "subscan_state":
            # if subscan enabled, ensure there is a subscan region
            if self.__stem_controller.subscan_state == stem_controller.SubscanState.ENABLED and not self.__stem_controller.subscan_region:
                self.__stem_controller.subscan_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
                self.__stem_controller.subscan_rotation = 0.0
            # otherwise let __set_current_frame_parameters clean up existing __frame_parameters
            self.__set_current_frame_parameters(self.__frame_parameters, False)

    def __subscan_region_changed(self, name: str) -> None:
        if name == "subscan_region":
            subscan_region = self.subscan_region
            if not subscan_region:
                self.subscan_enabled = False
            self.__set_current_frame_parameters(self.__frame_parameters, False)

    def __subscan_rotation_changed(self, name: str) -> None:
        if name == "subscan_rotation":
            self.__set_current_frame_parameters(self.__frame_parameters, False)

    def __line_scan_state_changed(self, name: str) -> None:
        if name == "line_scan_state":
            # if line scan enabled, ensure there is a line scan region
            if self.__stem_controller.line_scan_state == stem_controller.LineScanState.ENABLED and not self.__stem_controller.line_scan_vector:
                self.__stem_controller.line_scan_vector = (0.25, 0.25), (0.75, 0.75)
            # otherwise let __set_current_frame_parameters clean up existing __frame_parameters
            self.__set_current_frame_parameters(self.__frame_parameters, False)

    def __line_scan_vector_changed(self, name: str) -> None:
        if name == "line_scan_vector":
            line_scan_vector = self.line_scan_vector
            if not line_scan_vector:
                self.line_scan_enabled = False
            self.__set_current_frame_parameters(self.__frame_parameters, False)

    @property
    def drift_channel_id(self) -> typing.Optional[str]:
        return self.__stem_controller.drift_channel_id

    @drift_channel_id.setter
    def drift_channel_id(self, value: typing.Optional[str]) -> None:
        self.__stem_controller.drift_channel_id = value

    @property
    def drift_region(self) -> typing.Optional[Geometry.FloatRect]:
        return self.__stem_controller.drift_region

    @drift_region.setter
    def drift_region(self, value: typing.Optional[Geometry.FloatRect]) -> None:
        self.__stem_controller.drift_region = value

    @property
    def drift_rotation(self) -> float:
        return self.__stem_controller.drift_rotation

    @drift_rotation.setter
    def drift_rotation(self, value: float) -> None:
        self.__stem_controller.drift_rotation = value

    @property
    def drift_settings(self) -> DriftCorrectionSettings:
        return self.__stem_controller.drift_settings

    @drift_settings.setter
    def drift_settings(self, value: DriftCorrectionSettings) -> None:
        self.__stem_controller.drift_settings = value

    @property
    def drift_enabled(self) -> bool:
        return self.drift_channel_id is not None and self.drift_region is not None

    @drift_enabled.setter
    def drift_enabled(self, enabled: bool) -> None:
        if enabled:
            if not self.drift_channel_id:
                self.drift_channel_id = self.data_channels[0].channel_id
            if not self.drift_region:
                self.drift_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.25, x=0.75), Geometry.FloatSize(h=0.25, w=0.25))
        else:
            self.drift_channel_id = None
            self.drift_region = None

    @property
    def drift_valid(self) -> bool:
        return self.drift_enabled and self.drift_settings.interval > 0

    def calculate_drift_lines(self, width: int, frame_time: float) -> int:
        if self.drift_valid:
            assert isinstance(self.drift_settings.interval_units, stem_controller.DriftIntervalUnit)
            if self.drift_settings.interval_units == stem_controller.DriftIntervalUnit.FRAME:
                lines = max(1, math.ceil(self.drift_settings.interval / width))
            elif self.drift_settings.interval_units == stem_controller.DriftIntervalUnit.TIME:
                lines = max(1, math.ceil(self.drift_settings.interval / frame_time / width))
            else:
                lines = self.drift_settings.interval
            return int(lines)
        return 0

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        channel_count = self.__device.channel_count
        channel_states = [self.get_channel_state(i) for i in range(channel_count)]
        if not self.subscan_enabled:
            self.__stem_controller._update_scan_context(self.__frame_parameters.size, self.__frame_parameters.center_nm, self.__frame_parameters.fov_nm, self.__frame_parameters.rotation_rad)
        frame_parameters = copy.deepcopy(self.__frame_parameters)
        channel_ids = [channel_state.channel_id for channel_state in channel_states]
        return ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, True, frame_parameters, channel_ids, self.display_name)

    def _view_task_updated(self, view_task):
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__record_parameters is not None
        channel_count = self.__device.channel_count
        channel_states = [self.get_channel_state(i) for i in range(channel_count)]
        frame_parameters = copy.deepcopy(self.__record_parameters)
        channel_ids = [channel_state.channel_id for channel_state in channel_states]
        return ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, False, frame_parameters, channel_ids, self.display_name)

    def record_immediate(self, frame_parameters: ScanFrameParameters, enabled_channels: typing.Sequence[int] = None,
                         sync_timeout: float = None) -> typing.List[DataAndMetadata.DataAndMetadata]:
        assert not self.is_recording
        frame_parameters = copy.deepcopy(frame_parameters)
        old_enabled_channels = self.get_enabled_channels()
        channel_states = [self.get_channel_state(i) for i in range(self.__device.channel_count)]
        channel_ids = [channel_state.channel_id for channel_state in channel_states]
        if enabled_channels is not None:
            self.set_enabled_channels(enabled_channels)
        record_task = ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, False, frame_parameters, channel_ids, self.display_name)
        finished_event = threading.Event()
        xdatas = list()
        def finished(xdatas_: typing.Sequence[DataAndMetadata.DataAndMetadata]) -> None:
            nonlocal xdatas
            xdatas = [copy.deepcopy(xdata) for xdata in xdatas_]  # low level may be reused; copy here
            self.set_enabled_channels(old_enabled_channels)
            finished_event.set()
        record_task.finished_callback_fn = finished
        self._record_task_updated(record_task)
        self.start_task('record', record_task)
        # loop will break on finished or error (not recording). maybe a race condition?
        while self.is_recording:
            if finished_event.wait(0.01):  # 10 msec
                break
        # self.stop_task('record')
        self._record_task_updated(None)
        sync_timeout = sync_timeout or 3.0
        start = time.time()
        while self.is_recording:
            time.sleep(0.01)  # 10 msec
            assert time.time() - start < float(sync_timeout)
        return xdatas

    def set_frame_parameters(self, profile_index, frame_parameters):
        frame_parameters = ScanFrameParameters(frame_parameters)
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
        self.__set_current_frame_parameters(frame_parameters, True)

    def __set_current_frame_parameters(self, frame_parameters, is_context: bool, update_task: bool = True) -> None:
        frame_parameters = ScanFrameParameters(frame_parameters)
        self.__apply_subscan_parameters(frame_parameters)
        if frame_parameters.subscan_pixel_size:
            frame_parameters.channel_modifier = "subscan"
        else:
            frame_parameters.channel_modifier = None
        if self.__acquisition_task:
            if update_task:
                self.__acquisition_task.set_frame_parameters(frame_parameters)
            if not self.subscan_enabled:
                self.__stem_controller._update_scan_context(frame_parameters.size, frame_parameters.center_nm, frame_parameters.fov_nm, frame_parameters.rotation_rad)
            elif is_context:
                self.__stem_controller._clear_scan_context()
        elif update_task:
            # handle case where current profile has been changed but scan is not running.
            device_frame_parameters = ScanFrameParameters(frame_parameters)
            context_size = Geometry.FloatSize.make(device_frame_parameters.size)
            device_frame_parameters.fov_size_nm = device_frame_parameters.fov_nm * context_size.aspect_ratio, device_frame_parameters.fov_nm
            self.__device.set_frame_parameters(device_frame_parameters)
            self.__stem_controller._confirm_scan_context(frame_parameters.size, frame_parameters.center_nm, frame_parameters.fov_nm, frame_parameters.rotation_rad)
        self.__frame_parameters = ScanFrameParameters(frame_parameters)

    def get_current_frame_parameters(self) -> ScanFrameParameters:
        return ScanFrameParameters(self.__frame_parameters)

    def set_record_frame_parameters(self, frame_parameters):
        self.__record_parameters = ScanFrameParameters(frame_parameters)

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

    def get_channel_enabled(self, channel_index: int) -> bool:
        assert 0 <= channel_index < self.__device.channel_count
        return self.__device.channels_enabled[channel_index]

    def set_channel_enabled(self, channel_index, enabled):
        changed = self.__device.set_channel_enabled(channel_index, enabled)
        if changed:
            self.__channel_states_changed([self.get_channel_state(i) for i in range(self.channel_count)])

    def get_subscan_channel_info(self, channel_index: int, channel_id: str, channel_name: str) -> typing.Tuple[int, str, str]:
        return channel_index + self.channel_count, channel_id + "_subscan", " ".join((channel_name, _("SubScan")))

    def get_data_channel_state(self, channel_index) -> typing.Tuple[str, str, bool]:
        # channel indexes larger than then the channel count will be subscan channels
        if channel_index < self.channel_count:
            channel_id, name, enabled = self.get_channel_state(channel_index)
            return channel_id, name, enabled if not self.subscan_enabled else False
        elif channel_index < self.channel_count * 2:
            channel_id, name, enabled = self.get_channel_state(channel_index - self.channel_count)
            subscan_channel_index, subscan_channel_id, subscan_channel_name = self.get_subscan_channel_info(channel_index, channel_id, name)
            return subscan_channel_id, subscan_channel_name, enabled if self.subscan_enabled else False
        else:
            return self.data_channels[channel_index].channel_id, self.data_channels[channel_index].name, False

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

    def __update_frame_parameters(self, profile_index, frame_parameters):
        # update the frame parameters as they are changed from the low level.
        # the low level frame parameters should be treated as if they are updating
        # the existing profiles since the low level device has no way to know
        # the complete set of frame parameters being used at this level.
        frame_parameters = ScanFrameParameters(frame_parameters)
        self.__profiles[profile_index].update(frame_parameters)
        if profile_index == self.__current_profile_index:
            self.__frame_parameters.update(frame_parameters)
            # validates the frame parameters (applies subscan, etc.)
            self.__set_current_frame_parameters(self.__frame_parameters, False, update_task=False)
        if profile_index == 2:
            self.__record_parameters.update(frame_parameters)
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def _update_frame_parameters_test(self, profile_index, frame_parameters):
        self.__update_frame_parameters(profile_index, frame_parameters)

    def __profile_frame_parameters_changed(self, profile_index, frame_parameters):
        # this method will be called when the device changes parameters (via a dialog or something similar).
        # it calls __update_frame_parameters instead of set_frame_parameters so that we do _not_ update the
        # current acquisition (which can cause a cycle in that it would again set the low level values, which
        # itself wouldn't be an issue unless the user makes multiple changes in quick succession). not setting
        # current values is different semantics than the scan control panel, which _does_ set current values if
        # the current profile is selected. Hrrmmm.
        with self.__latest_values_lock:
            self.__latest_values[profile_index] = ScanFrameParameters(frame_parameters)
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

    ChannelState = collections.namedtuple("ChannelState", ["channel_id", "name", "enabled"])

    def get_channel_index(self, channel_id: str) -> typing.Optional[int]:
        for channel_index in range(self.channel_count):
            if self.get_channel_state(channel_index).channel_id == channel_id:
                return channel_index
        return None

    def __make_channel_id(self, channel_index) -> str:
        return "abcdefgh"[channel_index]

    def __make_channel_state(self, channel_index, channel_name, channel_enabled) -> ChannelState:
        return ScanHardwareSource.ChannelState(self.__make_channel_id(channel_index), channel_name, channel_enabled)

    def __device_state_changed(self, profile_frame_parameters_list, device_channel_states) -> None:
        for profile_index, profile_frame_parameters in enumerate(profile_frame_parameters_list):
            self.__profile_frame_parameters_changed(profile_index, profile_frame_parameters)
        channel_states = list()
        for channel_index, (channel_name, channel_enabled) in enumerate(device_channel_states):
            channel_states.append(self.__make_channel_state(channel_index, channel_name, channel_enabled))
        self.__channel_states_changed(channel_states)

    def get_frame_parameters_from_dict(self, d):
        return ScanFrameParameters(d)

    def calculate_frame_time(self, frame_parameters: dict) -> float:
        size = frame_parameters["size"]
        pixel_time_us = frame_parameters["pixel_time_us"]
        return size[0] * size[1] * pixel_time_us / 1000000.0

    def get_current_frame_time(self):
        return self.calculate_frame_time(self.get_current_frame_parameters())

    def get_record_frame_time(self):
        return self.calculate_frame_time(self.get_record_frame_parameters())

    def make_reference_key(self, **kwargs) -> str:
        # TODO: specifying the channel key in an acquisition? and sub channels?
        is_subscan = kwargs.get("subscan", False)
        channel_index = kwargs.get("channel_index")
        reference_key = kwargs.get("reference_key")
        if reference_key:
            return "_".join([self.hardware_source_id, str(reference_key)])
        if channel_index is not None:
            if is_subscan:
                return "_".join([self.hardware_source_id, self.__make_channel_id(channel_index), "subscan"])
            else:
                return "_".join([self.hardware_source_id, self.__make_channel_id(channel_index)])
        return self.hardware_source_id

    def clean_display_items(self, document_model, display_items, **kwargs) -> None:
        for display_item in display_items:
            for graphic in copy.copy(display_item.graphics):
                graphic_id = graphic.graphic_id
                if graphic_id in ("probe", "subscan", "line_scan", "drift"):
                    display_item.remove_graphic(graphic)

    def get_buffer_data(self, start: int, count: int) -> typing.Optional[typing.List[typing.List[typing.Dict]]]:
        """Get recently acquired (buffered) data.

        The start parameter can be negative to index backwards from the end.

        If start refers to a buffer item that doesn't exist or if count requests too many buffer items given
        the start value, the returned list may have fewer elements than count.

        Returns None if buffering is not enabled.
        """
        if hasattr(self.__device, "get_buffer_data"):
            buffer_data = self.__device.get_buffer_data(start, count)

            enabled_channel_states = list()
            for channel_index in range(self.channel_count):
                channel_state = self.get_channel_state(channel_index)
                if channel_state.enabled:
                    enabled_channel_states.append(channel_state)

            scan_id = uuid.uuid4()

            data_element_groups = list()

            for data_element_group in buffer_data:
                new_data_element_group = list()
                for channel_index, (_data_element, channel_state) in enumerate(zip(data_element_group, enabled_channel_states)):
                    channel_name = channel_state.name
                    channel_id = channel_state.channel_id
                    if self.subscan_enabled:
                        channel_id += "_subscan"
                    _data = _data_element["data"]
                    _scan_properties = _data_element["properties"]

                    # create the 'data_element' in the format that must be returned from this method
                    # '_data_element' is the format returned from the Device.
                    data_element = {"metadata": dict()}
                    instrument_metadata = dict()
                    update_instrument_properties(instrument_metadata, self.__stem_controller, self.__device)
                    if instrument_metadata:
                        data_element["metadata"].setdefault("instrument", dict()).update(instrument_metadata)
                    update_scan_data_element(data_element, self.__frame_parameters, _data.shape, channel_name, channel_id, _scan_properties)
                    update_scan_metadata(data_element["metadata"].setdefault("scan", dict()), self.hardware_source_id, self.display_name, self.__frame_parameters, scan_id, _scan_properties)
                    update_detector_metadata(data_element["metadata"].setdefault("hardware_source", dict()), self.hardware_source_id, self.display_name, _data.shape, None, channel_name, channel_id, _scan_properties)
                    data_element["data"] = _data
                    new_data_element_group.append(data_element)
                data_element_groups.append(new_data_element_group)

            return data_element_groups

        return None

    def __probe_state_changed(self, probe_state: str, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
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
    def probe_state(self) -> str:
        return self.__stem_controller.probe_state

    def _set_probe_position(self, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        if probe_position is not None:
            if hasattr(self.__device, "set_scan_context_probe_position"):
                self.__device.set_scan_context_probe_position(self.__stem_controller.scan_context, probe_position)
            else:
                self.__device.set_idle_position_by_percentage(probe_position.x, probe_position.y)
            self.__last_idle_position = probe_position
        else:
            if hasattr(self.__device, "set_scan_context_probe_position"):
                self.__device.set_scan_context_probe_position(self.__stem_controller.scan_context, None)
            else:
                # pass magic value to position to default position which may be top left or center depending on configuration.
                self.__device.set_idle_position_by_percentage(-1.0, -1.0)
            self.__last_idle_position = -1.0, -1.0

    def _get_last_idle_position_for_test(self):
        return self.__last_idle_position

    @property
    def probe_position(self) -> typing.Optional[Geometry.FloatPoint]:
        return self.__stem_controller.probe_position

    @probe_position.setter
    def probe_position(self, probe_position: typing.Optional[typing.Union[Geometry.FloatPoint, typing.Tuple]]):
        probe_position = Geometry.FloatPoint.make(probe_position) if probe_position else None
        self.__stem_controller.set_probe_position(probe_position)

    def validate_probe_position(self) -> None:
        self.__stem_controller.validate_probe_position()

    # override from the HardwareSource parent class.
    def data_item_states_changed(self, data_item_states):
        channel_ids = {self.get_channel_state(channel_index).channel_id for channel_index in range(self.channel_count)}
        channel_map = dict()
        for data_item_state in data_item_states:
            if data_item_state.get("channel_id") in channel_ids:
                channel_map[data_item_state.get("channel_id")] = data_item_state.get("data_item")
        self.__stem_controller._update_scan_channel_map(channel_map)

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

    def shift_click(self, mouse_position, camera_shape, logger: logging.Logger) -> None:
        frame_parameters = self.__device.current_frame_parameters
        width, height = frame_parameters.size
        fov_nm = frame_parameters.fov_nm
        pixel_size_nm = fov_nm / max(width, height)
        # calculate dx, dy in meters
        dx = 1e-9 * pixel_size_nm * (mouse_position[1] - (camera_shape[1] / 2))
        dy = 1e-9 * pixel_size_nm * (mouse_position[0] - (camera_shape[0] / 2))
        logger.info("Shifting (%s,%s) um.\n", -dx * 1e6, -dy * 1e6)
        self.__stem_controller.change_stage_position(dy=dy, dx=dx)

    def increase_pmt(self, channel_index):
        self.__stem_controller.change_pmt_gain(channel_index, factor=2.0)

    def decrease_pmt(self, channel_index):
        self.__stem_controller.change_pmt_gain(channel_index, factor=0.5)

    def get_api(self, version):
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade:

            def __init__(self):
                pass

        return CameraFacade()


class InstrumentController(abc.ABC):

    def apply_metadata_groups(self, properties: typing.MutableMapping, metatdata_groups: typing.Sequence[typing.Tuple[typing.Sequence[str], str]]) -> None: pass

    def get_autostem_properties(self) -> typing.Dict: return dict()

    def handle_shift_click(self, **kwargs) -> None: pass

    def handle_tilt_click(self, **kwargs) -> None: pass


def update_instrument_properties(stem_properties: typing.MutableMapping, instrument_controller: InstrumentController, scan_device) -> None:
    if instrument_controller:
        # give the instrument controller opportunity to add properties
        if callable(getattr(instrument_controller, "get_autostem_properties", None)):
            try:
                autostem_properties = instrument_controller.get_autostem_properties()
                stem_properties.update(autostem_properties)
            except Exception as e:
                pass
        # give the instrument controller opportunity to update metadata groups specified by the camera
        if hasattr(scan_device, "acquisition_metatdata_groups"):
            acquisition_metatdata_groups = scan_device.acquisition_metatdata_groups
            instrument_controller.apply_metadata_groups(stem_properties, acquisition_metatdata_groups)


_component_registered_listener = None
_component_unregistered_listener = None

def run():
    def component_registered(component, component_types):
        if "scan_device" in component_types:
            stem_controller = None
            stem_controller_id = getattr(component, "stem_controller_id", None)
            if not stem_controller and stem_controller_id:
                stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id(component.stem_controller_id)
            if not stem_controller and not stem_controller_id:
                stem_controller = Registry.get_component("stem_controller")
            if not stem_controller:
                print("STEM Controller (" + component.stem_controller_id + ") for (" + component.scan_device_id + ") not found. Using proxy.")
                from nion.instrumentation import stem_controller
                stem_controller = stem_controller.STEMController()
            scan_hardware_source = ScanHardwareSource(stem_controller, component, component.scan_device_id, component.scan_device_name)
            if hasattr(component, "priority"):
                scan_hardware_source.priority = component.priority
            Registry.register_component(scan_hardware_source, {"hardware_source", "scan_hardware_source"})
            HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)
            component.hardware_source = scan_hardware_source

    def component_unregistered(component, component_types):
        if "scan_device" in component_types:
            scan_hardware_source = component.hardware_source
            Registry.unregister_component(scan_hardware_source)
            HardwareSource.HardwareSourceManager().unregister_hardware_source(scan_hardware_source)

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    for component in Registry.get_components_by_type("scan_device"):
        component_registered(component, {"scan_device"})


class ScanInterface:
    # preliminary interface (v1.0.0) for scan hardware source
    def get_current_frame_parameters(self) -> dict: ...
    def create_frame_parameters(self, d: dict) -> dict: ...
    def get_enabled_channels(self) -> typing.Sequence[int]: ...
    def set_enabled_channels(self, channel_indexes: typing.Sequence[int]) -> None: ...
    def start_playing(self, frame_parameters: dict) -> None: ...
    def stop_playing(self) -> None: ...
    def abort_playing(self) -> None: ...
    def is_playing(self) -> bool: ...
    def grab_next_to_start(self) -> typing.List[DataAndMetadata.DataAndMetadata]: ...
    def grab_next_to_finish(self) -> typing.List[DataAndMetadata.DataAndMetadata]: ...
    def grab_sequence_prepare(self, count: int) -> bool: ...
    def grab_sequence(self, count: int) -> typing.Optional[typing.List[DataAndMetadata.DataAndMetadata]]: ...
    def grab_sequence_abort(self) -> None: ...
    def grab_sequence_get_progress(self) -> typing.Optional[float]: ...
    def grab_synchronized(self, *, scan_frame_parameters: dict=None, camera=None, camera_frame_parameters: dict=None) -> typing.Tuple[typing.List[DataAndMetadata.DataAndMetadata], typing.List[DataAndMetadata.DataAndMetadata]]: ...
    def grab_synchronized_abort(self) -> None: ...
    def grab_synchronized_get_progress(self) -> typing.Optional[float]: ...
    def grab_buffer(self, count: int, *, start: int = None) -> typing.Optional[typing.List[typing.List[DataAndMetadata.DataAndMetadata]]]: ...
    def calculate_frame_time(self, frame_parameters: dict) -> float: ...
    def calculate_line_scan_frame_parameters(self, frame_parameters: dict, start: typing.Tuple[float, float], end: typing.Tuple[float, float], length: int) -> dict: ...
    def make_reference_key(self, **kwargs) -> str: ...

    def get_current_frame_id(self) -> int: ...
    def get_frame_progress(self, frame_id: int) -> float: ...

