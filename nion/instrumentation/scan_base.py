from __future__ import annotations

# standard libraries
import abc
import asyncio
import collections
import contextlib
import copy
import dataclasses
import datetime
import functools
import gettext
import json
import logging
import math
import numpy
import numpy.typing
import os
import pathlib
import queue
import threading
import time
import typing
import uuid

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import camera_base
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import stem_controller as stem_controller_module
from nion.swift.model import ImportExportManager
from nion.swift.model import Utility
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Process
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.swift.model import DataItem

_NDArray = numpy.typing.NDArray[typing.Any]

_ = gettext.gettext


class ParametersBase:
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        self.__d: typing.Dict[str, typing.Any] = dict()
        assert not args or isinstance(args[0], dict)
        if args and isinstance(args[0], dict):
            self.__d.update(args[0])
        self.__d.update(kwargs)

    def __copy__(self) -> ParametersBase:
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> ParametersBase:
        deepcopy = self.__class__(copy.deepcopy(self.as_dict()))
        memo[id(self)] = deepcopy
        return deepcopy

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        return copy.deepcopy(self.__d)

    def has_parameter(self, name: str) -> bool:
        return name in self.__d

    def get_parameter(self, name: str, default_value: typing.Any = None) -> typing.Any:
        return self.__d.get(name, default_value)

    def set_parameter(self, name: str, value: typing.Any) -> None:
        if value is not None:
            self.__d[name] = value
        else:
            self.__d.pop(name, None)

    def clear_parameter(self, name: str) -> None:
        self.__d.pop(name, None)


# subclass of ScanFrameParameters when mypy #13954 is fixed.
class ScanFrameParameters(ParametersBase):
    """Scan frame parameters.

    This should almost never be instantiated directly. The settings objects is responsible for constructing scan
    frame parameters suitable for a particular scan module. Scan frame parameters can be copied using `copy.copy`
    or they can be created from a dictionary using `settings.get_frame_parameters_from_dict`.

    NOTE: some scan modules do not customize the scan frame parameters. The settings object for those scan modules
    are allowed to use this class directly.
    """

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        if self.has_parameter("size") and not self.has_parameter("pixel_size"):
            self.pixel_size = Geometry.IntSize.make(self.get_parameter("size"))

    @property
    def fov_nm(self) -> float:
        return typing.cast(float, self.get_parameter("fov_nm", 8.0))

    @fov_nm.setter
    def fov_nm(self, value: float) -> None:
        self.set_parameter("fov_nm", value)

    @property
    def rotation_rad(self) -> float:
        return typing.cast(float, self.get_parameter("rotation_rad", 0.0))

    @rotation_rad.setter
    def rotation_rad(self, value: float) -> None:
        self.set_parameter("rotation_rad", value)

    @property
    def center_nm(self) -> Geometry.FloatPoint:
        # parameters must be convertible to JSON; center_nm must be stored as a tuple.
        return Geometry.FloatPoint.make(self.get_parameter("center_nm", (0, 0)))

    @center_nm.setter
    def center_nm(self, value: Geometry.FloatPointTuple) -> None:
        # parameters must be convertible to JSON; center_nm must be stored as a tuple.
        self.set_parameter("center_nm", Geometry.FloatPoint.make(value).as_tuple())

    @property
    def pixel_time_us(self) -> float:
        return typing.cast(float, self.get_parameter("pixel_time_us", 10.0))

    @pixel_time_us.setter
    def pixel_time_us(self, value: float) -> None:
        self.set_parameter("pixel_time_us", value)

    @property
    def pixel_size(self) -> Geometry.IntSize:
        # parameters must be convertible to JSON; pixel_size must be stored as a tuple.
        return Geometry.IntSize.make(self.get_parameter("pixel_size", (512, 512)))

    @pixel_size.setter
    def pixel_size(self, value: Geometry.IntSizeTuple) -> None:
        # parameters must be convertible to JSON; center_nm must be stored as a tuple.
        self.set_parameter("pixel_size", Geometry.IntSize.make(value).as_tuple())

    @property
    def size(self) -> Geometry.IntSize:
        return self.pixel_size

    @size.setter
    def size(self, value: Geometry.IntSize) -> None:
        self.pixel_size = value

    @property
    def pixel_size_nm(self) -> Geometry.FloatSize:
        return Geometry.FloatSize(height=self.fov_nm / self.pixel_size.height, width=self.fov_nm / self.pixel_size.width)

    @pixel_size_nm.setter
    def pixel_size_nm(self, pixel_size_nm: Geometry.FloatSize) -> None:
        # the free parameter is always the size; keep the fov fixed.
        if pixel_size_nm and pixel_size_nm.height > 0.0 and pixel_size_nm.width > 0.0:
            self.pixel_size = Geometry.IntSize(
                height=max(1, round(self.fov_nm / pixel_size_nm.height)),
                width=max(1, round(self.fov_nm / pixel_size_nm.width))
            )

    @property
    def subscan_pixel_size(self) -> typing.Optional[Geometry.IntSize]:
        # parameters must be convertible to JSON; subscan_pixel_size must be stored as a tuple.
        subscan_pixel_size_tuple = self.get_parameter("subscan_pixel_size", None)
        return Geometry.IntSize.make(subscan_pixel_size_tuple) if subscan_pixel_size_tuple else None

    @subscan_pixel_size.setter
    def subscan_pixel_size(self, value: typing.Optional[Geometry.IntSizeTuple]) -> None:
        # parameters must be convertible to JSON; subscan_pixel_size must be stored as a tuple.
        self.set_parameter("subscan_pixel_size", Geometry.IntSize.make(value).as_tuple() if value else None)

    @property
    def subscan_fractional_size(self) -> typing.Optional[Geometry.FloatSize]:
        # parameters must be convertible to JSON; subscan_fractional_size must be stored as a tuple.
        subscan_fractional_size_tuple = self.get_parameter("subscan_fractional_size", None)
        return Geometry.FloatSize.make(subscan_fractional_size_tuple) if subscan_fractional_size_tuple else None

    @subscan_fractional_size.setter
    def subscan_fractional_size(self, value: typing.Optional[Geometry.FloatSizeTuple]) -> None:
        # parameters must be convertible to JSON; subscan_fractional_size must be stored as a tuple.
        self.set_parameter("subscan_fractional_size", Geometry.FloatSize.make(value).as_tuple() if value else None)

    @property
    def subscan_fractional_center(self) -> typing.Optional[Geometry.FloatPoint]:
        # parameters must be convertible to JSON; subscan_fractional_center must be stored as a tuple.
        subscan_fractional_center_tuple = self.get_parameter("subscan_fractional_center", None)
        return Geometry.FloatPoint.make(subscan_fractional_center_tuple) if subscan_fractional_center_tuple else None

    @subscan_fractional_center.setter
    def subscan_fractional_center(self, value: typing.Optional[Geometry.FloatPointTuple]) -> None:
        # parameters must be convertible to JSON; subscan_fractional_center must be stored as a tuple.
        self.set_parameter("subscan_fractional_center", Geometry.FloatPoint.make(value).as_tuple() if value else None)

    @property
    def subscan_pixel_size_nm(self) -> typing.Optional[Geometry.FloatSize]:
        if self.subscan_pixel_size and self.subscan_fractional_size and self.subscan_pixel_size.height > 0.0 and self.subscan_pixel_size.width > 0.0:
            return Geometry.FloatSize(
                height=self.subscan_fractional_size.height * self.fov_nm / self.subscan_pixel_size.height,
                width=self.subscan_fractional_size.width * self.fov_nm / self.subscan_pixel_size.width
            )
        return None

    @subscan_pixel_size_nm.setter
    def subscan_pixel_size_nm(self, pixel_size_nm: typing.Optional[Geometry.FloatSize]) -> None:
        # the free parameter is always the subscan size; keep the fov fixed.
        if pixel_size_nm and pixel_size_nm.height > 0.0 and pixel_size_nm.width > 0.0 and self.subscan_fractional_size:
            self.subscan_pixel_size = Geometry.IntSize(
                height=max(1, round(self.subscan_fractional_size.height * self.fov_nm / pixel_size_nm.height)),
                width=max(1, round(self.subscan_fractional_size.width * self.fov_nm / pixel_size_nm.width))
            )

    @property
    def subscan_rotation(self) -> float:
        return typing.cast(float, self.get_parameter("subscan_rotation", 0.0))

    @subscan_rotation.setter
    def subscan_rotation(self, value: float) -> None:
        self.set_parameter("subscan_rotation", value)

    @property
    def ac_line_sync(self) -> bool:
        return typing.cast(bool, self.get_parameter("ac_line_sync", False))

    @ac_line_sync.setter
    def ac_line_sync(self, value: bool) -> None:
        self.set_parameter("ac_line_sync", value)

    @property
    def channel_indexes_enabled(self) -> typing.Optional[typing.Sequence[int]]:
        maybe_channel_indexes_enabled = self.get_parameter("channel_indexes_enabled", list())
        if maybe_channel_indexes_enabled is not None:
            return typing.cast(typing.Sequence[int], maybe_channel_indexes_enabled)
        return None

    @channel_indexes_enabled.setter
    def channel_indexes_enabled(self, value: typing.Sequence[int]) -> None:
        self.set_parameter("channel_indexes_enabled", list(value) if value is not None else None)

    @property
    def channel_override(self) -> typing.Optional[str]:
        return typing.cast(typing.Optional[str], self.get_parameter("channel_override", None))

    @channel_override.setter
    def channel_override(self, value: typing.Optional[str]) -> None:
        self.set_parameter("channel_override", value)

    @property
    def fov_size_nm(self) -> typing.Optional[Geometry.FloatSize]:
        # return the fov size with the same aspect ratio as the size
        # the width of the fov_size_nm will be the same as the fov_nm
        # the height will depend on the aspect ratio of the pixel shape.
        return Geometry.FloatSize(self.fov_nm * self.pixel_size.aspect_ratio, self.fov_nm)

    @property
    def rotation_deg(self) -> float:
        return math.degrees(self.rotation_rad)


def get_scan_calibrations(frame_parameters: ScanFrameParameters) -> typing.Tuple[Calibration.Calibration, Calibration.Calibration]:
    scan_shape = frame_parameters.pixel_size
    center_x_nm = frame_parameters.center_nm.x
    center_y_nm = frame_parameters.center_nm.y
    fov_nm = frame_parameters.fov_nm
    pixel_size_nm = fov_nm / max(scan_shape)
    scan_calibrations = (
        Calibration.Calibration(-center_y_nm - pixel_size_nm * scan_shape[0] * 0.5, pixel_size_nm, "nm"),
        Calibration.Calibration(-center_x_nm - pixel_size_nm * scan_shape[1] * 0.5, pixel_size_nm, "nm")
    )
    return scan_calibrations


def update_scan_properties(properties: typing.MutableMapping[str, typing.Any], scan_frame_parameters: ScanFrameParameters, scan_id_str: typing.Optional[str]) -> None:
    if scan_id_str:
        properties["scan_id"] = scan_id_str
    properties["center_x_nm"] = scan_frame_parameters.center_nm.x
    properties["center_y_nm"] = scan_frame_parameters.center_nm.y
    properties["fov_nm"] = scan_frame_parameters.fov_nm
    properties["rotation"] = scan_frame_parameters.rotation_rad
    properties["rotation_deg"] = math.degrees(scan_frame_parameters.rotation_rad)
    properties["scan_context_size"] = scan_frame_parameters.pixel_size.as_tuple()
    if scan_frame_parameters.subscan_fractional_size is not None:
        properties["subscan_fractional_size"] = scan_frame_parameters.subscan_fractional_size.as_tuple()
    if scan_frame_parameters.subscan_pixel_size is not None:
        properties["scan_size"] = scan_frame_parameters.subscan_pixel_size.to_float_size().as_tuple()
    elif scan_frame_parameters.subscan_fractional_size is not None:
        properties["scan_size"] = (int(scan_frame_parameters.pixel_size.height * scan_frame_parameters.subscan_fractional_size.height),
                                   int(scan_frame_parameters.pixel_size.height * scan_frame_parameters.subscan_fractional_size.width))
    else:
        properties["scan_size"] = scan_frame_parameters.pixel_size.as_tuple()
    if scan_frame_parameters.subscan_fractional_center is not None:
        properties["subscan_fractional_center"] = scan_frame_parameters.subscan_fractional_center.as_tuple()
    if scan_frame_parameters.subscan_rotation:
        properties["subscan_rotation"] = scan_frame_parameters.subscan_rotation


# set the calibrations for this image. does not touch metadata.
def update_scan_data_element(data_element: typing.MutableMapping[str, typing.Any], scan_frame_parameters: ScanFrameParameters, data_shape: typing.Tuple[int, int], channel_name: str, channel_id: str, scan_properties: typing.Mapping[str, typing.Any]) -> None:
    pixel_time_us = float(scan_properties["pixel_time_us"])
    line_time_us = float(scan_properties["line_time_us"]) if "line_time_us" in scan_properties else pixel_time_us * data_shape[1]
    center_x_nm = scan_frame_parameters.center_nm.x
    center_y_nm = scan_frame_parameters.center_nm.y
    fov_nm = scan_frame_parameters.fov_nm  # context fov_nm, not actual fov_nm returned from low level
    if scan_frame_parameters.pixel_size[0] > scan_frame_parameters.pixel_size[1]:
        fractional_size = scan_frame_parameters.subscan_fractional_size[0] if scan_frame_parameters.subscan_fractional_size else 1.0
        pixel_size = scan_frame_parameters.subscan_pixel_size[0] if scan_frame_parameters.subscan_pixel_size else scan_frame_parameters.pixel_size[0]
        pixel_size_nm = fov_nm * fractional_size / pixel_size
    else:
        fractional_size = scan_frame_parameters.subscan_fractional_size[1] if scan_frame_parameters.subscan_fractional_size else 1.0
        pixel_size = scan_frame_parameters.subscan_pixel_size[1] if scan_frame_parameters.subscan_pixel_size else scan_frame_parameters.pixel_size[1]
        pixel_size_nm = fov_nm * fractional_size / pixel_size
    data_element["title"] = channel_name
    data_element["version"] = 1
    data_element["channel_id"] = channel_id  # needed to match to the channel
    data_element["channel_name"] = channel_name  # needed to match to the channel
    if not "spatial_calibrations" in data_element:
        assert len(data_shape) in (1, 2)
        if scan_properties.get("calibration_style") == "time":
            if len(data_shape) == 2:
                data_element["spatial_calibrations"] = (
                    {"offset": 0.0, "scale": line_time_us / 1E6, "units": "s"},
                    {"offset": 0.0, "scale": pixel_time_us / 1E6, "units": "s"}
                )
            elif len(data_shape) == 1:
                data_element["spatial_calibrations"] = (
                    {"offset": 0.0, "scale": pixel_time_us / 1E6, "units": "s"}
                )
        else:
            if len(data_shape) == 2:
                data_element["spatial_calibrations"] = (
                    {"offset": -center_y_nm - pixel_size_nm * data_shape[0] * 0.5, "scale": pixel_size_nm, "units": "nm"},
                    {"offset": -center_x_nm - pixel_size_nm * data_shape[1] * 0.5, "scale": pixel_size_nm, "units": "nm"}
                )
            elif len(data_shape) == 1:
                data_element["spatial_calibrations"] = (
                    {"offset": -center_x_nm - pixel_size_nm * data_shape[1] * 0.5, "scale": pixel_size_nm, "units": "nm"}
                )


def update_scan_metadata(scan_metadata: typing.MutableMapping[str, typing.Any], hardware_source_id: str, display_name: str, scan_frame_parameters: ScanFrameParameters, scan_id: typing.Optional[uuid.UUID], scan_properties: typing.Mapping[str, typing.Any]) -> None:
    scan_metadata["hardware_source_id"] = hardware_source_id
    scan_metadata["hardware_source_name"] = display_name
    update_scan_properties(scan_metadata, scan_frame_parameters, str(scan_id) if scan_id else None)
    if scan_frame_parameters:
        scan_metadata["scan_device_parameters"] = scan_frame_parameters.as_dict()
    if scan_properties:
        scan_properties = dict(scan_properties)
        scan_properties.pop("channel_id", None)  # not part of scan description
        scan_metadata["scan_device_properties"] = scan_properties


def update_detector_metadata(detector_metadata: typing.MutableMapping[str, typing.Any], hardware_source_id: str, display_name: str, data_shape: DataAndMetadata.ShapeType, frame_number: typing.Optional[int], channel_name: str, channel_id: str, scan_properties: typing.Mapping[str, typing.Any]) -> None:
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



@dataclasses.dataclass
class ScanAcquisitionTaskParameters(HardwareSource.AcquisitionTaskParameters):
    scan_id: typing.Optional[uuid.UUID] = None
    data_shape_override: typing.Optional[Geometry.IntSize] = None
    top_left_override: typing.Optional[Geometry.IntPoint] = None
    state_override: typing.Optional[str] = None


class ScanAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, stem_controller_: stem_controller_module.STEMController, scan_hardware_source: ScanHardwareSource,
                 device: ScanDevice, hardware_source_id: str, is_continuous: bool, frame_parameters: ScanFrameParameters,
                 acquisition_task_parameters: typing.Optional[ScanAcquisitionTaskParameters], channel_ids: typing.Sequence[str],
                 display_name: str) -> None:
        # acquisition_task_parameters are internal parameters that are needed by this package. do not use externally.
        # channel_ids is the channel id for each acquired channel
        # for instance, there may be 4 possible channels (0-3, a-d) and acquisition from channels 1,2
        # in that case channel_ids would be [b, c]
        super().__init__(is_continuous)
        self.__stem_controller = stem_controller_
        self.hardware_source_id = hardware_source_id
        self.__device = device
        self.__scan_hardware_source = scan_hardware_source
        self.__is_continuous = is_continuous
        self.__display_name = display_name
        self.__hardware_source_id = hardware_source_id
        self.__frame_parameters = copy.copy(frame_parameters)
        self.__acquisition_task_parameters = acquisition_task_parameters or ScanAcquisitionTaskParameters()
        self.__frame_number: typing.Optional[int] = None
        self.__scan_id: typing.Optional[uuid.UUID] = None
        self.__last_scan_id: typing.Optional[uuid.UUID] = None
        self.__fixed_scan_id = self.__acquisition_task_parameters.scan_id
        self.__pixels_to_skip = 0
        self.__channel_ids = list(channel_ids)
        self.__last_read_time = 0.0
        self.__subscan_enabled = False

    def set_frame_parameters(self, frame_parameters: ScanFrameParameters) -> None:
        if self.__frame_parameters.as_dict() != frame_parameters.as_dict():
            self.__frame_parameters = copy.copy(frame_parameters)
            self.__activate_frame_parameters()

    @property
    def frame_parameters(self) -> typing.Optional[ScanFrameParameters]:
        return copy.copy(self.__frame_parameters)

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self.__stem_controller._enter_scanning_state()
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
        self.__stem_controller._exit_scanning_state()

    def _acquire_data_elements(self) -> typing.List[typing.Dict[str, typing.Any]]:

        def update_data_element(data_element: typing.MutableMapping[str, typing.Any], complete: bool, sub_area: typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]], npdata: _NDArray) -> None:
            data_element["data"] = npdata
            if self.__acquisition_task_parameters.data_shape_override:
                # data_shape of None is handled specially in DataChannel.update
                data_element["data_shape"] = self.__acquisition_task_parameters.data_shape_override.as_tuple()
            data_element["sub_area"] = sub_area
            data_element["dest_sub_area"] = (Geometry.IntRect.make(sub_area) + (self.__acquisition_task_parameters.top_left_override if self.__acquisition_task_parameters.top_left_override else Geometry.IntPoint())).as_tuple()
            data_element["state"] = self.__acquisition_task_parameters.state_override or "complete" if complete else "partial"
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
            channel_id = channel_override or (channel_id + ("_subscan" if self.__frame_parameters.subscan_pixel_size else ""))

            # create the 'data_element' in the format that must be returned from this method
            # '_data_element' is the format returned from the Device.
            data_element: typing.Dict[str, typing.Any] = {"metadata": dict()}
            instrument_metadata: typing.Dict[str, typing.Any] = dict()
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

    def __activate_frame_parameters(self) -> None:
        device_frame_parameters = copy.copy(self.__frame_parameters)
        self.__device.set_frame_parameters(device_frame_parameters)


def apply_section_rect(scan_frame_parameters: ScanFrameParameters, acquisition_task_parameters: ScanAcquisitionTaskParameters,
                       section_rect: Geometry.IntRect, scan_size: Geometry.IntSize, fractional_area: Geometry.FloatRect) -> ScanFrameParameters:
    section_rect_f = section_rect.to_float_rect()
    subscan_rotation = scan_frame_parameters.subscan_rotation
    subscan_fractional_center0 = scan_frame_parameters.subscan_fractional_center or Geometry.FloatPoint(0.5, 0.5)
    subscan_fractional_size = Geometry.FloatSize(h=fractional_area.height * section_rect_f.height / scan_size.height,
                                                 w=fractional_area.width * section_rect_f.width / scan_size.width)
    subscan_fractional_center = Geometry.FloatPoint(y=fractional_area.top + fractional_area.height * section_rect_f.center.y / scan_size.height,
                                                    x=fractional_area.left + fractional_area.width * section_rect_f.center.x / scan_size.width)
    subscan_fractional_center = subscan_fractional_center.rotate(-subscan_rotation, subscan_fractional_center0)
    section_frame_parameters = copy.deepcopy(scan_frame_parameters)
    section_frame_parameters.subscan_pixel_size = section_rect.size
    section_frame_parameters.subscan_fractional_size = subscan_fractional_size
    section_frame_parameters.subscan_fractional_center = subscan_fractional_center
    acquisition_task_parameters.data_shape_override = scan_size  # no flyback addition since this is data from scan device
    acquisition_task_parameters.state_override = "complete" if section_rect.bottom == scan_size.height and section_rect.right == scan_size.width else "partial"
    acquisition_task_parameters.top_left_override = section_rect.top_left
    return section_frame_parameters


class ScanDevice(typing.Protocol):
    scan_device_id: str
    scan_device_name: str

    def close(self) -> None: ...
    def get_channel_name(self, channel_index: int) -> str: ...
    def set_channel_enabled(self, channel_index: int, enabled: bool) -> bool: ...
    def set_frame_parameters(self, frame_parameters: ScanFrameParameters) -> None: ...
    def save_frame_parameters(self) -> None: ...
    def start_frame(self, is_continuous: bool) -> int: ...
    def cancel(self) -> None: ...
    def stop(self) -> None: ...
    def read_partial(self, frame_number: typing.Optional[int], pixels_to_skip: int) -> typing.Tuple[typing.Sequence[ImportExportManager.DataElementType], bool, bool, typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]], typing.Optional[int], int]: ...
    def get_buffer_data(self, start: int, count: int) -> typing.List[typing.List[typing.Dict[str, typing.Any]]]: ...
    def set_scan_context_probe_position(self, scan_context: stem_controller_module.ScanContext, probe_position: typing.Optional[Geometry.FloatPoint]) -> None: ...
    def set_idle_position_by_percentage(self, x: float, y: float) -> None: ...
    def prepare_synchronized_scan(self, scan_frame_parameters: ScanFrameParameters, *, camera_exposure_ms: float, **kwargs: typing.Any) -> None: ...
    def calculate_flyback_pixels(self, frame_parameters: ScanFrameParameters) -> int: return 2

    # default implementation
    def wait_for_frame(self, frame_number: int) -> None:
        pixels_to_skip = 0
        while True:
            _data_elements, complete, bad_frame, sub_area, _frame_number, pixels_to_skip = self.read_partial(frame_number, pixels_to_skip)
            if complete:
                break

    @property
    def channel_count(self) -> int: raise NotImplementedError()

    @property
    def channels_enabled(self) -> typing.Tuple[bool, ...]: raise NotImplementedError()

    @property
    def is_scanning(self) -> bool: raise NotImplementedError()

    @property
    def current_frame_parameters(self) -> ScanFrameParameters: raise NotImplementedError()

    @property
    def acquisition_metatdata_groups(self) -> typing.Sequence[typing.Tuple[typing.Sequence[str], str]]:
        return list()

    on_device_state_changed: typing.Optional[typing.Callable[[typing.Sequence[ScanFrameParameters], typing.Sequence[typing.Tuple[str, bool]]], None]]


@typing.runtime_checkable
class ScanHardwareSource(HardwareSource.HardwareSource, typing.Protocol):

    # public methods

    def grab_synchronized(self, *,
                          data_channel: typing.Optional[Acquisition.DataChannel] = None,
                          scan_frame_parameters: ScanFrameParameters,
                          camera: camera_base.CameraHardwareSource,
                          camera_frame_parameters: camera_base.CameraFrameParameters,
                          camera_data_channel: typing.Optional[camera_base.SynchronizedDataChannelInterface] = None,
                          section_height: typing.Optional[int] = None,
                          scan_data_stream_functor: typing.Optional[Acquisition.DataStreamFunctor] = None,
                          scan_count: int = 1) -> GrabSynchronizedResult: ...

    def record_immediate(self, frame_parameters: ScanFrameParameters,
                         enabled_channels: typing.Optional[typing.Sequence[int]] = None,
                         sync_timeout: typing.Optional[float] = None) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]: ...

    # used in Facade

    @property
    def selected_profile_index(self) -> int: raise NotImplementedError()

    def set_channel_enabled(self, channel_index: int, enabled: bool) -> None: ...
    def get_frame_parameters(self, profile_index: int) -> ScanFrameParameters: ...
    def set_frame_parameters(self, profile_index: int, frame_parameters: ScanFrameParameters) -> None: ...

    # properties

    @property
    def scan_device(self) -> ScanDevice: raise NotImplementedError()

    @property
    def scan_settings(self) -> ScanSettingsProtocol: raise NotImplementedError()

    @property
    def stem_controller(self) -> stem_controller_module.STEMController: raise NotImplementedError()

    @property
    def channel_count(self) -> int: raise NotImplementedError()

    @property
    def scan_context(self) -> stem_controller_module.ScanContext: raise NotImplementedError()

    @property
    def probe_position(self) -> typing.Optional[Geometry.FloatPoint]: raise NotImplementedError()

    @probe_position.setter
    def probe_position(self, probe_position: typing.Optional[Geometry.FloatPointTuple]) -> None: ...

    @property
    def probe_state(self) -> str: raise NotImplementedError()

    @property
    def subscan_state(self) -> stem_controller_module.SubscanState: raise NotImplementedError()

    @property
    def subscan_enabled(self) -> bool: raise NotImplementedError()

    @subscan_enabled.setter
    def subscan_enabled(self, enabled: bool) -> None: ...

    @property
    def subscan_region(self) -> typing.Optional[Geometry.FloatRect]: raise NotImplementedError()

    @subscan_region.setter
    def subscan_region(self, value: typing.Optional[Geometry.FloatRect]) -> None: ...

    @property
    def line_scan_state(self) -> stem_controller_module.LineScanState: raise NotImplementedError()

    @property
    def line_scan_enabled(self) -> bool: raise NotImplementedError()

    @line_scan_enabled.setter
    def line_scan_enabled(self, enabled: bool) -> None: ...

    @property
    def line_scan_vector(self) -> typing.Optional[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]: raise NotImplementedError()

    @property
    def drift_channel_id(self) -> typing.Optional[str]: raise NotImplementedError()

    @drift_channel_id.setter
    def drift_channel_id(self, value: typing.Optional[str]) -> None: ...

    @property
    def drift_region(self) -> typing.Optional[Geometry.FloatRect]: raise NotImplementedError()

    @drift_region.setter
    def drift_region(self, value: typing.Optional[Geometry.FloatRect]) -> None: ...

    @property
    def drift_rotation(self) -> float: raise NotImplementedError()

    @drift_rotation.setter
    def drift_rotation(self, value: float) -> None: ...

    @property
    def drift_settings(self) -> stem_controller_module.DriftCorrectionSettings: raise NotImplementedError()

    @drift_settings.setter
    def drift_settings(self, value: stem_controller_module.DriftCorrectionSettings) -> None: ...

    @property
    def drift_enabled(self) -> bool: raise NotImplementedError()

    @drift_enabled.setter
    def drift_enabled(self, enabled: bool) -> None: ...

    # private. do not use outside instrumentation-kit.

    def get_current_frame_parameters(self) -> ScanFrameParameters: ...
    def set_record_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None: ...
    def get_record_frame_parameters(self) -> HardwareSource.FrameParameters: ...
    def periodic(self) -> None: ...
    def get_enabled_channels(self) -> typing.Sequence[int]: ...
    def get_channel_state(self, channel_index: int) -> ChannelState: ...
    def get_channel_enabled(self, channel_index: int) -> bool: ...
    def get_channel_index(self, channel_id: str) -> typing.Optional[int]: ...
    def get_subscan_channel_info(self, channel_index: int, channel_id: str, channel_name: str) -> typing.Tuple[int, str, str]: ...
    def apply_scan_context_subscan(self, frame_parameters: ScanFrameParameters, size: typing.Optional[typing.Tuple[int, int]] = None) -> None: ...
    def calculate_drift_lines(self, width: int, frame_time: float) -> int: ...
    def calculate_drift_scans(self) -> int: ...
    def shift_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None: ...
    def grab_synchronized_abort(self) -> None: ...
    def set_selected_profile_index(self, profile_index: int) -> None: ...
    def record_async(self, callback_fn: typing.Callable[[typing.Sequence[HardwareSource.DataAndMetadataPromise]], None]) -> None: ...
    def open_configuration_interface(self, api_broker: typing.Any) -> None: ...
    def validate_probe_position(self) -> None: ...
    def increase_pmt(self, channel_index: int) -> None: ...
    def decrease_pmt(self, channel_index: int) -> None: ...
    def get_channel_index_for_data_channel_index(self, data_channel_index: int) -> int: ...
    def get_data_channel_state(self, channel_index: int) -> typing.Tuple[typing.Optional[str], typing.Optional[str], bool]: ...
    def grab_synchronized_get_info(self, *, scan_frame_parameters: ScanFrameParameters, camera: camera_base.CameraHardwareSource, camera_frame_parameters: camera_base.CameraFrameParameters) -> GrabSynchronizedInfo: ...
    def get_current_frame_time(self) -> float: ...
    def get_buffer_data(self, start: int, count: int) -> typing.List[typing.List[typing.Dict[str, typing.Any]]]: ...
    def scan_immediate(self, frame_parameters: ScanFrameParameters) -> None: ...
    def calculate_flyback_pixels(self, frame_parameters: ScanFrameParameters) -> int: ...

    def get_context_data_channels(self) -> typing.Sequence[HardwareSource.DataChannel]:
        # return the data channels functioning as scan contexts.
        return tuple(filter(lambda data_channel: data_channel.is_context, self.data_channels))

    record_index: int
    priority: int = 100

    @property
    def drift_tracker(self) -> typing.Optional[DriftTracker.DriftTracker]:
        raise NotImplementedError()

    probe_state_changed_event: Event.Event
    channel_state_changed_event: Event.Event


@dataclasses.dataclass
class AxesDescriptor:
    sequence_axes: typing.Optional[int]
    collection_axes: typing.Sequence[int]
    data_axes: typing.Sequence[int]


@dataclasses.dataclass
class GrabSynchronizedInfo:
    scan_size: Geometry.IntSize
    fractional_area: Geometry.FloatRect
    is_subscan: bool
    camera_readout_size: Geometry.IntSize
    camera_readout_size_squeezed: DataAndMetadata.ShapeType
    scan_calibrations: typing.Tuple[Calibration.Calibration, Calibration.Calibration]
    data_calibrations: typing.Tuple[Calibration.Calibration, ...]
    data_intensity_calibration: Calibration.Calibration
    instrument_metadata: typing.Dict[str, typing.Any]
    camera_metadata: typing.Dict[str, typing.Any]
    scan_metadata: typing.Dict[str, typing.Any]
    axes_descriptor: AxesDescriptor


GrabSynchronizedResult = typing.Optional[typing.Tuple[typing.List[DataAndMetadata.DataAndMetadata], typing.List[DataAndMetadata.DataAndMetadata]]]


@dataclasses.dataclass
class ChannelState:
    channel_id: str
    name: str
    enabled: bool


@dataclasses.dataclass
class ScanSettingsMode:
    """Define a scan settings mode.

    name is a user visible name.

    mode_id is used for storing the mode reference in a file. it is not user visible and should be an identifier.

    frame_parameters are the associated frame parameters. these may be a subclass of ScanFrameParameters.
    """
    name: str
    mode_id: str
    frame_parameters: ScanFrameParameters


class ScanSettingsProtocol(typing.Protocol):
    """Define a protocol for storing and observing a set of frame parameters.

    """

    # events that might be used within the UI.
    current_frame_parameters_changed_event: Event.Event
    record_frame_parameters_changed_event: Event.Event
    profile_changed_event: Event.Event
    frame_parameters_changed_event: Event.Event

    # optional event and identifier for settings. defining settings_id signals that
    # the settings should be managed as a dict by the container of this class. the container
    # will call apply_settings to initialize settings and then expect settings_changed_event
    # to be fired when settings change.
    settings_changed_event: Event.Event
    settings_id: str

    def close(self) -> None:
        ...

    def initialize(self, configuration_location: typing.Optional[pathlib.Path] = None, event_loop: typing.Optional[asyncio.AbstractEventLoop] = None, **kwargs: typing.Any) -> None:
        ...

    def apply_settings(self, settings_dict: typing.Mapping[str, typing.Any]) -> None:
        """Initialize the settings with the settings_dict."""
        ...

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> ScanFrameParameters:
        """Return camera frame parameters from dict.

        Subclasses may override this method to provide a custom subclass of ScanFrameParameters.
        """
        raise NotImplementedError()

    def get_current_frame_parameters(self) -> ScanFrameParameters:
        """Get the current frame parameters."""
        raise NotImplementedError()

    def get_record_frame_parameters(self) -> ScanFrameParameters:
        """Get the record frame parameters."""
        raise NotImplementedError()

    def set_frame_parameters(self, settings_index: int, frame_parameters: ScanFrameParameters) -> None:
        """Set the frame parameters with the settings index and fire the frame parameters changed event.

        If the settings index matches the current settings index, call set current frame parameters.

        If the settings index matches the record settings index, call set record frame parameters.
        """
        ...

    def get_frame_parameters(self, settings_index: int) -> ScanFrameParameters:
        """Get the frame parameters for the settings index."""
        ...

    def set_selected_profile_index(self, settings_index: int) -> None:
        """Set the current settings index.

        Call set current frame parameters if it changed.

        Fire profile changed event if it changed.
        """
        ...

    @property
    def selected_profile_index(self) -> int:
        raise NotImplementedError()

    def open_configuration_interface(self, api_broker: typing.Any) -> None:
        return  # use return here to signal it is not abstract


ScanFrameParametersFactory = typing.Callable[[typing.Mapping[str, typing.Any]], ScanFrameParameters]


class ScanSettings(ScanSettingsProtocol):
    """A concrete implementation of ScanSettingsProtocol.

    Pass a list of scan modes (name/identifier/frame-parameters), the current settings index, and the index of the
    scan mode to be used for record behavior.

    If your scan module subclasses ScanFrameParameters, provide a frame_parameters_factory function to construct
    your subclass from a dictionary.
    """

    def __init__(self,
                 scan_modes: typing.Sequence[ScanSettingsMode],
                 frame_parameters_factory: ScanFrameParametersFactory,
                 current_settings_index: int = 0,
                 record_settings_index: int = 0,
                 open_configuration_dialog_fn: typing.Optional[typing.Callable[[], None]] = None) -> None:
        assert len(scan_modes) > 0

        # these events must be defined
        self.current_frame_parameters_changed_event = Event.Event()
        self.record_frame_parameters_changed_event = Event.Event()
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()

        # optional event and identifier for settings. defining settings_id signals that
        # the settings should be managed as a dict by the container of this class. the container
        # will call apply_settings to initialize settings and then expect settings_changed_event
        # to be fired when settings change.
        self.settings_changed_event = Event.Event()
        self.settings_id = str()

        # scan specific
        self.__scan_modes = tuple(scan_modes)
        self.__current_settings_index = current_settings_index
        self.__record_settings_index = record_settings_index
        self.__frame_parameters_factory = frame_parameters_factory
        self.__frame_parameters = self.__scan_modes[self.__current_settings_index].frame_parameters
        self.__record_parameters = self.__scan_modes[self.__record_settings_index].frame_parameters

        # dialogs
        self.__open_configuration_dialog_fn = open_configuration_dialog_fn

    def close(self) -> None:
        pass

    def initialize(self, configuration_location: typing.Optional[pathlib.Path] = None, event_loop: typing.Optional[asyncio.AbstractEventLoop] = None, **kwargs: typing.Any) -> None:
        pass

    def apply_settings(self, settings_dict: typing.Mapping[str, typing.Any]) -> None:
        """Initialize the settings with the settings_dict."""
        pass

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> ScanFrameParameters:
        """Return camera frame parameters from dict."""
        return self.__frame_parameters_factory(d)

    def __set_current_frame_parameters(self, frame_parameters: ScanFrameParameters) -> None:
        """Set the current frame parameters.

        Fire the current frame parameters changed event and optionally the settings changed event.
        """
        self.__frame_parameters = copy.copy(frame_parameters)
        # self.settings_changed_event.fire(self.__save_settings())
        self.current_frame_parameters_changed_event.fire(frame_parameters)

    def get_current_frame_parameters(self) -> ScanFrameParameters:
        """Get the current frame parameters."""
        return copy.copy(self.__frame_parameters)

    def __set_record_frame_parameters(self, frame_parameters: ScanFrameParameters) -> None:
        """Set the record frame parameters.

        Fire the record frame parameters changed event and optionally the settings changed event.
        """
        self.__record_parameters = copy.copy(frame_parameters)
        self.record_frame_parameters_changed_event.fire(frame_parameters)

    def get_record_frame_parameters(self) -> ScanFrameParameters:
        """Get the record frame parameters."""
        return self.__record_parameters

    def set_frame_parameters(self, settings_index: int, frame_parameters: ScanFrameParameters) -> None:
        """Set the frame parameters with the settings index and fire the frame parameters changed event.

        If the settings index matches the current settings index, call set current frame parameters.

        If the settings index matches the record settings index, call set record frame parameters.
        """
        assert 0 <= settings_index < len(self.__scan_modes)
        frame_parameters = copy.copy(frame_parameters)
        self.__scan_modes[settings_index].frame_parameters = frame_parameters
        # self.settings_changed_event.fire(self.__save_settings())
        self.frame_parameters_changed_event.fire(settings_index, frame_parameters)
        # update the local frame parameters.
        # in order to facilitate profile changes from the SuperScan, which are sent partially
        # when setting the frame parameters, do this _after_ the profile parameters have changed
        # (in the frame_parameters_changed_event). this way, setting the current frame parameters
        # shouldn't cause partially updated profile parameters to be sent from the low level.
        # this all needs to be reworked once the SuperScan UI behaves properly.
        if settings_index == self.__current_settings_index:
            self.__set_current_frame_parameters(frame_parameters)
        if settings_index == self.__record_settings_index:
            self.__set_record_frame_parameters(frame_parameters)

    def get_frame_parameters(self, settings_index: int) -> ScanFrameParameters:
        """Get the frame parameters for the settings index."""
        return copy.copy(self.__scan_modes[settings_index].frame_parameters)

    def set_selected_profile_index(self, settings_index: int) -> None:
        """Set the current settings index.

        Call set current frame parameters if it changed.

        Fire profile changed event if it changed.
        """
        assert 0 <= settings_index < len(self.__scan_modes)
        if self.__current_settings_index != settings_index:
            self.__current_settings_index = settings_index
            # set current frame parameters
            self.__set_current_frame_parameters(self.__scan_modes[self.__current_settings_index].frame_parameters)
            # self.settings_changed_event.fire(self.__save_settings())
            self.profile_changed_event.fire(settings_index)

    @property
    def selected_profile_index(self) -> int:
        """Return the current settings index."""
        return self.__current_settings_index

    @property
    def scan_modes(self) -> typing.Sequence[ScanSettingsMode]:
        return tuple(self.__scan_modes)

    @property
    def modes(self) -> typing.Sequence[str]:
        return tuple(scan_mode.name for scan_mode in self.__scan_modes)

    def get_mode(self) -> str:
        """Return the current mode (named version of current settings index)."""
        return self.__scan_modes[self.__current_settings_index].name

    def set_mode(self, mode: str) -> None:
        """Set the current mode (named version of current settings index)."""
        self.set_selected_profile_index(self.modes.index(mode))

    def open_configuration_interface(self, api_broker: typing.Any) -> None:
        if callable(self.__open_configuration_dialog_fn):
            self.__open_configuration_dialog_fn()


class ConcreteScanHardwareSource(HardwareSource.ConcreteHardwareSource, ScanHardwareSource):

    def __init__(self, stem_controller_: stem_controller_module.STEMController, device: ScanDevice, settings: ScanSettingsProtocol, configuration_location: typing.Optional[pathlib.Path], panel_type: typing.Optional[str] = None) -> None:
        super().__init__(device.scan_device_id, device.scan_device_name)

        # configure the event loop object
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(logging.INFO)
        self.__event_loop = asyncio.new_event_loop()  # outputs a debugger message!
        logger.setLevel(old_level)

        self.__settings = settings
        self.__settings.initialize(configuration_location=configuration_location, event_loop=self.__event_loop)

        self.__current_frame_parameters_changed_event_listener = self.__settings.current_frame_parameters_changed_event.listen(self.__set_current_frame_parameters)
        self.__record_frame_parameters_changed_event_listener = self.__settings.record_frame_parameters_changed_event.listen(self.__set_record_frame_parameters)

        # add optional support for settings. to enable auto settings handling, the camera settings object must define
        # a settings_id property (which can just be the camera id), an apply_settings method which takes a settings
        # dict read from the config file and applies it as the settings, and a settings_changed_event which must be
        # fired when the settings changed (at which point they will be written to the config file).
        self.__settings_changed_event_listener = None
        if configuration_location and self.__settings.settings_id:
            config_file = configuration_location / pathlib.Path(self.__settings.settings_id + "_config.json")
            logging.info("Scan device configuration: " + str(config_file))
            if config_file.is_file():
                with open(config_file) as f:
                    settings_dict = json.load(f)
                self.__settings.apply_settings(settings_dict)

            def settings_changed(settings_dict: typing.Mapping[str, typing.Any]) -> None:
                # atomically overwrite
                temp_filepath = config_file.with_suffix(".temp")
                with open(temp_filepath, "w") as fp:
                    json.dump(settings_dict, fp, skipkeys=True, indent=4)
                os.replace(temp_filepath, config_file)

            self.__settings_changed_event_listener = self.__settings.settings_changed_event.listen(settings_changed)

        self.features["is_scanning"] = True
        if panel_type:
            self.features["panel_type"] = panel_type

        # define events
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()
        self.probe_state_changed_event = Event.Event()
        self.channel_state_changed_event = Event.Event()

        self.__channel_states_lock = threading.RLock()
        self.__channel_states: typing.List[ChannelState] = list()
        self.__pending_channel_states: typing.Optional[typing.List[ChannelState]] = None

        self.__profile_changed_event_listener = self.__settings.profile_changed_event.listen(self.profile_changed_event.fire)
        self.__frame_parameters_changed_event_listener = self.__settings.frame_parameters_changed_event.listen(self.frame_parameters_changed_event.fire)

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
            self.add_data_channel(channel_info.channel_id, channel_info.name, is_context=True)
        # add an associated sub-scan channel for each device channel
        for channel_index, channel_info in enumerate(channel_info_list):
            subscan_channel_index, subscan_channel_id, subscan_channel_name = self.get_subscan_channel_info(channel_index, channel_info.channel_id , channel_info.name)
            self.add_data_channel(subscan_channel_id, subscan_channel_name)
        self.add_data_channel("drift", _("Drift"))

        self.__last_idle_position: typing.Optional[Geometry.FloatPoint] = None  # used for testing

        self.__frame_parameters = self.__settings.get_current_frame_parameters()
        self.__record_parameters = self.__settings.get_record_frame_parameters()

        self.__acquisition_task: typing.Optional[HardwareSource.AcquisitionTask] = None
        # the task queue is a list of tasks that must be executed on the UI thread. items are added to the queue
        # and executed at a later time in the __handle_executing_task_queue method.
        self.__task_queue: queue.Queue[typing.Callable[[], None]] = queue.Queue()
        self.__latest_values_lock = threading.RLock()
        self.__latest_values: typing.Dict[int, ScanFrameParameters] = dict()
        self.record_index = 1  # use to give unique name to recorded images

        # synchronized acquisition
        self.acquisition_state_changed_event = Event.Event()

    def close(self) -> None:
        Process.close_event_loop(self.__event_loop)
        self.__event_loop = typing.cast(asyncio.AbstractEventLoop, None)
        # thread needs to close before closing the stem controller. so use this method to
        # do it slightly out of order for this class.
        self.close_thread()
        # when overriding hardware source close, the acquisition loop may still be running
        # so nothing can be changed here that will make the acquisition loop fail.
        self.__stem_controller.disconnect_probe_connections()
        if self.__probe_state_changed_event_listener:
            self.__probe_state_changed_event_listener.close()
            self.__probe_state_changed_event_listener = typing.cast(typing.Any, None)
        if self.__subscan_region_changed_event_listener:
            self.__subscan_region_changed_event_listener.close()
            self.__subscan_region_changed_event_listener = typing.cast(typing.Any, None)
        if self.__subscan_rotation_changed_event_listener:
            self.__subscan_rotation_changed_event_listener.close()
            self.__subscan_rotation_changed_event_listener = typing.cast(typing.Any, None)
        if self.__line_scan_vector_changed_event_listener:
            self.__line_scan_vector_changed_event_listener.close()
            self.__line_scan_vector_changed_event_listener = typing.cast(typing.Any, None)
        super().close()
        if self.__settings_changed_event_listener:
            self.__settings_changed_event_listener.close()
            self.__settings_changed_event_listener = None
        self.__profile_changed_event_listener.close()
        self.__profile_changed_event_listener = typing.cast(typing.Any, None)
        self.__frame_parameters_changed_event_listener.close()
        self.__frame_parameters_changed_event_listener = typing.cast(typing.Any, None)
        self.__current_frame_parameters_changed_event_listener.close()
        self.__current_frame_parameters_changed_event_listener = typing.cast(typing.Any, None)
        self.__record_frame_parameters_changed_event_listener.close()
        self.__record_frame_parameters_changed_event_listener = typing.cast(typing.Any, None)
        self.__settings.close()
        self.__settings = typing.cast(typing.Any, None)

        # keep the device around until super close is called, since super
        # may do something that requires the device.
        self.__device.save_frame_parameters()
        self.__device.close()
        self.__device = typing.cast(ScanDevice, None)

    def periodic(self) -> None:
        self.__event_loop.stop()
        self.__event_loop.run_forever()
        self.__handle_executing_task_queue()

    @property
    def drift_tracker(self) -> typing.Optional[DriftTracker.DriftTracker]:
        return self.__stem_controller.drift_tracker

    def __handle_executing_task_queue(self) -> None:
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
    def scan_context(self) -> stem_controller_module.ScanContext:
        return self.__stem_controller.scan_context

    @property
    def scan_settings(self) -> ScanSettingsProtocol:
        return self.__settings

    @property
    def stem_controller(self) -> stem_controller_module.STEMController:
        return self.__stem_controller

    @property
    def scan_device(self) -> ScanDevice:
        return self.__device

    def start_playing(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if "frame_parameters" in kwargs:
            frame_parameters = self.__settings.get_frame_parameters_from_dict(kwargs["frame_parameters"])
            self.set_current_frame_parameters(frame_parameters)
        elif len(args) == 1 and isinstance(args[0], dict):
            frame_parameters = self.__settings.get_frame_parameters_from_dict(args[0])
            self.set_current_frame_parameters(frame_parameters)
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

    def grab_next_to_start(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        self.start_playing()
        return self.get_next_xdatas_to_start(timeout)

    def grab_next_to_finish(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        self.start_playing()
        return self.get_next_xdatas_to_finish(timeout)

    def grab_sequence_prepare(self, count: int, **kwargs: typing.Any) -> bool:
        return False

    def grab_sequence(self, count: int, **kwargs: typing.Any) -> typing.Optional[typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]]:
        return None

    def grab_synchronized_get_info(self, *, scan_frame_parameters: ScanFrameParameters, camera: camera_base.CameraHardwareSource, camera_frame_parameters: camera_base.CameraFrameParameters) -> GrabSynchronizedInfo:
        scan_max_area = 2048 * 2048
        subscan_pixel_size = scan_frame_parameters.subscan_pixel_size
        if subscan_pixel_size:
            scan_param_height = subscan_pixel_size.height
            scan_param_width = subscan_pixel_size.width
            if scan_param_height * scan_param_width > scan_max_area:
                scan_param_height = scan_max_area // scan_param_width
            fractional_size = scan_frame_parameters.subscan_fractional_size
            fractional_center = scan_frame_parameters.subscan_fractional_center
            assert fractional_size and fractional_center
            fractional_area = Geometry.FloatRect.from_center_and_size(fractional_center, fractional_size)
            is_subscan = True
        else:
            scan_param_height = scan_frame_parameters.pixel_size.height
            scan_param_width = scan_frame_parameters.pixel_size.width
            if scan_param_height * scan_param_width > scan_max_area:
                scan_param_height = scan_max_area // scan_param_width
            fractional_area = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.5, x=0.5), Geometry.FloatSize(h=1.0, w=1.0))
            is_subscan = False

        camera_readout_size = Geometry.IntSize.make(camera.get_expected_dimensions(camera_frame_parameters.binning))

        scan_size = Geometry.IntSize(h=scan_param_height, w=scan_param_width)

        camera_readout_size_squeezed: typing.Tuple[int, ...]
        if camera_frame_parameters.processing == "sum_project":
            camera_readout_size_squeezed = (camera_readout_size.width,)
            axes_descriptor = AxesDescriptor(None, [0, 1], [2])
        elif camera_frame_parameters.processing == "sum_masked":
            camera_readout_size_squeezed = (max(len(camera_frame_parameters.active_masks), 1),)
            axes_descriptor = AxesDescriptor(None, [2], [0, 1])
        else:
            camera_readout_size_squeezed = tuple(camera_readout_size)
            axes_descriptor = AxesDescriptor(None, [0, 1], [2, 3])

        scan_calibrations = get_scan_calibrations(scan_frame_parameters)

        data_calibrations = camera.get_camera_calibrations(camera_frame_parameters)
        data_intensity_calibration = camera.get_camera_intensity_calibration(camera_frame_parameters)

        camera_metadata: typing.Dict[str, typing.Any] = dict()
        camera.update_camera_properties(camera_metadata, camera_frame_parameters)

        scan_metadata: typing.Dict[str, typing.Any] = dict()
        update_scan_metadata(scan_metadata, self.hardware_source_id, self.display_name, copy.copy(scan_frame_parameters), None, dict())

        instrument_metadata: typing.Dict[str, typing.Any] = dict()
        update_instrument_properties(instrument_metadata, self.__stem_controller, self.__device)

        return GrabSynchronizedInfo(scan_size, fractional_area, is_subscan, camera_readout_size,
                                    camera_readout_size_squeezed, scan_calibrations, data_calibrations,
                                    data_intensity_calibration, instrument_metadata, camera_metadata,
                                    scan_metadata, axes_descriptor)

    def grab_synchronized(self, *,
                          data_channel: typing.Optional[Acquisition.DataChannel] = None,
                          scan_frame_parameters: ScanFrameParameters,
                          camera: camera_base.CameraHardwareSource,
                          camera_frame_parameters: camera_base.CameraFrameParameters,
                          camera_data_channel: typing.Optional[camera_base.SynchronizedDataChannelInterface] = None,
                          section_height: typing.Optional[int] = None,
                          scan_data_stream_functor: typing.Optional[Acquisition.DataStreamFunctor] = None,
                          scan_count: int = 1) -> GrabSynchronizedResult:
        synchronized_scan_data_stream = make_synchronized_scan_data_stream(scan_hardware_source=self,
                                                                           scan_frame_parameters=copy.copy(scan_frame_parameters),
                                                                           camera_hardware_source=camera,
                                                                           camera_frame_parameters=camera_frame_parameters,
                                                                           camera_data_channel=camera_data_channel,
                                                                           scan_data_stream_functor=scan_data_stream_functor,
                                                                           section_height=section_height,
                                                                           scan_count=scan_count,
                                                                           old_move_axis=camera_data_channel is not None)
        result_data_stream = Acquisition.FramedDataStream(synchronized_scan_data_stream, data_channel=data_channel)
        scan_acquisition = Acquisition.Acquisition(result_data_stream)
        with result_data_stream.ref(), contextlib.closing(scan_acquisition):
            results: GrabSynchronizedResult = None
            self.__scan_acquisition = scan_acquisition
            try:
                scan_acquisition.prepare_acquire()
                scan_acquisition.acquire()
                if scan_acquisition.is_error:
                    raise RuntimeError("grab_synchronized failed.")
                if not scan_acquisition.is_aborted:
                    scan_results = [result_data_stream.get_data(c) for c in result_data_stream.channels if c.segments[0] == self.hardware_source_id]
                    camera_results = [result_data_stream.get_data(c) for c in result_data_stream.channels if c.segments[0] == camera.hardware_source_id]
                    results = (scan_results, camera_results)
            finally:
                self.__scan_acquisition = typing.cast(typing.Any, None)
            return results

    def grab_synchronized_abort(self) -> None:
        if self.__scan_acquisition:
            self.__scan_acquisition.abort_acquire()

    def grab_synchronized_get_progress(self) -> typing.Optional[float]:
        return None

    def grab_buffer(self, count: int, *, start: typing.Optional[int] = None, **kwargs: typing.Any) -> typing.Optional[typing.List[typing.List[DataAndMetadata.DataAndMetadata]]]:
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
    def subscan_state(self) -> stem_controller_module.SubscanState:
        return self.__stem_controller.subscan_state

    @property
    def subscan_enabled(self) -> bool:
        return self.__stem_controller.subscan_state == stem_controller_module.SubscanState.ENABLED

    @subscan_enabled.setter
    def subscan_enabled(self, enabled: bool) -> None:
        if enabled:
            self.__stem_controller.subscan_state = stem_controller_module.SubscanState.ENABLED
        else:
            self.__stem_controller.subscan_state = stem_controller_module.SubscanState.DISABLED
            self.__stem_controller._update_scan_context(self.__frame_parameters.pixel_size, self.__frame_parameters.center_nm, self.__frame_parameters.fov_nm, self.__frame_parameters.rotation_rad)

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
    def line_scan_state(self) -> stem_controller_module.LineScanState:
        return self.__stem_controller.line_scan_state

    @property
    def line_scan_enabled(self) -> bool:
        return self.__stem_controller.line_scan_state == stem_controller_module.LineScanState.ENABLED

    @line_scan_enabled.setter
    def line_scan_enabled(self, enabled: bool) -> None:
        if enabled:
            self.__stem_controller.line_scan_state = stem_controller_module.LineScanState.ENABLED
        else:
            self.__stem_controller.line_scan_state = stem_controller_module.LineScanState.DISABLED
            self.__stem_controller._update_scan_context(self.__frame_parameters.pixel_size, self.__frame_parameters.center_nm, self.__frame_parameters.fov_nm, self.__frame_parameters.rotation_rad)

    @property
    def line_scan_vector(self) -> typing.Optional[typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]]:
        return self.__stem_controller.line_scan_vector

    def apply_scan_context_subscan(self, frame_parameters: ScanFrameParameters, size: typing.Optional[typing.Tuple[int, int]] = None) -> None:
        scan_context = self.scan_context
        if scan_context.is_valid:
            scan_context_size = scan_context.size
            scan_context_center_nm = scan_context.center_nm
            assert scan_context_size
            assert scan_context_center_nm
            frame_parameters.pixel_size = scan_context_size
            frame_parameters.center_nm = scan_context_center_nm
            frame_parameters.fov_nm = scan_context.fov_nm or 8.0
            frame_parameters.rotation_rad = scan_context.rotation_rad or 0.0
        self.__apply_subscan_parameters(frame_parameters, size)

    def __apply_subscan_parameters(self, frame_parameters: ScanFrameParameters, size_tuple: typing.Optional[typing.Tuple[int, int]] = None) -> None:
        context_size = frame_parameters.pixel_size.to_float_size()
        size = Geometry.IntSize.make(size_tuple) if size_tuple else None
        if self.subscan_enabled and self.subscan_region:
            subscan_region = self.subscan_region
            subscan_pixel_size = size or Geometry.IntSize(max(int(context_size.height * subscan_region.height), 1), max(int(context_size.width * subscan_region.width), 1))
            frame_parameters.subscan_pixel_size = subscan_pixel_size
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(max(subscan_region.height, 1 / context_size.height), max(subscan_region.width, 1 / context_size.width))
            frame_parameters.subscan_fractional_center = subscan_region.center
            frame_parameters.subscan_rotation = self.subscan_rotation
        elif self.line_scan_enabled and self.line_scan_vector:
            line_scan_vector = self.line_scan_vector
            start = Geometry.FloatPoint.make(line_scan_vector[0])
            end = Geometry.FloatPoint.make(line_scan_vector[1])
            length = Geometry.distance(start, end)
            length = max(length, max(1 / context_size.width, 1 / context_size.height))
            subscan_pixel_length = max(int(context_size.width * length), 1)
            subscan_pixel_size = size or Geometry.IntSize(1, subscan_pixel_length)
            frame_parameters.subscan_pixel_size = subscan_pixel_size
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(1 / context_size.height, length)
            frame_parameters.subscan_fractional_center = Geometry.midpoint(start, end)
            frame_parameters.subscan_rotation = -math.atan2(end.y - start.y, end.x - start.x)
        else:
            frame_parameters.pixel_size = size if size else frame_parameters.pixel_size
            frame_parameters.subscan_pixel_size = None
            frame_parameters.subscan_fractional_size = None
            frame_parameters.subscan_fractional_center = None
            frame_parameters.subscan_rotation = 0.0

    def apply_subscan(self, frame_parameters: ScanFrameParameters) -> None:
        if frame_parameters.subscan_fractional_size and frame_parameters.subscan_fractional_center:
            pass  # let the parameters speak for themselves
        else:
            self.__apply_subscan_parameters(frame_parameters)

    def __subscan_state_changed(self, name: str) -> None:
        if name == "subscan_state":
            # if subscan enabled, ensure there is a subscan region
            if self.__stem_controller.subscan_state == stem_controller_module.SubscanState.ENABLED and not self.__stem_controller.subscan_region:
                self.__stem_controller.subscan_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
                self.__stem_controller.subscan_rotation = 0.0
            # otherwise let __set_current_frame_parameters clean up existing __frame_parameters
            self.__set_current_frame_parameters(self.__frame_parameters)

    def __subscan_region_changed(self, name: str) -> None:
        if name == "subscan_region":
            subscan_region = self.subscan_region
            if not subscan_region:
                self.subscan_enabled = False
            self.__set_current_frame_parameters(self.__frame_parameters)

    def __subscan_rotation_changed(self, name: str) -> None:
        if name == "subscan_rotation":
            self.__set_current_frame_parameters(self.__frame_parameters)

    def __line_scan_state_changed(self, name: str) -> None:
        if name == "line_scan_state":
            # if line scan enabled, ensure there is a line scan region
            if self.__stem_controller.line_scan_state == stem_controller_module.LineScanState.ENABLED and not self.__stem_controller.line_scan_vector:
                self.__stem_controller.line_scan_vector = (0.25, 0.25), (0.75, 0.75)
            # otherwise let __set_current_frame_parameters clean up existing __frame_parameters
            self.__set_current_frame_parameters(self.__frame_parameters)

    def __line_scan_vector_changed(self, name: str) -> None:
        if name == "line_scan_vector":
            line_scan_vector = self.line_scan_vector
            if not line_scan_vector:
                self.line_scan_enabled = False
            self.__set_current_frame_parameters(self.__frame_parameters)

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
    def drift_settings(self) -> stem_controller_module.DriftCorrectionSettings:
        return self.__stem_controller.drift_settings

    @drift_settings.setter
    def drift_settings(self, value: stem_controller_module.DriftCorrectionSettings) -> None:
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
            assert isinstance(self.drift_settings.interval_units, stem_controller_module.DriftIntervalUnit)
            if self.drift_settings.interval_units == stem_controller_module.DriftIntervalUnit.FRAME:
                lines = max(1, math.ceil(self.drift_settings.interval / width))
            elif self.drift_settings.interval_units == stem_controller_module.DriftIntervalUnit.TIME:
                lines = max(1, math.ceil(self.drift_settings.interval / frame_time / width))
            elif self.drift_settings.interval_units == stem_controller_module.DriftIntervalUnit.LINE:
                lines = self.drift_settings.interval
            else:  # drift per scans
                lines = 0
            return int(lines)
        return 0

    def calculate_drift_scans(self) -> int:
        if self.drift_valid:
            assert isinstance(self.drift_settings.interval_units, stem_controller_module.DriftIntervalUnit)
            if self.drift_settings.interval_units == stem_controller_module.DriftIntervalUnit.SCAN:
                return max(0, int(self.drift_settings.interval))
        return 0

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        channel_count = self.__device.channel_count
        channel_states = [self.get_channel_state(i) for i in range(channel_count)]
        if not self.subscan_enabled:
            self.__stem_controller._update_scan_context(self.__frame_parameters.pixel_size, self.__frame_parameters.center_nm, self.__frame_parameters.fov_nm, self.__frame_parameters.rotation_rad)
        frame_parameters = copy.deepcopy(self.__frame_parameters)
        channel_ids = [channel_state.channel_id for channel_state in channel_states]
        return ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, True, frame_parameters, None, channel_ids, self.display_name)

    def _view_task_updated(self, view_task: typing.Optional[HardwareSource.AcquisitionTask]) -> None:
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self, *, frame_parameters: typing.Optional[HardwareSource.FrameParameters] = None, **kwargs: typing.Any) -> HardwareSource.AcquisitionTask:
        # frame_parameters is typed as HardwareSource.FrameParameters. construct a ScanFrameParameters from
        # their dict via the settings to allow it to customize the frame parameters.
        record_parameters = self.__settings.get_frame_parameters_from_dict(frame_parameters.as_dict()) if frame_parameters else self.__record_parameters
        assert record_parameters is not None
        acquisition_task_parameters = kwargs.get("acquisition_task_parameters", ScanAcquisitionTaskParameters())
        channel_count = self.__device.channel_count
        channel_states = [self.get_channel_state(i) for i in range(channel_count)]
        frame_parameters = copy.deepcopy(record_parameters)
        channel_ids = [channel_state.channel_id for channel_state in channel_states]
        return ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, False, frame_parameters, acquisition_task_parameters, channel_ids, self.display_name)

    def record_immediate(self, frame_parameters: ScanFrameParameters,
                         enabled_channels: typing.Optional[typing.Sequence[int]] = None,
                         sync_timeout: typing.Optional[float] = None) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        assert not self.is_recording
        frame_parameters = copy.deepcopy(frame_parameters)
        old_enabled_channels = self.get_enabled_channels()
        channel_states = [self.get_channel_state(i) for i in range(self.__device.channel_count)]
        channel_ids = [channel_state.channel_id for channel_state in channel_states]
        if enabled_channels is not None:
            self.set_enabled_channels(enabled_channels)
        record_task = ScanAcquisitionTask(self.__stem_controller, self, self.__device, self.hardware_source_id, False, frame_parameters, None, channel_ids, self.display_name)
        finished_event = threading.Event()
        xdatas: typing.List[typing.Optional[DataAndMetadata.DataAndMetadata]] = list()
        def finished(datas_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:
            nonlocal xdatas
            xdatas = [data_promise.xdata for data_promise in datas_promises]
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

    def set_frame_parameters(self, profile_index: int, frame_parameters: ScanFrameParameters) -> None:
        self.__settings.set_frame_parameters(profile_index, frame_parameters)

    def get_frame_parameters(self, profile_index: int) -> ScanFrameParameters:
        return self.__settings.get_frame_parameters(profile_index)

    def set_current_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None:
        scan_frame_parameters = typing.cast(ScanFrameParameters, frame_parameters)
        self.__set_current_frame_parameters(scan_frame_parameters)

    def __set_current_frame_parameters(self, frame_parameters: ScanFrameParameters, update_task: bool = True) -> None:
        frame_parameters = copy.copy(frame_parameters)
        self.__apply_subscan_parameters(frame_parameters)
        acquisition_task = self.__acquisition_task
        if isinstance(acquisition_task, ScanAcquisitionTask):
            if update_task:
                acquisition_task.set_frame_parameters(frame_parameters)
            if not self.subscan_enabled:
                self.__stem_controller._update_scan_context(frame_parameters.pixel_size, frame_parameters.center_nm, frame_parameters.fov_nm, frame_parameters.rotation_rad)
            else:
                # subscan is active, decide if the changed frame parameters should clear the scan context
                self.__stem_controller._confirm_scan_context(frame_parameters.pixel_size, frame_parameters.center_nm, frame_parameters.fov_nm, frame_parameters.rotation_rad)
        elif update_task:
            # handle case where current profile has been changed but scan is not running.
            device_frame_parameters = copy.copy(frame_parameters)
            self.__device.set_frame_parameters(device_frame_parameters)
            self.__stem_controller._confirm_scan_context(frame_parameters.pixel_size, frame_parameters.center_nm, frame_parameters.fov_nm, frame_parameters.rotation_rad)
        self.__frame_parameters = copy.copy(frame_parameters)

    def get_current_frame_parameters(self) -> ScanFrameParameters:
        return copy.copy(self.__frame_parameters)

    def set_record_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None:
        self.__set_record_frame_parameters(frame_parameters)

    def __set_record_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None:
        # frame_parameters is typed as HardwareSource.FrameParameters. construct a ScanFrameParameters from
        # their dict via the settings to allow it to customize the frame parameters.
        self.__record_parameters = self.__settings.get_frame_parameters_from_dict(frame_parameters.as_dict())

    def get_record_frame_parameters(self) -> HardwareSource.FrameParameters:
        return copy.copy(self.__record_parameters)

    @property
    def channel_count(self) -> int:
        return len(self.__device.channels_enabled)

    def get_channel_state(self, channel_index: int) -> ChannelState:
        channels_enabled = self.__device.channels_enabled
        assert 0 <= channel_index < len(channels_enabled)
        name = self.__device.get_channel_name(channel_index)
        return self.__make_channel_state(channel_index, name, channels_enabled[channel_index])

    def get_channel_enabled(self, channel_index: int) -> bool:
        assert 0 <= channel_index < self.__device.channel_count
        return self.__device.channels_enabled[channel_index]

    def set_channel_enabled(self, channel_index: int, enabled: bool) -> None:
        changed = self.__device.set_channel_enabled(channel_index, enabled)
        if changed:
            self.__channel_states_changed([self.get_channel_state(i) for i in range(self.channel_count)])

    def get_subscan_channel_info(self, channel_index: int, channel_id: str, channel_name: str) -> typing.Tuple[int, str, str]:
        return channel_index + self.channel_count, channel_id + "_subscan", " ".join((channel_name, _("SubScan")))

    def get_data_channel_state(self, channel_index: int) -> typing.Tuple[typing.Optional[str], typing.Optional[str], bool]:
        # channel indexes larger than then the channel count will be subscan channels
        if channel_index < self.channel_count:
            channel_state = self.get_channel_state(channel_index)
            return channel_state.channel_id, channel_state.name, channel_state.enabled if not self.subscan_enabled else False
        elif channel_index < self.channel_count * 2:
            channel_state = self.get_channel_state(channel_index - self.channel_count)
            subscan_channel_index, subscan_channel_id, subscan_channel_name = self.get_subscan_channel_info(channel_index, channel_state.channel_id, channel_state.name)
            return subscan_channel_id, subscan_channel_name, channel_state.enabled if self.subscan_enabled else False
        else:
            return self.data_channels[channel_index].channel_id, self.data_channels[channel_index].name, False

    def get_channel_index_for_data_channel_index(self, data_channel_index: int) -> int:
        return data_channel_index % self.channel_count

    def record_async(self, callback_fn: typing.Callable[[typing.Sequence[HardwareSource.DataAndMetadataPromise]], None]) -> None:
        """ Call this when the user clicks the record button. """
        assert callable(callback_fn)

        def record_thread() -> None:
            current_frame_time = self.get_current_frame_time()

            def handle_finished(data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:
                callback_fn(data_promises)

            self.start_recording(current_frame_time, finished_callback_fn=handle_finished)

        self.__thread = threading.Thread(target=record_thread)
        self.__thread.start()

    def set_selected_profile_index(self, profile_index: int) -> None:
        self.__settings.set_selected_profile_index(profile_index)

    @property
    def selected_profile_index(self) -> int:
        return self.__settings.selected_profile_index

    def __profile_frame_parameters_changed(self, profile_index: int, frame_parameters: ScanFrameParameters) -> None:
        # to avoid set-notify cycles, only set the low level again if parameters have changed. this is handled
        # in the scan acquisition task.
        with self.__latest_values_lock:
            self.__latest_values[profile_index] = copy.copy(frame_parameters)
        def do_update_parameters() -> None:
            with self.__latest_values_lock:
                for profile_index in self.__latest_values.keys():
                    self.__settings.set_frame_parameters(profile_index, self.__latest_values[profile_index])
                self.__latest_values = dict()
        self.__task_queue.put(do_update_parameters)

    def __channel_states_changed(self, channel_states: typing.List[ChannelState]) -> None:
        # this method will be called when the device changes channels enabled (via dialog or script).
        # it updates the channels internally but does not send out a message to set the channels to the
        # hardware, since they're already set, and doing so can cause strange change loops.
        channel_count = self.channel_count
        assert len(channel_states) == channel_count
        def channel_states_changed() -> None:
            with self.__channel_states_lock:
                self.__channel_states = self.__pending_channel_states or list()
                self.__pending_channel_states = None
            for channel_index, channel_state in enumerate(self.__channel_states):
                self.channel_state_changed_event.fire(channel_index, channel_state.channel_id, channel_state.name, channel_state.enabled)
            at_least_one_enabled = False
            for channel_index in range(channel_count):
                if self.get_channel_state(channel_index).enabled:
                    at_least_one_enabled = True
                    break
            if not at_least_one_enabled:
                self.stop_playing()
        with self.__channel_states_lock:
            if list(channel_states) != list(self.__channel_states):
                self.__pending_channel_states = list(channel_states)
                self.__task_queue.put(channel_states_changed)

    def get_channel_index(self, channel_id: str) -> typing.Optional[int]:
        for channel_index in range(self.channel_count):
            if self.get_channel_state(channel_index).channel_id == channel_id:
                return channel_index
        return None

    def __make_channel_id(self, channel_index: int) -> str:
        return "abcdefghijklmnopqrstuvwxyz"[channel_index]

    def __make_channel_state(self, channel_index: int, channel_name: str, channel_enabled: bool) -> ChannelState:
        return ChannelState(self.__make_channel_id(channel_index), channel_name, channel_enabled)

    def __device_state_changed(self, profile_frame_parameters_list: typing.Sequence[ScanFrameParameters], device_channel_states: typing.Sequence[typing.Tuple[str, bool]]) -> None:
        for profile_index, profile_frame_parameters in enumerate(profile_frame_parameters_list):
            self.__profile_frame_parameters_changed(profile_index, profile_frame_parameters)
        channel_states = list()
        for channel_index, (channel_name, channel_enabled) in enumerate(device_channel_states):
            channel_states.append(self.__make_channel_state(channel_index, channel_name, channel_enabled))
        self.__channel_states_changed(channel_states)

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> ScanFrameParameters:
        return self.__settings.get_frame_parameters_from_dict(d)

    def calculate_frame_time(self, frame_parameters: ScanFrameParameters) -> float:
        size = frame_parameters.pixel_size
        pixel_time_us = frame_parameters.pixel_time_us
        return size.height * size.width * pixel_time_us / 1000000.0

    def get_current_frame_time(self) -> float:
        return self.calculate_frame_time(self.get_current_frame_parameters())

    def get_record_frame_time(self) -> float:
        frame_parameters = self.get_record_frame_parameters()
        scan_frame_parameters = typing.cast(ScanFrameParameters, frame_parameters)
        return self.calculate_frame_time(scan_frame_parameters)

    def make_reference_key(self, **kwargs: typing.Any) -> str:
        # NOTE: this is a public API method currently. see scanning.rst.
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

    def get_buffer_data(self, start: int, count: int) -> typing.List[typing.List[typing.Dict[str, typing.Any]]]:
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
                    data_element: typing.Dict[str, typing.Any] = {"metadata": dict()}
                    instrument_metadata: typing.Dict[str, typing.Any] = dict()
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

        return list()

    def scan_immediate(self, frame_parameters: ScanFrameParameters) -> None:
        old_frame_parameters = self.__device.current_frame_parameters
        self.__device.set_frame_parameters(frame_parameters)
        frame_number = self.__device.start_frame(False)
        self.__device.wait_for_frame(frame_number)
        self.__device.stop()
        self.__device.set_frame_parameters(old_frame_parameters)

    def calculate_flyback_pixels(self, frame_parameters: ScanFrameParameters) -> int:
        if callable(getattr(self.__device, "calculate_flyback_pixels", None)):
            return self.__device.calculate_flyback_pixels(frame_parameters)
        return getattr(self.__device, "flyback_pixels", 0)

    def __probe_state_changed(self, probe_state: str, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        # subclasses will override _set_probe_position
        # probe_state can be 'parked', or 'scanning'
        self._set_probe_position(probe_position)
        # update the probe position for listeners and also explicitly update for probe_graphic_connections.
        self.probe_state_changed_event.fire(probe_state, probe_position)

    def _enter_scanning_state(self) -> None:
        """Enter scanning state. Acquisition task will call this. Tell the STEM controller."""
        self.__stem_controller._enter_scanning_state()

    def _exit_scanning_state(self) -> None:
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
            self.__last_idle_position = Geometry.FloatPoint(x=-1.0, y=-1.0)

    def _get_last_idle_position_for_test(self) -> typing.Optional[Geometry.FloatPoint]:
        return self.__last_idle_position

    @property
    def probe_position(self) -> typing.Optional[Geometry.FloatPoint]:
        return self.__stem_controller.probe_position

    @probe_position.setter
    def probe_position(self, probe_position: typing.Optional[Geometry.FloatPointTuple]) -> None:
        probe_position = Geometry.FloatPoint.make(probe_position) if probe_position else None
        self.__stem_controller.set_probe_position(probe_position)

    def validate_probe_position(self) -> None:
        self.__stem_controller.validate_probe_position()

    # override from the HardwareSource parent class.
    def data_channel_map_updated(self, data_channel_map: typing.Mapping[str, DataItem.DataItem]) -> None:
        self.__stem_controller._update_scan_channel_map(data_channel_map)

    @property
    def use_hardware_simulator(self) -> bool:
        return False

    def get_property(self, name: str) -> typing.Any:
        return getattr(self, name)

    def set_property(self, name: str, value: typing.Any) -> None:
        setattr(self, name, value)

    def open_configuration_interface(self, api_broker: typing.Any) -> None:
        self.__settings.open_configuration_interface(api_broker)

    def shift_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None:
        frame_parameters = self.__device.current_frame_parameters
        width, height = frame_parameters.pixel_size
        fov_nm = frame_parameters.fov_nm
        pixel_size_nm = fov_nm / max(width, height)
        # calculate dx, dy in meters
        dx = 1e-9 * pixel_size_nm * (mouse_position.x - (camera_shape[1] / 2))
        dy = 1e-9 * pixel_size_nm * (mouse_position.y - (camera_shape[0] / 2))
        logger.info("Shifting (%s,%s) um.\n", -dx * 1e6, -dy * 1e6)
        self.__stem_controller.change_stage_position(dy=dy, dx=dx)

    def increase_pmt(self, channel_index: int) -> None:
        self.__stem_controller.change_pmt_gain(channel_index, factor=2.0)

    def decrease_pmt(self, channel_index: int) -> None:
        self.__stem_controller.change_pmt_gain(channel_index, factor=0.5)

    def get_api(self, version: str) -> typing.Any:
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade:

            def __init__(self) -> None:
                pass

        return CameraFacade()


class ScanFrameDataStream(Acquisition.DataStream):
    def __init__(self, scan_hardware_source: ScanHardwareSource,
                 scan_frame_parameters: ScanFrameParameters, scan_id: uuid.UUID,
                 drift_tracker: typing.Optional[DriftTracker.DriftTracker] = None,
                 camera_exposure_ms: typing.Optional[float] = None,
                 camera_data_stream: typing.Optional[camera_base.CameraFrameDataStream] = None):
        super().__init__()
        scan_frame_parameters = copy.deepcopy(scan_frame_parameters)
        self.__scan_hardware_source = scan_hardware_source
        self.__scan_frame_parameters = scan_frame_parameters
        self.__scan_id = scan_id
        self.__scan_frame_parameters_center_nm = Geometry.FloatPoint.make(scan_frame_parameters.center_nm)
        self.__section_offset = 0
        self.__drift_tracker = drift_tracker
        self.__camera_data_stream = camera_data_stream
        self.__camera_exposure_ms = camera_exposure_ms
        self.__enabled_channels = scan_hardware_source.get_enabled_channels()
        scan_max_area = 2048 * 2048
        subscan_pixel_size = scan_frame_parameters.subscan_pixel_size
        if subscan_pixel_size:
            scan_param_height = subscan_pixel_size.height
            scan_param_width = subscan_pixel_size.width
            if scan_param_height * scan_param_width > scan_max_area:
                scan_param_height = scan_max_area // scan_param_width
            fractional_size = scan_frame_parameters.subscan_fractional_size
            fractional_center = scan_frame_parameters.subscan_fractional_center
            assert fractional_size and fractional_center
            self.__fractional_area = Geometry.FloatRect.from_center_and_size(fractional_center, fractional_size)
            is_subscan = True
        else:
            scan_param_height = scan_frame_parameters.pixel_size.height
            scan_param_width = scan_frame_parameters.pixel_size.width
            if scan_param_height * scan_param_width > scan_max_area:
                scan_param_height = scan_max_area // scan_param_width
            self.__fractional_area = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.5, x=0.5), Geometry.FloatSize(h=1.0, w=1.0))
            is_subscan = False
        self.__scan_size = Geometry.IntSize(h=scan_param_height, w=scan_param_width)
        if is_subscan:
            self.__scan_frame_parameters.subscan_pixel_size = self.__scan_size
        else:
            self.__scan_frame_parameters.pixel_size = self.__scan_size
        self.__section_rect = Geometry.IntRect.from_tlbr(0, 0, 0, 0)
        self.__record_task = typing.cast(HardwareSource.RecordTask, None)

        self.__lock = threading.RLock()
        self.__buffers: typing.Dict[Acquisition.Channel, DataAndMetadata.DataAndMetadata] = dict()
        self.__sent_rows: typing.Dict[Acquisition.Channel, int] = dict()
        self.__available_rows: typing.Dict[Acquisition.Channel, int] = dict()

        self.__started = False

        def update_data(data_channel: HardwareSource.DataChannel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
            # when data arrives here, it will be part of the overall data item, even if it is only a partial
            # acquire of the data item. so the buffer data shape will reflect the overall data item.
            if self.__started:
                with self.__lock:
                    channel_index = data_channel.index
                    if scan_hardware_source.channel_count <= data_channel.index < scan_hardware_source.channel_count * 2:
                        channel_index -= scan_hardware_source.channel_count
                    channel = Acquisition.Channel(scan_hardware_source.hardware_source_id, str(channel_index))
                    # valid_rows will represent the number of valid rows within this section, not within the overall
                    # data. so valid and available rows need to be offset by the section rect top.
                    valid_rows = self.__section_rect.top + data_and_metadata.metadata.get("hardware_source", dict()).get("valid_rows", 0)
                    available_rows = self.__available_rows.get(channel, self.__section_rect.top)
                    if valid_rows > available_rows:
                        if channel not in self.__buffers:
                            self.__buffers[channel] = copy.deepcopy(data_and_metadata)
                        buffer = self.__buffers[channel]
                        assert buffer
                        buffer_data = buffer.data
                        assert buffer_data is not None
                        buffer_data[available_rows:valid_rows] = data_and_metadata[available_rows:valid_rows]
                        self.__available_rows[channel] = valid_rows

        self.__data_channel_listeners = list()
        for data_channel in scan_hardware_source.data_channels:
            self.__data_channel_listeners.append(data_channel.data_channel_updated_event.listen(functools.partial(update_data, data_channel)))

    def about_to_delete(self) -> None:
        if self.__record_task:
            self.__record_task.close()
            self.__record_task = typing.cast(typing.Any, None)
        for data_channel_listener in self.__data_channel_listeners:
            data_channel_listener.close()
        self.__data_channel_listeners.clear()
        super().about_to_delete()

    @property
    def scan_size(self) -> Geometry.IntSize:
        return self.__scan_size

    def get_info(self, channel: Acquisition.Channel) -> Acquisition.DataStreamInfo:
        return Acquisition.DataStreamInfo(DataAndMetadata.DataMetadata(((), numpy.float32)), 0.0)

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return tuple(Acquisition.Channel(self.__scan_hardware_source.hardware_source_id, str(c)) for c in self.__enabled_channels)

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs: typing.Any) -> None:
        self.__scan_hardware_source.abort_playing(sync_timeout=5.0)
        # update the center_nm of scan parameters by adding the accumulated value from drift to the original value.
        # NOTE: this fails if we independently move the center_nm, at which point the original would have to be
        # updated and the accumulated would have to be reset.
        if self.__drift_tracker:
            camera_sequence_overhead = self.__camera_data_stream.camera_sequence_overhead if self.__camera_data_stream else 0.0
            delta_nm = self.__drift_tracker.predict_drift(datetime.datetime.utcnow() + datetime.timedelta(seconds=camera_sequence_overhead))
            # print(f"predicted {delta_nm}")
            self.__scan_frame_parameters.center_nm = self.__scan_frame_parameters_center_nm - delta_nm
        # print(f"scan center_nm={self.__scan_frame_parameters.center_nm}")
        if self.__camera_data_stream:
            assert self.__camera_exposure_ms is not None
            self.__scan_hardware_source.scan_device.prepare_synchronized_scan(self.__scan_frame_parameters, camera_exposure_ms=self.__camera_exposure_ms)

    def _start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        scan_frame_parameters = self.__scan_frame_parameters
        scan_size = self.__scan_size
        # configure the section rect. to make this class work as automatically as possible, the next section
        # will start at the bottom of this section modulo the height of the acquisition. this allows this
        # class to be continually started but still produce the proper output in the view channels.
        section_rect = stream_args.slice_rect
        section_rect = section_rect + Geometry.IntPoint(y=self.__section_offset)
        acquisition_task_parameters = ScanAcquisitionTaskParameters(scan_id=self.__scan_id)
        section_frame_parameters = apply_section_rect(scan_frame_parameters, acquisition_task_parameters, section_rect, scan_size, self.__fractional_area)
        self.__section_rect = section_rect
        self.__section_offset = section_rect.bottom % scan_size.height
        self.__started = True
        with self.__lock:
            self.__buffers.clear()
            self.__sent_rows.clear()
            self.__available_rows.clear()
        acquisition_parameters = HardwareSource.AcquisitionParameters(section_frame_parameters, acquisition_task_parameters)
        self.__record_task = HardwareSource.RecordTask(self.__scan_hardware_source, acquisition_parameters)

    def _finish_stream(self) -> None:
        if self.__record_task:
            self.__record_task.close()
            self.__record_task = typing.cast(typing.Any, None)
        self.__started = False

    def _abort_stream(self) -> None:
        self.__scan_hardware_source.abort_recording()

    @property
    def _progress(self) -> float:
        return sum(self.__sent_rows.values()) / self.__scan_size[0] / len(self.channels)

    def _send_next(self) -> None:
        with self.__lock:
            for channel in self.__buffers.keys():
                sent_rows = self.__sent_rows.get(channel, self.__section_rect.top)
                available_rows = self.__available_rows.get(channel, self.__section_rect.top)
                if sent_rows < available_rows:
                    # when we extract data from the buffer, extract only the part that is the section rect.
                    # the buffer represents the entire data item; but only is updated with the section.
                    scan_data = self.__buffers[channel]
                    is_complete = available_rows == self.__section_rect.bottom
                    # only complete when the record task is finished. this prevents a race condition when restarting.
                    if not is_complete or (is_complete and self.__record_task.is_finished):
                        start = self.__section_rect.width * sent_rows
                        stop = self.__section_rect.width * available_rows
                        data_dtype = scan_data.data_dtype
                        assert data_dtype is not None
                        data_metadata = DataAndMetadata.DataMetadata(((), data_dtype),
                                                                     scan_data.intensity_calibration,
                                                                     (),
                                                                     scan_data.metadata,
                                                                     scan_data.timestamp,
                                                                     DataAndMetadata.DataDescriptor(False, 0, 0),
                                                                     scan_data.timezone,
                                                                     scan_data.timezone_offset)
                        source_slice = (slice(start, stop),)
                        scan_data_data = scan_data.data
                        assert scan_data_data is not None
                        data_stream_event = Acquisition.DataStreamEventArgs(self,
                                                                            channel,
                                                                            data_metadata,
                                                                            scan_data_data.reshape(-1),
                                                                            stop - start,
                                                                            source_slice,
                                                                            Acquisition.DataStreamStateEnum.COMPLETE)
                        if stop - start > 0:
                            self.fire_data_available(data_stream_event)
                            self.__sent_rows[channel] = available_rows


class SynchronizedDataStream(Acquisition.ContainerDataStream):
    def __init__(self, data_stream: Acquisition.DataStream, scan_hardware_source: ScanHardwareSource,
                 camera_hardware_source: camera_base.CameraHardwareSource):
        super().__init__(data_stream)
        self.__scan_hardware_source = scan_hardware_source
        self.__camera_hardware_source = camera_hardware_source
        self.__stem_controller = scan_hardware_source.stem_controller

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs: typing.Any) -> None:
        self.__stem_controller._enter_synchronized_state(self.__scan_hardware_source,
                                                         camera=self.__camera_hardware_source)
        self.__scan_hardware_source.acquisition_state_changed_event.fire(True)
        self.__old_record_parameters = self.__scan_hardware_source.get_record_frame_parameters()
        super()._prepare_stream(stream_args, **kwargs)

    def _finish_stream(self) -> None:
        super()._finish_stream()
        self.__scan_hardware_source.set_record_frame_parameters(self.__old_record_parameters)
        self.__stem_controller._exit_synchronized_state(self.__scan_hardware_source,
                                                        camera=self.__camera_hardware_source)
        self.__scan_hardware_source.acquisition_state_changed_event.fire(False)


def make_synchronized_scan_data_stream(
        scan_hardware_source: ScanHardwareSource,
        scan_frame_parameters: ScanFrameParameters,
        camera_hardware_source: camera_base.CameraHardwareSource,
        camera_frame_parameters: camera_base.CameraFrameParameters,
        camera_data_channel: typing.Optional[camera_base.SynchronizedDataChannelInterface] = None,
        section_height: typing.Optional[int] = None,
        scan_data_stream_functor: typing.Optional[Acquisition.DataStreamFunctor] = None,
        scan_count: int = 1,
        include_raw: bool = True,
        include_summed: bool = False,
        enable_drift_tracker: bool = False,
        old_move_axis: bool = False) -> Acquisition.DataStream:

    # there are two separate drift corrector possibilities:
    #   1 - a drift corrector that takes a separate scan, implemented using the scan_data_stream_functor
    #   2 - a drift corrector that uses the entire result of the scan, implemented by passing enable_drift_corrector = True

    scan_metadata: typing.Dict[str, typing.Any] = dict()
    scan_id = uuid.uuid4()
    update_scan_metadata(scan_metadata, scan_hardware_source.hardware_source_id, scan_hardware_source.display_name,
                         scan_frame_parameters, scan_id, dict())

    instrument_metadata: typing.Dict[str, typing.Any] = dict()
    update_instrument_properties(instrument_metadata, scan_hardware_source.stem_controller,
                                 scan_hardware_source.scan_device)

    camera_exposure_ms = camera_frame_parameters.exposure_ms
    additional_camera_metadata = {"scan": copy.deepcopy(scan_metadata),
                                  "instrument": copy.deepcopy(instrument_metadata)}
    # calculate the flyback pixels by using the scan device. this is fragile because the exposure
    # for a particular section gets calculated here and in ScanFrameDataStream.prepare. if they
    # don't match, the total pixel count for the camera will not match the scan pixels.
    scan_paramaters_copy = copy.deepcopy(scan_frame_parameters)
    scan_hardware_source.scan_device.prepare_synchronized_scan(scan_paramaters_copy, camera_exposure_ms=camera_frame_parameters.exposure_ms)
    flyback_pixels = scan_hardware_source.calculate_flyback_pixels(scan_paramaters_copy)
    camera_data_stream = camera_base.CameraFrameDataStream(camera_hardware_source, camera_frame_parameters,
                                                            camera_base.CameraDeviceSynchronizedStream(camera_hardware_source,
                                                                                           camera_frame_parameters,
                                                                                           flyback_pixels,
                                                                                           additional_camera_metadata))
    processed_camera_data_stream: Acquisition.DataStream = camera_data_stream
    if camera_frame_parameters.processing == "sum_project":
        processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream, operator=Acquisition.SumOperator(axis=0))
    elif camera_frame_parameters.processing == "sum_masked":
        active_masks = camera_frame_parameters.active_masks
        if active_masks:
            operator = Acquisition.StackedDataStreamOperator(
                [Acquisition.MaskedSumOperator(active_mask) for active_mask in active_masks])
            processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream, operator=operator)
        else:
            operator = Acquisition.StackedDataStreamOperator([Acquisition.SumOperator()])
            processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream, operator=operator)
    scan_data_stream = ScanFrameDataStream(scan_hardware_source, scan_frame_parameters, scan_id, scan_hardware_source.drift_tracker, camera_exposure_ms, camera_data_stream)
    scan_size = scan_data_stream.scan_size
    scan_like_data_stream: Acquisition.DataStream = scan_data_stream
    if scan_data_stream_functor:
        scan_like_data_stream = scan_data_stream_functor.apply(scan_like_data_stream)
    combined_data_stream = Acquisition.CombinedDataStream([scan_like_data_stream, processed_camera_data_stream])
    section_height = section_height or scan_size.height
    section_count = (scan_size.height + section_height - 1) // section_height
    collectors: typing.List[Acquisition.CollectedDataStream] = list()
    for section in range(section_count):
        start = section * section_height
        stop = min(start + section_height, scan_size.height)
        collectors.append(Acquisition.CollectedDataStream(combined_data_stream, (stop - start, scan_size.width), get_scan_calibrations(scan_frame_parameters)))
    collector: Acquisition.DataStream = Acquisition.StackedDataStream(collectors)
    if not old_move_axis and camera_frame_parameters.processing == "sum_masked":
        active_masks = camera_frame_parameters.active_masks
        if active_masks and len(active_masks) > 1:
            collector = Acquisition.FramedDataStream(collector, operator=Acquisition.MoveAxisDataStreamOperator(
                processed_camera_data_stream.channels[0]))
    # SynchronizedDataStream saves and restores the scan parameters; also enters/exits synchronized state
    collector = SynchronizedDataStream(collector, scan_hardware_source, camera_hardware_source)
    if scan_count > 1:
        # DriftUpdaterDataStream watches the first channel (HAADF) and sends its frames to the drift compensator
        drift_tracker = scan_hardware_source.drift_tracker
        if drift_tracker and enable_drift_tracker:
            collector = DriftTracker.DriftUpdaterDataStream(collector, drift_tracker, scan_hardware_source.drift_rotation)
        # SequenceDataStream puts all streams in the collector into a sequence
        collector = Acquisition.SequenceDataStream(collector, scan_count)
        assert include_raw or include_summed
        if include_raw and include_summed:
            # AccumulateDataStream sums the successive frames in each channel
            monitor = Acquisition.MonitorDataStream(collector, "raw")
            collector = Acquisition.AccumulatedDataStream(collector)
            collector = Acquisition.CombinedDataStream([collector, monitor])
        elif include_summed:
            collector = Acquisition.AccumulatedDataStream(collector)
        # include_raw is the default behavior
    # the optional ChannelDataStream updates the camera data channel for the stream matching 999
    data_stream: Acquisition.DataStream
    if camera_data_channel:
        data_stream = camera_base.ChannelDataStream(collector, camera_data_channel, Acquisition.Channel(camera_hardware_source.hardware_source_id))
    else:
        data_stream = collector
    # return the top level stream
    return data_stream


class InstrumentController(abc.ABC):

    def apply_metadata_groups(self, properties: typing.MutableMapping[str, typing.Any], metatdata_groups: typing.Sequence[typing.Tuple[typing.Sequence[str], str]]) -> None: pass

    def get_autostem_properties(self) -> typing.Mapping[str, typing.Any]: return dict()

    def handle_shift_click(self, **kwargs: typing.Any) -> None: pass

    def handle_tilt_click(self, **kwargs: typing.Any) -> None: pass


def update_instrument_properties(stem_properties: typing.MutableMapping[str, typing.Any], instrument_controller: stem_controller_module.STEMController, scan_device: typing.Optional[ScanDevice]) -> None:
    if instrument_controller:
        # give the instrument controller opportunity to add properties
        get_autostem_properties_fn = getattr(instrument_controller, "get_autostem_properties", None)
        if callable(get_autostem_properties_fn):
            try:
                autostem_properties = get_autostem_properties_fn()
                stem_properties.update(autostem_properties)
            except Exception as e:
                pass
        # give the instrument controller opportunity to update metadata groups specified by the camera
        acquisition_metatdata_groups = getattr(scan_device, "acquisition_metatdata_groups", None)
        if acquisition_metatdata_groups:
            instrument_controller.apply_metadata_groups(stem_properties, acquisition_metatdata_groups)


def find_stem_controller(stem_controller_id: typing.Optional[str]) -> typing.Optional[stem_controller_module.STEMController]:
    stem_controller: typing.Optional[stem_controller_module.STEMController] = None
    if not stem_controller and stem_controller_id:
        stem_controller = typing.cast(typing.Any, HardwareSource.HardwareSourceManager().get_instrument_by_id(stem_controller_id))
        if not stem_controller:
            stem_controller = Registry.get_component("stem_controller")
    return stem_controller


class ScanModule(typing.Protocol):
    stem_controller_id: str
    device: ScanDevice
    settings: ScanSettingsProtocol
    panel_type: typing.Optional[str] = None  # match a 'scan_panel' registry component
    panel_delegate_type: typing.Optional[str] = None  # currently unused


_component_registered_listener = None
_component_unregistered_listener = None

def run(configuration_location: pathlib.Path) -> None:
    def component_registered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
        stem_controller: typing.Optional[stem_controller_module.STEMController]
        if "scan_module" in component_types:
            scan_module = typing.cast(ScanModule, component)
            stem_controller = find_stem_controller(scan_module.stem_controller_id)
            if not stem_controller:
                print("STEM Controller (" + component.stem_controller_id + ") for (" + component.scan_device_id + ") not found. Using proxy.")
                stem_controller = stem_controller_module.STEMController()
            scan_hardware_source = ConcreteScanHardwareSource(stem_controller, scan_module.device, scan_module.settings, configuration_location, scan_module.panel_type)
            if hasattr(scan_module, "priority"):
                setattr(scan_hardware_source, "priority", getattr(scan_module, "priority"))
            Registry.register_component(scan_hardware_source, {"hardware_source", "scan_hardware_source"})
            HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)
            component.hardware_source = scan_hardware_source

    def component_unregistered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
        if "scan_module" in component_types:
            scan_hardware_source = component.hardware_source
            Registry.unregister_component(scan_hardware_source)
            HardwareSource.HardwareSourceManager().unregister_hardware_source(scan_hardware_source)
            scan_hardware_source.close()

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    # handle any scan modules that have already been registered
    for component in Registry.get_components_by_type("scan_module"):
        component_registered(component, {"scan_module"})


def stop() -> None:
    global _component_registered_listener
    global _component_unregistered_listener
    if _component_registered_listener:
        _component_registered_listener.close()
    if _component_unregistered_listener:
        _component_unregistered_listener.close()
    _component_registered_listener = None
    _component_unregistered_listener = None
