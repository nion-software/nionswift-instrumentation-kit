from __future__ import annotations

# standard libraries
import abc
import asyncio
import copy
import dataclasses
import datetime
import gettext
import json
import logging
import os
import pathlib
import time
import typing
import traceback
import uuid

# typing
# None

# third party libraries
import numpy
import numpy.typing

# local libraries
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import HardwareSource
from nion.swift.model import ImportExportManager
from nion.swift.model import Utility
from nion.swift.model import Graphics
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Process
from nion.utils import Registry

_NDArray = numpy.typing.NDArray[typing.Any]

_ = gettext.gettext


class CameraDevice(typing.Protocol):
    """Camera device.

    Camera devices can support several modes of operation:
        - continue acquire (acquire single images repeatedly for viewing)
        - single acquire (acquire one image)
        - sequence acquire (acquire n continuous images)
        - synchronized acquire (synchronized with scan controller)

    It is recommended to implement all optional methods that are not deprecated. Some methods are deprecated but still
    valid for backward compatibility.

    Properties that should be defined for the camera device:

    camera_id (required, the camera identifier, must be unique. example: 'pci_camera_1')
    camera_name (required, the name of the camera. example: 'Bottom Camera')
    camera_type (required, the camera type. examples: 'eels' or 'ronchigram')
    signal_type (optional, falls back to camera_type if 'eels' or 'ronchigram' otherwise empty)
    has_processed_channel (optional, whether to automatically include a processed (vertical sum) channel of data)
    """

    def close(self) -> None:
        """Close the camera."""
        ...

    @property
    def sensor_dimensions(self) -> typing.Tuple[int, int]:
        """Read-only property for the native sensor size (no binning).

        Returns (height, width) in pixels.

        This is a global property, meaning it affects all profiles, and is assumed to be constant.
        """
        raise NotImplementedError()

    @property
    def readout_area(self) -> typing.Tuple[int, int, int, int]:
        """Return the detector readout area.

        Accepts tuple of (top, left, bottom, right) readout rectangle, specified in sensor coordinates.

        There are restrictions on the valid values, depending on camera. This property should use the closest
        appropriate values, rounding up when necessary.

        This is a global property, meaning it affects all profiles.
        """
        raise NotImplementedError()

    @readout_area.setter
    def readout_area(self, readout_area_TLBR: typing.Tuple[int, int, int, int]) -> None:
        """Set the detector readout area.

        The coordinates, top, left, bottom, right, are specified in sensor coordinates.

        There are restrictions on the valid values, depending on camera. This property should always return
        valid values.

        This is a global property, meaning it affects all profiles.
        """
        ...

    @property
    def flip(self) -> bool:
        """Return whether data is flipped left-right (last dimension).

        This is a global property, meaning it affects all profiles.
        """
        raise NotImplementedError()

    @flip.setter
    def flip(self, do_flip: bool) -> None:
        """Set whether data is flipped left-right (last dimension).

        This is a global property, meaning it affects all profiles.
        """
        ...

    @property
    def is_dark_subtraction_enabled(self) -> bool:
        """Return whether dark subtraction is enabled.

        This is a global property, meaning it affects all profiles.
        """
        return False

    @is_dark_subtraction_enabled.setter
    def is_dark_subtraction_enabled(self, is_dark_subtraction_enabled: bool) -> None:
        """Set whether dark subtraction is enabled.

        This is a global property, meaning it affects all profiles.
        """
        pass

    @property
    def is_dark_subtraction_available(self) -> bool:
        """Return whether dark subtraction is available on this camera.
        """
        return False

    def set_dark_image(self, data: _NDArray, exposure: float, bin: int, t: int, l: int, b: int, r: int) -> bool:
        """Set the dark image for the given exposure, bin, and readout area."""
        return False

    def remove_all_dark_images(self) -> None:
        """Remove all dark reference images."""
        pass

    @property
    def is_gain_normalization_enabled(self) -> bool:
        """Return whether gain normalization is enabled.

        This is a global property, meaning it affects all profiles.
        """
        return False

    @is_gain_normalization_enabled.setter
    def is_gain_normalization_enabled(self, is_gain_normalization_enabled: bool) -> None:
        """Set whether gain normalization is enabled.

        This is a global property, meaning it affects all profiles.
        """
        pass

    @property
    def is_gain_normalization_available(self) -> bool:
        """Return whether gain normalization is available on this camera.
        """
        return False

    def set_gain_image(self, data: _NDArray, voltage: int, bin: int) -> bool:
        """Set the gain image for the given voltage and binning."""
        return False

    def remove_all_gain_images(self) -> None:
        """Remove all gain reference images."""
        pass

    @property
    def binning_values(self) -> typing.List[int]:
        """Return a list of valid binning values (int's).

        This is a global property, meaning it affects all profiles, and is assumed to be constant.
        """
        raise NotImplementedError()

    def get_expected_dimensions(self, binning: int) -> typing.Tuple[int, int]:
        """Read-only property for the expected image size (binning and readout area included).

        Returns (height, width).

        Cameras are allowed to bin in one dimension and not the other.
        """
        ...

    def set_frame_parameters(self, frame_parameters: typing.Any) -> None:
        """Set the pending frame parameters (exposure_ms, binning, processing, integration_count).

        processing and integration_count are optional, in which case they are handled at a higher level.
        """
        ...

    @property
    def calibration_controls(self) -> typing.Mapping[str, typing.Union[str, int, float]]:
        """Return lookup dict of calibration controls to be read from instrument controller for this device.

        The dict should have keys of the form <axis>_<field>_<type> where <axis> is "x", "y", "z", or "intensity",
        <field> is "scale", "offset", or "units", and <type> is "control" or "value". If <type> is "control", then
        the value for that axis/field will use the value of that key to read the calibration field value from the
        instrument controller. If <type> is "value", then the calibration field value will be the value of that key.

        For example, the dict with the following keys will read x_scale and x_offset from the instrument controller
        values "cam_scale' and "cam_offset", but supply the units directly as "nm".

        { "x_scale_control": "cam_scale", "x_offset_control": "cam_offset", "x_units_value": "nm" }

        The dict can contain the key <axis>_origin_override with the value "center" to indicate that the origin is
        at the center of the data for that axis.

        { "x_scale_control": "cam_scale",
          "x_offset_control": "cam_offset",
          "x_units_value": "nm",
          "x_origin_override": "center" }

        In addition to the calibration controls, a "counts_per_electron" control or value can also be specified.

        { "counts_per_electron_control": "Camera1_CountsPerElectron" }
        """
        return dict()

    @property
    def acquisition_metatdata_groups(self) -> typing.Sequence[typing.Tuple[typing.Sequence[str], str]]:
        """Return metadata groups to be read from instrument controller and stored in metadata.

        Metadata groups are a list of tuples where the first item is the destination path and the second item is the
        control group name.

        This method is optional. Default is to return an empty list.

        Note: for backward compatibility, if the destination root is 'autostem', it is skipped and the rest of the path
        is used. this can be removed in the future.
        """
        return list()

    def start_live(self) -> None:
        """Start live acquisition. Required before using acquire_image."""
        ...

    def stop_live(self) -> None:
        """Stop live acquisition."""
        ...

    def acquire_image(self) -> ImportExportManager.DataElementType:
        """Acquire the most recent image and return a data element dict.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict. Inside the 'properties' dict you must include 'frame_number' as an int.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).

        If integration_count is non-zero and is handled directly in this method, the 'properties' should also contain
        a 'integration_count' value to indicate how many frames were integrated. If the value is missing, a default
        value of 1 is assumed.

        Calibrations can be directly by including a 'calibration_controls' dictionary that follows the guidelines
        described under that property, by including keys 'intensity calibration' and 'spatial_calibrations', or by
        doing nothing and falling back to the having the 'calibration_controls' property define the calibrations.

        Use the 'calibration_controls' dictionary when the calibrations are read from the instrument and dependent on
        the camera state. Use the 'calibration_controls' property when read from the instrument but independent of
        camera state. Use 'intensity_calibration' and 'spatial_calibrations' when the calibrations are only dependent
        on the camera state but not the instrument.

        If the calibrations are specified with the 'intensity calibration' and 'spatial_calibrations' keys, the
        `intensity_calibration` should be a dict with `offset`, `scale`, and `units` key and the `spatial_calibrations`
        should be a list of dicts, one for each dimension, with the same keys.

        Specifying calibrations directly using 'intensity_calibration' and 'spatial_calibrations' take precedence over
        supplying `calibration_controls` in the data_element. And that takes precedence over using the
        'calibration_controls' property.
        """
        ...

    def get_acquire_sequence_metrics(self, frame_parameters: CameraFrameParameters) -> typing.Mapping[str, typing.Any]:
        """Return the acquire sequence metrics for the frame parameters dict.

        The frame parameters will contain extra keys 'acquisition_frame_count' and 'storage_frame_count' to indicate
        the number of frames in the sequence.

        The frame parameters will contain a key 'processing' set to 'sum_project' if 1D summing or binning
        is requested.

        The dictionary returned should include keys for 'acquisition_time' (in seconds), 'storage_memory' (in bytes) and
         'acquisition_memory' (in bytes). The default values will be the exposure time times the acquisition frame
         count and the camera readout size times the number of frames.
        """
        return dict()

    def acquire_sequence_begin(self, camera_frame_parameters: CameraFrameParameters, count: int, **kwargs: typing.Any) -> PartialData:
        """Begin sequence acquire.

        The camera device can allocate memory to accommodate the count and begin acquisition immediately.

        The camera device will typically populate the PartialData with the data array (xdata), is_complete set to
        False, is_canceled set to False, and valid_rows and valid_count set to 0.

        Returns PartialData.
        """
        pass

    def acquire_sequence_continue(self, *, update_period: float = 1.0, **kwargs: typing.Any) -> PartialData:
        """Continue sequence acquire.

        The camera device should wait up to update_period seconds for data and populate PartialData with data and
        information about the acquisition.

        The valid_count field of PartialData indicates how many items are valid in xdata. The grab_synchronized method
        will keep track of the last valid item and copy data from the last valid item to valid_count into the acquisition
        data and then update last valid item with valid_count.

        The xdata field of PartialData must be filled with the data allocated during acquire_synchronized_begin. The
        section of data up to valid_rows must remain valid until the last Python reference to xdata is released.

        Returns PartialData.
        """
        pass

    def acquire_sequence_end(self, **kwargs: typing.Any) -> None:
        """Clean up sequence acquire.

        The camera device can clean up anything internal that was required for acquisition.

        The memory returned during acquire_sequence_begin or acquire_sequence_continue must remain valid until
        the last Python reference to that memory is released.
        """

    def acquire_sequence_cancel(self) -> None:
        """Request to cancel a sequence acquisition.

        Pending acquire_sequence calls should return None to indicate cancellation.
        """
        pass

    def show_config_window(self) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def show_configuration_dialog(self, api_broker: typing.Any) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def start_monitor(self) -> None:
        """Show a monitor dialog, if needed. Dialog can be modal or modeless."""
        pass


class CameraDevice3(typing.Protocol):
    """Camera device.

    Camera devices can support several modes of operation:
        - continue acquire (acquire single images repeatedly for viewing)
        - single acquire (acquire one image)
        - sequence acquire (acquire n continuous images)
        - synchronized acquire (synchronized with scan controller)

    It is recommended to implement all optional methods that are not deprecated. Some methods are deprecated but still
    valid for backward compatibility.

    Properties that should be defined for the camera device:

    camera_id (required, the camera identifier, must be unique. example: 'pci_camera_1')
    camera_name (required, the name of the camera. example: 'Bottom Camera')
    camera_type (required, the camera type. examples: 'eels' or 'ronchigram')
    signal_type (optional, falls back to camera_type if 'eels' or 'ronchigram' otherwise empty)
    has_processed_channel (optional, whether to automatically include a processed (vertical sum) channel of data)
    """

    def close(self) -> None:
        """Close the camera."""
        ...

    @property
    def camera_version(self) -> int:
        return 3

    @property
    def sensor_dimensions(self) -> typing.Tuple[int, int]:
        """Read-only property for the native sensor size (no binning).

        Returns (height, width) in pixels.

        This is a global property, meaning it affects all profiles, and is assumed to be constant.
        """
        raise NotImplementedError()

    @property
    def exposure_precision(self) -> int:
        """Read-only property for the exposure precision, expressed as a negative integer exponent."""
        return -3  # ms

    @property
    def readout_area(self) -> typing.Tuple[int, int, int, int]:
        """Return the detector readout area.

        Accepts tuple of (top, left, bottom, right) readout rectangle, specified in sensor coordinates.

        There are restrictions on the valid values, depending on camera. This property should use the closest
        appropriate values, rounding up when necessary.

        This is a global property, meaning it affects all profiles.
        """
        raise NotImplementedError()

    @readout_area.setter
    def readout_area(self, readout_area_TLBR: typing.Tuple[int, int, int, int]) -> None:
        """Set the detector readout area.

        The coordinates, top, left, bottom, right, are specified in sensor coordinates.

        There are restrictions on the valid values, depending on camera. This property should always return
        valid values.

        This is a global property, meaning it affects all profiles.
        """
        ...

    @property
    def flip(self) -> bool:
        """Return whether data is flipped left-right (last dimension).

        This is a global property, meaning it affects all profiles.
        """
        raise NotImplementedError()

    @flip.setter
    def flip(self, do_flip: bool) -> None:
        """Set whether data is flipped left-right (last dimension).

        This is a global property, meaning it affects all profiles.
        """
        ...

    @property
    def is_dark_subtraction_enabled(self) -> bool:
        """Return whether dark subtraction is enabled.

        This is a global property, meaning it affects all profiles.
        """
        return False

    @is_dark_subtraction_enabled.setter
    def is_dark_subtraction_enabled(self, is_dark_subtraction_enabled: bool) -> None:
        """Set whether dark subtraction is enabled.

        This is a global property, meaning it affects all profiles.
        """
        pass

    @property
    def is_dark_subtraction_available(self) -> bool:
        """Return whether dark subtraction is available on this camera.
        """
        return False

    def set_dark_image(self, data: _NDArray, exposure: float, bin: int, t: int, l: int, b: int, r: int) -> bool:
        """Set the dark image for the given exposure, bin, and readout area."""
        return False

    def remove_all_dark_images(self) -> None:
        """Remove all dark reference images."""
        pass

    @property
    def is_gain_normalization_enabled(self) -> bool:
        """Return whether gain normalization is enabled.

        This is a global property, meaning it affects all profiles.
        """
        return False

    @is_gain_normalization_enabled.setter
    def is_gain_normalization_enabled(self, is_gain_normalization_enabled: bool) -> None:
        """Set whether gain normalization is enabled.

        This is a global property, meaning it affects all profiles.
        """
        pass

    @property
    def is_gain_normalization_available(self) -> bool:
        """Return whether gain normalization is available on this camera.
        """
        return False

    def set_gain_image(self, data: _NDArray, voltage: int, bin: int) -> bool:
        """Set the gain image for the given voltage and binning."""
        return False

    def remove_all_gain_images(self) -> None:
        """Remove all gain reference images."""
        pass

    @property
    def binning_values(self) -> typing.List[int]:
        """Return a list of valid binning values (int's).

        This is a global property, meaning it affects all profiles, and is assumed to be constant.
        """
        raise NotImplementedError()

    def get_expected_dimensions(self, binning: int) -> typing.Tuple[int, int]:
        """Read-only property for the expected image size (binning and readout area included).

        Returns (height, width).

        Cameras are allowed to bin in one dimension and not the other.
        """
        ...

    def validate_frame_parameters(self, frame_parameters: CameraFrameParameters) -> CameraFrameParameters:
        """Validate the frame parameters.

        The returned frame parameters should be a copy of the input values with modifications so that the
        requested values are within range and precision.
        """
        return CameraFrameParameters(frame_parameters.as_dict())

    def set_frame_parameters(self, frame_parameters: typing.Any) -> None:
        """Set the pending frame parameters (exposure_ms, binning, processing, integration_count).

        The parameters may be out of range or not precise; the device is free to use the closest value.

        The caller should call validate_frame_parameters if worried about range or precision.

        processing and integration_count are optional, in which case they are handled at a higher level.
        """
        ...

    @property
    def calibration_controls(self) -> typing.Mapping[str, typing.Union[str, int, float]]:
        """Return lookup dict of calibration controls to be read from instrument controller for this device.

        Only used if a camera calibrator is not supplied.

        The dict should have keys of the form <axis>_<field>_<type> where <axis> is "x", "y", "z", or "intensity",
        <field> is "scale", "offset", or "units", and <type> is "control" or "value". If <type> is "control", then
        the value for that axis/field will use the value of that key to read the calibration field value from the
        instrument controller. If <type> is "value", then the calibration field value will be the value of that key.

        For example, the dict with the following keys will read x_scale and x_offset from the instrument controller
        values "cam_scale' and "cam_offset", but supply the units directly as "nm".

        { "x_scale_control": "cam_scale", "x_offset_control": "cam_offset", "x_units_value": "nm" }

        The dict can contain the key <axis>_origin_override with the value "center" to indicate that the origin is
        at the center of the data for that axis.

        { "x_scale_control": "cam_scale",
          "x_offset_control": "cam_offset",
          "x_units_value": "nm",
          "x_origin_override": "center" }

        In addition to the calibration controls, a "counts_per_electron" control or value can also be specified.

        { "counts_per_electron_control": "Camera1_CountsPerElectron" }
        """
        return dict()

    def get_camera_calibrator(self, *, instrument_controller: InstrumentController, **kwargs: typing.Any) -> typing.Optional[CameraCalibrator]:
        """Return a camera calibrator object.

        The default is to return a camera calibrator based on camera controls.
        """
        return CalibrationControlsCalibrator(instrument_controller, self)

    @property
    def acquisition_metatdata_groups(self) -> typing.Sequence[typing.Tuple[typing.Sequence[str], str]]:
        """Return metadata groups to be read from instrument controller and stored in metadata.

        Metadata groups are a list of tuples where the first item is the destination path and the second item is the
        control group name.

        This method is optional. Default is to return an empty list.

        Note: for backward compatibility, if the destination root is 'autostem', it is skipped and the rest of the path
        is used. this can be removed in the future.
        """
        return list()

    def start_live(self) -> None:
        """Start live acquisition. Required before using acquire_image."""
        ...

    def stop_live(self) -> None:
        """Stop live acquisition."""
        ...

    def acquire_image(self) -> ImportExportManager.DataElementType:
        """Acquire the most recent image and return a data element dict.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict. Inside the 'properties' dict you must include 'frame_number' as an int.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).

        If integration_count is non-zero and is handled directly in this method, the 'properties' should also contain
        a 'integration_count' value to indicate how many frames were integrated. If the value is missing, a default
        value of 1 is assumed.

        Calibrations can be directly by including a 'calibration_controls' dictionary that follows the guidelines
        described under that property, by including keys 'intensity calibration' and 'spatial_calibrations', or by
        doing nothing and falling back to the having the 'calibration_controls' property define the calibrations.

        Use the 'calibration_controls' dictionary when the calibrations are read from the instrument and dependent on
        the camera state. Use the 'calibration_controls' property when read from the instrument but independent of
        camera state. Use 'intensity_calibration' and 'spatial_calibrations' when the calibrations are only dependent
        on the camera state but not the instrument.

        If the calibrations are specified with the 'intensity calibration' and 'spatial_calibrations' keys, the
        `intensity_calibration` should be a dict with `offset`, `scale`, and `units` key and the `spatial_calibrations`
        should be a list of dicts, one for each dimension, with the same keys.

        Specifying calibrations directly using 'intensity_calibration' and 'spatial_calibrations' take precedence over
        supplying `calibration_controls` in the data_element. And that takes precedence over using the
        'calibration_controls' property.
        """
        ...

    def get_acquire_sequence_metrics(self, frame_parameters: CameraFrameParameters) -> typing.Mapping[str, typing.Any]:
        """Return the acquire sequence metrics for the frame parameters dict.

        The frame parameters will contain extra keys 'acquisition_frame_count' and 'storage_frame_count' to indicate
        the number of frames in the sequence.

        The frame parameters will contain a key 'processing' set to 'sum_project' if 1D summing or binning
        is requested.

        The dictionary returned should include keys for 'acquisition_time' (in seconds), 'storage_memory' (in bytes) and
         'acquisition_memory' (in bytes). The default values will be the exposure time times the acquisition frame
         count and the camera readout size times the number of frames.
        """
        return dict()

    def acquire_synchronized_begin(self, camera_frame_parameters: CameraFrameParameters, collection_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> PartialData:
        """Begin synchronized acquire.

        The camera device can allocate memory to accommodate the collection_shape and begin acquisition immediately.

        The camera device will typically populate the PartialData with the data array (xdata), is_complete set to
        False, is_canceled set to False, and valid_rows and valid_count set to 0.

        Returns PartialData.
        """
        ...

    def acquire_synchronized_continue(self, *, update_period: float = 1.0, **kwargs: typing.Any) -> PartialData:
        """Continue synchronized acquire.

        The camera device should wait up to update_period seconds for data and populate PartialData with data and
        information about the acquisition.

        Deprecated: The valid_rows field of PartialData indicates how many rows are valid in xdata. The
        grab_synchronized method will keep track of the last valid row and copy data from the last valid row to
        valid_rows into the acquisition data and then update last valid row with valid_rows.

        The valid_count field of PartialData indicates how many items are valid in xdata. The grab_synchronized method
        will keep track of the last valid item and copy data from the last valid item to valid_count into the acquisition
        data and then update last valid item with valid_count.

        The xdata field of PartialData must be filled with the data allocated during acquire_synchronized_begin. The
        section of data up to valid_rows must remain valid until the last Python reference to xdata is released.

        Returns PartialData.
        """
        ...

    def acquire_synchronized_end(self, **kwargs: typing.Any) -> None:
        """Clean up synchronized acquire.

        The camera device can clean up anything internal that was required for acquisition.

        The memory returned during acquire_synchronized_begin or acquire_synchronized_continue must remain valid until
        the last Python reference to that memory is released.
        """
        ...

    def acquire_synchronized_cancel(self) -> None:
        """Request to cancel a synchronized acquisition.

        Future calls to acquire_synchronized_continue should have the is_cancelled flag set.
        """
        pass

    def acquire_sequence_begin(self, camera_frame_parameters: CameraFrameParameters, count: int, **kwargs: typing.Any) -> PartialData:
        """Begin sequence acquire.

        The camera device can allocate memory to accommodate the count and begin acquisition immediately.

        The camera device will typically populate the PartialData with the data array (xdata), is_complete set to
        False, is_canceled set to False, and valid_rows and valid_count set to 0.

        Returns PartialData.
        """
        pass

    def acquire_sequence_continue(self, *, update_period: float = 1.0, **kwargs: typing.Any) -> PartialData:
        """Continue sequence acquire.

        The camera device should wait up to update_period seconds for data and populate PartialData with data and
        information about the acquisition.

        The valid_count field of PartialData indicates how many items are valid in xdata. The grab_synchronized method
        will keep track of the last valid item and copy data from the last valid item to valid_count into the acquisition
        data and then update last valid item with valid_count.

        The xdata field of PartialData must be filled with the data allocated during acquire_synchronized_begin. The
        section of data up to valid_rows must remain valid until the last Python reference to xdata is released.

        Returns PartialData.
        """
        pass

    def acquire_sequence_end(self, **kwargs: typing.Any) -> None:
        """Clean up sequence acquire.

        The camera device can clean up anything internal that was required for acquisition.

        The memory returned during acquire_sequence_begin or acquire_sequence_continue must remain valid until
        the last Python reference to that memory is released.
        """

    def acquire_sequence_cancel(self) -> None:
        """Request to cancel a sequence acquisition.

        Future calls to acquire_sequence_continue should have the is_cancelled flag set.
        """
        pass

    def show_config_window(self) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def show_configuration_dialog(self, api_broker: typing.Any) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def start_monitor(self) -> None:
        """Show a monitor dialog, if needed. Dialog can be modal or modeless."""
        pass


class InstrumentController(abc.ABC):

    @abc.abstractmethod
    def TryGetVal(self, s: str) -> typing.Tuple[bool, typing.Optional[float]]: ...

    @abc.abstractmethod
    def get_value(self, value_id: str, default_value: typing.Optional[float] = None) -> typing.Optional[float]: ...

    @abc.abstractmethod
    def set_value(self, value_id: str, value: float) -> None: ...

    def get_autostem_properties(self) -> typing.Mapping[str, typing.Any]: return dict()

    def apply_metadata_groups(self, properties: typing.MutableMapping[str, typing.Any], metatdata_groups: typing.Sequence[typing.Tuple[typing.Sequence[str], str]]) -> None: pass

    def handle_shift_click(self, **kwargs: typing.Any) -> None: pass

    def handle_tilt_click(self, **kwargs: typing.Any) -> None: pass


class AcquisitionData:
    def __init__(self, data_element: typing.Optional[ImportExportManager.DataElementType] = None) -> None:
        self.__data_element: ImportExportManager.DataElementType = data_element or dict()

    @property
    def data_element(self) -> ImportExportManager.DataElementType:
        return self.__data_element

    @property
    def is_signal_calibrated(self) -> bool:
        return "spatial_calibrations" in self.__data_element

    @property
    def signal_calibrations(self) -> DataAndMetadata.CalibrationListType:
        return tuple(Calibration.Calibration.from_rpc_dict(d) or Calibration.Calibration() for d in self.__data_element.get("spatial_calibrations", list()))

    @signal_calibrations.setter
    def signal_calibrations(self, calibrations: DataAndMetadata.CalibrationListType) -> None:
        self.__data_element["spatial_calibrations"] = [calibration.rpc_dict for calibration in calibrations]

    def apply_signal_calibrations(self, calibrations: DataAndMetadata.CalibrationListType) -> None:
        if not self.is_signal_calibrated:
            self.signal_calibrations = calibrations

    @property
    def is_intensity_calibrated(self) -> bool:
        return "intensity_calibration" in self.__data_element

    @property
    def intensity_calibration(self) -> Calibration.Calibration:
        return Calibration.Calibration.from_rpc_dict(self.__data_element.get("intensity_calibration", dict())) or Calibration.Calibration()

    @intensity_calibration.setter
    def intensity_calibration(self, calibration: Calibration.Calibration) -> None:
        self.__data_element["intensity_calibration"] = calibration.rpc_dict

    def apply_intensity_calibration(self, calibration: Calibration.Calibration) -> None:
        if not self.is_intensity_calibrated:
            self.intensity_calibration = calibration

    @property
    def counts_per_electron(self) -> typing.Optional[float]:
        return typing.cast(typing.Optional[float], self.__data_element.get("metadata", dict()).get("hardware_source", dict()).get("counts_per_electron"))

    @counts_per_electron.setter
    def counts_per_electron(self, counts_per_electron: typing.Optional[float]) -> None:
        if counts_per_electron:
            self.__data_element.setdefault("metadata", dict()).setdefault("hardware_source", dict())["counts_per_electron"] = counts_per_electron
        else:
            self.__data_element.get("metadata", dict()).get("hardware_source", dict()).pop("counts_per_electron", None)


class CameraAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, instrument_controller: InstrumentController, camera_hardware_source: CameraHardwareSource, is_continuous: bool,
                 camera_category: str, signal_type: typing.Optional[str], frame_parameters: CameraFrameParameters) -> None:
        super().__init__(is_continuous)
        self.__instrument_controller = instrument_controller
        self.hardware_source_id = camera_hardware_source.hardware_source_id
        self.is_continuous = is_continuous
        self.__camera = camera_hardware_source.camera
        self.__camera_settings = camera_hardware_source.camera_settings
        self.__camera_hardware_source = camera_hardware_source
        self.__camera_category = camera_category
        self.__signal_type = signal_type
        self.__display_name = camera_hardware_source.display_name
        self.__frame_parameters: typing.Optional[CameraFrameParameters] = None
        self.__pending_frame_parameters: typing.Optional[CameraFrameParameters] = CameraFrameParameters(frame_parameters.as_dict())

    def set_frame_parameters(self, frame_parameters: CameraFrameParameters) -> None:
        self.__pending_frame_parameters = CameraFrameParameters(frame_parameters.as_dict())
        self.__activate_frame_parameters()

    @property
    def frame_parameters(self) -> typing.Optional[CameraFrameParameters]:
        if self.__pending_frame_parameters:
            return self.__pending_frame_parameters
        return self.__frame_parameters

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

    def _acquire_data_elements(self) -> typing.List[typing.Dict[str, typing.Any]]:
        if self.__pending_frame_parameters:
            self.__activate_frame_parameters()
        assert self.__frame_parameters is not None
        frame_parameters = self.__frame_parameters
        binning = frame_parameters.binning
        integration_count = frame_parameters.integration_count if frame_parameters.integration_count else 1
        cumulative_frame_count = 0  # used for integration_count
        cumulative_data: typing.Optional[_NDArray] = None
        _data_element: typing.Dict[str, typing.Any] = dict()  # avoid use-before-set warning
        had_grace_frame = False  # whether grace frame has been used up (allows for extra frame during accumulation startup)
        while cumulative_frame_count < integration_count:
            _data_element = self.__camera.acquire_image()
            frames_acquired = _data_element["properties"].get("integration_count", 1)
            if cumulative_data is None:
                cumulative_data = _data_element["data"]
                cumulative_frame_count += frames_acquired
            else:
                # if the cumulative shape does not match in size, assume it is an acquisition steady state problem
                # and start again with the newer frame. only allow this to occur once.
                if cumulative_data.shape != _data_element["data"].shape:
                    assert not had_grace_frame
                    cumulative_data = _data_element["data"]
                    had_grace_frame = True
                else:
                    cumulative_data += _data_element["data"]
                    cumulative_frame_count += frames_acquired
            assert cumulative_frame_count <= integration_count
        assert cumulative_data is not None
        if self.__stop_after_acquire:
            self.__camera.stop_live()
        # camera data is always assumed to be full frame, otherwise deal with subarea 1d and 2d
        return [self.__camera_hardware_source.make_live_data_element(cumulative_data, _data_element["properties"], _data_element.get("timestamp", datetime.datetime.utcnow()), frame_parameters, cumulative_frame_count)]

    def __activate_frame_parameters(self) -> None:
        self.__frame_parameters = self.frame_parameters
        self.__pending_frame_parameters = None
        self.__camera.set_frame_parameters(self.__frame_parameters)


class Mask:
    def __init__(self) -> None:
        self._layers: typing.List[typing.Dict[str, typing.Any]] = list()
        self.name: typing.Optional[str] = None
        self.uuid = uuid.uuid4()

    def add_layer(self, graphic: Graphics.Graphic, value: typing.Union[float, _NDArray], inverted: bool = False) -> None:
        if isinstance(value, numpy.ndarray):
            value = value.tolist()
        self._layers.append({"value": value, "inverted": inverted, "graphic_dict": graphic.mime_data_dict()})

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {"name": self.name, "uuid": str(self.uuid), "layers": self._layers}

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        return self.to_dict()

    @classmethod
    def from_dict(cls, mask_description: typing.Mapping[str, typing.Any]) -> "Mask":
        mask = cls()
        mask.name = mask_description["name"]
        if "uuid" in mask_description:
            mask.uuid = uuid.UUID(mask_description["uuid"])
        mask._layers = mask_description["layers"]
        return mask

    def get_mask_array(self, data_shape: typing.Sequence[int]) -> _NDArray:
        if len(self._layers) == 0:
            return numpy.ones(data_shape)
        mask = numpy.zeros(data_shape)
        for layer in self._layers:
            graphic_dict = typing.cast(typing.Dict[str, typing.Any], layer["graphic_dict"])
            value = layer["value"]
            inverted = layer.get("inverted", False)

            def graphic_type_lookup(t: str) -> str:
                return typing.cast(str, graphic_dict["type"])

            graphic = Graphics.factory(graphic_type_lookup)
            graphic.read_from_mime_data(graphic_dict)
            if graphic:
                part_mask = graphic.get_mask(tuple(data_shape)).astype(bool)  # type: ignore
                if inverted:
                    part_mask = numpy.logical_not(part_mask)
                if isinstance(value, (float, int)):
                    mask[part_mask] = value
                else:
                    # For backwards compatibility accept "grad_x" and "grad_y" and convert to new general system
                    if value == "grad_x":
                        value = [[0, 0], [1, 0]]
                    elif value == "grad_y":
                        value = [[0, 1], [0, 0]]
                    if hasattr(graphic, "center"):
                        center = getattr(graphic, "center")
                    else:
                        center = (0.5, 0.5)
                    center_coords = (center[0] * data_shape[0], center[1] * data_shape[1])
                    y, x = numpy.mgrid[:data_shape[0], :data_shape[1]].astype(numpy.float32)
                    y -= center_coords[0]
                    x -= center_coords[1]
                    poly = numpy.polynomial.polynomial.polyval2d(x, y, value)  # type: ignore
                    mask[part_mask] = poly[part_mask]
        return mask

    def copy(self) -> Mask:
        return Mask.from_dict(copy.deepcopy(self.to_dict()))


class CameraSettings:
    """Document and define types for camera settings.

    IMPORTANT NOTE: Used for typing. Not intended to serve as a base class.

    The camera settings object facilitates persistence and tracking of configuration and frame parameters for the
    camera. When used with the standard UI, it is only accessed through the CameraHardwareSource and not used directly.
    However, when used with a custom UI, it may be accessed directly.

    Configuration parameters are settings that apply to the camera as a whole, as opposed to settings for a specific
    acquisition sequence.

    Frame parameters are settings that apply to a specific frame acquisition sequence.

    The current frame parameters refer to the frame parameters being used for the current acquisition (if running) or
    pending acquisition (if stopped).

    For backwards compatibility, the record frame parameters are a special set of parameters used for
    higher quality data acquisition (recording).

    To facilitate the user being able to switch between frame parameter settings quickly, sets of frame parameters
    called profiles and the current selected profile can be tracked. The standard UI supports this capability but custom
    UIs may choose not to support this.

    The manner in which a change to the current frame parameters is propagated to the frame parameters associated with
    the current profile is implementation dependent. The suggested behavior is to apply user initiated changes in the
    current frame parameters to the frame parameters associated with the current profile.

    For backwards compatibility, the profiles may also be referred to by named modes. Up to now, exactly three modes
    have been supported: Run (0), Tune (1), and Snap (2), with the mode name and profile index listed in parenthesis.

    When `set_current_frame_parameters` is called, it should fire the `current_frame_parameters_changed_event` with
    the frame parameters as the only parameter; this will result in a `set_frame_parameters` call to the camera device.

    TODO: write about threading (events must be triggered on main thread)
    """

    def __init__(self) -> None:
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

        # the list of possible modes should be defined here
        self.modes = [str()]

    def close(self) -> None:
        pass

    def initialize(self, configuration_location: typing.Optional[pathlib.Path] = None, event_loop: typing.Optional[asyncio.AbstractEventLoop] = None, **kwargs: typing.Any) -> None:
        pass

    def apply_settings(self, settings_dict: typing.Mapping[str, typing.Any]) -> None:
        """Initialize the settings with the settings_dict."""
        pass

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> CameraFrameParameters:
        """Return camera frame parameters from dict.

        Subclasses should no longer override this method as it is required to return CameraFrameParameters
        instead of another frame parameters class.
        """
        return CameraFrameParameters(d)

    def set_current_frame_parameters(self, frame_parameters: CameraFrameParameters) -> None:
        """Set the current frame parameters.

        Fire the current frame parameters changed event and optionally the settings changed event.
        """
        self.current_frame_parameters_changed_event.fire(frame_parameters)

    def get_current_frame_parameters(self) -> typing.Optional[CameraFrameParameters]:
        """Get the current frame parameters."""
        return None

    def set_record_frame_parameters(self, frame_parameters: CameraFrameParameters) -> None:
        """Set the record frame parameters.

        Fire the record frame parameters changed event and optionally the settings changed event.
        """
        self.record_frame_parameters_changed_event.fire(frame_parameters)

    def get_record_frame_parameters(self) -> typing.Optional[CameraFrameParameters]:
        """Get the record frame parameters."""
        return None

    def set_frame_parameters(self, profile_index: int, frame_parameters: CameraFrameParameters) -> None:
        """Set the frame parameters with the settings index and fire the frame parameters changed event.

        If the settings index matches the current settings index, call set current frame parameters.

        If the settings index matches the record settings index, call set record frame parameters.
        """
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def get_frame_parameters(self, profile_index: int) -> CameraFrameParameters:
        """Get the frame parameters for the settings index."""
        return CameraFrameParameters()

    def set_selected_profile_index(self, profile_index: int) -> None:
        """Set the current settings index.

        Call set current frame parameters if it changed.

        Fire profile changed event if it changed.
        """
        pass

    @property
    def selected_profile_index(self) -> int:
        """Return the current settings index."""
        return 0

#    @property
#    def masks(self) -> typing.Sequence[Mask]:
        """
        If the camera supports masked summing for synchronized acquisition, this attribute is used to access the masks
        provided by the camera. Whether this attribute exists or not is also used to decide if the camera supports
        masked summing.
        Note that the active masks which will be used in the acquisition are defined in the CameraFrameParameters object.
        """

    def get_mode(self) -> str:
        """Return the current mode (named version of current settings index)."""
        return str()

    def set_mode(self, mode: str) -> None:
        """Set the current mode (named version of current settings index)."""
        pass

    def open_configuration_interface(self, api_broker: typing.Any) -> None:
        pass


class PartialData:
    """Represents data returned from acquisition.

    The xdata should be the entire data array.

    valid_count gives the number of valid pixels.

    valid_rows is deprecated and does not need to be supplied if valid_count is supplied.

    is_complete and is_canceled should be set as required.
    """
    def __init__(self, xdata: DataAndMetadata.DataAndMetadata, is_complete: bool, is_canceled: bool,
                 valid_rows: typing.Optional[int] = None, valid_count: typing.Optional[int] = None) -> None:
        self.xdata = xdata
        self.is_complete = is_complete
        self.is_canceled = is_canceled
        self.valid_rows = valid_rows
        self.valid_count = valid_count


@typing.runtime_checkable
class CameraHardwareSource(HardwareSource.HardwareSource, typing.Protocol):
    """Define the camera hardware source protocol.

     This protocol is not intended to be implemented outside of the instrumentation kit.

     The public methods are intended to be stable as much as possible. When
     """

    # protected methods

    def get_expected_dimensions(self, binning: int) -> typing.Tuple[int, int]: ...
    def get_signal_name(self, camera_frame_parameters: CameraFrameParameters) -> str: ...
    def grab_next_to_start(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]: ...
    def grab_next_to_finish(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]: ...
    def grab_sequence_prepare(self, count: int, **kwargs: typing.Any) -> bool: ...
    def grab_sequence(self, count: int, **kwargs: typing.Any) -> typing.Optional[typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]]: ...
    def acquire_synchronized_begin(self, camera_frame_parameters: CameraFrameParameters, collection_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> PartialData: ...
    def acquire_synchronized_continue(self, *, update_period: float = 1.0) -> PartialData: ...
    def acquire_synchronized_end(self) -> None: ...
    def acquire_synchronized_prepare(self, data_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> None: ...
    def acquire_synchronized(self, data_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> typing.Sequence[ImportExportManager.DataElementType]: ...
    def acquire_sequence_prepare(self, n: int) -> None: ...
    def acquire_sequence(self, n: int) -> typing.Sequence[ImportExportManager.DataElementType]: ...
    def acquire_sequence_begin(self, camera_frame_parameters: CameraFrameParameters, count: int, **kwargs: typing.Any) -> PartialData: ...
    def acquire_sequence_continue(self, *, update_period: float = 1.0) -> PartialData: ...
    def acquire_sequence_end(self) -> None: ...
    def acquire_sequence_cancel(self) -> None: ...

    # properties

    @property
    def camera(self) -> CameraDevice: raise NotImplementedError()

    @property
    def camera_settings(self) -> CameraSettings: raise NotImplementedError()

    @property
    def binning_values(self) -> typing.Sequence[int]: raise NotImplementedError()

    @property
    def exposure_precision(self) -> int: raise NotImplementedError()

    # used in Facade. should be considered private.

    def set_current_frame_parameters(self, camera_frame_parameters: HardwareSource.FrameParameters) -> None: ...
    def get_current_frame_parameters(self) -> CameraFrameParameters: ...
    def set_record_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None: ...
    def get_record_frame_parameters(self) -> HardwareSource.FrameParameters: ...
    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> CameraFrameParameters: ...
    def get_frame_parameters(self, profile_index: int) -> CameraFrameParameters: ...
    def set_frame_parameters(self, profile_index: int, frame_parameters: CameraFrameParameters) -> None: ...
    def set_selected_profile_index(self, profile_index: int) -> None: ...
    def validate_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> CameraFrameParameters: ...

    @property
    def selected_profile_index(self) -> int: raise NotImplementedError()

    # private. do not use outside of instrumentation-kit.

    @property
    def modes(self) -> typing.Sequence[str]: raise NotImplementedError()

    def get_acquire_sequence_metrics(self, frame_parameters: CameraFrameParameters) -> typing.Mapping[str, typing.Any]: ...
    def make_live_data_element(self, data: _NDArray, properties: typing.Mapping[str, typing.Any], timestamp: datetime.datetime, frame_parameters: CameraFrameParameters, frame_count: int) -> ImportExportManager.DataElementType: ...
    def update_camera_properties(self, properties: typing.MutableMapping[str, typing.Any], frame_parameters: CameraFrameParameters, signal_type: typing.Optional[str] = None) -> None: ...
    def get_camera_calibrations(self, camera_frame_parameters: CameraFrameParameters) -> typing.Tuple[Calibration.Calibration, ...]: ...
    def get_camera_intensity_calibration(self, camera_frame_parameters: CameraFrameParameters) -> Calibration.Calibration: ...
    def shift_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None: ...
    def tilt_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None: ...
    def open_configuration_interface(self, api_broker: typing.Any) -> None: ...
    def periodic(self) -> None: ...

    profile_changed_event: Event.Event
    frame_parameters_changed_event: Event.Event
    log_messages_event: Event.Event


class CameraHardwareSource2(HardwareSource.ConcreteHardwareSource, CameraHardwareSource):

    def __init__(self, instrument_controller_id: typing.Optional[str], camera: CameraDevice, camera_settings: CameraSettings, configuration_location: typing.Optional[pathlib.Path], camera_panel_type: typing.Optional[str], camera_panel_delegate_type: typing.Optional[str] = None):
        super().__init__(typing.cast(typing.Any, camera).camera_id, typing.cast(typing.Any, camera).camera_name)

        # configure the event loop object
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(logging.INFO)
        self.__event_loop = asyncio.new_event_loop()  # outputs a debugger message!
        logger.setLevel(old_level)

        self.__camera_settings = camera_settings
        self.__camera_settings.initialize(configuration_location=configuration_location, event_loop=self.__event_loop)

        self.__current_frame_parameters_changed_event_listener = self.__camera_settings.current_frame_parameters_changed_event.listen(self.__current_frame_parameters_changed)
        self.__record_frame_parameters_changed_event_listener = self.__camera_settings.record_frame_parameters_changed_event.listen(self.__record_frame_parameters_changed)

        # add optional support for settings. to enable auto settings handling, the camera settings object must define
        # a settings_id property (which can just be the camera id), an apply_settings method which takes a settings
        # dict read from the config file and applies it as the settings, and a settings_changed_event which must be
        # fired when the settings changed (at which point they will be written to the config file).
        self.__settings_changed_event_listener = None
        if configuration_location and hasattr(self.__camera_settings, "settings_id"):
            config_file = configuration_location / pathlib.Path(self.__camera_settings.settings_id + "_config.json")
            logging.info("Camera device configuration: " + str(config_file))
            if config_file.is_file():
                with open(config_file) as f:
                    settings_dict = json.load(f)
                self.__camera_settings.apply_settings(settings_dict)

            def settings_changed(settings_dict: typing.Mapping[str, typing.Any]) -> None:
                # atomically overwrite
                temp_filepath = config_file.with_suffix(".temp")
                with open(temp_filepath, "w") as fp:
                    json.dump(settings_dict, fp, skipkeys=True, indent=4)
                os.replace(temp_filepath, config_file)

            self.__settings_changed_event_listener = self.__camera_settings.settings_changed_event.listen(settings_changed)

        self.__instrument_controller_id = instrument_controller_id
        self.__instrument_controller: typing.Optional[InstrumentController] = None

        self.__camera = camera
        self.__camera_category = typing.cast(typing.Any, camera).camera_type
        # signal type falls back to camera category if camera category is "eels" or "ronchigram". this is only for
        # backward compatibility. new camera instances should define signal_type directly.
        self.__signal_type = getattr(camera, "signal_type", self.__camera_category if self.__camera_category in ("eels", "ronchigram") else None)
        self.processor = None

        # configure the features. putting the features into this object is for convenience of access. the features
        # should not be considered as part of this class. instead, the features should be thought of as being stored
        # here as a convenient location where the UI has access to them.
        self.features["is_camera"] = True
        if camera_panel_type:
            self.features["camera_panel_type"] = camera_panel_type
        if camera_panel_delegate_type:
            self.features["camera_panel_delegate_type"] = camera_panel_delegate_type
        if self.__camera_category.lower() == "ronchigram":
            self.features["is_ronchigram_camera"] = True
        if self.__camera_category.lower() == "eels":
            self.features["is_eels_camera"] = True
        if getattr(camera, "has_processed_channel", True if self.__camera_category.lower() == "eels" else False):
            self.processor = HardwareSource.SumProcessor(Geometry.FloatRect(Geometry.FloatPoint(0.25, 0.0), Geometry.FloatSize(0.5, 1.0)))
        self.features["has_processed_channel"] = self.processor is not None
        if hasattr(camera_settings, "masks"):
            self.features["has_masked_sum_option"] = True
        # future version will also include the processed channel type;
        # candidates for the official name are "vertical_sum" or "vertical_projection_profile"

        # add channels
        self.add_data_channel()
        if self.processor:
            self.add_channel_processor(0, self.processor)

        # define deprecated events. both are used in camera control panel. frame_parameter_changed_event used in scan acquisition.
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()

        self.__profile_changed_event_listener = self.__camera_settings.profile_changed_event.listen(self.profile_changed_event.fire)
        self.__frame_parameters_changed_event_listener = self.__camera_settings.frame_parameters_changed_event.listen(self.frame_parameters_changed_event.fire)

        # define events
        self.log_messages_event = Event.Event()

        self.__frame_parameters = self.__camera_settings.get_current_frame_parameters()
        self.__record_parameters = self.__camera_settings.get_record_frame_parameters()

        self.__acquisition_task: typing.Optional[HardwareSource.AcquisitionTask] = None

        # the periodic logger function retrieves any log messages from the camera. it is called during
        # __handle_log_messages_event. any messages are sent out on the log_messages_event.
        periodic_logger_fn = getattr(self.__camera, "periodic_logger_fn", None)
        self.__periodic_logger_fn = periodic_logger_fn if callable(periodic_logger_fn) else None

    def close(self) -> None:
        Process.close_event_loop(self.__event_loop)
        self.__event_loop = typing.cast(asyncio.AbstractEventLoop, None)
        self.__periodic_logger_fn = None
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
        self.__camera_settings.close()
        self.__camera_settings = typing.cast(typing.Any, None)
        camera_close_method = getattr(self.__camera, "close", None)
        if callable(camera_close_method):
            camera_close_method()
        self.__camera = typing.cast(typing.Any, None)

    def periodic(self) -> None:
        self.__event_loop.stop()
        self.__event_loop.run_forever()
        self.__handle_log_messages_event()

    def __get_instrument_controller(self) -> InstrumentController:
        if not self.__instrument_controller and self.__instrument_controller_id:
            self.__instrument_controller = typing.cast(typing.Any, HardwareSource.HardwareSourceManager().get_instrument_by_id(self.__instrument_controller_id))
        if not self.__instrument_controller and not self.__instrument_controller_id:
            self.__instrument_controller = Registry.get_component("instrument_controller")
        if not self.__instrument_controller and not self.__instrument_controller_id:
            self.__instrument_controller = Registry.get_component("stem_controller")
        if not self.__instrument_controller:
            print(f"Instrument Controller ({self.__instrument_controller_id}) for ({self.hardware_source_id}) not found. Using proxy.")
            from nion.instrumentation import stem_controller
            self.__instrument_controller = self.__instrument_controller or typing.cast(InstrumentController, stem_controller.STEMController())
        return self.__instrument_controller

    def __handle_log_messages_event(self) -> None:
        if callable(self.__periodic_logger_fn):
            messages, data_elements = self.__periodic_logger_fn()
            if len(messages) > 0 or len(data_elements) > 0:
                self.log_messages_event.fire(messages, data_elements)

    def start_playing(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # note: sum_project has been mishandled in the past. it is not valid for view modes. clear it here.
        if "frame_parameters" in kwargs:
            camera_frame_parameters = typing.cast(CameraFrameParameters, kwargs["frame_parameters"])
            camera_frame_parameters.processing = None
            self.set_current_frame_parameters(camera_frame_parameters)
        elif len(args) == 1 and isinstance(args[0], dict):
            camera_frame_parameters = CameraFrameParameters(args[0])
            camera_frame_parameters.processing = None
            self.set_current_frame_parameters(camera_frame_parameters)
        super().start_playing(*args, **kwargs)

    def grab_next_to_start(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        self.start_playing()
        return self.get_next_xdatas_to_start(timeout)

    def grab_next_to_finish(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        self.start_playing()
        return self.get_next_xdatas_to_finish(timeout)

    def grab_sequence_prepare(self, count: int, **kwargs: typing.Any) -> bool:
        self.acquire_sequence_prepare(count)
        return True

    def grab_sequence(self, count: int, **kwargs: typing.Any) -> typing.Optional[typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]]:
        self.start_playing()
        frames = self.acquire_sequence(count)
        if frames is not None:
            xdatas = list()
            for data_element in frames:
                data_element["is_sequence"] = True
                data_element["collection_dimension_count"] = 0
                data_element["datum_dimension_count"] = len(data_element["data"].shape) - 1
                xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
                xdatas.append(xdata)
            return xdatas
        return None

    def grab_buffer(self, count: int, *, start: typing.Optional[int] = None, **kwargs: typing.Any) -> typing.Optional[typing.List[typing.List[DataAndMetadata.DataAndMetadata]]]:
        return None

    def make_reference_key(self, **kwargs: typing.Any) -> str:
        reference_key = kwargs.get("reference_key")
        if reference_key:
            return "_".join([self.hardware_source_id, str(reference_key)])
        return self.hardware_source_id

    @property
    def camera_settings(self) -> CameraSettings:
        return self.__camera_settings

    @property
    def camera(self) -> CameraDevice:
        return self.__camera

    @property
    def sensor_dimensions(self) -> typing.Tuple[int, int]:
        return self.__camera.sensor_dimensions

    @property
    def exposure_precision(self) -> int:
        return -3

    @property
    def binning_values(self) -> typing.Sequence[int]:
        return self.__camera.binning_values

    @property
    def readout_area(self) -> typing.Tuple[int, int, int, int]:
        return self.__camera.readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR: typing.Tuple[int, int, int, int]) -> None:
        self.__camera.readout_area = readout_area_TLBR

    def get_expected_dimensions(self, binning: int) -> typing.Tuple[int, int]:
        return self.__camera.get_expected_dimensions(binning)

    def get_signal_name(self, camera_frame_parameters: CameraFrameParameters) -> str:
        if self.__signal_type == "eels":
            if camera_frame_parameters.processing == "sum_project":
                return _("EELS")
            else:
                return _("EELS Image")
        elif self.__signal_type == "ronchigram":
            return _("Ronchigram")
        else:
            return _("Camera Data")

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        return CameraAcquisitionTask(self.__get_instrument_controller(), self, True, self.__camera_category, self.__signal_type, self.__frame_parameters)

    def _view_task_updated(self, view_task: typing.Optional[HardwareSource.AcquisitionTask]) -> None:
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self, *, frame_parameters: typing.Optional[HardwareSource.FrameParameters] = None, **kwargs: typing.Any) -> HardwareSource.AcquisitionTask:
        record_parameters = CameraFrameParameters(frame_parameters.as_dict()) if frame_parameters else self.__record_parameters
        assert record_parameters is not None
        return CameraAcquisitionTask(self.__get_instrument_controller(), self, False, self.__camera_category, self.__signal_type, record_parameters)

    def acquire_synchronized_begin(self, camera_frame_parameters: CameraFrameParameters, collection_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> PartialData:
        acquire_synchronized_begin = getattr(self.__camera, "acquire_synchronized_begin", None)
        if callable(acquire_synchronized_begin):
            return typing.cast(PartialData, acquire_synchronized_begin(camera_frame_parameters, collection_shape))
        else:
            data_elements = self.acquire_synchronized(collection_shape)
            if len(data_elements) > 0:
                data_elements[0]["data"] = data_elements[0]["data"].reshape(*collection_shape, *(data_elements[0]["data"].shape[1:]))
                xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_elements[0])
                return PartialData(xdata, True, False, collection_shape[0])
        return PartialData(DataAndMetadata.new_data_and_metadata(numpy.zeros(())), True, True, 0)

    def acquire_synchronized_continue(self, *, update_period: float = 1.0) -> PartialData:
        acquire_synchronized_continue = getattr(self.__camera, "acquire_synchronized_continue", None)
        if callable(acquire_synchronized_continue):
            return typing.cast(PartialData, acquire_synchronized_continue(update_period=update_period))
        return PartialData(DataAndMetadata.new_data_and_metadata(numpy.zeros(())), True, True, 0)

    def acquire_synchronized_end(self) -> None:
        acquire_synchronized_end = getattr(self.__camera, "acquire_synchronized_end", None)
        if callable(acquire_synchronized_end):
            acquire_synchronized_end()

    def acquire_synchronized_prepare(self, data_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> None:
        acquire_synchronized_prepare = getattr(self.__camera, "acquire_synchronized_prepare", None)
        if callable(acquire_synchronized_prepare):
            frame_parameters = self.get_current_frame_parameters()
            self.__camera.set_frame_parameters(frame_parameters)
            acquire_synchronized_prepare(data_shape, **kwargs)
        else:
            self.acquire_sequence_prepare(int(numpy.product(data_shape)))  # type: ignore

    def acquire_synchronized(self, data_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> typing.Sequence[ImportExportManager.DataElementType]:
        acquire_synchronized = getattr(self.__camera, "acquire_synchronized", None)
        if callable(acquire_synchronized):
            frame_parameters = self.get_current_frame_parameters()
            data_element = acquire_synchronized(data_shape, **kwargs)
            if data_element:
                self.__update_data_element_for_sequence(data_element, frame_parameters)
                return [data_element]
            return []
        else:
            return self.acquire_sequence(int(numpy.product(data_shape)))  # type: ignore

    def acquire_sequence_prepare(self, n: int) -> None:
        frame_parameters = self.get_current_frame_parameters()
        self.__camera.set_frame_parameters(frame_parameters)
        acquire_sequence_prepare = getattr(self.__camera, "acquire_sequence_prepare", None)
        if callable(acquire_sequence_prepare):
            acquire_sequence_prepare(n)

    def __acquire_sequence_fallback(self, n: int, frame_parameters: CameraFrameParameters) -> typing.Optional[ImportExportManager.DataElementType]:
        # if the device does not implement acquire_sequence, fall back to looping acquisition.
        processing = frame_parameters.processing
        acquisition_task = CameraAcquisitionTask(self.__get_instrument_controller(), self, True, self.__camera_category, self.__signal_type, frame_parameters)
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
                    summed_xdata = Core.function_sum(DataAndMetadata.new_data_and_metadata(frame_data), 0)
                    assert summed_xdata
                    data[index] = summed_xdata.data
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
        data_element: typing.Dict[str, typing.Any] = dict()
        data_element["data"] = data
        data_element["metadata"] = dict()
        data_element["hardware_source"] = properties
        return data_element

    def acquire_sequence(self, n: int) -> typing.Sequence[ImportExportManager.DataElementType]:
        frame_parameters = self.get_current_frame_parameters()
        acquire_sequence = getattr(self.__camera, "acquire_sequence", None)
        if callable(acquire_sequence):
            data_element = acquire_sequence(n)
        else:
            data_element = self.__acquire_sequence_fallback(n, frame_parameters)
        if data_element:
            self.__update_data_element_for_sequence(data_element, frame_parameters)
            return [data_element]
        return []

    def __build_calibration_dict(self, instrument_controller: InstrumentController, calibration_controls: typing.Mapping[str, typing.Union[str, int, float]], prefix: str, relative_scale: float = 1, data_len: int = 0) -> typing.Dict[str, typing.Any]:
        return build_calibration(instrument_controller, calibration_controls, prefix, relative_scale, data_len).rpc_dict

    def __update_spatial_calibrations(self, data_element: ImportExportManager.DataElementType, instrument_controller: InstrumentController, camera: CameraDevice, camera_category: str, data_shape: DataAndMetadata.ShapeType, scaling_x: float, scaling_y: float) -> None:
        if "spatial_calibrations" not in data_element:
            if "spatial_calibrations" in data_element.get("hardware_source", dict()):
                data_element["spatial_calibrations"] = data_element["hardware_source"]["spatial_calibrations"]
            elif hasattr(camera, "calibration"):  # used in nionccd1010
                data_element["spatial_calibrations"] = getattr(camera, "calibration")
            elif instrument_controller:
                if "calibration_controls" in data_element:
                    calibration_controls = data_element["calibration_controls"]
                elif hasattr(camera, "calibration_controls"):
                    calibration_controls = camera.calibration_controls
                else:
                    calibration_controls = None
                if calibration_controls is not None:
                    x_calibration_dict = self.__build_calibration_dict(instrument_controller, calibration_controls, "x", scaling_x, data_shape[0])
                    y_calibration_dict = self.__build_calibration_dict(instrument_controller, calibration_controls, "y", scaling_y, data_shape[1] if len(data_shape) > 1 else 0)
                    z_calibration_dict = self.__build_calibration_dict(instrument_controller, calibration_controls, "z", 1, data_shape[2] if len(data_shape) > 2 else 0)
                    # leave this here for backwards compatibility until origin override is specified in NionCameraManager.py
                    if camera_category.lower() == "ronchigram" and len(data_shape) == 2:
                        y_calibration_dict["offset"] = -y_calibration_dict.get("scale", 1) * data_shape[0] * 0.5
                        x_calibration_dict["offset"] = -x_calibration_dict.get("scale", 1) * data_shape[1] * 0.5
                    if len(data_shape) == 1:
                        data_element["spatial_calibrations"] = [x_calibration_dict]
                    if len(data_shape) == 2:
                        data_element["spatial_calibrations"] = [y_calibration_dict, x_calibration_dict]
                    if len(data_shape) == 3:
                        data_element["spatial_calibrations"] = [z_calibration_dict, y_calibration_dict, x_calibration_dict]

    def __update_intensity_calibration(self, data_element: ImportExportManager.DataElementType, instrument_controller: InstrumentController, camera: CameraDevice) -> None:
        if instrument_controller and "calibration_controls" in data_element:
            calibration_controls = data_element["calibration_controls"]
        elif instrument_controller and hasattr(camera, "calibration_controls"):
            calibration_controls = camera.calibration_controls
        else:
            calibration_controls = None
        if "intensity_calibration" not in data_element:
            if "intensity_calibration" in data_element.get("hardware_source", dict()):
                data_element["intensity_calibration"] = data_element["hardware_source"]["intensity_calibration"]
            elif calibration_controls is not None:
                data_element["intensity_calibration"] = self.__build_calibration_dict(instrument_controller, calibration_controls, "intensity")
        if "counts_per_electron" not in data_element:
            if calibration_controls is not None:
                counts_per_electron = get_instrument_calibration_value(instrument_controller, calibration_controls, "counts_per_electron")
                if counts_per_electron:
                    data_element.setdefault("metadata", dict()).setdefault("hardware_source", dict())["counts_per_electron"] = counts_per_electron

    def __update_data_element_for_sequence(self, data_element: ImportExportManager.DataElementType, frame_parameters: CameraFrameParameters) -> None:
        binning = frame_parameters.binning
        data_element["version"] = 1
        data_element["state"] = "complete"
        instrument_controller = self.__get_instrument_controller()
        if "spatial_calibrations" not in data_element:
            self.__update_spatial_calibrations(data_element, instrument_controller, self.__camera, self.__camera_category,
                                               data_element["data"].shape[1:], binning, binning)
            if "spatial_calibrations" in data_element:
                data_element["spatial_calibrations"] = [dict(), ] + data_element["spatial_calibrations"]
        self.__update_intensity_calibration(data_element, instrument_controller, self.__camera)
        update_instrument_properties(data_element.setdefault("metadata", dict()).setdefault("instrument", dict()), instrument_controller, self.__camera)
        update_camera_properties(data_element.setdefault("metadata", dict()).setdefault("hardware_source", dict()), frame_parameters, self.hardware_source_id, self.display_name, data_element.get("signal_type", self.__signal_type))

    def make_live_data_element(self, data: _NDArray, properties: typing.Mapping[str, typing.Any], timestamp: datetime.datetime, frame_parameters: CameraFrameParameters, frame_count: int) -> ImportExportManager.DataElementType:
        data_element: ImportExportManager.DataElementType = dict()
        data_element["metadata"] = dict()
        data_element["metadata"]["hardware_source"] = copy.deepcopy(dict(properties))
        data_element["data"] = data
        data_element["version"] = 1
        data_element["state"] = "complete"
        data_element["timestamp"] = timestamp
        instrument_controller = self.__instrument_controller
        assert instrument_controller
        self.__update_spatial_calibrations(data_element, instrument_controller, self.__camera, self.__camera_category, data.shape, frame_parameters.binning, frame_parameters.binning)
        self.__update_intensity_calibration(data_element, instrument_controller, self.__camera)
        instrument_metadata: typing.Dict[str, typing.Any] = dict()
        update_instrument_properties(instrument_metadata, instrument_controller, self.__camera)
        if instrument_metadata:
            data_element["metadata"].setdefault("instrument", dict()).update(instrument_metadata)
        update_camera_properties(data_element["metadata"]["hardware_source"], frame_parameters, self.hardware_source_id, self.display_name, data_element.get("signal_type", self.__signal_type))
        data_element["metadata"]["hardware_source"]["valid_rows"] = data.shape[0]
        data_element["metadata"]["hardware_source"]["frame_index"] = data_element["metadata"]["hardware_source"]["frame_number"]
        data_element["metadata"]["hardware_source"]["integration_count"] = frame_count
        return data_element

    def update_camera_properties(self, properties: typing.MutableMapping[str, typing.Any], frame_parameters: CameraFrameParameters, signal_type: typing.Optional[str] = None) -> None:
        update_instrument_properties(properties, self.__get_instrument_controller(), self.__camera)
        update_camera_properties(properties, frame_parameters, self.hardware_source_id, self.display_name, signal_type or self.__signal_type)

    def get_camera_calibrations(self, camera_frame_parameters: CameraFrameParameters) -> typing.Tuple[Calibration.Calibration, ...]:
        processing = camera_frame_parameters.processing
        instrument_controller = self.__get_instrument_controller()
        calibration_controls = self.__camera.calibration_controls
        binning = camera_frame_parameters.binning
        data_shape = self.get_expected_dimensions(binning)
        if processing in {"sum_masked"}:
            return (Calibration.Calibration(),)  # a dummy calibration; the masked dimension is 1 so this is needed
        elif processing in {"sum_project"}:
            x_calibration = build_calibration(instrument_controller, calibration_controls, "x", binning, data_shape[0])
            return (x_calibration,)
        else:
            y_calibration = build_calibration(instrument_controller, calibration_controls, "y", binning, data_shape[1] if len(data_shape) > 1 else 0)
            x_calibration = build_calibration(instrument_controller, calibration_controls, "x", binning, data_shape[0])
            return (y_calibration, x_calibration)

    def get_camera_intensity_calibration(self, camera_frame_parameters: CameraFrameParameters) -> Calibration.Calibration:
        instrument_controller = self.__instrument_controller
        assert instrument_controller
        return build_calibration(instrument_controller, self.__camera.calibration_controls, "intensity")

    def acquire_sequence_begin(self, camera_frame_parameters: CameraFrameParameters, count: int, **kwargs: typing.Any) -> PartialData:
        acquire_sequence_begin = getattr(self.__camera, "acquire_sequence_begin", None)
        if callable(acquire_sequence_begin):
            return typing.cast(PartialData, acquire_sequence_begin(camera_frame_parameters, count, **kwargs))
        raise NotImplementedError()

    def acquire_sequence_continue(self, *, update_period: float = 1.0) -> PartialData:
        acquire_sequence_continue = getattr(self.__camera, "acquire_sequence_continue", None)
        if callable(acquire_sequence_continue):
            return typing.cast(PartialData, acquire_sequence_continue())
        raise NotImplementedError()

    def acquire_sequence_end(self) -> None:
        acquire_sequence_end = getattr(self.__camera, "acquire_sequence_end", None)
        if callable(acquire_sequence_end):
            acquire_sequence_end()

    def acquire_sequence_cancel(self) -> None:
        acquire_sequence_cancel = getattr(self.__camera, "acquire_sequence_cancel", None)
        if callable(acquire_sequence_cancel):
            acquire_sequence_cancel()

    def get_acquire_sequence_metrics(self, frame_parameters: CameraFrameParameters) -> typing.Mapping[str, typing.Any]:
        get_acquire_sequence_metrics = getattr(self.__camera, "get_acquire_sequence_metrics", None)
        if callable(get_acquire_sequence_metrics):
            return typing.cast(typing.Mapping[str, typing.Any], get_acquire_sequence_metrics(frame_parameters))
        return dict()

    def __current_frame_parameters_changed(self, frame_parameters: CameraFrameParameters) -> None:
        acquisition_task = self.__acquisition_task
        if isinstance(acquisition_task, CameraAcquisitionTask):
            acquisition_task.set_frame_parameters(CameraFrameParameters(frame_parameters.as_dict()))
        self.__frame_parameters = CameraFrameParameters(frame_parameters.as_dict())

    def set_current_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None:
        frame_parameters = CameraFrameParameters(frame_parameters.as_dict())
        self.__camera_settings.set_current_frame_parameters(frame_parameters)
        # __current_frame_parameters_changed will be called by the controller

    def get_current_frame_parameters(self) -> CameraFrameParameters:
        return CameraFrameParameters(self.__frame_parameters.as_dict() if self.__frame_parameters else dict())

    def __record_frame_parameters_changed(self, frame_parameters: CameraFrameParameters) -> None:
        self.__record_parameters = CameraFrameParameters(frame_parameters.as_dict())

    def set_record_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None:
        frame_parameters = CameraFrameParameters(frame_parameters.as_dict())
        self.__camera_settings.set_record_frame_parameters(frame_parameters)
        # __record_frame_parameters_changed will be called by the controller

    def get_record_frame_parameters(self) -> HardwareSource.FrameParameters:
        return CameraFrameParameters(self.__record_parameters.as_dict() if self.__record_parameters else dict())

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> CameraFrameParameters:
        return CameraFrameParameters(d)

    def validate_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> CameraFrameParameters:
        return CameraFrameParameters(frame_parameters.as_dict())

    def shift_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None:
        instrument_controller = self.__get_instrument_controller()
        if callable(getattr(instrument_controller, "handle_shift_click", None)):
            instrument_controller.handle_shift_click(mouse_position=mouse_position, data_shape=camera_shape, camera=self.camera, logger=logger)
        else:
            # TODO: remove this backwards compatibility code once everyone updated to new technique above
            if self.__camera_category.lower() == "ronchigram":
                radians_per_pixel = typing.cast(float, instrument_controller.get_value("TVPixelAngle", 0.0))
                defocus_value = typing.cast(float, instrument_controller.get_value("C10", 0.0))
                dx = radians_per_pixel * defocus_value * (mouse_position[1] - (camera_shape[1] / 2))
                dy = radians_per_pixel * defocus_value * (mouse_position[0] - (camera_shape[0] / 2))
                logger.info("Shifting (%s,%s) um.\n", -dx * 1e6, -dy * 1e6)
                sx = instrument_controller.get_value("SShft.x")
                sy = instrument_controller.get_value("SShft.y")
                if sx is not None and sy is not None:
                    instrument_controller.set_value("SShft.x", sx - dx)
                    instrument_controller.set_value("SShft.y", sy - dy)

    def tilt_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None:
        instrument_controller = self.__get_instrument_controller()
        if callable(getattr(instrument_controller, "handle_tilt_click", None)):
            instrument_controller.handle_tilt_click(mouse_position=mouse_position, data_shape=camera_shape, camera=self.camera, logger=logger)
        else:
            # TODO: remove this backwards compatibility code once everyone updated to new technique above
            if self.__camera_category.lower() == "ronchigram":
                radians_per_pixel = instrument_controller.get_value("TVPixelAngle")
                tx = instrument_controller.get_value("STilt.x")
                ty = instrument_controller.get_value("STilt.y")
                if radians_per_pixel is not None and tx is not None and ty is not None:
                    da = radians_per_pixel * (mouse_position.x - (camera_shape[1] / 2))
                    db = radians_per_pixel * (mouse_position.y - (camera_shape[0] / 2))
                    logger.info("Tilting (%s,%s) rad.\n", -da, -db)
                    instrument_controller.set_value("STilt.x", tx - da)
                    instrument_controller.set_value("STilt.y", ty - db)

    def get_property(self, name: str) -> typing.Any:
        return getattr(self.__camera, name)

    def set_property(self, name: str, value: typing.Any) -> None:
        setattr(self.__camera, name, value)

    def get_api(self, version: str) -> typing.Any:
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade:

            def __init__(self) -> None:
                pass

        return CameraFacade()

    # Compatibility functions

    # used in camera control panel
    @property
    def modes(self) -> typing.Sequence[str]:
        return self.__camera_settings.modes

    # used in service scripts
    def get_mode(self) -> str:
        return self.__camera_settings.get_mode()

    # used in service scripts
    def set_mode(self, mode: str) -> None:
        self.__camera_settings.set_mode(mode)

    # used in api, tests, camera control panel
    def set_frame_parameters(self, profile_index: int, frame_parameters: CameraFrameParameters) -> None:
        self.__camera_settings.set_frame_parameters(profile_index, frame_parameters)

    # used in tuning, api, tests, camera control panel
    def get_frame_parameters(self, profile_index: int) -> CameraFrameParameters:
        return self.__camera_settings.get_frame_parameters(profile_index)

    # used in api, tests, camera control panel
    def set_selected_profile_index(self, profile_index: int) -> None:
        self.__camera_settings.set_selected_profile_index(profile_index)

    # used in api, camera control panel
    @property
    def selected_profile_index(self) -> int:
        return self.__camera_settings.selected_profile_index

    # used in camera control panel
    def open_configuration_interface(self, api_broker: typing.Any) -> None:
        self.__camera_settings.open_configuration_interface(api_broker)


class CameraHardwareSource3(HardwareSource.ConcreteHardwareSource, CameraHardwareSource):
    """Construct hardware source from the device.

    Adds a primary channel and any processor channels automatically included with the device, e.g. a sum-processed
    channel for a 2D spectral device such as an EELS detector.
    """

    def __init__(self, instrument_controller_id: typing.Optional[str], camera: CameraDevice3, camera_settings: CameraSettings, configuration_location: typing.Optional[pathlib.Path], camera_panel_type: typing.Optional[str], camera_panel_delegate_type: typing.Optional[str] = None):
        super().__init__(typing.cast(typing.Any, camera).camera_id, typing.cast(typing.Any, camera).camera_name)

        # configure the event loop object
        logger = logging.getLogger()
        old_level = logger.level
        logger.setLevel(logging.INFO)
        self.__event_loop = asyncio.new_event_loop()  # outputs a debugger message!
        logger.setLevel(old_level)

        self.__camera_settings = camera_settings
        self.__camera_settings.initialize(configuration_location=configuration_location, event_loop=self.__event_loop)

        self.__current_frame_parameters_changed_event_listener = self.__camera_settings.current_frame_parameters_changed_event.listen(self.__current_frame_parameters_changed)
        self.__record_frame_parameters_changed_event_listener = self.__camera_settings.record_frame_parameters_changed_event.listen(self.__record_frame_parameters_changed)

        # add optional support for settings. to enable auto settings handling, the camera settings object must define
        # a settings_id property (which can just be the camera id), an apply_settings method which takes a settings
        # dict read from the config file and applies it as the settings, and a settings_changed_event which must be
        # fired when the settings changed (at which point they will be written to the config file).
        self.__settings_changed_event_listener = None
        if configuration_location and hasattr(self.__camera_settings, "settings_id"):
            config_file = configuration_location / pathlib.Path(self.__camera_settings.settings_id + "_config.json")
            logging.info("Camera device configuration: " + str(config_file))
            if config_file.is_file():
                with open(config_file) as f:
                    settings_dict = json.load(f)
                self.__camera_settings.apply_settings(settings_dict)

            def settings_changed(settings_dict: typing.Mapping[str, typing.Any]) -> None:
                # atomically overwrite
                temp_filepath = config_file.with_suffix(".temp")
                with open(temp_filepath, "w") as fp:
                    json.dump(settings_dict, fp, skipkeys=True, indent=4)
                os.replace(temp_filepath, config_file)

            self.__settings_changed_event_listener = self.__camera_settings.settings_changed_event.listen(settings_changed)

        self.__instrument_controller_id = instrument_controller_id
        self.__instrument_controller: typing.Optional[InstrumentController] = None

        self.__camera = camera
        self.__camera_category = typing.cast(typing.Any, camera).camera_type
        # signal type falls back to camera category if camera category is "eels" or "ronchigram". this is only for
        # backward compatibility. new camera instances should define signal_type directly.
        self.__signal_type = getattr(camera, "signal_type", self.__camera_category if self.__camera_category in ("eels", "ronchigram") else None)
        self.processor = None

        self.__camera_calibrator: typing.Optional[CameraCalibrator] = None

        # configure the features. putting the features into this object is for convenience of access. the features
        # should not be considered as part of this class. instead, the features should be thought of as being stored
        # here as a convenient location where the UI has access to them.
        self.features["is_camera"] = True
        if camera_panel_type:
            self.features["camera_panel_type"] = camera_panel_type
        if camera_panel_delegate_type:
            self.features["camera_panel_delegate_type"] = camera_panel_delegate_type
        if self.__camera_category.lower() == "ronchigram":
            self.features["is_ronchigram_camera"] = True
        if self.__camera_category.lower() == "eels":
            self.features["is_eels_camera"] = True
        if getattr(camera, "has_processed_channel", True if self.__camera_category.lower() == "eels" else False):
            self.processor = HardwareSource.SumProcessor(Geometry.FloatRect(Geometry.FloatPoint(0.25, 0.0), Geometry.FloatSize(0.5, 1.0)))
        self.features["has_processed_channel"] = self.processor is not None
        if hasattr(camera_settings, "masks"):
            self.features["has_masked_sum_option"] = True
        # future version will also include the processed channel type;
        # candidates for the official name are "vertical_sum" or "vertical_projection_profile"

        # add channels
        self.add_data_channel()
        if self.processor:
            self.add_channel_processor(0, self.processor)

        # define deprecated events. both are used in camera control panel. frame_parameter_changed_event used in scan acquisition.
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()

        self.__profile_changed_event_listener = self.__camera_settings.profile_changed_event.listen(self.profile_changed_event.fire)
        self.__frame_parameters_changed_event_listener = self.__camera_settings.frame_parameters_changed_event.listen(self.frame_parameters_changed_event.fire)

        # define events
        self.log_messages_event = Event.Event()

        self.__frame_parameters = self.__camera_settings.get_current_frame_parameters()
        self.__record_parameters = self.__camera_settings.get_record_frame_parameters()

        # view mode "processing" has been misused in the past; clear it here. it should never be enabled, at least in its current form.
        # the y-summed channel is handled below in the has_processed_channel/SumProcessor code.
        if self.__frame_parameters:
            self.__frame_parameters.processing = None
        if self.__record_parameters:
            self.__record_parameters.processing = None

        self.__acquisition_task: typing.Optional[HardwareSource.AcquisitionTask] = None

        self.__grab_sequence_partial_data: typing.Optional[PartialData] = None
        self.__grab_sequence_frame_parameters: typing.Optional[CameraFrameParameters] = None

        # the periodic logger function retrieves any log messages from the camera. it is called during
        # __handle_log_messages_event. any messages are sent out on the log_messages_event.
        periodic_logger_fn = getattr(self.__camera, "periodic_logger_fn", None)
        self.__periodic_logger_fn = periodic_logger_fn if callable(periodic_logger_fn) else None

    def close(self) -> None:
        Process.close_event_loop(self.__event_loop)
        self.__event_loop = typing.cast(asyncio.AbstractEventLoop, None)
        self.__periodic_logger_fn = None
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
        self.__camera_settings.close()
        self.__camera_settings = typing.cast(typing.Any, None)
        self.__camera.close()
        self.__camera = typing.cast(typing.Any, None)

    def periodic(self) -> None:
        self.__event_loop.stop()
        self.__event_loop.run_forever()
        self.__handle_log_messages_event()

    def __get_instrument_controller(self) -> InstrumentController:
        if not self.__instrument_controller and self.__instrument_controller_id:
            self.__instrument_controller = typing.cast(typing.Any, HardwareSource.HardwareSourceManager().get_instrument_by_id(self.__instrument_controller_id))
        if not self.__instrument_controller and not self.__instrument_controller_id:
            self.__instrument_controller = Registry.get_component("instrument_controller")
        if not self.__instrument_controller and not self.__instrument_controller_id:
            self.__instrument_controller = Registry.get_component("stem_controller")
        if not self.__instrument_controller:
            print(f"Instrument Controller ({self.__instrument_controller_id}) for ({self.hardware_source_id}) not found. Using proxy.")
            from nion.instrumentation import stem_controller
            self.__instrument_controller = self.__instrument_controller or typing.cast(InstrumentController, stem_controller.STEMController())
        return self.__instrument_controller

    def __get_camera_calibrator(self) -> CameraCalibrator:
        if not self.__camera_calibrator:
            self.__camera_calibrator = self.__camera.get_camera_calibrator(instrument_controller=self.__get_instrument_controller())
        assert self.__camera_calibrator
        return self.__camera_calibrator

    def __handle_log_messages_event(self) -> None:
        if callable(self.__periodic_logger_fn):
            messages, data_elements = self.__periodic_logger_fn()
            if len(messages) > 0 or len(data_elements) > 0:
                self.log_messages_event.fire(messages, data_elements)

    def start_playing(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # note: sum_project has been mishandled in the past. it is not valid for view modes. clear it here.
        if "frame_parameters" in kwargs:
            camera_frame_parameters = typing.cast(CameraFrameParameters, kwargs["frame_parameters"])
            camera_frame_parameters.processing = None
            self.set_current_frame_parameters(camera_frame_parameters)
        elif len(args) == 1 and isinstance(args[0], dict):
            camera_frame_parameters = CameraFrameParameters(args[0])
            camera_frame_parameters.processing = None
            self.set_current_frame_parameters(camera_frame_parameters)
        else:
            # hack in case camera_frame_parameters is already sum_project. ugh.
            if self.__frame_parameters:
                camera_frame_parameters = CameraFrameParameters(self.__frame_parameters.as_dict())
                camera_frame_parameters.processing = None
                self.set_current_frame_parameters(camera_frame_parameters)
        super().start_playing(*args, **kwargs)

    def grab_next_to_start(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        self.start_playing()
        return self.get_next_xdatas_to_start(timeout)

    def grab_next_to_finish(self, *, timeout: typing.Optional[float] = None, **kwargs: typing.Any) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        self.start_playing()
        return self.get_next_xdatas_to_finish(timeout)

    def grab_sequence_prepare(self, count: int, **kwargs: typing.Any) -> bool:
        self.__grab_sequence_frame_parameters = self.get_current_frame_parameters()
        self.__grab_sequence_partial_data = self.acquire_sequence_begin(self.__grab_sequence_frame_parameters, count)
        return True

    def grab_sequence(self, count: int, **kwargs: typing.Any) -> typing.Optional[typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]]:
        self.start_playing()  # backwards compatibility
        assert self.__grab_sequence_frame_parameters
        assert self.__grab_sequence_partial_data
        while not self.__grab_sequence_partial_data.is_complete and not self.__grab_sequence_partial_data.is_canceled:
            self.__grab_sequence_partial_data = self.acquire_sequence_continue()
        self.acquire_sequence_end()
        data_element: typing.Dict[str, typing.Any] = {"data": self.__grab_sequence_partial_data.xdata.data}
        self.__grab_sequence_partial_data = None
        self.__update_data_element_for_sequence(data_element, self.__grab_sequence_frame_parameters)
        self.__grab_sequence_frame_parameters = None
        frames = [data_element]
        xdatas = list()
        for data_element in frames:
            data_element["is_sequence"] = True
            data_element["collection_dimension_count"] = 0
            data_element["datum_dimension_count"] = len(data_element["data"].shape) - 1
            xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
            xdatas.append(xdata)
        return xdatas

    def grab_buffer(self, count: int, *, start: typing.Optional[int] = None, **kwargs: typing.Any) -> typing.Optional[typing.List[typing.List[DataAndMetadata.DataAndMetadata]]]:
        return None

    def make_reference_key(self, **kwargs: typing.Any) -> str:
        reference_key = kwargs.get("reference_key")
        if reference_key:
            return "_".join([self.hardware_source_id, str(reference_key)])
        return self.hardware_source_id

    @property
    def camera_settings(self) -> CameraSettings:
        return self.__camera_settings

    @property
    def camera(self) -> CameraDevice:
        return self.__camera

    @property
    def sensor_dimensions(self) -> typing.Tuple[int, int]:
        return self.__camera.sensor_dimensions

    @property
    def exposure_precision(self) -> int:
        return self.__camera.exposure_precision

    @property
    def binning_values(self) -> typing.Sequence[int]:
        return self.__camera.binning_values

    @property
    def readout_area(self) -> typing.Tuple[int, int, int, int]:
        return self.__camera.readout_area

    @readout_area.setter
    def readout_area(self, readout_area_TLBR: typing.Tuple[int, int, int, int]) -> None:
        self.__camera.readout_area = readout_area_TLBR

    def get_expected_dimensions(self, binning: int) -> typing.Tuple[int, int]:
        return self.__camera.get_expected_dimensions(binning)

    def get_signal_name(self, camera_frame_parameters: CameraFrameParameters) -> str:
        if self.__signal_type == "eels":
            if camera_frame_parameters.processing == "sum_project":
                return _("EELS")
            else:
                return _("EELS Image")
        elif self.__signal_type == "ronchigram":
            return _("Ronchigram")
        else:
            return _("Camera Data")

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        return CameraAcquisitionTask(self.__get_instrument_controller(), self, True, self.__camera_category, self.__signal_type, self.__frame_parameters)

    def _view_task_updated(self, view_task: typing.Optional[HardwareSource.AcquisitionTask]) -> None:
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self, *, frame_parameters: typing.Optional[HardwareSource.FrameParameters] = None, **kwargs: typing.Any) -> HardwareSource.AcquisitionTask:
        record_parameters = CameraFrameParameters(frame_parameters.as_dict()) if frame_parameters else self.__record_parameters
        assert record_parameters is not None
        return CameraAcquisitionTask(self.__get_instrument_controller(), self, False, self.__camera_category, self.__signal_type, record_parameters)

    def acquire_synchronized_begin(self, camera_frame_parameters: CameraFrameParameters, collection_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> PartialData:
        return self.__camera.acquire_synchronized_begin(camera_frame_parameters, collection_shape, **kwargs)

    def acquire_synchronized_continue(self, *, update_period: float = 1.0) -> PartialData:
        return self.__camera.acquire_synchronized_continue(update_period=update_period)

    def acquire_synchronized_end(self) -> None:
        self.__camera.acquire_synchronized_end()

    def acquire_synchronized_prepare(self, data_shape: DataAndMetadata.ShapeType, **kwargs: typing.Any) -> None:
        # prepare does nothing in camera device 3
        pass

    def acquire_sequence(self, n: int) -> typing.Sequence[ImportExportManager.DataElementType]:
        frame_parameters = self.get_current_frame_parameters()
        partial_data = self.acquire_sequence_begin(frame_parameters, n)
        while not partial_data.is_complete and not partial_data.is_canceled:
            partial_data = self.acquire_sequence_continue()
        self.acquire_sequence_end()
        data_element = {"data": partial_data.xdata.data}
        self.__update_data_element_for_sequence(data_element, frame_parameters)
        return [data_element]

    def __update_data_element_for_sequence(self, data_element: ImportExportManager.DataElementType, frame_parameters: CameraFrameParameters) -> None:
        acquisition_data = AcquisitionData(data_element)
        data_element["version"] = 1
        data_element["state"] = "complete"
        instrument_controller = self.__get_instrument_controller()
        camera_calibrations = self.get_camera_calibrations(frame_parameters)
        acquisition_data.apply_signal_calibrations([Calibration.Calibration()] + list(camera_calibrations))
        acquisition_data.apply_intensity_calibration(self.get_camera_intensity_calibration(frame_parameters))
        acquisition_data.counts_per_electron = self.get_counts_per_electron()
        update_instrument_properties(data_element.setdefault("metadata", dict()).setdefault("instrument", dict()), instrument_controller, self.__camera)
        update_camera_properties(data_element.setdefault("metadata", dict()).setdefault("hardware_source", dict()), frame_parameters, self.hardware_source_id, self.display_name, data_element.get("signal_type", self.__signal_type))

    def make_live_data_element(self, data: _NDArray, properties: typing.Mapping[str, typing.Any], timestamp: datetime.datetime, frame_parameters: CameraFrameParameters, frame_count: int) -> ImportExportManager.DataElementType:
        acquisition_data = AcquisitionData()
        data_element = acquisition_data.data_element
        data_element["metadata"] = dict()
        data_element["metadata"]["hardware_source"] = copy.deepcopy(dict(properties))
        data_element["data"] = data
        data_element["version"] = 1
        data_element["state"] = "complete"
        data_element["timestamp"] = timestamp
        # use a frame parameters copy without processing to hack around calibration issue where view mode is
        # has wrong calibration during SI because processing is enabled is frame parameters when SI starts but
        # the view acquisition is 2D with the sum processor. strong code smell.
        frame_parameters_copy = copy.deepcopy(frame_parameters)
        frame_parameters_copy.processing = None
        camera_calibrations = self.get_camera_calibrations(frame_parameters_copy)
        acquisition_data.apply_signal_calibrations(list(camera_calibrations))
        acquisition_data.apply_intensity_calibration(self.get_camera_intensity_calibration(frame_parameters))
        acquisition_data.counts_per_electron = self.get_counts_per_electron()
        instrument_metadata: typing.Dict[str, typing.Any] = dict()
        instrument_controller = self.__instrument_controller
        assert instrument_controller
        update_instrument_properties(instrument_metadata, instrument_controller, self.__camera)
        if instrument_metadata:
            data_element["metadata"].setdefault("instrument", dict()).update(instrument_metadata)
        update_camera_properties(data_element["metadata"]["hardware_source"], frame_parameters, self.hardware_source_id, self.display_name, data_element.get("signal_type", self.__signal_type))
        data_element["metadata"]["hardware_source"]["valid_rows"] = data.shape[0]
        data_element["metadata"]["hardware_source"]["frame_index"] = data_element["metadata"]["hardware_source"]["frame_number"]
        data_element["metadata"]["hardware_source"]["integration_count"] = frame_count
        return data_element

    def update_camera_properties(self, properties: typing.MutableMapping[str, typing.Any], frame_parameters: CameraFrameParameters, signal_type: typing.Optional[str] = None) -> None:
        update_instrument_properties(properties, self.__get_instrument_controller(), self.__camera)
        update_camera_properties(properties, frame_parameters, self.hardware_source_id, self.display_name, signal_type or self.__signal_type)

    def get_camera_calibrations(self, camera_frame_parameters: CameraFrameParameters) -> typing.Tuple[Calibration.Calibration, ...]:
        calibrator = self.__get_camera_calibrator()
        processing = camera_frame_parameters.processing
        binning = camera_frame_parameters.binning
        data_shape = self.get_expected_dimensions(binning)
        if processing in {"sum_masked"}:
            return (Calibration.Calibration(),)  # a dummy calibration; the masked dimension is 1 so this is needed
        elif processing in {"sum_project"}:
            return tuple(calibrator.get_signal_calibrations(camera_frame_parameters, data_shape[0:1]))
        else:
            return tuple(calibrator.get_signal_calibrations(camera_frame_parameters, data_shape))

    def get_camera_intensity_calibration(self, camera_frame_parameters: CameraFrameParameters) -> Calibration.Calibration:
        return self.__get_camera_calibrator().get_intensity_calibration(camera_frame_parameters)

    def get_counts_per_electron(self) -> typing.Optional[float]:
        return self.__get_camera_calibrator().get_counts_per_electron()

    def acquire_sequence_prepare(self, n: int) -> None:
        # prepare does nothing in camera device 3
        pass

    def acquire_sequence_begin(self, camera_frame_parameters: CameraFrameParameters, count: int, **kwargs: typing.Any) -> PartialData:
        return self.__camera.acquire_sequence_begin(camera_frame_parameters, count, **kwargs)

    def acquire_sequence_continue(self, *, update_period: float = 1.0) -> PartialData:
        return self.__camera.acquire_sequence_continue(update_period=update_period)

    def acquire_sequence_end(self) -> None:
        self.__camera.acquire_sequence_end()

    def acquire_sequence_cancel(self) -> None:
        self.__camera.acquire_sequence_cancel()

    def get_acquire_sequence_metrics(self, frame_parameters: CameraFrameParameters) -> typing.Mapping[str, typing.Any]:
        get_acquire_sequence_metrics = getattr(self.__camera, "get_acquire_sequence_metrics", None)
        if callable(get_acquire_sequence_metrics):
            return typing.cast(typing.Mapping[str, typing.Any], get_acquire_sequence_metrics(frame_parameters))
        return dict()

    def __current_frame_parameters_changed(self, frame_parameters: CameraFrameParameters) -> None:
        acquisition_task = self.__acquisition_task
        if isinstance(acquisition_task, CameraAcquisitionTask):
            acquisition_task.set_frame_parameters(CameraFrameParameters(frame_parameters.as_dict()))
        self.__frame_parameters = CameraFrameParameters(frame_parameters.as_dict())

    def set_current_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None:
        frame_parameters = CameraFrameParameters(frame_parameters.as_dict())
        self.__camera_settings.set_current_frame_parameters(frame_parameters)
        # __current_frame_parameters_changed will be called by the controller

    def get_current_frame_parameters(self) -> CameraFrameParameters:
        return CameraFrameParameters(self.__frame_parameters.as_dict() if self.__frame_parameters else dict())

    def __record_frame_parameters_changed(self, frame_parameters: CameraFrameParameters) -> None:
        self.__record_parameters = CameraFrameParameters(frame_parameters.as_dict())

    def set_record_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> None:
        frame_parameters = CameraFrameParameters(frame_parameters.as_dict())
        self.__camera_settings.set_record_frame_parameters(frame_parameters)
        # __record_frame_parameters_changed will be called by the controller

    def get_record_frame_parameters(self) -> HardwareSource.FrameParameters:
        return CameraFrameParameters(self.__record_parameters.as_dict() if self.__record_parameters else dict())

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> CameraFrameParameters:
        return CameraFrameParameters(d)

    def validate_frame_parameters(self, frame_parameters: HardwareSource.FrameParameters) -> CameraFrameParameters:
        frame_parameters = CameraFrameParameters(frame_parameters.as_dict())
        return self.__camera.validate_frame_parameters(frame_parameters)

    def shift_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None:
        instrument_controller = self.__get_instrument_controller()
        if callable(getattr(instrument_controller, "handle_shift_click", None)):
            instrument_controller.handle_shift_click(mouse_position=mouse_position, data_shape=camera_shape, camera=self.camera, logger=logger)

    def tilt_click(self, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> None:
        instrument_controller = self.__get_instrument_controller()
        if callable(getattr(instrument_controller, "handle_tilt_click", None)):
            instrument_controller.handle_tilt_click(mouse_position=mouse_position, data_shape=camera_shape, camera=self.camera, logger=logger)

    def get_property(self, name: str) -> typing.Any:
        return getattr(self.__camera, name)

    def set_property(self, name: str, value: typing.Any) -> None:
        setattr(self.__camera, name, value)

    def get_api(self, version: str) -> typing.Any:
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade:

            def __init__(self) -> None:
                pass

        return CameraFacade()

    # Compatibility functions

    # used in camera control panel
    @property
    def modes(self) -> typing.Sequence[str]:
        return self.__camera_settings.modes

    # used in service scripts
    def get_mode(self) -> str:
        return self.__camera_settings.get_mode()

    # used in service scripts
    def set_mode(self, mode: str) -> None:
        self.__camera_settings.set_mode(mode)

    # used in api, tests, camera control panel
    def set_frame_parameters(self, profile_index: int, frame_parameters: CameraFrameParameters) -> None:
        self.__camera_settings.set_frame_parameters(profile_index, frame_parameters)

    # used in tuning, api, tests, camera control panel
    def get_frame_parameters(self, profile_index: int) -> CameraFrameParameters:
        return self.__camera_settings.get_frame_parameters(profile_index)

    # used in api, tests, camera control panel
    def set_selected_profile_index(self, profile_index: int) -> None:
        self.__camera_settings.set_selected_profile_index(profile_index)

    # used in api, camera control panel
    @property
    def selected_profile_index(self) -> int:
        return self.__camera_settings.selected_profile_index

    # used in camera control panel
    def open_configuration_interface(self, api_broker: typing.Any) -> None:
        self.__camera_settings.open_configuration_interface(api_broker)


class CameraFrameParameters:
    """Camera frame parameters."""

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        d: typing.Dict[str, typing.Any] = dict()
        assert not args or isinstance(args[0], dict)
        if isinstance(args[0], dict):
            d.update(args[0])
        d.update(kwargs)
        self.exposure_ms: float = d.pop("exposure_ms", 125)
        self.binning = d.pop("binning", 1)
        self.processing = d.pop("processing", None)
        self.integration_count = d.pop("integration_count", 1)
        self.__is_validated = False
        self.__active_masks = [Mask.from_dict(mask) if not isinstance(mask, Mask) else mask for mask in d.pop("active_masks", [])]
        self.__extra = d

    def __copy__(self) -> CameraFrameParameters:
        return copy.deepcopy(self)

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> CameraFrameParameters:
        deepcopy = self.__class__(copy.deepcopy(self.as_dict()))
        memo[id(self)] = deepcopy
        return deepcopy

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        d = {
            "exposure_ms": self.exposure_ms,
            "exposure": self.exposure_ms / 1000,
            "binning": self.binning,
            "processing": self.processing,
            "integration_count": self.integration_count,
            "active_masks": [mask.as_dict() for mask in self.active_masks],
        }
        d.update(self.__extra)
        return d

    def __getitem__(self, item: str) -> typing.Any:
        if hasattr(self, item):
            return getattr(self, item)
        else:
            return self.__extra[item]

    def __setitem__(self, key: str, value: typing.Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.__extra[key] = value

    @property
    def exposure(self) -> float:
        return self.exposure_ms / 1000

    @exposure.setter
    def exposure(self, value: float) -> None:
        self.exposure_ms = value * 1000

    @property
    def is_validated(self) -> bool:
        return self.__is_validated

    def _set_is_validated(self, is_validated: bool) -> None:
        self.__is_validated = is_validated

    @property
    def active_masks(self) -> typing.Sequence[Mask]:
        return self.__active_masks

    @active_masks.setter
    def active_masks(self, value: typing.Union[typing.Sequence[Mask], typing.List[typing.Dict[str, typing.Any]]]) -> None:
        self.__active_masks = [Mask.from_dict(mask) if not isinstance(mask, Mask) else mask for mask in value]


def crop_and_calibrate(uncropped_xdata: DataAndMetadata.DataAndMetadata, flyback_pixels: int,
                       scan_calibrations: typing.Optional[DataAndMetadata.CalibrationListType],
                       data_calibrations: DataAndMetadata.CalibrationListType,
                       data_intensity_calibration: typing.Optional[Calibration.Calibration],
                       metadata: DataAndMetadata.MetadataType) -> DataAndMetadata.DataAndMetadata:
    data_shape = uncropped_xdata.data_shape
    collection_shape = uncropped_xdata.collection_dimension_shape
    scan_calibrations = scan_calibrations or uncropped_xdata.collection_dimensional_calibrations
    uncropped_data = uncropped_xdata.data
    assert uncropped_data is not None, "Device data was None."
    if flyback_pixels > 0:
        data = uncropped_data.reshape(*collection_shape, *data_shape[len(collection_shape):])[:, flyback_pixels:collection_shape[1], :]
    else:
        data = uncropped_data.reshape(*collection_shape, *data_shape[len(collection_shape):])
    dimensional_calibrations = tuple(scan_calibrations) + tuple(data_calibrations)
    return DataAndMetadata.new_data_and_metadata(data, data_intensity_calibration,
                                                 dimensional_calibrations,
                                                 dict(metadata), None,
                                                 uncropped_xdata.data_descriptor, None,
                                                 None)


@dataclasses.dataclass
class CameraDeviceStreamPartialData:
    """Represents the data returned from get_next_data in the CameraDeviceStreamInterface."""
    valid_index: int
    is_complete: bool
    xdata: DataAndMetadata.DataAndMetadata


class CameraDeviceStreamInterface(typing.Protocol):
    """An interface to a camera device to help implementation in a data stream."""

    def prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs: typing.Any) -> None: ...

    def start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None: ...

    def finish_stream(self) -> None: ...

    def abort_stream(self) -> None: ...

    def get_next_data(self) -> typing.Optional[CameraDeviceStreamPartialData]:
        """Return the partial data; return None if nothing is available."""
        ...

    def continue_data(self, partial_data: typing.Optional[CameraDeviceStreamPartialData]) -> None: ...


class CameraDeviceSynchronizedStream(CameraDeviceStreamInterface):
    """An interface using the 'synchronized' style methods of the camera."""
    def __init__(self, camera_hardware_source: CameraHardwareSource, camera_frame_parameters: CameraFrameParameters, flyback_pixels: int = 0, additional_metadata: typing.Optional[DataAndMetadata.MetadataType] = None) -> None:
        self.__camera_hardware_source = camera_hardware_source
        self.__camera_frame_parameters = camera_frame_parameters
        self.__camera_frame_parameters_original: typing.Optional[CameraFrameParameters] = None
        self.__additional_metadata = additional_metadata or dict()
        self.__flyback_pixels = flyback_pixels
        self.__partial_data_info = typing.cast(PartialData, None)
        self.__slice: typing.List[slice] = list()

    def prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs: typing.Any) -> None:
        camera_frame_parameters = self.__camera_frame_parameters
        # clear the processing parameters in the original camera frame parameters.
        # processing will be configured based on the operator kwarg instead.
        camera_frame_parameters.processing = None
        camera_frame_parameters.active_masks = list()
        # get the operator.
        operator = typing.cast(Acquisition.DataStreamOperator, kwargs.get("operator", Acquisition.NullDataStreamOperator()))
        # rebuild the low level processing commands using the operator.
        if isinstance(operator, Acquisition.SumOperator):
            if operator.axis == 0:
                camera_frame_parameters.processing = "sum_project"
                operator.apply()
            else:
                camera_frame_parameters.processing = "sum_masked"
                operator.apply()
        elif isinstance(operator, Acquisition.StackedDataStreamOperator) and all(isinstance(o, Acquisition.SumOperator) for o in operator.operators):
            camera_frame_parameters.processing = "sum_masked"
            operator.apply()
        elif isinstance(operator, Acquisition.StackedDataStreamOperator) and all(isinstance(o, Acquisition.MaskedSumOperator) for o in operator.operators):
            camera_frame_parameters.processing = "sum_masked"
            camera_frame_parameters.active_masks = [typing.cast(Mask, typing.cast(Acquisition.MaskedSumOperator, o).mask) for o in operator.operators]
            operator.apply()
        # save original current camera frame parameters. these will be restored in finish stream.
        self.__camera_frame_parameters_original = self.__camera_hardware_source.get_current_frame_parameters()
        self.__camera_hardware_source.set_current_frame_parameters(camera_frame_parameters)
        collection_shape = (stream_args.slice_rect.height, stream_args.slice_rect.width + self.__flyback_pixels)  # includes flyback pixels
        self.__camera_hardware_source.acquire_synchronized_prepare(collection_shape)

    def start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        self.__slice = list(stream_args.slice)
        collection_shape = (stream_args.slice_rect.height, stream_args.slice_rect.width + self.__flyback_pixels)  # includes flyback pixels
        self.__partial_data_info = self.__camera_hardware_source.acquire_synchronized_begin(self.__camera_frame_parameters, collection_shape)

    def finish_stream(self) -> None:
        self.__camera_hardware_source.acquire_synchronized_end()
        # restore camera frame parameters.
        assert self.__camera_frame_parameters_original
        self.__camera_hardware_source.set_current_frame_parameters(self.__camera_frame_parameters_original)

    def abort_stream(self) -> None:
        self.__camera_hardware_source.acquire_sequence_cancel()

    def get_next_data(self) -> typing.Optional[CameraDeviceStreamPartialData]:
        valid_rows = self.__partial_data_info.valid_rows
        width = self.__slice[1].stop - self.__slice[1].start
        valid_count = self.__partial_data_info.valid_count if self.__partial_data_info.valid_count is not None else valid_rows * (width + self.__flyback_pixels)
        assert valid_count is not None
        if valid_count > 0:
            uncropped_xdata = self.__partial_data_info.xdata  # this returns the entire result data array
            is_complete = self.__partial_data_info.is_complete
            camera_metadata: typing.Dict[str, typing.Any] = dict()
            self.__camera_hardware_source.update_camera_properties(camera_metadata, self.__camera_frame_parameters)
            metadata = dict(copy.deepcopy(uncropped_xdata.metadata))
            # this is a hack to prevent some of the potentially misleading metadata
            # from getting saved into the synchronized data. while it is acceptable to
            # assume that the hardware_source properties will get copied to the final
            # metadata for now, camera implementers should be aware that this is likely
            # to change behavior in the future. please write tests if you make this
            # assumption so that they fail when this behavior is changed.
            metadata.setdefault("hardware_source", dict()).pop("frame_number", None)
            metadata.setdefault("hardware_source", dict()).pop("integration_count", None)
            metadata.setdefault("hardware_source", dict()).pop("valid_rows", None)
            metadata.setdefault("hardware_source", dict()).update(camera_metadata)
            metadata.update(copy.deepcopy(self.__additional_metadata))

            # TODO: this should be tracked elsewhere than here.
            if "scan" in metadata:
                metadata["scan"]["valid_rows"] = self.__slice[0].start + valid_count // (width + self.__flyback_pixels)

            # note: collection calibrations will be added in the collections stream
            data_calibrations = self.__camera_hardware_source.get_camera_calibrations(self.__camera_frame_parameters)
            data_intensity_calibration = self.__camera_hardware_source.get_camera_intensity_calibration(self.__camera_frame_parameters)
            cropped_xdata = crop_and_calibrate(uncropped_xdata, self.__flyback_pixels, None, data_calibrations, data_intensity_calibration, metadata)
            # convert the valid count to valid index. valid count includes flyback pixels. valid index does not.
            valid_index = valid_count // (width + self.__flyback_pixels) * width + max(0, valid_count % (width + self.__flyback_pixels) - self.__flyback_pixels)
            return CameraDeviceStreamPartialData(valid_index, is_complete, cropped_xdata)
        return None

    def continue_data(self, partial_data: typing.Optional[CameraDeviceStreamPartialData]) -> None:
        # acquire the next section and continue
        if not partial_data or not partial_data.is_complete:
            self.__partial_data_info = self.__camera_hardware_source.acquire_synchronized_continue()
        else:
            self.__partial_data_info = typing.cast(typing.Any, None)


class CameraDeviceSequenceStream(CameraDeviceStreamInterface):
    """An interface using the 'sequence' style methods of the camera."""
    def __init__(self, camera_hardware_source: CameraHardwareSource, camera_frame_parameters: CameraFrameParameters, additional_metadata: typing.Optional[DataAndMetadata.MetadataType] = None) -> None:
        self.__camera_hardware_source = camera_hardware_source
        self.__camera_frame_parameters = camera_frame_parameters
        self.__camera_frame_parameters_original: typing.Optional[CameraFrameParameters] = None
        self.__additional_metadata = additional_metadata or dict()
        self.__partial_data_info = typing.cast(PartialData, None)
        self.__slice: typing.List[slice] = list()

    def prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs: typing.Any) -> None:
        camera_frame_parameters = self.__camera_frame_parameters
        # clear the processing parameters in the original camera frame parameters.
        # processing will be configured based on the operator kwarg instead.
        camera_frame_parameters.processing = None
        camera_frame_parameters.active_masks = list()
        # get the operator.
        operator = typing.cast(Acquisition.DataStreamOperator, kwargs.get("operator", Acquisition.NullDataStreamOperator()))
        # rebuild the low level processing commands using the operator.
        if isinstance(operator, Acquisition.SumOperator):
            if operator.axis == 0:
                camera_frame_parameters.processing = "sum_project"
                operator.apply()
            else:
                camera_frame_parameters.processing = "sum_masked"
                operator.apply()
        elif isinstance(operator, Acquisition.StackedDataStreamOperator) and all(isinstance(o, Acquisition.SumOperator) for o in operator.operators):
            camera_frame_parameters.processing = "sum_masked"
            operator.apply()
        elif isinstance(operator, Acquisition.StackedDataStreamOperator) and all(isinstance(o, Acquisition.MaskedSumOperator) for o in operator.operators):
            camera_frame_parameters.processing = "sum_masked"
            camera_frame_parameters.active_masks = [typing.cast(Mask, typing.cast(Acquisition.MaskedSumOperator, o).mask) for o in operator.operators]
            operator.apply()
        # save original current camera frame parameters. these will be restored in finish stream.
        self.__camera_frame_parameters_original = self.__camera_hardware_source.get_current_frame_parameters()
        self.__camera_hardware_source.set_current_frame_parameters(camera_frame_parameters)
        self.__camera_hardware_source.acquire_sequence_prepare(stream_args.sequence_count)

    def start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        self.__slice = list(stream_args.slice)
        self.__partial_data_info = self.__camera_hardware_source.acquire_sequence_begin(self.__camera_frame_parameters, stream_args.sequence_count)

    def finish_stream(self) -> None:
        self.__camera_hardware_source.acquire_sequence_end()
        # restore camera frame parameters.
        assert self.__camera_frame_parameters_original
        self.__camera_hardware_source.set_current_frame_parameters(self.__camera_frame_parameters_original)

    def abort_stream(self) -> None:
        self.__camera_hardware_source.acquire_sequence_cancel()

    def get_next_data(self) -> typing.Optional[CameraDeviceStreamPartialData]:
        valid_count = self.__partial_data_info.valid_count
        assert valid_count is not None
        if valid_count > 0:
            uncropped_xdata = self.__partial_data_info.xdata  # this returns the entire result data array
            is_complete = self.__partial_data_info.is_complete
            camera_metadata: typing.Dict[str, typing.Any] = dict()
            self.__camera_hardware_source.update_camera_properties(camera_metadata, self.__camera_frame_parameters)
            metadata = dict(copy.deepcopy(uncropped_xdata.metadata))
            # this is a hack to prevent some of the potentially misleading metadata
            # from getting saved into the synchronized data. while it is acceptable to
            # assume that the hardware_source properties will get copied to the final
            # metadata for now, camera implementers should be aware that this is likely
            # to change behavior in the future. please write tests if you make this
            # assumption so that they fail when this behavior is changed.
            metadata.setdefault("hardware_source", dict()).pop("frame_number", None)
            metadata.setdefault("hardware_source", dict()).pop("integration_count", None)
            metadata.setdefault("hardware_source", dict()).pop("valid_rows", None)
            metadata.setdefault("hardware_source", dict()).update(camera_metadata)
            metadata.update(copy.deepcopy(self.__additional_metadata))
            # note: collection calibrations will be added in the collections stream
            data_calibrations = self.__camera_hardware_source.get_camera_calibrations(self.__camera_frame_parameters)
            data_intensity_calibration = self.__camera_hardware_source.get_camera_intensity_calibration(self.__camera_frame_parameters)
            cropped_xdata = crop_and_calibrate(uncropped_xdata, 0, None, data_calibrations, data_intensity_calibration, metadata)
            return CameraDeviceStreamPartialData(valid_count, is_complete, cropped_xdata)
        return None

    def continue_data(self, partial_data: typing.Optional[CameraDeviceStreamPartialData]) -> None:
        # acquire the next section and continue
        if not partial_data or not partial_data.is_complete:
            self.__partial_data_info = self.__camera_hardware_source.acquire_sequence_continue()
        else:
            self.__partial_data_info = typing.cast(typing.Any, None)


class CameraFrameDataStream(Acquisition.DataStream):
    """A data stream of individual camera frames, for use in synchronized/sequence acquisition.

    The data stream may utilize the sequence acquisition mode if the number of expected frames (passed in
    data stream args) is greater than one and the max count is unspecified. Otherwise, frames will be acquired
    one by one.
    """

    def __init__(self, camera_hardware_source: CameraHardwareSource,
                 camera_frame_parameters: CameraFrameParameters,
                 camera_device_stream_delegate: typing.Optional[CameraDeviceStreamInterface] = None) -> None:
        super().__init__()
        self.__camera_device_stream_interface = camera_device_stream_delegate
        self.__camera_hardware_source = camera_hardware_source
        self.__camera_frame_parameters = camera_frame_parameters
        self.__record_task = typing.cast(HardwareSource.RecordTask, None)  # used for single frames
        self.__record_count = 0
        self.__frame_shape = camera_hardware_source.get_expected_dimensions(camera_frame_parameters.binning)
        self.__channel = Acquisition.Channel(self.__camera_hardware_source.hardware_source_id)
        self.__last_index = 0
        self.__camera_sequence_overheads: typing.List[float] = list()
        self.camera_sequence_overhead = 0.0
        self.__start = 0.0
        self.__progress = 0.0

    def about_to_delete(self) -> None:
        if self.__record_task:
            self.__record_task = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return (self.__channel,)

    def get_info(self, channel: Acquisition.Channel) -> Acquisition.DataStreamInfo:
        data_shape = tuple(self.__camera_hardware_source.get_expected_dimensions(self.__camera_frame_parameters.binning))
        data_metadata = DataAndMetadata.DataMetadata((data_shape, numpy.float32))
        return Acquisition.DataStreamInfo(data_metadata, self.__camera_frame_parameters.exposure_ms / 1000)

    @property
    def progress(self) -> float:
        if not self.is_finished:
            return self.__progress
        return super().progress

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs: typing.Any) -> None:
        if stream_args.max_count == 1 or stream_args.shape == (1,):
            self.__camera_hardware_source.abort_playing(sync_timeout=5.0)
        else:
            assert self.__camera_device_stream_interface
            self.__start = time.perf_counter()
            self.__camera_device_stream_interface.prepare_stream(stream_args, **kwargs)
            self.__camera_sequence_overheads.append(time.perf_counter() - self.__start)
            while len(self.__camera_sequence_overheads) > 4:
                self.__camera_sequence_overheads.pop(0)

    def _start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        if stream_args.max_count == 1 or stream_args.shape == (1,):
            self.__record_task = HardwareSource.RecordTask(self.__camera_hardware_source, self.__camera_frame_parameters)
            self.__record_count = numpy.product(stream_args.shape, dtype=numpy.uint64)  # type: ignore
        else:
            assert self.__camera_device_stream_interface
            self.__last_index = 0
            self.__progress = 0.0
            self.__start = time.perf_counter()
            self.__camera_device_stream_interface.start_stream(stream_args)
            self.__camera_sequence_overheads.append(time.perf_counter() - self.__start)
            while len(self.__camera_sequence_overheads) > 4:
                self.__camera_sequence_overheads.pop(0)
            self.camera_sequence_overhead = sum(self.__camera_sequence_overheads) / (len(self.__camera_sequence_overheads) / 2)

    def _finish_stream(self) -> None:
        if self.__record_task:
            self.__record_task.grab()  # ensure grab is finished
            self.__record_task = typing.cast(typing.Any, None)
        else:
            assert self.__camera_device_stream_interface
            self.__camera_device_stream_interface.finish_stream()

    def _abort_stream(self) -> None:
        if self.__record_task:
            self.__camera_hardware_source.abort_recording()
        else:
            assert self.__camera_device_stream_interface
            self.__camera_device_stream_interface.abort_stream()

    def _send_next(self) -> None:
        if self.__record_task:
            if self.__record_task.is_finished:
                # data metadata describes the data being sent from this stream: shape, data type, and descriptor
                data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape))
                data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape, numpy.float32), data_descriptor=data_descriptor)
                source_data_slice: typing.Tuple[slice, ...] = (slice(0, self.__frame_shape[0]), slice(None))
                state = Acquisition.DataStreamStateEnum.COMPLETE
                xdatas = self.__record_task.grab()
                xdata = xdatas[0] if xdatas else None
                data = xdata.data if xdata else None
                assert data is not None
                data_stream_event = Acquisition.DataStreamEventArgs(self, self.__channel, data_metadata, data, None,
                                                                    source_data_slice, state)
                self.fire_data_available(data_stream_event)
                self.__record_count -= 1
                if self.__record_count > 0:
                    self.__record_task = HardwareSource.RecordTask(self.__camera_hardware_source, self.__camera_frame_parameters)
        else:
            assert self.__camera_device_stream_interface
            partial_data = self.__camera_device_stream_interface.get_next_data()
            if partial_data:
                valid_index = partial_data.valid_index
                xdata = partial_data.xdata
                start_index = self.__last_index
                stop_index = valid_index
                count = stop_index - start_index
                if count > 0:
                    data_channel_data = xdata.data
                    assert data_channel_data is not None
                    data_channel_data_metadata = xdata.data_metadata
                    data_channel_data_dtype = data_channel_data_metadata.data_dtype
                    assert data_channel_data_dtype is not None
                    channel = Acquisition.Channel(self.__camera_hardware_source.hardware_source_id)
                    data_metadata = DataAndMetadata.DataMetadata(
                        (tuple(data_channel_data_metadata.datum_dimension_shape), data_channel_data_dtype),
                        xdata.intensity_calibration,
                        xdata.dimensional_calibrations[-len(data_channel_data_metadata.datum_dimension_shape):],
                        xdata.metadata,
                        xdata.timestamp,
                        DataAndMetadata.DataDescriptor(False, 0, xdata.datum_dimension_count),
                        xdata.timezone,
                        xdata.timezone_offset)
                    # data_count is the total for the data provided by the child data stream. some data streams will
                    # provide a slice into a chunk of data representing the entire stream; whereas others will provide
                    # smaller chunks.
                    data_count = numpy.product(xdata.navigation_dimension_shape, dtype=numpy.int64)
                    data = data_channel_data.reshape((data_count,) + tuple(xdata.datum_dimension_shape))
                    source_slice = (slice(start_index, stop_index),) + (slice(None),) * len(xdata.datum_dimension_shape)
                    data_stream_event = Acquisition.DataStreamEventArgs(self,
                                                                        channel,
                                                                        data_metadata,
                                                                        data,
                                                                        count,
                                                                        source_slice,
                                                                        Acquisition.DataStreamStateEnum.COMPLETE)
                    self.fire_data_available(data_stream_event)
                    # total_count is the total for this entire stream.
                    total_count = numpy.product(self.get_info(channel).data_metadata.data_shape, dtype=numpy.int64).item()
                    self.__progress = valid_index / total_count
                self.__last_index = valid_index
            self.__camera_device_stream_interface.continue_data(partial_data)

    def wrap_in_sequence(self, length: int) -> Acquisition.DataStream:
        return make_sequence_data_stream(self.__camera_hardware_source, self.__camera_frame_parameters, length)
        # return Acquisition.SequenceDataStream(self, length)


def get_instrument_calibration_value(instrument_controller: InstrumentController, calibration_controls: typing.Mapping[str, typing.Union[str, int, float]], key: str) -> typing.Optional[typing.Union[float, str]]:
    if key + "_control" in calibration_controls:
        valid, value = instrument_controller.TryGetVal(typing.cast(str, calibration_controls[key + "_control"]))
        if valid:
            return value
    if key + "_value" in calibration_controls:
        return calibration_controls.get(key + "_value")
    return None


def build_calibration(instrument_controller: InstrumentController, calibration_controls: typing.Mapping[str, typing.Union[str, int, float]], prefix: str, relative_scale: float = 1, data_len: int = 0) -> Calibration.Calibration:
    scale = typing.cast(float, get_instrument_calibration_value(instrument_controller, calibration_controls, prefix + "_" + "scale"))
    scale = scale * relative_scale if scale is not None else scale
    offset = typing.cast(float, get_instrument_calibration_value(instrument_controller, calibration_controls, prefix + "_" + "offset"))
    units = typing.cast(str, get_instrument_calibration_value(instrument_controller, calibration_controls, prefix + "_" + "units"))
    if calibration_controls.get(prefix + "_origin_override", None) == "center" and scale is not None and data_len:
        offset = -scale * data_len * 0.5
    return Calibration.Calibration(offset, scale, units)


class CameraCalibrator(typing.Protocol):
    """A protocol for adding calibrations to data acquired from a camera device.

    The calibration of most cameras on a microscope will depend on other instrument parameters.
    """

    def get_signal_calibrations(self, frame_parameters: CameraFrameParameters, data_shape: typing.Sequence[int], **kwargs: typing.Any) -> typing.Sequence[Calibration.Calibration]: ...

    def get_intensity_calibration(self, camera_frame_parameters: CameraFrameParameters, **kwargs: typing.Any) -> Calibration.Calibration: ...

    def get_counts_per_electron(self, **kwargs: typing.Any) -> typing.Optional[float]: ...


class CalibrationControlsCalibrator(CameraCalibrator):
    """Calibrator v1. Uses calibration controls dictionary."""

    def __init__(self, instrument_controller: InstrumentController, camera_device: CameraDevice3) -> None:
        self.__instrument_controller = instrument_controller
        self.__camera_device = camera_device

    def get_signal_calibrations(self, frame_parameters: CameraFrameParameters, data_shape: typing.Sequence[int], **kwargs: typing.Any) -> typing.Sequence[Calibration.Calibration]:
        binning = frame_parameters.binning
        calibration_controls = self.__camera_device.calibration_controls
        if len(data_shape) == 2:
            y_calibration = build_calibration(self.__instrument_controller, calibration_controls, "y", binning, data_shape[1] if len(data_shape) > 1 else 0)
            x_calibration = build_calibration(self.__instrument_controller, calibration_controls, "x", binning, data_shape[0])
            return (y_calibration, x_calibration)
        else:
            x_calibration = build_calibration(self.__instrument_controller, calibration_controls, "x", binning, data_shape[0])
            return (x_calibration,)

    def get_intensity_calibration(self, camera_frame_parameters: CameraFrameParameters, **kwargs: typing.Any) -> Calibration.Calibration:
        instrument_controller = self.__instrument_controller
        return build_calibration(instrument_controller, self.__camera_device.calibration_controls, "intensity")

    def get_counts_per_electron(self, **kwargs: typing.Any) -> typing.Optional[float]:
        instrument_controller = self.__instrument_controller
        return typing.cast(typing.Optional[float], get_instrument_calibration_value(instrument_controller, self.__camera_device.calibration_controls, "counts_per_electron"))


class CalibrationControlsCalibrator2(CameraCalibrator):
    """Calibrator v2. Uses calibration config dictionary.

    The config mapping should have the following keys:

    (optional) calibrationModeIndexControl: name of control to use as index to other controls. substituted into <<n>> below if 1+
    (required) calibXScaleControl<<n>>: name of control to be used for x-scale; <<nn>> should be empty or 1+
    (required) calibXOffsetControl<<n>>: name of control to be used for x-offset; <<nn>> should be empty or 1+
    (required) calibXUnits<<n>>: x-unit string; <<nn>> should be empty or 1+
    (required) calibYScaleControl<<n>>: name of control to be used for y-scale; <<nn>> should be empty or 1+
    (required) calibYOffsetControl<<n>>: name of control to be used for y-offset; <<nn>> should be empty or 1+
    (required) calibYUnits<<n>>: y-unit string; <<nn>> should be empty or 1+
    (required) calibIntensityScaleControl<<n>>: name of control to be used for intensity-scale; <<nn>> should be empty or 1+
    (required) calibIntensityOffsetControl<<n>>: name of control to be used for intensity-offset; <<nn>> should be empty or 1+
    (required) calibIntensityUnits<<n>>: intensity-unit string; <<nn>> should be empty or 1+

    If calibration control for a particular index is empty, the appropriate default value will be used (scale=1.0, offset=0.0, units='').

    NOTE: counts_per_electron control is configured as before, using calibration_controls.
    """

    def __init__(self, instrument_controller: InstrumentController, camera_device: CameraDevice3, config: typing.Mapping[str, typing.Any]) -> None:
        self.__instrument_controller = instrument_controller
        self.__camera_device = camera_device
        self.__config = config

    def __construct_suffix(self) -> str:
        control = self.__config.get("calibrationModeIndexControl".lower(), None)
        if control:
            valid, value = self.__instrument_controller.TryGetVal(typing.cast(str, control))
            if valid and value:
                return str(int(value))
        return str()

    def __construct_calibration(self, prefix: str, suffix: str, relative_scale: float = 1.0, is_center_origin: bool = False, data_len: int = 0) -> Calibration.Calibration:
        scale = None
        scale_control = self.__config.get((prefix + "ScaleControl" + suffix).lower(), None)
        if scale_control:
            valid, value = self.__instrument_controller.TryGetVal(typing.cast(str, scale_control))
            if valid:
                scale = value
        offset = None
        offset_control = self.__config.get((prefix + "OffsetControl" + suffix).lower(), None)
        if offset_control:
            valid, value = self.__instrument_controller.TryGetVal(typing.cast(str, offset_control))
            if valid:
                offset = value
        units = self.__config.get((prefix + "Units" + suffix).lower(), None)
        scale = scale * relative_scale if scale is not None else scale
        if is_center_origin and scale is not None and data_len:
            offset = -scale * data_len * 0.5
        return Calibration.Calibration(offset, scale, units)

    def get_signal_calibrations(self, frame_parameters: CameraFrameParameters, data_shape: typing.Sequence[int], **kwargs: typing.Any) -> typing.Sequence[Calibration.Calibration]:
        binning = frame_parameters.binning
        is_center_origin = getattr(self.__camera_device, "camera_type", str()) != "eels"
        suffix = self.__construct_suffix()
        if len(data_shape) == 2:
            x_calibration = self.__construct_calibration("calibX", suffix, binning, is_center_origin, data_shape[1] if len(data_shape) > 1 else 0)
            y_calibration = self.__construct_calibration("calibY", suffix, binning, is_center_origin, data_shape[0])
            return (y_calibration, x_calibration)
        else:
            x_calibration = self.__construct_calibration("calibX", suffix, binning, is_center_origin, data_shape[0])
            return (x_calibration,)

    def get_intensity_calibration(self, camera_frame_parameters: CameraFrameParameters, **kwargs: typing.Any) -> Calibration.Calibration:
        suffix = self.__construct_suffix()
        return self.__construct_calibration("calibIntensity", suffix)

    def get_counts_per_electron(self, **kwargs: typing.Any) -> typing.Optional[float]:
        instrument_controller = self.__instrument_controller
        return typing.cast(typing.Optional[float], self.__get_instrument_calibration_value(instrument_controller, self.__camera_device.calibration_controls, "counts_per_electron"))

    def __get_instrument_calibration_value(self, instrument_controller: InstrumentController, calibration_controls: typing.Mapping[str, typing.Union[str, int, float]], key: str) -> typing.Optional[typing.Union[float, str]]:
        if key + "_control" in calibration_controls:
            valid, value = instrument_controller.TryGetVal(typing.cast(str, calibration_controls[key + "_control"]))
            if valid:
                return value
        if key + "_value" in calibration_controls:
            return calibration_controls.get(key + "_value")
        return None


def update_instrument_properties(stem_properties: typing.MutableMapping[str, typing.Any], instrument_controller: InstrumentController, camera: CameraDevice) -> None:
    if instrument_controller:
        # give the instrument controller opportunity to add properties
        if callable(getattr(instrument_controller, "get_autostem_properties", None)):
            try:
                autostem_properties = instrument_controller.get_autostem_properties()
                stem_properties.update(autostem_properties)
            except Exception as e:
                pass
        # give the instrument controller opportunity to update metadata groups specified by the camera
        if hasattr(camera, "acquisition_metatdata_groups"):
            acquisition_metatdata_groups = getattr(camera, "acquisition_metatdata_groups")
            instrument_controller.apply_metadata_groups(stem_properties, acquisition_metatdata_groups)


def update_camera_properties(properties: typing.MutableMapping[str, typing.Any],
                             frame_parameters: CameraFrameParameters, hardware_source_id: str, display_name: str,
                             signal_type: typing.Optional[str] = None) -> None:
    properties["hardware_source_id"] = hardware_source_id
    properties["hardware_source_name"] = display_name
    properties["exposure"] = frame_parameters.exposure_ms / 1000.0
    properties["binning"] = frame_parameters.binning
    if signal_type:
        properties["signal_type"] = signal_type


class ChannelDataStream(Acquisition.ContainerDataStream):
    def __init__(self, data_stream: Acquisition.DataStream, camera_data_channel: typing.Optional[SynchronizedDataChannelInterface], channel: typing.Optional[Acquisition.Channel] = None):
        super().__init__(data_stream)
        self.__camera_data_channel = camera_data_channel
        self.__channel = channel
        self.__dst_index = 0

    def _start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        super()._start_stream(stream_args)
        self.__dst_index = 0

    def _fire_data_available(self, data_stream_event: Acquisition.DataStreamEventArgs) -> None:
        if self.__channel is None or self.__channel == data_stream_event.channel:
            data_channel_state = "complete" if data_stream_event.state == Acquisition.DataStreamStateEnum.COMPLETE else "partial"
            assert data_stream_event.count is None
            assert not data_stream_event.data_metadata.is_sequence
            assert data_stream_event.source_slice[1].start is None or data_stream_event.source_slice[1].start == 0
            if data_stream_event.data_metadata.is_navigable:
                collection_shape = data_stream_event.data_metadata.navigation_dimension_shape
                data_shape = Geometry.IntSize(h=collection_shape[0], w=collection_shape[1])
                data = data_stream_event.source_data
                dimensional_calibrations = tuple(data_stream_event.data_metadata.dimensional_calibrations)
                data_descriptor = data_stream_event.data_metadata.data_descriptor
            else:
                collection_shape = data_stream_event.data_metadata.datum_dimension_shape
                data_shape = Geometry.IntSize(h=collection_shape[0], w=collection_shape[1])
                data = data_stream_event.source_data[..., numpy.newaxis]
                dimensional_calibrations = tuple(data_stream_event.data_metadata.dimensional_calibrations) + (Calibration.Calibration(),)
                data_descriptor = DataAndMetadata.DataDescriptor(False, data_stream_event.data_metadata.data_descriptor.datum_dimension_count, 1)
            start_index = Acquisition.ravel_slice_start(data_stream_event.source_slice[0:2], data_shape.as_tuple())
            stop_index = Acquisition.ravel_slice_stop(data_stream_event.source_slice[0:2], data_shape.as_tuple())
            src_row = data_stream_event.source_slice[0].start
            width = data_shape[1]
            # handle the case where a partial row has been acquired; try to finish the row
            if self.__dst_index % width:
                length = min(width - self.__dst_index % width, stop_index - start_index)
                if self.__camera_data_channel:
                    source_data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                                     data_stream_event.data_metadata.intensity_calibration,
                                                                                     dimensional_calibrations,
                                                                                     data_stream_event.data_metadata.metadata,
                                                                                     data_stream_event.data_metadata.timestamp,
                                                                                     data_descriptor,
                                                                                     data_stream_event.data_metadata.timezone,
                                                                                     data_stream_event.data_metadata.timezone_offset)
                    dst_rect = Geometry.IntRect.from_tlhw(self.__dst_index // width, self.__dst_index % width, 1, length)
                    src_rect = Geometry.IntRect.from_tlhw(src_row, 0, 1, length)
                    data_channel_view_id = None
                    self.__camera_data_channel.update(source_data_and_metadata, data_channel_state, data_shape, dst_rect, src_rect, data_channel_view_id)
                src_row += 1
                start_index += length
                self.__dst_index += length
            # handle the case where one or more full rows remain; send full rows.
            if stop_index - start_index >= width:
                assert data_stream_event.data_metadata.data_descriptor.collection_dimension_count == 2 or (data_stream_event.data_metadata.data_descriptor.collection_dimension_count == 0 and data_stream_event.data_metadata.data_descriptor.datum_dimension_count == 2)
                assert data_stream_event.source_slice[1].stop is None  # or data_stream_event.source_slice[1].stop == data_shape.width
                height = (stop_index - start_index) // width
                if self.__camera_data_channel:
                    source_data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                                     data_stream_event.data_metadata.intensity_calibration,
                                                                                     dimensional_calibrations,
                                                                                     data_stream_event.data_metadata.metadata,
                                                                                     data_stream_event.data_metadata.timestamp,
                                                                                     data_descriptor,
                                                                                     data_stream_event.data_metadata.timezone,
                                                                                     data_stream_event.data_metadata.timezone_offset)
                    dst_rect = Geometry.IntRect.from_tlhw(self.__dst_index // width, 0, height, width)
                    src_rect = Geometry.IntRect.from_tlhw(src_row, 0, height, width)
                    data_channel_view_id = None
                    self.__camera_data_channel.update(source_data_and_metadata, data_channel_state, data_shape, dst_rect, src_rect, data_channel_view_id)
                src_row += height
                start_index += height * width
                self.__dst_index += height * width
            # handle remaining data as a partial row.
            if start_index < stop_index:
                length = stop_index - start_index
                if self.__camera_data_channel:
                    source_data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                                     data_stream_event.data_metadata.intensity_calibration,
                                                                                     dimensional_calibrations,
                                                                                     data_stream_event.data_metadata.metadata,
                                                                                     data_stream_event.data_metadata.timestamp,
                                                                                     data_descriptor,
                                                                                     data_stream_event.data_metadata.timezone,
                                                                                     data_stream_event.data_metadata.timezone_offset)
                    dst_rect = Geometry.IntRect.from_tlhw(self.__dst_index // width, 0, 1, length)
                    src_rect = Geometry.IntRect.from_tlhw(src_row, start_index % width, 1, length)
                    data_channel_view_id = None
                    self.__camera_data_channel.update(source_data_and_metadata, data_channel_state, data_shape, dst_rect, src_rect, data_channel_view_id)
                src_row += 1
                start_index += length
                self.__dst_index += length

        super()._fire_data_available(data_stream_event)


class SynchronizedDataChannelInterface:
    """Update the data.

    This method is always called with a collection of 1d or 2d data.
    """
    def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, data_shape: Geometry.IntSize, dest_sub_area: Geometry.IntRect, sub_area: Geometry.IntRect, view_id: typing.Optional[str]) -> None: ...


def make_sequence_data_stream(
        camera_hardware_source: CameraHardwareSource,
        camera_frame_parameters: CameraFrameParameters,
        count: int,
        camera_data_channel: typing.Optional[SynchronizedDataChannelInterface] = None,
        include_raw: bool = True,
        include_summed: bool = False) -> Acquisition.DataStream:

    instrument_controller = typing.cast(InstrumentController, Registry.get_component("stem_controller"))

    instrument_metadata: typing.Dict[str, typing.Any] = dict()
    update_instrument_properties(instrument_metadata, instrument_controller, camera_hardware_source.camera)

    additional_camera_metadata = {"instrument": copy.deepcopy(instrument_metadata)}
    camera_data_stream = CameraFrameDataStream(camera_hardware_source, camera_frame_parameters,
                                                CameraDeviceSequenceStream(
                                                    camera_hardware_source,
                                                    camera_frame_parameters,
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
    sequence: Acquisition.DataStream = Acquisition.SequenceDataStream(processed_camera_data_stream, count)
    if camera_frame_parameters.processing == "sum_masked":
        active_masks = camera_frame_parameters.active_masks
        if active_masks and len(active_masks) > 1:
            sequence = Acquisition.FramedDataStream(sequence, operator=Acquisition.MoveAxisDataStreamOperator(
                processed_camera_data_stream.channels[0]))
    # SynchronizedDataStream saves and restores the scan parameters; also enters/exits synchronized state
    if count > 1:
        assert include_raw or include_summed
        if include_raw and include_summed:
            # AccumulateDataStream sums the successive frames in each channel
            monitor = Acquisition.MonitorDataStream(sequence, "raw")
            sequence = Acquisition.AccumulatedDataStream(sequence)
            sequence = Acquisition.CombinedDataStream([sequence, monitor])
        elif include_summed:
            sequence = Acquisition.AccumulatedDataStream(sequence)
        # include_raw is the default behavior
    # the optional ChannelDataStream updates the camera data channel for the stream matching 999
    data_stream: Acquisition.DataStream
    if camera_data_channel:
        data_stream = ChannelDataStream(sequence, camera_data_channel, Acquisition.Channel(camera_hardware_source.hardware_source_id))
    else:
        data_stream = sequence
    # return the top level stream
    return data_stream


_component_registered_listener = None
_component_unregistered_listener = None

def run(configuration_location: pathlib.Path) -> None:
    def component_registered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
        if "camera_module" in component_types:
            camera_module = component
            instrument_controller_id = typing.cast(typing.Optional[str], getattr(camera_module, "instrument_controller_id", None))
            # TODO: remove next line when backwards compatibility no longer needed
            instrument_controller_id = instrument_controller_id or typing.cast(typing.Optional[str], getattr(camera_module, "stem_controller_id", None))
            # grab the settings and camera panel info from the camera module
            camera_settings = camera_module.camera_settings
            camera_device = camera_module.camera_device
            camera_panel_type = getattr(camera_module, "camera_panel_type", None)  # a replacement camera panel
            camera_panel_delegate_type = getattr(camera_module, "camera_panel_delegate_type", None)  # a delegate for the default camera panel
            try:
                camera_version = getattr(camera_device, "camera_version", 2)
                camera_hardware_source: CameraHardwareSource
                if camera_version == 3:
                    camera_hardware_source = CameraHardwareSource3(instrument_controller_id, camera_device, camera_settings, configuration_location, camera_panel_type, camera_panel_delegate_type)
                else:
                    camera_hardware_source = CameraHardwareSource2(instrument_controller_id, camera_device, camera_settings, configuration_location, camera_panel_type, camera_panel_delegate_type)
                if hasattr(camera_module, "priority"):
                    setattr(camera_hardware_source, "priority", getattr(camera_module, "priority"))
                component_types = {"hardware_source", "camera_hardware_source"}.union({camera_device.camera_type + "_camera_hardware_source"})
                Registry.register_component(camera_hardware_source, component_types)
                HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)
                camera_module.hardware_source = camera_hardware_source
            except Exception as e:
                camera_id = str(getattr(getattr(component, "camera_device", None), "camera_id", None))
                camera_id = camera_id or "UNKNOWN"
                logging.info("Camera Plug-in '" + camera_id + "' exception during initialization.")
                logging.info(traceback.format_exc())

    def component_unregistered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
        if "camera_module" in component_types:
            camera_hardware_source = component.hardware_source
            Registry.unregister_component(camera_hardware_source)
            HardwareSource.HardwareSourceManager().unregister_hardware_source(camera_hardware_source)

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    for component in Registry.get_components_by_type("camera_module"):
        component_registered(component, {"camera_module"})
