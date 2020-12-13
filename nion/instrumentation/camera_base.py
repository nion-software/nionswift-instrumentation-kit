from __future__ import annotations

# standard libraries
import abc
import asyncio
import collections
import copy
import datetime
import gettext
import json
import logging
import os
import pathlib
import typing
import traceback
import uuid

# typing
# None

# third party libraries
import numpy

# local libraries
from nion.data import Calibration
from nion.data import Core
from nion.data import DataAndMetadata
from nion.swift.model import HardwareSource
from nion.swift.model import ImportExportManager
from nion.swift.model import Utility
from nion.swift.model import Graphics
from nion.utils import Event
from nion.utils import Process
from nion.utils import Registry


_ = gettext.gettext


class Camera(abc.ABC):
    """DEPRECATED. Here for backwards compatibility.

    The method implementations only exist since classes derived from this base class may have assumed these methods
    would be implemented. Methods marked as abstract have been removed since they must have already been implemented in
    any class derived from this one.
    """

    def set_integration_count(self, integration_count: int, mode_id: str) -> None:
        pass

    def get_acquire_sequence_metrics(self, frame_parameters: typing.Dict) -> typing.Dict:
        return dict()

    def acquire_sequence_prepare(self, n: int) -> None:
        pass

    def acquire_sequence(self, n: int) -> typing.Optional[typing.Dict]:
        return None

    def show_config_window(self) -> None:
        pass

    def show_configuration_dialog(self, api_broker) -> None:
        pass

    def start_monitor(self) -> None:
        pass


class CameraDevice(abc.ABC):
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

    def set_dark_image(self, data: numpy.ndarray, exposure: float, bin: int, t: int, l: int, b: int, r: int) -> bool:
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

    def set_gain_image(self, data: numpy.ndarray, voltage: int, bin: int) -> bool:
        """Set the gain image for the given voltage and binning."""
        return False

    def remove_all_gain_images(self) -> None:
        """Remove all gain reference images."""
        pass

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

    @abc.abstractmethod
    def set_frame_parameters(self, frame_parameters: typing.Any) -> None:
        """Set the pending frame parameters (exposure_ms, binning, processing, integration_count).

        processing and integration_count are optional, in which case they are handled at a higher level.
        """
        ...

    @property
    @abc.abstractmethod
    def calibration_controls(self) -> dict:
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

    # @property
    # @abc.abstractmethod
    # def acquisition_metatdata_groups(self) -> typing.Sequence[typing.Tuple[typing.Sequence[str], str]]):
        """Return metadata groups to be read from instrument controller and stored in metadata.

        Metadata groups are a list of tuples where the first item is the destination path and the second item is the
        control group name.

        This method is optional. Default is to return an empty list.

        Note: for backward compatibility, if the destination root is 'autostem', it is skipped and the rest of the path
        is used. this can be removed in the future.
        """

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

    def get_acquire_sequence_metrics(self, frame_parameters: typing.Dict) -> typing.Dict:
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

    # def acquire_synchronized_prepare(self, data_shape, **kwargs) -> None:
        """Prepare for synchronized acquisition.

        THIS METHOD IS DEPRECATED TO ADD SUPPORT FOR PARTIAL ACQUISITION.

        Default implementation calls acquire_sequence_prepare.
        """
        # pass

    # def acquire_synchronized(self, data_shape, **kwargs) -> typing.Optional[typing.Dict]:
        """Acquire a sequence of images with the data_shape. Return a single data element with two dimensions n x data_shape.

        THIS METHOD IS DEPRECATED TO ADD SUPPORT FOR PARTIAL ACQUISITION.

        Default implementation calls acquire_sequence.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).

        Return None for cancellation.

        Raise exception for error.
        """
        # pass

    # def acquire_synchronized_begin(self, camera_frame_parameters: typing.Mapping, scan_shape: typing.Tuple[int, ...], **kwargs) -> PartialData: ...
        """Begin synchronized acquire.

        The camera device can allocate memory to accommodate the scan_shape and begin acquisition immediately.

        The camera device will typically populate the PartialData with the data array (xdata), is_complete set to
        False, is_canceled set to False, and valid_rows set to 0 or valid_row_range set to 0, 0.

        Returns PartialData.
        """

    # def acquire_synchronized_continue(self, *, update_period: float = 1.0, **kwargs) -> PartialData: ...
        """Continue synchronized acquire.

        The camera device should wait up to update_period seconds for data and populate PartialData with data and
        information about the acquisition.

        The valid_rows field of PartialData indicates how many rows are valid in xdata. The grab_synchronized method
        will keep track of the last valid row and copy data from the last valid row to valid_rows into the acquisition
        data and then update last valid row with valid_rows.

        The xdata field of PartialData must be filled with the data allocated during acquire_synchronized_begin. The
        section of data up to valid_rows must remain valid until the last Python reference to xdata is released.

        Returns PartialData.
        """

    # def acquire_synchronized_end(self, **kwargs) -> None: ...
        """Clean up synchronized acquire.

        The camera device can clean up anything internal that was required for acquisition.

        The memory returned during acquire_synchronized_begin or acquire_synchronized_continue must remain valid until
        the last Python reference to that memory is released.
        """

    # def acquire_sequence_prepare(self, n: int) -> None:
        """Prepare for acquire_sequence."""
        # pass

    # def acquire_sequence(self, n: int) -> typing.Optional[typing.Dict]:
        """Acquire a sequence of n images. Return a single data element with two dimensions n x h, w.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).

        Return None for cancellation.

        Raise exception for error.
        """
        # return None

    def acquire_sequence_cancel(self) -> None:
        """Request to cancel a sequence acquisition.

        Pending acquire_sequence calls should return None to indicate cancellation.
        """
        pass

    def show_config_window(self) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def show_configuration_dialog(self, api_broker) -> None:
        """Show a configuration dialog, if needed. Dialog can be modal or modeless."""
        pass

    def start_monitor(self) -> None:
        """Show a monitor dialog, if needed. Dialog can be modal or modeless."""
        pass


class InstrumentController(abc.ABC):

    @abc.abstractmethod
    def TryGetVal(self, s: str) -> (bool, float): ...

    @abc.abstractmethod
    def get_value(self, value_id: str, default_value: float=None) -> typing.Optional[float]: ...

    @abc.abstractmethod
    def set_value(self, value_id: str, value: float) -> None: ...

    def get_autostem_properties(self) -> typing.Dict: return dict()

    def apply_metadata_groups(self, properties: typing.MutableMapping, metatdata_groups: typing.Sequence[typing.Tuple[typing.Sequence[str], str]]) -> None: pass

    def handle_shift_click(self, **kwargs) -> None: pass

    def handle_tilt_click(self, **kwargs) -> None: pass


class CameraAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, instrument_controller: InstrumentController, hardware_source_id, is_continuous: bool, camera: CameraDevice, camera_settings: "CameraSettings", camera_category: str, signal_type: typing.Optional[str], frame_parameters, display_name):
        super().__init__(is_continuous)
        self.__instrument_controller = instrument_controller
        self.hardware_source_id = hardware_source_id
        self.is_continuous = is_continuous
        self.__camera = camera
        self.__camera_settings = camera_settings
        self.__camera_category = camera_category
        self.__signal_type = signal_type
        self.__display_name = display_name
        self.__frame_parameters = None
        self.__pending_frame_parameters = self.__camera_settings.get_frame_parameters_from_dict(frame_parameters)

    def set_frame_parameters(self, frame_parameters):
        self.__pending_frame_parameters = self.__camera_settings.get_frame_parameters_from_dict(frame_parameters)
        self.__activate_frame_parameters()

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
        binning = frame_parameters.binning
        integration_count = frame_parameters.integration_count if frame_parameters.integration_count else 1
        cumulative_frame_count = 0  # used for integration_count
        cumulative_data = None
        _data_element = None  # avoid use-before-set warning
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
        if self.__stop_after_acquire:
            self.__camera.stop_live()
        # camera data is always assumed to be full frame, otherwise deal with subarea 1d and 2d
        data_element = dict()
        data_element["metadata"] = dict()
        data_element["metadata"]["hardware_source"] = copy.deepcopy(_data_element["properties"])
        data_element["data"] = cumulative_data
        data_element["version"] = 1
        data_element["state"] = "complete"
        data_element["timestamp"] = _data_element.get("timestamp", datetime.datetime.utcnow())
        update_spatial_calibrations(data_element, self.__instrument_controller, self.__camera, self.__camera_category, cumulative_data.shape, binning, binning)
        update_intensity_calibration(data_element, self.__instrument_controller, self.__camera)
        instrument_metadata = dict()
        update_instrument_properties(instrument_metadata, self.__instrument_controller, self.__camera)
        if instrument_metadata:
            data_element["metadata"].setdefault("instrument", dict()).update(instrument_metadata)
        update_camera_properties(data_element["metadata"]["hardware_source"], frame_parameters, self.hardware_source_id, self.__display_name, data_element.get("signal_type", self.__signal_type))
        data_element["metadata"]["hardware_source"]["valid_rows"] = cumulative_data.shape[0]
        data_element["metadata"]["hardware_source"]["frame_index"] = data_element["metadata"]["hardware_source"]["frame_number"]
        data_element["metadata"]["hardware_source"]["integration_count"] = cumulative_frame_count
        return [data_element]

    def __activate_frame_parameters(self):
        self.__frame_parameters = self.frame_parameters
        self.__pending_frame_parameters = None
        self.__camera.set_frame_parameters(self.__frame_parameters)


class Mask:
    def __init__(self):
        self._layers = list()
        self.name = None
        self.uuid = uuid.uuid4()

    def add_layer(self, graphic: Graphics.Graphic, value: typing.Union[float, str], inverted: bool = False):
        self._layers.append({"value": value, "inverted": inverted, "graphic_dict": graphic.mime_data_dict()})

    def to_dict(self):
        return {"name": self.name, "uuid": str(self.uuid), "layers": self._layers}

    def as_dict(self):
        return self.to_dict()

    @classmethod
    def from_dict(cls, mask_description: typing.Mapping) -> "Mask":
        mask = cls()
        mask.name = mask_description["name"]
        if "uuid" in mask_description:
            mask.uuid = uuid.UUID(mask_description["uuid"])
        mask._layers = mask_description["layers"]
        return mask

    def get_mask_array(self, data_shape: typing.Sequence[int]) -> numpy.ndarray:
        if len(self._layers) == 0:
            return numpy.ones(data_shape)
        mask = numpy.zeros(data_shape)
        for layer in self._layers:
            graphic_dict = layer["graphic_dict"]
            value = layer["value"]
            inverted = layer.get("inverted", False)
            graphic = Graphics.factory(lambda t: graphic_dict["type"])
            graphic.read_from_mime_data(graphic_dict)
            if graphic:
                part_mask = graphic.get_mask(data_shape).astype(numpy.bool)
                if inverted:
                    part_mask = numpy.logical_not(part_mask)
                if value == "grad_x":
                    if hasattr(graphic, "center"):
                        center = graphic.center
                    else:
                        center = (0.5, 0.5)
                    center_coords = (center[0] * data_shape[0], center[1] * data_shape[1])
                    grad = numpy.tile(numpy.linspace(-center_coords[1], center_coords[1], data_shape[1]), (data_shape[0], 1))
                    mask[part_mask] = grad[part_mask]
                elif value == "grad_y":
                    if hasattr(graphic, "center"):
                        center = graphic.center
                    else:
                        center = (0.5, 0.5)
                    center_coords = (center[0] * data_shape[0], center[1] * data_shape[1])
                    grad = numpy.tile(numpy.linspace(-center_coords[0], center_coords[0], data_shape[0]), (data_shape[1], 1)).T
                    mask[part_mask] = grad[part_mask]
                else:
                    mask[part_mask] = value
        return mask

    def copy(self) -> 'Mask':
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

    def __init__(self):
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

    def close(self):
        pass

    def initialize(self, configuration_location: pathlib.Path = None, event_loop: asyncio.AbstractEventLoop = None, **kwargs):
        pass

    def apply_settings(self, settings_dict: typing.Dict) -> None:
        """Initialize the settings with the settings_dict."""
        pass

    def get_frame_parameters_from_dict(self, d: typing.Mapping):
        pass

    def set_current_frame_parameters(self, frame_parameters) -> None:
        """Set the current frame parameters.

        Fire the current frame parameters changed event and optionally the settings changed event.
        """
        self.current_frame_parameters_changed_event.fire(frame_parameters)

    def get_current_frame_parameters(self):
        """Get the current frame parameters."""
        return None

    def set_record_frame_parameters(self, frame_parameters) -> None:
        """Set the record frame parameters.

        Fire the record frame parameters changed event and optionally the settings changed event.
        """
        self.record_frame_parameters_changed_event.fire(frame_parameters)

    def get_record_frame_parameters(self):
        """Get the record frame parameters."""
        return None

    def set_frame_parameters(self, profile_index: int, frame_parameters) -> None:
        """Set the frame parameters with the settings index and fire the frame parameters changed event.

        If the settings index matches the current settings index, call set current frame parameters.

        If the settings index matches the record settings index, call set record frame parameters.
        """
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def get_frame_parameters(self, profile_index: int):
        """Get the frame parameters for the settings index."""
        return None

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

    def open_configuration_interface(self, api_broker) -> None:
        pass

    def open_monitor(self) -> None:
        pass


class CameraHardwareSource(HardwareSource.HardwareSource):

    def __init__(self, instrument_controller_id: str, camera: CameraDevice, camera_settings: CameraSettings, configuration_location: pathlib.Path, camera_panel_type: typing.Optional[str], camera_panel_delegate_type: typing.Optional[str] = None):
        super().__init__(camera.camera_id, camera.camera_name)

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

            def settings_changed(settings_dict: typing.Dict) -> None:
                # atomically overwrite
                temp_filepath = config_file.with_suffix(".temp")
                with open(temp_filepath, "w") as fp:
                    json.dump(settings_dict, fp, skipkeys=True, indent=4)
                os.replace(temp_filepath, config_file)

            self.__settings_changed_event_listener = self.__camera_settings.settings_changed_event.listen(settings_changed)

        self.__instrument_controller_id = instrument_controller_id
        self.__instrument_controller = None

        self.__camera = camera
        self.__camera_category = camera.camera_type
        # signal type falls back to camera category if camera category is "eels" or "ronchigram". this is only for
        # backward compatibility. new camera instances should define signal_type directly.
        self.__signal_type = getattr(camera, "signal_type", self.__camera_category if self.__camera_category in ("eels", "ronchigram") else None)
        self.processor = None

        # configure the features. putting the features into this object is for convenience of access. the features
        # should not be considered as part of this class. instead, the features should be thought of as being stored
        # here as a convenient location where the UI has access to them.
        self.features = dict()
        self.features["is_camera"] = True
        self.features["has_monitor"] = True
        if camera_panel_type:
            self.features["camera_panel_type"] = camera_panel_type
        if camera_panel_delegate_type:
            self.features["camera_panel_delegate_type"] = camera_panel_delegate_type
        if self.__camera_category.lower() == "ronchigram":
            self.features["is_ronchigram_camera"] = True
        if self.__camera_category.lower() == "eels":
            self.features["is_eels_camera"] = True
        if getattr(camera, "has_processed_channel", True if self.__camera_category.lower() == "eels" else False):
            self.processor = HardwareSource.SumProcessor(((0.25, 0.0), (0.5, 1.0)))
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

        self.__frame_parameters = self.__camera_settings.get_frame_parameters_from_dict(self.__camera_settings.get_current_frame_parameters())
        self.__record_parameters = self.__camera_settings.get_frame_parameters_from_dict(self.__camera_settings.get_record_frame_parameters())

        self.__acquisition_task = None

        # the periodic logger function retrieves any log messages from the camera. it is called during
        # __handle_log_messages_event. any messages are sent out on the log_messages_event.
        periodic_logger_fn = getattr(self.__camera, "periodic_logger_fn", None)
        self.__periodic_logger_fn = periodic_logger_fn if callable(periodic_logger_fn) else None

    def close(self):
        Process.close_event_loop(self.__event_loop)
        self.__event_loop = None
        self.__periodic_logger_fn = None
        super().close()
        if self.__settings_changed_event_listener:
            self.__settings_changed_event_listener.close()
            self.__settings_changed_event_listener = None
        self.__profile_changed_event_listener.close()
        self.__profile_changed_event_listener = None
        self.__frame_parameters_changed_event_listener.close()
        self.__frame_parameters_changed_event_listener = None
        self.__current_frame_parameters_changed_event_listener.close()
        self.__current_frame_parameters_changed_event_listener = None
        self.__record_frame_parameters_changed_event_listener.close()
        self.__record_frame_parameters_changed_event_listener = None
        self.__camera_settings.close()
        self.__camera_settings = None
        camera_close_method = getattr(self.__camera, "close", None)
        if callable(camera_close_method):
            camera_close_method()
        self.__camera = None

    def periodic(self):
        self.__event_loop.stop()
        self.__event_loop.run_forever()
        self.__handle_log_messages_event()

    def __get_instrument_controller(self) -> InstrumentController:
        if not self.__instrument_controller and self.__instrument_controller_id:
            self.__instrument_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id(self.__instrument_controller_id)
        if not self.__instrument_controller and not self.__instrument_controller_id:
            self.__instrument_controller = Registry.get_component("instrument_controller")
        if not self.__instrument_controller and not self.__instrument_controller_id:
            self.__instrument_controller = Registry.get_component("stem_controller")
        if not self.__instrument_controller:
            print(f"Instrument Controller ({self.__instrument_controller_id}) for ({self.hardware_source_id}) not found. Using proxy.")
            from nion.instrumentation import stem_controller
            self.__instrument_controller = self.__instrument_controller or stem_controller.STEMController()
        return self.__instrument_controller

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
        self.acquire_sequence_prepare(count)
        return True

    def grab_sequence(self, count: int, **kwargs) -> typing.Optional[typing.List[DataAndMetadata.DataAndMetadata]]:
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

    def grab_sequence_abort(self) -> None:
        self.acquire_sequence_cancel()

    def grab_sequence_get_progress(self) -> typing.Optional[float]:
        return None

    def grab_buffer(self, count: int, *, start: int=None, **kwargs) -> typing.Optional[typing.List[typing.List[DataAndMetadata.DataAndMetadata]]]:
        return None

    def make_reference_key(self, **kwargs) -> str:
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
        return CameraAcquisitionTask(self.__get_instrument_controller(), self.hardware_source_id, True, self.__camera, self.__camera_settings, self.__camera_category, self.__signal_type, self.__frame_parameters, self.display_name)

    def _view_task_updated(self, view_task):
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__record_parameters is not None
        return CameraAcquisitionTask(self.__get_instrument_controller(), self.hardware_source_id, False, self.__camera, self.__camera_settings, self.__camera_category, self.__signal_type, self.__record_parameters, self.display_name)

    class PartialData:
        def __init__(self, xdata: numpy.ndarray, is_complete: bool, is_canceled: bool, valid_rows: typing.Optional[int] = None):
            self.xdata = xdata
            self.is_complete = is_complete
            self.is_canceled = is_canceled
            self.valid_rows = valid_rows

    def acquire_synchronized_begin(self, camera_frame_parameters: typing.Mapping, scan_shape: typing.Tuple[int, ...]) -> PartialData:
        if callable(getattr(self.__camera, "acquire_synchronized_begin", None)):
            return self.__camera.acquire_synchronized_begin(camera_frame_parameters, scan_shape)
        else:
            data_elements = self.acquire_synchronized(scan_shape)
            if len(data_elements) > 0:
                data_elements[0]["data"] = data_elements[0]["data"].reshape(*scan_shape, *(data_elements[0]["data"].shape[1:]))
                xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_elements[0])
                return CameraHardwareSource.PartialData(xdata, True, False, scan_shape[0])
        return CameraHardwareSource.PartialData(None, True, True, 0)


    def acquire_synchronized_continue(self, *, update_period: float = 1.0) -> PartialData:
        if callable(getattr(self.__camera, "acquire_synchronized_continue", None)):
            return self.__camera.acquire_synchronized_continue(update_period=update_period)
        return CameraHardwareSource.PartialData(None, True, True, 0)

    def acquire_synchronized_end(self) -> None:
        if callable(getattr(self.__camera, "acquire_synchronized_end", None)):
            self.__camera.acquire_synchronized_end()

    def acquire_synchronized_prepare(self, data_shape, **kwargs) -> None:
        if callable(getattr(self.__camera, "acquire_synchronized_prepare", None)):
            frame_parameters = self.get_current_frame_parameters()
            self.__camera.set_frame_parameters(frame_parameters)
            self.__camera.acquire_synchronized_prepare(data_shape, **kwargs)
        else:
            self.acquire_sequence_prepare(int(numpy.product(data_shape)))

    def acquire_synchronized(self, data_shape, **kwargs) -> typing.Sequence[typing.Dict]:
        if callable(getattr(self.__camera, "acquire_synchronized", None)):
            frame_parameters = self.get_current_frame_parameters()
            data_element = self.__camera.acquire_synchronized(data_shape, **kwargs)
            if data_element:
                self.__update_data_element_for_sequence(data_element, frame_parameters)
                return [data_element]
            return []
        else:
            return self.acquire_sequence(int(numpy.product(data_shape)))

    def acquire_sequence_prepare(self, n: int) -> None:
        frame_parameters = self.get_current_frame_parameters()
        self.__camera.set_frame_parameters(frame_parameters)
        if callable(getattr(self.__camera, "acquire_sequence_prepare", None)):
            self.__camera.acquire_sequence_prepare(n)

    def __acquire_sequence_fallback(self, n: int, frame_parameters) -> dict:
        # if the device does not implement acquire_sequence, fall back to looping acquisition.
        processing = frame_parameters.processing
        acquisition_task = CameraAcquisitionTask(self.__get_instrument_controller(), self.hardware_source_id, True, self.__camera, self.__camera_settings, self.__camera_category, self.__signal_type, frame_parameters, self.display_name)
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
        data_element["metadata"] = dict()
        data_element["hardware_source"] = properties
        return data_element

    def acquire_sequence(self, n: int) -> typing.Sequence[typing.Dict]:
        frame_parameters = self.get_current_frame_parameters()
        if callable(getattr(self.__camera, "acquire_sequence", None)):
            data_element = self.__camera.acquire_sequence(n)
        else:
            data_element = self.__acquire_sequence_fallback(n, frame_parameters)
        if data_element:
            self.__update_data_element_for_sequence(data_element, frame_parameters)
            return [data_element]
        return []

    def __update_data_element_for_sequence(self, data_element, frame_parameters):
        binning = frame_parameters.binning
        data_element["version"] = 1
        data_element["state"] = "complete"
        instrument_controller = self.__get_instrument_controller()
        if "spatial_calibrations" not in data_element:
            update_spatial_calibrations(data_element, instrument_controller, self.__camera, self.__camera_category,
                                        data_element["data"].shape[1:], binning, binning)
            if "spatial_calibrations" in data_element:
                data_element["spatial_calibrations"] = [dict(), ] + data_element["spatial_calibrations"]
        update_intensity_calibration(data_element, instrument_controller, self.__camera)
        update_instrument_properties(data_element.setdefault("metadata", dict()).setdefault("instrument", dict()), instrument_controller, self.__camera)
        update_camera_properties(data_element.setdefault("metadata", dict()).setdefault("hardware_source", dict()), frame_parameters, self.hardware_source_id, self.display_name, data_element.get("signal_type", self.__signal_type))

    def update_camera_properties(self, properties: typing.MutableMapping, frame_parameters: CameraFrameParameters, signal_type: str = None) -> None:
        update_instrument_properties(properties, self.__get_instrument_controller(), self.__camera)
        update_camera_properties(properties, frame_parameters, self.hardware_source_id, self.display_name, signal_type or self.__signal_type)

    def get_camera_calibrations(self, camera_frame_parameters: CameraFrameParameters) -> typing.Tuple[Calibration.Calibration, ...]:
        processing = camera_frame_parameters.get("processing")
        instrument_controller = self.__get_instrument_controller()
        calibration_controls = self.__camera.calibration_controls
        binning = camera_frame_parameters.get("binning", 1)
        data_shape = self.get_expected_dimensions(binning)
        if processing in {"sum_project", "sum_masked"}:
            x_calibration = build_calibration(instrument_controller, calibration_controls, "x", binning, data_shape[0])
            return (x_calibration,)
        else:
            y_calibration = build_calibration(instrument_controller, calibration_controls, "y", binning, data_shape[1] if len(data_shape) > 1 else 0)
            x_calibration = build_calibration(instrument_controller, calibration_controls, "x", binning, data_shape[0])
            return (y_calibration, x_calibration)


    def get_camera_intensity_calibration(self, camera_frame_parameters: CameraFrameParameters) -> Calibration.Calibration:
        return build_calibration(self.__instrument_controller, self.__camera.calibration_controls, "intensity")

    def acquire_sequence_cancel(self) -> None:
        if callable(getattr(self.__camera, "acquire_sequence_cancel", None)):
            self.__camera.acquire_sequence_cancel()

    def get_acquire_sequence_metrics(self, frame_parameters: typing.Dict) -> typing.Dict:
        if hasattr(self.__camera, "get_acquire_sequence_metrics"):
            return self.__camera.get_acquire_sequence_metrics(frame_parameters)
        return dict()

    def __current_frame_parameters_changed(self, frame_parameters):
        if self.__acquisition_task:
            self.__acquisition_task.set_frame_parameters(self.__camera_settings.get_frame_parameters_from_dict(frame_parameters))
        self.__frame_parameters = self.__camera_settings.get_frame_parameters_from_dict(frame_parameters)

    def set_current_frame_parameters(self, frame_parameters):
        frame_parameters = self.__camera_settings.get_frame_parameters_from_dict(frame_parameters)
        self.__camera_settings.set_current_frame_parameters(frame_parameters)
        # __current_frame_parameters_changed will be called by the controller

    def get_current_frame_parameters(self):
        return self.__camera_settings.get_frame_parameters_from_dict(self.__frame_parameters)

    def __record_frame_parameters_changed(self, frame_parameters):
        self.__record_parameters = self.__camera_settings.get_frame_parameters_from_dict(frame_parameters)

    def set_record_frame_parameters(self, frame_parameters):
        frame_parameters = self.__camera_settings.get_frame_parameters_from_dict(frame_parameters)
        self.__camera_settings.set_record_frame_parameters(frame_parameters)
        # __record_frame_parameters_changed will be called by the controller

    def get_record_frame_parameters(self):
        return self.__record_parameters

    def get_frame_parameters_from_dict(self, d):
        return self.__camera_settings.get_frame_parameters_from_dict(d)

    def shift_click(self, mouse_position, camera_shape, logger: logging.Logger) -> None:
        instrument_controller = self.__get_instrument_controller()
        if callable(getattr(instrument_controller, "handle_shift_click", None)):
            instrument_controller.handle_shift_click(mouse_position=mouse_position, data_shape=camera_shape, camera=self.camera, logger=logger)
        else:
            # TODO: remove this backwards compatibility code once everyone updated to new technique above
            if self.__camera_category.lower() == "ronchigram":
                radians_per_pixel = instrument_controller.get_value("TVPixelAngle")
                defocus_value = instrument_controller.get_value("C10")  # get the defocus
                dx = radians_per_pixel * defocus_value * (mouse_position[1] - (camera_shape[1] / 2))
                dy = radians_per_pixel * defocus_value * (mouse_position[0] - (camera_shape[0] / 2))
                logger.info("Shifting (%s,%s) um.\n", -dx * 1e6, -dy * 1e6)
                instrument_controller.set_value("SShft.x", instrument_controller.get_value("SShft.x") - dx)
                instrument_controller.set_value("SShft.y", instrument_controller.get_value("SShft.y") - dy)

    def tilt_click(self, mouse_position, camera_shape, logger: logging.Logger) -> None:
        instrument_controller = self.__get_instrument_controller()
        if callable(getattr(instrument_controller, "handle_tilt_click", None)):
            instrument_controller.handle_tilt_click(mouse_position=mouse_position, data_shape=camera_shape, camera=self.camera, logger=logger)
        else:
            # TODO: remove this backwards compatibility code once everyone updated to new technique above
            if self.__camera_category.lower() == "ronchigram":
                radians_per_pixel = instrument_controller.get_value("TVPixelAngle")
                da = radians_per_pixel * (mouse_position[1] - (camera_shape[1] / 2))
                db = radians_per_pixel * (mouse_position[0] - (camera_shape[0] / 2))
                logger.info("Tilting (%s,%s) rad.\n", -da, -db)
                instrument_controller.set_value("STilt.x", instrument_controller.get_value("STilt.x") - da)
                instrument_controller.set_value("STilt.y", instrument_controller.get_value("STilt.y") - db)

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

    # Compatibility functions

    # used in camera control panel
    @property
    def modes(self):
        return self.__camera_settings.modes

    # used in service scripts
    def get_mode(self):
        return self.__camera_settings.get_mode()

    # used in service scripts
    def set_mode(self, mode):
        self.__camera_settings.set_mode(mode)

    # used in api, tests, camera control panel
    def set_frame_parameters(self, profile_index, frame_parameters):
        frame_parameters = self.__camera_settings.get_frame_parameters_from_dict(frame_parameters)
        self.__camera_settings.set_frame_parameters(profile_index, frame_parameters)

    # used in tuning, api, tests, camera control panel
    def get_frame_parameters(self, profile_index):
        return self.__camera_settings.get_frame_parameters(profile_index)

    # used in api, tests, camera control panel
    def set_selected_profile_index(self, profile_index):
        self.__camera_settings.set_selected_profile_index(profile_index)

    # used in api, camera control panel
    @property
    def selected_profile_index(self):
        return self.__camera_settings.selected_profile_index

    # used in camera control panel
    def open_configuration_interface(self, api_broker):
        self.__camera_settings.open_configuration_interface(api_broker)

    # used in camera control panel
    def open_monitor(self):
        self.__camera_settings.open_monitor()


class CameraFrameParameters(dict):
    """Example implementation for camera frame parameters; used in tests too."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        self.exposure_ms = self.get("exposure_ms", 125)
        self.binning = self.get("binning", 1)
        self.processing = self.get("processing")
        self.integration_count = self.get("integration_count")
        self.active_masks = [Mask.from_dict(mask) if not isinstance(mask, Mask) else mask for mask in self.get("active_masks", [])]

    def __copy__(self):
        return self.__class__(copy.copy(self.as_dict()))

    def __deepcopy__(self, memo):
        deepcopy = self.__class__(copy.deepcopy(self.as_dict()))
        memo[id(self)] = deepcopy
        return deepcopy

    def as_dict(self):
        return {
            "exposure_ms": self.exposure_ms,
            "binning": self.binning,
            "processing": self.processing,
            "integration_count": self.integration_count,
            "sum_channels": self.sum_channels,
            "active_masks": [mask.as_dict() for mask in self.active_masks],
        }


def get_instrument_calibration_value(instrument_controller: InstrumentController, calibration_controls, key) -> typing.Optional[float]:
    if key + "_control" in calibration_controls:
        valid, value = instrument_controller.TryGetVal(calibration_controls[key + "_control"])
        if valid:
            return value
    if key + "_value" in calibration_controls:
        return calibration_controls.get(key + "_value")
    return None


def build_calibration(instrument_controller: InstrumentController, calibration_controls: typing.Mapping, prefix: str,
                      relative_scale: float = 1, data_len: int = 0) -> Calibration.Calibration:
    scale = get_instrument_calibration_value(instrument_controller, calibration_controls, prefix + "_" + "scale")
    scale = scale * relative_scale if scale is not None else scale
    offset = get_instrument_calibration_value(instrument_controller, calibration_controls, prefix + "_" + "offset")
    units = get_instrument_calibration_value(instrument_controller, calibration_controls, prefix + "_" + "units")
    if calibration_controls.get(prefix + "_origin_override", None) == "center" and scale is not None and data_len:
        offset = -scale * data_len * 0.5
    return Calibration.Calibration(offset, scale, units)


def build_calibration_dict(instrument_controller: InstrumentController, calibration_controls: typing.Mapping,
                           prefix: str, relative_scale: float = 1, data_len: int = 0) -> typing.Dict:
    return build_calibration(instrument_controller, calibration_controls, prefix, relative_scale, data_len).rpc_dict


def update_spatial_calibrations(data_element, instrument_controller: InstrumentController, camera, camera_category, data_shape, scaling_x, scaling_y):
    if "spatial_calibrations" not in data_element:
        if "spatial_calibrations" in data_element.get("hardware_source", dict()):
            data_element["spatial_calibrations"] = data_element["hardware_source"]["spatial_calibrations"]
        elif hasattr(camera, "calibration"):  # used in nionccd1010
            data_element["spatial_calibrations"] = camera.calibration
        elif instrument_controller:
            if "calibration_controls" in data_element:
                calibration_controls = data_element["calibration_controls"]
            elif hasattr(camera, "calibration_controls"):
                calibration_controls = camera.calibration_controls
            else:
                calibration_controls = None
            if calibration_controls is not None:
                x_calibration_dict = build_calibration_dict(instrument_controller, calibration_controls, "x", scaling_x, data_shape[0])
                y_calibration_dict = build_calibration_dict(instrument_controller, calibration_controls, "y", scaling_y, data_shape[1] if len(data_shape) > 1 else 0)
                z_calibration_dict = build_calibration_dict(instrument_controller, calibration_controls, "z", 1, data_shape[2] if len(data_shape) > 2 else 0)
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


def update_intensity_calibration(data_element, instrument_controller: InstrumentController, camera):
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
            data_element["intensity_calibration"] = build_calibration_dict(instrument_controller, calibration_controls, "intensity")
    if "counts_per_electron" not in data_element:
        if calibration_controls is not None:
            counts_per_electron = get_instrument_calibration_value(instrument_controller, calibration_controls, "counts_per_electron")
            if counts_per_electron:
                data_element.setdefault("metadata", dict()).setdefault("hardware_source", dict())["counts_per_electron"] = counts_per_electron


def update_instrument_properties(stem_properties: typing.MutableMapping, instrument_controller: InstrumentController, camera: CameraDevice) -> None:
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
            acquisition_metatdata_groups = camera.acquisition_metatdata_groups
            instrument_controller.apply_metadata_groups(stem_properties, acquisition_metatdata_groups)


def update_camera_properties(properties: typing.MutableMapping, frame_parameters: CameraFrameParameters, hardware_source_id: str, display_name: str, signal_type: str = None) -> None:
    properties["hardware_source_id"] = hardware_source_id
    properties["hardware_source_name"] = display_name
    properties["exposure"] = frame_parameters.exposure_ms / 1000.0
    properties["binning"] = frame_parameters.binning
    if signal_type:
        properties["signal_type"] = signal_type


_component_registered_listener = None
_component_unregistered_listener = None

def run(configuration_location: pathlib.Path):
    def component_registered(component, component_types: typing.Set[str]) -> None:
        if "camera_module" in component_types:
            camera_module = component
            instrument_controller_id = getattr(camera_module, "instrument_controller_id", None)
            # TODO: remove next line when backwards compatibility no longer needed
            instrument_controller_id = instrument_controller_id or getattr(camera_module, "stem_controller_id", None)
            # grab the settings and camera panel info from the camera module
            camera_settings = camera_module.camera_settings
            camera_device = camera_module.camera_device
            camera_panel_type = getattr(camera_module, "camera_panel_type", None)  # a replacement camera panel
            camera_panel_delegate_type = getattr(camera_module, "camera_panel_delegate_type", None)  # a delegate for the default camera panel
            try:
                camera_hardware_source = CameraHardwareSource(instrument_controller_id, camera_device, camera_settings, configuration_location, camera_panel_type, camera_panel_delegate_type)
                if hasattr(camera_module, "priority"):
                    camera_hardware_source.priority = camera_module.priority
                component_types = {"hardware_source", "camera_hardware_source"}.union({camera_device.camera_type + "_camera_hardware_source"})
                Registry.register_component(camera_hardware_source, component_types)
                HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)
                camera_module.hardware_source = camera_hardware_source
            except Exception as e:
                camera_id = str(getattr(getattr(component, "camera_device", None), "camera_id", None))
                camera_id = camera_id or "UNKNOWN"
                logging.info("Camera Plug-in '" + camera_id + "' exception during initialization.")
                logging.info(traceback.format_exc())

    def component_unregistered(component, component_types):
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


class CameraInterface:
    # preliminary interface (v1.0.0) for camera hardware source
    def get_current_frame_parameters(self) -> dict: ...
    def create_frame_parameters(self, d: dict) -> dict: ...
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
    def grab_buffer(self, count: int, *, start: int = None) -> typing.Optional[typing.List[typing.List[DataAndMetadata.DataAndMetadata]]]: ...
    def make_reference_key(self, **kwargs) -> str: ...
