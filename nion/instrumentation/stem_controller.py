from __future__ import annotations

# standard libraries
import abc
import asyncio
import copy
import enum
import functools
import gettext
import math
import threading
import typing
import uuid

# third party libraries
# None

# local libraries
from nion.data import Calibration
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Observable
from nion.utils import ReferenceCounting
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.swift.model import DataItem
    from nion.swift.model import DisplayItem
    from nion.instrumentation import camera_base
    from nion.instrumentation import scan_base

_VectorType = typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]

_ = gettext.gettext


class PMTTypeEnum(enum.IntEnum):
    DF = 0
    BF = 1

PMTType = typing.Union[PMTTypeEnum, int]


class SubscanState(enum.Enum):
    INVALID = -1
    DISABLED = 0
    ENABLED = 1


class LineScanState(enum.Enum):
    INVALID = -1
    DISABLED = 0
    ENABLED = 1


class DriftIntervalUnit(enum.IntEnum):
    FRAME = 0
    TIME = 1
    LINE = 2
    SCAN = 3


class DriftCorrectionSettings:
    def __init__(self) -> None:
        self.interval = 0
        self.interval_units = DriftIntervalUnit.FRAME


AxisType = typing.Tuple[str, str]


class AxisDescription(typing.Protocol):

    @property
    def axis_id(self) -> str:
        """Read-only property for the (ideally unique) identifier of this axis.

        """
        raise NotImplementedError()

    @property
    def axis_type(self) -> typing.Tuple[str, str]:
        """Read-only property for the co-ordinate names of this axis.

        Note: This might be removed in a future release
        """
        raise NotImplementedError()

    @property
    def display_name(self) -> str:
        """Read-only property for the name of this axis as it appears in the UI

        """
        raise NotImplementedError()


class ScanContext:
    def __init__(self) -> None:
        self.size: typing.Optional[Geometry.IntSize] = None
        self.center_nm: typing.Optional[Geometry.FloatPoint] = None
        self.fov_nm: typing.Optional[float] = None
        self.rotation_rad: typing.Optional[float] = None

    def __repr__(self) -> str:
        if self.fov_nm and self.size and self.rotation_rad:
            return f"{self.size} {self.fov_nm}nm {math.degrees(self.rotation_rad)}deg"
        else:
            return "NO CONTEXT"

    def __eq__(self, other: typing.Any) -> bool:
        if other is None:
            return False
        if not isinstance(other, self.__class__):
            return False
        if other.size != self.size:
            return False
        if other.center_nm != self.center_nm:
            return False
        if other.fov_nm != self.fov_nm:
            return False
        if other.rotation_rad != self.rotation_rad:
            return False
        return True

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> ScanContext:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.size = self.size
        result.center_nm = self.center_nm
        result.fov_nm = self.fov_nm
        result.rotation_rad = self.rotation_rad
        return result

    @property
    def is_valid(self) -> bool:
        return self.size is not None and self.fov_nm is not None and self.rotation_rad is not None and self.center_nm is not None

    def clear(self) -> None:
        self.size = None
        self.center_nm = None
        self.fov_nm = None
        self.rotation_rad = None

    def update(self, size: Geometry.IntSize, center_nm: Geometry.FloatPoint, fov_nm: float, rotation_rad: float) -> None:
        self.size = Geometry.IntSize.make(size)
        self.center_nm = Geometry.FloatPoint.make(center_nm)
        self.fov_nm = fov_nm
        self.rotation_rad = rotation_rad

    @property
    def fov_size_nm(self) -> typing.Optional[Geometry.FloatSize]:
        if self.size and self.fov_nm is not None:
            if self.size.aspect_ratio > 1.0:
                return Geometry.FloatSize(height=self.fov_nm / self.size.aspect_ratio, width=self.fov_nm)
            else:
                return Geometry.FloatSize(height=self.fov_nm, width=self.fov_nm * self.size.aspect_ratio)
        else:
            return None

    @property
    def calibration(self) -> Calibration.Calibration:
        if self.size and self.fov_nm is not None:
            return Calibration.Calibration(scale=self.fov_nm / max(self.size.width, self.size.height), units="nm")
        return Calibration.Calibration()


class STEMController(Observable.Observable):
    """An interface to a STEM microscope.

    Methods and properties starting with a single underscore are called internally and shouldn't be called by general
    clients.

    Methods starting with double underscores are private.

    Probe
    -----
    probe_state (parked, blanked, scanning)
    probe_position (fractional coordinates, optional)
    set_probe_position(probe_position)
    validate_probe_position()

    probe_state_changed_event (probe_state, probe_position)
    """

    def __init__(self) -> None:
        super().__init__()
        self.__probe_position: typing.Optional[Geometry.FloatPoint] = None
        self.__probe_state_stack = list()  # parked, or scanning
        self.__probe_state_stack.append("parked")
        self.__scan_context = ScanContext()
        self.probe_state_changed_event = Event.Event()
        self.__subscan_state = SubscanState.INVALID
        self.__subscan_region: typing.Optional[Geometry.FloatRect] = None
        self.__subscan_rotation = 0.0
        self.__line_scan_state = LineScanState.INVALID
        self.__line_scan_vector: typing.Optional[_VectorType] = None
        self.__drift_channel_id: typing.Optional[str] = None
        self.__drift_region: typing.Optional[Geometry.FloatRect] = None
        self.__drift_rotation = 0.0
        self.__drift_settings = DriftCorrectionSettings()
        self.__scan_context_channel_map : typing.Dict[str, DataItem.DataItem] = dict()
        self.scan_context_data_items_changed_event = Event.Event()
        self.scan_context_changed_event = Event.Event()
        self.__ronchigram_camera: typing.Optional[camera_base.CameraHardwareSource] = None
        self.__eels_camera: typing.Optional[camera_base.CameraHardwareSource] = None
        self.__scan_controller: typing.Optional[scan_base.ScanHardwareSource] = None
        self.__drift_tracker: typing.Optional[DriftTracker.DriftTracker] = None

    def close(self) -> None:
        self.__scan_context_channel_map = typing.cast(typing.Any, None)

    def reset(self) -> None:
        self.__probe_position = None
        self.__probe_state_stack.clear()
        self.__probe_state_stack.append("parked")
        self.__scan_context.clear()
        self.__subscan_state = SubscanState.INVALID
        self.__subscan_region = None
        self.__subscan_rotation = 0.0
        self.__line_scan_state = LineScanState.INVALID
        self.__line_scan_vector = None
        self.__drift_channel_id = None
        self.__drift_region = None
        self.__drift_rotation = 0.0
        self.__drift_settings = DriftCorrectionSettings()
        self.__scan_context_channel_map.clear()

    # configuration methods

    @property
    def ronchigram_camera(self) -> typing.Optional[camera_base.CameraHardwareSource]:
        if self.__ronchigram_camera:
            return self.__ronchigram_camera
        return typing.cast(typing.Optional["camera_base.CameraHardwareSource"],
                           Registry.get_component("ronchigram_camera_hardware_source"))

    def set_ronchigram_camera(self, camera: typing.Optional[HardwareSource.HardwareSource]) -> None:
        assert camera is None or camera.features.get("is_ronchigram_camera", False)
        self.__ronchigram_camera = typing.cast(typing.Optional["camera_base.CameraHardwareSource"], camera)

    @property
    def eels_camera(self) -> typing.Optional[camera_base.CameraHardwareSource]:
        if self.__eels_camera:
            return self.__eels_camera
        return typing.cast(typing.Optional["camera_base.CameraHardwareSource"],
                           Registry.get_component("eels_camera_hardware_source"))

    def set_eels_camera(self, camera: typing.Optional[HardwareSource.HardwareSource]) -> None:
        assert camera is None or camera.features.get("is_eels_camera", False)
        self.__eels_camera = typing.cast(typing.Optional["camera_base.CameraHardwareSource"], camera)

    @property
    def scan_controller(self) -> typing.Optional[scan_base.ScanHardwareSource]:
        if self.__scan_controller:
            return self.__scan_controller
        return typing.cast(typing.Optional["scan_base.ScanHardwareSource"],
                           Registry.get_component("scan_hardware_source"))

    def set_scan_controller(self, scan_controller: typing.Optional[HardwareSource.HardwareSource]) -> None:
        self.__scan_controller = typing.cast(typing.Optional["scan_base.ScanHardwareSource"], scan_controller)

    @property
    def drift_tracker(self) -> typing.Optional[DriftTracker.DriftTracker]:
        if self.__drift_tracker:
            return self.__drift_tracker
        return typing.cast(typing.Optional[DriftTracker.DriftTracker],
                           Registry.get_component("drift_tracker"))

    # end configuration methods

    def _enter_scanning_state(self) -> None:
        # push 'scanning' onto the probe state stack; the `probe_state` will now be `scanning`
        self.__probe_state_stack.append("scanning")
        # fire off the probe state changed event.
        self.probe_state_changed_event.fire(self.probe_state, self.probe_position)
        # ensure that SubscanState is valid (ENABLED or DISABLED, not INVALID)
        if self.subscan_state == SubscanState.INVALID:
            self.subscan_state = SubscanState.DISABLED
        if self.line_scan_state == LineScanState.INVALID:
            self.line_scan_state = LineScanState.DISABLED

    def _exit_scanning_state(self) -> None:
        # pop the 'scanning' probe state and fire off the probe state changed event.
        self.__probe_state_stack.pop()
        self.probe_state_changed_event.fire(self.probe_state, self.probe_position)

    def _enter_synchronized_state(self, scan_controller: HardwareSource.HardwareSource, *, camera: typing.Optional[HardwareSource.HardwareSource] = None) -> None:
        pass

    def _exit_synchronized_state(self, scan_controller: HardwareSource.HardwareSource, *, camera: typing.Optional[HardwareSource.HardwareSource] = None) -> None:
        pass

    @property
    def subscan_state(self) -> SubscanState:
        return self.__subscan_state

    @subscan_state.setter
    def subscan_state(self, value: SubscanState) -> None:
        if self.__subscan_state != value:
            self.__subscan_state = value
            self.notify_property_changed("subscan_state")

    @property
    def subscan_region(self) -> typing.Optional[Geometry.FloatRect]:
        return self.__subscan_region

    @subscan_region.setter
    def subscan_region(self, value: typing.Optional[Geometry.FloatRect]) -> None:
        if self.__subscan_region != value:
            self.__subscan_region = value
            self.notify_property_changed("subscan_region")

    @property
    def subscan_rotation(self) -> float:
        return self.__subscan_rotation

    @subscan_rotation.setter
    def subscan_rotation(self, value: float) -> None:
        if self.__subscan_rotation != value:
            self.__subscan_rotation = value
            self.notify_property_changed("subscan_rotation")

    @property
    def line_scan_state(self) -> LineScanState:
        return self.__line_scan_state

    @line_scan_state.setter
    def line_scan_state(self, value: LineScanState) -> None:
        if self.__line_scan_state != value:
            self.__line_scan_state = value
            self.notify_property_changed("line_scan_state")

    @property
    def line_scan_vector(self) -> typing.Optional[_VectorType]:
        return self.__line_scan_vector

    @line_scan_vector.setter
    def line_scan_vector(self, value: typing.Optional[_VectorType]) -> None:
        if self.__line_scan_vector != value:
            self.__line_scan_vector = value
            self.notify_property_changed("line_scan_vector")

    @property
    def drift_channel_id(self) -> typing.Optional[str]:
        return self.__drift_channel_id

    @drift_channel_id.setter
    def drift_channel_id(self, value: typing.Optional[str]) -> None:
        if self.__drift_channel_id != value:
            self.__drift_channel_id = value
            self.notify_property_changed("drift_channel_id")

    @property
    def drift_region(self) -> typing.Optional[Geometry.FloatRect]:
        return self.__drift_region

    @drift_region.setter
    def drift_region(self, value: typing.Optional[Geometry.FloatRect]) -> None:
        if self.__drift_region != value:
            self.__drift_region = value
            self.notify_property_changed("drift_region")

    @property
    def drift_rotation(self) -> float:
        return self.__drift_rotation

    @drift_rotation.setter
    def drift_rotation(self, value: float) -> None:
        if self.__drift_rotation != value:
            self.__drift_rotation = value
            self.notify_property_changed("drift_rotation")

    @property
    def drift_settings(self) -> DriftCorrectionSettings:
        return self.__drift_settings

    @drift_settings.setter
    def drift_settings(self, value: DriftCorrectionSettings) -> None:
        if self.__drift_settings != value:
            self.__drift_settings = value
            self.notify_property_changed("drift_settings")

    def disconnect_probe_connections(self) -> None:
        self.__scan_context_channel_map = dict()
        self.scan_context_data_items_changed_event.fire()

    def _update_scan_channel_map(self, channel_map: typing.Mapping[str, DataItem.DataItem]) -> None:
        old_scan_context_channel_map = copy.copy(self.__scan_context_channel_map)
        self.__scan_context_channel_map.update(channel_map)
        if old_scan_context_channel_map != self.__scan_context_channel_map:
            self.scan_context_data_items_changed_event.fire()

    @property
    def scan_context(self) -> ScanContext:
        return self.__scan_context

    def _update_scan_context(self, size: Geometry.IntSize, center_nm: Geometry.FloatPoint, fov_nm: float, rotation_rad: float) -> None:
        old_context = copy.deepcopy(self.scan_context)
        self.__scan_context.update(size, center_nm, fov_nm, rotation_rad)
        if old_context != self.scan_context:
            self.scan_context_changed_event.fire()

    def _clear_scan_context(self) -> None:
        old_context = copy.deepcopy(self.scan_context)
        self.__scan_context.clear()
        if old_context != self.scan_context:
            self.scan_context_changed_event.fire()

    def _confirm_scan_context(self, size: Geometry.IntSize, center_nm: Geometry.FloatPoint, fov_nm: float, rotation_rad: float) -> None:
        current_context = copy.deepcopy(self.scan_context)
        current_context.update(size, center_nm, fov_nm, rotation_rad)
        if current_context != self.scan_context:
            self._clear_scan_context()

    @property
    def probe_position(self) -> typing.Optional[Geometry.FloatPoint]:
        """ Return the probe position, in normalized coordinates with origin at top left. Only valid if probe_state is 'parked'."""
        return self.__probe_position

    @probe_position.setter
    def probe_position(self, value: typing.Optional[Geometry.FloatPoint]) -> None:
        if value is not None:
            # convert the probe position to a FloatPoint and limit it to the 0.0 to 1.0 range in both axes.
            value = Geometry.FloatPoint(y=max(min(value.y, 1.0), 0.0), x=max(min(value.x, 1.0), 0.0))
        if self.probe_position != value:
            self.__probe_position = value
            self.notify_property_changed("probe_position")
            # update the probe position for listeners and also explicitly update for probe_graphic_connections.
            self.probe_state_changed_event.fire(self.probe_state, self.probe_position)

    def set_probe_position(self, new_probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        self.probe_position = new_probe_position

    def validate_probe_position(self) -> None:
        """Validate the probe position.

        This is called when the user switches from not controlling to controlling the position."""
        self.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))

    @property
    def probe_state(self) -> str:
        """Probe state is the current probe state and can be 'parked', or 'scanning'."""
        return self.__probe_state_stack[-1]

    # instrument API

    def set_control_output(self, name: str, value: float, options: typing.Optional[typing.Mapping[str, typing.Any]] = None) -> None:
        options = options if options else dict()
        value_type = options.get("value_type", "output")
        inform = options.get("inform", False)
        confirm = options.get("confirm", False)
        confirm_tolerance_factor = options.get("confirm_tolerance_factor", 1.0)  # instrument keeps track of default; this is a factor applied to the default
        confirm_timeout = options.get("confirm_timeout", 16.0)
        if value_type == "output":
            if inform:
                self.InformControl(name, value)
            elif confirm:
                if not self.SetValAndConfirm(name, value, confirm_tolerance_factor, int(confirm_timeout * 1000)):
                    raise TimeoutError("Setting '" + name + "'.")
            else:
                self.SetVal(name, value)
        elif value_type == "delta" and not inform:
            self.SetValDelta(name, value)
        else:
            raise NotImplementedError()

    def get_control_output(self, name: str) -> float:
        return self.GetVal(name)

    def get_control_state(self, name: str) -> typing.Optional[str]:
        value_exists, value = self.TryGetVal(name)
        return "unknown" if value_exists else None

    def get_property(self, name: str) -> typing.Any:
        if name in ("probe_position", "probe_state"):
            return getattr(self, name)
        return self.get_control_output(name)

    def set_property(self, name: str, value: typing.Any) -> None:
        if name in ("probe_position"):
            return setattr(self, name, value)
        return self.set_control_output(name, value)

    def _get_config_property(self, name: str) -> typing.Any:
        """Get a configuration property.

        Concrete STEM controllers can optionally return configuration properties using this method.
        A specific use case is for configuration of a simulated device for testing.
        """
        raise NotImplementedError()

    def _set_config_property(self, name: str, value: typing.Any) -> None:
        """Set a configuration property.

        Most configuration properties are immutable. However, devices may choose to allow them to be
        changed programmatically for various reasons. A specific use case is for configuration of
        a simulated device for testing.
        """
        raise NotImplementedError()

    def apply_metadata_groups(self, properties: typing.MutableMapping[str, typing.Any], metatdata_groups: typing.Sequence[typing.Tuple[typing.Sequence[str], str]]) -> None:
        """Apply metadata groups to properties.

        Metadata groups is a tuple with two elements. The first is a list of strings representing a dict-path in which
        to add the controls. The second is a control group from which to read a list of controls to be added as name
        value pairs to the dict-path.
        """
        pass

    @property
    def axis_descriptions(self) -> typing.Sequence[AxisDescription]:
        return list()

    def get_reference_setting_index(self, settings_control: str) -> typing.Optional[int]:
        """
        Queries the instrument for the reference setting number of a settings control.

        A reference setting is the setting that you would typically use for tuning and alignment of the microscope. In
        this setting it is guaranteed that rotations, strenghts, etc. are calibrated and set up so that (automated)
        procedures can work properly. There is always exactly one reference setting per settings control, because it
        is important to always align the microscope in the same setting to avoid creating diverging alignments and
        settings.
        """
        raise NotImplementedError()

    def axis_transform_point(self, point: Geometry.FloatPoint, from_axis: AxisDescription, to_axis: AxisDescription) -> Geometry.FloatPoint:
        """
        Convert the vector "value" from "from_axis" to "to_axis".

        Existing axis descriptions can be retrieved via:
        `STEMController.axis_descriptions`

        Raises `ValueError` if an invalid axis is passed as "from_axis" or "to_axis".
        """
        raise NotImplementedError()

    # end instrument API

    # required functions (templates). subclasses should override.

    def TryGetVal(self, s: str) -> typing.Tuple[bool, typing.Optional[float]]:
        return False, None

    def GetVal(self, s: str, default_value: typing.Optional[float] = None) -> float:
        raise Exception(f"No element named '{s}' exists! Cannot get value.")

    def SetVal(self, s: str, val: float) -> bool:
        return False

    def SetValWait(self, s: str, val: float, timeout_ms: int) -> bool:
        return False

    def SetValAndConfirm(self, s: str, val: float, tolfactor: float, timeout_ms: int) -> bool:
        return False

    def SetValDelta(self, s: str, delta: float) -> bool:
        return False

    def SetValDeltaAndConfirm(self, s: str, delta: float, tolfactor: float, timeout_ms: int) -> bool:
        return False

    def InformControl(self, s: str, val: float) -> bool:
        return False

    def GetVal2D(self, s: str, default_value: typing.Optional[Geometry.FloatPoint] = None, *, axis: AxisType) -> Geometry.FloatPoint:
        raise Exception(f"No 2D element named '{s}' exists! Cannot get value.")

    def SetVal2D(self, s:str, value: Geometry.FloatPoint, *, axis: typing.Optional[AxisType] = None) -> bool:
        return False

    def SetVal2DAndConfirm(self, s: str, val: Geometry.FloatPoint, tolfactor: float, timeout_ms: int, *, axis: AxisType) -> bool:
        return False

    def SetVal2DDelta(self, s: str, delta: Geometry.FloatPoint, *, axis: AxisType) -> bool:
        return False

    def SetVal2DDeltaAndConfirm(self, s: str, delta: Geometry.FloatPoint, tolfactor: float, timeout_ms: int, *, axis: AxisType) -> bool:
        return False

    def InformControl2D(self, s: str, val: Geometry.FloatPoint, *, axis: AxisType) -> bool:
        return False

    def HasValError(self, s: str) -> bool:
        return False

    # end required functions

    # high level commands

    def change_stage_position(self, *, dy: typing.Optional[float] = None, dx: typing.Optional[float] = None) -> None:
        """Shift the stage by dx, dy (meters). Do not wait for confirmation."""
        raise NotImplementedError()

    def change_pmt_gain(self, pmt_type: PMTType, *, factor: float) -> None:
        """Change specified PMT by factor. Do not wait for confirmation."""
        raise NotImplementedError()

    # end high level commands


class AbstractGraphicSetHandler(abc.ABC):
    """Handle callbacks from the graphic set controller to the model."""

    @abc.abstractmethod
    def _create_graphic(self) -> Graphics.Graphic:
        """Called to create a new graphic for a new display item."""
        ...

    @abc.abstractmethod
    def _update_graphic(self, graphic: Graphics.Graphic) -> None:
        """Called to update the graphic when the model changes."""
        ...

    @abc.abstractmethod
    def _graphic_property_changed(self, graphic: Graphics.Graphic, name: str) -> None:
        """Called to update the model when the graphic changes."""
        ...

    @abc.abstractmethod
    def _graphic_removed(self, graphic: Graphics.Graphic) -> None:
        """Called when one of the graphics are removed."""
        ...


class GraphicSetController:

    def __init__(self, handler: AbstractGraphicSetHandler):
        self.__graphic_trackers : typing.List[typing.Tuple[Graphics.Graphic, Event.EventListener, Event.EventListener, Event.EventListener]] = list()
        self.__handler = handler

    def close(self) -> None:
        for _, graphic_property_changed_listener, remove_region_graphic_event_listener, display_about_to_be_removed_listener in self.__graphic_trackers:
            graphic_property_changed_listener.close()
            remove_region_graphic_event_listener.close()
            display_about_to_be_removed_listener.close()
        self.__graphic_trackers = list()

    @property
    def graphics(self) -> typing.Sequence[Graphics.Graphic]:
        return [t[0] for t in self.__graphic_trackers]

    def synchronize_graphics(self, display_items: typing.Sequence[DisplayItem.DisplayItem]) -> None:
        # create graphics for each scan data item if it doesn't exist
        if not self.__graphic_trackers:
            for display_item in display_items:
                graphic = self.__handler._create_graphic()

                def graphic_property_changed(graphic: Graphics.Graphic, name: str) -> None:
                    self.__handler._graphic_property_changed(graphic, name)

                graphic_property_changed_listener = graphic.property_changed_event.listen(functools.partial(graphic_property_changed, graphic))

                def graphic_removed(graphic: Graphics.Graphic) -> None:
                    self.__remove_one_graphic(graphic)
                    self.__handler._graphic_removed(graphic)

                def display_removed(graphic: Graphics.Graphic) -> None:
                    self.__remove_one_graphic(graphic)

                remove_region_graphic_event_listener = graphic.about_to_be_removed_event.listen(functools.partial(graphic_removed, graphic))
                display_about_to_be_removed_listener = display_item.about_to_be_removed_event.listen(functools.partial(display_removed, graphic))
                self.__graphic_trackers.append((graphic, graphic_property_changed_listener, remove_region_graphic_event_listener, display_about_to_be_removed_listener))
                display_item.add_graphic(graphic)
        # apply new value to any existing graphics
        for graphic in self.graphics:
            self.__handler._update_graphic(graphic)

    def remove_all_graphics(self) -> None:
        # remove any graphics, doing it in a way such that all references to the listeners
        # are out of scope when the graphic is removed from its container via `remove_graphic`.
        # this ensures the listeners are inactive when `remove_graphic` is called.
        graphics = list()
        for graphic, graphic_property_changed_listener, remove_region_graphic_event_listener, display_about_to_be_removed_listener in self.__graphic_trackers:
            graphic_property_changed_listener.close()
            remove_region_graphic_event_listener.close()
            display_about_to_be_removed_listener.close()
            graphics.append(graphic)
            del graphic_property_changed_listener
            del remove_region_graphic_event_listener
            del display_about_to_be_removed_listener
        self.__graphic_trackers = list()
        for graphic in graphics:
            graphic.display_item.remove_graphic(graphic).close()

    def __remove_one_graphic(self, graphic_to_remove: Graphics.Graphic) -> None:
        graphic_trackers = list()
        for graphic, graphic_property_changed_listener, remove_region_graphic_event_listener, display_about_to_be_removed_listener in self.__graphic_trackers:
            if graphic_to_remove != graphic:
                graphic_trackers.append((graphic, graphic_property_changed_listener, remove_region_graphic_event_listener, display_about_to_be_removed_listener))
            else:
                graphic_property_changed_listener.close()
                remove_region_graphic_event_listener.close()
                display_about_to_be_removed_listener.close()
        self.__graphic_trackers = graphic_trackers


class DisplayItemListModel(Observable.Observable):
    """Make an observable list model from the item source with a list as the item."""

    def __init__(self, document_model: DocumentModel.DocumentModel, item_key: str,
                 predicate: typing.Callable[[DisplayItem.DisplayItem], bool],
                 change_event: typing.Optional[Event.Event] = None):
        super().__init__()
        self.__document_model = document_model
        self.__item_key = item_key
        self.__predicate = predicate
        self.__items : typing.List[DisplayItem.DisplayItem] = list()

        self.__item_inserted_listener = document_model.item_inserted_event.listen(self.__item_inserted)
        self.__item_removed_listener = document_model.item_removed_event.listen(self.__item_removed)

        for index, display_item in enumerate(document_model.display_items):
            self.__item_inserted("display_items", display_item, index)

        self.__change_event_listener = change_event.listen(self.refilter) if change_event else None

        # special handling when document closes
        def unlisten() -> None:
            if self.__change_event_listener:
                self.__change_event_listener.close()
                self.__change_event_listener = None

        self.__document_close_listener = document_model.about_to_close_event.listen(unlisten)

    def close(self) -> None:
        if self.__change_event_listener:
            self.__change_event_listener.close()
            self.__change_event_listener = None
        self.__item_inserted_listener.close()
        self.__item_inserted_listener = typing.cast(typing.Any, None)
        self.__item_removed_listener.close()
        self.__item_removed_listener = typing.cast(typing.Any, None)
        self.__document_close_listener.close()
        self.__document_close_listener = typing.cast(typing.Any, None)
        self.__document_model = typing.cast(typing.Any, None)

    def __item_inserted(self, key: str, display_item: DisplayItem.DisplayItem, index: int) -> None:
        if key == "display_items" and not display_item in self.__items and self.__predicate(display_item):
            index = len(self.__items)
            self.__items.append(display_item)
            self.notify_insert_item(self.__item_key, display_item, index)

    def __item_removed(self, key: str, display_item: DisplayItem.DisplayItem, index: int) -> None:
        if key == "display_items" and display_item in self.__items:
            index = self.__items.index(display_item)
            self.__items.pop(index)
            self.notify_remove_item(self.__item_key, display_item, index)

    @property
    def items(self) -> typing.Sequence[DisplayItem.DisplayItem]:
        return self.__items

    def __getattr__(self, item: str) -> typing.Any:
        if item == self.__item_key:
            return self.items
        raise AttributeError()

    def refilter(self) -> None:
        self.item_set = set(self.__items)
        for display_item in self.__document_model.display_items:
            if self.__predicate(display_item):
                # insert item if not already inserted
                if not display_item in self.__items:
                    index = len(self.__items)
                    self.__items.append(display_item)
                    self.notify_insert_item(self.__item_key, display_item, index)
            else:
                # remove item if in list
                if display_item in self.__items:
                    index = self.__items.index(display_item)
                    self.__items.pop(index)
                    self.notify_remove_item(self.__item_key, display_item, index)

    def clean(self, graphic_id: str) -> None:
        display_items = self.__document_model.display_items
        for display_item in display_items:
            for graphic in display_item.graphics:
                if graphic.graphic_id == graphic_id:
                    display_item.remove_graphic(graphic).close()

def make_scan_display_item_list_model(document_model: DocumentModel.DocumentModel, stem_controller: STEMController) -> DisplayItemListModel:
    def is_scan_context_display_item(display_item: DisplayItem.DisplayItem) -> bool:
        scan_controller = stem_controller.scan_controller
        if scan_controller:
            for data_channel in scan_controller.data_channels:
                channel_id = data_channel.channel_id
                if channel_id and not channel_id.endswith("subscan") and channel_id != "drift":
                    data_item_channel_reference = document_model.get_data_item_channel_reference(scan_controller.hardware_source_id, channel_id)
                    if data_item_channel_reference and data_item_channel_reference.display_item == display_item:
                        return True
        return False

    return DisplayItemListModel(document_model, "display_items", is_scan_context_display_item, stem_controller.scan_context_data_items_changed_event)


class EventLoopMonitor:
    """Utility base class to monitor availability of event loop."""

    def __init__(self, document_model: DocumentModel.DocumentModel, event_loop: asyncio.AbstractEventLoop):
        self.__event_loop : typing.Optional[asyncio.AbstractEventLoop] = event_loop
        self.__document_close_listener = document_model.about_to_close_event.listen(self._unlisten)
        self.__closed = False

    def _unlisten(self) -> None:
        pass

    def _mark_closed(self) -> None:
        self.__closed = True
        self.__document_close_listener.close()
        self.__document_close_listener = typing.cast(typing.Any, None)
        self.__event_loop = None

    def _call_soon_threadsafe(self, fn: typing.Callable[...,  None], *args: typing.Any) -> None:
        if not self.__closed:
            def safe_fn() -> None:
                if not self.__closed:
                    fn(*args)

            assert self.__event_loop
            self.__event_loop.call_soon_threadsafe(safe_fn)


class ProbeView(EventLoopMonitor, AbstractGraphicSetHandler, DocumentModel.AbstractImplicitDependency):
    """Observes the probe (STEM controller) and updates data items and graphics."""
    count = 0

    def __init__(self, stem_controller: STEMController, document_model: DocumentModel.DocumentModel, event_loop: asyncio.AbstractEventLoop):
        super().__init__(document_model, event_loop)
        self.__class__.count += 1
        self.__stem_controller = stem_controller
        self.__document_model = document_model
        self.__scan_display_items_model = make_scan_display_item_list_model(document_model, stem_controller)
        self.__project_loaded_event_listener = document_model.project_loaded_event.listen(ReferenceCounting.weak_partial(ProbeView.__refilter_scan_display_items, self))
        self.__graphic_set = GraphicSetController(self)
        # note: these property changed listeners can all possibly be fired from a thread.
        self.__probe_state = None
        self.__probe_state_changed_listener = stem_controller.probe_state_changed_event.listen(self.__probe_state_changed)
        self.__document_model.register_implicit_dependency(self)
        # update in case a new document model is opened with the line scan already enabled
        self.__update_probe_state(stem_controller.probe_state, stem_controller.probe_position)

    def close(self) -> None:
        self._mark_closed()
        if self.__probe_state_changed_listener:
            self.__probe_state_changed_listener.close()
            self.__probe_state_changed_listener = typing.cast(typing.Any, None)
        self.__document_model.unregister_implicit_dependency(self)
        self.__graphic_set.close()
        self.__graphic_set = typing.cast(typing.Any, None)
        self.__scan_display_items_model.close()
        self.__scan_display_items_model = typing.cast(typing.Any, None)
        self.__document_model = typing.cast(typing.Any, None)
        self.__stem_controller = typing.cast(typing.Any, None)
        self.__project_loaded_event_listener = typing.cast(typing.Any, None)
        self.__class__.count -= 1

    def _unlisten(self) -> None:
        if self.__probe_state_changed_listener:
            self.__probe_state_changed_listener.close()
            self.__probe_state_changed_listener = typing.cast(typing.Any, None)

    def __probe_state_changed(self, probe_state: str, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        # thread safe. move actual call to main thread using the event loop.
        self._call_soon_threadsafe(self.__update_probe_state, probe_state, probe_position)

    def __update_probe_state(self, probe_state: str, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        assert threading.current_thread() == threading.main_thread()
        if probe_state != "scanning" and probe_position is not None:
            self.__graphic_set.synchronize_graphics(self.__scan_display_items_model.display_items)
        else:
            self.__graphic_set.remove_all_graphics()

    def __refilter_scan_display_items(self) -> None:
        stem_controller = self.__stem_controller
        self.__scan_display_items_model.clean("probe")
        self.__scan_display_items_model.refilter()
        self.__update_probe_state(stem_controller.probe_state, stem_controller.probe_position)

    # implement methods for the graphic set handler

    def _graphic_removed(self, probe_graphic: Graphics.Graphic) -> None:
        # clear probe state
        self.__stem_controller.probe_position = None

    def _create_graphic(self) -> Graphics.PointGraphic:
        graphic = Graphics.PointGraphic()
        graphic.graphic_id = "probe"
        graphic.label = _("Probe")
        graphic.position = self.__stem_controller.probe_position or Geometry.FloatPoint()
        graphic.is_bounds_constrained = True
        graphic.color = "#F80"
        return graphic

    def _update_graphic(self, graphic: Graphics.Graphic) -> None:
        assert isinstance(graphic, Graphics.PointGraphic)
        probe_position = self.__stem_controller.probe_position
        if probe_position and graphic.position != probe_position:
            graphic.position = probe_position

    def _graphic_property_changed(self, graphic: Graphics.Graphic, name: str) -> None:
        if name == "position":
            assert isinstance(graphic, Graphics.PointGraphic)
            self.__stem_controller.probe_position = Geometry.FloatPoint.make(graphic.position)

    def get_dependents(self, item: Graphics.Graphic) -> typing.Sequence[Graphics.Graphic]:
        graphics = self.__graphic_set.graphics
        if item in graphics:
            return list(set(graphics) - {item})
        return list()


class SubscanView(EventLoopMonitor, AbstractGraphicSetHandler, DocumentModel.AbstractImplicitDependency):
    """Observes the STEM controller and updates data items and graphics."""
    count = 0

    def __init__(self, stem_controller: STEMController, document_model: DocumentModel.DocumentModel, event_loop: asyncio.AbstractEventLoop):
        super().__init__(document_model, event_loop)
        self.__class__.count += 1
        self.__stem_controller = stem_controller
        self.__document_model = document_model
        self.__scan_display_items_model = make_scan_display_item_list_model(document_model, stem_controller)
        self.__project_loaded_event_listener = document_model.project_loaded_event.listen(ReferenceCounting.weak_partial(SubscanView.__refilter_scan_display_items, self))
        self.__graphic_set = GraphicSetController(self)
        # note: these property changed listeners can all possibly be fired from a thread.
        self.__subscan_region_changed_listener = stem_controller.property_changed_event.listen(self.__subscan_region_changed)
        self.__subscan_rotation_changed_listener = stem_controller.property_changed_event.listen(self.__subscan_rotation_changed)
        self.__document_model.register_implicit_dependency(self)
        # update in case a new document model is opened with the line scan already enabled
        self.__update_subscan_region()

    def close(self) -> None:
        self._mark_closed()
        self.__document_model.unregister_implicit_dependency(self)
        self.__graphic_set.close()
        self.__graphic_set = typing.cast(typing.Any, None)
        self.__scan_display_items_model.close()
        self.__scan_display_items_model = typing.cast(typing.Any, None)
        self.__document_model = typing.cast(typing.Any, None)
        self.__stem_controller = typing.cast(typing.Any, None)
        self.__project_loaded_event_listener = typing.cast(typing.Any, None)
        self.__class__.count -= 1

    def _unlisten(self) -> None:
        # unlisten to the event loop dependent listeners
        self.__subscan_region_changed_listener.close()
        self.__subscan_region_changed_listener = typing.cast(typing.Any, None)
        self.__subscan_rotation_changed_listener.close()
        self.__subscan_rotation_changed_listener = typing.cast(typing.Any, None)

    # methods for handling changes to the subscan region

    def __subscan_region_changed(self, name: str) -> None:
        # must be thread safe
        if name == "subscan_region":
            self._call_soon_threadsafe(self.__update_subscan_region)

    def __subscan_rotation_changed(self, name: str) -> None:
        # must be thread safe
        if name == "subscan_rotation":
            self._call_soon_threadsafe(self.__update_subscan_region)

    def __update_subscan_region(self) -> None:
        assert threading.current_thread() == threading.main_thread()
        if self.__stem_controller.subscan_region:
            self.__graphic_set.synchronize_graphics(self.__scan_display_items_model.display_items)
        else:
            self.__graphic_set.remove_all_graphics()

    def __refilter_scan_display_items(self) -> None:
        self.__scan_display_items_model.clean("subscan")
        self.__scan_display_items_model.refilter()
        self.__update_subscan_region()

    # implement methods for the graphic set handler

    def _graphic_removed(self, subscan_graphic: Graphics.Graphic) -> None:
        # clear subscan state
        self.__stem_controller.subscan_state = SubscanState.DISABLED
        self.__stem_controller.subscan_region = None
        self.__stem_controller.subscan_rotation = 0

    def _create_graphic(self) -> Graphics.RectangleGraphic:
        subscan_graphic = Graphics.RectangleGraphic()
        subscan_graphic.graphic_id = "subscan"
        subscan_graphic.label = _("Subscan")
        subscan_graphic.bounds = self.__stem_controller.subscan_region or Geometry.FloatRect.empty_rect()
        subscan_graphic.rotation = self.__stem_controller.subscan_rotation
        subscan_graphic.is_bounds_constrained = True
        return subscan_graphic

    def _update_graphic(self, subscan_graphic: Graphics.Graphic) -> None:
        assert isinstance(subscan_graphic, Graphics.RectangleGraphic)
        subscan_region = self.__stem_controller.subscan_region
        if subscan_region and subscan_graphic.bounds != subscan_region:
            subscan_graphic.bounds = subscan_region
        if subscan_graphic.rotation != self.__stem_controller.subscan_rotation:
            subscan_graphic.rotation = self.__stem_controller.subscan_rotation

    def _graphic_property_changed(self, subscan_graphic: Graphics.Graphic, name: str) -> None:
        if name == "bounds":
            assert isinstance(subscan_graphic, Graphics.RectangleGraphic)
            self.__stem_controller.subscan_region = Geometry.FloatRect.make(subscan_graphic.bounds)
        if name == "rotation":
            assert isinstance(subscan_graphic, Graphics.RectangleGraphic)
            self.__stem_controller.subscan_rotation = subscan_graphic.rotation

    def get_dependents(self, item: Graphics.Graphic) -> typing.Sequence[Graphics.Graphic]:
        graphics = self.__graphic_set.graphics
        if item in graphics:
            return list(set(graphics) - {item})
        return list()


class LineScanView(EventLoopMonitor, AbstractGraphicSetHandler, DocumentModel.AbstractImplicitDependency):
    """Observes the STEM controller and updates data items and graphics."""
    count = 0

    def __init__(self, stem_controller: STEMController, document_model: DocumentModel.DocumentModel, event_loop: asyncio.AbstractEventLoop):
        super().__init__(document_model, event_loop)
        self.__class__.count += 1
        self.__stem_controller = stem_controller
        self.__document_model = document_model
        self.__scan_display_items_model = make_scan_display_item_list_model(document_model, stem_controller)
        self.__project_loaded_event_listener = document_model.project_loaded_event.listen(ReferenceCounting.weak_partial(LineScanView.__refilter_scan_display_items, self))
        self.__graphic_set = GraphicSetController(self)
        # note: these property changed listeners can all possibly be fired from a thread.
        self.__line_scan_vector_changed_listener = stem_controller.property_changed_event.listen(self.__line_scan_vector_changed)
        self.__document_model.register_implicit_dependency(self)
        # update in case a new document model is opened with the line scan already enabled
        self.__update_line_scan_vector()

    def close(self) -> None:
        self._mark_closed()
        self.__document_model.unregister_implicit_dependency(self)
        self.__graphic_set.close()
        self.__graphic_set = typing.cast(typing.Any, None)
        self.__scan_display_items_model.close()
        self.__scan_display_items_model = typing.cast(typing.Any, None)
        self.__document_model = typing.cast(typing.Any, None)
        self.__stem_controller = typing.cast(typing.Any, None)
        self.__project_loaded_event_listener = typing.cast(typing.Any, None)
        self.__class__.count -= 1

    def _unlisten(self) -> None:
        # unlisten to the event loop dependent listeners
        self.__line_scan_vector_changed_listener.close()
        self.__line_scan_vector_changed_listener = typing.cast(typing.Any, None)

    # methods for handling changes to the line scan region

    def __line_scan_vector_changed(self, name: str) -> None:
        # must be thread safe
        if name == "line_scan_vector":
            self._call_soon_threadsafe(self.__update_line_scan_vector)

    def __update_line_scan_vector(self) -> None:
        assert threading.current_thread() == threading.main_thread()
        if self.__stem_controller.line_scan_vector:
            self.__graphic_set.synchronize_graphics(self.__scan_display_items_model.display_items)
        else:
            self.__graphic_set.remove_all_graphics()

    def __refilter_scan_display_items(self) -> None:
        self.__scan_display_items_model.clean("line_scan")
        self.__scan_display_items_model.refilter()
        self.__update_line_scan_vector()

    # implement methods for the graphic set handler

    def _graphic_removed(self, line_scan_graphic: Graphics.Graphic) -> None:
        # clear line scan state
        self.__stem_controller.line_scan_state = LineScanState.DISABLED
        self.__stem_controller.line_scan_vector = None

    def _create_graphic(self) -> Graphics.LineGraphic:
        line_scan_graphic = Graphics.LineGraphic()
        line_scan_graphic.graphic_id = "line_scan"
        line_scan_graphic.label = _("Line Scan")
        self._update_graphic(line_scan_graphic)
        line_scan_graphic.is_bounds_constrained = True
        return line_scan_graphic

    def _update_graphic(self, line_scan_graphic: Graphics.Graphic) -> None:
        assert isinstance(line_scan_graphic, Graphics.LineTypeGraphic)
        line_scan_vector = self.__stem_controller.line_scan_vector
        graphic_vector = line_scan_graphic.vector
        graphic_vector_tuple = graphic_vector[0].as_tuple(), graphic_vector[1].as_tuple()
        if line_scan_vector and graphic_vector_tuple != line_scan_vector:
            line_scan_graphic.vector = Geometry.FloatPoint.make(line_scan_vector[0]), Geometry.FloatPoint.make(line_scan_vector[1])

    def _graphic_property_changed(self, line_scan_graphic: Graphics.Graphic, name: str) -> None:
        if name == "vector":
            assert isinstance(line_scan_graphic, Graphics.LineTypeGraphic)
            graphic_vector = line_scan_graphic.vector
            graphic_vector_tuple = graphic_vector[0].as_tuple(), graphic_vector[1].as_tuple()
            self.__stem_controller.line_scan_vector = graphic_vector_tuple

    def get_dependents(self, item: Graphics.Graphic) -> typing.Sequence[Graphics.Graphic]:
        graphics = self.__graphic_set.graphics
        if item in graphics:
            return list(set(graphics) - {item})
        return list()


class DriftView(EventLoopMonitor):
    """Observes the STEM controller and updates drift data item and graphic."""
    count = 0

    def __init__(self, stem_controller: STEMController, document_model: DocumentModel.DocumentModel, event_loop: asyncio.AbstractEventLoop):
        super().__init__(document_model, event_loop)
        self.__class__.count += 1
        self.__stem_controller = stem_controller
        self.__document_model = document_model
        self.__scan_display_items_model = make_scan_display_item_list_model(document_model, stem_controller)
        self.__project_loaded_event_listener = document_model.project_loaded_event.listen(ReferenceCounting.weak_partial(DriftView.__clean, self))
        self.__graphic_display_item : typing.Optional[DisplayItem.DisplayItem] = None
        self.__graphic : typing.Optional[Graphics.RectangleGraphic] = None
        self.__graphic_property_changed_listener: typing.Optional[Event.EventListener] = None
        self.__graphic_about_to_be_removed_listener: typing.Optional[Event.EventListener] = None
        # note: these property changed listeners can all possibly be fired from a thread.
        self.__scan_context_data_items_changed_listener = stem_controller.scan_context_data_items_changed_event.listen(self.__scan_context_data_items_changed)
        self.__drift_channel_id_changed_listener = stem_controller.property_changed_event.listen(self.__drift_channel_id_changed)
        self.__drift_region_changed_listener = stem_controller.property_changed_event.listen(self.__drift_region_changed)
        self.__drift_rotation_changed_listener = stem_controller.property_changed_event.listen(self.__drift_rotation_changed)
        # update in case a new document model is opened with the line scan already enabled
        self.__update_drift_region()

    def close(self) -> None:
        self._mark_closed()
        if self.__graphic_property_changed_listener:
            self.__graphic_property_changed_listener.close()
            self.__graphic_property_changed_listener = None
        if self.__graphic_about_to_be_removed_listener:
            self.__graphic_about_to_be_removed_listener.close()
            self.__graphic_about_to_be_removed_listener = None
        self.__graphic_display_item = None
        self.__graphic = None
        self.__scan_display_items_model.close()
        self.__scan_display_items_model = typing.cast(typing.Any, None)
        self.__document_model = typing.cast(typing.Any, None)
        self.__stem_controller = typing.cast(typing.Any, None)
        self.__project_loaded_event_listener = typing.cast(typing.Any, None)
        self.__class__.count -= 1

    def _unlisten(self) -> None:
        # unlisten to the event loop dependent listeners
        self.__drift_region_changed_listener.close()
        self.__drift_region_changed_listener = typing.cast(typing.Any, None)
        self.__drift_rotation_changed_listener.close()
        self.__drift_rotation_changed_listener = typing.cast(typing.Any, None)
        self.__drift_channel_id_changed_listener.close()
        self.__drift_channel_id_changed_listener = typing.cast(typing.Any, None)
        self.__scan_context_data_items_changed_listener.close()
        self.__scan_context_data_items_changed_listener = typing.cast(typing.Any, None)

    def __scan_context_data_items_changed(self) -> None:
        # must be thread safe
        self._call_soon_threadsafe(self.__update_drift_region)

    # methods for handling changes to the drift region

    def __drift_channel_id_changed(self, name: str) -> None:
        # must be thread safe
        if name == "drift_channel_id":
            self._call_soon_threadsafe(self.__update_drift_region)

    def __drift_region_changed(self, name: str) -> None:
        # must be thread safe
        if name == "drift_region":
            self._call_soon_threadsafe(self.__update_drift_region)

    def __drift_rotation_changed(self, name: str) -> None:
        # must be thread safe
        if name == "drift_rotation":
            self._call_soon_threadsafe(self.__update_drift_region)

    def __update_drift_region(self) -> None:
        assert threading.current_thread() == threading.main_thread()
        if self.__stem_controller.drift_channel_id:
            scan_controller = self.__stem_controller.scan_controller
            assert scan_controller
            data_item_channel_reference = self.__document_model.get_data_item_channel_reference(scan_controller.hardware_source_id, self.__stem_controller.drift_channel_id)
            drift_data_item = data_item_channel_reference.display_item.data_item if data_item_channel_reference and data_item_channel_reference.display_item else None
        else:
            drift_data_item = None
        # determine if a new graphic should exist and if it exists already
        drift_region = self.__stem_controller.drift_region
        if self.__stem_controller.drift_channel_id and drift_region and drift_data_item:
            drift_display_item = self.__document_model.get_display_item_for_data_item(drift_data_item)
            # remove the graphic if it already exists on the wrong display item
            if self.__graphic and (not drift_display_item or not self.__graphic in drift_display_item.graphics):
                self.__remove_graphic()
            # it already exists on the correct display item, update it.
            if self.__graphic:
                # only fire messages when something changes to avoid flickering, difficulty editing.
                if self.__graphic.bounds != drift_region:
                    self.__graphic.bounds = drift_region
                if self.__graphic.rotation != self.__stem_controller.drift_rotation:
                    self.__graphic.rotation = self.__stem_controller.drift_rotation
            # otherwise create it if there is a display item for it
            elif drift_display_item:
                drift_graphic = Graphics.RectangleGraphic()
                drift_graphic.graphic_id = "drift"
                drift_graphic.label = _("Drift")
                drift_graphic.bounds = drift_region
                drift_graphic.rotation = self.__stem_controller.drift_rotation
                drift_graphic.is_bounds_constrained = True
                drift_graphic.color = "#F0F"  # purple
                drift_display_item.add_graphic(drift_graphic)
                self.__graphic_display_item = drift_display_item
                self.__graphic = drift_graphic
                self.__graphic_property_changed_listener = self.__graphic.property_changed_event.listen(self.__graphic_property_changed)
                self.__graphic_about_to_be_removed_listener = self.__graphic.about_to_be_removed_event.listen(self.__graphic_about_to_be_removed)
            # otherwise do nothing, graphic is removed and not tracked.
        else:
            # either no drift channel_id or drift region - so remove the graphic
            if self.__graphic:
                self.__remove_graphic()

    def __remove_graphic(self) -> None:
        if self.__graphic_property_changed_listener:
            self.__graphic_property_changed_listener.close()
            self.__graphic_property_changed_listener = None
        if self.__graphic_about_to_be_removed_listener:
            self.__graphic_about_to_be_removed_listener.close()
            self.__graphic_about_to_be_removed_listener = None
        if self.__graphic_display_item and self.__graphic:
            self.__graphic_display_item.remove_graphic(self.__graphic).close()
            self.__graphic_display_item = None
        self.__graphic = None

    def __graphic_about_to_be_removed(self) -> None:
        # clear drift state
        self.__stem_controller.drift_channel_id = None
        self.__stem_controller.drift_region = None
        self.__stem_controller.drift_rotation = 0

    def __graphic_property_changed(self, key: str) -> None:
        assert self.__graphic
        if key == "bounds":
            self.__stem_controller.drift_region = Geometry.FloatRect.make(self.__graphic.bounds)
        if key == "rotation":
            self.__stem_controller.drift_rotation = self.__graphic.rotation

    def __clean(self) -> None:
        self.__scan_display_items_model.clean("drift")
        self.__update_drift_region()


class ScanContextController:
    """Manage probe view, subscan, and drift area for each instrument (STEMController) that gets registered."""
    count = 0

    def __init__(self, document_model: DocumentModel.DocumentModel, event_loop: asyncio.AbstractEventLoop) -> None:
        self.__class__.count += 1
        assert event_loop is not None
        self.__document_model = document_model
        self.__event_loop = event_loop
        self.__m: typing.Dict[typing.Any, typing.Dict[typing.Any, typing.Any]] = dict()
        # be sure to keep a reference or it will be closed immediately.
        self.__instrument_added_event_listener = HardwareSource.HardwareSourceManager().instrument_added_event.listen(self.register_instrument)
        self.__instrument_removed_event_listener = HardwareSource.HardwareSourceManager().instrument_removed_event.listen(self.unregister_instrument)
        for instrument in HardwareSource.HardwareSourceManager().instruments:
            self.register_instrument(instrument)

    def close(self) -> None:
        # any instrument that was registered needs to be unregistered.
        for instrument in HardwareSource.HardwareSourceManager().instruments:
            self.unregister_instrument(instrument)
        self.__instrument_added_event_listener.close()
        self.__instrument_added_event_listener = typing.cast(typing.Any, None)
        self.__instrument_removed_event_listener.close()
        self.__instrument_removed_event_listener = typing.cast(typing.Any, None)
        self.__class__.count -= 1

    def register_instrument(self, instrument: typing.Any) -> None:
        # if this is a stem controller, add a probe view
        if hasattr(instrument, "probe_position"):
            self.__m.setdefault(instrument, dict())["probe_view"] = ProbeView(instrument, self.__document_model, self.__event_loop)
        if hasattr(instrument, "subscan_region"):
            self.__m.setdefault(instrument, dict())["subscan_view"] = SubscanView(instrument, self.__document_model, self.__event_loop)
        if hasattr(instrument, "line_scan_vector"):
            self.__m.setdefault(instrument, dict())["line_scan_view"] = LineScanView(instrument, self.__document_model, self.__event_loop)
        if hasattr(instrument, "drift_region"):
            self.__m.setdefault(instrument, dict())["drift_view"] = DriftView(instrument, self.__document_model, self.__event_loop)

    def unregister_instrument(self, instrument: typing.Any) -> None:
        probe_view = self.__m.get(instrument, dict()).pop("probe_view")
        if probe_view:
            probe_view.close()
        subscan_view = self.__m.get(instrument, dict()).pop("subscan_view")
        if subscan_view:
            subscan_view.close()
        line_scan_view = self.__m.get(instrument, dict()).pop("line_scan_view")
        if line_scan_view:
            line_scan_view.close()
        drift_view = self.__m.get(instrument, dict()).pop("drift_view")
        if drift_view:
            drift_view.close()


# the plan is to migrate away from the hardware manager as a registration system.
# but keep the hardware source manager registrations here until that migration is complete.

_scan_context_controllers: typing.Dict[uuid.UUID, ScanContextController] = dict()
_event_loop: typing.Optional[asyncio.AbstractEventLoop] = None
_pending_document_models: typing.List[DocumentModel.DocumentModel] = list()


def component_registered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
    if "stem_controller" in component_types:
        HardwareSource.HardwareSourceManager().register_instrument(component.instrument_id, component)
    if "document_model" in component_types:
        document_model = typing.cast(DocumentModel.DocumentModel, component)
        if _event_loop:
            _scan_context_controllers[document_model.uuid] = ScanContextController(document_model, _event_loop)
        else:
            _pending_document_models.append(document_model)


def component_unregistered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
    if "stem_controller" in component_types:
        HardwareSource.HardwareSourceManager().unregister_instrument(component.instrument_id)
    if "document_model" in component_types:
        document_model = typing.cast(DocumentModel.DocumentModel, component)
        if _event_loop:
            _scan_context_controllers.pop(document_model.uuid).close()
        else:
            _pending_document_models.remove(document_model)


component_registered_listener = Registry.listen_component_registered_event(component_registered)
component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

for component in Registry.get_components_by_type("stem_controller"):
    component_registered(component, {"stem_controller"})


def register_event_loop(event_loop: asyncio.AbstractEventLoop) -> None:
    global _event_loop
    _event_loop = event_loop
    for document_model in _pending_document_models:
        _scan_context_controllers[document_model.uuid] = ScanContextController(document_model, _event_loop)
    _pending_document_models.clear()


def unregister_event_loop() -> None:
    global _event_loop
    for scan_context_controller in _scan_context_controllers.values():
        scan_context_controller.close()
    _scan_context_controllers.clear()
    _event_loop = None
