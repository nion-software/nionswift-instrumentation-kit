from __future__ import annotations

# standard libraries
import copy
import typing

import numpy.typing

from nion.instrumentation import HardwareSource
from nion.instrumentation import stem_controller
from nion.utils import Event
from nion.utils import Geometry

if typing.TYPE_CHECKING:
    from nion.device_kit import ScanDevice


_NDArray = numpy.typing.NDArray[typing.Any]


class ValueManagerLike(typing.Protocol):
    def get_value(self, name: str) -> typing.Optional[float]:
        ...

    def set_value(self, name: str, value: float) -> bool:
        ...

    def inform_value(self, name: str, value: float) -> bool:
        ...

    def get_value_2d(self, s: str, default_value: typing.Optional[Geometry.FloatPoint] = None, *, axis: typing.Optional[stem_controller.AxisType] = None) -> Geometry.FloatPoint:
        ...


    def set_value_2d(self, s: str, value: Geometry.FloatPoint, *, axis: typing.Optional[stem_controller.AxisType] = None) -> bool:
        ...

    def inform_control_2d(self, s: str, value: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        ...

    def get_reference_setting_index(self, settings_control: str) -> int:
        ...


class AxisManagerLike(typing.Protocol):

    @property
    def supported_axis_descriptions(self) -> typing.Sequence[stem_controller.AxisDescription]:
        raise NotImplementedError()

    def axis_transform_point(self, point: Geometry.FloatPoint, from_axis: stem_controller.AxisDescription, to_axis: stem_controller.AxisDescription) -> Geometry.FloatPoint:
        ...


class Instrument(stem_controller.STEMController):
    """
    TODO: add temporal supersampling for cameras (to produce blurred data when things are changing).
    """

    def __init__(self, instrument_id: str, value_manager: ValueManagerLike, axis_manager: AxisManagerLike) -> None:
        super().__init__()
        self.priority = 20
        self.instrument_id = instrument_id
        self.property_changed_event = Event.Event()

        self.__value_manager = value_manager
        self.__axis_manager = axis_manager

        # define the STEM geometry limits
        self.stage_size_nm = 1000
        self.max_defocus = 5000 / 1E9

        self.__ronchigram_shape = Geometry.IntSize(2048, 2048)
        self.__eels_shape = Geometry.IntSize(256, 1024)
        self.__probe_position: typing.Optional[Geometry.FloatPoint] = None
        self.__live_probe_position: typing.Optional[Geometry.FloatPoint] = None
        self._is_synchronized = False

    def _get_config_property(self, name: str) -> typing.Any:
        if name in ("stage_size_nm", "max_defocus"):
            return getattr(self, name)
        raise AttributeError()

    def _set_config_property(self, name: str, value: typing.Any) -> None:
        if name in ("stage_size_nm", "max_defocus"):
            return setattr(self, name, value)
        raise AttributeError()

    @property
    def value_manager(self) -> ValueManagerLike:
        return self.__value_manager

    @property
    def axis_manager(self) -> AxisManagerLike:
        return self.__axis_manager

    @property
    def live_probe_position(self) -> typing.Optional[Geometry.FloatPoint]:
        return self.__live_probe_position

    @live_probe_position.setter
    def live_probe_position(self, position: typing.Optional[Geometry.FloatPoint]) -> None:
        self.__live_probe_position = position
        self.property_changed_event.fire("live_probe_position")

    def _set_scan_context_probe_position(self, scan_context: stem_controller.ScanContext, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        self.__probe_position = probe_position

    def _enter_synchronized_state(self, scan_controller: HardwareSource.HardwareSource, *, camera: typing.Optional[HardwareSource.HardwareSource] = None) -> None:
        self._is_synchronized = True

    def _exit_synchronized_state(self, scan_controller: HardwareSource.HardwareSource, *, camera: typing.Optional[HardwareSource.HardwareSource] = None) -> None:
        self._is_synchronized = False

    def camera_sensor_dimensions(self, camera_type: str) -> typing.Tuple[int, int]:
        if camera_type == "ronchigram":
            return self.__ronchigram_shape[0], self.__ronchigram_shape[1]
        else:
            return self.__eels_shape[0], self.__eels_shape[1]

    def camera_readout_area(self, camera_type: str) -> typing.Tuple[int, int, int, int]:
        # returns readout area TLBR
        if camera_type == "ronchigram":
            return 0, 0, self.__ronchigram_shape[0], self.__ronchigram_shape[1]
        else:
            return 0, 0, self.__eels_shape[0], self.__eels_shape[1]

    @property
    def counts_per_electron(self) -> int:
        return 40

    def get_electrons_per_pixel(self, pixel_count: int, exposure_s: float) -> float:
        beam_current_pa = self.GetVal("BeamCurrent") * 1E12
        e_per_pa = 6.241509074E18 / 1E12
        beam_e = beam_current_pa * e_per_pa
        e_per_pixel_per_second = beam_e / pixel_count
        return e_per_pixel_per_second * exposure_s

    @property
    def stage_position_m(self) -> Geometry.FloatPoint:
        return self.GetVal2D("stage_position_m")

    @stage_position_m.setter
    def stage_position_m(self, value: Geometry.FloatPoint) -> None:
        self.SetVal2D("stage_position_m", value)

    @property
    def defocus_m(self) -> float:
        return self.GetVal("C10")

    @defocus_m.setter
    def defocus_m(self, value: float) -> None:
        self.SetVal("C10", value)

    @property
    def voltage(self) -> float:
        return self.GetVal("EHT")

    @voltage.setter
    def voltage(self, value: float) -> None:
        self.SetVal("EHT", value)
        self.property_changed_event.fire("voltage")

    @property
    def energy_offset_eV(self) -> float:
        return self.GetVal("ZLPoffset")

    @energy_offset_eV.setter
    def energy_offset_eV(self, value: float) -> None:
        self.SetVal("ZLPoffset", value)
        self.property_changed_event.fire("energy_offset_eV")

    def get_autostem_properties(self) -> typing.Mapping[str, typing.Any]:
        """Return a new autostem properties (dict) to be recorded with an acquisition.

           * use property names that are lower case and separated by underscores
           * use property names that include the unit attached to the end
           * avoid using abbreviations
           * avoid adding None entries
           * dict must be serializable using json.dumps(dict)

           Be aware that these properties may be used far into the future so take care when designing additions and
           discuss/review with team members.
        """
        return {
            "high_tension": self.voltage,
            "defocus": self.defocus_m,
        }

    # these are required functions to implement the standard stem controller interface.

    def TryGetVal(self, s: str) -> typing.Tuple[bool, typing.Optional[float]]:
        value = self.__value_manager.get_value(s)
        return value is not None, value

    def GetVal(self, s: str, default_value: typing.Optional[float] = None) -> float:
        good, d = self.TryGetVal(s)
        if not good or d is None:
            if default_value is None:
                raise Exception(f"No element named '{s}' exists! Cannot get value.")
            else:
                return default_value
        return d

    def SetVal(self, s: str, val: float) -> bool:
        return self.__value_manager.set_value(s, val)

    def SetValWait(self, s: str, val: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValAndConfirm(self, s: str, val: float, tolfactor: float, timeout_ms: int) -> bool:
        return self.SetVal(s, val)

    def SetValDelta(self, s: str, delta: float) -> bool:
        return self.SetVal(s, self.GetVal(s) + delta)

    def SetValDeltaAndConfirm(self, s: str, delta: float, tolfactor: float, timeout_ms: int) -> bool:
        return self.SetValAndConfirm(s, self.GetVal(s) + delta, tolfactor, timeout_ms)

    def InformControl(self, s: str, val: float) -> bool:
        return self.__value_manager.inform_value(s, val)

    def GetVal2D(self, s: str, default_value: typing.Optional[Geometry.FloatPoint] = None, *, axis: typing.Optional[stem_controller.AxisType] = None) -> Geometry.FloatPoint:
        return self.__value_manager.get_value_2d(s, default_value, axis=axis)

    def SetVal2D(self, s: str, value: Geometry.FloatPoint, *, axis: typing.Optional[stem_controller.AxisType] = None) -> bool:
        return self.__value_manager.set_value_2d(s, value, axis=axis)

    def SetVal2DAndConfirm(self, s: str, value: Geometry.FloatPoint, tolfactor: float, timeout_ms: int, *, axis: stem_controller.AxisType) -> bool:
        return self.SetVal2D(s, value, axis=axis)

    def SetVal2DDelta(self, s: str, delta: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        return self.SetVal2D(s, self.GetVal2D(s, axis=axis) + delta, axis=axis)

    def SetVal2DDeltaAndConfirm(self, s: str, delta: Geometry.FloatPoint, tolfactor: float, timeout_ms: int, *, axis: stem_controller.AxisType) -> bool:
        return self.SetVal2DAndConfirm(s, self.GetVal2D(s, axis=axis) + delta, tolfactor, timeout_ms, axis=axis)

    def InformControl2D(self, s: str, value: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        return self.__value_manager.inform_control_2d(s, value, axis=axis)

    def HasValError(self, s: str) -> bool:
        return False

    @property
    def axis_descriptions(self) -> typing.Sequence[stem_controller.AxisDescription]:
        return self.__axis_manager.supported_axis_descriptions

    def get_reference_setting_index(self, settings_control: str) -> int:
        return self.__value_manager.get_reference_setting_index(settings_control)

    def axis_transform_point(self, point: Geometry.FloatPoint, from_axis: stem_controller.AxisDescription, to_axis: stem_controller.AxisDescription) -> Geometry.FloatPoint:
        return self.__axis_manager.axis_transform_point(point, from_axis, to_axis)

    def change_stage_position(self, *, dy: typing.Optional[float] = None, dx: typing.Optional[float] = None) -> None:
        """Shift the stage by dx, dy (meters). Do not wait for confirmation."""
        dx = dx or 0
        dy = dy or 0
        self.stage_position_m += Geometry.FloatPoint(y=-dy, x=-dx)

    def change_pmt_gain(self, pmt_type: stem_controller.PMTType, *, factor: float) -> None:
        """Change specified PMT by factor. Do not wait for confirmation."""
        pass
