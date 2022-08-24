from __future__ import annotations

# system imports
import copy
import gettext
import pathlib
import typing

# local libraries
from nion.swift.model import ApplicationData
from nion.swift.model import Schema
from nion.utils import Observable
from nion.utils import Recorder


_ = gettext.gettext


class ControlDescription(Observable.Observable):
    # describes a control. entity-compatible
    # control_id is an identifier for this control, may include periods and dashes
    # display_name is control name as displayed in the UI
    # device_id is the device identifier; a specific device or general, i.e. "stem", "camera", or "scan"
    # device_control_name control name on the device. may be case sensitive, depending on device
    # control_type is "1d", "2d", or "bool"
    # default_value is initial value
    # multiplier is conversion from intrinsic control unit to SI
    # units is displayed unit (use names from https://github.com/hgrecco/pint/blob/master/pint/default_en.txt)
    # delay applied after control is set but before acquisition proceeds
    # native axis identifier, specific to device
    def __init__(self, control_id: str, name: str, device_id: str, device_control_id: str, control_type: str,
                 default_value: typing.Any, multiplier: float, units: str, delay: float, axis: typing.Optional[str]):
        super().__init__()
        self.control_id = control_id
        self.name = name
        self.device_id = device_id
        self.__device_control_id = device_control_id
        self.control_type = control_type
        self.default_value = default_value
        self.multiplier = multiplier
        self.units = units
        self.__delay = delay
        self.axis = axis

    @property
    def is_customizable(self) -> bool:
        return self.device_id == "stem"

    @property
    def device_control_id(self) -> str:
        return self.__device_control_id

    @device_control_id.setter
    def device_control_id(self, value: str) -> None:
        self.__device_control_id = value
        self.notify_property_changed("device_control_id")

    @property
    def delay(self) -> float:
        return self.__delay

    @delay.setter
    def delay(self, value: float) -> None:
        self.__delay = value
        self.notify_property_changed("delay")


acquisition_controls = [
    ControlDescription("blanker", _("Blanking"), "stem", "C_Blank", "bool", False, 1, "", 0.05, None),
    ControlDescription("defocus", _("Defocus"), "stem", "C10", "1d", 500E-9, 1, "nm", 0.05, None),
    ControlDescription("control_nm", _("Control 1E-9"), "stem", "ControlName", "1d", 1E-9, 1, "nm", 0.05, None),
    ControlDescription("control_um", _("Control 1E-6"), "stem", "ControlName", "1d", 1E-6, 1, "um", 0.05, None),
    ControlDescription("control_mm", _("Control 1E-3"), "stem", "ControlName", "1d", 1E-3, 1, "mm", 0.05, None),
    ControlDescription("control_m", _("Control"), "stem", "ControlName", "1d", 1, 1, "m", 0.05, None),
    ControlDescription("control2_nm", _("Control 2D 1E-9"), "stem", "ControlName", "2d", (0, 0), 1, "nm", 0.05, "tv"),
    ControlDescription("control2_um", _("Control 2D 1E-6"), "stem", "ControlName", "2d", (0, 0), 1, "um", 0.05, "tv"),
    ControlDescription("control2_mm", _("Control 2D 1E-3"), "stem", "ControlName", "2d", (0, 0), 1, "mm", 0.05, "tv"),
    ControlDescription("control2_m", _("Control 2D"), "stem", "ControlName", "2d", (0, 0), 1, "m", 0.05, "tv"),
    ControlDescription("field_of_view", _("FoV"), "magnification", "fov_nm", "1d", 100E-9, 1E9, "nm", 0.00, None),
    ControlDescription("energy_offset", _("Energy Offset"), "stem", "EELS_MagneticShift_Offset", "1d", 0, 1, "eV", 0.05, None),
    ControlDescription("stage_position", _("Stage Position"), "stem", "stage_position_m", "2d", (0, 0), 1, "nm", 0.00, "tv"),
    ControlDescription("exposure", _("Exposure"), "camera", "exposure_ms", "1d", 100E-3, 1E3, "ms", 0.00, None),
]


class ControlCustomization(Schema.Entity):
    def __init__(self, entity_type: Schema.EntityType, context: typing.Optional[Schema.EntityContext]):
        super().__init__(entity_type, context)

    def __str__(self) -> str:
        return self.name

    @property
    def control_description(self) -> typing.Optional[ControlDescription]:
        for acquisition_control in acquisition_controls:
            if acquisition_control.control_id == self.control_id:
                return acquisition_control
        return None

    @property
    def control_id(self) -> str:
        return typing.cast(str, self._get_field_value("control_id"))

    @property
    def name(self) -> str:
        control_description = self.control_description
        return control_description.name if control_description else _("N/A")

    @property
    def is_customizable(self) -> bool:
        control_description = self.control_description
        return control_description.is_customizable if control_description else False


ControlCustomizationSchema = Schema.entity("control_customization", None, None, {
    "control_id": Schema.prop(Schema.STRING),
    "device_control_id": Schema.prop(Schema.STRING),
    "delay": Schema.prop(Schema.FLOAT),
}, ControlCustomization)

default_drift_frame_parameters = {
                "scan_width_pixels": 64,
                "dwell_time_us": 16.0}

# Create an entity for customizing the drift scan that is created when "Drift correct" is checked in the ScanControlPanel
DriftFrameParameters = Schema.entity("drift_frame_parameters", None, None, {
    "scan_width_pixels": Schema.prop(Schema.INT),
    "dwell_time_us": Schema.prop(Schema.FLOAT),
})

AcquisitionPreferencesSchema = Schema.entity("acquisition_preferences", None, None, {
    "control_customizations": Schema.array(Schema.component(ControlCustomizationSchema)),
    "drift_scan_customization": Schema.component(DriftFrameParameters),
})


class DictRecorderLoggerDictInterface(typing.Protocol):
    def get_data_dict(self) -> typing.Dict[str, typing.Any]: ...
    def set_data_dict(self, d: typing.Mapping[str, typing.Any]) -> None: ...


class DictRecorderLogger(Recorder.RecorderLogger):
    def __init__(self, field: Schema.Field, app_data: DictRecorderLoggerDictInterface):
        super().__init__()
        self.__app_data = app_data
        self.__field = field
        self.__d = copy.deepcopy(app_data.get_data_dict())

    def __resolve_accessor(self, field: Schema.Field, d: typing.Any, accessor: Recorder.Accessor) -> typing.Tuple[Schema.Field, typing.Any]:
        if isinstance(accessor, Recorder.DirectAccessor):
            return field, d
        elif isinstance(accessor, Recorder.KeyAccessor):
            field, d = self.__resolve_accessor(field, d, accessor.accessor)
            return field.field_by_key(accessor.key), d[accessor.key]
        elif isinstance(accessor, Recorder.IndexAccessor):
            field, d = self.__resolve_accessor(field, d, accessor.accessor)
            return field.field_by_index(accessor.index), d[accessor.index]
        assert False, f"Unknown accessor type ({accessor})."

    def append(self, recorder_entry: Recorder.RecorderEntry) -> None:
        super().append(recorder_entry)
        if isinstance(recorder_entry, Recorder.KeyRecorderEntry):
            field, d = self.__resolve_accessor(self.__field, self.__d, recorder_entry.accessor)
            field = field.field_by_key(recorder_entry.key)
            d[recorder_entry.key] = field.write()
            self.__app_data.set_data_dict(self.__d)
        elif isinstance(recorder_entry, Recorder.InsertRecorderEntry):
            field, d = self.__resolve_accessor(self.__field, self.__d, recorder_entry.accessor)
            field = field.field_by_key(recorder_entry.key).field_by_index(recorder_entry.index)
            d.setdefault(recorder_entry.key, list()).insert(recorder_entry.index, field.write())
            self.__app_data.set_data_dict(self.__d)
        elif isinstance(recorder_entry, Recorder.RemoveRecorderEntry):
            field, d = self.__resolve_accessor(self.__field, self.__d, recorder_entry.accessor)
            d[recorder_entry.key].pop(recorder_entry.index)
            self.__app_data.set_data_dict(self.__d)
        else:
            assert False, f"Unknown recorder entry ({recorder_entry})"


class AcquisitionPreferences(Schema.Entity):
    def __init__(self, app_data: DictRecorderLoggerDictInterface):
        super().__init__(AcquisitionPreferencesSchema)
        self.read_from_dict(app_data.get_data_dict())
        field = Schema.ComponentField(None, self.entity_type.entity_id)
        field.set_field_value(None, self)
        self.__logger = DictRecorderLogger(field, app_data)
        self.__recorder = Recorder.Recorder(self, None, self.__logger)

    def close(self) -> None:
        self.__recorder.close()
        self.__recorder = typing.cast(typing.Any, None)
        super().close()

    def _create(self, context: typing.Optional[Schema.EntityContext]) -> Schema.Entity:
        raise NotImplementedError()


acquisition_preferences: typing.Optional[AcquisitionPreferences] = None


def init_acquisition_preferences(file_path: pathlib.Path) -> None:
    global acquisition_preferences
    acquisition_preferences = AcquisitionPreferences(ApplicationData.ApplicationData(file_path))
    assert acquisition_preferences
    # determine missing controls. build the master list of controls. remove the ones already there. add remaining.
    control_id_set = {control_description.control_id for control_description in acquisition_controls}
    for control_customization in acquisition_preferences.control_customizations:
        control_id_set.discard(control_customization.control_id)
    for control_description in acquisition_controls:
        if control_description.control_id in control_id_set:
            acquisition_preferences._append_item("control_customizations", ControlCustomizationSchema.create(None, {
                "control_id": control_description.control_id,
                "device_control_id": control_description.device_control_id, "delay": control_description.delay}))
    # We want to reset to defaults after each reastart of Swift, so simply always set "drift_scan_customization"
    # Uncomment the line below to return to persistent saving.
    # if not acquisition_preferences.drift_scan_customization:
    acquisition_preferences._set_field_value("drift_scan_customization", DriftFrameParameters.create(None, default_drift_frame_parameters))



def deinit_acquisition_preferences() -> None:
    global acquisition_preferences
    acquisition_preferences = None
