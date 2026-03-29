import asyncio
import dataclasses
import enum
import json
import logging
import pkgutil
import time
import typing

from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import camera_base as CameraBase
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base as ScanBase
from nion.instrumentation import stem_controller as STEMController
from nion.swift import DocumentController
from nion.swift.model import DataItem
from nionswift_plugin.nion_instrumentation_ui import AcquisitionPanel
from nion.ui import Declarative
from nion.ui import Dialog
from nion.utils import Converter
from nion.utils import Model
from nion.utils import Observable
from nion.utils import Registry
from nion.utils import Stream


class AcquisitionTestState(enum.Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    SUCCEEDED = 2
    FAILED = 3
    NOT_AVAILABLE = 4


class AcquisitionTestResultMessage:
    def __init__(self, status: bool = False, message: str = str(), state: AcquisitionTestState | None = None) -> None:
        self.status = state if state is not None else (AcquisitionTestState.SUCCEEDED if status else AcquisitionTestState.FAILED)
        self.message = message


class AcquisitionTestResult:
    def __init__(self) -> None:
        self.messages = list[AcquisitionTestResultMessage]()
        self.result: AcquisitionPanel._AcquireResult | None = None

    def delete_display_items(self, window: DocumentController.DocumentController) -> None:
        ar = self.result
        assert ar
        data_item_map = ar.data_item_map
        for channel in data_item_map.keys():
            data_item = data_item_map[channel]
            if data_item:
                display_item = window.document_model.get_display_item_for_data_item(data_item)
                if display_item:
                    window.delete_display_items([display_item])


def make_not_available_result(message: str) -> AcquisitionTestResult:
    result = AcquisitionTestResult()
    result.messages.append(AcquisitionTestResultMessage(message=message, state=AcquisitionTestState.NOT_AVAILABLE))
    return result


PersistentDictType = typing.Mapping[str, typing.Any]


def validate_keys(cls: typing.Any, d: PersistentDictType) -> None:
    field_names = {f.name for f in dataclasses.fields(cls)}
    for key in d.keys():
        if key not in field_names:
            raise ValueError(f"Unexpected key: '{key}'. Expected keys for {cls.__name__} are: {field_names}")


@dataclasses.dataclass
class ChannelRecord:
    channel_type: str | None
    channel_id: str | None
    channel_index: int | None

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> ChannelRecord:
        validate_keys(cls, d)
        channel_type = d.get("channel_type", None)
        channel_id = d.get("channel_id", None)
        channel_index = d.get("channel_index", None)
        return cls(channel_type, channel_id, channel_index)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        if self.channel_type is not None:
            d["channel_type"] = self.channel_type
        if self.channel_id is not None:
            d["channel_id"] = self.channel_id
        if self.channel_index is not None:
            d["channel_index"] = self.channel_index
        return d


@dataclasses.dataclass
class DeviceParametersRecord:
    parameters: typing.Mapping[str, typing.Any]

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> DeviceParametersRecord:
        parameters = dict(d)
        return cls(parameters)

    def to_dict(self) -> PersistentDictType:
        return dict(self.parameters)


@dataclasses.dataclass
class DetectorRecord:
    device_type_id: str
    device_parameters: DeviceParametersRecord
    channels: typing.Sequence[ChannelRecord]

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> DetectorRecord:
        validate_keys(cls, d)
        device_type_id = typing.cast(str, d["device_type_id"])
        device_parameters = DeviceParametersRecord.from_dict(d.get("device_parameters", dict()))
        channels = [ChannelRecord.from_dict(channel_d) for channel_d in d["channels"]]
        return cls(device_type_id, device_parameters, channels)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["device_type_id"] = self.device_type_id
        device_parameters_d = self.device_parameters.to_dict()
        if device_parameters_d:
            d["device_parameters"] = device_parameters_d
        d["channels"] = [channel.to_dict() for channel in self.channels]
        return d


@dataclasses.dataclass
class MagnificationRecord:
    fov_nm: float | None
    rotation: float | None

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> MagnificationRecord:
        validate_keys(cls, d)
        fov_nm = typing.cast(float | None, d.get("fov_nm", None))
        rotation = typing.cast(float | None, d.get("rotation", None))
        return cls(fov_nm, rotation)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        if self.fov_nm is not None:
            d["fov_nm"] = self.fov_nm
        if self.rotation is not None:
            d["rotation"] = self.rotation
        return d


@dataclasses.dataclass
class ClockRecord:
    type: str
    pixel_time_us: float | None
    device_type_id: str | None

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> ClockRecord:
        validate_keys(cls, d)
        type = typing.cast(str, d["type"])
        pixel_time_us = typing.cast(float | None, d.get("pixel_time_us", None))
        device_type_id = typing.cast(str | None, d.get("device_type_id", None))
        return cls(type, pixel_time_us, device_type_id)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["type"] = self.type
        if self.pixel_time_us is not None:
            d["pixel_time_us"] = self.pixel_time_us
        if self.device_type_id is not None:
            d["device_type_id"] = self.device_type_id
        return d


@dataclasses.dataclass
class ScanRecord:
    pixel_size: tuple[int, int]
    subscan_pixel_size: tuple[int, int] | None
    subscan_fractional_center: tuple[float, float] | None
    subscan_fractional_size: tuple[float, float] | None
    clock: ClockRecord

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> ScanRecord:
        validate_keys(cls, d)
        pixel_size_l = typing.cast(typing.Sequence[int], d["pixel_size"])
        pixel_size = (pixel_size_l[0], pixel_size_l[1])
        subscan_pixel_size_l = typing.cast(typing.Sequence[int] | None, d.get("subscan_pixel_size", None))
        subscan_pixel_size = (subscan_pixel_size_l[0], subscan_pixel_size_l[1]) if subscan_pixel_size_l else None
        subscan_fractional_center_l = typing.cast(typing.Sequence[float] | None, d.get("subscan_fractional_center", None))
        subscan_fractional_center = (subscan_fractional_center_l[0], subscan_fractional_center_l[1]) if subscan_fractional_center_l else None
        subscan_fractional_size_l = typing.cast(typing.Sequence[float] | None, d.get("subscan_fractional_size", None))
        subscan_fractional_size = (subscan_fractional_size_l[0], subscan_fractional_size_l[1]) if subscan_fractional_size_l else None
        clock = ClockRecord.from_dict(typing.cast(typing.Mapping[str, typing.Any], d["clock"]))
        return cls(pixel_size, subscan_pixel_size, subscan_fractional_center, subscan_fractional_size, clock)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["pixel_size"] = list(self.pixel_size)
        if self.subscan_pixel_size is not None:
            d["subscan_pixel_size"] = list(self.subscan_pixel_size)
        if self.subscan_fractional_center is not None:
            d["subscan_fractional_center"] = list(self.subscan_fractional_center)
        if self.subscan_fractional_size is not None:
            d["subscan_fractional_size"] = list(self.subscan_fractional_size)
        d["clock"] = self.clock.to_dict()
        return d


@dataclasses.dataclass
class ProcedureRecord:
    type: str
    magnification: MagnificationRecord | None
    scan: ScanRecord | None

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> ProcedureRecord:
        type = typing.cast(str, d["type"])
        if type == "view":
            return ViewProcedureRecord.from_dict(d)
        elif type == "acquire":
            return AcquireProcedureRecord.from_dict(d)
        raise ValueError(f"Unknown procedure type: {type}")

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["type"] = self.type
        if self.magnification is not None:
            d["magnification"] = self.magnification.to_dict()
        if self.scan is not None:
            d["scan"] = self.scan.to_dict()
        return d


@dataclasses.dataclass
class ViewProcedureRecord(ProcedureRecord):
    detector: DetectorRecord
    duration: float

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> ViewProcedureRecord:
        validate_keys(cls, d)
        magnification = MagnificationRecord.from_dict(d.get("magnification", dict())) if "magnification" in d else None
        scan = ScanRecord.from_dict(d.get("scan", dict())) if "scan" in d else None
        detector = DetectorRecord.from_dict(d["detector"])
        duration = typing.cast(float, d["duration"])
        return cls("view", magnification, scan, detector, duration)

    def to_dict(self) -> PersistentDictType:
        d = dict(super().to_dict())
        d["detector"] = self.detector.to_dict()
        d["duration"] = self.duration
        return d


@dataclasses.dataclass
class IterationRecord:
    type: str

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> IterationRecord:
        type = typing.cast(str, d["type"])
        if type == "basic":
            return BasicIterationRecord(type)
        elif type == "sequence":
            return SequenceIterationRecord.from_dict(d)
        elif type == "series":
            return SeriesIterationRecord.from_dict(d)
        elif type == "multiple-series":
            return MultipleSeriesIterationRecord.from_dict(d)
        raise ValueError(f"Unknown iteration type: {type}")

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["type"] = self.type
        return d


@dataclasses.dataclass
class BasicIterationRecord(IterationRecord):

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> BasicIterationRecord:
        validate_keys(cls, d)
        return cls("basic")

    def to_dict(self) -> PersistentDictType:
        return super().to_dict()


@dataclasses.dataclass
class SequenceIterationRecord(IterationRecord):
    count: int

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> SequenceIterationRecord:
        validate_keys(cls, d)
        count = typing.cast(int, d["count"])
        return cls("sequence", count)

    def to_dict(self) -> PersistentDictType:
        d = dict(super().to_dict())
        d["count"] = self.count
        return d


@dataclasses.dataclass
class ControlValuesRecord:
    count: int
    start: float | int
    step: float | int

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> ControlValuesRecord:
        validate_keys(cls, d)
        count = typing.cast(int, d["count"])
        start = typing.cast(float | int, d["start"])
        step = typing.cast(float | int, d["step"])
        return cls(count, start, step)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["count"] = self.count
        d["start"] = self.start
        d["step"] = self.step
        return d


@dataclasses.dataclass
class SeriesIterationRecord(IterationRecord):
    control_id: str
    control_values: ControlValuesRecord

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> SeriesIterationRecord:
        validate_keys(cls, d)
        control_id = typing.cast(str, d["control_id"])
        control_values = ControlValuesRecord.from_dict(d["control_values"])
        return cls("series", control_id, control_values)

    def to_dict(self) -> PersistentDictType:
        d = dict(super().to_dict())
        d["control_id"] = self.control_id
        d["control_values"] = self.control_values.to_dict()
        return d


@dataclasses.dataclass
class MultipleSeriesSectionRecord:
    offset: float
    exposure_ms: float
    count: int
    include_sum: bool = False

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> MultipleSeriesSectionRecord:
        validate_keys(cls, d)
        offset = typing.cast(float, d["offset"])
        exposure_ms = typing.cast(float, d["exposure_ms"])
        count = typing.cast(int, d["count"])
        include_sum = typing.cast(bool, d.get("include_sum", False))
        return cls(offset, exposure_ms, count, include_sum)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["offset"] = self.offset
        d["exposure_ms"] = self.exposure_ms
        d["count"] = self.count
        if self.include_sum:
            d["include_sum"] = self.include_sum
        return d


@dataclasses.dataclass
class MultipleSeriesIterationRecord(IterationRecord):
    sections: typing.Sequence[MultipleSeriesSectionRecord]

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> MultipleSeriesIterationRecord:
        validate_keys(cls, d)
        sections = [MultipleSeriesSectionRecord.from_dict(section_d) for section_d in d["sections"]]
        return cls("multiple-series", sections)

    def to_dict(self) -> PersistentDictType:
        d = dict(super().to_dict())
        d["sections"] = [section.to_dict() for section in self.sections]
        return d


@dataclasses.dataclass
class AcquireProcedureRecord(ProcedureRecord):
    detectors: typing.Sequence[DetectorRecord]
    iteration: IterationRecord

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> AcquireProcedureRecord:
        validate_keys(cls, d)
        magnification = MagnificationRecord.from_dict(d.get("magnification", dict())) if "magnification" in d else None
        scan = ScanRecord.from_dict(d.get("scan", dict())) if "scan" in d else None
        detectors = [DetectorRecord.from_dict(detector_d) for detector_d in typing.cast(typing.Sequence[typing.Mapping[str, typing.Any]], d["detectors"])]
        iteration = IterationRecord.from_dict(d["iteration"]) if "iteration" in d else BasicIterationRecord("basic")
        return cls("acquire", magnification, scan, detectors, iteration)

    def to_dict(self) -> PersistentDictType:
        d = dict(super().to_dict())
        d["detectors"] = [detector.to_dict() for detector in self.detectors]
        if not isinstance(self.iteration, BasicIterationRecord):
            d["iteration"] = self.iteration.to_dict()
        return d


@dataclasses.dataclass
class TestRecord:
    title: str
    procedure: ProcedureRecord

    @classmethod
    def from_dict(cls, d: PersistentDictType) -> TestRecord:
        validate_keys(cls, d)
        title = typing.cast(str, d["title"])
        procedure = ProcedureRecord.from_dict(d["procedure"])
        return cls(title, procedure)

    def to_dict(self) -> PersistentDictType:
        d = dict[str, typing.Any]()
        d["title"] = self.title
        d["procedure"] = self.procedure.to_dict()
        return d



@dataclasses.dataclass
class ExpectedShapeAndDescriptor:
    data_shape: DataAndMetadata.ShapeType
    data_descriptor: DataAndMetadata.DataDescriptor


ExpectedResults = typing.Mapping[Acquisition.Channel, ExpectedShapeAndDescriptor]


def check_results(ar: AcquisitionPanel._AcquireResult, logger: logging.Logger, expected_results: ExpectedResults) -> AcquisitionTestResult:
    result = AcquisitionTestResult()
    result.result = ar
    if not ar.success:
        result.messages.append(AcquisitionTestResultMessage(False, "Acquisition failed"))
    for channel, expected_result in expected_results.items():
        data_item = ar.data_item_map.get(channel)
        if data_item:
            data_shape = data_item.data_shape
            data_and_metadata = data_item.data_and_metadata
            if data_and_metadata:
                data_descriptor = data_and_metadata.data_descriptor
                result.messages.append(AcquisitionTestResultMessage(data_shape == expected_result.data_shape, f"Channel {channel} data shape {data_shape} matches expected {expected_result.data_shape}"))
                result.messages.append(AcquisitionTestResultMessage(data_descriptor == expected_result.data_descriptor, f"Channel {channel} data descriptor {data_descriptor} matches expected {expected_result.data_descriptor}"))
            else:
                result.messages.append(AcquisitionTestResultMessage(False, f"Channel {channel} data and metadata not found"))
        else:
            result.messages.append(AcquisitionTestResultMessage(False, f"Channel {channel} not found in acquisition result"))
    for channel in ar.data_item_map.keys():
        if channel not in expected_results:
            result.messages.append(AcquisitionTestResultMessage(False, f"Channel {channel} not expected in acquisition result"))
    return result


def acquisition_test_state_to_string(state: AcquisitionTestState | None) -> str:
    if state == AcquisitionTestState.NOT_STARTED:
        return "\N{WHITE CIRCLE}"
    elif state == AcquisitionTestState.IN_PROGRESS:
        return "\N{CIRCLE WITH LEFT HALF BLACK}"
    elif state == AcquisitionTestState.SUCCEEDED:
        return "\N{WHITE HEAVY CHECK MARK}"
    elif state == AcquisitionTestState.FAILED:
        return "\N{CROSS MARK}"
    else:
        return "\N{LARGE BLUE CIRCLE}"


class AcquisitionProcedureTest(Observable.Observable):
    def __init__(self, test: TestRecord) -> None:
        super().__init__()
        self.title = test.title
        self.__test = test
        self.__state = AcquisitionTestState.NOT_STARTED
        self.status_stream = Model.StreamValueModel(Stream.MapStream[AcquisitionTestState, str](Stream.PropertyChangedEventStream(self, "state"), acquisition_test_state_to_string))
        self.__elapsed_time: float | None = None
        self.tooltip = Model.PropertyModel[str]()
        self.__result: AcquisitionTestResult | None = None

    @property
    def state(self) -> AcquisitionTestState:
        return self.__state

    @state.setter
    def state(self, value: AcquisitionTestState) -> None:
        self.__state = value
        self.notify_property_changed("state")

    @property
    def elapsed_time(self) -> float | None:
        return self.__elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, value: float | None) -> None:
        self.__elapsed_time = value
        self.notify_property_changed("elapsed_time")

    @property
    def has_result(self) -> bool:
        return self.__result is not None

    @property
    def result(self) -> AcquisitionTestResult | None:
        return self.__result

    @result.setter
    def result(self, value: AcquisitionTestResult | None) -> None:
        self.__result = value
        self.notify_property_changed("result")
        self.notify_property_changed("has_result")

    async def run_test_async(self, window: DocumentController.DocumentController, is_stopped_model: Model.ValueModel[bool], delete_results: bool) -> None:
        was_stopped = is_stopped_model.value
        is_stopped_model.value = False
        try:
            self.state = AcquisitionTestState.IN_PROGRESS
            stem_controller = typing.cast(STEMController.STEMController | None, Registry.get_component("stem_controller"))
            assert stem_controller

            ap = typing.cast(AcquisitionPanel.AcquisitionPanel | None, window.find_dock_panel("acquisition-panel"))
            assert ap
            ap.show()

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            logger.info(f"Starting acquisition test: {self.title}")

            result: AcquisitionTestResult | None = None
            exception: Exception | None = None

            try:
                start_time = time.time()
                result = await self.acquire(ap, stem_controller, logger)
                self.elapsed_time = time.time() - start_time
            except Exception as e:
                import traceback
                traceback.print_exc()
                exception = e

            logger.info(f"Finishing acquisition test: {self.title}")

            result_messages = result.messages if result else list()

            if not result:
                result_messages.append(AcquisitionTestResultMessage(False, f"Acquisition test failed: {exception}"))
                self.state = AcquisitionTestState.FAILED

            self.tooltip.value = "\n".join(f"{acquisition_test_state_to_string(r.status)} {r.message}" for r in result_messages)

            if all(r.status == AcquisitionTestState.SUCCEEDED for r in result_messages):
                self.state = AcquisitionTestState.SUCCEEDED
            elif any(r.status == AcquisitionTestState.NOT_AVAILABLE for r in result_messages):
                self.state = AcquisitionTestState.NOT_AVAILABLE
            else:
                self.state = AcquisitionTestState.FAILED

            self.result = result

            if result and delete_results:
                result.delete_display_items(window)
        finally:
            is_stopped_model.value = was_stopped

    async def acquire(self, ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
        procedure = create_acquisition_procedure(self.__test.procedure, ap, stem_controller, logger)
        if preflight_result := procedure.preflight():
            return preflight_result
        procedure_results = await procedure.run_procedure()
        expected_results = procedure.get_expected_results()
        return check_results(procedure_results, logger, expected_results)


class AcquisitionProcedure:
    def __init__(self, procedure: ProcedureRecord, ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> None:
        self.procedure = procedure
        self.ap = ap
        self.stem_controller = stem_controller
        self.logger = logger

    def preflight(self) -> AcquisitionTestResult | None:
        return None

    async def run_procedure(self) -> AcquisitionPanel._AcquireResult:
        raise NotImplementedError()

    def get_expected_results(self) -> ExpectedResults:
        raise NotImplementedError()


def get_device(detector_record: DetectorRecord, stem_controller: STEMController.STEMController) -> HardwareSource.HardwareSource | None:
    device_type_id = detector_record.device_type_id
    match device_type_id:
        case "ronchigram_camera":
            return stem_controller.ronchigram_camera
        case "eels_camera":
            return stem_controller.eels_camera
        case "scan_controller":
            return stem_controller.scan_controller
        case _:
            return None


@dataclasses.dataclass
class Detector:
    device_type_id: str
    device: HardwareSource.HardwareSource
    frame_parameters: HardwareSource.FrameParameters
    channels: typing.Sequence[Acquisition.Channel]
    expected_results: ExpectedResults
    hardware_source_channel_description_map: dict[Acquisition.Channel, str]

    def prepare(self) -> None:
        self.device.set_current_frame_parameters(self.frame_parameters)
        if isinstance(self.device, ScanBase.ScanHardwareSource) and isinstance(self.frame_parameters, ScanBase.ScanFrameParameters):
            if self.frame_parameters.subscan_pixel_size:
                self.device.subscan_enabled = True
            else:
                self.device.subscan_enabled = False
        enabled_channel_indexes = [self.device.get_channel_index(channel.segments[1]) for channel in self.channels if len(channel.segments) > 1]
        if enabled_channel_indexes:
            for channel_index in range(self.device.get_channel_count()):
                self.device.set_channel_enabled(channel_index, channel_index in enabled_channel_indexes)

    @property
    def is_camera(self) -> bool:
        return isinstance(self.device, CameraBase.CameraHardwareSource)

    @property
    def view_channels(self) -> typing.Iterable[Acquisition.Channel]:
        return self.expected_results.keys()

    def get_expected_results(self, channel: Acquisition.Channel) -> ExpectedShapeAndDescriptor:
        return self.expected_results[channel]


def parse_detector_channels(channel_records: typing.Sequence[ChannelRecord], device: HardwareSource.HardwareSource) -> typing.Sequence[Acquisition.Channel]:
    channels = list[Acquisition.Channel]()
    for channel_record in channel_records:
        channel = Acquisition.Channel(device.hardware_source_id)
        channel_id = channel_record.channel_id
        if channel_id is None and channel_record.channel_index is not None:
            channel_id = device.get_channel_id(channel_record.channel_index)
        if channel_id is None and channel_record.channel_type == "primary":
            pass
        if channel_id:
            channel = channel.join_segment(channel_id)
        channels.append(channel)
    return tuple(channels)


def get_detector(detector_record: DetectorRecord, stem_controller: STEMController.STEMController, magnification: MagnificationRecord | None, scan: ScanRecord | None) -> Detector:
    device_type_id = detector_record.device_type_id
    device = get_device(detector_record, stem_controller)
    if not device:
        raise ValueError(f"Device of type {device_type_id} not found in STEM controller.")
    frame_parameters: HardwareSource.FrameParameters
    channels = parse_detector_channels(detector_record.channels, device)
    expected_results = dict[Acquisition.Channel, ExpectedShapeAndDescriptor]()
    hardware_source_channel_description_map = dict[Acquisition.Channel, str]()
    match device_type_id:
        case "ronchigram_camera":
            assert isinstance(device, CameraBase.CameraHardwareSource)
            camera_frame_parameters = device.get_frame_parameters_from_dict(detector_record.device_parameters.parameters)
            channel = Acquisition.Channel(device.hardware_source_id)
            data_shape = device.get_expected_dimensions(camera_frame_parameters)
            expected_results[channel] = ExpectedShapeAndDescriptor(data_shape, DataAndMetadata.DataDescriptor(False, 0, 2))
            hardware_source_channel_description_map[channel] = "ronchigram"
            frame_parameters = camera_frame_parameters
        case "eels_camera":
            assert isinstance(device, CameraBase.CameraHardwareSource)
            camera_frame_parameters = device.get_frame_parameters_from_dict(detector_record.device_parameters.parameters)
            channel = Acquisition.Channel(device.hardware_source_id)
            channel_summed = Acquisition.Channel(device.hardware_source_id, "summed")
            data_shape = device.get_expected_dimensions(camera_frame_parameters)
            expected_results[channel] = ExpectedShapeAndDescriptor(data_shape, DataAndMetadata.DataDescriptor(False, 0, 2))
            expected_results[channel_summed] = ExpectedShapeAndDescriptor(data_shape[1:], DataAndMetadata.DataDescriptor(False, 0, 1))
            hardware_source_channel_description_map[channel] = "eels_image"
            hardware_source_channel_description_map[channel_summed] = "eels_spectrum"
            frame_parameters = camera_frame_parameters
        case "scan_controller":
            assert isinstance(device, ScanBase.ScanHardwareSource)
            # hack to put together the scan frame parameters
            frame_parameters_d = dict[str, typing.Any]()
            if magnification and magnification.fov_nm:
                frame_parameters_d["fov_nm"] = magnification.fov_nm
            if magnification and magnification.rotation:
                frame_parameters_d["rotation"] = magnification.rotation
            if scan and scan.pixel_size:
                frame_parameters_d["pixel_size"] = scan.pixel_size
            if scan and scan.subscan_pixel_size:
                frame_parameters_d["subscan_pixel_size"] = scan.subscan_pixel_size
            if scan and scan.subscan_fractional_center:
                frame_parameters_d["subscan_fractional_center"] = scan.subscan_fractional_center
            if scan and scan.subscan_fractional_size:
                frame_parameters_d["subscan_fractional_size"] = scan.subscan_fractional_size
            if scan and scan.clock and scan.clock.pixel_time_us:
                frame_parameters_d["pixel_time_us"] = scan.clock.pixel_time_us
            scan_frame_parameters = typing.cast(ScanBase.ScanFrameParameters, device.get_frame_parameters_from_dict(frame_parameters_d))
            enabled_channel_indexes = [device.get_channel_index(channel.segments[1]) for channel in channels]
            for channel_index in enabled_channel_indexes:
                channel = Acquisition.Channel(device.hardware_source_id, str(channel_index))
                expected_results[channel] = ExpectedShapeAndDescriptor(scan_frame_parameters.scan_size.as_tuple(), DataAndMetadata.DataDescriptor(False, 0, 2))
                hardware_source_channel_description_map[channel] = ".".join(channel.segments)
            frame_parameters = scan_frame_parameters
        case _:
            raise ValueError(f"Unknown device type: {device_type_id}")
    assert frame_parameters is not None
    return Detector(device_type_id, device, frame_parameters, channels, expected_results, hardware_source_channel_description_map)


class ViewAcquisitionProcedure(AcquisitionProcedure):
    def __init__(self, procedure: ViewProcedureRecord, ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> None:
        super().__init__(procedure, ap, stem_controller, logger)
        magnification = procedure.magnification
        scan = procedure.scan
        self.detector = get_detector(procedure.detector, stem_controller, magnification, scan)
        self.duration = procedure.duration

    def preflight(self) -> AcquisitionTestResult | None:
        if not self.detector.device:
            return make_not_available_result(f"Device type {self.detector.device_type_id} not found.")
        return None

    async def run_procedure(self) -> AcquisitionPanel._AcquireResult:
        self.detector.prepare()
        self.detector.device.start_playing(sync_timeout=3.0)
        start = time.time()
        while time.time() - start < 5.0:
            await asyncio.sleep(0.1)
        self.detector.device.stop_playing(sync_timeout=3.0)
        model = self.ap.document_controller.document_model
        result_map = dict[Acquisition.Channel, DataItem.DataItem]()
        for channel in self.detector.view_channels:
            hardware_source_id = channel.segments[0]
            # some real messiness resulting from inconsistent use of channel_id and channel_index in get_data_item_channel_reference
            if isinstance(self.detector.device, ScanBase.ScanHardwareSource):
                channel_index = int(channel.segments[1]) if len(channel.segments) > 1 else None
                channel_id = self.detector.device.get_channel_id(channel_index) if channel_index is not None else None
            else:
                channel_id = channel.segments[1] if len(channel.segments) > 1 else None
            data_item = model.get_data_item_channel_reference(hardware_source_id, channel_id).data_item
            if data_item:
                result_map[channel] = data_item
        return AcquisitionPanel._AcquireResult(True, result_map)

    def get_expected_results(self) -> ExpectedResults:
        return self.detector.expected_results


class Iteration:
    def __init__(self) -> None:
        pass

    def activate(self, ap: AcquisitionPanel.AcquisitionPanel) -> None:
        raise NotImplementedError()

    def update_expected_results(self, expected_results: ExpectedResults) -> ExpectedResults:
        raise NotImplementedError()


class BasicIteration(Iteration):
    def activate(self, ap: AcquisitionPanel.AcquisitionPanel) -> None:
        ap._activate_basic_acquire()

    def update_expected_results(self, expected_results: ExpectedResults) -> ExpectedResults:
        return expected_results


class SequenceIteration(Iteration):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    def activate(self, ap: AcquisitionPanel.AcquisitionPanel) -> None:
        ap._activate_sequence_acquire(self.count)

    def update_expected_results(self, expected_results: ExpectedResults) -> ExpectedResults:
        results = dict[Acquisition.Channel, ExpectedShapeAndDescriptor]()
        for channel, expected_shape_and_descriptor in expected_results.items():
            data_shape = (self.count,) + expected_shape_and_descriptor.data_shape
            assert not expected_shape_and_descriptor.data_descriptor.is_sequence
            data_descriptor = DataAndMetadata.DataDescriptor(True, expected_shape_and_descriptor.data_descriptor.collection_dimension_count, expected_shape_and_descriptor.data_descriptor.datum_dimension_count)
            results[channel] = ExpectedShapeAndDescriptor(data_shape, data_descriptor)
        return results


class SeriesIteration(Iteration):
    def __init__(self, control_id: str, control_values: AcquisitionPanel.ControlValues) -> None:
        super().__init__()
        self.control_id = control_id
        self.control_values = control_values

    def activate(self, ap: AcquisitionPanel.AcquisitionPanel) -> None:
        ap._activate_series_acquire(self.control_id, self.control_values)

    def update_expected_results(self, expected_results: ExpectedResults) -> ExpectedResults:
        results = dict[Acquisition.Channel, ExpectedShapeAndDescriptor]()
        for channel, expected_shape_and_descriptor in expected_results.items():
            data_shape = (self.control_values.count,) + expected_shape_and_descriptor.data_shape
            assert not expected_shape_and_descriptor.data_descriptor.is_sequence
            data_descriptor = DataAndMetadata.DataDescriptor(True, expected_shape_and_descriptor.data_descriptor.collection_dimension_count, expected_shape_and_descriptor.data_descriptor.datum_dimension_count)
            results[channel] = ExpectedShapeAndDescriptor(data_shape, data_descriptor)
        return results


class MultipleSeriesIteration(Iteration):
    def __init__(self, sections: typing.Sequence[AcquisitionPanel.MultipleAcquireEntry]) -> None:
        super().__init__()
        self.sections = sections

    def activate(self, ap: AcquisitionPanel.AcquisitionPanel) -> None:
        ap._activate_multiple_acquire(self.sections)

    def update_expected_results(self, expected_results: ExpectedResults) -> ExpectedResults:
        results = dict[Acquisition.Channel, ExpectedShapeAndDescriptor]()
        for index, section in enumerate(self.sections):
            for channel, expected_shape_and_descriptor in expected_results.items():
                data_shape = (section.count,) + expected_shape_and_descriptor.data_shape
                assert expected_shape_and_descriptor.data_descriptor.is_sequence == False
                data_descriptor = DataAndMetadata.DataDescriptor(True, expected_shape_and_descriptor.data_descriptor.collection_dimension_count, expected_shape_and_descriptor.data_descriptor.datum_dimension_count)
                results[Acquisition.Channel(str(index), *channel.segments)] = ExpectedShapeAndDescriptor(data_shape, data_descriptor)
                if section.include_sum:
                    results[Acquisition.Channel(str(index), *channel.segments, "sum")] = expected_shape_and_descriptor
        return results


def create_iteration(iteration_record: IterationRecord) -> Iteration:
    match iteration_record:
        case BasicIterationRecord():
            return BasicIteration()
        case SequenceIterationRecord():
            return SequenceIteration(iteration_record.count)
        case SeriesIterationRecord():
            control_values_record = iteration_record.control_values
            control_values = AcquisitionPanel.ControlValues(control_values_record.count, control_values_record.start, control_values_record.step)
            return SeriesIteration(iteration_record.control_id, control_values)
        case MultipleSeriesIterationRecord():
            section_records = iteration_record.sections
            sections = list[AcquisitionPanel.MultipleAcquireEntry]()
            for section_record in section_records:
                offset = section_record.offset
                exposure_ms = section_record.exposure_ms
                count = section_record.count
                include_sum = section_record.include_sum
                sections.append(AcquisitionPanel.MultipleAcquireEntry(offset, exposure_ms / 1e3, count, include_sum))
            return MultipleSeriesIteration(sections)
        case _:
            raise ValueError(f"Unknown iteration type: {iteration_record.type}")


class AcquireAcquisitionProcedure(AcquisitionProcedure):
    def __init__(self, procedure: AcquireProcedureRecord, ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> None:
        super().__init__(procedure, ap, stem_controller, logger)
        magnification = procedure.magnification
        scan = procedure.scan
        self.detectors = [get_detector(detector, stem_controller, magnification, scan) for detector in procedure.detectors]
        self.iteration = create_iteration(procedure.iteration)

    def preflight(self) -> AcquisitionTestResult | None:
        camera_count = 0  # count cameras, only a single camera allowed in detectors
        for detector in self.detectors:
            if not detector.device:
                return make_not_available_result(f"Device type {detector.device_type_id} not found.")
            if detector.is_camera:
                camera_count += 1
        if camera_count > 1:
            return make_not_available_result(f"Only a single camera-like detector is allowed.")
        return None

    async def run_procedure(self) -> AcquisitionPanel._AcquireResult:
        self.iteration.activate(self.ap)
        if len(self.detectors) == 1:
            detector = self.detectors[0]
            detector.prepare()
            if isinstance(detector.device, CameraBase.CameraHardwareSource):
                # hack until we allow multiple camera channels
                selected_channel = detector.channels[0]
                hardware_source_channel_description_id = detector.hardware_source_channel_description_map[selected_channel]
                exposure = typing.cast(CameraBase.CameraFrameParameters, detector.frame_parameters).exposure
                self.ap._activate_camera_acquire(detector.device.hardware_source_id, hardware_source_channel_description_id, exposure)
            else:
                self.ap._activate_scan_acquire(detector.device.hardware_source_id)
            acquire_result = await self.ap._acquire()
        elif len(self.detectors) == 2:
            for detector in self.detectors:
                detector.prepare()
            # assumes scan detector is first, camera is second
            scan_detector = self.detectors[0]
            camera_detector = self.detectors[1]
            selected_channel = camera_detector.channels[0]
            hardware_source_channel_description_id = camera_detector.hardware_source_channel_description_map[selected_channel]
            exposure = typing.cast(CameraBase.CameraFrameParameters, camera_detector.frame_parameters).exposure
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, scan_detector.device)
            scan_size = scan_hardware_source.get_current_frame_parameters().scan_size
            self.ap._activate_synchronized_acquire(scan_detector.device.hardware_source_id, scan_size.width, camera_detector.device.hardware_source_id, hardware_source_channel_description_id, exposure)
            acquire_result = await self.ap._acquire()
        else:
            raise NotImplementedError()
        # special hack until channel results are cleaned up. this is necessary because the acquisition panel current
        # uses a result key of the eels_camera_id instead of the eels_camera_id + processing.
        for detector in self.detectors:
            if isinstance(detector.device, CameraBase.CameraHardwareSource):
                # hack until we allow multiple camera channels
                selected_channel = detector.channels[0]
                hardware_source_channel_description_id = detector.hardware_source_channel_description_map[selected_channel]
                if hardware_source_channel_description_id == "eels_spectrum":
                    for channel in list(acquire_result.data_item_map.keys()):
                        new_channel_segments = list[str]()
                        for segment in channel.segments:
                            if segment != detector.device.hardware_source_id:
                                new_channel_segments.append(segment)
                            else:
                                new_channel_segments.extend(selected_channel.segments)
                        new_channel = Acquisition.Channel(*new_channel_segments)
                        acquire_result.data_item_map[new_channel] = acquire_result.data_item_map.pop(channel)
        return acquire_result

    def get_expected_results(self) -> ExpectedResults:
        expected_results = dict(self.detectors[0].expected_results)
        if len(self.detectors) == 2:
            # assumes scan detector is first, camera is second
            scan_detector = self.detectors[0]
            camera_detector = self.detectors[1]
            scan_size = typing.cast(ScanBase.ScanFrameParameters, scan_detector.frame_parameters).scan_size
            for camera_channel, camera_expected_result in camera_detector.expected_results.items():
                data_descriptor = DataAndMetadata.DataDescriptor(False, len(scan_size), camera_expected_result.data_descriptor.datum_dimension_count)
                expected_results[camera_channel] = ExpectedShapeAndDescriptor(scan_size.as_tuple() + camera_expected_result.data_shape, data_descriptor)
        # special hack since acquisition only acquires one of the two channels for camera detectors currently.
        for detector in self.detectors:
            if isinstance(detector.device, CameraBase.CameraHardwareSource):
                # hack until we allow multiple camera channels
                selected_channel = detector.channels[0]
                hardware_source_channel_description_id = detector.hardware_source_channel_description_map[selected_channel]
                if hardware_source_channel_description_id == "eels_spectrum":
                    expected_results.pop(Acquisition.Channel(detector.device.hardware_source_id))
                elif hardware_source_channel_description_id == "eels_image":
                    expected_results.pop(Acquisition.Channel(detector.device.hardware_source_id, "summed"))
        return self.iteration.update_expected_results(expected_results)


def create_acquisition_procedure(procedure_record: ProcedureRecord, ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionProcedure:
    match procedure_record.type:
        case "view":
            return ViewAcquisitionProcedure(typing.cast(ViewProcedureRecord, procedure_record), ap, stem_controller, logger)
        case "acquire":
            return AcquireAcquisitionProcedure(typing.cast(AcquireProcedureRecord, procedure_record), ap, stem_controller, logger)
        case _:
            raise ValueError(f"Unknown acquisition procedure type: {procedure_record.type}")


def parse_expected_result_channel(channel_record: ChannelRecord, device: HardwareSource.HardwareSource) -> Acquisition.Channel:
    channel = Acquisition.Channel(device.hardware_source_id)
    channel_id = channel_record.channel_id
    if channel_id is None and channel_record.channel_index is not None:
        channel_id = str(channel_record.channel_index)
    if channel_id:
        channel = channel.join_segment(channel_id)
    return channel


class AcquisitionTestHandler(Declarative.Handler):
    def __init__(self, acquisition_test: AcquisitionProcedureTest, window: DocumentController.DocumentController, is_stopped_model: Model.ValueModel[bool]) -> None:
        super().__init__()
        self.acquisition_test = acquisition_test
        self.window = window
        self.is_stopped_model = is_stopped_model
        self.elapsed_time_converter = Converter.PhysicalValueToStringConverter("s", 1, "{:.1f}")
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_row(
            # u.create_check_box(enabled="@binding(is_stopped_model.value)", style="minimal", width=24),
            u.create_label(text="@binding(acquisition_test.status_stream.value)", width=18),
            u.create_label(text=acquisition_test.title, tool_tip="@binding(acquisition_test.tooltip.value)"),
            u.create_stretch(),
            u.create_label(text="@binding(acquisition_test.elapsed_time, converter=elapsed_time_converter)", width=48),
            {"type": "nionswift.text_push_button",
             "text": "\u23F5",
             "enabled": "@binding(is_stopped_model.value)",
             "tool_tip": "Run test.",
             "on_clicked": "handle_run_clicked"},
            {"type": "nionswift.text_push_button",
             "text": "\N{N-ARY CIRCLED TIMES OPERATOR}",
             "enabled": "@binding(acquisition_test.has_result)",
             "tool_tip": "Delete results.",
             "on_clicked": "handle_delete_clicked"},
            # u.create_push_button(text="Run", enabled="@binding(is_stopped_model.value)", on_clicked="handle_run_clicked", style="minimal"),
            # u.create_push_button(text="Delete", enabled="@binding(acquisition_test.has_result)", on_clicked="handle_delete_clicked", style="minimal"),
            spacing=8,
            margin_horizontal=8
        )

    def handle_run_clicked(self, widget: Declarative.UIWidget) -> None:
        self.window.event_loop.create_task(self.acquisition_test.run_test_async(self.window, self.is_stopped_model, False))

    def handle_delete_clicked(self, widget: Declarative.UIWidget) -> None:
        result = self.acquisition_test.result
        if result:
            result.delete_display_items(self.window)
            self.acquisition_test.result = None
            self.acquisition_test.elapsed_time = None


class AcquisitionTestDialog(Declarative.Handler):
    def __init__(self, window: DocumentController.DocumentController) -> None:
        super().__init__()
        self.window = window
        self.acquisition_tests = list[AcquisitionProcedureTest]()
        self.is_stopped_model = Model.PropertyModel[bool](True)

        self.is_run_all_model = Model.PropertyModel[bool](False)
        self.__cancel = False

        def is_enabled(b: typing.Optional[bool]) -> str:
            return "Run All" if not b else "Cancel"

        self.run_all_button_text_model = Model.StreamValueModel(Stream.MapStream(
            Stream.PropertyChangedEventStream(self.is_run_all_model, "value"),
            is_enabled))

        def is_run_all_enabled(b: typing.Optional[bool]) -> str:
            return "Run All" if not b else "Cancel"

        self.run_all_enabled_model: Model.ValueModel[bool] = Model.StreamValueModel(Stream.CombineLatestStream([Stream.PropertyChangedEventStream(self.is_stopped_model, "value"), Stream.PropertyChangedEventStream(self.is_run_all_model, "value")], lambda a, b: a or b))

        json_bytes = pkgutil.get_data(AcquisitionPanel.__name__, "resources/acquisition_tests.json")
        if json_bytes:
            for test_d in json.loads(json_bytes.decode("utf-8")):
                test_record = TestRecord.from_dict(test_d)
                self.acquisition_tests.append(AcquisitionProcedureTest(test_record))

        u = Declarative.DeclarativeUI()
        ui_view = u.create_column(
            u.create_scroll_area(u.create_column(items="acquisition_tests", item_component_id="acquisition_test", margin_vertical=4), width=480, height=400),
            u.create_row(
                u.create_stretch(),
                u.create_push_button(text="Reset", enabled="@binding(is_stopped_model.value)", on_clicked="handle_reset_clicked", style="minimal"),
                u.create_push_button(text="@binding(run_all_button_text_model.value)", enabled="@binding(run_all_enabled_model.value)", on_clicked="handle_run_all_clicked", style="minimal"), spacing=8, margin=8),
        )
        dialog = typing.cast(Dialog.ActionDialog, Declarative.construct(window.ui, window, u.create_modeless_dialog(ui_view, title="Acquisition Test"), self))
        dialog.show()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        if component_id == "acquisition_test":
            return AcquisitionTestHandler(typing.cast(AcquisitionProcedureTest, item), self.window, self.is_stopped_model)
        return None

    def handle_reset_clicked(self, widget: Declarative.UIWidget) -> None:
        for acquisition_test in self.acquisition_tests:
            acquisition_test.state = AcquisitionTestState.NOT_STARTED
            acquisition_test.elapsed_time = None
            acquisition_test.result = None

    def handle_run_all_clicked(self, widget: Declarative.UIWidget) -> None:
        if self.is_run_all_model.value:
            self.__cancel = True
            return

        for acquisition_test in self.acquisition_tests:
            acquisition_test.state = AcquisitionTestState.NOT_STARTED

        async def run_all() -> None:
            self.__cancel = False
            self.is_run_all_model.value = True
            self.is_stopped_model.value = False
            try:
                for acquisition_test in self.acquisition_tests:
                    try:
                        await acquisition_test.run_test_async(self.window, self.is_stopped_model, True)
                        if self.__cancel:
                            break
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
            finally:
                self.is_stopped_model.value = True
                self.is_run_all_model.value = False

        self.window.event_loop.create_task(run_all())



def _internal_main(window: DocumentController.DocumentController) -> None:
    # this will be called from run script on a thread. to run on the ui thread, schedule an async task.

    async def main() -> None:
        AcquisitionTestDialog(window)

    window.event_loop.create_task(main())
