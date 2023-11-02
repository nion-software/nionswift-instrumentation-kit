"""
Acquisition configurator and controller.

Changes to this API need to be strictly backwards compatible. Breaking changes should a new acquisition
controller with a name like AcquisitionConfiguration<n>, where <n> is the version number of the API.

The goal is to be able to define these acquisition descriptions and examine them in a GUI now and in the future.
The acquisitions could be edited and restarted (making a copy). The resulting acquisition objects should contain
pointers to the acquired data items and associated metadata.

As much as possible, the acquisition object should be able to fully specify the acquisition, meaning that there
may be default values, but the caller should be able to specify them using this API. This includes items such as
which channels are enabled, gain and dark images, etc. If these items are left unspecified, a suitable default or
global value may be used.
"""

from __future__ import annotations

import dataclasses
import typing

import numpy.typing

from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition as Acquisition_
from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller as STEMController
from nion.swift.model import Schema
from nion.utils import Geometry
from nion.utils import Registry


@dataclasses.dataclass
class Device(Acquisition_.AcquisitionProcedureFactoryInterface.Device):
    device_type_id: str
    device_id: typing.Optional[str] = None


@dataclasses.dataclass
class DeviceParametersLike(Acquisition_.AcquisitionProcedureFactoryInterface.DeviceParametersLike, typing.Protocol):
    pass


@dataclasses.dataclass
class DeviceChannelSpecifier(Acquisition_.AcquisitionProcedureFactoryInterface.DeviceChannelSpecifier):
    channel_index: typing.Optional[int]
    channel_type_id: typing.Optional[str]
    channel_id: typing.Optional[str]


@dataclasses.dataclass
class ProcessingChannelLike(Acquisition_.AcquisitionProcedureFactoryInterface.ProcessingChannelLike, typing.Protocol):
    pass


@dataclasses.dataclass
class MagnificationParameters(DeviceParametersLike):
    fov_nm: typing.Optional[float] = None
    rotation_rad: typing.Optional[float] = None


@dataclasses.dataclass
class ScanParameters(DeviceParametersLike, Acquisition_.AcquisitionProcedureFactoryInterface.ScanParameters):
    pixel_time_us: typing.Optional[float] = None
    pixel_size: typing.Optional[Geometry.IntSize] = None
    fov_nm: typing.Optional[float] = None
    rotation_rad: typing.Optional[float] = None
    center_nm: typing.Optional[Geometry.FloatPoint] = None
    subscan_pixel_size: typing.Optional[Geometry.IntSize] = None
    subscan_fractional_size: typing.Optional[Geometry.FloatSize] = None
    subscan_fractional_center: typing.Optional[Geometry.FloatPoint] = None
    subscan_rotation: typing.Optional[float] = None
    ac_line_sync: typing.Optional[bool] = None
    # flyback_time_us: typing.Optional[float] = None
    # ac_frame_sync: typing.Optional[bool] = None
    kwargs: typing.Optional[typing.Mapping[str, typing.Any]] = None

    def get_value(self, name: str) -> typing.Optional[typing.Any]:
        return self.kwargs.get(name) if self.kwargs else None

@dataclasses.dataclass
class CameraParameters(DeviceParametersLike, Acquisition_.AcquisitionProcedureFactoryInterface.CameraParameters):
    exposure_ms: typing.Optional[float] = None
    binning: typing.Optional[int] = None
    kwargs: typing.Optional[typing.Mapping[str, typing.Any]] = None


@dataclasses.dataclass
class DeviceAcquisitionParameters(Acquisition_.AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters):
    device_type_id: str
    device_parameters: typing.Optional[DeviceParametersLike] = None
    device_channels: typing.Optional[typing.Sequence[DeviceChannelSpecifier]] = None
    processing_channels: typing.Optional[typing.Sequence[ProcessingChannelLike]] = None


@dataclasses.dataclass
class DriftCorrectionParameters:
    drift_correction_enabled: bool
    drift_interval_lines: int
    drift_interval_scans: int
    drift_channel: typing.Optional[DeviceChannelSpecifier]
    drift_region: typing.Optional[Geometry.FloatRect]
    drift_rotation: float


@dataclasses.dataclass
class ProcedureStepLike(Acquisition_.AcquisitionProcedureFactoryInterface.ProcedureStepLike, typing.Protocol):
    step_id: str


@dataclasses.dataclass
class DeviceAcquisitionStep(ProcedureStepLike, Acquisition_.AcquisitionProcedureFactoryInterface.DeviceAcquisitionStep):
    step_id: str
    device_acquisition_parameters: DeviceAcquisitionParameters

    def __init__(self, device_acquisition_parameters: DeviceAcquisitionParameters) -> None:
        self.step_id = "device-acquisition"
        self.device_acquisition_parameters = device_acquisition_parameters


@dataclasses.dataclass
class MultiDeviceAcquisitionStep(ProcedureStepLike, Acquisition_.AcquisitionProcedureFactoryInterface.MultiDeviceAcquisitionStep):
    step_id: str
    primary_device_acquisition_parameters: DeviceAcquisitionParameters
    secondary_device_acquisition_parameters: typing.Sequence[DeviceAcquisitionParameters]
    drift_parameters: typing.Optional[DriftCorrectionParameters]

    def __init__(self, primary_device_acquisition_parameters: DeviceAcquisitionParameters, secondary_device_acquisition_parameters: typing.Sequence[DeviceAcquisitionParameters], drift_parameters: typing.Optional[DriftCorrectionParameters]) -> None:
        self.step_id = "multi-device-acquisition"
        self.primary_device_acquisition_parameters = primary_device_acquisition_parameters
        self.secondary_device_acquisition_parameters = secondary_device_acquisition_parameters
        self.drift_parameters = drift_parameters


@dataclasses.dataclass
class ControlController(Acquisition_.AcquisitionProcedureFactoryInterface.ControlController):
    controller_id: str


@dataclasses.dataclass
class DeviceControlController(ControlController, Acquisition_.AcquisitionProcedureFactoryInterface.DeviceControlController):
    """Configurable device control controller.

    - the controller_id is "device-controller"
    - the device_type_id is the device type id, such as "scan", "eels", "stem", etc.
    - the control_id is the general control id. there are built-ins like "defocus" but also used to specify the units of
      the device control "control2_m", etc.
    - the device_control_id is the specific control in AS2
    - the values is an object that can be converted to a numpy array.
    - the delay is the delay between values in seconds.
    - the axis_id is the axis id (if any) used for 2D controls.
    """
    controller_id: str
    device_type_id: str
    control_id: str
    device_control_id: typing.Optional[str]
    values: typing.Optional[numpy.typing.ArrayLike] = None
    delay: typing.Optional[float] = None
    axis_id: typing.Optional[str] = None

    def __init__(self, device_type_id: str, control_id: str, device_control_id: typing.Optional[str], values: typing.Optional[numpy.typing.ArrayLike], delay: typing.Optional[float] = None, axis_id: typing.Optional[str] = None) -> None:
        self.controller_id = "device-controller"
        self.device_type_id = device_type_id
        self.control_id = control_id
        self.device_control_id = device_control_id
        self.values = values
        self.delay = delay
        self.axis_id = axis_id

    def get_values(self) -> numpy.typing.NDArray[typing.Any]:
        return numpy.array(self.values)


@dataclasses.dataclass
class CollectionStep(ProcedureStepLike, Acquisition_.AcquisitionProcedureFactoryInterface.CollectionStep):
    step_id: str
    sub_step: ProcedureStepLike
    control_controller: ControlController

    def __init__(self, sub_step: ProcedureStepLike, control_controller: ControlController) -> None:
        self.step_id = "collection-step"
        self.sub_step = sub_step
        self.control_controller = control_controller


@dataclasses.dataclass
class SequentialStep(ProcedureStepLike, Acquisition_.AcquisitionProcedureFactoryInterface.SequentialStep):
    step_id: str
    sub_steps: typing.Sequence[ProcedureStepLike]

    def __init__(self, sub_steps: typing.Sequence[ProcedureStepLike]) -> None:
        self.step_id = "sequential-steps"
        self.sub_steps = sub_steps


@dataclasses.dataclass
class Acquisition(Acquisition_.AcquisitionProcedureFactoryInterface.AcquisitionProcedure):
    devices: typing.Mapping[str, Device]
    steps: typing.Sequence[ProcedureStepLike]


@dataclasses.dataclass
class ProcessingChannel(Acquisition_.AcquisitionProcedureFactoryInterface.ProcessingChannelLike):
    processing_id: str
    processing_parameters: typing.Optional[typing.Mapping[str, typing.Any]] = None


class DataStreamProducer(typing.Protocol):
    def get_data_stream(self) -> Acquisition_.DataStream:
        pass


class AcquisitionDeviceDataStreamProducer(DataStreamProducer):
    def __init__(self, acquisition_device: Acquisition_.AcquisitionDeviceLike, device_map: typing.MutableMapping[str, STEMController.DeviceController]) -> None:
        self.acquisition_device = acquisition_device
        self.device_map = device_map

    def get_data_stream(self) -> Acquisition_.DataStream:
        return self.acquisition_device.build_acquisition_device_data_stream(self.device_map)


class AcquisitionMethodDataStreamProducer(DataStreamProducer):
    def __init__(self, acquisition_method: Acquisition_.AcquisitionMethodLike, data_stream_producer: DataStreamProducer, device_map: typing.MutableMapping[str, STEMController.DeviceController]) -> None:
        self.acquisition_method = acquisition_method
        self.data_stream_producer = data_stream_producer
        self.device_map = device_map

    def get_data_stream(self) -> Acquisition_.DataStream:
        return self.acquisition_method.wrap_acquisition_device_data_stream(self.data_stream_producer.get_data_stream(), self.device_map)


class SequentialDataStreamProducer(DataStreamProducer):
    def __init__(self, data_stream_producers: typing.Sequence[DataStreamProducer]) -> None:
        self.data_stream_producers = data_stream_producers

    def get_data_stream(self) -> Acquisition_.DataStream:
        return Acquisition_.SequentialDataStream(tuple(data_stream_producer.get_data_stream() for data_stream_producer in self.data_stream_producers))


class AcquisitionController(Acquisition_.AcquisitionProcedureFactoryInterface.AcquisitionController):
    """Configurable acquisition controller.

    The acquisition controller is responsible for acquiring data from the instrument. It is created with an
    acquisition configuration.

    Changes to this API need to be strictly backwards compatible. Breaking changes should a new acquisition
    controller with a name like AcquisitionController<n>, where <n> is the version number of the API.
    """
    def __init__(self, acquisition: Acquisition) -> None:
        self.__stem_controller = self.__create_stem_controller(acquisition.devices)
        self.__devices = acquisition.devices
        self.__device_map: typing.MutableMapping[str, STEMController.DeviceController] = dict()
        self.__device_map["stem"] = STEMController.STEMDeviceController()
        self.__data_stream_producer = self.__create_data_stream(acquisition.steps[0] if len(acquisition.steps) == 1 else SequentialStep(acquisition.steps))

    def acquire_immediate(self) -> typing.Mapping[Acquisition_.Channel, DataAndMetadata.DataAndMetadata]:
        return Acquisition_.acquire_immediate(self.__data_stream_producer.get_data_stream())

    def __create_stem_controller(self, devices: typing.Mapping[str, Device]) -> STEMController.STEMController:
        stem_controller: typing.Optional[STEMController.STEMController] = None
        stem_controller_d = devices.get("stem")
        if not stem_controller and stem_controller_d and (device_id := stem_controller_d.device_id):
            stem_controller = typing.cast(typing.Optional[STEMController.STEMController], Registry.get_component(device_id))
        if not stem_controller:
            stem_controller = typing.cast(typing.Optional[STEMController.STEMController], Registry.get_component("stem_controller"))
        assert stem_controller
        return stem_controller

    def __create_acquisition_device(self,
                                    device: Device,
                                    device_parameters: typing.Optional[DeviceParametersLike],
                                    device_channels: typing.Optional[typing.Sequence[DeviceChannelSpecifier]],
                                    processing_channels: typing.Optional[typing.Sequence[ProcessingChannelLike]]) -> Acquisition_.AcquisitionDeviceLike:
        device_type_id = device.device_type_id
        stem_controller = self.__stem_controller
        if device_type_id == "scan":
            scan_hardware_source: typing.Optional[scan_base.ScanHardwareSource] = None
            if not scan_hardware_source and (device_id := device.device_id):
                scan_hardware_source = typing.cast(typing.Optional[scan_base.ScanHardwareSource], Registry.get_components_by_type(device_id))
            elif not scan_hardware_source:
                scan_hardware_source = stem_controller.scan_controller
            assert scan_hardware_source
            assert not processing_channels
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            if device_parameters:
                assert isinstance(device_parameters, ScanParameters)
                if device_parameters.pixel_time_us is not None:
                    scan_frame_parameters.pixel_time_us = device_parameters.pixel_time_us
                if device_parameters.pixel_size is not None:
                    scan_frame_parameters.pixel_size = device_parameters.pixel_size
                if device_parameters.fov_nm is not None:
                    scan_frame_parameters.fov_nm = device_parameters.fov_nm
                if device_parameters.rotation_rad is not None:
                    scan_frame_parameters.rotation_rad = device_parameters.rotation_rad
                if device_parameters.center_nm is not None:
                    scan_frame_parameters.center_nm = device_parameters.center_nm
                if device_parameters.subscan_pixel_size is not None:
                    scan_frame_parameters.subscan_pixel_size = device_parameters.subscan_pixel_size
                if device_parameters.subscan_fractional_size is not None:
                    scan_frame_parameters.subscan_fractional_size = device_parameters.subscan_fractional_size
                if device_parameters.subscan_fractional_center is not None:
                    scan_frame_parameters.subscan_fractional_center = device_parameters.subscan_fractional_center
                if device_parameters.subscan_rotation is not None:
                    scan_frame_parameters.subscan_rotation = device_parameters.subscan_rotation
                if device_parameters.ac_line_sync is not None:
                    scan_frame_parameters.ac_line_sync = device_parameters.ac_line_sync
                # TODO: move these to a plug-in handler
                if (flyback_time_us := typing.cast(typing.Optional[float], device_parameters.get_value("flyback_time_us"))) is not None:
                    setattr(scan_frame_parameters, "flyback_time_us", flyback_time_us)
                if (ac_frame_sync := typing.cast(typing.Optional[bool], device_parameters.get_value("ac_frame_sync"))) is not None:
                    setattr(scan_frame_parameters, "ac_frame_sync", ac_frame_sync)
            if device_channels is not None:
                channel_indexes = list()
                for device_channel in device_channels:
                    if device_channel.channel_index is not None:
                        channel_indexes.append(device_channel.channel_index)
                assert len(channel_indexes) > 0
                scan_frame_parameters.enabled_channel_indexes = channel_indexes
            return scan_base.ScanAcquisitionDevice(scan_hardware_source, scan_frame_parameters)
        elif device_type_id in ("ronchigram", "camera"):
            camera_hardware_source: typing.Optional[camera_base.CameraHardwareSource] = None
            if not camera_hardware_source and (device_id := device.device_id):
                camera_hardware_source = typing.cast(typing.Optional[camera_base.CameraHardwareSource], Registry.get_components_by_type(device_id))
            elif not camera_hardware_source:
                camera_hardware_source = stem_controller.ronchigram_camera
            assert camera_hardware_source
            assert not processing_channels
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_channel = "ronchigram" if device_type_id == "ronchigram" else None
            if device_parameters:
                assert isinstance(device_parameters, CameraParameters)
                if device_parameters.exposure_ms is not None:
                    camera_frame_parameters.exposure_ms = device_parameters.exposure_ms
            return camera_base.CameraAcquisitionDevice(camera_hardware_source, camera_frame_parameters, camera_channel)
        elif device_type_id == "eels":
            eels_hardware_source: typing.Optional[camera_base.CameraHardwareSource] = None
            if not eels_hardware_source and (device_id := device.device_id):
                eels_hardware_source = typing.cast(typing.Optional[camera_base.CameraHardwareSource], Registry.get_components_by_type(device_id))
            elif not eels_hardware_source:
                eels_hardware_source = stem_controller.eels_camera
            assert eels_hardware_source
            eels_frame_parameters = eels_hardware_source.get_current_frame_parameters()
            eels_camera_channel = "eels_spectrum" if processing_channels else "eels_image"
            # if processing_id == "sum" and processing_parameters and processing_parameters.get("axis") == 0 and isinstance(camera_acquisition_step, DeviceAcquisitionStep) and camera_acquisition_step.device_acquisition_parameters.device_type_id in ("eels", "camera", "ronchigrma"):
            if device_parameters:
                assert isinstance(device_parameters, CameraParameters)
                if device_parameters.exposure_ms is not None:
                    eels_frame_parameters.exposure_ms = device_parameters.exposure_ms
            return camera_base.CameraAcquisitionDevice(eels_hardware_source, eels_frame_parameters, eels_camera_channel)
        raise NotImplementedError(f"device {device.device_type_id=}/{device.device_id=} not supported")

    def __create_control_customization(self, controller: DeviceControlController) -> AcquisitionPreferences.ControlCustomization:
        # this is a hack to configure a control customization from a control controller.
        # take the device_control_id from the control_description.
        control_customization_entity_type = Schema.get_entity_type("control_customization")
        assert control_customization_entity_type
        control_customization = AcquisitionPreferences.ControlCustomization(control_customization_entity_type, None)
        control_customization._set_field_value("control_id", controller.control_id)
        control_description = control_customization.control_description
        assert control_description
        control_customization.device_control_id = controller.device_control_id or control_description.device_control_id
        control_customization.delay = controller.delay
        return control_customization

    def __create_data_stream(self, step: ProcedureStepLike) -> DataStreamProducer:
        if isinstance(step, DeviceAcquisitionStep):
            device_acquisition_parameters = step.device_acquisition_parameters
            device_type_id = device_acquisition_parameters.device_type_id
            acquisition_device = self.__create_acquisition_device(self.__devices[device_type_id], device_acquisition_parameters.device_parameters, device_acquisition_parameters.device_channels, device_acquisition_parameters.processing_channels)
            return AcquisitionDeviceDataStreamProducer(acquisition_device, self.__device_map)
        elif isinstance(step, MultiDeviceAcquisitionStep):
            primary_device_acquisition_parameters = step.primary_device_acquisition_parameters
            primary_data_stream_producer = self.__create_data_stream(DeviceAcquisitionStep(primary_device_acquisition_parameters))
            secondary_data_stream_producers: typing.List[DataStreamProducer] = list()
            for secondary_device_acquisition_parameters in step.secondary_device_acquisition_parameters:
                secondary_data_stream_producers.append(self.__create_data_stream(DeviceAcquisitionStep(secondary_device_acquisition_parameters)))
            secondary_data_stream_producer = secondary_data_stream_producers[0] if len(secondary_data_stream_producers) == 1 else None
            if isinstance(primary_data_stream_producer, AcquisitionDeviceDataStreamProducer) and isinstance(secondary_data_stream_producer, AcquisitionDeviceDataStreamProducer):
                if isinstance(primary_data_stream_producer.acquisition_device, camera_base.CameraAcquisitionDevice) and isinstance(secondary_data_stream_producer.acquisition_device, scan_base.ScanAcquisitionDevice):
                    camera_acquisition_device = primary_data_stream_producer.acquisition_device
                    scan_acquisition_device = secondary_data_stream_producer.acquisition_device
                    drift_parameters = step.drift_parameters
                    drift_correction_enabled = drift_parameters.drift_correction_enabled if drift_parameters else False
                    drift_interval_lines = drift_parameters.drift_interval_lines if drift_parameters else 0
                    drift_interval_scans = drift_parameters.drift_interval_scans if drift_parameters else 0
                    drift_channel_id: typing.Optional[str] = None
                    if drift_parameters and (drift_channel := drift_parameters.drift_channel):
                        if drift_channel.channel_id:
                            drift_channel_id = drift_channel.channel_id
                        elif (drift_channel_index := drift_channel.channel_index) is not None:
                            drift_channel_id  = scan_acquisition_device._scan_hardware_source.get_channel_id(drift_channel_index)
                    drift_region = drift_parameters.drift_region if drift_parameters else None
                    drift_rotation = drift_parameters.drift_rotation if drift_parameters else 0.0
                    acquisition_device = scan_base.SynchronizedScanAcquisitionDevice(
                        scan_acquisition_device._scan_hardware_source,
                        scan_acquisition_device._scan_frame_parameters,
                        camera_acquisition_device._camera_hardware_source,
                        camera_acquisition_device._camera_frame_parameters,
                        camera_acquisition_device._camera_channel,
                        drift_correction_enabled,
                        drift_interval_lines,
                        drift_interval_scans,
                        drift_channel_id,
                        drift_region,
                        drift_rotation)
                    return AcquisitionDeviceDataStreamProducer(acquisition_device, self.__device_map)
        elif isinstance(step, CollectionStep):
            device_data_stream = self.__create_data_stream(step.sub_step)
            acquisition_method: Acquisition_.AcquisitionMethodLike
            if isinstance(step.control_controller, DeviceControlController):
                values = step.control_controller.get_values()
                values_shape = values.shape
                if len(values_shape) == 1:
                    values = values.reshape((values_shape[0], 1))
                    values_shape = values.shape
                if len(values_shape) == 2:
                    acquisition_method = Acquisition_.SeriesAcquisitionMethod(
                        self.__create_control_customization(step.control_controller),
                        values)
                    return AcquisitionMethodDataStreamProducer(acquisition_method, device_data_stream, self.__device_map)
                elif len(values_shape) == 3:
                    acquisition_method = Acquisition_.TableAcquisitionMethod(
                        self.__create_control_customization(step.control_controller),
                        step.control_controller.axis_id,
                        values)
                    return AcquisitionMethodDataStreamProducer(acquisition_method, device_data_stream, self.__device_map)
                else:
                    raise NotImplementedError(f"control controller {step.control_controller=} with shape {values_shape} not supported")
            raise NotImplementedError(f"control controller {step.control_controller=} not supported")
        elif isinstance(step, SequentialStep):
            sequential_data_streams = [self.__create_data_stream(ds) for ds in step.sub_steps]
            return SequentialDataStreamProducer(sequential_data_streams)
        raise NotImplementedError(f"data stream {step.step_id=} not supported")


class AcquisitionProcedureFactoryInterface(Acquisition_.AcquisitionProcedureFactoryInterface):
    def create_device(self,
                      device_type_id: str, *,
                      device_id: typing.Optional[str] = None) -> Device:
        return Device(device_type_id, device_id)

    def create_stem_device(self, *, device_id: typing.Optional[str] = None) -> Device:
        return Device("stem", device_id)

    def create_scan_device(self, *, device_id: typing.Optional[str] = None) -> Device:
        return Device("scan", device_id)

    def create_ronchigram_device(self, *, device_id: typing.Optional[str] = None) -> Device:
        return Device("ronchigram", device_id)

    def create_eels_device(self, *, device_id: typing.Optional[str] = None) -> Device:
        return Device("eels", device_id)

    def create_device_channel_specifier(self, *,
                                        channel_index: typing.Optional[int] = None,
                                        channel_type_id: typing.Optional[str] = None,
                                        channel_id: typing.Optional[str] = None) -> DeviceChannelSpecifier:
        return DeviceChannelSpecifier(channel_index, channel_type_id, channel_id)

    def create_magnification_parameters(self, *,
                                        fov_nm: typing.Optional[float] = None,
                                        rotation_rad: typing.Optional[float] = None) -> MagnificationParameters:
        return MagnificationParameters(fov_nm, rotation_rad)

    def create_scan_parameters(self, *,
                               pixel_time_us: typing.Optional[float] = None,
                               pixel_size: typing.Optional[Geometry.IntSize] = None,
                               fov_nm: typing.Optional[float] = None,
                               rotation_rad: typing.Optional[float] = None,
                               center_nm: typing.Optional[Geometry.FloatPoint] = None,
                               subscan_pixel_size: typing.Optional[Geometry.IntSize] = None,
                               subscan_fractional_size: typing.Optional[Geometry.FloatSize] = None,
                               subscan_fractional_center: typing.Optional[Geometry.FloatPoint] = None,
                               subscan_rotation: typing.Optional[float] = None,
                               ac_line_sync: typing.Optional[bool] = None,
                               # flyback_time_us: typing.Optional[float] = None,
                               # ac_frame_sync: typing.Optional[bool] = None,
                               **kwargs: typing.Any
                               ) -> ScanParameters:
        return ScanParameters(pixel_time_us, pixel_size, fov_nm, rotation_rad, center_nm, subscan_pixel_size,
                              subscan_fractional_size, subscan_fractional_center, subscan_rotation,
                              ac_line_sync, **kwargs)

    def create_camera_parameters(self, *,
                                 exposure_ms: typing.Optional[float] = None,
                                 binning: typing.Optional[int] = None,
                                 **kwargs: typing.Any
                                 ) -> CameraParameters:
        return CameraParameters(exposure_ms, binning, **kwargs)

    def create_drift_parameters(self, *,
                                drift_correction_enabled: bool = False,
                                drift_interval_lines: int = 0,
                                drift_scan_lines: int = 0,
                                drift_channel: typing.Optional[AcquisitionProcedureFactoryInterface.DeviceChannelSpecifier] = None,
                                drift_region: typing.Optional[Geometry.FloatRect] = None,
                                drift_rotation: float = 0.0,
                                **kwargs: typing.Any,
                                ) -> AcquisitionProcedureFactoryInterface.DriftCorrectionParameters:
        return DriftCorrectionParameters(drift_correction_enabled,
                                         drift_interval_lines,
                                         drift_scan_lines,
                                         typing.cast(typing.Optional[DeviceChannelSpecifier], drift_channel),
                                         drift_region,
                                         drift_rotation)

    def create_device_acquisition_parameters(self, *,
                                             device: AcquisitionProcedureFactoryInterface.Device,
                                             device_parameters: typing.Optional[AcquisitionProcedureFactoryInterface.DeviceParametersLike] = None,
                                             device_channels: typing.Optional[typing.Sequence[AcquisitionProcedureFactoryInterface.DeviceChannelSpecifier]] = None,
                                             processing_channels: typing.Optional[typing.Sequence[AcquisitionProcedureFactoryInterface.ProcessingChannelLike]] = None,
                                             ) -> AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters:
        return DeviceAcquisitionParameters(typing.cast(Device, device).device_type_id, typing.cast(DeviceParametersLike, device_parameters),
                                           typing.cast(typing.Optional[typing.Sequence[DeviceChannelSpecifier]], device_channels),
                                           typing.cast(typing.Optional[typing.Sequence[ProcessingChannelLike]], processing_channels))

    def create_device_acquisition_step(self, *,
                                       device_acquisition_parameters: AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters,
                                       ) -> AcquisitionProcedureFactoryInterface.DeviceAcquisitionStep:
        return DeviceAcquisitionStep(typing.cast(DeviceAcquisitionParameters, device_acquisition_parameters))

    def create_multi_device_acquisition_step(self, *,
                                             primary_device_acquisition_parameters: AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters,
                                             secondary_device_acquisition_parameters: typing.Sequence[Acquisition_.AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters],
                                             drift_parameters: typing.Optional[Acquisition_.AcquisitionProcedureFactoryInterface.DriftCorrectionParameters] = None) -> MultiDeviceAcquisitionStep:
        return MultiDeviceAcquisitionStep(typing.cast(DeviceAcquisitionParameters, primary_device_acquisition_parameters), typing.cast(typing.Sequence[DeviceAcquisitionParameters], secondary_device_acquisition_parameters), typing.cast(DriftCorrectionParameters, drift_parameters))

    def create_device_controller(self, *,
                                 device: Acquisition_.AcquisitionProcedureFactoryInterface.Device,
                                 control_id: str,
                                 device_control_id: typing.Optional[str] = None,
                                 values: typing.Optional[numpy.typing.ArrayLike] = None,
                                 delay: typing.Optional[float] = None,
                                 axis_id: typing.Optional[str] = None) -> Acquisition_.AcquisitionProcedureFactoryInterface.DeviceControlController:
        return DeviceControlController(typing.cast(Device, device).device_type_id, control_id, device_control_id, values, delay, axis_id)

    def create_collection_step(self, *,
                               sub_step: AcquisitionProcedureFactoryInterface.ProcedureStepLike,
                               control_controller: Acquisition_.AcquisitionProcedureFactoryInterface.ControlController) -> CollectionStep:
        """Create a table from an input data stream and a controller."""
        return CollectionStep(typing.cast(ProcedureStepLike, sub_step), typing.cast(ControlController, control_controller))

    def create_sequential_step(self, *,
                               sub_steps: typing.Sequence[Acquisition_.AcquisitionProcedureFactoryInterface.ProcedureStepLike]) -> SequentialStep:
        """Create a sequential acquisition from a sequence of data streams."""
        return SequentialStep(typing.cast(typing.Sequence[ProcedureStepLike], sub_steps))

    def create_acquisition_procedure(self, *,
                                     devices: typing.Sequence[Acquisition_.AcquisitionProcedureFactoryInterface.Device],
                                     steps: typing.Sequence[Acquisition_.AcquisitionProcedureFactoryInterface.ProcedureStepLike]) -> Acquisition:
        """Create an acquisition from a table of devices and a data stream acquiring from those devices."""
        return Acquisition({device.device_type_id: device for device in typing.cast(typing.Sequence[Device], devices)}, typing.cast(typing.Sequence[ProcedureStepLike], steps))

    def create_processing_channel(self, *,
                                  processing_id: str,
                                  processing_parameters: typing.Optional[typing.Mapping[str, typing.Any]] = None
                                  ) -> Acquisition_.AcquisitionProcedureFactoryInterface.ProcessingChannelLike:
        """Create a processing data stream from an input data stream and a processing description."""
        return ProcessingChannel(processing_id, processing_parameters)

    def create_acquisition_controller(self, *, acquisition_procedure: Acquisition_.AcquisitionProcedureFactoryInterface.AcquisitionProcedure) -> AcquisitionController:
        return AcquisitionController(typing.cast(Acquisition, acquisition_procedure))


Registry.register_component(AcquisitionProcedureFactoryInterface(), {"acquisition_procedure_factory_interface"})


"""
Architectural Decision Records.

ADR FUTURE: Reference images, if used, may be hashed and stored in the acquisition description. This would allow future
callers to confirm the data had not changed.

ADR FUTURE: How are reference images included into the final output?

ADR 2023-09-21: By default, enclosing data streams capture the enclosed data streams and the enclosed streams do not 
produce output on their own. This heuristic makes top level streams be the output streams. However, the user may also 
specify that an enclosed stream should produce output. If this is done at the enclosing data stream level, 
then the enclosing data stream collates the output from the enclosed data streams. If this is done independently of 
the enclosing data stream, then the enclosed data stream produces output on its own but in a view-style mode. This is 
useful for debugging and viewing.

ADR 2023-09-20: Processing and analysis can be applied to device data streams and produce new data streams. The 
caller should have the option to suppress the output for the primary device data stream and only see the processed 
data streams or to include both. The low level device may support some forms of processing and should be allowed to 
do so (e.g. camera dark/gain; or camera integration in y-axis; or camera 4D STEM processing).

ADR 2023-09-20: Channels should have the option of being specified by identifier, instance-specific identifier, 
or by index. This helps with making the acquisition reusable and able to handle dynamic channel changes within a 
session (e.g. a different microscope mode produces different scan channels). The identifier may be device specific; 
but may also be a predefined identifier which represents commonly used channels (e.g. HAADF, MAADF, etc.). Also, 
see NOTE 2023-09-14.

NOTE 2023-09-14: Channel id and name are not used on cameras. On scans, channel id's are trivial ('a', 'b', etc.) and
channel names are valid. Channel indexes are always valid.

ADR 2023-09-13. Allow callers to indicate the magnification settings with the scan parameters since scanning is the 
way they get activated anyway.
"""
