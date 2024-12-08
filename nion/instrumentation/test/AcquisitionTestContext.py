import gc
import logging
import math
import typing
import unittest

import numpy
import numpy.typing
import scipy

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import camera_base
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.swift.test import TestContext
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry
from nion.device_kit import CameraDevice
from nion.device_kit import InstrumentDevice
from nion.device_kit import ScanDevice


class ScanModule(scan_base.ScanModule):
    def __init__(self, instrument: InstrumentDevice.Instrument) -> None:
        self.stem_controller_id = instrument.instrument_id
        self.device = ScanDevice.Device("usim_scan_device", "uSim Scan", instrument)
        setattr(self.device, "priority", 20)
        scan_modes = (
            scan_base.ScanSettingsMode("Fast", "fast", ScanDevice.ScanFrameParameters(pixel_size=(256, 256), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.1)),
            scan_base.ScanSettingsMode("Slow", "slow", ScanDevice.ScanFrameParameters(pixel_size=(512, 512), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.4)),
            scan_base.ScanSettingsMode("Record", "record", ScanDevice.ScanFrameParameters(pixel_size=(1024, 1024), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 1.0))
        )
        self.settings = scan_base.ScanSettings(scan_modes, lambda d: ScanDevice.ScanFrameParameters(d), 0, 2)


class CameraSimulator:
    def __init__(self, sensor_dimensions: typing.Optional[Geometry.IntSize]) -> None:
        self.__data_value = 0
        self.__sensor_dimensions = sensor_dimensions

    def close(self) -> None:
        pass

    def get_dimensional_calibrations(self, readout_area: typing.Optional[Geometry.IntRect], binning_shape: typing.Optional[Geometry.IntSize]) -> typing.Sequence[Calibration.Calibration]:
        dimensional_calibrations = [
            Calibration.Calibration(),
            Calibration.Calibration()
        ]
        return dimensional_calibrations

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, probe_position: typing.Optional[Geometry.FloatPoint]) -> DataAndMetadata.DataAndMetadata:
        self.__data_value += 1
        shape = self.__sensor_dimensions if self.__sensor_dimensions else readout_area.size
        data = numpy.random.randn(shape.height // binning_shape.height, shape.width // binning_shape.width) * exposure_s
        return DataAndMetadata.new_data_and_metadata(data)


class ScanDataGenerator:
    def __init__(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(100)
        pattern = scipy.ndimage.zoom(numpy.abs(numpy.random.randn(40, 40)), 25) + scipy.ndimage.zoom(numpy.abs(numpy.random.randn(100, 100)), 10)
        numpy.random.set_state(random_state)
        self.__pattern = typing.cast(numpy.typing.NDArray[numpy.float32], pattern)

    def generate_scan_data(self, instrument: InstrumentDevice.Instrument, scan_frame_parameters: ScanDevice.ScanFrameParameters) -> numpy.typing.NDArray[numpy.float32]:
        pattern = self.__pattern
        shift_nm = Geometry.FloatPoint(instrument.GetVal("CSH.y") * 1e9, instrument.GetVal("CSH.x") * 1e9)  # for drift tests
        size = scan_frame_parameters.size
        fov_size_nm = scan_frame_parameters.fov_size_nm
        rotation = scan_frame_parameters.rotation_rad
        center_nm = scan_frame_parameters.center_nm
        y_start = (50 + center_nm.y + shift_nm.y - fov_size_nm.height / 2) / 100 * pattern.shape[0]
        y_length = fov_size_nm.height / 100 * pattern.shape[0]
        x_start = (50 + center_nm.x + shift_nm.x - fov_size_nm.width / 2) / 100 * pattern.shape[1]
        x_length = fov_size_nm.width / 100 * pattern.shape[1]
        iy, ix = numpy.meshgrid(numpy.arange(size.width), numpy.arange(size.height))
        y = iy * y_length / size.height - y_length / 2
        x = ix * x_length / size.width - x_length / 2
        angle_sin = math.sin(-rotation)
        angle_cos = math.cos(-rotation)
        coordinates = [y_start + y_length / 2 + (x * angle_cos - y * angle_sin), x_start + x_length / 2 + (y * angle_cos + x * angle_sin)]
        return typing.cast(numpy.typing.NDArray[numpy.float32], scipy.ndimage.map_coordinates(pattern, coordinates, order=1) + numpy.random.randn(*size) * 0.1)


class AcquisitionTestContext(TestContext.MemoryProfileContext):
    def __init__(self, *, is_eels: bool = False, camera_exposure: float = 0.025, is_both_cameras: bool = False):
        super().__init__()
        assert not is_eels or not is_both_cameras
        logging.getLogger("acquisition").setLevel(logging.ERROR)
        HardwareSource.run()
        instrument = self.setup_stem_controller()
        DriftTracker.run()
        Registry.register_component(ScanModule(instrument), {"scan_module"})
        scan_hardware_source = self.setup_scan_hardware_source(instrument)
        camera_hardware_source = self.setup_camera_hardware_source(instrument, camera_exposure, is_eels)
        eels_hardware_source = self.setup_camera_hardware_source(instrument, camera_exposure, True) if is_both_cameras else None
        HardwareSource.HardwareSourceManager()._hardware_source_list_model.clear_items()
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()
        self.instrument = instrument
        self.scan_hardware_source = scan_hardware_source
        self.camera_hardware_source = camera_hardware_source
        self.eels_hardware_source = eels_hardware_source
        HardwareSource.HardwareSourceManager().register_hardware_source(self.camera_hardware_source)
        if self.eels_hardware_source:
            HardwareSource.HardwareSourceManager().register_hardware_source(self.eels_hardware_source)
        HardwareSource.HardwareSourceManager().register_hardware_source(self.scan_hardware_source)
        self.document_controller = self.create_document_controller(auto_close=False)
        self.document_model = self.document_controller.document_model
        stem_controller.register_event_loop(self.document_controller.event_loop)
        self.__exit_stack: typing.List[typing.Any] = list()

    def close(self) -> None:
        gc.collect()  # allow acquisition objects to be garbage collected
        self.document_controller.periodic()
        self.document_controller.close()
        for ex in self.__exit_stack:
            ex.close()
        stem_controller.unregister_event_loop()
        camera_type = self.camera_hardware_source.camera.camera_type
        eels_type = self.eels_hardware_source.camera.camera_type if self.eels_hardware_source else None
        self.camera_hardware_source.close()
        if self.eels_hardware_source:
            self.eels_hardware_source.close()
        self.scan_hardware_source.close()
        HardwareSource.HardwareSourceManager().unregister_hardware_source(self.camera_hardware_source)
        if self.eels_hardware_source:
            HardwareSource.HardwareSourceManager().unregister_hardware_source(self.eels_hardware_source)
        HardwareSource.HardwareSourceManager().unregister_hardware_source(self.scan_hardware_source)
        Registry.unregister_component(Registry.get_component("scan_device"), {"scan_device"})
        DriftTracker.stop()
        Registry.unregister_component(Registry.get_component("scan_module"), {"scan_module"})
        Registry.unregister_component(Registry.get_component("stem_controller"), {"stem_controller"})
        Registry.unregister_component(Registry.get_component("scan_hardware_source"), {"hardware_source", "scan_hardware_source"})
        Registry.unregister_component(self.camera_hardware_source, {"hardware_source", "camera_hardware_source", camera_type + "_camera_hardware_source"})
        if self.eels_hardware_source:
            Registry.unregister_component(self.eels_hardware_source, {"hardware_source", "camera_hardware_source", eels_type + "_camera_hardware_source"})
        HardwareSource.HardwareSourceManager()._close_instruments()
        HardwareSource.stop()
        super().close()

    def push(self, ex) -> None:
        self.__exit_stack.append(ex)

    def setup_stem_controller(self) -> stem_controller.STEMController:
        instrument = InstrumentDevice.Instrument("usim_stem_controller", ScanDataGenerator())
        Registry.register_component(instrument, {"stem_controller"})
        return instrument

    def setup_scan_hardware_source(self, stem_controller: stem_controller.STEMController) -> scan_base.ScanHardwareSource:
        instrument = typing.cast(InstrumentDevice.Instrument, stem_controller)
        scan_module = ScanModule(instrument)
        scan_device = scan_module.device
        scan_settings = scan_module.settings
        Registry.register_component(scan_device, {"scan_device"})
        scan_hardware_source = scan_base.ConcreteScanHardwareSource(stem_controller, scan_device, scan_settings, None)
        setattr(scan_device, "hardware_source", scan_hardware_source)
        Registry.register_component(scan_hardware_source, {"hardware_source", "scan_hardware_source"})  # allows stem controller to find scan controller
        HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)
        return scan_hardware_source

    def setup_camera_hardware_source(self, stem_controller: stem_controller.STEMController, camera_exposure: float, is_eels: bool) -> camera_base.CameraHardwareSource:
        instrument = typing.cast(InstrumentDevice.Instrument, stem_controller)
        camera_id = "usim_ronchigram_camera" if not is_eels else "usim_eels_camera"
        camera_type = "ronchigram" if not is_eels else "eels"
        camera_name = "uSim Camera"
        camera_settings = CameraDevice.CameraSettings(camera_id)
        sensor_dimensions = Geometry.IntSize(256, 1024) if is_eels else None
        camera_device = CameraDevice.Camera(camera_id, camera_type, camera_name, CameraSimulator(sensor_dimensions), instrument)
        camera_hardware_source = camera_base.CameraHardwareSource3("usim_stem_controller", camera_device, camera_settings, None, None)
        if is_eels:
            camera_hardware_source.features["is_eels_camera"] = True
        camera_hardware_source.set_frame_parameters(0, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(1, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(2, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000 * 2, "binning": 1}))
        camera_hardware_source.set_selected_profile_index(0)
        Registry.register_component(camera_hardware_source, {"hardware_source", "camera_hardware_source", camera_device.camera_type + "_camera_hardware_source"})  # allows stem controller to find camera controller
        HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)
        return camera_hardware_source


def test_context(*, is_eels: bool = False, camera_exposure: float = 0.025, is_both_cameras: bool = False) -> AcquisitionTestContext:
    return AcquisitionTestContext(is_eels=is_eels, camera_exposure=camera_exposure, is_both_cameras=is_both_cameras)


def begin_leaks() -> None:
    TestContext.begin_leaks()

def end_leaks(test_case: unittest.TestCase) -> None:
    test_case.assertEqual(0, len(Registry.get_components_by_type("hardware_source_manager")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("stem_controller")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("scan_device")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("camera_device")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("video_device")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("scan_hardware_source")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("camera_hardware_source")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("video_hardware_source")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("hardware_source")))
    test_case.assertEqual(0, len(Registry.get_components_by_type("document_model")))
    test_case.assertEqual(0, stem_controller.ScanContextController.count)
    test_case.assertEqual(0, stem_controller.ProbeView.count)
    test_case.assertEqual(0, stem_controller.SubscanView.count)
    test_case.assertEqual(0, stem_controller.LineScanView.count)
    test_case.assertEqual(0, stem_controller.DriftView.count)
    TestContext.end_leaks(test_case)
