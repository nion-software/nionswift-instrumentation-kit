import gc
import logging
import typing
import unittest

from nion.instrumentation import camera_base
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.swift.test import TestContext
from nion.utils import Event
from nion.utils import Registry


class AcquisitionTestContextConfigurationLike(typing.Protocol):
    configuration_location: str
    instrument_id: str
    instrument: stem_controller.STEMController
    scan_module: scan_base.ScanModule
    ronchigram_camera_device_id: str
    ronchigram_camera_device: camera_base.CameraDevice3
    ronchigram_camera_settings: camera_base.CameraSettings
    eels_camera_device_id: str
    eels_camera_device: camera_base.CameraDevice3
    eels_camera_settings: camera_base.CameraSettings

    def run(self) -> None: ...

    def stop(self) -> None: ...


class AcquisitionTestContext(TestContext.MemoryProfileContext):
    def __init__(self, configuration: AcquisitionTestContextConfigurationLike, *, is_eels: bool = False, camera_exposure: float = 0.025, is_both_cameras: bool = False, is_app: bool = False) -> None:
        super().__init__()
        assert not is_eels or not is_both_cameras
        logging.getLogger("acquisition").setLevel(logging.ERROR)
        from nionswift_plugin import nion_instrumentation_ui
        nion_instrumentation_ui.configuration_location = configuration.configuration_location
        nion_instrumentation_ui.run()
        HardwareSource.HardwareSourceManager()._hardware_source_list_model.clear_items()
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()
        configuration.run()
        scan_hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(configuration.scan_module.device.scan_device_id)
        self._ronchigram_camera_hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(configuration.ronchigram_camera_device_id)
        self._eels_camera_hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(configuration.eels_camera_device_id)
        self.configuration = configuration
        self.instrument = configuration.instrument
        self.scan_hardware_source = scan_hardware_source
        self.camera_hardware_source = self._eels_camera_hardware_source if is_eels else self._ronchigram_camera_hardware_source
        self.camera_device_id = configuration.eels_camera_device_id if is_eels else configuration.ronchigram_camera_device_id
        self.camera_type = "eels" if is_eels else "ronchigram"
        self.eels_hardware_source = self._eels_camera_hardware_source if is_both_cameras else None
        self.eels_camera_device_id = configuration.eels_camera_device_id if is_both_cameras else None
        self.document_controller = self.create_document_controller_with_application(auto_close=False) if is_app else self.create_document_controller(auto_close=False)
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
        self.configuration.stop()
        from nionswift_plugin import nion_instrumentation_ui
        nion_instrumentation_ui.stop()
        super().close()

    def push(self, ex: typing.Any) -> None:
        self.__exit_stack.append(ex)

    def __setup_scan_hardware_source(self, stem_controller: stem_controller.STEMController, scan_device: scan_base.ScanDevice, scan_settings: scan_base.ScanSettingsProtocol) -> scan_base.ScanHardwareSource:
        scan_hardware_source = scan_base.ConcreteScanHardwareSource(stem_controller, scan_device, scan_settings, None)
        setattr(scan_device, "hardware_source", scan_hardware_source)
        Registry.register_component(scan_hardware_source, {"hardware_source", "scan_hardware_source"})  # allows stem controller to find scan controller
        HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)
        return scan_hardware_source

    def __setup_camera_hardware_source(self, instrument_id: str, camera_device: camera_base.CameraDevice3, camera_settings: camera_base.CameraSettings, camera_exposure: float, is_eels: bool) -> camera_base.CameraHardwareSource:
        camera_type = "eels" if is_eels else "ronchigram"
        camera_hardware_source = camera_base.CameraHardwareSource3(instrument_id, camera_device, camera_settings, None, None)
        if is_eels:
            camera_hardware_source.features["is_eels_camera"] = True
        camera_hardware_source.set_frame_parameters(0, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(1, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(2, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000 * 2, "binning": 1}))
        camera_hardware_source.set_selected_profile_index(0)
        Registry.register_component(camera_hardware_source, {"hardware_source", "camera_hardware_source", camera_type + "_camera_hardware_source"})  # allows stem controller to find camera controller
        HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)
        return camera_hardware_source


def test_context(configuration: typing.Optional[AcquisitionTestContextConfigurationLike] = None, *, is_eels: bool = False, camera_exposure: float = 0.025, is_both_cameras: bool = False, is_app: bool = False) -> AcquisitionTestContext:
    configuration_ = configuration or AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration()
    return AcquisitionTestContext(configuration_, is_eels=is_eels, camera_exposure=camera_exposure, is_both_cameras=is_both_cameras, is_app=is_app)


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
