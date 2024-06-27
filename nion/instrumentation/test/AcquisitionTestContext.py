import gc
import logging
import typing
import unittest

from nion.instrumentation import camera_base
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.swift.test import TestContext
from nion.utils import Event
from nion.utils import Registry
from nionswift_plugin.usim import CameraDevice
from nionswift_plugin.usim import InstrumentDevice
from nionswift_plugin.usim import ScanDevice


class AcquisitionTestContext(TestContext.MemoryProfileContext):
    def __init__(self, *, is_eels: bool = False, camera_exposure: float = 0.025, is_both_cameras: bool = False):
        super().__init__()
        assert not is_eels or not is_both_cameras
        logging.getLogger("acquisition").setLevel(logging.ERROR)
        HardwareSource.run()
        instrument = self.setup_stem_controller()
        DriftTracker.run()
        ScanDevice.run(typing.cast(InstrumentDevice.Instrument, instrument))
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
        ScanDevice.stop()
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
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        Registry.register_component(instrument, {"stem_controller"})
        return instrument

    def setup_scan_hardware_source(self, stem_controller: stem_controller.STEMController) -> scan_base.ScanHardwareSource:
        instrument = typing.cast(InstrumentDevice.Instrument, stem_controller)
        scan_module = ScanDevice.ScanModule(instrument)
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
        camera_device = CameraDevice.Camera(camera_id, camera_type, camera_name, instrument)
        if getattr(camera_device, "camera_version", 2) == 3:
            camera_hardware_source = camera_base.CameraHardwareSource3("usim_stem_controller", camera_device, camera_settings, None, None)
        else:
            camera_hardware_source = camera_base.CameraHardwareSource2("usim_stem_controller", camera_device, camera_settings, None, None)
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
