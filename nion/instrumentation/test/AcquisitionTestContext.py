import typing

from nion.instrumentation import camera_base
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.swift.test import TestContext
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry
from nionswift_plugin.usim import CameraDevice
from nionswift_plugin.usim import InstrumentDevice
from nionswift_plugin.usim import ScanDevice


class AcquisitionTestContext(TestContext.MemoryProfileContext):
    def __init__(self, *, is_eels: bool = False, camera_exposure: float = 0.025):
        super().__init__()

        # HardwareSource.run()
        # camera_base.run(configuration_location)
        # scan_base.run()
        # video_base.run()
        # CameraControlPanel.run()
        # ScanControlPanel.run()
        # MultipleShiftEELSAcquire.run()
        # VideoControlPanel.run()

        HardwareSource.run()
        instrument = self.setup_stem_controller()
        DriftTracker.run()
        ScanDevice.run(typing.cast(InstrumentDevice.Instrument, instrument))
        scan_base.run()
        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, Registry.get_component("scan_hardware_source"))
        camera_hardware_source = self.setup_camera_hardware_source(instrument, camera_exposure, is_eels)
        HardwareSource.HardwareSourceManager().hardware_sources = []
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()
        self.instrument = instrument
        self.scan_hardware_source = scan_hardware_source
        self.camera_hardware_source = camera_hardware_source
        HardwareSource.HardwareSourceManager().register_hardware_source(self.camera_hardware_source)
        HardwareSource.HardwareSourceManager().register_hardware_source(self.scan_hardware_source)
        self.document_controller = self.create_document_controller(auto_close=False)
        self.document_model = self.document_controller.document_model
        stem_controller.register_event_loop(self.document_controller.event_loop)
        self.__exit_stack: typing.List[typing.Any] = list()

    def close(self) -> None:
        self.document_controller.periodic()
        self.document_controller.close()
        for ex in self.__exit_stack:
            ex.close()
        stem_controller.unregister_event_loop()
        self.camera_hardware_source.close()
        HardwareSource.HardwareSourceManager().unregister_hardware_source(self.camera_hardware_source)
        DriftTracker.stop()
        ScanDevice.stop()
        scan_base.stop()
        Registry.unregister_component(Registry.get_component("stem_controller"), {"stem_controller"})
        HardwareSource.HardwareSourceManager()._close_instruments()
        HardwareSource.stop()
        super().close()

    def push(self, ex) -> None:
        self.__exit_stack.append(ex)

    def setup_stem_controller(self) -> stem_controller.STEMController:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        Registry.register_component(instrument, {"stem_controller"})
        return instrument

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
        return camera_hardware_source


def test_context(*, is_eels: bool = False, camera_exposure: float = 0.025) -> AcquisitionTestContext:
    return AcquisitionTestContext(is_eels=is_eels, camera_exposure=camera_exposure)
