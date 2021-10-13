import typing

from nion.instrumentation import camera_base
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
    def __init__(self, instrument: stem_controller.STEMController, scan_hardware_source, camera_hardware_source):
        super().__init__()
        HardwareSource.HardwareSourceManager().hardware_sources = []
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()
        self.document_controller = self.create_document_controller(auto_close=False)
        self.document_model = self.document_controller.document_model
        self.instrument = instrument
        self.scan_hardware_source = scan_hardware_source
        self.camera_hardware_source = camera_hardware_source
        HardwareSource.HardwareSourceManager().register_hardware_source(self.camera_hardware_source)
        HardwareSource.HardwareSourceManager().register_hardware_source(self.scan_hardware_source)
        self.scan_context_controller = stem_controller.ScanContextController(self.document_model,
                                                                             self.document_controller.event_loop)
        self.__exit_stack: typing.List[typing.Any] = list()

    def close(self) -> None:
        self.document_controller.periodic()
        self.document_controller.close()
        for ex in self.__exit_stack:
            ex.close()
        self.scan_context_controller.close()
        self.scan_context_controller = typing.cast(typing.Any, None)
        self.scan_hardware_source.close()
        self.camera_hardware_source.close()
        HardwareSource.HardwareSourceManager().unregister_hardware_source(self.camera_hardware_source)
        HardwareSource.HardwareSourceManager().unregister_hardware_source(self.scan_hardware_source)
        HardwareSource.HardwareSourceManager()._close_instruments()
        super().close()

    def push(self, ex) -> None:
        self.__exit_stack.append(ex)


class AcquisitionTestContextBehavior:

    def setup_stem_controller(self) -> stem_controller.STEMController:
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        Registry.register_component(instrument, {"stem_controller"})
        return instrument

    def setup_scan_hardware_source(self, stem_controller: stem_controller.STEMController) -> scan_base.ScanHardwareSource:
        instrument = typing.cast(InstrumentDevice.Instrument, stem_controller)
        scan_hardware_source = scan_base.ConcreteScanHardwareSource(stem_controller, ScanDevice.Device(instrument),
                                                                    "usim_scan_device", "uSim Scan")
        return scan_hardware_source

    def setup_camera_hardware_source(self, stem_controller: stem_controller.STEMController, camera_exposure: float, is_eels: bool) -> HardwareSource.HardwareSource:
        instrument = typing.cast(InstrumentDevice.Instrument, stem_controller)
        camera_id = "usim_ronchigram_camera" if not is_eels else "usim_eels_camera"
        camera_type = "ronchigram" if not is_eels else "eels"
        camera_name = "uSim Camera"
        camera_settings = CameraDevice.CameraSettings(camera_id)
        camera_device = CameraDevice.Camera(camera_id, camera_type, camera_name, instrument)
        camera_hardware_source = camera_base.CameraHardwareSource2("usim_stem_controller", camera_device, camera_settings,
                                                                   None, None)
        if is_eels:
            camera_hardware_source.features["is_eels_camera"] = True
            camera_hardware_source.add_channel_processor(0, HardwareSource.SumProcessor(Geometry.FloatRect(Geometry.FloatPoint(0.25, 0.0), Geometry.FloatSize(0.5, 1.0))))
        camera_hardware_source.set_frame_parameters(0, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(1, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(2, camera_base.CameraFrameParameters(
            {"exposure_ms": camera_exposure * 1000 * 2, "binning": 1}))
        camera_hardware_source.set_selected_profile_index(0)
        return camera_hardware_source


def test_context(*, is_eels: bool = False, camera_exposure: float = 0.025) -> AcquisitionTestContext:
    behavior = AcquisitionTestContextBehavior()
    instrument = behavior.setup_stem_controller()
    scan_hardware_source = behavior.setup_scan_hardware_source(instrument)
    camera_hardware_source = behavior.setup_camera_hardware_source(instrument, camera_exposure, is_eels)
    return AcquisitionTestContext(instrument, scan_hardware_source, camera_hardware_source)
