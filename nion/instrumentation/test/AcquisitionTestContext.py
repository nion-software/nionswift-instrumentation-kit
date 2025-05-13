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

    def run(self) -> None:
        """Register the instrument and cameras with the registry.

        The instrument should be registered with the type "instrument_controller" and "stem_controller".

        The cameras should be registered with the type "camera_module".

        The scan devices should be registered with the type "scan_module".

        The camera device objects themselves should already be registered with the type "{camera_id}_device" when
        this method is called. That would typically be done when this object is constructed.
        """
        ...

    def stop(self) -> None:
        """Unregister the instrument and cameras from the registry.

        The camera modules should be unregistered with the type "camera_module".

        The scan devices should be unregistered with the type "scan_module".

        The instrument should be unregistered with the type "instrument_controller" and "stem_controller".

        The camera device objects should be unregistered in their close method, which will be called when the module
        is unloaded. The camera device objects should be registered with the type "{camera_id}_device" when this
        method is called. That would typically be done when this object is constructed.
        """
        ...


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
