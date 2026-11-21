import contextlib
import typing
import unittest

from nion.device_kit import CameraDevice
from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import HardwareSource
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.swift import Application
from nion.swift.model import ApplicationData
from nion.swift.model import Schema
from nion.swift.test import TestContext
from nion.ui import Declarative
from nion.ui import TestUI
from nion.utils import Binding
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Registry
from nion.utils import Stream
from nionswift_plugin.nion_instrumentation_ui import AcquisitionPanel


# a test configuration whose spectrometer is an EDXS detector.
class EDXSAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.eels_camera_device.close()
        simulator = AcquisitionTestContextConfiguration.CameraSimulator(("", "eV"))
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "edxs", "EDXS", simulator, self.instrument)


# a test configuration whose cameras are offline at launch. a test brings them online with register_cameras.
class NoCameraAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.__are_cameras_registered = False

    def run(self) -> None:
        Registry.register_component(self.instrument, {"instrument_controller", "stem_controller"})
        Registry.register_component(self.scan_module, {"scan_module"})

    def register_cameras(self) -> None:
        Registry.register_component(CameraDevice.CameraModule(self.instrument_id, self.ronchigram_camera_device, self.ronchigram_camera_settings), {"camera_module"})
        Registry.register_component(CameraDevice.CameraModule(self.instrument_id, self.eels_camera_device, self.eels_camera_settings), {"camera_module"})
        self.__are_cameras_registered = True

    def stop(self) -> None:
        super().stop()
        # unregistering a camera module closes its device, so only devices which were never registered need closing.
        if not self.__are_cameras_registered:
            self.ronchigram_camera_device.close()
            self.eels_camera_device.close()


class TestCameraControlClass(unittest.TestCase):

    def setUp(self) -> None:
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()
        self.exposure = 0.04

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def __test_context(self, is_eels: bool=False, is_both_cameras: bool=False):
        return AcquisitionTestContext.test_context(is_eels=is_eels, camera_exposure=self.exposure, is_both_cameras=is_both_cameras)

    def test_camera_channel_updated_when_camera_choice_changed(self) -> None:
        # this bug showed up when launching with acquisition panel sequence mode in camera mode and camera set to
        # Ronchigram, then switching to EELS camera. The camera channel was not updated to EELS Spectrum.
        with self.__test_context(is_both_cameras=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            c = Schema.Entity(Schema.get_entity_type("acquisition_device_component_camera"))
            c.camera_device_id = hardware_source.hardware_source_id

            # define a function for the camera settings scope; it will be de-ref'd when the scope exits.
            def camera_settings_scope() -> None:
                camera_settings = AcquisitionPanel.CameraSettingsModel(c)
                self.assertEqual("ronchigram", camera_settings.camera_channel)
                # simulate switching the hardware source to EELS via the combo box.
                # setting channel_index to None will be done by the combo box as part of switching its list of items,
                # Since it isn't connected, do it here.
                self.assertEqual("ronchigram", camera_settings.camera_channel)
                camera_settings.hardware_source_choice_model.hardware_source_choice.hardware_source_index_model.value = 1
                camera_settings.channel_index = None
                # check result
                self.assertEqual("ronchigram", camera_settings.camera_channel)

            camera_settings_scope()

    def test_eels_size_estimate_is_one_row_for_spectra_and_the_whole_frame_for_images(self) -> None:
        with self.__test_context(is_eels=True) as test_context:
            for camera_channel_id, expected_byte_dimensions in (("eels_spectrum", (1024, 4)), ("eels_image", (256, 1024, 4))):
                with self.subTest(camera_channel_id=camera_channel_id):
                    c = Schema.Entity(Schema.get_entity_type("acquisition_device_component_camera"))
                    c.camera_device_id = test_context.camera_hardware_source.hardware_source_id
                    c.camera_channel_id = camera_channel_id

                    # define a function for the camera settings scope; it will be de-ref'd when the scope exits.
                    def camera_settings_scope() -> None:
                        camera_settings = AcquisitionPanel.CameraSettingsModel(c)
                        self.assertEqual(expected_byte_dimensions, camera_settings.get_byte_dimensions((256, 1024)))

                    camera_settings_scope()

    def test_edxs_camera_offers_only_spectra(self) -> None:
        with AcquisitionTestContext.test_context(EDXSAcquisitionTestContextConfiguration(), is_eels=True) as test_context:
            c = Schema.Entity(Schema.get_entity_type("acquisition_device_component_camera"))
            c.camera_device_id = test_context.camera_hardware_source.hardware_source_id
            # an acquisition panel saved by an earlier release has the image channel selected for an EDXS camera.
            c.camera_channel_id = "image"

            # define a function for the camera settings scope; it will be de-ref'd when the scope exits.
            def camera_settings_scope() -> None:
                camera_settings = AcquisitionPanel.CameraSettingsModel(c)
                self.assertEqual(["edxs_spectrum"], [channel_description.channel_id for channel_description in camera_settings.channel_descriptions])
                self.assertEqual("edxs_spectrum", camera_settings.camera_channel)

            camera_settings_scope()

    def test_acquisition_panel_opens_without_camera(self) -> None:
        with AcquisitionTestContext.test_context(NoCameraAcquisitionTestContextConfiguration()) as test_context:
            document_controller = test_context.document_controller
            for device_component_id in ("camera", "synchronized-scan"):
                for method_component_id in ("basic-acquire", "sequence-acquire", "series-acquire", "tableau-acquire", "multiple-acquire"):
                    with self.subTest(device_component_id=device_component_id, method_component_id=method_component_id):
                        acquisition_configuration = AcquisitionPanel.AcquisitionConfiguration(ApplicationData.ApplicationData())
                        acquisition_configuration.acquisition_device_component_id = device_component_id
                        acquisition_configuration.acquisition_method_component_id = method_component_id
                        acquisition_preferences = AcquisitionPreferences.AcquisitionPreferences(ApplicationData.ApplicationData())
                        acquisition_controller = AcquisitionPanel.AcquisitionController(document_controller, acquisition_configuration, acquisition_preferences)
                        # the widget closes the acquisition controller when it closes.
                        widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, acquisition_controller)
                        with contextlib.closing(widget):
                            document_controller.periodic()
                            self.assertIsNone(acquisition_controller.time_usage_model.value)

    def test_acquisition_panel_estimates_time_once_a_camera_comes_online(self) -> None:
        configuration = NoCameraAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration) as test_context:
            acquisition_configuration = AcquisitionPanel.AcquisitionConfiguration(ApplicationData.ApplicationData())
            acquisition_configuration.acquisition_device_component_id = "camera"
            acquisition_preferences = AcquisitionPreferences.AcquisitionPreferences(ApplicationData.ApplicationData())
            acquisition_controller = AcquisitionPanel.AcquisitionController(test_context.document_controller, acquisition_configuration, acquisition_preferences)
            with contextlib.closing(acquisition_controller):
                configuration.register_cameras()
                self.assertIsNotNone(acquisition_controller.time_usage_model.value)

    def test_acquire_is_enabled_once_a_camera_comes_online(self) -> None:
        configuration = NoCameraAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration) as test_context:
            # a synchronized scan also needs a scan context.
            test_context.scan_hardware_source.scan_context.fov_nm = 100.0
            test_context.scan_hardware_source.scan_context.size = Geometry.IntSize(100, 100)
            test_context.scan_hardware_source.scan_context.rotation_rad = 0.0
            test_context.scan_hardware_source.scan_context.center_nm = Geometry.IntPoint(0, 0)
            acquisition_controllers = list()
            for device_component_id in ("camera", "synchronized-scan"):
                acquisition_configuration = AcquisitionPanel.AcquisitionConfiguration(ApplicationData.ApplicationData())
                acquisition_configuration.acquisition_device_component_id = device_component_id
                acquisition_preferences = AcquisitionPreferences.AcquisitionPreferences(ApplicationData.ApplicationData())
                acquisition_controllers.append(AcquisitionPanel.AcquisitionController(test_context.document_controller, acquisition_configuration, acquisition_preferences))
            try:
                self.assertEqual([False, False], [acquisition_controller.button_enabled_model.value for acquisition_controller in acquisition_controllers])
                configuration.register_cameras()
                self.assertEqual([True, True], [acquisition_controller.button_enabled_model.value for acquisition_controller in acquisition_controllers])
            finally:
                for acquisition_controller in acquisition_controllers:
                    acquisition_controller.close()

    def test_acquisition_panel_opens_with_acquire_disabled_in_scan_modes_without_scan_device(self) -> None:
        configuration = AcquisitionTestContextConfiguration.NoScanDeviceAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration) as test_context:
            document_controller = test_context.document_controller
            for device_component_id in ("scan", "synchronized-scan"):
                with self.subTest(device_component_id=device_component_id):
                    acquisition_configuration = AcquisitionPanel.AcquisitionConfiguration(ApplicationData.ApplicationData())
                    acquisition_configuration.acquisition_device_component_id = device_component_id
                    acquisition_preferences = AcquisitionPreferences.AcquisitionPreferences(ApplicationData.ApplicationData())
                    acquisition_controller = AcquisitionPanel.AcquisitionController(document_controller, acquisition_configuration, acquisition_preferences)
                    # the widget closes the acquisition controller when it closes.
                    widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, acquisition_controller)
                    with contextlib.closing(widget):
                        document_controller.periodic()
                        self.assertFalse(acquisition_controller.button_enabled_model.value)

    def test_saved_camera_choice_is_kept_once_the_last_camera_unregisters(self) -> None:
        with self.__test_context() as test_context:
            camera_device_id = test_context.camera_hardware_source.hardware_source_id
            c = Schema.Entity(Schema.get_entity_type("acquisition_device_component_camera"))
            c.camera_device_id = camera_device_id

            # define a function for the camera settings scope; it will be de-ref'd when the scope exits.
            def camera_settings_scope() -> None:
                camera_settings = AcquisitionPanel.CameraSettingsModel(c)
                # unregister the chosen camera last, since the choice moves to another camera while one remains.
                camera_modules = sorted(Registry.get_components_by_type("camera_module"), key=lambda camera_module: camera_module.camera_device.camera_id == camera_device_id)
                for camera_module in camera_modules:
                    Registry.unregister_component(camera_module, {"camera_module"})
                self.assertEqual(camera_device_id, c.camera_device_id)

            camera_settings_scope()

    def test_camera_choice_shows_no_camera_once_the_last_camera_unregisters(self) -> None:
        with self.__test_context() as test_context:
            c = Schema.Entity(Schema.get_entity_type("acquisition_device_component_camera"))
            c.camera_device_id = test_context.camera_hardware_source.hardware_source_id

            # define a function for the camera settings scope; it will be de-ref'd when the scope exits.
            def camera_settings_scope() -> None:
                camera_settings = AcquisitionPanel.CameraSettingsModel(c)
                # bind the hardware source index the way the camera combo box does.
                index_binding = Binding.PropertyBinding(camera_settings.hardware_source_choice_model.hardware_source_choice.hardware_source_index_model, "value")
                combo_box_indexes: list[int | None] = list()
                index_binding.target_setter = combo_box_indexes.append
                for camera_module in list(Registry.get_components_by_type("camera_module")):
                    Registry.unregister_component(camera_module, {"camera_module"})
                index_binding.close()
                self.assertIsNone(combo_box_indexes[-1])

            camera_settings_scope()

    def test_saved_camera_channel_is_kept_while_no_camera_is_available(self) -> None:
        configuration = NoCameraAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration):
            # an earlier launch saved the EELS image channel.
            c = Schema.Entity(Schema.get_entity_type("acquisition_device_component_camera"))
            c.camera_device_id = configuration.eels_camera_device_id
            c.camera_channel_id = "eels_image"

            # define a function for the camera settings scope; it will be de-ref'd when the scope exits.
            def camera_settings_scope() -> None:
                camera_settings = AcquisitionPanel.CameraSettingsModel(c)
                # the combo box sets channel_index to None as part of switching its list of items. since it isn't
                # connected, do it here.
                camera_settings.channel_index = None
                self.assertEqual("eels_image", c.camera_channel_id)

            camera_settings_scope()

    def test_scan_drift_interval_without_camera_follows_scan_pixel_time(self) -> None:
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            # a pixel time edit in the scan panel changes the current frame parameters.
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_time_us = 4.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            scan_hardware_source.drift_enabled = True
            scan_hardware_source.drift_settings = stem_controller.DriftCorrectionSettings(1, stem_controller.DriftIntervalUnit.TIME)
            # without a camera, the scan description estimates the drift interval from the scan's pixel time.
            camera_hardware_source_stream = Stream.ValueStream[HardwareSource.HardwareSource]().add_ref()
            scan_hardware_source_stream = Stream.ValueStream[HardwareSource.HardwareSource](scan_hardware_source).add_ref()
            scan_specifier_value_stream = AcquisitionPanel.ScanSpecifierValueStream(camera_hardware_source_stream, scan_hardware_source_stream, Model.PropertyModel[int](32), test_context.document_controller.event_loop).add_ref()
            try:
                scan_specifier = scan_specifier_value_stream.value
                assert scan_specifier
                # a drift correction every second is one every 7813 lines of 32 pixels at 4 us per pixel.
                self.assertEqual(7813, scan_specifier.drift_interval_lines)
            finally:
                scan_specifier_value_stream.remove_ref()
                scan_hardware_source_stream.remove_ref()
                camera_hardware_source_stream.remove_ref()
