import typing
import unittest

from nion.instrumentation import HardwareSource
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.swift import Application
from nion.swift.model import Schema
from nion.swift.test import TestContext
from nion.ui import TestUI
from nion.utils import Model
from nion.utils import Stream
from nionswift_plugin.nion_instrumentation_ui import AcquisitionPanel


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
