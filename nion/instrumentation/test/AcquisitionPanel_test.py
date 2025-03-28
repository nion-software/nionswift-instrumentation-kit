import typing
import unittest

from nion.instrumentation.test import AcquisitionTestContext
from nion.swift import Application
from nion.swift.model import Schema
from nion.swift.test import TestContext
from nion.ui import TestUI
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
