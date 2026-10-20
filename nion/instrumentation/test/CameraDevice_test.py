import typing
import unittest

from nion.device_kit import CameraDevice
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.swift.test import TestContext
from nion.utils import Geometry


# a test configuration whose spectrometer is an EDXS detector, a camera type with no units of its own.
class EDXSAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.eels_camera_device.close()
        simulator = AcquisitionTestContextConfiguration.CameraSimulator(Geometry.IntSize(256, 1024), ("", "eV"))
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "edxs", "EDXS", simulator, self.instrument)


class TestCameraDeviceClass(unittest.TestCase):

    def setUp(self) -> None:
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def test_camera_data_has_the_units_of_its_simulator(self) -> None:
        with AcquisitionTestContext.test_context(EDXSAcquisitionTestContextConfiguration(), is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                xdata = hardware_source.get_next_xdatas_to_finish()[0]
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)
            self.assertEqual(["", "eV"], [calibration.units for calibration in xdata.dimensional_calibrations])


if __name__ == '__main__':
    unittest.main()
