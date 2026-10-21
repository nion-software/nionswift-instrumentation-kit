import contextlib
import typing
import unittest

from nion.device_kit import CameraDevice
from nion.device_kit import InstrumentDevice
from nion.instrumentation import camera_base
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.swift.test import TestContext


# a test configuration whose spectrometer is an EDXS detector, a camera type with no units of its own.
class EDXSAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.eels_camera_device.close()
        simulator = AcquisitionTestContextConfiguration.CameraSimulator(("", "eV"))
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "edxs", "EDXS", simulator, self.instrument)


# an instrument whose cameras all have a single row of 1024 pixels, like a spectrometer.
class OneRowInstrument(InstrumentDevice.Instrument):

    def camera_sensor_dimensions(self, camera_type: str) -> typing.Tuple[int, int]:
        return 1, 1024

    def camera_readout_area(self, camera_type: str) -> typing.Tuple[int, int, int, int]:
        return 0, 0, 1, 1024


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

    def test_binning_a_one_row_camera_bins_its_width_only(self) -> None:
        instrument = OneRowInstrument("test_stem_controller", AcquisitionTestContextConfiguration.ValueManager(), AcquisitionTestContextConfiguration.AxisManager())
        simulator = AcquisitionTestContextConfiguration.CameraSimulator()
        camera = CameraDevice.Camera("test_one_row_camera", "edxs", "EDXS", simulator, instrument)
        with contextlib.closing(camera):
            camera.set_frame_parameters(camera_base.CameraFrameParameters({"exposure_ms": 5, "binning": 2}))
            camera.start_live()
            try:
                data_element = camera.acquire_image()
            finally:
                camera.stop_live()
            self.assertEqual((512,), data_element["data"].shape)

    def test_binning_a_one_row_readout_area_of_a_two_dimensional_sensor_bins_its_width_only(self) -> None:
        instrument = InstrumentDevice.Instrument("test_stem_controller", AcquisitionTestContextConfiguration.ValueManager(), AcquisitionTestContextConfiguration.AxisManager())
        simulator = AcquisitionTestContextConfiguration.CameraSimulator()
        camera = CameraDevice.Camera("test_eels_camera", "eels", "EELS", simulator, instrument)
        with contextlib.closing(camera):
            camera.readout_area = (0, 0, 1, 1024)
            camera.set_frame_parameters(camera_base.CameraFrameParameters({"exposure_ms": 5, "binning": 2}))
            camera.start_live()
            try:
                data_element = camera.acquire_image()
            finally:
                camera.stop_live()
            self.assertEqual((512,), data_element["data"].shape)


if __name__ == '__main__':
    unittest.main()
