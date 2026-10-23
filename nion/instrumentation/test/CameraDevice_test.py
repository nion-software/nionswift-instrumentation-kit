import contextlib
import typing
import unittest

from nion.device_kit import CameraDevice
from nion.device_kit import InstrumentDevice
from nion.instrumentation import camera_base
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.swift.test import TestContext


# a test configuration whose spectrometer is an EDXS detector, a camera type with no units or signal type of its own,
# and with no counts per electron.
class EDXSAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.eels_camera_device.close()
        simulator = AcquisitionTestContextConfiguration.CameraSimulator(("", "eV"))
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "edxs", "EDXS", simulator, self.instrument, signal_type="edxs")


# an instrument whose cameras all have a single row of 1024 pixels, like a spectrometer.
class OneRowInstrument(InstrumentDevice.Instrument):

    def camera_sensor_dimensions(self, camera_type: str) -> typing.Tuple[int, int]:
        return 1, 1024

    def camera_readout_area(self, camera_type: str) -> typing.Tuple[int, int, int, int]:
        return 0, 0, 1, 1024


# a camera whose x and y calibrations have their origin at the center of the frame.
class CenteredOriginCamera(CameraDevice.Camera):

    @property
    def calibration_controls(self) -> typing.Mapping[str, str | int | float]:
        return dict(super().calibration_controls) | {"x_origin_override": "center", "y_origin_override": "center"}


# a test configuration whose ronchigram camera has its x and y origin at the center of the frame.
class CenteredOriginAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.ronchigram_camera_device.close()
        simulator = AcquisitionTestContextConfiguration.CameraSimulator()
        self.ronchigram_camera_device = CenteredOriginCamera(self.ronchigram_camera_device_id, "ronchigram", "Ronchigram", simulator, self.instrument)


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

    def test_camera_data_has_the_signal_type_of_its_camera(self) -> None:
        with AcquisitionTestContext.test_context(EDXSAcquisitionTestContextConfiguration(), is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                xdata = hardware_source.get_next_xdatas_to_finish()[0]
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)
            self.assertEqual("edxs", xdata.metadata["hardware_source"]["signal_type"])

    def test_camera_data_has_no_counts_per_electron_when_its_simulator_has_none(self) -> None:
        with AcquisitionTestContext.test_context(EDXSAcquisitionTestContextConfiguration(), is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                xdata = hardware_source.get_next_xdatas_to_finish()[0]
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)
            self.assertNotIn("counts_per_electron", xdata.metadata["hardware_source"])

    def test_binning_a_one_row_camera_bins_its_width_only(self) -> None:
        instrument = OneRowInstrument("test_stem_controller", AcquisitionTestContextConfiguration.ValueManager(), AcquisitionTestContextConfiguration.AxisManager())
        simulator = AcquisitionTestContextConfiguration.CameraSimulator()
        camera = CameraDevice.Camera("test_one_row_camera", "edxs", "EDXS", simulator, instrument)
        with contextlib.closing(camera):
            camera.set_frame_parameters(camera_base.CameraFrameParameters({"exposure_ms": 5, "binning": 2}))
            # a readout area one bin wide still gives 1D data, of one channel.
            for readout_area, expected_shape in (((0, 0, 1, 1024), (512,)), ((0, 0, 1, 2), (1,))):
                camera.readout_area = readout_area
                camera.start_live()
                try:
                    data_element = camera.acquire_image()
                finally:
                    camera.stop_live()
                self.assertEqual(expected_shape, data_element["data"].shape)

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

    def test_setting_a_readout_area_which_is_empty_or_not_within_the_sensor_raises_and_keeps_the_readout_area(self) -> None:
        with AcquisitionTestContext.test_context(is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.readout_area = (0, 0, 1, 1024)
            for readout_area in ((0, -100, 1, 412), (0, 512, 1, 1536), (256, 0, 257, 1024), (0, 0, 0, 1024), (0, 100, 1, 100)):
                with self.subTest(readout_area=readout_area):
                    with self.assertRaises(ValueError):
                        hardware_source.readout_area = readout_area
                    self.assertEqual((0, 0, 1, 1024), hardware_source.readout_area)

    def test_sum_project_sequence_of_a_camera_with_a_centered_origin_is_centered_on_the_frame_width(self) -> None:
        with AcquisitionTestContext.test_context(CenteredOriginAcquisitionTestContextConfiguration()) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.readout_area = (0, 0, 256, 1024)
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.binning = 1
            frame_parameters.processing = "sum_project"
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.acquire_sequence_prepare(3)
            data_element = hardware_source.acquire_sequence(3)[0]
            self.assertEqual((3, 1024), data_element["data"].shape)
            self.assertEqual(-512.0, data_element["spatial_calibrations"][-1]["offset"])

    def test_view_of_a_camera_with_a_centered_origin_is_centered_on_the_frame(self) -> None:
        with AcquisitionTestContext.test_context(CenteredOriginAcquisitionTestContextConfiguration()) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.readout_area = (0, 0, 256, 1024)
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.binning = 1
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                xdata = hardware_source.get_next_xdatas_to_finish()[0]
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)
            self.assertEqual((256, 1024), xdata.data_shape)
            self.assertEqual(-128.0, xdata.dimensional_calibrations[0].offset)
            self.assertEqual(-512.0, xdata.dimensional_calibrations[1].offset)


if __name__ == '__main__':
    unittest.main()
