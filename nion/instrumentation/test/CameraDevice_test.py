import contextlib
import copy
import threading
import time
import typing
import unittest
import uuid

from nion.data import DataAndMetadata
from nion.device_kit import CameraDevice
from nion.device_kit import InstrumentDevice
from nion.instrumentation import camera_base
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.swift.test import TestContext
from nion.utils import Geometry


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


# a camera simulator which records the largest number of its get_frame_data calls in progress at once. each call takes
# 20 ms so that calls from two threads overlap.
class ConcurrencyRecordingCameraSimulator(AcquisitionTestContextConfiguration.CameraSimulator):

    def __init__(self) -> None:
        super().__init__(("", "eV"))
        self.__lock = threading.Lock()
        self.__active_call_count = 0
        self.max_active_call_count = 0

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, probe_position: Geometry.FloatPoint | None) -> DataAndMetadata.DataAndMetadata:
        with self.__lock:
            self.__active_call_count += 1
            self.max_active_call_count = max(self.max_active_call_count, self.__active_call_count)
        try:
            time.sleep(0.02)
            return super().get_frame_data(readout_area, binning_shape, exposure_s, scan_context, probe_position)
        finally:
            with self.__lock:
                self.__active_call_count -= 1


# a test configuration whose spectrometer has a simulator which records overlapping calls.
class ConcurrencyRecordingAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.eels_camera_device.close()
        self.simulator = ConcurrencyRecordingCameraSimulator()
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "eels", "EELS", self.simulator, self.instrument)


# a camera simulator which records the scan context of its latest frame. the camera thread writes it, and a test reads
# it only after the frame is delivered.
class ScanContextRecordingCameraSimulator(AcquisitionTestContextConfiguration.CameraSimulator):

    def __init__(self) -> None:
        super().__init__(("", "eV"))
        self.scan_context: stem_controller.ScanContext | None = None

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, probe_position: Geometry.FloatPoint | None) -> DataAndMetadata.DataAndMetadata:
        self.scan_context = copy.deepcopy(scan_context)
        return super().get_frame_data(readout_area, binning_shape, exposure_s, scan_context, probe_position)


# a test configuration whose spectrometer has a simulator which records the scan context of its latest frame.
class ScanContextRecordingAcquisitionTestContextConfiguration(AcquisitionTestContextConfiguration.AcquisitionTestContextConfiguration):

    def __init__(self) -> None:
        super().__init__()
        self.eels_camera_device.close()
        self.simulator = ScanContextRecordingCameraSimulator()
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "eels", "EELS", self.simulator, self.instrument)


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
            self.assertEqual(["eV"], [calibration.units for calibration in xdata.dimensional_calibrations])

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

    def test_sequence_acquired_while_the_view_plays_does_not_overlap_the_view_frames(self) -> None:
        configuration = ConcurrencyRecordingAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration, is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                hardware_source.get_next_xdatas_to_finish()
                hardware_source.acquire_sequence_prepare(10)
                hardware_source.acquire_sequence(10)
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)
            self.assertEqual(1, configuration.simulator.max_active_call_count, "the simulator was asked for frames by two threads at once")

    def test_sequence_acquired_just_after_stopping_the_view_does_not_overlap_the_view_frames(self) -> None:
        configuration = ConcurrencyRecordingAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration, is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing(sync_timeout=3.0)
            hardware_source.get_next_xdatas_to_finish()
            hardware_source.abort_playing(sync_timeout=3.0)
            hardware_source.acquire_sequence_prepare(10)
            hardware_source.acquire_sequence(10)
            self.assertEqual(1, configuration.simulator.max_active_call_count, "the simulator was asked for frames by two threads at once")

    def test_synchronized_acquisition_delivered_as_partial_data_succeeds_when_the_view_starts_during_it(self) -> None:
        with AcquisitionTestContext.test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            # acquire the spectrum image in parts of ten pixels, so that the view has a chance to acquire between parts.
            setattr(camera_hardware_source, "_update_period_for_testing", 0.05)
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(16, 16)
            # starting the view applies its frame parameters to the camera, so use frame parameters which the view
            # can share; a view has no processing.
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.exposure_ms = 5
            camera_frame_parameters.binning = 4
            camera_frame_parameters.processing = None
            results: list[typing.Any] = list()

            def grab_synchronized() -> None:
                results.append(scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters, camera=camera_hardware_source, camera_frame_parameters=camera_frame_parameters))

            synchronized_thread = threading.Thread(target=grab_synchronized)
            synchronized_thread.start()
            try:
                while synchronized_thread.is_alive() and not camera_hardware_source.camera._is_acquire_synchronized_running:
                    time.sleep(0.005)
                camera_hardware_source.start_playing()
            finally:
                synchronized_thread.join()
                camera_hardware_source.abort_playing(sync_timeout=3.0)
            self.assertEqual(1, len(results), "the synchronized acquisition failed")
            scans, spectrum_images = results[0]
            self.assertEqual((16, 16), spectrum_images[0].data_shape[:2])

    def test_view_playing_during_a_sequence_receives_the_sequence_frames(self) -> None:
        with AcquisitionTestContext.test_context(is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            # a view frame which waits for new data gives up after 200 exposures, here 4 s, longer than the 2 s sequence.
            frame_parameters = hardware_source.get_current_frame_parameters()
            frame_parameters.exposure_ms = 20
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                hardware_source.get_next_xdatas_to_finish()
                hardware_source.acquire_sequence_prepare(100)
                sequence_thread = threading.Thread(target=hardware_source.acquire_sequence, args=(100,))
                start = time.perf_counter()
                sequence_thread.start()
                try:
                    time.sleep(0.2)
                    hardware_source.get_next_xdatas_to_finish()
                    hardware_source.get_next_xdatas_to_finish()
                    elapsed = time.perf_counter() - start
                finally:
                    sequence_thread.join()
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)
            self.assertLess(elapsed, 1.0, "the view stalled until the sequence finished")

    def test_parked_probe_is_positioned_within_the_last_scan(self) -> None:
        configuration = ScanContextRecordingAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration, is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=16, w=16)
            frame_parameters.fov_nm = 50.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
            # the user changes the field of view with no view playing, so it isn't scanned, and parks the probe. the
            # probe is off center only as a user would park it; the result doesn't depend on its position.
            frame_parameters.fov_nm = 200.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            test_context.instrument.probe_position = Geometry.FloatPoint(y=0.25, x=0.25)
            test_context.camera_hardware_source.grab_next_to_start()
            scan_context = configuration.simulator.scan_context
            assert scan_context is not None
            self.assertEqual(50.0, scan_context.fov_nm, "the parked probe moved to frame parameters which were never scanned")


    def test_parked_probe_is_positioned_within_frame_parameters_changed_while_scanning(self) -> None:
        configuration = ScanContextRecordingAcquisitionTestContextConfiguration()
        with AcquisitionTestContext.test_context(configuration, is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=256, w=256)
            frame_parameters.pixel_time_us = 10.0
            frame_parameters.fov_nm = 50.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                # a frame takes about 0.65 s, so the change below is made while it scans.
                start_time = time.time()
                while not scan_hardware_source.scan_device.is_scanning:
                    self.assertLess(time.time() - start_time, 3.0)
                    time.sleep(0.01)
                frame_parameters.fov_nm = 200.0
                scan_hardware_source.set_current_frame_parameters(frame_parameters)
            finally:
                # stopping lets the frame being scanned finish without starting another.
                scan_hardware_source.stop_playing(sync_timeout=3.0)
            test_context.instrument.probe_position = Geometry.FloatPoint(y=0.25, x=0.25)
            test_context.camera_hardware_source.grab_next_to_start()
            scan_context = configuration.simulator.scan_context
            assert scan_context is not None
            self.assertEqual(200.0, scan_context.fov_nm, "the parked probe stayed within frame parameters which were replaced while scanning")

if __name__ == '__main__':
    unittest.main()
