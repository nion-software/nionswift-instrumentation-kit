import copy
import datetime
import math
import numpy
import threading
import time
import typing
import unittest
import uuid

from nion.data import DataAndMetadata
from nion.data import Calibration
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import ApplicationData
from nion.swift.model import Metadata
from nion.ui import TestUI
from nion.utils import Geometry
from nion.instrumentation import Acquisition
from nion.instrumentation import camera_base
from nion.instrumentation import DataChannel
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller as STEMController
from nion.instrumentation.test import AcquisitionTestContext
from nionswift_plugin.nion_instrumentation_ui import ScanAcquisition


class TestDriftTrackerClass(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)

    def __test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.test_context(is_eels=is_eels)

    def _acquire_one(self, document_controller, hardware_source):
        hardware_source.start_playing(sync_timeout=3.0)
        hardware_source.stop_playing(sync_timeout=3.0)
        document_controller.periodic()

    def test_drift_corrector_basic_with_correction(self):
        # for this test, drift should be in a constant direction
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            test_context.instrument.sample_index = 2  # use CTS sample, custom position chosen using view mode in Swift
            drift_tracker = scan_hardware_source.drift_tracker
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.drift_channel_id = scan_hardware_source.data_channels[0].channel_id
            scan_hardware_source.drift_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(0.6554, 0.2932), Geometry.FloatSize(0.15, 0.15))
            document_controller.periodic()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            stem_controller = scan_hardware_source.stem_controller
            axis = None
            for axis in stem_controller.axis_descriptions:
                if axis.axis_id == "scan":
                    break
            assert axis is not None
            drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters, axis=axis)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.width)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.height)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertLess(dist_nm, 0.1)
            stem_controller.SetValDeltaAndConfirm("CSH.x", 1e-9, 1.0, 1000)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertAlmostEqual(dist_nm, 1.0, delta=0.3)
            self.assertAlmostEqual(abs(last_delta_nm.width), 1.0, delta=0.3)
            self.assertLess(abs(last_delta_nm.height), 0.1)
            stem_controller.SetValDeltaAndConfirm("CSH.x", 2e-9, 2.0, 1000)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertAlmostEqual(dist_nm, 2.0, delta=0.3)
            self.assertAlmostEqual(abs(last_delta_nm.width), 2.0, delta=0.3)
            self.assertLess(abs(last_delta_nm.height), 0.1)
            total_delta_nm = drift_tracker.total_delta_nm
            self.assertAlmostEqual(abs(total_delta_nm.width), 3.0, delta=0.3)
            self.assertLess(abs(last_delta_nm.height), 0.1)
            expected_drift_data_frame = [[0.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0]]
            self.assertTrue(numpy.allclose(drift_tracker.drift_data_frame[:-1], expected_drift_data_frame, atol=0.2))

    def test_drift_corrector_basic_without_correction(self):
        # for this test, drift should be in a constant direction
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            test_context.instrument.sample_index = 2  # use CTS sample, custom position chosen using view mode in Swift
            drift_tracker = scan_hardware_source.drift_tracker
            self._acquire_one(document_controller, scan_hardware_source)
            # scan_hardware_source.drift_channel_id = scan_hardware_source.data_channels[0].channel_id
            # scan_hardware_source.drift_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(0.6554, 0.2932), Geometry.FloatSize(0.15, 0.15))
            document_controller.periodic()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            # drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.width)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.height)
            xdata_list = scan_hardware_source.record_immediate(scan_frame_parameters)
            self.assertTrue(xdata_list)
            self.assertIsNotNone(xdata_list[0])
            starttime = datetime.datetime.utcnow()
            xdata_list[0].timestamp = starttime
            drift_tracker.submit_image(xdata_list[0], 0.0, drift_data_source_id="test_drift_corrector", axis="ScanAxis", drift_correction_applied=False, wait=True)
            # drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertLess(dist_nm, 0.1)
            stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id("usim_stem_controller")
            stem_controller.SetValDeltaAndConfirm("CSH.x", 1e-9, 1.0, 1000)
            xdata_list = scan_hardware_source.record_immediate(scan_frame_parameters)
            self.assertTrue(xdata_list)
            self.assertIsNotNone(xdata_list[0])
            xdata_list[0].timestamp = starttime + datetime.timedelta(seconds=1.0)
            drift_tracker.submit_image(xdata_list[0], 0.0, drift_data_source_id="test_drift_corrector", axis="ScanAxis", drift_correction_applied=False, wait=True)
            # drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            # print(last_delta_nm)
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertAlmostEqual(dist_nm, 1.0, delta=0.2)
            self.assertAlmostEqual(abs(last_delta_nm.width), 1.0, delta=0.2)
            self.assertLess(abs(last_delta_nm.height), 0.1)
            stem_controller.SetValDeltaAndConfirm("CSH.x", 2e-9, 1.0, 1000)
            xdata_list = scan_hardware_source.record_immediate(scan_frame_parameters)
            self.assertTrue(xdata_list)
            self.assertIsNotNone(xdata_list[0])
            xdata_list[0].timestamp = starttime + datetime.timedelta(seconds=3.0)
            drift_tracker.submit_image(xdata_list[0], 0.0, drift_data_source_id="test_drift_corrector", axis="ScanAxis", drift_correction_applied=False, wait=True)
            # drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            # print(last_delta_nm)
            # print(drift_tracker._DriftTracker__drift_history)
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertAlmostEqual(dist_nm, 2.0, delta=0.2)
            self.assertAlmostEqual(abs(last_delta_nm.width), 2.0, delta=0.2)
            self.assertLess(abs(last_delta_nm.height), 0.1)
            total_delta_nm = drift_tracker.total_delta_nm
            self.assertAlmostEqual(abs(total_delta_nm.width), 3.0, delta=0.2)
            self.assertLess(abs(last_delta_nm.height), 0.1)
            expected_drift_data_frame = [[0.0, 0.0, 0.0], [0.0, 1.0, 3.0], [0.0, 1.0, 3.0]]
            self.assertTrue(numpy.allclose(drift_tracker.drift_data_frame[:-1], expected_drift_data_frame, atol=0.2))

    def test_drift_corrector_constant_drift_rate(self):
        drift_rate = Geometry.FloatSize(height=2.0, width=3.0) # in nm / s
        n_points = 5
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            test_context.instrument.sample_index = 2  # use CTS sample, custom position chosen using view mode in Swift
            drift_tracker = scan_hardware_source.drift_tracker
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            # print(scan_frame_parameters)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.width)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.height)
            starttime = datetime.datetime.utcnow()
            xdata_list = scan_hardware_source.record_immediate(scan_frame_parameters)
            self.assertTrue(xdata_list)
            self.assertIsNotNone(xdata_list[0])
            acquisition_time = (xdata_list[0].timestamp - starttime).total_seconds()
            # print(f"Acquisition time: {acquisition_time}")
            xdata_list[0].timestamp = starttime
            drift_tracker.submit_image(xdata_list[0], 0.0, drift_data_source_id="test_drift_corrector", axis="ScanAxis", drift_correction_applied=False, wait=True)
            # numpy.save(f"C:/Users/Andi/Downloads/image_{xdata_list[0].timestamp.timestamp()}.npy", xdata_list[0].data)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertLess(dist_nm, 0.1)
            stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id("usim_stem_controller")
            for i in range(n_points):
                now = datetime.datetime.utcnow()
                acquisition_time = (xdata_list[0].timestamp - starttime).total_seconds()
                with self.subTest(point=i, now=now, starttime=starttime, time_delta=(now - starttime).total_seconds()):
                    expected_delta_nm = drift_rate * (now - starttime).total_seconds()
                    stem_controller.SetValAndConfirm("CSH.x", expected_delta_nm.width * 1e-9, 1.0, 1000)
                    stem_controller.SetValAndConfirm("CSH.y", expected_delta_nm.height * 1e-9, 1.0, 1000)
                    xdata_list = scan_hardware_source.record_immediate(scan_frame_parameters)
                    # print(f"Acquisition time: {(datetime.datetime.utcnow() - now).total_seconds()}")
                    self.assertTrue(xdata_list)
                    self.assertIsNotNone(xdata_list[0])
                    xdata_list[0].timestamp = now
                    drift_tracker.submit_image(xdata_list[0], 0.0, drift_data_source_id="test_drift_corrector", axis="ScanAxis", drift_correction_applied=False, wait=True)
                    # numpy.save(f"C:/Users/Andi/Downloads/image_{xdata_list[0].timestamp.timestamp()}.npy", xdata_list[0].data)
                    last_delta_nm = drift_tracker.total_delta_nm
                    # print(f"{last_delta_nm=}\n{expected_delta_nm=}")
                    dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
                    # expected_delta_nm = drift_rate * (now - starttime)
                    expected_dist_nm = numpy.sqrt(numpy.sum(numpy.power(expected_delta_nm, 2)))
                    self.assertAlmostEqual(dist_nm, expected_dist_nm, delta=0.8)
                    self.assertAlmostEqual(last_delta_nm.width, expected_delta_nm.width, delta=0.8)
                    self.assertAlmostEqual(last_delta_nm.height, expected_delta_nm.height, delta=0.8)
                    measured_drift_rate = drift_tracker.get_drift_rate() * 1e9
                    self.assertAlmostEqual(measured_drift_rate.width, drift_rate.width, delta=0.8)
                    self.assertAlmostEqual(measured_drift_rate.height, drift_rate.height, delta=0.8)

    def test_drift_corrector_constant_drift_rate_multiple_sources(self):
        drift_rate = Geometry.FloatSize(height=.8, width=1.3) # in nm / s
        n_points = 3
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id("usim_stem_controller")
            test_context.instrument.sample_index = 2  # use CTS sample, custom position chosen using view mode in Swift
            drift_tracker = scan_hardware_source.drift_tracker
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            # It seems that the rotation in the simulator goes in the wrong direction: If we use the same rotation here
            # as we pass to "submit_image", the test fails. In DriftCorrector we rotate the measured drift by the negative
            # rotation passed to "submit_image" which should be the correct thing to do. It needs to be tested on real
            # hardware to see if it is functioning correctly.
            scan_frame_parameters.rotation_rad = -0.5 # ~28.6 deg
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            self.assertEqual(0.0, drift_tracker.last_delta_nm.width)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.height)
            starttime = datetime.datetime.utcnow()
            # Acquire a camera image and submit it to the drift tracker
            for _ in range(2):
                xdata_list = camera_hardware_source.grab_next_to_start()
            self.assertTrue(xdata_list)
            self.assertIsNotNone(xdata_list[0])
            camera_xdata = xdata_list[0]
            camera_xdata.timestamp = starttime
            defocus = stem_controller.GetVal("C10") * 1e9
            # We need to update the calibration from rad to nm
            old_calibration = camera_xdata.dimensional_calibrations[0]
            new_calibration = Calibration.Calibration(old_calibration.offset * defocus, old_calibration.scale * defocus, "nm")
            camera_xdata._set_dimensional_calibrations([new_calibration, new_calibration])
            drift_tracker.submit_image(camera_xdata, 0.0, drift_data_source_id="test_drift_corrector_camera", axis="TV", drift_correction_applied=False, wait=True)
            # numpy.save(f"C:/Users/Andi/Downloads/image_{xdata_list[0].timestamp.timestamp()}.npy", xdata_list[0].data)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertLess(dist_nm, 0.1)

            for i in range(n_points):
                now = datetime.datetime.utcnow()
                acquisition_time = (xdata_list[0].timestamp - starttime).total_seconds()
                with self.subTest(camera=True, point=i, now=now, starttime=starttime, time_delta=(now - starttime).total_seconds()):
                    expected_delta_nm = drift_rate * (now - starttime).total_seconds()
                    stem_controller.SetValAndConfirm("CSH.x", expected_delta_nm.width * 1e-9, 1.0, 1000)
                    stem_controller.SetValAndConfirm("CSH.y", expected_delta_nm.height * 1e-9, 1.0, 1000)
                    for _ in range(2):
                        xdata_list = camera_hardware_source.grab_next_to_start()
                    # print(f"Acquisition time: {(datetime.datetime.utcnow() - now).total_seconds()}")
                    camera_xdata = xdata_list[0]
                    camera_xdata.timestamp = now
                    defocus = stem_controller.GetVal("C10") * 1e9
                    # We need to update the calibration from rad to nm
                    old_calibration = camera_xdata.dimensional_calibrations[0]
                    new_calibration = Calibration.Calibration(old_calibration.offset * defocus, old_calibration.scale * defocus, "nm")
                    camera_xdata._set_dimensional_calibrations([new_calibration, new_calibration])
                    drift_tracker.submit_image(camera_xdata, 0.0, drift_data_source_id="test_drift_corrector_camera", axis="TV", drift_correction_applied=False, wait=True)
                    # numpy.save(f"C:/Users/Andi/Downloads/image_{xdata_list[0].timestamp.timestamp()}.npy", xdata_list[0].data)
                    last_delta_nm = drift_tracker.total_delta_nm
                    # print(f"Camera: {last_delta_nm=}\n{expected_delta_nm=}")
                    dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
                    # expected_delta_nm = drift_rate * (now - starttime)
                    expected_dist_nm = numpy.sqrt(numpy.sum(numpy.power(expected_delta_nm, 2)))
                    self.assertAlmostEqual(dist_nm, expected_dist_nm, delta=0.8)
                    self.assertAlmostEqual(last_delta_nm.width, expected_delta_nm.width, delta=0.8)
                    self.assertAlmostEqual(last_delta_nm.height, expected_delta_nm.height, delta=0.8)
                    measured_drift_rate = drift_tracker.get_drift_rate() * 1e9
                    self.assertAlmostEqual(measured_drift_rate.width, drift_rate.width, delta=0.8)
                    self.assertAlmostEqual(measured_drift_rate.height, drift_rate.height, delta=0.8)

            # Now submit some scan data
            xdata_list = scan_hardware_source.record_immediate(scan_frame_parameters)
            self.assertTrue(xdata_list)
            self.assertIsNotNone(xdata_list[0])
            acquisition_time = (xdata_list[0].timestamp - starttime).total_seconds()
            # print(f"Acquisition time: {acquisition_time}")
            xdata_list[0].timestamp = now
            drift_tracker.submit_image(xdata_list[0], 0.5, drift_data_source_id="test_drift_corrector_scan", axis="ScanAxis", drift_correction_applied=False, wait=True)
            # numpy.save(f"C:/Users/Andi/Downloads/image_{xdata_list[0].timestamp.timestamp()}.npy", xdata_list[0].data)

            for i in range(n_points):
                now = datetime.datetime.utcnow()
                acquisition_time = (xdata_list[0].timestamp - starttime).total_seconds()
                with self.subTest(scan=True, point=i, now=now, starttime=starttime, time_delta=(now - starttime).total_seconds()):
                    expected_delta_nm = drift_rate * ((now - starttime).total_seconds())
                    stem_controller.SetValAndConfirm("CSH.x", expected_delta_nm.width * 1e-9, 1.0, 1000)
                    stem_controller.SetValAndConfirm("CSH.y", expected_delta_nm.height * 1e-9, 1.0, 1000)
                    xdata_list = scan_hardware_source.record_immediate(scan_frame_parameters)
                    # print(f"Acquisition time: {(datetime.datetime.utcnow() - now).total_seconds()}")
                    self.assertTrue(xdata_list)
                    self.assertIsNotNone(xdata_list[0])
                    xdata_list[0].timestamp = now
                    drift_tracker.submit_image(xdata_list[0], 0.5, drift_data_source_id="test_drift_corrector_scan", axis="ScanAxis", drift_correction_applied=False, wait=True)
                    # numpy.save(f"C:/Users/Andi/Downloads/image_{xdata_list[0].timestamp.timestamp()}.npy", xdata_list[0].data)
                    last_delta_nm = drift_tracker.total_delta_nm
                    # print(f"Scan: {last_delta_nm=}\n{expected_delta_nm=}")
                    dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
                    # expected_delta_nm = drift_rate * (now - starttime)
                    expected_dist_nm = numpy.sqrt(numpy.sum(numpy.power(expected_delta_nm, 2)))
                    self.assertAlmostEqual(dist_nm, expected_dist_nm, delta=0.8)
                    self.assertAlmostEqual(last_delta_nm.width, expected_delta_nm.width, delta=0.8)
                    self.assertAlmostEqual(last_delta_nm.height, expected_delta_nm.height, delta=0.8)
                    measured_drift_rate = drift_tracker.get_drift_rate() * 1e9
                    self.assertAlmostEqual(measured_drift_rate.width, drift_rate.width, delta=0.8)
                    self.assertAlmostEqual(measured_drift_rate.height, drift_rate.height, delta=0.8)