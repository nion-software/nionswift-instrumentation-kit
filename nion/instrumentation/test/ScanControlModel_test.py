from __future__ import annotations

import time
import types
import typing
import unittest

from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base as ScanBase
from nion.instrumentation import stem_controller as STEMController
from nion.instrumentation.test import AcquisitionTestContext
from nion.swift import DocumentController
from nion.swift.test import TestContext
from nion.utils import Geometry
from nion.utils import Observable
from nion.utils import ReferenceCounting
from nionswift_plugin.nion_instrumentation_ui import ScanControlPanel


class PropertyChangedEventWatcher:
    def __init__(self, model: Observable.ObservableLike, property_name: str) -> None:
        self.listener = model.property_changed_event.listen(ReferenceCounting.weak_partial(PropertyChangedEventWatcher.__handle_property_changed, self))
        self.property_name = property_name
        self.changed = False

    def __enter__(self) -> PropertyChangedEventWatcher:
        return self

    def __exit__(self, exception_type: typing.Optional[typing.Type[BaseException]], value: typing.Optional[BaseException], traceback: typing.Optional[types.TracebackType]) -> typing.Optional[bool]:
        self.listener = None
        return None

    def __handle_property_changed(self, property_name: str) -> None:
        if property_name == self.property_name:
            self.changed = True


class TestScanControlClass(unittest.TestCase):

    def setUp(self):
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def _test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        # subclasses may override this to provide a different configuration
        return AcquisitionTestContext.test_context()

    def _acquire_one(self, document_controller: DocumentController.DocumentController, hardware_source: HardwareSource.HardwareSource) -> None:
        hardware_source.start_playing(sync_timeout=5.0)
        hardware_source.stop_playing(sync_timeout=5.0)
        document_controller.periodic()

    def test_profile_index(self):
        # test changing profile index updates frame parameters, only checking fov here
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)
            scan_settings = scan_hardware_source.scan_settings
            profile_frame_parameters_list = [scan_settings.get_frame_parameters(i) for i in range(2)]
            self.assertNotAlmostEqual(profile_frame_parameters_list[0].fov_nm, profile_frame_parameters_list[1].fov_nm)
            self.assertEqual(model.profile_index, 0)
            self.assertAlmostEqual(profile_frame_parameters_list[0].fov_nm, float(model.fov_str))
            model.profile_index = 1
            self.assertEqual(model.profile_index, 1)
            self.assertAlmostEqual(profile_frame_parameters_list[1].fov_nm, float(model.fov_str))
            model.profile_index = 0
            self.assertEqual(model.profile_index, 0)
            self.assertAlmostEqual(profile_frame_parameters_list[0].fov_nm, float(model.fov_str))
            # test that property changed event is emitted
            with PropertyChangedEventWatcher(model, "profile_index") as watcher:
                model.profile_index = 1
                self.assertTrue(watcher.changed)
            # test changing settings updates model
            with PropertyChangedEventWatcher(model, "profile_index") as watcher:
                scan_settings.set_selected_profile_index(0)
                self.assertTrue(watcher.changed)

    def test_changing_frame_parameters(self):
        # test changing the frame parameters fields; ensure other profile frame parameters are not changed

        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)
            scan_settings = scan_hardware_source.scan_settings

            # test changing pixel time
            profile_frame_parameters_list = [scan_settings.get_frame_parameters(i) for i in range(2)]
            self.assertAlmostEqual(scan_settings.get_frame_parameters(0).pixel_time_us, float(model.pixel_time_str))
            model.pixel_time_str = "2.5"
            self.assertAlmostEqual(2.5, scan_settings.get_frame_parameters(0).pixel_time_us)
            self.assertNotEqual(profile_frame_parameters_list[0].as_dict(), scan_settings.get_frame_parameters(0).as_dict())
            self.assertEqual(profile_frame_parameters_list[1].as_dict(), scan_settings.get_frame_parameters(1).as_dict())
            # test that property changed event is emitted
            with PropertyChangedEventWatcher(model, "pixel_time_str") as watcher:
                model.pixel_time_str = "2.55"
                self.assertTrue(watcher.changed)

            # test changing fov
            profile_frame_parameters_list = [scan_settings.get_frame_parameters(i) for i in range(2)]
            self.assertAlmostEqual(scan_settings.get_frame_parameters(0).fov_nm, float(model.fov_str))
            model.fov_str = "123.4"
            self.assertAlmostEqual(123.4, scan_settings.get_frame_parameters(0).fov_nm)
            self.assertNotEqual(profile_frame_parameters_list[0].as_dict(), scan_settings.get_frame_parameters(0).as_dict())
            self.assertEqual(profile_frame_parameters_list[1].as_dict(), scan_settings.get_frame_parameters(1).as_dict())
            # test that property changed event is emitted
            with PropertyChangedEventWatcher(model, "fov_str") as watcher:
                model.fov_str = "123.45"
                self.assertTrue(watcher.changed)

            # test changing width
            profile_frame_parameters_list = [scan_settings.get_frame_parameters(i) for i in range(2)]
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, int(model.width_str))
            model.width_str = "514"
            self.assertEqual(514, scan_settings.get_frame_parameters(0).pixel_size.width)
            self.assertNotEqual(profile_frame_parameters_list[0].as_dict(), scan_settings.get_frame_parameters(0).as_dict())
            self.assertEqual(profile_frame_parameters_list[1].as_dict(), scan_settings.get_frame_parameters(1).as_dict())
            # test that property changed event is emitted
            with PropertyChangedEventWatcher(model, "width_str") as watcher:
                model.width_str = "512"
                self.assertTrue(watcher.changed)

            # test changing height. initially will be None if not set
            profile_frame_parameters_list = [scan_settings.get_frame_parameters(i) for i in range(2)]
            self.assertIsNone(model.height_str)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, int(model.placeholder_height_str))
            model.height_str = "515"
            self.assertEqual(515, scan_settings.get_frame_parameters(0).pixel_size.height)
            self.assertNotEqual(profile_frame_parameters_list[0].as_dict(), scan_settings.get_frame_parameters(0).as_dict())
            self.assertEqual(profile_frame_parameters_list[1].as_dict(), scan_settings.get_frame_parameters(1).as_dict())
            # test that property changed event is emitted
            with PropertyChangedEventWatcher(model, "height_str") as watcher:
                model.height_str = "516"
                self.assertTrue(watcher.changed)

            # test changing rotation
            profile_frame_parameters_list = [scan_settings.get_frame_parameters(i) for i in range(2)]
            self.assertAlmostEqual(scan_settings.get_frame_parameters(0).rotation_deg, float(model.rotation_deg_str))
            model.rotation_deg_str = "12.3"
            self.assertAlmostEqual(12.3, scan_settings.get_frame_parameters(0).rotation_deg)
            self.assertNotEqual(profile_frame_parameters_list[0].as_dict(), scan_settings.get_frame_parameters(0).as_dict())
            self.assertEqual(profile_frame_parameters_list[1].as_dict(), scan_settings.get_frame_parameters(1).as_dict())
            # test that property changed event is emitted
            with PropertyChangedEventWatcher(model, "rotation_deg_str") as watcher:
                model.rotation_deg_str = "12.4"
                self.assertTrue(watcher.changed)

    def test_width_height_placeholder(self):
        # test that height field placeholder updates with width when height is empty, and that height syncs/unsyncs properly
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)
            scan_settings = scan_hardware_source.scan_settings

            # initial assumptions
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, int(model.width_str))
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, int(model.width_str))
            self.assertIsNone(model.height_str)
            self.assertEqual(model.placeholder_height_str, model.width_str)

            # change the width, height should update too
            model.width_str = "600"
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, 600)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, 600)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, int(model.width_str))
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, int(model.width_str))
            self.assertIsNone(model.height_str)
            self.assertEqual(model.placeholder_height_str, model.width_str)

            # change the height, should be independent of width now
            model.height_str = "500"
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, 600)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, 500)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, int(model.width_str))
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, int(model.height_str))

            # change the width again, height should remain unchanged
            model.width_str = "400"
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, 400)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, 500)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, int(model.width_str))
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, int(model.height_str))

            # change height back to empty, should sync to width again
            model.height_str = ""
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, 400)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, 400)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, int(model.width_str))
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, int(model.width_str))
            self.assertIsNone(model.height_str)
            self.assertEqual(model.placeholder_height_str, model.width_str)

            # change width again, height should follow
            model.width_str = "300"
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, 300)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, 300)
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, int(model.width_str))
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, int(model.width_str))
            self.assertIsNone(model.height_str)
            self.assertEqual(model.placeholder_height_str, model.width_str)

            # test changing width generates placeholder_height_str property changed event when height is synced
            model.height_str = ""
            with PropertyChangedEventWatcher(model, "placeholder_height_str") as watcher:
                model.width_str = "250"
                self.assertTrue(watcher.changed)

    def test_increase_decrease_fields(self):
        # test that the increase/decrease methods work for pixel time, fov, width, and height
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)
            scan_settings = scan_hardware_source.scan_settings

            # test increase/decrease pixel time
            initial_pixel_time = scan_settings.get_frame_parameters(0).pixel_time_us
            model.increase_pixel_time()
            self.assertGreater(scan_settings.get_frame_parameters(0).pixel_time_us, initial_pixel_time)
            model.decrease_pixel_time()
            self.assertAlmostEqual(scan_settings.get_frame_parameters(0).pixel_time_us, initial_pixel_time)

            # test increase/decrease fov
            initial_fov = scan_settings.get_frame_parameters(0).fov_nm
            model.increase_fov()
            self.assertGreater(scan_settings.get_frame_parameters(0).fov_nm, initial_fov)
            model.decrease_fov()
            self.assertAlmostEqual(scan_settings.get_frame_parameters(0).fov_nm, initial_fov)

            # test increase/decrease width
            initial_width = scan_settings.get_frame_parameters(0).pixel_size.width
            model.increase_width()
            self.assertGreater(scan_settings.get_frame_parameters(0).pixel_size.width, initial_width)
            model.decrease_width()
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.width, initial_width)

            # test increase/decrease height
            initial_height = scan_settings.get_frame_parameters(0).pixel_size.height
            model.increase_height()
            self.assertGreater(scan_settings.get_frame_parameters(0).pixel_size.height, initial_height)
            model.decrease_height()
            self.assertEqual(scan_settings.get_frame_parameters(0).pixel_size.height, initial_height)

    def test_subscan_and_line_scan_checkboxes(self):
        # test that the increase/decrease methods work for pixel time, fov, width, and height
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertFalse(model.subscan_checkbox_enabled)
            self.assertFalse(model.line_scan_checkbox_enabled)
            self.assertFalse(model.subscan_checkbox_checked)
            self.assertFalse(model.line_scan_checkbox_checked)
            self.assertIsNone(scan_hardware_source.get_current_frame_parameters().subscan_fractional_size)

            # acquire one scan
            with PropertyChangedEventWatcher(model, "subscan_checkbox_enabled") as watcher:
                self._acquire_one(document_controller, scan_hardware_source)
                self.assertTrue(model.subscan_checkbox_enabled)
                self.assertTrue(model.line_scan_checkbox_enabled)
                self.assertTrue(watcher.changed)

            # enable subscan mode
            with PropertyChangedEventWatcher(model, "subscan_checkbox_checked") as watcher:
                model.subscan_checkbox_checked = True
                self.assertFalse(model.line_scan_checkbox_checked)
                self.assertTrue(watcher.changed)
                self.assertIsNotNone(scan_hardware_source.get_current_frame_parameters().subscan_fractional_size)
                model.subscan_checkbox_checked = False
                self.assertIsNone(scan_hardware_source.get_current_frame_parameters().subscan_fractional_size)

            # enable line scan mode
            with PropertyChangedEventWatcher(model, "line_scan_checkbox_checked") as watcher:
                model.line_scan_checkbox_checked = True
                self.assertFalse(model.subscan_checkbox_checked)
                self.assertTrue(watcher.changed)

    def test_drift_correction_checkbox(self):
        # test that the drift correction checkbox works
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertFalse(model.drift_controls_enabled)
            self.assertFalse(model.drift_checkbox_checked)

            # acquire one scan
            with PropertyChangedEventWatcher(model, "drift_controls_enabled") as watcher:
                self._acquire_one(document_controller, scan_hardware_source)
                self.assertTrue(watcher.changed)

            # enable drift correction
            with PropertyChangedEventWatcher(model, "drift_checkbox_checked") as watcher:
                model.drift_checkbox_checked = True
                self.assertTrue(watcher.changed)
                model.drift_checkbox_checked = False

            # enable drift correction outside model
            with PropertyChangedEventWatcher(model, "drift_checkbox_checked") as watcher:
                scan_hardware_source.drift_enabled = True
                self.assertTrue(watcher.changed)

    def test_drift_interval_settings(self):
        # test that changing the drift interval string works
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertFalse(model.drift_controls_enabled)
            self.assertEqual(model.drift_settings_interval_str, "0")

            # acquire one scan
            self._acquire_one(document_controller, scan_hardware_source)

            # change drift interval
            with PropertyChangedEventWatcher(model, "drift_settings_interval_str") as watcher:
                model.drift_settings_interval_str = "1"
                self.assertTrue(watcher.changed)
                self.assertEqual("1", model.drift_settings_interval_str)
                self.assertEqual(1.0, scan_hardware_source.stem_controller.drift_settings.interval)

            # change drift interval externally
            with PropertyChangedEventWatcher(model, "drift_settings_interval_str") as watcher:
                scan_hardware_source.stem_controller.drift_settings = STEMController.DriftCorrectionSettings(2, STEMController.DriftIntervalUnit.SCAN)
                self.assertTrue(watcher.changed)
                scan_hardware_source.stem_controller.drift_settings = STEMController.DriftCorrectionSettings(0, STEMController.DriftIntervalUnit.FRAME)

            # change the drift units
            with PropertyChangedEventWatcher(model, "drift_settings_interval_units_index") as watcher:
                model.drift_settings_interval_units_index = 1
                self.assertTrue(watcher.changed)
                self.assertEqual(STEMController.DriftIntervalUnit.SCAN, scan_hardware_source.stem_controller.drift_settings.interval_units)
                scan_hardware_source.stem_controller.drift_settings = STEMController.DriftCorrectionSettings(0, STEMController.DriftIntervalUnit.FRAME)

            # change the driver units externally
            with PropertyChangedEventWatcher(model, "drift_settings_interval_units_index") as watcher:
                scan_hardware_source.stem_controller.drift_settings = STEMController.DriftCorrectionSettings(0, STEMController.DriftIntervalUnit.SCAN)
                self.assertTrue(watcher.changed)
                scan_hardware_source.stem_controller.drift_settings = STEMController.DriftCorrectionSettings(0, STEMController.DriftIntervalUnit.FRAME)

    def test_scan_and_abort_button(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertTrue(model.scan_button_enabled)
            self.assertEqual("Scan", model.scan_button_title)
            self.assertFalse(model.scan_abort_button_enabled)
            self.assertEqual("Stopped", model.play_state_text)

            # start playing
            with PropertyChangedEventWatcher(model, "scan_button_title") as watcher:
                with PropertyChangedEventWatcher(model, "scan_abort_button_enabled") as watcher2:
                    with PropertyChangedEventWatcher(model, "play_state_text") as watcher3:
                        try:
                            scan_hardware_source.start_playing(sync_timeout=5.0)
                            scan_hardware_source.get_next_xdatas_to_finish(timeout=5.0)
                            document_controller.periodic()
                            self.assertTrue(watcher.changed)
                            self.assertTrue(watcher2.changed)
                            self.assertTrue(watcher3.changed)
                            self.assertTrue(model.scan_button_enabled)
                            self.assertEqual("Stop", model.scan_button_title)
                            self.assertTrue(model.scan_abort_button_enabled)
                            self.assertTrue(model.record_button_enabled)
                            self.assertFalse(model.record_abort_button_enabled)
                            self.assertEqual("Acquiring", model.play_state_text)
                        except Exception as e:
                            scan_hardware_source.stop_playing(sync_timeout=5.0)
                            raise e

            # stop playing
            with PropertyChangedEventWatcher(model, "scan_button_title") as watcher:
                with PropertyChangedEventWatcher(model, "scan_abort_button_enabled") as watcher2:
                    with PropertyChangedEventWatcher(model, "play_state_text") as watcher3:
                        scan_hardware_source.stop_playing(sync_timeout=5.0)
                        document_controller.periodic()
                        self.assertTrue(watcher.changed)
                        self.assertTrue(watcher2.changed)
                        self.assertTrue(watcher3.changed)
                        self.assertTrue(model.scan_button_enabled)
                        self.assertEqual("Scan", model.scan_button_title)
                        self.assertFalse(model.scan_abort_button_enabled)
                        self.assertTrue(model.record_button_enabled)
                        self.assertFalse(model.record_abort_button_enabled)
                        self.assertEqual("Stopped", model.play_state_text)

            # start playing with button
            try:
                model.handle_scan_button_clicked()
                start_time = time.time()
                while not scan_hardware_source.is_playing and (time.time() - start_time) < 5.0:
                    time.sleep(0.1)
                    document_controller.periodic()
                self.assertTrue(scan_hardware_source.is_playing)
                model.handle_scan_button_clicked()
                start_time = time.time()
                while scan_hardware_source.is_playing and (time.time() - start_time) < 5.0:
                    time.sleep(0.1)
                    document_controller.periodic()
                self.assertFalse(scan_hardware_source.is_playing)
            finally:
                scan_hardware_source.abort_playing(sync_timeout=5.0)

            # start playing with button and abort
            try:
                model.handle_scan_button_clicked()
                start_time = time.time()
                while not scan_hardware_source.is_playing and (time.time() - start_time) < 5.0:
                    time.sleep(0.1)
                    document_controller.periodic()
                self.assertTrue(scan_hardware_source.is_playing)
                model.handle_scan_abort_button_clicked()
                start_time = time.time()
                while scan_hardware_source.is_playing and (time.time() - start_time) < 5.0:
                    time.sleep(0.1)
                    document_controller.periodic()
                self.assertFalse(scan_hardware_source.is_playing)
            finally:
                scan_hardware_source.abort_playing(sync_timeout=5.0)

    def test_record_and_abort_button(self):
        # this test may be unreliable on due to timing issues
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertTrue(model.record_button_enabled)
            self.assertFalse(model.record_abort_button_enabled)

            # start playing with button
            try:
                model.pixel_time_str = "5"  # slow down for testing
                model.handle_scan_button_clicked()
                start_time = time.time()
                while not scan_hardware_source.is_playing and (time.time() - start_time) < 5.0:
                    time.sleep(0.1)
                    document_controller.periodic()
                time.sleep(0.1)
                self.assertTrue(scan_hardware_source.is_playing)
                self.assertTrue(model.record_button_enabled)
                self.assertFalse(model.record_abort_button_enabled)
                with PropertyChangedEventWatcher(model, "record_button_enabled") as watcher:
                    with PropertyChangedEventWatcher(model, "record_abort_button_enabled") as watcher2:
                        model.handle_record_button_clicked()
                        while not scan_hardware_source.is_recording and (time.time() - start_time) < 5.0:
                            time.sleep(0.1)
                            document_controller.periodic()
                        time.sleep(0.1)
                        document_controller.periodic()
                        self.assertTrue(scan_hardware_source.is_recording)
                        self.assertFalse(model.record_button_enabled)
                        self.assertTrue(model.record_abort_button_enabled)
                with PropertyChangedEventWatcher(model, "record_button_enabled") as watcher:
                    with PropertyChangedEventWatcher(model, "record_abort_button_enabled") as watcher2:
                        while scan_hardware_source.is_recording and (time.time() - start_time) < 5.0:
                            time.sleep(0.1)
                            document_controller.periodic()
                        time.sleep(0.1)
                        self.assertFalse(scan_hardware_source.is_recording)
                        self.assertTrue(model.record_button_enabled)
                        self.assertFalse(model.record_abort_button_enabled)
                model.handle_scan_abort_button_clicked()
                start_time = time.time()
                while scan_hardware_source.is_playing and (time.time() - start_time) < 5.0:
                    time.sleep(0.1)
                    document_controller.periodic()
                self.assertFalse(scan_hardware_source.is_playing)
            finally:
                scan_hardware_source.abort_recording(sync_timeout=5.0)
                scan_hardware_source.abort_playing(sync_timeout=5.0)

    def test_probe_checkbox_and_text(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertFalse(model.probe_position_enabled)
            self.assertEqual("Parked Default", model.probe_state_text)

            # enable probe positioning
            with PropertyChangedEventWatcher(model, "probe_position_enabled") as watcher:
                with PropertyChangedEventWatcher(model, "probe_state_text") as watcher2:
                    model.probe_position_enabled = True
                    self.assertTrue(model.probe_position_enabled)
                    self.assertTrue(watcher.changed)
                    self.assertTrue(watcher2.changed)
                    self.assertEqual("Parked 50%, 50%", model.probe_state_text)

            # change the probe position externally
            with PropertyChangedEventWatcher(model, "probe_state_text") as watcher:
                scan_hardware_source.stem_controller.probe_position = Geometry.FloatPoint(0.25, 0.75)
                self.assertTrue(watcher.changed)
                self.assertEqual("Parked 75%, 25%", model.probe_state_text)

            # change probe state externally
            with PropertyChangedEventWatcher(model, "probe_position_enabled") as watcher:
                with PropertyChangedEventWatcher(model, "probe_state_text") as watcher2:
                    scan_hardware_source.stem_controller.probe_position = None
                    self.assertFalse(model.probe_position_enabled)
                    self.assertTrue(watcher.changed)
                    self.assertTrue(watcher2.changed)
                    self.assertEqual("Parked Default", model.probe_state_text)

    def test_ac_line_sync_checkbox(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertFalse(model.ac_line_sync_enabled)

            with PropertyChangedEventWatcher(model, "ac_line_sync_enabled") as watcher:
                model.ac_line_sync_enabled = True
                self.assertTrue(model.ac_line_sync_enabled)
                self.assertTrue(watcher.changed)
                self.assertTrue(scan_hardware_source.get_frame_parameters(scan_hardware_source.selected_profile_index).ac_line_sync)

    def test_fov_out_of_range(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            model = ScanControlPanel.ScanControlPanelModel(scan_hardware_source, document_controller)

            # check assumptions
            self.assertEqual("black", model.fov_label_color)
            self.assertEqual("Maximum field of view: 1600nm", model.fov_label_tool_tip)

            # set fov to out of range value
            with PropertyChangedEventWatcher(model, "fov_label_color") as watcher:
                with PropertyChangedEventWatcher(model, "fov_label_tool_tip") as watcher2:
                    model.fov_str = "2000"
                    document_controller.periodic()
                    self.assertEqual("red", model.fov_label_color)
                    self.assertEqual("Exceeds maximum field of view: 1600nm", model.fov_label_tool_tip)
                    self.assertTrue(watcher.changed)
                    self.assertTrue(watcher2.changed)

            # set fov to nearly out of range value
            with PropertyChangedEventWatcher(model, "fov_label_color") as watcher:
                with PropertyChangedEventWatcher(model, "fov_label_tool_tip") as watcher2:
                    model.fov_str = "1500"
                    document_controller.periodic()
                    self.assertEqual("orange", model.fov_label_color)
                    self.assertEqual("Near maximum field of view: 1600nm", model.fov_label_tool_tip)
                    self.assertTrue(watcher.changed)
                    self.assertTrue(watcher2.changed)
