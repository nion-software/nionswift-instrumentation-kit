import contextlib
import threading
import time
import unittest

import numpy

from nion.swift import Application
from nion.swift import DocumentController
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.swift.model import Graphics
from nion.swift.model import HardwareSource
from nion.swift.test import HardwareSource_test
from nion.ui import TestUI
from nion.utils import Event
from nion.utils import Geometry
from nion.instrumentation import stem_controller
from nion.instrumentation import scan_base
from nionswift_plugin.nion_instrumentation_ui import ScanControlPanel

"""
# running in Swift
import sys, unittest
from superscan import SimulatorScanControl_test
suite = unittest.TestLoader().loadTestsFromTestCase(SimulatorScanControl_test.TestSimulatorScanControlClass)
result = unittest.TextTestResult(sys.stdout, True, True)
suite.run(result)
"""

class TestScanControlClass(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)
        HardwareSource.HardwareSourceManager().hardware_sources = []
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()

    def tearDown(self):
        HardwareSource.HardwareSourceManager()._close_hardware_sources()
        HardwareSource.HardwareSourceManager()._close_instruments()

    def _acquire_one(self, document_controller, hardware_source):
        hardware_source.start_playing(3.0)
        hardware_source.stop_playing(3.0)
        document_controller.periodic()

    def _record_one(self, document_controller, hardware_source, scan_state_controller):
        frame_time = hardware_source.get_current_frame_time()

        enabled_channel_count = sum(hardware_source.get_channel_state(channel_index).enabled for channel_index in range(hardware_source.channel_count))

        event = threading.Event()
        event_count_ref = [0]
        def handle_record_data_item(data_item):  # called from thread
            def perform():
                document_controller.document_model.append_data_item(data_item)
                result_display_panel = document_controller.next_result_display_panel()
                if result_display_panel:
                    result_display_panel.set_display_panel_data_item(data_item)
            document_controller.queue_task(perform)
            event_count_ref[0] += 1
            if event_count_ref[0] == enabled_channel_count:
                event.set()

        scan_state_controller.handle_record_clicked(handle_record_data_item)
        time.sleep(frame_time * 0.5)
        start_time = time.time()
        while hardware_source.is_recording:
            time.sleep(frame_time * 0.5)
            self.assertTrue(time.time() - start_time < 30.0)
        self.assertTrue(event.wait(3.0))
        document_controller.periodic()

    def _setup_scan_hardware_source(self, channel_id=None):
        document_model = DocumentModel.DocumentModel()
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        instrument = self._setup_instrument()
        hardware_source = self._setup_hardware_source(instrument)
        HardwareSource.HardwareSourceManager().register_hardware_source(hardware_source)
        scan_state_controller = ScanControlPanel.ScanControlStateController(hardware_source, document_controller.queue_task, document_controller.document_model, channel_id)
        scan_state_controller.initialize_state()
        return document_controller, document_model, instrument, hardware_source, scan_state_controller

    def _setup_instrument(self):
        raise NotImplementedError()

    def _close_instrument(self, instrument) -> None:
        raise NotImplementedError()

    def _setup_hardware_source(self, instrument) -> scan_base.ScanHardwareSource:
        raise NotImplementedError()

    def _close_hardware_source(self) -> None:
        raise NotImplementedError()

    def _close_scan_hardware_source(self, document_controller, document_model, instrument, hardware_source, scan_state_controller):
        scan_state_controller.close()
        hardware_source.close()
        HardwareSource.HardwareSourceManager().unregister_hardware_source(hardware_source)
        self._close_hardware_source()
        self._close_instrument(instrument)
        document_controller.close()

    def _make_scan_context(self, channel_id=None):

        class ScanContext:

            def __init__(self, scan_test):
                self.__scan_test = scan_test

            def __enter__(self):
                document_controller, document_model, instrument, hardware_source, scan_state_controller = self.__scan_test._setup_scan_hardware_source(channel_id)

                self.document_controller = document_controller
                self.document_model = document_model
                self.instrument = instrument
                self.hardware_source = hardware_source
                self.scan_state_controller = scan_state_controller
                self.probe_view = stem_controller.ProbeView(self.instrument, document_controller.event_loop)

                return self

            def __exit__(self, *exc_details):
                self.probe_view.close()
                self.probe_view = None
                self.__scan_test._close_scan_hardware_source(self.document_controller, self.document_model, self.instrument, self.hardware_source, self.scan_state_controller)

            @property
            def objects(self):
                return self.document_controller, self.document_model, self.hardware_source, self.scan_state_controller

        return ScanContext(self)

    ## STANDARD ACQUISITION TESTS ##

    # Do not change the comment above as it is used to search for places needing updates when a new
    # standard acquisition test is added.

    def test_acquiring_frames_with_generator_produces_correct_frame_numbers(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_acquiring_frames_with_generator_produces_correct_frame_numbers(self, hardware_source, document_controller)

    def test_acquire_multiple_frames_reuses_same_data_item(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_acquire_multiple_frames_reuses_same_data_item(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_stop_actually_stops_acquisition(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_simple_hardware_start_and_stop_actually_stops_acquisition(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_abort_works_as_expected(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_simple_hardware_start_and_abort_works_as_expected(self, hardware_source, document_controller)

    def test_record_only_acquires_one_item(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_record_only_acquires_one_item(self, hardware_source, document_controller)

    def test_record_during_view_records_one_item_and_keeps_viewing(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_record_during_view_records_one_item_and_keeps_viewing(self, hardware_source, document_controller)

    def test_abort_record_during_view_returns_to_view(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_abort_record_during_view_returns_to_view(self, hardware_source, document_controller)

    def test_view_reuses_single_data_item(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_view_reuses_single_data_item(self, hardware_source, document_controller)

    def test_get_next_data_elements_to_finish_returns_full_frames(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_get_next_data_elements_to_finish_returns_full_frames(self, hardware_source, document_controller)

    def test_exception_during_view_halts_scan(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_exception_during_view_halts_playback(self, hardware_source, hardware_source.get_current_frame_time())

    def test_exception_during_record_halts_scan(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_selected_profile_index(2)
            HardwareSource_test._test_exception_during_record_halts_playback(self, hardware_source, hardware_source.get_current_frame_time())

    def test_able_to_restart_scan_after_exception_scan(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_able_to_restart_view_after_exception(self, hardware_source, hardware_source.get_current_frame_time())

    def test_record_starts_and_finishes_in_reasonable_time(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            HardwareSource_test._test_record_starts_and_finishes_in_reasonable_time(self, hardware_source, hardware_source.get_current_frame_time() * 16)

    # End of standard acquisition tests.

    def test_record_followed_by_view_updates_display(self):
        with self._make_scan_context("a") as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            displayed_data_item = [None]
            def display_data_item_changed(data_item):
                displayed_data_item[0] = data_item
            scan_state_controller.on_display_data_item_changed = display_data_item_changed
            self._record_one(document_controller, hardware_source, scan_state_controller)
            document_controller.periodic()  # extra to handle binding
            self.assertEqual(document_model.data_items[0], displayed_data_item[0])
            self.assertTrue(numpy.array_equal(document_model.data_items[0].data, document_model.data_items[1].data))
            self._acquire_one(document_controller, hardware_source)
            document_controller.periodic()  # extra to handle binding
            self.assertEqual(document_model.data_items[0], displayed_data_item[0])
            self.assertFalse(numpy.array_equal(document_model.data_items[0].data, document_model.data_items[1].data))
            # clean up
            hardware_source.abort_playing()

    def test_record_gets_correct_data_when_record_started_during_view(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.pixel_time_us = 1
            hardware_source.set_frame_parameters(0, frame_parameters)
            frame_time = hardware_source.get_current_frame_time()
            hardware_source.start_playing()
            time.sleep(frame_time * 0.25)
            self._record_one(document_controller, hardware_source, scan_state_controller)
            self.assertEqual(len(document_model.data_items), 2)  # check assumptions
            hardware_source.stop_playing(3.0)
            document_controller.periodic()
            self.assertEqual(document_model.data_items[1].dimensional_shape, hardware_source.get_frame_parameters(2).size)
            self.assertNotEqual(document_model.data_items[0].dimensional_shape, document_model.data_items[1].dimensional_shape)

    def test_record_names_results_correctly(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, True)
            hardware_source.set_channel_enabled(1, True)
            self._record_one(document_controller, hardware_source, scan_state_controller)
            self.assertTrue(document_model.data_items[2].title.startswith(document_model.data_items[0].title))
            self.assertTrue(len(document_model.data_items[2].title) > len(document_model.data_items[0].title))
            self.assertTrue(document_model.data_items[3].title.startswith(document_model.data_items[1].title))
            self.assertTrue(len(document_model.data_items[3].title) > len(document_model.data_items[1].title))

    def test_enable_positioned_after_one_frame_acquisition_should_add_graphic(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 0)
            scan_state_controller.handle_positioned_check_box(True)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 1)

    def test_disable_positioned_after_one_frame_acquisition_should_remove_graphic(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 1)
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 0)

    def test_deleting_probe_graphic_after_one_frame_acquisition_should_disable_positioned(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            display = document_model.data_items[0].primary_display_specifier.display
            probe_graphic = display.graphics[0]
            display.remove_graphic(probe_graphic)
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 0)
            self.assertIsNone(hardware_source.probe_position)

    def test_deleting_display_after_one_frame_acquisition_should_disable_positioned(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            data_item = document_model.data_items[0]
            document_model.remove_data_item(data_item)
            self.assertEqual(len(document_model.data_items), 0)
            self.assertIsNone(hardware_source.probe_position)

    # TODO: test case where one of two displays is deleted, what happens to probe graphic on other one?
    # TODO: test case where probe graphic should be removed when a channel is disabled

    def test_start_playing_after_stopping_should_remove_probe_position_graphic(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            display = document_model.data_items[0].primary_display_specifier.display
            self.assertEqual(len(display.graphics), 1)
            hardware_source.start_playing()
            hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            self.assertEqual(len(display.graphics), 0)
            hardware_source.stop_playing()

    def test_start_playing_with_existing_probe_position_graphic_should_remove_it(self):
        document_model = DocumentModel.DocumentModel()
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        data_item = DataItem.DataItem(numpy.zeros((256, 256)) + 16)
        graphic = Graphics.PointGraphic()
        graphic.graphic_id = "probe"
        graphic.position = 0.5, 0.5
        graphic.is_bounds_constrained = True
        data_item.displays[0].add_graphic(graphic)
        document_model.append_data_item(data_item)
        instrument = self._setup_instrument()
        hardware_source = self._setup_hardware_source(instrument)
        document_model.setup_channel(document_model.make_data_item_reference_key(hardware_source.hardware_source_id, hardware_source.data_channels[0].channel_id), data_item)
        HardwareSource.HardwareSourceManager().register_hardware_source(hardware_source)
        display = document_model.data_items[0].primary_display_specifier.display
        scan_state_controller = ScanControlPanel.ScanControlStateController(hardware_source, document_controller.queue_task, document_controller.document_model, None)
        scan_state_controller.initialize_state()
        probe_view = stem_controller.ProbeView(instrument, document_controller.event_loop)
        self.assertEqual(len(display.graphics), 0)
        with contextlib.closing(document_controller):
            try:
                with contextlib.closing(hardware_source), contextlib.closing(probe_view), contextlib.closing(scan_state_controller):
                    scan_state_controller.handle_positioned_check_box(True)
                    self._acquire_one(document_controller, hardware_source)
                    self.assertEqual(len(display.graphics), 1)
                    hardware_source.start_playing()
                    hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                    document_controller.periodic()
                    self.assertEqual(len(display.graphics), 0)
                    hardware_source.stop_playing()
            finally:
                HardwareSource.HardwareSourceManager().unregister_hardware_source(hardware_source)
                self._close_hardware_source()
                self._close_instrument(instrument)

    def test_acquiring_multiple_channels_attaches_common_scan_id(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, True)
            hardware_source.set_channel_enabled(2, True)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items), 2)
            metadata0 = document_model.data_items[0].metadata
            metadata1 = document_model.data_items[1].metadata
            scan_id0 = metadata0.get("hardware_source", dict()).get("scan_id")
            scan_id1 = metadata1.get("hardware_source", dict()).get("scan_id")
            self.assertIsNotNone(scan_id0)
            self.assertEqual(scan_id0, scan_id1)

    def test_acquiring_multiple_channels_twice_generates_different_scan_ids(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, True)
            hardware_source.set_channel_enabled(2, True)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items), 2)
            metadata0 = document_model.data_items[0].metadata
            metadata1 = document_model.data_items[1].metadata
            scan_id00 = metadata0.get("hardware_source", dict()).get("scan_id")
            scan_id01 = metadata1.get("hardware_source", dict()).get("scan_id")
            self._acquire_one(document_controller, hardware_source)
            metadata0 = document_model.data_items[0].metadata
            metadata1 = document_model.data_items[1].metadata
            scan_id10 = metadata0.get("hardware_source", dict()).get("scan_id")
            scan_id11 = metadata1.get("hardware_source", dict()).get("scan_id")
            self.assertIsNotNone(scan_id00)
            self.assertIsNotNone(scan_id10)
            self.assertEqual(scan_id00, scan_id01)
            self.assertEqual(scan_id10, scan_id11)

    def test_ui_generates_default_channel_names_when_none_are_provided(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_state_controller = ScanControlPanel.ScanControlStateController(hardware_source, document_controller.queue_task, document_controller.document_model, None)
            names = {}

            def channel_state_changed(channel_index, channel_id, name, enabled):
                names[channel_index] = name

            scan_state_controller.on_channel_state_changed = channel_state_changed
            scan_state_controller.initialize_state()
            hardware_source.channel_state_changed_event.fire(0, "a", str(), True)
            self.assertNotEqual(names[0], str())
            self.assertIsNotNone(names[0])

    def test_start_playing_with_no_channels_enabled_does_nothing(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, False)
            frame_time = hardware_source.get_current_frame_time()
            hardware_source.start_playing()
            time.sleep(frame_time * 0.5)
            is_playing = hardware_source.is_playing
            hardware_source.stop_playing()
            self.assertFalse(is_playing)

    def test_disabling_all_channels_during_play_stops_playback(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            self._acquire_one(document_controller, hardware_source)  # throw out first scan, which can take longer due to initial conditions
            frame_time = hardware_source.get_current_frame_time()
            hardware_source.start_playing()
            time.sleep(frame_time * 0.25)
            hardware_source.set_channel_enabled(0, False)
            time.sleep(frame_time * 0.25)  # this sleep is not really necessary but leave it here for extra robustness
            was_playing = hardware_source.is_playing
            hardware_source.periodic()
            time.sleep(frame_time * 2)  # scan should definitely be stopped after 2 frames
            is_playing = hardware_source.is_playing
            hardware_source.stop_playing()
            self.assertTrue(was_playing)
            self.assertFalse(is_playing)

    def test_scan_and_record_button_disabled_when_no_channels_are_enabled(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_enabled_ref = [True]
            def scan_button_state_changed(enabled, play_button_state):
                scan_enabled_ref[0] = enabled

            record_enabled_ref = [True]
            def record_button_state_changed(visible, enabled):
                record_enabled_ref[0] = enabled

            scan_state_controller.on_scan_button_state_changed = scan_button_state_changed
            scan_state_controller.on_record_button_state_changed = record_button_state_changed
            hardware_source.set_channel_enabled(0, False)
            hardware_source.periodic()
            self.assertFalse(scan_enabled_ref[0])
            self.assertFalse(record_enabled_ref[0])
            hardware_source.set_channel_enabled(0, True)
            hardware_source.periodic()
            self.assertTrue(scan_enabled_ref[0])
            self.assertTrue(record_enabled_ref[0])

    def test_record_generates_a_data_item(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            for i in range(3):
                self._record_one(document_controller, hardware_source, scan_state_controller)
            self.assertEqual(len(document_model.data_items), 4)  # 1 view image (always acquired), 3 records

    def test_ability_to_set_profile_parameters_is_reflected_in_acquisition(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = (300, 300)
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.size = (500, 500)
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (300, 300))
            hardware_source.set_selected_profile_index(1)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (500, 500))

    def test_change_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = (300, 300)
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.size = (500, 500)
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            hardware_source.start_playing()
            self.assertEqual(hardware_source.get_next_xdatas_to_start()[0].data.shape, (300, 300))
            hardware_source.set_selected_profile_index(1)
            self.assertEqual(hardware_source.get_next_xdatas_to_start()[0].data.shape, (500, 500))
            hardware_source.abort_playing()

    def test_mutating_current_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = (300, 300)
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.size = (500, 500)
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            hardware_source.start_playing()
            self.assertEqual(hardware_source.get_next_xdatas_to_start()[0].data.shape, (300, 300))
            hardware_source.set_frame_parameters(0, frame_parameters_1)
            self.assertEqual(hardware_source.get_next_xdatas_to_start()[0].data.shape, (500, 500))
            hardware_source.abort_playing()

    def test_changing_frame_parameters_during_record_does_not_affect_recording(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            # verify that the frame parameters are applied to the _next_ frame.
            frame_parameters_2 = hardware_source.get_frame_parameters(2)
            frame_parameters_2.size = (500, 500)
            frame_time = frame_parameters_2.pixel_time_us * 500 * 500 / 1000000.0
            hardware_source.start_recording()
            time.sleep(frame_time * 0.6)
            hardware_source.set_frame_parameters(2, frame_parameters_2)
            self.assertEqual(hardware_source.get_next_xdatas_to_finish(10.0)[0].data.shape, (1024, 1024))
            start_time = time.time()
            while hardware_source.is_recording:
                time.sleep(frame_time)
                self.assertTrue(time.time() - start_time < 3.0)
            # confirm that recording is done, then verify that the frame parameters are actually applied to the _next_ frame.
            hardware_source.start_recording()
            time.sleep(frame_time * 0.6)
            self.assertEqual(hardware_source.get_next_xdatas_to_finish(10.0)[0].data.shape, (500, 500))

    def test_capturing_during_view_captures_new_data_items(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(1, True)
            hardware_source.start_playing()
            hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)

            def display_new_data_item(data_item):
                document_model.append_data_item(data_item)

            scan_state_controller.on_display_new_data_item = display_new_data_item
            scan_state_controller.handle_capture_clicked()
            hardware_source.stop_playing()
            hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 4)

    def test_ability_to_start_playing_with_custom_parameters(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = (200, 200)
            hardware_source.set_current_frame_parameters(frame_parameters_0)
            hardware_source.set_channel_enabled(1, True)
            hardware_source.start_playing()
            hardware_source.get_next_xdatas_to_finish()
            hardware_source.stop_playing()
            #hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)
            self.assertEqual(document_model.data_items[0].dimensional_shape, (200, 200))
            self.assertEqual(document_model.data_items[1].dimensional_shape, (200, 200))

    def test_ability_to_start_recording_with_custom_parameters(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = (200, 200)
            hardware_source.set_record_frame_parameters(frame_parameters_0)
            hardware_source.set_channel_enabled(1, True)
            hardware_source.start_recording()
            recorded_extended_data_list = hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)  # 2 view images (always grabbed)
            self.assertEqual(len(recorded_extended_data_list), 2)  # 2 record images
            self.assertEqual(document_model.data_items[0].dimensional_shape, (200, 200))
            self.assertEqual(document_model.data_items[1].dimensional_shape, (200, 200))
            self.assertEqual(recorded_extended_data_list[0].data.shape, (200, 200))
            self.assertEqual(recorded_extended_data_list[1].data.shape, (200, 200))

    def test_changing_profile_updates_frame_parameters_in_ui(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            scan_state_controller.on_frame_parameters_changed = frame_parameters_changed
            hardware_source.set_selected_profile_index(1)
            self.assertIsNotNone(frame_parameters_ref[0])

    def test_changing_current_profiles_frame_parameters_updates_frame_parameters_in_ui(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            scan_state_controller.on_frame_parameters_changed = frame_parameters_changed
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = (200, 200)
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            self.assertIsNotNone(frame_parameters_ref[0])
            self.assertEqual(frame_parameters_ref[0].size, (200, 200))

    def test_changing_fov_updates_only_that_profile(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_selected_profile_index(0)
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 10
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            self.assertEqual(hardware_source.get_frame_parameters(0).fov_nm, 10)
            hardware_source.set_selected_profile_index(1)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.fov_nm = 20
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            self.assertEqual(hardware_source.get_frame_parameters(1).fov_nm, 20)
            hardware_source.set_selected_profile_index(2)
            frame_parameters_2 = hardware_source.get_frame_parameters(2)
            frame_parameters_2.fov_nm = 30
            hardware_source.set_frame_parameters(2, frame_parameters_2)
            self.assertEqual(hardware_source.get_frame_parameters(2).fov_nm, 30)
            hardware_source.set_selected_profile_index(0)
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 1
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            self.assertEqual(frame_parameters_1.fov_nm, 20)
            frame_parameters_2 = hardware_source.get_frame_parameters(2)
            self.assertEqual(frame_parameters_2.fov_nm, 30)

    def test_changing_size_is_reflected_in_new_acquisition(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_state_controller.handle_linked_changed(False)
            scan_state_controller.handle_decrease_width()
            scan_state_controller.handle_decrease_height()
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (128, 128))

    def test_record_during_view_uses_record_mode(self):
        # NOTE: requires stages_per_frame == 2
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 10
            frame_parameters_0.pixel_time_us = 10
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_2 = hardware_source.get_frame_parameters(2)
            frame_parameters_2.fov_nm = 100
            hardware_source.set_frame_parameters(2, frame_parameters_2)
            frame_time = hardware_source.get_current_frame_time()
            hardware_source.start_playing()
            self.assertEqual(hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["fov_nm"], 10)
            # give view a chance to start before recording. this must finished _before_ the final segment of partial
            # acquisition for this test to work; otherwise the final segment finishes and notifies and gets picked up
            # in the first wait call after start_recording below.
            # Also, time of partial acquisition  must be less than 50% of the frame time
            time.sleep(frame_time * 0.1)
            hardware_source.start_recording()
            time.sleep(frame_time * 0.1)  # give record a chance to start; starting record will abort view immediately
            self.assertEqual(hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["fov_nm"], 100)
            self.assertEqual(hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["fov_nm"], 10)

    def test_consecutive_frames_have_unique_data(self):
        # this test will fail if the scan is saturated (or otherwise produces identical values naturally)
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = 1024, 1024
            frame_parameters_0.pixel_time_us = 2
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            hardware_source.start_playing()
            data_list = list()
            for _ in range(16):
                data_list.append(hardware_source.get_next_xdatas_to_finish()[0].data)
            for row in range(0, 1024, 32):
                s = slice(row, row+32), slice(0, 1024)
                for data in data_list[1:]:
                    self.assertFalse(numpy.array_equal(data_list[0][s], data[s]))

    def test_changing_width_when_linked_changes_height_too(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_state_controller.handle_decrease_width()
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (128, 128))

    def test_changing_width_when_unlinked_does_not_change_height(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_state_controller.handle_linked_changed(False)
            scan_state_controller.handle_decrease_width()
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 128))

    def test_obscenely_large_scan_does_not_crash(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = 32768, 32768
            frame_parameters_0.pixel_time_us = 1
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            hardware_source.start_playing()
            hardware_source.abort_playing()

    def test_big_scan_does_not_prevent_further_playing(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            original_size = frame_parameters_0.size
            original_pixel_time_us = frame_parameters_0.pixel_time_us
            frame_parameters_0.size = 8192, 8192
            frame_parameters_0.pixel_time_us = 1
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            hardware_source.start_playing(sync_timeout=3.0)
            hardware_source.abort_playing(sync_timeout=3.0)
            frame_parameters_0.size = original_size
            frame_parameters_0.pixel_time_us = original_pixel_time_us
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            hardware_source.start_playing(sync_timeout=3.0)
            hardware_source.abort_playing(sync_timeout=3.0)

    def test_closing_display_panel_with_scan_controller_shuts_down_controller_correctly(self):
        # NOTE: this is a duplicate of test_closing_display_panel_with_display_controller_shuts_down_controller_correctly
        ScanControlPanel.run()
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            d = {"type": "splitter", "orientation": "vertical", "splits": [0.5, 0.5], "children": [
                {"type": "image", "uuid": "0569ca31-afd7-48bd-ad54-5e2bb9f21102", "identifier": "a", "selected": True},
                {"type": "image", "uuid": "acd77f9f-2f6f-4fbf-af5e-94330b73b997", "identifier": "b"}]}
            workspace_2x1 = document_controller.workspace_controller.new_workspace("2x1", d)
            document_controller.workspace_controller.change_workspace(workspace_2x1)
            root_canvas_item = document_controller.workspace_controller.image_row.children[0]._root_canvas_item()
            root_canvas_item.update_layout(Geometry.IntPoint(), Geometry.IntSize(width=640, height=480))
            hardware_source.start_playing()
            try:
                frame_time = hardware_source.get_current_frame_time()
                time.sleep(frame_time * 2)  # let view start
                d = {"type": "image", "controller_type": "scan-live", "hardware_source_id": hardware_source.hardware_source_id, "channel_id": "a"}
                display_panel = document_controller.selected_display_panel
                display_panel.change_display_panel_content(d)
                time.sleep(frame_time * 2)  # let view start
                document_controller.periodic()
                document_controller.workspace_controller.remove_display_panel(display_panel)
                time.sleep(frame_time * 2)  # let view start
                document_controller.periodic()
            finally:
                hardware_source.abort_playing()

    def test_probe_graphic_gets_closed(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            probe_graphic = document_model.data_items[0].primary_display_specifier.display.graphics[0]
            self.assertFalse(probe_graphic._closed)
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            self.assertIsNone(hardware_source.probe_position)
            self.assertTrue(probe_graphic._closed)

    def test_probe_appears_on_all_channels(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, True)
            hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            self.assertIsNotNone(document_model.data_items[0].primary_display_specifier.display.graphics[0])
            self.assertIsNotNone(document_model.data_items[1].primary_display_specifier.display.graphics[0])
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            self.assertIsNone(hardware_source.probe_position)
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 0)
            self.assertEqual(len(document_model.data_items[1].primary_display_specifier.display.graphics), 0)

    def test_probe_on_multiple_channels_shuts_down_properly(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, True)
            hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            document_controller.periodic()
            document_model.data_items[0].primary_display_specifier.display.graphics[0].position = 0.3, 0.4
            document_controller.periodic()
            # will output extraneous messages when it fails

    def test_probe_on_multiple_channels_deletes_properly(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, True)
            hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            document_controller.periodic()
            display = document_model.data_items[0].primary_display_specifier.display
            probe_graphic = display.graphics[0]
            display.remove_graphic(probe_graphic)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 0)
            self.assertEqual(len(document_model.data_items[1].primary_display_specifier.display.graphics), 0)
            self.assertIsNone(hardware_source.probe_position)
            # will output extraneous messages when it fails

    def test_probe_on_multiple_channels_sets_to_none_properly(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            hardware_source.set_channel_enabled(0, True)
            hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            document_controller.periodic()
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items[0].primary_display_specifier.display.graphics), 0)
            self.assertEqual(len(document_model.data_items[1].primary_display_specifier.display.graphics), 0)
            self.assertIsNone(hardware_source.probe_position)

    def test_setting_probe_position_updates_probe_graphic(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_context.instrument.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            document_controller.periodic()
            probe_graphic = document_model.data_items[0].primary_display_specifier.display.graphics[0]
            self.assertEqual(scan_context.instrument.probe_position, probe_graphic.position)
            scan_context.instrument.set_probe_position(Geometry.FloatPoint(y=0.45, x=0.65))
            document_controller.periodic()
            self.assertEqual(scan_context.instrument.probe_position, probe_graphic.position)

    def test_setting_probe_graphic_updates_probe_position(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            scan_context.instrument.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)
            document_controller.periodic()
            probe_graphic = document_model.data_items[0].primary_display_specifier.display.graphics[0]
            self.assertEqual(scan_context.instrument.probe_position, probe_graphic.position)
            probe_graphic.position = Geometry.FloatPoint(y=0.45, x=0.65)
            document_controller.periodic()
            self.assertEqual(scan_context.instrument.probe_position, probe_graphic.position)
            self.assertEqual(scan_context.instrument.probe_position, hardware_source._get_last_idle_position_for_test())

    def test_acquire_into_empty_scan_controlled_display_panel(self):
        ScanControlPanel.run()
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            display_panel = document_controller.selected_display_panel
            display_panel.change_display_panel_content({"controller_type": "scan-live", "hardware_source_id": hardware_source.hardware_source_id, "channel_id": hardware_source.get_channel_state(0).channel_id})
            scan_context.instrument.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, hardware_source)

    def test_subscan_state_goes_from_invalid_to_disabled_upon_first_acquisition(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            self.assertEqual(stem_controller.SubscanState.INVALID, hardware_source.subscan_state)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(stem_controller.SubscanState.DISABLED, hardware_source.subscan_state)

    def test_enabling_subscan_during_initial_acquisition_puts_graphic_on_context(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            with contextlib.closing(stem_controller.ProbeViewController(document_controller.event_loop)):
                hardware_source.start_playing()
                hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                display = document_model.data_items[0].primary_display_specifier.display
                self.assertEqual(len(display.graphics), 0)
                hardware_source.subscan_enabled = True
                hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                hardware_source.stop_playing()
                self.assertEqual(2, len(document_model.data_items))
                self.assertEqual(1, len(display.graphics))
                self.assertEqual(0, len(document_model.data_items[1].primary_display_specifier.display.graphics))

    def test_removing_subscan_graphic_disables_subscan_when_acquisition_running(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            with contextlib.closing(stem_controller.ProbeViewController(document_controller.event_loop)):
                hardware_source.start_playing()
                hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                display = document_model.data_items[0].primary_display_specifier.display
                self.assertEqual(len(display.graphics), 0)
                hardware_source.subscan_enabled = True
                hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                display.remove_graphic(display.graphics[0])
                hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                hardware_source.stop_playing()
                self.assertFalse(hardware_source.subscan_enabled)

    def test_removing_subscan_graphic_disables_subscan_when_acquisition_stopped(self):
        with self._make_scan_context() as scan_context:
            document_controller, document_model, hardware_source, scan_state_controller = scan_context.objects
            with contextlib.closing(stem_controller.ProbeViewController(document_controller.event_loop)):
                hardware_source.start_playing()
                hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                display = document_model.data_items[0].primary_display_specifier.display
                self.assertEqual(len(display.graphics), 0)
                hardware_source.subscan_enabled = True
                hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                hardware_source.stop_playing()
                display.remove_graphic(display.graphics[0])
                self.assertFalse(hardware_source.subscan_enabled)

    def planned_test_changing_pixel_count_mid_scan_does_not_change_nm_per_pixel(self):
        pass

    def planned_test_custom_record_followed_by_ui_record_uses_ui_frame_parameters(self):
        pass

    def planned_test_setting_custom_frame_parameters_updates_ui(self):
        pass

    def planned_test_cold_start_acquisition_from_thread_produces_data(self):
        pass

    def planned_test_hot_start_acquisition_from_thread_produces_data(self):
        pass

    def planned_test_hot_start_acquisition_from_thread_with_custom_parameters_produces_data(self):
        pass

    def planned_test_record_mode_displays_acquisition_and_leaves_it_displayed(self):
        pass


if __name__ == '__main__':
    unittest.main()
