import contextlib
import copy
import json
import os
import pathlib
import random
import shutil
import threading
import time
import typing
import unittest
import uuid

import numpy

from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import HardwareSource_test
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import ImportExportManager
from nion.swift.model import Metadata
from nion.swift.test import TestContext
from nion.ui import TestUI
from nion.utils import Geometry
from nion.utils import Registry
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
        TestContext.begin_leaks()
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)

    def tearDown(self) -> None:
        self.assertEqual(0, len(Registry.get_components_by_type("hardware_source_manager")))
        self.assertEqual(0, len(Registry.get_components_by_type("stem_controller")))
        self.assertEqual(0, len(Registry.get_components_by_type("scan_device")))
        self.assertEqual(0, len(Registry.get_components_by_type("scan_hardware_source")))
        self.assertEqual(0, len(Registry.get_components_by_type("hardware_source")))
        self.assertEqual(0, len(Registry.get_components_by_type("document_model")))
        self.assertEqual(0, stem_controller.ScanContextController.count)
        self.assertEqual(0, stem_controller.ProbeView.count)
        self.assertEqual(0, stem_controller.SubscanView.count)
        self.assertEqual(0, stem_controller.LineScanView.count)
        self.assertEqual(0, stem_controller.DriftView.count)
        TestContext.end_leaks(self)

    def _acquire_one(self, document_controller, hardware_source):
        hardware_source.start_playing(sync_timeout=3.0)
        hardware_source.stop_playing(sync_timeout=3.0)
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

    def __create_state_controller(self, acquisition_test_context: AcquisitionTestContext.AcquisitionTestContext,
                                  channel_id: typing.Optional[str] = None) -> ScanControlPanel.ScanControlStateController:
        state_controller = ScanControlPanel.ScanControlStateController(acquisition_test_context.scan_hardware_source,
                                                                       acquisition_test_context.document_controller.queue_task,
                                                                       acquisition_test_context.document_model,
                                                                       channel_id)
        state_controller.initialize_state()
        acquisition_test_context.push(state_controller)
        return state_controller

    def __test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.test_context()

    def _test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        return self.__test_context()

    ## STANDARD ACQUISITION TESTS ##

    # Do not change the comment above as it is used to search for places needing updates when a new
    # standard acquisition test is added.

    def test_acquiring_frames_with_generator_produces_correct_frame_numbers(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_acquiring_frames_with_generator_produces_correct_frame_numbers(self, scan_hardware_source, document_controller)

    def test_acquire_multiple_frames_reuses_same_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_acquire_multiple_frames_reuses_same_data_item(self, scan_hardware_source, document_controller)

    def test_simple_hardware_start_and_stop_actually_stops_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_stop_actually_stops_acquisition(self, scan_hardware_source, document_controller)

    def test_simple_hardware_start_and_abort_works_as_expected(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_abort_works_as_expected(self, scan_hardware_source, document_controller)

    def test_record_only_acquires_one_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_record_only_acquires_one_item(self, scan_hardware_source, document_controller)

    def test_record_during_view_records_one_item_and_keeps_viewing(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_record_during_view_records_one_item_and_keeps_viewing(self, scan_hardware_source, document_controller)

    def test_abort_record_during_view_returns_to_view(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_abort_record_during_view_returns_to_view(self, scan_hardware_source, document_controller)

    def test_view_reuses_single_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_view_reuses_single_data_item(self, scan_hardware_source, document_controller)

    def test_get_next_data_elements_to_finish_returns_full_frames(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_get_next_data_elements_to_finish_returns_full_frames(self, scan_hardware_source, document_controller)

    def test_exception_during_view_halts_scan(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_exception_during_view_halts_playback(self, scan_hardware_source, scan_hardware_source.get_current_frame_time())

    def test_exception_during_record_halts_scan(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_selected_profile_index(2)
            HardwareSource_test._test_exception_during_record_halts_playback(self, scan_hardware_source, scan_hardware_source.get_current_frame_time())

    def test_able_to_restart_scan_after_exception_scan(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_able_to_restart_view_after_exception(self, scan_hardware_source, scan_hardware_source.get_current_frame_time())

    def test_record_starts_and_finishes_in_reasonable_time(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_record_starts_and_finishes_in_reasonable_time(self, scan_hardware_source, scan_hardware_source.get_current_frame_time() * 16)

    # End of standard acquisition tests.

    def test_record_followed_by_view_updates_display(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context, "a")
            displayed_data_item = [None]
            def display_data_item_changed():
                displayed_data_item[0] = scan_state_controller.data_item_reference.data_item
            data_item_reference_changed_listener = scan_state_controller.data_item_reference.data_item_reference_changed_event.listen(display_data_item_changed)
            self._record_one(document_controller, scan_hardware_source, scan_state_controller)
            document_controller.periodic()  # extra to handle binding
            self.assertEqual(document_model.data_items[0], displayed_data_item[0])
            self.assertTrue(numpy.array_equal(document_model.data_items[0].data, document_model.data_items[1].data))
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()  # extra to handle binding
            self.assertEqual(document_model.data_items[0], displayed_data_item[0])
            self.assertFalse(numpy.array_equal(document_model.data_items[0].data, document_model.data_items[1].data))
            # clean up
            scan_hardware_source.abort_playing()
            data_item_reference_changed_listener.close()

    def test_record_gets_correct_data_when_record_started_during_view(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            frame_parameters = scan_hardware_source.get_frame_parameters(0)
            frame_parameters.pixel_time_us = 1
            scan_hardware_source.set_frame_parameters(0, frame_parameters)
            frame_time = scan_hardware_source.get_current_frame_time()
            scan_hardware_source.start_playing()
            time.sleep(frame_time * 0.25)
            self._record_one(document_controller, scan_hardware_source, scan_state_controller)
            self.assertEqual(len(document_model.data_items), 2)  # check assumptions
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(document_model.data_items[1].dimensional_shape, scan_hardware_source.get_frame_parameters(2).size)
            self.assertNotEqual(document_model.data_items[0].dimensional_shape, document_model.data_items[1].dimensional_shape)

    def test_record_names_results_correctly(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            self._record_one(document_controller, scan_hardware_source, scan_state_controller)
            self.assertTrue(document_model.data_items[2].title.startswith(document_model.data_items[0].title))
            self.assertTrue(len(document_model.data_items[2].title) > len(document_model.data_items[0].title))
            self.assertTrue(document_model.data_items[3].title.startswith(document_model.data_items[1].title))
            self.assertTrue(len(document_model.data_items[3].title) > len(document_model.data_items[1].title))

    def test_enable_positioned_after_one_frame_acquisition_should_add_graphic(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 0)
            scan_state_controller.handle_positioned_check_box(True)
            document_controller.periodic()
            self.assertEqual(len(display_item.graphics), 1)

    def test_disable_positioned_after_one_frame_acquisition_should_remove_graphic(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 1)
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            self.assertEqual(len(display_item.graphics), 0)

    def test_deleting_probe_graphic_after_one_frame_acquisition_should_disable_positioned(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            display_item.remove_graphic(probe_graphic).close()
            self.assertEqual(len(display_item.graphics), 0)
            self.assertIsNone(scan_hardware_source.probe_position)

    def test_deleting_display_after_one_frame_acquisition_should_disable_positioned(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            data_item = document_model.data_items[0]
            document_model.remove_data_item(data_item)
            self.assertEqual(len(document_model.data_items), 0)
            self.assertIsNone(scan_hardware_source.probe_position)

    # TODO: test case where one of two displays is deleted, what happens to probe graphic on other one?
    # TODO: test case where probe graphic should be removed when a channel is disabled

    def test_start_playing_after_stopping_should_remove_probe_position_graphic(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 1)
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            self.assertEqual(len(display_item.graphics), 0)
            scan_hardware_source.stop_playing()

    def test_context_scan_attaches_required_metadata(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(2, True)
            self._acquire_one(document_controller, scan_hardware_source)
            metadata_source0 = document_model.data_items[0]
            metadata_source1 = document_model.data_items[1]
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            for metadata_source, channel_index in zip([metadata_source0, metadata_source1], [0, 2]):
                # import pprint; print(pprint.pformat(metadata_source.metadata))
                # note: frame_time and line_time_us do not currently handle flyback - so this is intentionally wrong
                self.assertEqual(scan_hardware_source.hardware_source_id, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.id"))
                self.assertEqual(scan_hardware_source.display_name, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.name"))
                self.assertNotIn("autostem", metadata_source.metadata["hardware_source"])
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("EHT"), Metadata.get_metadata_value(metadata_source, "stem.high_tension"))
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("C10"), Metadata.get_metadata_value(metadata_source, "stem.defocus"))
                self.assertEqual(channel_index, Metadata.get_metadata_value(metadata_source, "stem.scan.channel_index"))
                self.assertEqual(scan_hardware_source.get_channel_state(channel_index).channel_id, Metadata.get_metadata_value(metadata_source, "stem.scan.channel_id"))
                self.assertEqual(scan_hardware_source.get_channel_state(channel_index).name, Metadata.get_metadata_value(metadata_source, "stem.scan.channel_name"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.center_x_nm"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.center_x_nm"))
                self.assertAlmostEqual(frame_parameters.size[0] * frame_parameters.size[1] * frame_parameters.pixel_time_us / 1E6, Metadata.get_metadata_value(metadata_source, "stem.scan.frame_time"))
                self.assertEqual(100.0, Metadata.get_metadata_value(metadata_source, "stem.scan.fov_nm"))
                self.assertEqual(0, Metadata.get_metadata_value(metadata_source, "stem.scan.frame_index"))
                self.assertAlmostEqual(frame_parameters.pixel_time_us, Metadata.get_metadata_value(metadata_source, "stem.scan.pixel_time_us"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.rotation"))
                self.assertIsNotNone(uuid.UUID(Metadata.get_metadata_value(metadata_source, "stem.scan.scan_id")))
                self.assertEqual(frame_parameters.size[1], Metadata.get_metadata_value(metadata_source, "stem.hardware_source.valid_rows"))
                self.assertEqual(frame_parameters.size[1], Metadata.get_metadata_value(metadata_source, "stem.scan.valid_rows"))
                self.assertAlmostEqual(frame_parameters.size[0] * frame_parameters.pixel_time_us, Metadata.get_metadata_value(metadata_source, "stem.scan.line_time_us"))
                self.assertEqual((256, 256), metadata_source.metadata["scan"]["scan_context_size"])
                self.assertEqual((256, 256), metadata_source.metadata["scan"]["scan_size"])

    def test_sub_scan_attaches_required_metadata(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(2, True)
            scan_hardware_source.subscan_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            metadata_source0 = document_model.data_items[0]
            metadata_source1 = document_model.data_items[1]
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            for metadata_source, channel_index in zip([metadata_source0, metadata_source1], [4, 6]):
                # import pprint; print(pprint.pformat(metadata_source.metadata))
                # note: frame_time and line_time_us do not currently handle flyback - so this is intentionally wrong
                self.assertEqual(scan_hardware_source.hardware_source_id, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.id"))
                self.assertEqual(scan_hardware_source.display_name, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.name"))
                self.assertNotIn("autostem", metadata_source.metadata["hardware_source"])
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("EHT"), Metadata.get_metadata_value(metadata_source, "stem.high_tension"))
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("C10"), Metadata.get_metadata_value(metadata_source, "stem.defocus"))
                self.assertEqual(channel_index, Metadata.get_metadata_value(metadata_source, "stem.scan.channel_index"))
                self.assertEqual(scan_hardware_source.get_channel_state(channel_index % 4).channel_id + "_subscan", Metadata.get_metadata_value(metadata_source, "stem.scan.channel_id"))
                self.assertEqual(scan_hardware_source.get_channel_state(channel_index % 4).name + " SubScan", Metadata.get_metadata_value(metadata_source, "stem.scan.channel_name"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.center_x_nm"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.center_x_nm"))
                self.assertAlmostEqual(frame_parameters.size[0] * frame_parameters.subscan_fractional_size[0] * frame_parameters.size[1] * frame_parameters.subscan_fractional_size[1] * frame_parameters.pixel_time_us / 1E6, Metadata.get_metadata_value(metadata_source, "stem.scan.frame_time"))
                self.assertEqual(100.0, Metadata.get_metadata_value(metadata_source, "stem.scan.fov_nm"))
                self.assertEqual(0, Metadata.get_metadata_value(metadata_source, "stem.scan.frame_index"))
                self.assertAlmostEqual(frame_parameters.pixel_time_us, Metadata.get_metadata_value(metadata_source, "stem.scan.pixel_time_us"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.rotation"))
                self.assertIsNotNone(uuid.UUID(Metadata.get_metadata_value(metadata_source, "stem.scan.scan_id")))
                self.assertEqual(frame_parameters.size[1] * frame_parameters.subscan_fractional_size[1], Metadata.get_metadata_value(metadata_source, "stem.hardware_source.valid_rows"))
                self.assertEqual(frame_parameters.size[1] * frame_parameters.subscan_fractional_size[1], Metadata.get_metadata_value(metadata_source, "stem.scan.valid_rows"))
                self.assertAlmostEqual(frame_parameters.size[0] * frame_parameters.subscan_fractional_size[0] * frame_parameters.pixel_time_us, Metadata.get_metadata_value(metadata_source, "stem.scan.line_time_us"))
                self.assertEqual((256, 256), metadata_source.metadata["scan"]["scan_context_size"])
                self.assertEqual((128, 128), metadata_source.metadata["scan"]["scan_size"])

    def test_acquiring_multiple_channels_attaches_common_scan_id(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(2, True)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(len(document_model.data_items), 2)
            scan_id0 = Metadata.get_metadata_value(document_model.data_items[0], "stem.scan.scan_id")
            scan_id1 = Metadata.get_metadata_value(document_model.data_items[1], "stem.scan.scan_id")
            self.assertIsNotNone(scan_id0)
            self.assertEqual(scan_id0, scan_id1)

    def test_acquiring_multiple_channels_twice_generates_different_scan_ids(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(2, True)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(len(document_model.data_items), 2)
            scan_id00 = Metadata.get_metadata_value(document_model.data_items[0], "stem.scan.scan_id")
            scan_id01 = Metadata.get_metadata_value(document_model.data_items[1], "stem.scan.scan_id")
            self._acquire_one(document_controller, scan_hardware_source)
            scan_id10 = Metadata.get_metadata_value(document_model.data_items[0], "stem.scan.scan_id")
            scan_id11 = Metadata.get_metadata_value(document_model.data_items[1], "stem.scan.scan_id")
            self.assertIsNotNone(scan_id00)
            self.assertIsNotNone(scan_id10)
            self.assertEqual(scan_id00, scan_id01)
            self.assertEqual(scan_id10, scan_id11)

    def test_start_playing_with_no_channels_enabled_does_nothing(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, False)
            frame_time = scan_hardware_source.get_current_frame_time()
            scan_hardware_source.start_playing()
            start = time.time()
            while scan_hardware_source.is_playing and time.time() - start < 3.0:
                time.sleep(frame_time * 0.2)
            is_playing = scan_hardware_source.is_playing
            scan_hardware_source.stop_playing()
            self.assertFalse(is_playing)

    def test_enabling_channel_during_acquisition_results_in_write_delayed_data_item(self):
        # this ensures that data items freshly created during acquisition don't write repeatedly to disk.
        # see also test_data_item_created_during_acquisition_is_write_delayed_during_and_not_after
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, True)
            frame_time = scan_hardware_source.get_current_frame_time()
            scan_hardware_source.start_playing()
            try:
                time.sleep(frame_time * 2.1)
                test_context.document_controller.periodic()
                self.assertEqual(1, len(test_context.document_model.data_items))
                self.assertTrue(test_context.document_model.data_items[0].is_write_delayed)
                scan_hardware_source.set_channel_enabled(1, True)
                scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
                test_context.document_controller.periodic()
                self.assertEqual(2, len(test_context.document_model.data_items))
                self.assertTrue(test_context.document_model.data_items[0].is_write_delayed)
                self.assertTrue(test_context.document_model.data_items[1].is_write_delayed)
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
                test_context.document_controller.periodic()
            self.assertFalse(test_context.document_model.data_items[0].is_write_delayed)
            self.assertFalse(test_context.document_model.data_items[1].is_write_delayed)

    def test_disabling_all_channels_during_play_stops_playback(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)  # throw out first scan, which can take longer due to initial conditions
            frame_time = scan_hardware_source.get_current_frame_time()
            scan_hardware_source.start_playing()
            time.sleep(frame_time * 0.25)
            was_playing = scan_hardware_source.is_playing
            scan_hardware_source.set_channel_enabled(0, False)
            time.sleep(frame_time * 0.25)  # this sleep is not really necessary but leave it here for extra robustness
            scan_hardware_source.periodic()
            time.sleep(frame_time * 2)  # scan should definitely be stopped after 4 frames
            # updating is_playing may take some time; sync here
            start = time.time()
            while time.time() - start < 3.0 and scan_hardware_source.is_playing:
                time.sleep(0.01)
            is_playing = scan_hardware_source.is_playing
            scan_hardware_source.stop_playing()
            self.assertTrue(was_playing)
            self.assertFalse(is_playing)

    def test_scan_and_record_button_disabled_when_no_channels_are_enabled(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_enabled_ref = [True]
            def scan_button_state_changed(enabled, play_button_state):
                scan_enabled_ref[0] = enabled

            record_enabled_ref = [True]
            def record_button_state_changed(visible, enabled):
                record_enabled_ref[0] = enabled

            scan_state_controller.on_scan_button_state_changed = scan_button_state_changed
            scan_state_controller.on_record_button_state_changed = record_button_state_changed
            scan_hardware_source.set_channel_enabled(0, False)
            scan_hardware_source.periodic()
            self.assertFalse(scan_enabled_ref[0])
            self.assertFalse(record_enabled_ref[0])
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.periodic()
            self.assertTrue(scan_enabled_ref[0])
            self.assertTrue(record_enabled_ref[0])

    def test_record_generates_a_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            for i in range(3):
                self._record_one(document_controller, scan_hardware_source, scan_state_controller)
            self.assertEqual(len(document_model.data_items), 4)  # 1 view image (always acquired), 3 records

    def test_ability_to_set_profile_parameters_is_reflected_in_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(300, 300)
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = scan_hardware_source.get_frame_parameters(1)
            frame_parameters_1.size = Geometry.IntSize(500, 500)
            scan_hardware_source.set_frame_parameters(1, frame_parameters_1)
            scan_hardware_source.set_selected_profile_index(0)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (300, 300))
            scan_hardware_source.set_selected_profile_index(1)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (500, 500))

    def test_change_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(300, 300)
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = scan_hardware_source.get_frame_parameters(1)
            frame_parameters_1.size = Geometry.IntSize(500, 500)
            scan_hardware_source.set_frame_parameters(1, frame_parameters_1)
            scan_hardware_source.set_selected_profile_index(0)
            scan_hardware_source.start_playing()
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (300, 300))
            scan_hardware_source.set_selected_profile_index(1)
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (500, 500))
            scan_hardware_source.abort_playing()

    def test_mutating_current_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(300, 300)
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = scan_hardware_source.get_frame_parameters(1)
            frame_parameters_1.size = Geometry.IntSize(500, 500)
            scan_hardware_source.set_frame_parameters(1, frame_parameters_1)
            scan_hardware_source.set_selected_profile_index(0)
            scan_hardware_source.start_playing()
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (300, 300))
            scan_hardware_source.set_frame_parameters(0, frame_parameters_1)
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (500, 500))
            scan_hardware_source.abort_playing()

    def test_changing_frame_parameters_during_record_does_not_affect_recording(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            # verify that the frame parameters are applied to the _next_ frame.
            frame_parameters_2 = scan_hardware_source.get_frame_parameters(2)
            frame_parameters_2.size = Geometry.IntSize(500, 500)
            frame_time = frame_parameters_2.pixel_time_us * 500 * 500 / 1000000.0
            scan_hardware_source.start_recording()
            time.sleep(frame_time * 0.6)
            scan_hardware_source.set_frame_parameters(2, frame_parameters_2)
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_finish(10.0)[0].data.shape, (1024, 1024))
            start_time = time.time()
            while scan_hardware_source.is_recording:
                time.sleep(frame_time)
                self.assertTrue(time.time() - start_time < 3.0)
            # confirm that recording is done, then verify that the frame parameters are actually applied to the _next_ frame.
            scan_hardware_source.start_recording()
            time.sleep(frame_time * 0.6)
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_finish(10.0)[0].data.shape, (500, 500))

    def test_capturing_during_view_captures_new_data_items(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)

            def display_new_data_item(data_item):
                document_model.append_data_item(data_item)

            scan_state_controller.on_display_new_data_item = display_new_data_item
            scan_state_controller.handle_capture_clicked()
            scan_hardware_source.stop_playing()
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 4)

    def test_ability_to_start_playing_with_custom_parameters(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(200, 200)
            scan_hardware_source.set_current_frame_parameters(frame_parameters_0)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()
            scan_hardware_source.stop_playing()
            #hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)
            self.assertEqual(document_model.data_items[0].dimensional_shape, (200, 200))
            self.assertEqual(document_model.data_items[1].dimensional_shape, (200, 200))

    def test_ability_to_start_recording_with_custom_parameters(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(200, 200)
            scan_hardware_source.set_record_frame_parameters(frame_parameters_0)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_hardware_source.start_recording()
            recorded_extended_data_list = scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)  # 2 view images (always grabbed)
            self.assertEqual(len(recorded_extended_data_list), 2)  # 2 record images
            self.assertEqual(document_model.data_items[0].dimensional_shape, (200, 200))
            self.assertEqual(document_model.data_items[1].dimensional_shape, (200, 200))
            self.assertEqual(recorded_extended_data_list[0].data.shape, (200, 200))
            self.assertEqual(recorded_extended_data_list[1].data.shape, (200, 200))

    def test_changing_profile_updates_frame_parameters_in_ui(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            scan_state_controller.on_frame_parameters_changed = frame_parameters_changed
            scan_hardware_source.set_selected_profile_index(1)
            self.assertIsNotNone(frame_parameters_ref[0])

    def test_changing_current_profiles_frame_parameters_updates_frame_parameters_in_ui(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            scan_state_controller.on_frame_parameters_changed = frame_parameters_changed
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(200, 200)
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            self.assertIsNotNone(frame_parameters_ref[0])
            self.assertEqual(frame_parameters_ref[0].size, (200, 200))

    def test_changing_fov_updates_only_that_profile(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_selected_profile_index(0)
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 10
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            self.assertEqual(scan_hardware_source.get_frame_parameters(0).fov_nm, 10)
            scan_hardware_source.set_selected_profile_index(1)
            frame_parameters_1 = scan_hardware_source.get_frame_parameters(1)
            frame_parameters_1.fov_nm = 20
            scan_hardware_source.set_frame_parameters(1, frame_parameters_1)
            self.assertEqual(scan_hardware_source.get_frame_parameters(1).fov_nm, 20)
            scan_hardware_source.set_selected_profile_index(2)
            frame_parameters_2 = scan_hardware_source.get_frame_parameters(2)
            frame_parameters_2.fov_nm = 30
            scan_hardware_source.set_frame_parameters(2, frame_parameters_2)
            self.assertEqual(scan_hardware_source.get_frame_parameters(2).fov_nm, 30)
            scan_hardware_source.set_selected_profile_index(0)
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 1
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = scan_hardware_source.get_frame_parameters(1)
            self.assertEqual(frame_parameters_1.fov_nm, 20)
            frame_parameters_2 = scan_hardware_source.get_frame_parameters(2)
            self.assertEqual(frame_parameters_2.fov_nm, 30)

    def test_changing_size_is_reflected_in_new_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_state_controller.handle_linked_changed(False)
            scan_state_controller.handle_decrease_width()
            scan_state_controller.handle_decrease_height()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (128, 128))

    def test_record_during_view_uses_record_mode(self):
        # NOTE: requires stages_per_frame == 2
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 10
            frame_parameters_0.pixel_time_us = 10
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_2 = scan_hardware_source.get_frame_parameters(2)
            frame_parameters_2.fov_nm = 100
            scan_hardware_source.set_frame_parameters(2, frame_parameters_2)
            frame_time = scan_hardware_source.get_current_frame_time()
            scan_hardware_source.start_playing()
            self.assertEqual(Metadata.get_metadata_value(scan_hardware_source.get_next_xdatas_to_finish()[0], "stem.scan.fov_nm"), 10)
            # give view a chance to start before recording. this must finished _before_ the final segment of partial
            # acquisition for this test to work; otherwise the final segment finishes and notifies and gets picked up
            # in the first wait call after start_recording below.
            # Also, time of partial acquisition  must be less than 50% of the frame time
            time.sleep(frame_time * 0.1)
            scan_hardware_source.start_recording()
            time.sleep(frame_time * 0.1)  # give record a chance to start; starting record will abort view immediately
            self.assertEqual(Metadata.get_metadata_value(scan_hardware_source.get_next_xdatas_to_finish()[0], "stem.scan.fov_nm"), 100)
            self.assertEqual(Metadata.get_metadata_value(scan_hardware_source.get_next_xdatas_to_finish()[0], "stem.scan.fov_nm"), 10)

    def test_consecutive_frames_have_unique_data(self):
        # this test will fail if the scan is saturated (or otherwise produces identical values naturally)
        numpy_random_state = numpy.random.get_state()
        random_state = random.getstate()
        numpy.random.seed(999)
        random.seed(999)
        try:
            with self.__test_context() as test_context:
                scan_hardware_source = test_context.scan_hardware_source
                frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
                frame_parameters_0.size = Geometry.IntSize(256, 256)
                frame_parameters_0.pixel_time_us = 2
                scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
                scan_hardware_source.start_playing()
                data_list = list()
                for i in range(16):
                    data = scan_hardware_source.get_next_xdatas_to_finish()[0].data
                    data_list.append(data)
                for row in range(0, 256, 32):
                    s = slice(row, row+32), slice(0, 256)
                    for i, data in enumerate(data_list[1:]):
                        self.assertFalse(numpy.array_equal(data_list[0][s], data[s]))
        finally:
            random.setstate(random_state)
            numpy.random.set_state(numpy_random_state)

    def test_frame_do_not_change_after_acquisition(self):
        # this test will fail if the scan is saturated (or otherwise produces identical values naturally)
        numpy_random_state = numpy.random.get_state()
        random_state = random.getstate()
        numpy.random.seed(999)
        random.seed(999)
        try:
            with self.__test_context() as test_context:
                scan_hardware_source = test_context.scan_hardware_source
                frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
                frame_parameters_0.size = Geometry.IntSize(256, 256)
                frame_parameters_0.pixel_time_us = 2
                scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
                scan_hardware_source.start_playing()
                data_list = list()
                checksums = list()
                for i in range(4):
                    data = scan_hardware_source.get_next_xdatas_to_finish()[0].data
                    data_list.append(data)
                    checksums.append(numpy.sum(data))
                for checksum, data in zip(checksums, data_list):
                    self.assertEqual(checksum, numpy.sum(data))
        finally:
            random.setstate(random_state)
            numpy.random.set_state(numpy_random_state)

    def test_changing_width_when_linked_changes_height_too(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_state_controller.handle_decrease_width()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (128, 128))

    def test_changing_width_when_unlinked_does_not_change_height(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_state_controller.handle_linked_changed(False)
            scan_state_controller.handle_decrease_width()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 128))

    def test_obscenely_large_scan_does_not_crash(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(32768, 32768)
            frame_parameters_0.pixel_time_us = 1
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            scan_hardware_source.start_playing()
            scan_hardware_source.abort_playing()

    def test_big_scan_does_not_prevent_further_playing(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            original_size = frame_parameters_0.size
            original_pixel_time_us = frame_parameters_0.pixel_time_us
            frame_parameters_0.size = Geometry.IntSize(8192, 8192)
            frame_parameters_0.pixel_time_us = 1
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            scan_hardware_source.abort_playing(sync_timeout=3.0)
            frame_parameters_0.size = original_size
            frame_parameters_0.pixel_time_us = original_pixel_time_us
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            scan_hardware_source.abort_playing(sync_timeout=3.0)

    def test_closing_display_panel_with_scan_controller_shuts_down_controller_correctly(self):
        # NOTE: this is a duplicate of test_closing_display_panel_with_display_controller_shuts_down_controller_correctly
        with self.__test_context() as test_context:
            ScanControlPanel.run()
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            d = {"type": "splitter", "orientation": "vertical", "splits": [0.5, 0.5], "children": [
                {"type": "image", "uuid": "0569ca31-afd7-48bd-ad54-5e2bb9f21102", "identifier": "a", "selected": True},
                {"type": "image", "uuid": "acd77f9f-2f6f-4fbf-af5e-94330b73b997", "identifier": "b"}]}
            workspace_2x1 = document_controller.workspace_controller.new_workspace("2x1", d)
            document_controller.workspace_controller.change_workspace(workspace_2x1)
            root_canvas_item = document_controller.workspace_controller.image_row.children[0]._root_canvas_item()
            root_canvas_item.update_layout(Geometry.IntPoint(), Geometry.IntSize(width=640, height=480))
            scan_hardware_source.start_playing()
            try:
                frame_time = scan_hardware_source.get_current_frame_time()
                time.sleep(frame_time * 2)  # let view start
                d = {"type": "image", "controller_type": "scan-live", "hardware_source_id": scan_hardware_source.hardware_source_id, "channel_id": "a"}
                display_panel = document_controller.selected_display_panel
                display_panel.change_display_panel_content(d)
                time.sleep(frame_time * 2)  # let view start
                document_controller.periodic()
                document_controller.workspace_controller.remove_display_panel(display_panel)
                time.sleep(frame_time * 2)  # let view start
                document_controller.periodic()
            finally:
                scan_hardware_source.abort_playing()
            ScanControlPanel.stop()

    def test_probe_graphic_gets_closed(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            self.assertFalse(probe_graphic._closed)
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            self.assertIsNone(scan_hardware_source.probe_position)
            self.assertTrue(probe_graphic._closed)

    def test_probe_appears_on_all_channels(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            display_item0 = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item1 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertIsNotNone(display_item0.graphics[0])
            self.assertIsNotNone(display_item1.graphics[0])
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            self.assertIsNone(scan_hardware_source.probe_position)
            self.assertEqual(len(display_item0.graphics), 0)
            self.assertEqual(len(display_item1.graphics), 0)

    def test_probe_on_multiple_channels_shuts_down_properly(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.graphics[0].position = 0.3, 0.4
            document_controller.periodic()
            # will output extraneous messages when it fails

    def test_probe_on_multiple_channels_deletes_properly(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            display_item.remove_graphic(probe_graphic).close()
            document_controller.periodic()
            display_item0 = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item1 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(len(display_item0.graphics), 0)
            self.assertEqual(len(display_item1.graphics), 0)
            self.assertIsNone(scan_hardware_source.probe_position)
            # will output extraneous messages when it fails

    def test_probe_on_multiple_channels_sets_to_none_properly(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            scan_state_controller.handle_positioned_check_box(False)
            document_controller.periodic()
            display_item0 = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item1 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(len(display_item0.graphics), 0)
            self.assertEqual(len(display_item1.graphics), 0)
            self.assertIsNone(scan_hardware_source.probe_position)

    def test_setting_probe_position_updates_probe_graphic(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.45, x=0.65))
            document_controller.periodic()
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)

    def test_setting_probe_graphic_updates_probe_position(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            probe_graphic.position = Geometry.FloatPoint(y=0.45, x=0.65)
            document_controller.periodic()
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            self.assertEqual(stem_controller_.probe_position, scan_hardware_source._get_last_idle_position_for_test())

    def test_setting_probe_graphic_updates_probe_position_with_multiple_channels_enabled(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            probe_graphic.position = Geometry.FloatPoint(y=0.45, x=0.65)
            document_controller.periodic()
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            self.assertEqual(stem_controller_.probe_position, scan_hardware_source._get_last_idle_position_for_test())

    def test_acquire_into_empty_scan_controlled_display_panel(self):
        with self.__test_context() as test_context:
            ScanControlPanel.run()
            document_controller = test_context.document_controller
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            display_panel = document_controller.selected_display_panel
            display_panel.change_display_panel_content({"controller_type": "scan-live", "hardware_source_id": scan_hardware_source.hardware_source_id, "channel_id": scan_hardware_source.get_channel_state(0).channel_id})
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_state_controller.handle_positioned_check_box(True)
            self._acquire_one(document_controller, scan_hardware_source)
            ScanControlPanel.stop()

    def test_subscan_state_goes_from_invalid_to_disabled_upon_first_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            self.assertEqual(stem_controller.SubscanState.INVALID, scan_hardware_source.subscan_state)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(stem_controller.SubscanState.DISABLED, scan_hardware_source.subscan_state)

    def test_enabling_subscan_during_initial_acquisition_puts_graphic_on_context(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(1, len(document_model.data_items))
            self.assertEqual(0, len(display_item.graphics))
            scan_hardware_source.validate_probe_position()  # enabled probe position
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            display_item1 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(2, len(document_model.data_items))
            self.assertEqual(2, len(display_item.graphics))  # subscan and position
            self.assertEqual(0, len(display_item1.graphics))

    def test_removing_subscan_graphic_disables_subscan_when_acquisition_running(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 0)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            display_item.remove_graphic(display_item.graphics[0]).close()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            scan_hardware_source.stop_playing()
            self.assertFalse(scan_hardware_source.subscan_enabled)

    def test_removing_line_scan_graphic_disables_line_scan_when_acquisition_running(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 0)
            scan_hardware_source.line_scan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            display_item.remove_graphic(display_item.graphics[0]).close()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            scan_hardware_source.stop_playing()
            self.assertFalse(scan_hardware_source.line_scan_enabled)

    def test_removing_subscan_graphic_disables_subscan(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_state_controller.handle_subscan_enabled(True)
            document_controller.periodic()
            self.assertTrue(scan_hardware_source.subscan_enabled)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.remove_graphic(display_item.graphics[0]).close()
            document_controller.periodic()
            self.assertFalse(scan_hardware_source.subscan_enabled)

    def test_subscan_not_allowed_with_width_or_height_of_zero(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_state_controller.handle_subscan_enabled(True)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.graphics[0].size = 0, 0.25
            document_controller.periodic()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual((1, 64), document_model.data_items[1].data_shape)

    def test_scan_context_updated_when_starting_playing(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            self.assertFalse(stem_controller_.scan_context.is_valid)
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            self.assertTrue(stem_controller_.scan_context.is_valid)
            scan_hardware_source.stop_playing()
            self.assertTrue(stem_controller_.scan_context.is_valid)

    def test_scan_context_updated_when_changing_parameters(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context1 = copy.deepcopy(stem_controller_.scan_context)
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 20
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            scan_hardware_source.stop_playing()
            scan_context2 = copy.deepcopy(stem_controller_.scan_context)
            self.assertEqual(20, scan_context2.fov_nm)
            self.assertNotEqual(20, scan_context1.fov_nm)

    def test_scan_context_update_after_changing_parameters_during_subscan_and_disabling(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context1 = copy.deepcopy(stem_controller_.scan_context)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context2 = copy.deepcopy(stem_controller_.scan_context)
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 20
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context3 = copy.deepcopy(stem_controller_.scan_context)
            scan_hardware_source.subscan_enabled = False
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context4 = copy.deepcopy(stem_controller_.scan_context)
            scan_hardware_source.stop_playing()
            self.assertNotEqual(20, scan_context1.fov_nm)
            self.assertNotEqual(20, scan_context2.fov_nm)
            self.assertFalse(scan_context3.is_valid)
            self.assertEqual(20, scan_context4.fov_nm)

    def test_scan_context_cleared_after_changing_parameters_during_subscan_and_stopping(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context1 = copy.deepcopy(stem_controller_.scan_context)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context2 = copy.deepcopy(stem_controller_.scan_context)
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 20
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            scan_context3 = copy.deepcopy(stem_controller_.scan_context)
            scan_hardware_source.stop_playing()
            scan_context4 = copy.deepcopy(stem_controller_.scan_context)
            self.assertNotEqual(20, scan_context1.fov_nm)
            self.assertNotEqual(20, scan_context2.fov_nm)
            self.assertFalse(scan_context3.is_valid)
            self.assertFalse(scan_context4.is_valid)

    def test_scan_context_not_cleared_after_stopping_subscan(self):
        # this tests a failure mode where the current profile is updated during acquisition on
        # a signal from the device.
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
            scan_hardware_source._update_frame_parameters_test(0, scan_hardware_source.get_current_frame_parameters())
            scan_context1 = copy.deepcopy(stem_controller_.scan_context)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
            scan_hardware_source._update_frame_parameters_test(0, scan_hardware_source.get_current_frame_parameters())
            scan_hardware_source.stop_playing()
            scan_context2 = copy.deepcopy(stem_controller_.scan_context)
            self.assertTrue(scan_context1.is_valid)
            self.assertTrue(scan_context2.is_valid)

    def test_changing_rotation_and_fov_update_device_parameters_immediately(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertTrue(stem_controller_.scan_context.is_valid)
            frame_parameters = copy.deepcopy(scan_hardware_source.scan_device.current_frame_parameters)
            fov_nm = frame_parameters.fov_nm
            frame_parameters.fov_nm //= 2
            scan_hardware_source.set_frame_parameters(0, frame_parameters)
            scan_hardware_source._update_frame_parameters_test(0, scan_hardware_source.get_current_frame_parameters())
            self.assertEqual(fov_nm // 2, scan_hardware_source.scan_device.current_frame_parameters.fov_nm)
            self.assertFalse(stem_controller_.scan_context.is_valid)

    def test_removing_subscan_graphic_disables_subscan_when_acquisition_stopped(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 0)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            scan_hardware_source.stop_playing()
            display_item.remove_graphic(display_item.graphics[0]).close()
            self.assertFalse(scan_hardware_source.subscan_enabled)

    def test_facade_record_data_with_immediate_close(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(scan_hardware_source.hardware_source_id, "~1.0")
            scan_frame_parameters = hardware_source_facade.get_frame_parameters_for_profile_by_index(2)
            scan_frame_parameters["external_clock_wait_time_ms"] = 20000 # int(camera_frame_parameters["exposure_ms"] * 1.5)
            scan_frame_parameters["external_clock_mode"] = 1
            scan_frame_parameters["ac_line_sync"] = False
            scan_frame_parameters["ac_frame_sync"] = False
            # this tests an issue for a race condition where thread for record task isn't started before the task
            # is canceled, resulting in the close waiting for the thread and the thread waiting for the acquire.
            # this reduces the problem, but it's still possible that during external sync, the acquisition starts
            # before being canceled and must timeout.
            with contextlib.closing(hardware_source_facade.create_record_task(scan_frame_parameters)) as task:
                pass

    def test_enabling_subscan_changes_output_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            hardware_source_id = scan_hardware_source.hardware_source_id
            self.assertEqual(len(document_model.data_items), 0)
            data_item = DataItem.DataItem()
            document_model.append_data_item(data_item)
            data_item2 = DataItem.DataItem()
            document_model.append_data_item(data_item2)
            document_model.setup_channel(document_model.make_data_item_reference_key(hardware_source_id, scan_hardware_source.get_channel_state(0).channel_id), data_item)
            document_model.setup_channel(document_model.make_data_item_reference_key(hardware_source_id, scan_hardware_source.get_channel_state(0).channel_id + "_subscan"), data_item2)
            scan_hardware_source.start_playing()
            try:
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                self.assertEqual(data_item.data.shape, (256, 256))
                self.assertIsNone(data_item2.data)
                # turn on subscan
                scan_hardware_source.subscan_enabled = True
                scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
                document_controller.periodic()
                # save modified times; only subscan should be modified
                modified = data_item.modified
                modified2 = data_item2.modified
                scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
                document_controller.periodic()
                self.assertEqual(modified, data_item.modified)
                self.assertLess(modified2, data_item2.modified)
                self.assertEqual(data_item.data.shape, (256, 256))
                self.assertEqual(data_item2.data.shape, (128, 128))
                # turn off subscan
                scan_hardware_source.subscan_enabled = False
                scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
                document_controller.periodic()
                # save modified times; only scan should be modified
                modified = data_item.modified
                modified2 = data_item2.modified
                scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
                document_controller.periodic()
                self.assertLess(modified, data_item.modified)
                self.assertEqual(modified2, data_item2.modified)
            finally:
                scan_hardware_source.abort_playing()

    def test_restarting_with_subscan_enabled_changes_correct_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            hardware_source_id = scan_hardware_source.hardware_source_id
            self.assertEqual(len(document_model.data_items), 0)
            data_item = DataItem.DataItem()
            document_model.append_data_item(data_item)
            data_item2 = DataItem.DataItem()
            document_model.append_data_item(data_item2)
            document_model.setup_channel(document_model.make_data_item_reference_key(hardware_source_id, scan_hardware_source.get_channel_state(0).channel_id), data_item)
            document_model.setup_channel(document_model.make_data_item_reference_key(hardware_source_id, scan_hardware_source.get_channel_state(0).channel_id + "_subscan"), data_item2)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                self.assertEqual(data_item.data.shape, (256, 256))
                self.assertIsNone(data_item2.data)
                # turn on subscan
                scan_hardware_source.subscan_enabled = True
                scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
                document_controller.periodic()
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(data_item.data.shape, (256, 256))
            self.assertEqual(data_item2.data.shape, (128, 128))
            # save modified times; only scan should be modified
            modified = data_item.modified
            modified2 = data_item2.modified
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(modified, data_item.modified)
            self.assertLess(modified2, data_item2.modified)

    def test_restarting_with_subscan_disabled_after_stopping_with_enabled_changes_correct_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            hardware_source_id = scan_hardware_source.hardware_source_id
            self.assertEqual(len(document_model.data_items), 0)
            data_item = DataItem.DataItem()
            document_model.append_data_item(data_item)
            data_item2 = DataItem.DataItem()
            document_model.append_data_item(data_item2)
            document_model.setup_channel(document_model.make_data_item_reference_key(hardware_source_id, scan_hardware_source.get_channel_state(0).channel_id), data_item)
            document_model.setup_channel(document_model.make_data_item_reference_key(hardware_source_id, scan_hardware_source.get_channel_state(0).channel_id + "_subscan"), data_item2)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
                self.assertEqual(data_item.data.shape, (256, 256))
                self.assertIsNone(data_item2.data)
                # turn on subscan
                scan_hardware_source.subscan_enabled = True
                scan_hardware_source.get_next_xdatas_to_start()  # grab at least one frame
                document_controller.periodic()
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(data_item.data.shape, (256, 256))
            self.assertEqual(data_item2.data.shape, (128, 128))
            # disable subscan
            scan_hardware_source.subscan_enabled = False
            # save modified times; only scan should be modified
            modified = data_item.modified
            modified2 = data_item2.modified
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertLess(modified, data_item.modified)
            self.assertEqual(modified2, data_item2.modified)

    def test_drift_rectangle_appears_when_setting_drift_region(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            data_item = document_model.data_items[0]
            display_item = document_model.get_display_item_for_data_item(data_item)
            stem_controller_ = typing.cast(stem_controller.STEMController, stem_controller_)
            stem_controller_.drift_channel_id = scan_hardware_source.get_channel_state(0).channel_id
            document_controller.periodic()
            self.assertEqual(0, len(display_item.graphics))
            stem_controller_.drift_region = Geometry.FloatRect.from_tlhw(0.2, 0.2, 0.4, 0.4)
            document_controller.periodic()
            self.assertEqual(1, len(display_item.graphics))

    def test_removing_drift_rectangle_disables_drift_correction(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            data_item = document_model.data_items[0]
            display_item = document_model.get_display_item_for_data_item(data_item)
            stem_controller_ = typing.cast(stem_controller.STEMController, stem_controller_)
            stem_controller_.drift_channel_id = scan_hardware_source.get_channel_state(0).channel_id
            stem_controller_.drift_region = Geometry.FloatRect.from_tlhw(0.2, 0.2, 0.4, 0.4)
            document_controller.periodic()
            display_item.remove_graphic(display_item.graphics[0]).close()
            self.assertIsNone(stem_controller_.drift_channel_id)
            self.assertIsNone(stem_controller_.drift_region)

    def test_drift_enabled_when_clicking_checkbox(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            stem_controller_ = typing.cast(stem_controller.STEMController, stem_controller_)
            self.assertIsNone(stem_controller_.drift_channel_id)
            self.assertIsNone(stem_controller_.drift_region)
            scan_state_controller.handle_drift_enabled(True)
            document_controller.periodic()
            self.assertIsNotNone(stem_controller_.drift_channel_id)
            self.assertIsNotNone(stem_controller_.drift_region)
            scan_state_controller.handle_drift_enabled(False)
            document_controller.periodic()
            self.assertIsNone(stem_controller_.drift_channel_id)
            self.assertIsNone(stem_controller_.drift_region)

    def test_subscan_has_proper_calibrations(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_state_controller.handle_subscan_enabled(True)
            document_controller.periodic()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].scale, document_model.data_items[1].dimensional_calibrations[1].scale)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].units, document_model.data_items[1].dimensional_calibrations[1].units)

    def test_record_immediate_has_proper_calibrations(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            frame_parameters = copy.copy(scan_hardware_source.get_current_frame_parameters())
            frame_parameters.subscan_pixel_size = Geometry.IntSize(128, 128)
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(0.5, 0.5)
            frame_parameters.subscan_fractional_center = Geometry.FloatPoint(0.5, 0.5)
            xdata = scan_hardware_source.record_immediate(frame_parameters, [0])[0]
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].scale, xdata.dimensional_calibrations[1].scale)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].units, xdata.dimensional_calibrations[1].units)
            frame_parameters.subscan_pixel_size = Geometry.IntSize(256, 256)
            xdata = scan_hardware_source.record_immediate(frame_parameters, [0])[0]
            self.assertAlmostEqual(document_model.data_items[0].dimensional_calibrations[0].scale, xdata.dimensional_calibrations[1].scale * 2)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].units, xdata.dimensional_calibrations[1].units)

    def test_get_buffer_data_basic_functionality(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                for i in range(4):
                    scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                document_controller.periodic()
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            data_element_groups = scan_hardware_source.get_buffer_data(-4, 4)
            # ensure we got 4 acquisitions
            self.assertEqual(4, len(data_element_groups))
            # ensure all of them are calibrated the same
            offsets = set()
            scales = set()
            units = set()
            for data_element_group in data_element_groups:
                for data_element  in data_element_group:
                    xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
                    offsets.add(xdata.dimensional_calibrations[0].offset)
                    scales.add(xdata.dimensional_calibrations[0].scale)
                    units.add(xdata.dimensional_calibrations[0].units)
            self.assertEqual(1, len(offsets))
            self.assertEqual(1, len(scales))
            self.assertEqual(1, len(units))
            self.assertTrue(next(iter(units)))  # confirm calibration is not empty

    def test_reloading_document_cleans_display_items(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = False
            scan_hardware_source.line_scan_enabled = True
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = False
            scan_hardware_source.line_scan_enabled = False
            test_context.instrument.probe_position = Geometry.FloatPoint(0.5, 0.5)
            document_controller.periodic()
            scan_hardware_source.drift_enabled = True
            document_controller.periodic()
            self.assertEqual(4, len(document_model.display_items[0].graphics))
            document_controller.close()
            test_context.instrument.probe_position = None
            scan_hardware_source.subscan_enabled = False
            test_context.instrument.subscan_region = None
            scan_hardware_source.line_scan_enabled = False
            test_context.instrument.line_scan_vector = None
            scan_hardware_source.drift_enabled = False
            test_context.document_controller = test_context.create_document_controller(auto_close=False)
            test_context.document_model = test_context.document_controller.document_model
            document_model = test_context.document_model
            self.assertEqual(0, len(document_model.display_items[0].graphics))

    def test_graphics_are_enabled_when_switching_project(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = False
            scan_hardware_source.line_scan_enabled = True
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = False
            scan_hardware_source.line_scan_enabled = False
            test_context.instrument.probe_position = Geometry.FloatPoint(0.5, 0.5)
            document_controller.periodic()
            scan_hardware_source.drift_enabled = True
            document_controller.periodic()
            self.assertEqual(4, len(document_model.display_items[0].graphics))
            document_controller.close()
            test_context.document_controller = test_context.create_document_controller(auto_close=False)
            test_context.document_model = test_context.document_controller.document_model
            document_model = test_context.document_model
            self.assertEqual(4, len(document_model.display_items[0].graphics))

    def test_acquisition_preferences(self):
        dir = pathlib.Path.cwd() / "__Test"
        if dir.exists():
            shutil.rmtree(dir)
        os.makedirs(dir)
        file_path = dir / pathlib.Path("test.json")
        try:
            acquisition_preferences = AcquisitionPreferences.AcquisitionPreferences(ApplicationData.ApplicationData(file_path))
            acquisition_preferences._append_item("control_customizations", AcquisitionPreferences.ControlCustomizationSchema.create(None, {"control_id": "blanker", "device_control_id": "BLANK", "delay": 0.05}))
            acquisition_preferences._append_item("control_customizations", AcquisitionPreferences.ControlCustomizationSchema.create(None, {"control_id": "defocus", "device_control_id": "C10", "delay": 0.05}))
            acquisition_preferences.control_customizations[0].delay = 0.03
            acquisition_preferences._remove_item("control_customizations", acquisition_preferences.control_customizations[0])
            acquisition_preferences.control_customizations[0].uuid = uuid.uuid4()
        finally:
            if dir.exists():
                shutil.rmtree(dir)

    def test_scan_frame_parameters(self) -> None:
        with self.__test_context() as test_context:
            hardware_source = test_context.scan_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            # ensure it is initially dict-like
            frame_parameters.fov_size_nm = Geometry.FloatSize(8, 8)
            self.assertEqual(frame_parameters.size, frame_parameters["size"])
            self.assertEqual(frame_parameters.center_nm, frame_parameters["center_nm"])
            self.assertEqual(frame_parameters.fov_nm, frame_parameters["fov_nm"])
            self.assertEqual(frame_parameters.fov_size_nm, frame_parameters["fov_size_nm"])
            # try setting values
            frame_parameters["size"] = Geometry.IntSize(4, 4)
            frame_parameters["center_nm"] = Geometry.FloatPoint(5.0, 5.0)
            frame_parameters["fov_nm"] = 16.0
            self.assertEqual(Geometry.IntSize(4, 4), frame_parameters.size)
            self.assertEqual(Geometry.FloatPoint(5.0, 5.0), frame_parameters.center_nm)
            self.assertEqual(16.0, frame_parameters.fov_nm)
            # test basic parameters take tuples
            frame_parameters.size = (40, 40)
            frame_parameters.center_nm = (50.0, 50.0)
            self.assertEqual(Geometry.IntSize(40, 40), frame_parameters.size)
            self.assertEqual(Geometry.FloatPoint(50.0, 50.0), frame_parameters.center_nm)
            frame_parameters["size"] = (41, 41)
            frame_parameters["center_nm"] = (51.0, 51.0)
            self.assertEqual(Geometry.IntSize(41, 41), frame_parameters.size)
            self.assertEqual(Geometry.FloatPoint(51.0, 51.0), frame_parameters.center_nm)
            # test optional parameters take tuples and None
            self.assertIsNone(frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = (20, 20)
            self.assertEqual(Geometry.IntSize(20, 20), frame_parameters.subscan_pixel_size)
            self.assertEqual(Geometry.IntSize(20, 20), frame_parameters["subscan_pixel_size"])
            frame_parameters.subscan_pixel_size = Geometry.IntSize(21, 21)
            self.assertEqual(Geometry.IntSize(21, 21), frame_parameters.subscan_pixel_size)
            frame_parameters["subscan_pixel_size"] = Geometry.IntSize(22, 22)
            self.assertEqual(Geometry.IntSize(22, 22), frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = None
            self.assertIsNone(frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = Geometry.IntSize(21, 21)
            frame_parameters["subscan_pixel_size"] = None
            # test extra parameters get copied
            frame_parameters["extra"] = 8
            self.assertEqual(8, frame_parameters["extra"])
            frame_parameters_copy = scan_base.ScanFrameParameters(frame_parameters.as_dict())
            self.assertEqual(8, frame_parameters_copy["extra"])
            # test dict is writeable to json
            json.dumps(frame_parameters.as_dict())

    # center_nm, center_x_nm, and center_y_nm are all sensible for context and subscans
    # all requested and actual frame parameters are recorded
    # stem values are recorded
    # external scan values are recorded
    # subscan info is recorded
    # image groups from stem controller are being added

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
