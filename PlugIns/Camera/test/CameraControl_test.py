import contextlib
import copy
import random
import time
import unittest

import numpy

from nion.swift.model import DocumentModel
from nion.swift.model import HardwareSource
from nion.swift import Application
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift.test import HardwareSource_test
from nion.ui import TestUI
from nion.utils import Event
from Camera import CameraControlPanel

"""
# running in Swift
import sys, unittest
from Camera import CameraControl_test
suite = unittest.TestLoader().loadTestsFromTestCase(CameraControl_test.TestCameraControlClass)
result = unittest.TextTestResult(sys.stdout, True, True)
suite.run(result)
"""

class TestCameraControlClass(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)
        self.source_image = numpy.random.randn(1024, 1024).astype(numpy.float32)
        self.exposure = 1.0
        HardwareSource.HardwareSourceManager().hardware_sources = []
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()
        # HardwareSource.HardwareSourceManager()._reset()

    def tearDown(self):
        HardwareSource.HardwareSourceManager()._close_hardware_sources()

    def __acquire_one(self, document_controller, hardware_source):
        hardware_source.start_playing()
        try:
            start_time = time.time()
            while not hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < 3.0)
        finally:
            hardware_source.stop_playing()
        start_time = time.time()
        while hardware_source.is_playing:
            time.sleep(self.exposure)
            self.assertTrue(time.time() - start_time < 3.0)
        document_controller.periodic()

    def _setup_hardware_source(self, initialize: bool=True, is_eels: bool=False) -> (DocumentController.DocumentController, DocumentModel.DocumentModel, HardwareSource.HardwareSource, CameraControlPanel.CameraControlStateController):
        raise NotImplementedError()

    def __setup_hardware_source(self, initialize: bool=True, is_eels: bool=False) -> (DocumentController.DocumentController, DocumentModel.DocumentModel, HardwareSource.HardwareSource, CameraControlPanel.CameraControlStateController):
        return self._setup_hardware_source(initialize, is_eels)

    ## STANDARD ACQUISITION TESTS ##

    # Do not change the comment above as it is used to search for places needing updates when a new
    # standard acquisition test is added.

    def test_acquiring_frames_with_generator_produces_correct_frame_numbers(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_acquiring_frames_with_generator_produces_correct_frame_numbers(self, hardware_source, document_controller)

    def test_acquire_multiple_frames_reuses_same_data_item(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_acquire_multiple_frames_reuses_same_data_item(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_stop_actually_stops_acquisition(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_simple_hardware_start_and_stop_actually_stops_acquisition(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_abort_works_as_expected(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_simple_hardware_start_and_abort_works_as_expected(self, hardware_source, document_controller)

    def test_view_reuses_single_data_item(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_view_reuses_single_data_item(self, hardware_source, document_controller)

    def test_get_next_data_elements_to_finish_returns_full_frames(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_get_next_data_elements_to_finish_returns_full_frames(self, hardware_source, document_controller)

    def test_get_next_data_elements_to_finish_produces_data_item_full_frames(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_get_next_data_elements_to_finish_produces_data_item_full_frames(self, hardware_source, document_controller)

    def test_exception_during_view_halts_playback(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_exception_during_view_halts_playback(self, hardware_source, self.exposure)

    def test_able_to_restart_view_after_exception(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            HardwareSource_test._test_able_to_restart_view_after_exception(self, hardware_source, self.exposure)

    # End of standard acquisition tests.

    def test_view_generates_a_data_item(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            self.__acquire_one(document_controller, hardware_source)
            self.__acquire_one(document_controller, hardware_source)
            self.__acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items), 1)

    def test_ability_to_set_profile_parameters_is_reflected_in_acquisition(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 2
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.binning = 1
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            self.__acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_sources[0].data_shape, hardware_source.get_expected_dimensions(2))
            hardware_source.set_selected_profile_index(1)
            self.__acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_sources[0].data_shape, hardware_source.get_expected_dimensions(1))

    def test_change_to_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 2
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.binning = 1
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            hardware_source.start_playing()
            try:
                self.assertEqual(hardware_source.get_next_data_elements_to_start()[0]["data"].shape, hardware_source.get_expected_dimensions(2))
                time.sleep(self.exposure * 0.1)
                hardware_source.set_selected_profile_index(1)
                self.assertEqual(hardware_source.get_next_data_elements_to_start()[0]["data"].shape, hardware_source.get_expected_dimensions(1))
            finally:
                hardware_source.abort_playing()

    def test_changing_frame_parameters_during_view_does_not_affect_current_acquisition(self):
        # NOTE: this currently fails on Orca camera because changing binning will immediately stop acquisition and restart.
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            # verify that the frame parameters are applied to the _next_ frame.
            profile_index = 0  # 1, 2 both have special cases that make this test fail.
            frame_parameters = hardware_source.get_frame_parameters(profile_index)
            frame_parameters.exposure_ms = 800
            hardware_source.set_frame_parameters(0, frame_parameters)
            hardware_source.set_selected_profile_index(profile_index)
            frame_parameters = hardware_source.get_frame_parameters(profile_index)
            frame_parameters.binning = 4
            frame_time = frame_parameters.exposure_ms / 1000.0
            hardware_source.start_playing()
            try:
                # the time taken to start playing is unpredictable,
                # so first make sure the camera is playing.
                start_time = time.time()
                while not hardware_source.is_playing:
                    time.sleep(self.exposure)
                    self.assertTrue(time.time() - start_time < 3.0)
                # now it is playing, so synchronize to the end of a frame
                hardware_source.get_next_data_elements_to_finish(10.0)
                # now wait long enough for the next frame to start, 50ms should be enough
                time.sleep(0.05)
                # now set the frame parameters. on the ccd1010, this takes a long time due to
                # intentional (but perhaps spurious) delays. so the only way this test will pass
                # is if 0.8 * exposure is greater than the total of the intentional delays.
                # the ccd1010 currently sleeps 600ms. so exposure must be about 800ms.
                hardware_source.set_frame_parameters(profile_index, frame_parameters)
                self.assertEqual(hardware_source.get_next_data_elements_to_finish(10.0)[0]["data"].shape, hardware_source.get_expected_dimensions(2))
                # now verify that the frame parameters are actually applied to the _next_ frame.
                time.sleep(frame_time * 0.8)
                self.assertEqual(hardware_source.get_next_data_elements_to_finish(10.0)[0]["data"].shape, hardware_source.get_expected_dimensions(4))
            finally:
                hardware_source.stop_playing()

    def test_capturing_during_view_captures_new_data_items(self):
        # NOTE: This fails on cameras where stop_playing does a synchronized stop rather than just requesting it to stop.
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            hardware_source.start_playing()
            try:
                hardware_source.get_next_data_elements_to_finish(5)
                document_controller.periodic()
                self.assertEqual(len(document_model.data_items), 1)
                state_controller.handle_capture_clicked()
            finally:
                hardware_source.stop_playing()
            # if stop is synchronized, the next statement will fail, since the hardware source is fully stopped already.
            hardware_source.get_next_data_elements_to_finish(5)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)

    def test_ability_to_start_playing_with_custom_parameters(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 4
            hardware_source.set_current_frame_parameters(frame_parameters_0)
            hardware_source.start_playing()
            try:
                hardware_source.get_next_data_elements_to_finish(10)
            finally:
                hardware_source.stop_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < 3.0)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 1)
            self.assertEqual(document_model.data_items[0].data_sources[0].dimensional_shape, hardware_source.get_expected_dimensions(4))

    def test_changing_profile_updates_frame_parameters_in_ui(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            state_controller.on_frame_parameters_changed = frame_parameters_changed
            hardware_source.set_selected_profile_index(1)
            document_controller.periodic()
            self.assertIsNotNone(frame_parameters_ref[0])

    def test_changing_current_profiles_frame_parameters_updates_frame_parameters_in_ui(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            state_controller.on_frame_parameters_changed = frame_parameters_changed
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 4
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            self.assertIsNotNone(frame_parameters_ref[0])
            self.assertEqual(frame_parameters_ref[0].binning, 4)

    def test_changing_binning_is_reflected_in_new_acquisition(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            self.__acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].maybe_data_source.data_shape, hardware_source.get_expected_dimensions(2))
            state_controller.handle_binning_changed("4")
            self.__acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].maybe_data_source.data_shape, hardware_source.get_expected_dimensions(4))

    def test_first_view_uses_correct_mode(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            hardware_source.start_playing()
            try:
                time.sleep(self.exposure * 0.5)
                hardware_source.get_next_data_elements_to_finish()  # view again
                document_controller.periodic()
                self.assertEqual(document_model.data_items[0].maybe_data_source.data_shape, hardware_source.get_expected_dimensions(4))
                hardware_source.get_next_data_elements_to_finish()  # view again
                document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            self.assertEqual(document_model.data_items[0].maybe_data_source.data_shape, hardware_source.get_expected_dimensions(4))

    def test_first_view_uses_correct_exposure(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            long_exposure = 0.5
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            state_controller.handle_exposure_changed(str(int(long_exposure * 1000)))
            start = time.time()
            hardware_source.start_playing()
            try:
                hardware_source.get_next_data_elements_to_finish()  # view again
                elapsed = time.time() - start
                document_controller.periodic()
                self.assertEqual(document_model.data_items[0].maybe_data_source.data_shape, hardware_source.get_expected_dimensions(4))
                self.assertTrue(elapsed > long_exposure)
            finally:
                hardware_source.abort_playing()

    def test_view_followed_by_frame_uses_correct_exposure(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            long_exposure = 0.5
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            state_controller.handle_exposure_changed(str(int(long_exposure * 1000)))
            state_controller.handle_change_profile("Run")
            hardware_source.start_playing()
            try:
                time.sleep(self.exposure * 0.5)
                data_and_metadata = hardware_source.get_next_data_and_metadata_list_to_finish()[0]  # view again
                self.assertEqual(data_and_metadata.data_shape, hardware_source.get_expected_dimensions(2))
            finally:
                hardware_source.stop_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < 3.0)
            state_controller.handle_change_profile("Snap")
            hardware_source.start_playing()
            try:
                start = time.time()
                data_and_metadata = hardware_source.get_next_data_and_metadata_list_to_finish()[0]  # frame now
                elapsed = time.time() - start
            finally:
                hardware_source.abort_playing()
            self.assertEqual(data_and_metadata.data_shape, hardware_source.get_expected_dimensions(4))
            self.assertTrue(elapsed > long_exposure)

    def test_exception_during_view_leaves_buttons_in_ready_state(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            play_enabled = [False]
            play_state = [None]
            def play_button_state_changed(enabled, state):
                play_enabled[0] = enabled
                play_state[0] = state
            state_controller.on_play_button_state_changed = play_button_state_changed
            enabled = [False]
            def raise_exception():
                if enabled[0]:
                    raise Exception("Error during acquisition")
            hardware_source._test_acquire_hook = raise_exception
            hardware_source._test_acquire_exception = lambda *args: None
            hardware_source.start_playing()
            try:
                hardware_source.get_next_data_elements_to_finish()
                document_controller.periodic()
                self.assertEqual(play_enabled[0], True)
                self.assertEqual(play_state[0], "pause")
                self.assertTrue(hardware_source.is_playing)
                enabled[0] = True
                hardware_source.get_next_data_elements_to_finish()
                # avoid a race condition and wait for is_playing to go false.
                start_time = time.time()
                while hardware_source.is_playing:
                    time.sleep(0.01)
                    self.assertTrue(time.time() - start_time < 3.0)
                document_controller.periodic()
                self.assertEqual(play_enabled[0], True)
                self.assertEqual(play_state[0], "play")
            finally:
                hardware_source.abort_playing()

    def test_profile_initialized_correctly(self):
        # this once failed due to incorrect closure handling in __update_profile_index
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(initialize=False)
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            profile_ref = [None]
            def profiles_changed(profiles):
                pass
            def profile_changed(profile):
                profile_ref[0] = profile
            state_controller.on_profiles_changed = profiles_changed
            state_controller.on_profile_changed = profile_changed
            state_controller.initialize_state()
            document_controller.periodic()
            self.assertEqual(profile_ref[0], "Run")

    def test_processed_data_is_produced_when_specified(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            data_items = [None, None]
            def display_data_item_changed(data_item):
                data_items[0] = data_item
            def processed_data_item_changed(data_item):
                data_items[1] = data_item
            state_controller.on_display_data_item_changed = display_data_item_changed
            state_controller.on_processed_data_item_changed = processed_data_item_changed
            hardware_source.start_playing()
            try:
                for _ in range(4):
                    hardware_source.get_next_data_elements_to_finish()
                    document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            document_model.recompute_all()
            self.assertEqual(data_items[0].maybe_data_source.data_shape, hardware_source.get_expected_dimensions(2))
            self.assertEqual(data_items[1].maybe_data_source.data_shape, (hardware_source.get_expected_dimensions(2)[1], ))

    def test_processed_data_is_reused(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            data_items = [None, None]
            def display_data_item_changed(data_item):
                data_items[0] = data_item
            def processed_data_item_changed(data_item):
                data_items[1] = data_item
            state_controller.on_display_data_item_changed = display_data_item_changed
            state_controller.on_processed_data_item_changed = processed_data_item_changed
            # first acquisition
            hardware_source.start_playing()
            try:
                for _ in range(4):
                    hardware_source.get_next_data_elements_to_finish()
                    document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            document_model.recompute_all()
            # make sure really stopped
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < 3.0)
            self.assertEqual(len(document_model.data_items), 2)
            # second acquisition
            first_data_items = copy.copy(data_items)
            hardware_source.start_playing()
            try:
                for _ in range(4):
                    hardware_source.get_next_data_elements_to_finish()
                    document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)
            self.assertEqual(data_items, first_data_items)

    def test_processed_data_is_regenerated_if_necessary(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            self.__acquire_one(document_controller, hardware_source)
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)
            document_model.remove_data_item(document_model.data_items[1])
            self.__acquire_one(document_controller, hardware_source)
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)
            document_model.remove_data_item(document_model.data_items[0])
            self.__acquire_one(document_controller, hardware_source)
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)

    def test_deleting_processed_data_item_during_acquisition_recovers_correctly(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller):
            hardware_source.start_playing()
            try:
                start_time = time.time()
                while not hardware_source.is_playing:
                    time.sleep(0.01)
                    self.assertTrue(time.time() - start_time < 3.0)
                start_time = time.time()
                while len(document_model.data_items) < 2:
                    time.sleep(0.01)
                    document_controller.periodic()
                    self.assertTrue(time.time() - start_time < 3.0)
                document_model.remove_data_item(document_model.data_items[1])
                start_time = time.time()
                while len(document_model.data_items) < 2:
                    time.sleep(0.01)
                    document_controller.periodic()
                    self.assertTrue(time.time() - start_time < 3.0)
            finally:
                hardware_source.abort_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(0.01)
                self.assertTrue(time.time() - start_time < 3.0)
            document_controller.periodic()

    def test_starting_processed_view_initializes_source_region(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller):
            self.__acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items[0].maybe_data_source.displays[0].graphics), 1)
            self.assertEqual(document_model.data_items[0].maybe_data_source.displays[0].graphics[0].bounds, hardware_source.data_channels[1].processor.bounds)

    def test_changing_processed_bounds_updates_region(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller):
            self.__acquire_one(document_controller, hardware_source)
            new_bounds = (0.45, 0.2), (0.1, 0.6)
            hardware_source.data_channels[1].processor.bounds = new_bounds
            self.assertEqual(document_model.data_items[0].maybe_data_source.displays[0].graphics[0].bounds, new_bounds)

    def test_changing_region_on_processed_view_updates_processor(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller):
            self.__acquire_one(document_controller, hardware_source)
            new_bounds = (0.45, 0.2), (0.1, 0.6)
            document_model.data_items[0].maybe_data_source.displays[0].graphics[0].bounds = new_bounds
            self.assertEqual(hardware_source.data_channels[1].processor.bounds, new_bounds)

    def test_restarting_processed_view_recreates_region_after_it_has_been_deleted(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source(is_eels=True)
        with contextlib.closing(document_controller):
            self.__acquire_one(document_controller, hardware_source)
            display = document_model.data_items[0].maybe_data_source.displays[0]
            display.remove_graphic(display.graphics[0])
            self.__acquire_one(document_controller, hardware_source)
            self.assertEqual(display.graphics[0].bounds, hardware_source.data_channels[1].processor.bounds)

    def test_consecutive_frames_have_unique_data(self):
        numpy.random.seed(999)
        random.seed(999)
        self.source_image = numpy.random.randn(1024, 1024).astype(numpy.float32)
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            hardware_source.start_playing()
            try:
                data = hardware_source.get_next_data_elements_to_finish()[0]["data"]
                last_average = numpy.average(data)
                for _ in range(16):
                    data = hardware_source.get_next_data_elements_to_finish()[0]["data"]
                    next_average = numpy.average(data)
                    self.assertNotEqual(last_average, next_average)
                    last_average = next_average
            finally:
                hardware_source.abort_playing()
            numpy.random.seed()
            random.seed()

    def test_facade_frame_parameter_methods(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            hardware_source_facade.get_default_frame_parameters()
            frame_parameters = hardware_source_facade.get_frame_parameters()
            record_frame_parameters = hardware_source_facade.get_record_frame_parameters()
            profile_frame_parameters = hardware_source_facade.get_frame_parameters_for_profile_by_index(0)
            hardware_source_facade.set_frame_parameters(frame_parameters)
            hardware_source_facade.set_record_frame_parameters(record_frame_parameters)
            hardware_source_facade.set_frame_parameters_for_profile_by_index(0, profile_frame_parameters)

    def test_facade_playback_stop(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            self.assertFalse(hardware_source.is_playing)  # we know this works
            self.assertFalse(hardware_source_facade.is_playing)
            hardware_source_facade.start_playing()
            try:
                time.sleep(self.exposure * 0.5)
                self.assertTrue(hardware_source.is_playing)  # we know this works
                self.assertTrue(hardware_source_facade.is_playing)
            finally:
                hardware_source_facade.stop_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure * 0.1)
                self.assertTrue(time.time() - start_time < 3.0)
            self.assertFalse(hardware_source.is_playing)  # we know this works
            self.assertFalse(hardware_source_facade.is_playing)

    def test_facade_playback_abort(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            self.assertFalse(hardware_source.is_playing)  # we know this works
            self.assertFalse(hardware_source_facade.is_playing)
            hardware_source_facade.start_playing()
            try:
                time.sleep(self.exposure * 0.5)
                self.assertTrue(hardware_source.is_playing)  # we know this works
                self.assertTrue(hardware_source_facade.is_playing)
            finally:
                hardware_source_facade.abort_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure * 0.1)
                self.assertTrue(time.time() - start_time < 3.0)
            self.assertFalse(hardware_source.is_playing)  # we know this works
            self.assertFalse(hardware_source_facade.is_playing)

    def test_facade_record(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        api = Facade.get_api("~1.0", "~1.0")
        hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)
            hardware_source_facade.start_recording()
            time.sleep(self.exposure * 0.5)
            self.assertTrue(hardware_source.is_recording)  # we know this works
            self.assertTrue(hardware_source_facade.is_recording)
            start_time = time.time()
            while hardware_source.is_recording:
                time.sleep(self.exposure * 0.1)
                self.assertTrue(time.time() - start_time < 4.0)
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_abort_record(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)
            # hardware_source.stages_per_frame = 5
            hardware_source_facade.start_recording()
            time.sleep(self.exposure * 0.1)
            self.assertTrue(hardware_source.is_recording)  # we know this works
            self.assertTrue(hardware_source_facade.is_recording)
            hardware_source_facade.abort_recording()
            start_time = time.time()
            while hardware_source.is_recording:
                time.sleep(self.exposure * 0.01)
                self.assertTrue(time.time() - start_time < 3.0)
            # TODO: figure out a way to test whether abort actually aborts or just stops
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_abort_record_and_return_data(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            data_and_metadata_list = hardware_source_facade.record()
            data_and_metadata = data_and_metadata_list[0]
            self.assertIsNotNone(data_and_metadata.data)

    def test_facade_view_task(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            view_task = hardware_source_facade.create_view_task()
            with contextlib.closing(view_task):
                data_and_metadata_list1 = view_task.grab_next_to_finish()
                self.assertTrue(hardware_source_facade.is_playing)
                data_and_metadata_list2 = view_task.grab_immediate()
                data_and_metadata_list3 = view_task.grab_next_to_start()
                self.assertIsNotNone(data_and_metadata_list1[0].data)
                self.assertIsNotNone(data_and_metadata_list2[0].data)
                self.assertIsNotNone(data_and_metadata_list3[0].data)

    def test_facade_record_task(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            record_task = hardware_source_facade.create_record_task()
            with contextlib.closing(record_task):
                data_and_metadata_list = record_task.grab()
                data_and_metadata = data_and_metadata_list[0]
                self.assertIsNotNone(data_and_metadata.data)
                # changing the ccd1010 (stopping) takes 600ms+ so wait until it stops recording
                start_time = time.time()
                while hardware_source.is_recording:
                    time.sleep(self.exposure * 0.01)
                    self.assertTrue(time.time() - start_time < 3.0)
                self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_record_task_cancel(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            record_task = hardware_source_facade.create_record_task()
            with contextlib.closing(record_task):
                time.sleep(self.exposure * 0.1)
                record_task.cancel()
                start_time = time.time()
                while hardware_source.is_recording:
                    time.sleep(self.exposure * 0.01)
                    self.assertTrue(time.time() - start_time < 4.0)
                self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_grab_data(self):
        document_controller, document_model, hardware_source, state_controller = self.__setup_hardware_source()
        with contextlib.closing(document_controller), contextlib.closing(state_controller):
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            hardware_source_facade.start_playing()
            try:
                time.sleep(self.exposure * 0.5)
                data_and_metadata_list1 = hardware_source_facade.grab_next_to_finish()
                data_and_metadata_list2 = hardware_source_facade.grab_next_to_start()
                self.assertIsNotNone(data_and_metadata_list1[0].data)
                self.assertIsNotNone(data_and_metadata_list2[0].data)
            finally:
                hardware_source_facade.stop_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure * 0.1)
                self.assertTrue(time.time() - start_time < 3.0)

    def planned_test_custom_view_followed_by_ui_view_uses_ui_frame_parameters(self):
        pass

    def planned_test_setting_custom_frame_parameters_updates_ui(self):
        pass

    def planned_test_cold_start_acquisition_from_thread_produces_data(self):
        pass

    def planned_test_hot_start_acquisition_from_thread_produces_data(self):
        pass

    def planned_test_hot_start_acquisition_from_thread_with_custom_parameters_produces_data(self):
        pass


if __name__ == '__main__':
    unittest.main()
