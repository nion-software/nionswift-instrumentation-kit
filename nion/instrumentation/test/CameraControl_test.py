import contextlib
import copy
import json
import random
import time
import unittest
import zlib

import numpy

from nion.instrumentation import camera_base
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import HardwareSource_test
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import Graphics
from nion.swift.model import Metadata
from nion.ui import TestUI
from nion.utils import Geometry
from nion.utils import Registry
from nionswift_plugin.nion_instrumentation_ui import CameraControlPanel

"""
# running in Swift
import sys, unittest
from nionswift_plugin.nion_instrumentation_ui import CameraControl_test
suite = unittest.TestLoader().loadTestsFromTestCase(CameraControl_test.TestCameraControlClass)
result = unittest.TextTestResult(sys.stdout, True, True)
suite.run(result)
"""

TIMEOUT = 20

class TestCameraControlClass(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)
        self.source_image = numpy.random.randn(1024, 1024).astype(numpy.float32)
        self.exposure = 1.0

    def _acquire_one(self, document_controller, hardware_source):
        hardware_source.start_playing()
        try:
            start_time = time.time()
            while not hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < TIMEOUT)
        finally:
            hardware_source.stop_playing()
        start_time = time.time()
        while hardware_source.is_playing:
            time.sleep(self.exposure)
            self.assertTrue(time.time() - start_time < TIMEOUT)
        document_controller.periodic()

    def __test_context(self, is_eels: bool=False):
        return AcquisitionTestContext.test_context(is_eels=is_eels, camera_exposure=0.04)

    def __create_state_controller(self, acquisition_test_context: AcquisitionTestContext.AcquisitionTestContext, *,
                                  initialize: bool = True) -> CameraControlPanel.CameraControlStateController:
        state_controller = CameraControlPanel.CameraControlStateController(acquisition_test_context.camera_hardware_source,
                                                                           acquisition_test_context.document_controller.queue_task,
                                                                           acquisition_test_context.document_model)
        if initialize:
            state_controller.initialize_state()
        acquisition_test_context.push(state_controller)
        return state_controller

    def _test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        return self.__test_context(is_eels=is_eels)

    ## STANDARD ACQUISITION TESTS ##

    # Do not change the comment above as it is used to search for places needing updates when a new
    # standard acquisition test is added.

    def test_acquiring_frames_with_generator_produces_correct_frame_numbers(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_acquiring_frames_with_generator_produces_correct_frame_numbers(self, hardware_source, document_controller)

    def test_acquire_multiple_frames_reuses_same_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_acquire_multiple_frames_reuses_same_data_item(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_stop_actually_stops_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_stop_actually_stops_acquisition(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_abort_works_as_expected(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_abort_works_as_expected(self, hardware_source, document_controller)

    def test_view_reuses_single_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_view_reuses_single_data_item(self, hardware_source, document_controller)

    def test_get_next_data_elements_to_finish_returns_full_frames(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_get_next_data_elements_to_finish_returns_full_frames(self, hardware_source, document_controller)

    def test_exception_during_view_halts_playback(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_exception_during_view_halts_playback(self, hardware_source, self.exposure)

    def test_able_to_restart_view_after_exception(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_able_to_restart_view_after_exception(self, hardware_source, self.exposure)

    # End of standard acquisition tests.

    def test_view_generates_a_data_item(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self._acquire_one(document_controller, hardware_source)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items), 1)

    def test_ability_to_set_profile_parameters_is_reflected_in_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 2
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.binning = 1
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(2))
            hardware_source.set_selected_profile_index(1)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(1))

    def test_change_to_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 2
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.binning = 1
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            hardware_source.start_playing()
            try:
                self.assertEqual(hardware_source.get_next_xdatas_to_start()[0].data.shape, hardware_source.get_expected_dimensions(2))
                time.sleep(self.exposure * 1.1)
                hardware_source.set_selected_profile_index(1)
                hardware_source.get_next_xdatas_to_start()  # skip one frame during testing to avoid timing variance
                self.assertEqual(hardware_source.get_next_xdatas_to_start()[0].data.shape, hardware_source.get_expected_dimensions(1))
            finally:
                hardware_source.abort_playing()

    def test_changing_frame_parameters_during_view_does_not_affect_current_acquisition(self):
        # NOTE: this currently fails on Orca camera because changing binning will immediately stop acquisition and restart.
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
                    self.assertTrue(time.time() - start_time < TIMEOUT)
                # now it is playing, so synchronize to the end of a frame
                hardware_source.get_next_xdatas_to_finish(10.0)
                # now wait long enough for the next frame to start, 50ms should be enough
                time.sleep(0.05)
                # now set the frame parameters. on the ccd1010, this takes a long time due to
                # intentional (but perhaps spurious) delays. so the only way this test will pass
                # is if 0.8 * exposure is greater than the total of the intentional delays.
                # the ccd1010 currently sleeps 600ms. so exposure must be about 800ms.
                hardware_source.set_frame_parameters(profile_index, frame_parameters)
                self.assertEqual(hardware_source.get_next_xdatas_to_finish(10.0)[0].data.shape, hardware_source.get_expected_dimensions(2))
                # now verify that the frame parameters are actually applied to the _next_ frame.
                time.sleep(frame_time * 0.8)
                self.assertEqual(hardware_source.get_next_xdatas_to_finish(10.0)[0].data.shape, hardware_source.get_expected_dimensions(4))
            finally:
                hardware_source.stop_playing()

    def test_capturing_during_view_captures_new_data_items(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
                self.assertEqual(len(document_model.data_items), 1)
                state_controller.handle_capture_clicked()
                hardware_source.get_next_xdatas_to_finish(5)
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)

    def test_capturing_during_view_captures_eels_2d(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
                self.assertEqual(len(document_model.data_items), 2)
                state_controller.use_processed_data = False
                state_controller.handle_capture_clicked()
                hardware_source.get_next_xdatas_to_finish(5)
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 3)
            self.assertEqual(len(document_model.data_items[2].dimensional_shape), 2)

    def test_capturing_during_view_captures_eels_1d(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
                self.assertEqual(len(document_model.data_items), 2)
                state_controller.use_processed_data = True
                state_controller.handle_capture_clicked()
                hardware_source.get_next_xdatas_to_finish(5)
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 3)
            self.assertEqual(len(document_model.data_items[2].dimensional_shape), 1)

    def test_ability_to_start_playing_with_custom_parameters(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 4
            hardware_source.set_current_frame_parameters(frame_parameters_0)
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish(10)
            finally:
                hardware_source.stop_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < TIMEOUT)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 1)
            self.assertEqual(document_model.data_items[0].dimensional_shape, hardware_source.get_expected_dimensions(4))

    def test_changing_profile_updates_frame_parameters_in_ui(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            state_controller.on_frame_parameters_changed = frame_parameters_changed
            hardware_source.set_selected_profile_index(1)
            document_controller.periodic()
            self.assertIsNotNone(frame_parameters_ref[0])

    def test_changing_current_profiles_frame_parameters_updates_frame_parameters_in_ui(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            frame_parameters_ref = [None]
            def frame_parameters_changed(frame_parameters):
                frame_parameters_ref[0] = frame_parameters
            state_controller.on_frame_parameters_changed = frame_parameters_changed
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 4
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            self.assertIsNotNone(frame_parameters_ref[0])
            self.assertEqual(frame_parameters_ref[0].binning, 4)

    def test_binning_values_are_not_empty(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            binning_values = hardware_source.binning_values
            self.assertTrue(len(binning_values) > 0)
            self.assertTrue(all(map(lambda x: isinstance(x, int), binning_values)))

    def test_changing_binning_is_reflected_in_new_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(2))
            state_controller.handle_binning_changed("4")
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(4))

    def test_first_view_uses_correct_mode(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            hardware_source.start_playing()
            try:
                time.sleep(self.exposure * 0.5)
                hardware_source.get_next_xdatas_to_finish()  # view again
                document_controller.periodic()
                self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(4))
                hardware_source.get_next_xdatas_to_finish()  # view again
                document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(4))

    def test_first_view_uses_correct_exposure(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            long_exposure = 0.5
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            state_controller.handle_exposure_changed(str(int(long_exposure * 1000)))
            start = time.time()
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish()  # view again
                elapsed = time.time() - start
                document_controller.periodic()
                self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(4))
                self.assertTrue(elapsed > long_exposure)
            finally:
                hardware_source.abort_playing()

    def test_view_followed_by_frame_uses_correct_exposure(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            long_exposure = 0.5
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            state_controller.handle_exposure_changed(str(int(long_exposure * 1000)))
            state_controller.handle_change_profile("Run")
            hardware_source.start_playing()
            try:
                time.sleep(self.exposure * 0.5)
                data_and_metadata = hardware_source.get_next_xdatas_to_finish()[0]  # view again
                self.assertEqual(data_and_metadata.data_shape, hardware_source.get_expected_dimensions(2))
            finally:
                hardware_source.stop_playing()
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < TIMEOUT)
            state_controller.handle_change_profile("Snap")
            hardware_source.start_playing()
            try:
                start = time.time()
                data_and_metadata = hardware_source.get_next_xdatas_to_finish()[0]  # frame now
                elapsed = time.time() - start
            finally:
                hardware_source.abort_playing()
            self.assertEqual(data_and_metadata.data_shape, hardware_source.get_expected_dimensions(4))
            self.assertTrue(elapsed > long_exposure)

    def test_exception_during_view_leaves_buttons_in_ready_state(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
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
                hardware_source.get_next_xdatas_to_finish()
                document_controller.periodic()
                self.assertEqual(play_enabled[0], True)
                self.assertEqual(play_state[0], "pause")
                self.assertTrue(hardware_source.is_playing)
                enabled[0] = True
                hardware_source.get_next_xdatas_to_finish()
                # avoid a race condition and wait for is_playing to go false.
                start_time = time.time()
                while hardware_source.is_playing:
                    time.sleep(0.01)
                    self.assertTrue(time.time() - start_time < TIMEOUT)
                document_controller.periodic()
                self.assertEqual(play_enabled[0], True)
                self.assertEqual(play_state[0], "play")
            finally:
                hardware_source.abort_playing()

    def test_profile_initialized_correctly(self):
        # this once failed due to incorrect closure handling in __update_profile_index
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            state_controller = self.__create_state_controller(test_context, initialize=False)
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
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            data_items = [None, None]
            def display_data_item_changed():
                data_items[0] = state_controller.data_item_reference.data_item
            def processed_data_item_changed():
                data_items[1] = state_controller.processed_data_item_reference.data_item
            data_item_reference_changed_listener = state_controller.data_item_reference.data_item_reference_changed_event.listen(display_data_item_changed)
            processed_data_item_reference_changed_listener = state_controller.processed_data_item_reference.data_item_reference_changed_event.listen(processed_data_item_changed)
            hardware_source.start_playing()
            try:
                for _ in range(4):
                    hardware_source.get_next_xdatas_to_finish()
                    document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            document_model.recompute_all()
            self.assertEqual(data_items[0].data_shape, hardware_source.get_expected_dimensions(2))
            self.assertEqual(data_items[1].data_shape, (hardware_source.get_expected_dimensions(2)[1], ))
            data_item_reference_changed_listener.close()
            processed_data_item_reference_changed_listener.close()

    def test_processed_data_is_reused(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            data_items = [None, None]
            def display_data_item_changed():
                data_items[0] = state_controller.data_item_reference.data_item
            def processed_data_item_changed():
                data_items[1] = state_controller.processed_data_item_reference.data_item
            data_item_reference_changed_listener = state_controller.data_item_reference.data_item_reference_changed_event.listen(display_data_item_changed)
            processed_data_item_reference_changed_listener = state_controller.processed_data_item_reference.data_item_reference_changed_event.listen(processed_data_item_changed)
            # first acquisition
            hardware_source.start_playing()
            try:
                for _ in range(4):
                    hardware_source.get_next_xdatas_to_finish()
                    document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            document_model.recompute_all()
            # make sure really stopped
            start_time = time.time()
            while hardware_source.is_playing:
                time.sleep(self.exposure)
                self.assertTrue(time.time() - start_time < TIMEOUT)
            self.assertEqual(len(document_model.data_items), 2)
            # second acquisition
            first_data_items = copy.copy(data_items)
            hardware_source.start_playing()
            try:
                for _ in range(4):
                    hardware_source.get_next_xdatas_to_finish()
                    document_controller.periodic()
            finally:
                hardware_source.abort_playing()
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)
            self.assertEqual(data_items, first_data_items)
            data_item_reference_changed_listener.close()
            processed_data_item_reference_changed_listener.close()

    def test_processed_data_is_regenerated_if_necessary(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)
            document_model.remove_data_item(document_model.data_items[1])
            self._acquire_one(document_controller, hardware_source)
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)
            document_model.remove_data_item(document_model.data_items[0])
            self._acquire_one(document_controller, hardware_source)
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)

    def test_deleting_processed_data_item_during_acquisition_recovers_correctly(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing(sync_timeout=TIMEOUT)
            try:
                start_time = time.time()
                while len(document_model.data_items) < 2:
                    time.sleep(0.01)
                    document_controller.periodic()
                    self.assertTrue(time.time() - start_time < TIMEOUT)
                document_model.remove_data_item(document_model.data_items[1])
                start_time = time.time()
                while len(document_model.data_items) < 2:
                    time.sleep(0.01)
                    document_controller.periodic()
                    self.assertTrue(time.time() - start_time < TIMEOUT)
            finally:
                hardware_source.abort_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()

    def test_starting_processed_view_initializes_source_region(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 1)
            self.assertEqual(display_item.graphics[0].bounds, hardware_source.data_channels[1].processor.bounds)

    def test_changing_processed_bounds_updates_region(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            new_bounds = (0.45, 0.2), (0.1, 0.6)
            hardware_source.data_channels[1].processor.bounds = new_bounds
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(display_item.graphics[0].bounds, new_bounds)

    def test_changing_region_on_processed_view_updates_processor(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            new_bounds = (0.45, 0.2), (0.1, 0.6)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.graphics[0].bounds = new_bounds
            self.assertEqual(hardware_source.data_channels[1].processor.bounds, new_bounds)

    def test_restarting_processed_view_recreates_region_after_it_has_been_deleted(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.remove_graphic(display_item.graphics[0])
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(display_item.graphics[0].bounds, hardware_source.data_channels[1].processor.bounds)

    def test_acquiring_eels_sets_signal_type_for_both_copies(self):
        # assumes EELS camera produces two data items
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual("eels", Metadata.get_metadata_value(document_model.data_items[0], "stem.signal_type"))
            self.assertEqual("eels", Metadata.get_metadata_value(document_model.data_items[1], "stem.signal_type"))

    def test_acquiring_ronchigram_sets_signal_type(self):
        # assumes EELS camera produces two data items
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual("ronchigram", Metadata.get_metadata_value(document_model.data_items[0], "stem.signal_type"))

    def test_acquire_attaches_required_metadata(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            stem_controller = Registry.get_component("stem_controller")
            self._acquire_one(document_controller, hardware_source)
            frame_parameters = hardware_source.get_current_frame_parameters()
            for metadata_source in [document_model.data_items[0]]:
                # import pprint; print(pprint.pformat(metadata_source.metadata))
                # note: frame_time and line_time_us do not currently handle flyback - so this is intentionally wrong
                self.assertEqual(hardware_source.hardware_source_id, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.id"))
                self.assertEqual(hardware_source.display_name, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.name"))
                self.assertNotIn("autostem", metadata_source.metadata["hardware_source"])
                self.assertIn("counts_per_electron", metadata_source.metadata["hardware_source"])
                self.assertEqual(stem_controller.GetVal("EHT"), Metadata.get_metadata_value(metadata_source, "stem.high_tension"))
                self.assertEqual(stem_controller.GetVal("C10"), Metadata.get_metadata_value(metadata_source, "stem.defocus"))
                self.assertEqual(frame_parameters.binning, Metadata.get_metadata_value(metadata_source, "stem.camera.binning"))
                self.assertEqual(frame_parameters.exposure_ms / 1000, Metadata.get_metadata_value(metadata_source, "stem.camera.exposure"))

    def test_consecutive_frames_have_unique_data(self):
        # this test will fail if the camera is saturated (or otherwise produces identical values naturally)
        numpy.random.seed(999)
        random.seed(999)
        self.source_image = numpy.random.randn(1024, 1024).astype(numpy.float32)
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing()
            try:
                data = hardware_source.get_next_xdatas_to_finish()[0].data
                last_hash = zlib.crc32(data)
                for _ in range(16):
                    data = hardware_source.get_next_xdatas_to_finish()[0].data
                    next_hash = zlib.crc32(data)
                    self.assertNotEqual(last_hash, next_hash)
                    last_hash = next_hash
            finally:
                hardware_source.abort_playing()
            numpy.random.seed()
            random.seed()

    def test_integrating_frames_updates_frame_count_by_integration_count(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.integration_count = 4
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.start_playing()
            try:
                frame0_integration_count = hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["integration_count"]
                frame1_integration_count = hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["integration_count"]
                self.assertEqual(frame0_integration_count, 4)
                self.assertEqual(frame1_integration_count, 4)
            finally:
                hardware_source.abort_playing(sync_timeout=TIMEOUT)

    def test_acquiring_attaches_timezone(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertIsNotNone(document_model.data_items[0].timezone)

    def test_acquire_sequence_2d_calibrations(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.acquire_sequence_prepare(4)
            data_elements = hardware_source.acquire_sequence(4)
            self.assertEqual(1, len(data_elements))
            data_element = data_elements[0]
            self.assertEqual(3, len(data_element["data"].shape))
            self.assertEqual(3, len(data_element["spatial_calibrations"]))

    def test_acquire_sequence_1d_calibrations(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.processing = "sum_project"
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.acquire_sequence_prepare(4)
            data_elements = hardware_source.acquire_sequence(4)
            self.assertEqual(1, len(data_elements))
            data_element = data_elements[0]
            self.assertEqual(2, len(data_element["data"].shape))
            self.assertEqual(2, len(data_element["spatial_calibrations"]))

    def test_acquire_with_probe_position(self):
        # used to test out the code path, but no specific asserts
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            stem_controller = Registry.get_component("stem_controller")
            stem_controller.validate_probe_position()
            stem_controller._update_scan_context(Geometry.IntSize(256, 256), Geometry.FloatPoint(), 12.0, 0.0)
            self._acquire_one(document_controller, hardware_source)
            self.assertIsNotNone(document_model.data_items[0].timezone)

    def test_eels_acquire_with_probe_position(self):
        # used to test out the code path, but no specific asserts
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            stem_controller = Registry.get_component("stem_controller")
            stem_controller.validate_probe_position()
            stem_controller._update_scan_context(Geometry.IntSize(256, 256), Geometry.FloatPoint(), 12.0, 0.0)
            self._acquire_one(document_controller, hardware_source)
            self.assertIsNotNone(document_model.data_items[0].timezone)

    def test_facade_frame_parameter_methods(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
                self.assertTrue(time.time() - start_time < TIMEOUT)
            self.assertFalse(hardware_source.is_playing)  # we know this works
            self.assertFalse(hardware_source_facade.is_playing)

    def test_facade_playback_abort(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
                self.assertTrue(time.time() - start_time < TIMEOUT)
            self.assertFalse(hardware_source.is_playing)  # we know this works
            self.assertFalse(hardware_source_facade.is_playing)

    def test_facade_record(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)
            hardware_source_facade.start_recording()
            time.sleep(self.exposure * 0.5)
            self.assertTrue(hardware_source.is_recording)  # we know this works
            self.assertTrue(hardware_source_facade.is_recording)
            start_time = time.time()
            while hardware_source.is_recording:
                time.sleep(self.exposure * 0.1)
                self.assertTrue(time.time() - start_time < TIMEOUT)
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_abort_record(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
                self.assertTrue(time.time() - start_time < TIMEOUT)
            # TODO: figure out a way to test whether abort actually aborts or just stops
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_abort_record_and_return_data(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            data_and_metadata_list = hardware_source_facade.record()
            data_and_metadata = data_and_metadata_list[0]
            self.assertIsNotNone(data_and_metadata.data)

    def test_facade_view_task(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
                    self.assertTrue(time.time() - start_time < TIMEOUT)
                self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_record_task_cancel(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            record_task = hardware_source_facade.create_record_task()
            with contextlib.closing(record_task):
                time.sleep(self.exposure * 0.1)
                record_task.cancel()
                start_time = time.time()
                while hardware_source.is_recording:
                    time.sleep(self.exposure * 0.01)
                    self.assertTrue(time.time() - start_time < TIMEOUT)
                self.assertFalse(hardware_source_facade.is_recording)

    def test_facade_grab_data(self):
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
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
                self.assertTrue(time.time() - start_time < TIMEOUT)

    def test_camera_frame_parameters(self) -> None:
        with self.__test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            # ensure it is initially dict-like
            self.assertEqual(frame_parameters.exposure_ms, frame_parameters["exposure_ms"])
            self.assertEqual(frame_parameters.binning, frame_parameters["binning"])
            self.assertEqual(frame_parameters.processing, frame_parameters["processing"])
            self.assertEqual(frame_parameters.integration_count, frame_parameters["integration_count"])
            # try setting values
            frame_parameters["exposure_ms"] = 8.0
            frame_parameters["binning"] = 8
            frame_parameters["processing"] = "processing"
            frame_parameters["integration_count"] = 8
            self.assertEqual(8.0, frame_parameters.exposure_ms)
            self.assertEqual(8, frame_parameters.binning)
            self.assertEqual("processing", frame_parameters.processing)
            self.assertEqual(8, frame_parameters.integration_count)
            # ensure masks are not None and can be set to masks or dict's but always return the object
            self.assertIsNotNone(frame_parameters.active_masks)
            self.assertIsNotNone(frame_parameters["active_masks"])
            mask = camera_base.Mask()
            graphic = Graphics.RectangleGraphic()
            mask.add_layer(graphic, 1.0)
            frame_parameters.active_masks = [mask]
            self.assertEqual(frame_parameters.active_masks[0].to_dict(), mask.to_dict())
            self.assertEqual(frame_parameters["active_masks"][0].to_dict(), mask.to_dict())  # mask is already a dict
            # test extra parameters
            frame_parameters["extra"] = 8
            self.assertEqual(8, frame_parameters["extra"])
            frame_parameters_copy = camera_base.CameraFrameParameters(frame_parameters.as_dict())
            self.assertEqual(8, frame_parameters_copy["extra"])
            # test dict is writeable to json
            json.dumps(frame_parameters.as_dict())

    def test_acquisition_state_updates_during_acquisition(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            self.assertEqual(state_controller.acquisition_state_model.value, "stopped")
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
                self.assertIn(state_controller.acquisition_state_model.value, ("partial", "complete"))
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
                document_controller.periodic()
            self.assertEqual(state_controller.acquisition_state_model.value, "stopped")

    def test_acquisition_state_after_exception_during_start(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            def raise_exception():
                raise Exception("Error during acquisition")
            hardware_source._test_start_hook = raise_exception
            hardware_source._test_acquire_exception = lambda *args: None
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()
            self.assertEqual(state_controller.acquisition_state_model.value, "error")

    def test_acquisition_state_after_exception_during_execute(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            def raise_exception():
                raise Exception("Error during acquisition")
            hardware_source._test_acquire_hook = raise_exception
            hardware_source._test_acquire_exception = lambda *args: None
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing()
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()
            self.assertEqual(state_controller.acquisition_state_model.value, "error")

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
