import contextlib
import copy
import dataclasses
import functools
import json
import pathlib
import random
import tempfile
import time
import typing
import unittest
import uuid
import zlib

import numpy

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import camera_base
from nion.instrumentation import DataChannel
from nion.instrumentation import DriftTracker
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import HardwareSource_test
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import Graphics
from nion.swift.model import Metadata
from nion.swift.model import Schema
from nion.swift.test import TestContext
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Registry
from nionswift_plugin.nion_instrumentation_ui import AcquisitionPanel
from nionswift_plugin.nion_instrumentation_ui import CameraControlPanel

"""
# running in Swift
import sys, unittest
from nionswift_plugin.nion_instrumentation_ui import CameraControl_test
suite = unittest.TestLoader().loadTestsFromTestCase(CameraControl_test.TestCameraControlClass)
result = unittest.TextTestResult(sys.stdout, True, True)
suite.run(result)
"""

TIMEOUT = 30.0


class ApplicationDataInMemory:
    def __init__(self) -> None:
        self.d = dict()

    def get_data_dict(self) -> typing.Dict[str, typing.Any]:
        return self.d

    def set_data_dict(self, d: typing.Mapping[str, typing.Any]) -> None:
        self.d = dict(d)


def make_camera_device(test_context: AcquisitionTestContext.test_context, camera_channel: typing.Optional[str]) -> Acquisition.AcquisitionDeviceLike:
    return camera_base.CameraAcquisitionDevice(test_context.camera_hardware_source, test_context.camera_hardware_source.get_frame_parameters(0), camera_channel)


def make_eels_device(test_context: AcquisitionTestContext.test_context, camera_channel: typing.Optional[str]) -> Acquisition.AcquisitionDeviceLike:
    return camera_base.CameraAcquisitionDevice(test_context.camera_hardware_source, test_context.camera_hardware_source.get_frame_parameters(0), camera_channel)


def make_scan_device(test_context: AcquisitionTestContext.test_context, *args: typing.Any) -> Acquisition.AcquisitionDeviceLike:
    return scan_base.ScanAcquisitionDevice(test_context.scan_hardware_source, test_context.scan_hardware_source.get_current_frame_parameters())


def _make_synchronized_device(test_context: AcquisitionTestContext.test_context, camera_channel: typing.Optional[str], *args: typing.Any) -> Acquisition.AcquisitionDeviceLike:
    scan_context_description = stem_controller.ScanSpecifier()
    scan_context_description.scan_context_valid = True
    scan_context_description.scan_size = Geometry.IntSize(6, 4)
    scan_frame_parameters = test_context.scan_hardware_source.get_current_frame_parameters()
    test_context.scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters, typing.cast(typing.Tuple[int, int], scan_context_description.scan_size))
    return scan_base.SynchronizedScanAcquisitionDevice(test_context.scan_hardware_source, scan_frame_parameters,
                                                       test_context.camera_hardware_source,
                                                       test_context.camera_hardware_source.get_frame_parameters(0),
                                                       camera_channel, False, 0, 0, None, None, 0.0)


def make_synchronized_device(test_context: AcquisitionTestContext.test_context, camera_channel: typing.Optional[str], *args: typing.Any) -> Acquisition.AcquisitionDeviceLike:
    return _make_synchronized_device(test_context, camera_channel)


def make_slow_synchronized_device(test_context: AcquisitionTestContext.test_context, camera_channel: typing.Optional[str], *args: typing.Any) -> Acquisition.AcquisitionDeviceLike:
    # update period gets set to 0
    setattr(test_context.camera_hardware_source, "_update_period_for_testing", 0.0)
    return _make_synchronized_device(test_context, camera_channel)


def make_sequence_acquisition_method() -> Acquisition.AcquisitionMethodLike:
    return Acquisition.SequenceAcquisitionMethod(4)


def make_basic_acquisition_method() -> Acquisition.AcquisitionMethodLike:
    return Acquisition.BasicAcquisitionMethod()


def make_series_acquisition_method() -> Acquisition.AcquisitionMethodLike:
    control_customization = AcquisitionPreferences.ControlCustomization(Schema.get_entity_type("control_customization"), None)
    control_customization._set_field_value("control_id", "defocus")
    control_customization.device_control_id = "C10"
    control_customization.delay = 0
    control_values = numpy.stack([numpy.fromfunction(lambda x: 500e-9 + 5e-9 * x, (4,))], axis=-1)
    return Acquisition.SeriesAcquisitionMethod(control_customization, control_values)


def make_tableau_acquisition_method() -> Acquisition.AcquisitionMethodLike:
    control_customization = AcquisitionPreferences.ControlCustomization(Schema.get_entity_type("control_customization"), None)
    control_customization._set_field_value("control_id", "stage_position")
    control_customization.device_control_id = "stage_position_m"
    control_customization.delay = 0
    shape: typing.Final = (3, 3)
    control_values = numpy.stack([
        numpy.fromfunction(lambda y, x: -1e-9 + 1e-9 * y, shape),
        numpy.fromfunction(lambda y, x: -1e-9 + 1e-9 * x, shape)
    ], axis=-1)
    return Acquisition.TableAcquisitionMethod(control_customization, "tv", control_values)


def make_multi_acquisition_method() -> Acquisition.AcquisitionMethodLike:
    @dataclasses.dataclass
    class MultiSection:
        offset: float
        exposure: float
        count: int
        include_sum: bool

    return Acquisition.MultipleAcquisitionMethod([MultiSection(0.0, 0.025, 4, False), MultiSection(5.0, 0.05, 2, False)])


def make_multi_acquisition_with_sum_method() -> Acquisition.AcquisitionMethodLike:
    @dataclasses.dataclass
    class MultiSection:
        offset: float
        exposure: float
        count: int
        include_sum: bool

    return Acquisition.MultipleAcquisitionMethod([MultiSection(0.0, 0.025, 4, True), MultiSection(5.0, 0.05, 2, False)])


class TestCameraControlClass(unittest.TestCase):

    def setUp(self):
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()
        self.source_image = numpy.random.randn(1024, 1024).astype(numpy.float32)
        self.exposure = 0.005

    def tearDown(self):
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def _acquire_one(self, document_controller: DocumentController.DocumentController, hardware_source: camera_base.CameraHardwareSource) -> None:
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

    def __create_state_controller(self, acquisition_test_context: AcquisitionTestContext.AcquisitionTestContext, *,
                                  initialize: bool = True) -> CameraControlPanel.CameraControlStateController:
        state_controller = CameraControlPanel.CameraControlStateController(acquisition_test_context.camera_hardware_source,
                                                                           acquisition_test_context.document_controller)
        if initialize:
            state_controller.initialize_state()
        acquisition_test_context.push(state_controller)
        return state_controller

    def _test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        # subclasses may override this to provide a different configuration
        return AcquisitionTestContext.test_context(is_eels=is_eels, camera_exposure=self.exposure)

    ## STANDARD ACQUISITION TESTS ##

    # Do not change the comment above as it is used to search for places needing updates when a new
    # standard acquisition test is added.

    def test_acquiring_frames_with_generator_produces_correct_frame_numbers(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_acquiring_frames_with_generator_produces_correct_frame_numbers(self, hardware_source, document_controller)

    def test_acquire_multiple_frames_reuses_same_data_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_acquire_multiple_frames_reuses_same_data_item(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_stop_actually_stops_acquisition(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_stop_actually_stops_acquisition(self, hardware_source, document_controller)

    def test_simple_hardware_start_and_abort_works_as_expected(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_abort_works_as_expected(self, hardware_source, document_controller)

    def test_view_reuses_single_data_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_view_reuses_single_data_item(self, hardware_source, document_controller)

    def test_get_next_data_elements_to_finish_returns_full_frames(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_get_next_data_elements_to_finish_returns_full_frames(self, hardware_source, document_controller)

    def test_exception_during_view_halts_playback(self):
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_exception_during_view_halts_playback(self, hardware_source, self.exposure)

    def test_able_to_restart_view_after_exception(self):
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            HardwareSource_test._test_able_to_restart_view_after_exception(self, hardware_source, self.exposure)

    # End of standard acquisition tests.

    def test_view_generates_a_data_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self._acquire_one(document_controller, hardware_source)
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual(len(document_model.data_items), 1)

    def test_record_acquires_properly_binned_data(self):
        with self._test_context() as test_context:
            for binning in (1, 2):
                hardware_source = test_context.camera_hardware_source
                frame_parameters = hardware_source.get_frame_parameters(2)
                frame_parameters.binning = binning
                hardware_source.set_record_frame_parameters(frame_parameters)
                recording_task = hardware_source.start_recording(sync_timeout=3.0)
                try:
                    results = recording_task.grab_xdatas()
                    self.assertEqual(results[0].data.shape, hardware_source.get_expected_dimensions(binning))
                finally:
                    hardware_source.stop_recording(sync_timeout=3.0)

    def test_ability_to_set_profile_parameters_is_reflected_in_acquisition(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 2
            hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = hardware_source.get_frame_parameters(1)
            frame_parameters_1.binning = 1
            hardware_source.set_frame_parameters(1, frame_parameters_1)
            hardware_source.set_selected_profile_index(0)
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context() as test_context:
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
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing(sync_timeout=3.0)
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
            self.assertTrue(document_model.data_items[1].title.endswith(" Capture 1"))

    def test_capturing_during_view_captures_eels_2d(self):
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing(sync_timeout=3.0)
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

    def test_capturing_during_view_captures_session(self):
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            ApplicationData.get_session_metadata_model().microscopist = "Ned Flanders"
            hardware_source.start_playing(sync_timeout=3.0)
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
            self.assertEqual("Ned Flanders", document_model.data_items[2].session_metadata["microscopist"])

    def test_ability_to_start_playing_with_custom_parameters(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            frame_parameters_0 = hardware_source.get_frame_parameters(0)
            frame_parameters_0.binning = 4
            hardware_source.set_current_frame_parameters(frame_parameters_0)
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            binning_values = hardware_source.binning_values
            self.assertTrue(len(binning_values) > 0)
            self.assertTrue(all(map(lambda x: isinstance(x, int), binning_values)))

    def test_changing_binning_is_reflected_in_new_acquisition(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            long_exposure = 0.5
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            state_controller.handle_exposure_changed(long_exposure)
            start = time.perf_counter()
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                hardware_source.get_next_xdatas_to_finish()  # view again
                elapsed = time.perf_counter() - start
                document_controller.periodic()
                self.assertEqual(document_model.data_items[0].data_shape, hardware_source.get_expected_dimensions(4))
                self.assertTrue(elapsed > long_exposure)
            finally:
                hardware_source.abort_playing()

    def test_view_followed_by_frame_uses_correct_exposure(self):
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            long_exposure = 0.5
            state_controller.handle_change_profile("Snap")
            state_controller.handle_binning_changed("4")
            state_controller.handle_exposure_changed(long_exposure)
            state_controller.handle_change_profile("Run")
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                time.sleep(self.exposure * 0.5)
                data_and_metadata = hardware_source.get_next_xdatas_to_finish()[0]  # view again
                self.assertEqual(data_and_metadata.data_shape, hardware_source.get_expected_dimensions(2))
            finally:
                hardware_source.stop_playing(sync_timeout=3.0)
            state_controller.handle_change_profile("Snap")
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                start = time.perf_counter()
                data_and_metadata = hardware_source.get_next_xdatas_to_finish()[0]  # frame now
                elapsed = time.perf_counter() - start
            finally:
                hardware_source.abort_playing()
            self.assertEqual(data_and_metadata.data_shape, hardware_source.get_expected_dimensions(4))
            self.assertTrue(elapsed > long_exposure)

    def test_exception_during_view_leaves_buttons_in_ready_state(self):
        with self._test_context() as test_context:
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
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context() as test_context:
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
        with self._test_context(is_eels=True) as test_context:
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
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context(is_eels=True) as test_context:
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
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                for _ in range(4):
                    hardware_source.get_next_xdatas_to_finish()
                    document_controller.periodic()
            finally:
                hardware_source.abort_playing(sync_timeout=3.0)
            document_model.recompute_all()
            self.assertEqual(len(document_model.data_items), 2)
            # second acquisition
            first_data_items = copy.copy(data_items)
            hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context(is_eels=True) as test_context:
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

    def test_deleting_data_item_during_acquisition_recovers_correctly(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            hardware_source.start_playing(sync_timeout=TIMEOUT)
            try:
                hardware_source.get_next_xdatas_to_start()
                document_controller.periodic()
                document_model.remove_data_item(document_model.data_items[0])
                hardware_source.get_next_xdatas_to_start()
                document_controller.periodic()
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()

    def test_deleting_processed_data_item_during_acquisition_recovers_correctly(self):
        with self._test_context(is_eels=True) as test_context:
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

    def test_acquiring_eels_sets_signal_type_for_both_copies(self):
        # assumes EELS camera produces two data items
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual("eels", Metadata.get_metadata_value(document_model.data_items[0], "stem.signal_type"))
            self.assertEqual("eels", Metadata.get_metadata_value(document_model.data_items[1], "stem.signal_type"))

    def test_acquiring_eels_with_existing_invalid_summed_reference_succeeds(self):
        # this tests a problem that occurred when the data reference of an old non-existent data item exists
        # and the EELS starts and tries to regenerate the data item.
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            project = test_context.document_model._project
            project._set_persistent_property_value("data_item_references", {f"{test_context.eels_camera_device_id}_summed": str(uuid.uuid4())})
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)

    def test_acquiring_ronchigram_sets_signal_type(self):
        # assumes EELS camera produces two data items
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual("ronchigram", Metadata.get_metadata_value(document_model.data_items[0], "stem.signal_type"))

    def test_acquire_attaches_required_metadata(self):
        with self._test_context() as test_context:
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
        numpy_random_state = numpy.random.get_state()
        random_state = random.getstate()
        numpy.random.seed(999)
        random.seed(999)
        try:
            self.source_image = numpy.random.randn(1024, 1024).astype(numpy.float32)
            with self._test_context() as test_context:
                hardware_source = test_context.camera_hardware_source
                hardware_source.start_playing(sync_timeout=3.0)
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
        finally:
            random.setstate(random_state)
            numpy.random.set_state(numpy_random_state)

    def test_integrating_frames_updates_frame_count_by_integration_count(self):
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.integration_count = 4
            hardware_source.set_current_frame_parameters(frame_parameters)
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                frame0_integration_count = hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["integration_count"]
                frame1_integration_count = hardware_source.get_next_xdatas_to_finish()[0].metadata["hardware_source"]["integration_count"]
                self.assertEqual(frame0_integration_count, 4)
                self.assertEqual(frame1_integration_count, 4)
            finally:
                hardware_source.abort_playing(sync_timeout=TIMEOUT)

    def test_acquiring_attaches_timezone(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertIsNotNone(document_model.data_items[0].timezone)

    def test_acquire_sequence_2d_calibrations(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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

    def test_ronchigram_calibrations(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual("rad", document_model.data_items[0].dimensional_calibrations[0].units)
            self.assertEqual("rad", document_model.data_items[0].dimensional_calibrations[1].units)

    def test_eels_calibrations(self):
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, hardware_source)
            self.assertEqual("eV", document_model.data_items[0].dimensional_calibrations[1].units)
            self.assertEqual("eV", document_model.data_items[1].dimensional_calibrations[0].units)
            # note: it is an error to run view mode with "sum_project" enabled.
            # "sum_project" is for sequence/SI only.

    def test_acquire_with_probe_position(self):
        # used to test out the code path, but no specific asserts
        with self._test_context() as test_context:
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
        with self._test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            stem_controller = Registry.get_component("stem_controller")
            stem_controller.validate_probe_position()
            stem_controller._update_scan_context(Geometry.IntSize(256, 256), Geometry.FloatPoint(), 12.0, 0.0)
            self._acquire_one(document_controller, hardware_source)
            self.assertIsNotNone(document_model.data_items[0].timezone)

    def test_facade_frame_parameter_methods(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)
            record_task = hardware_source_facade.start_recording()
            record_task.wait_started(timeout=TIMEOUT)
            self.assertTrue(record_task.is_started)
            record_task.wait_finished(timeout=TIMEOUT)
            self.assertTrue(record_task.is_finished)

    def test_facade_abort_record(self):
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            self.assertFalse(hardware_source.is_recording)  # we know this works
            self.assertFalse(hardware_source_facade.is_recording)
            # hardware_source.stages_per_frame = 5
            record_task = hardware_source_facade.start_recording()
            record_task.wait_started(timeout=TIMEOUT)
            hardware_source_facade.abort_recording()
            record_task.wait_finished(timeout=TIMEOUT)
            # TODO: figure out a way to test whether abort actually aborts or just stops

    def test_facade_abort_record_and_return_data(self):
        with self._test_context() as test_context:
            hardware_source = test_context.camera_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            data_and_metadata_list = hardware_source_facade.record()
            data_and_metadata = data_and_metadata_list[0]
            self.assertIsNotNone(data_and_metadata.data)

    def test_facade_view_task(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
            graphic.close()
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            state_controller = self.__create_state_controller(test_context)
            self.assertEqual(state_controller.acquisition_state_model.value, "stopped")
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
                self.assertIn(state_controller.acquisition_state_model.value, ("partial", "complete"))
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
                document_controller.periodic()
            self.assertEqual(state_controller.acquisition_state_model.value, "stopped")

    def test_acquisition_state_after_exception_during_start(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            def raise_exception():
                raise Exception("Error during acquisition")
            hardware_source._test_start_hook = raise_exception
            hardware_source._test_acquire_exception = lambda *args: None
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()
            self.assertEqual(state_controller.acquisition_state_model.value, "error")

    def test_acquisition_state_after_exception_during_execute(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            def raise_exception():
                raise Exception("Error during acquisition")
            hardware_source._test_acquire_hook = raise_exception
            hardware_source._test_acquire_exception = lambda *args: None
            state_controller = self.__create_state_controller(test_context)
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                hardware_source.get_next_xdatas_to_finish(5)
                document_controller.periodic()
            finally:
                hardware_source.stop_playing(sync_timeout=TIMEOUT)
            document_controller.periodic()
            self.assertEqual(state_controller.acquisition_state_model.value, "error")

    def test_acquisition_panel_sequence_acquisition(self):
        # this test is complicated to set up because it is testing the UI.
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            hardware_source = test_context.camera_hardware_source
            c = Schema.Entity(Schema.get_entity_type("acquisition_device_component_camera"))
            c.camera_device_id = hardware_source.hardware_source_id
            c2 = Schema.Entity(Schema.get_entity_type("acquisition_method_component_sequence_acquire"))
            c2.count = 4
            acquisition_configuration = AcquisitionPanel.AcquisitionConfiguration(ApplicationDataInMemory())
            acquisition_configuration.acquisition_device_component_id = "camera"
            acquisition_configuration.acquisition_method_component_id = "sequence-acquire"
            acquisition_preferences = AcquisitionPreferences.AcquisitionPreferences(ApplicationDataInMemory())
            ac = AcquisitionPanel.AcquisitionController(document_controller, acquisition_configuration, acquisition_preferences)
            with contextlib.closing(ac):
                h = AcquisitionPanel.CameraAcquisitionDeviceComponentHandler(c, acquisition_preferences)
                with contextlib.closing(h):
                    h2 = AcquisitionPanel.SequenceAcquisitionMethodComponentHandler(c2, acquisition_preferences)
                    with contextlib.closing(h2):
                        ac.start_acquisition(h, h2)
                        start_time = time.time()
                        while ac.is_acquiring_model.value:
                            document_controller.periodic()
                            time.sleep(1/200)
                            self.assertTrue(time.time() - start_time < TIMEOUT)
                        self.assertFalse(ac.is_error)
            # only one data item will be created: the sequence. the view data item does not exist since acquiring
            # a sequence will use the special sequence acquisition of the camera device.
            self.assertEqual(1, len(document_controller.document_model.data_items))

    def test_immediate_sequence_acquisition(self):
        with self._test_context() as test_context:
            acquisition_device = make_scan_device(test_context)
            acquisition_method = make_sequence_acquisition_method()
            stem_device_controller = stem_controller.STEMDeviceController()
            device_map: typing.MutableMapping[str, stem_controller.DeviceController] = dict()
            device_map["stem"] = stem_device_controller
            device_data_stream = acquisition_device.build_acquisition_device_data_stream(device_map)
            data_stream = acquisition_method.wrap_acquisition_device_data_stream(device_data_stream, device_map)
            self.assertEqual((4, 256, 256), list(Acquisition.acquire_immediate(data_stream).values())[0].data_shape)

    def __test_acq(self,
                   document_controller: DocumentController.DocumentController,
                   acquisition_device: Acquisition.AcquisitionDeviceLike,
                   acquisition_method: Acquisition.AcquisitionMethodLike,
                   expected_dimensions: typing.Sequence[typing.Tuple[DataAndMetadata.ShapeType, DataAndMetadata.DataDescriptor]],
                   expected_error: bool = False,
                   error_handler: typing.Optional[typing.Callable[[Exception], None]] = None,
                   ) -> None:

        class DataChannelProvider(Acquisition.DataChannelProviderLike):
            def __init__(self, document_controller: DocumentController.DocumentController) -> None:
                self.__document_controller = document_controller

            def get_data_channel(self, title_base: str, channel_names: typing.Mapping[Acquisition.Channel, str], **kwargs: typing.Any) -> Acquisition.DataChannel:
                # create a data item data channel for converting data streams to data items, using partial updates and
                # minimizing extra copies where possible.

                # define a callback method to display the data item.
                def display_data_item(document_controller: DocumentController.DocumentController, data_item: DataItem.DataItem) -> None:
                    Facade.DocumentWindow(document_controller).display_data_item(Facade.DataItem(data_item))

                data_item_data_channel = DataChannel.DataItemDataChannel(self.__document_controller.document_model, title_base, channel_names)
                data_item_data_channel.on_display_data_item = functools.partial(display_data_item, self.__document_controller)

                return data_item_data_channel

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = pathlib.Path(temp_dir) / pathlib.Path("nion_acquisition_preferences.json")
            AcquisitionPreferences.init_acquisition_preferences(file_path)
            progress_value_model = Model.PropertyModel[int](0)
            is_acquiring_model = Model.PropertyModel[bool](False)
            data_channel_provider = DataChannelProvider(document_controller)
            stem_device_controller = stem_controller.STEMDeviceController()
            device_map: typing.MutableMapping[str, stem_controller.DeviceController] = dict()
            device_map["stem"] = stem_device_controller
            device_data_stream = acquisition_device.build_acquisition_device_data_stream(device_map)
            data_stream = acquisition_method.wrap_acquisition_device_data_stream(device_data_stream, device_map)
            device_data_stream = None
            drift_tracker = stem_device_controller.stem_controller.drift_tracker
            drift_logger = DriftTracker.DriftLogger(document_controller.document_model, drift_tracker, document_controller.event_loop) if drift_tracker else None
            acquisition = Acquisition.start_acquire(data_stream,
                                                    data_stream.title or str(),
                                                    data_stream.channel_names,
                                                    data_channel_provider,
                                                    drift_logger,
                                                    progress_value_model,
                                                    is_acquiring_model,
                                                    document_controller.event_loop,
                                                    lambda: None,
                                                    error_handler=error_handler)
            last_progress_time = time.time()
            last_progress = progress_value_model.value
            while is_acquiring_model.value:
                if time.time() - last_progress_time > TIMEOUT:
                    raise Exception(f"Timeout {TIMEOUT}s")
                document_controller.periodic()
                progress = progress_value_model.value
                if progress > last_progress:
                    last_progress = progress
                    last_progress_time = time.time()
                time.sleep(0.005)
            self.assertEqual(expected_error, acquisition.is_error)
            self.assertTrue(acquisition.is_finished)
            # useful for debugging
            # print(document_controller.document_model.data_items)
            # print([di.data_shape for di in document_controller.document_model.data_items])
            # print([di.title for di in document_controller.document_model.data_items])
            if expected_dimensions:
                self.assertEqual(len(expected_dimensions), len(document_controller.document_model.data_items))
                for data_item, expected_dimension in zip(document_controller.document_model.data_items, expected_dimensions):
                    self.assertEqual(expected_dimension[0], data_item.data_shape)
                    self.assertEqual(expected_dimension[1], data_item.data_and_metadata.data_descriptor)
                    if expected_metadata_fn := expected_dimension[2]:
                        self.assertTrue(expected_metadata_fn(data_item.data_and_metadata.data_metadata) if expected_metadata_fn else True)
            data_stream = None
            acquisition = None

    def test_acquisition_panel_acquisition(self):
        def ensure_camera_metadata(data_metadata: DataAndMetadata.DataMetadata) -> bool:
            return data_metadata.metadata.get("hardware_source", dict()).get("counts_per_electron", None) is not None

        tc = [
            # only one data item will be created: the sequence. the view data item does not exist since acquiring
            # a sequence will use the special sequence acquisition of the camera device.
            (make_camera_device, False, "ronchigram", make_sequence_acquisition_method,
             [((4, 1024, 1024), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata)]),

            # only one data item will be created: the sequence. the view data item does not exist since acquiring
            # a sequence will use the special sequence acquisition of the camera device.
            (make_eels_device, True, "eels_spectrum", make_sequence_acquisition_method,
             [((4, 512), DataAndMetadata.DataDescriptor(True, 0, 1), ensure_camera_metadata)]),

            # only one data item will be created: the sequence. the view data item does not exist since acquiring
            # a sequence will use the special sequence acquisition of the camera device.
            (make_eels_device, True, "eels_image", make_sequence_acquisition_method,
             [((4, 128, 512), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata)]),

            # two data items will be created: the series and the camera view.
            (make_camera_device, False, "ronchigram", make_series_acquisition_method,
             [((4, 1024, 1024), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata),
              ((1024, 1024), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata)]),

            # three data items will be created: the series and the two camera view (full frame, summed).
            (make_eels_device, True, "eels_spectrum", make_series_acquisition_method,
             [((4, 512), DataAndMetadata.DataDescriptor(True, 0, 1), ensure_camera_metadata),
              ((128, 512), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
              ((512,), DataAndMetadata.DataDescriptor(False, 0, 1), ensure_camera_metadata)]),

            # three data items will be created: the series and the two camera view (full frame, summed).
            (make_eels_device, True, "eels_image", make_series_acquisition_method,
             [((4, 128, 512), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata),
              ((128, 512), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
              ((512,), DataAndMetadata.DataDescriptor(False, 0, 1), ensure_camera_metadata)]),

            # three data items will be created: the series and the two camera view (full frame, summed).
            (make_eels_device, True, "eels_image", make_multi_acquisition_method,
             [((4, 128, 512), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata),
              ((2, 128, 512), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata),
              ((128, 512), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
              ((512,), DataAndMetadata.DataDescriptor(False, 0, 1), ensure_camera_metadata)]),

            # data items: first multi-output (sequence), first multi-output (sum), second multi-output (sequence), camera view
            (make_camera_device, False, "ronchigram", make_multi_acquisition_with_sum_method,
             [
                 ((4, 1024, 1024), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata),
                 ((1024, 1024), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
                 ((2, 1024, 1024), DataAndMetadata.DataDescriptor(True, 0, 2), ensure_camera_metadata),
                 ((1024, 1024), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata)
             ]),

            # five data items will be created. scan/si for each section + haadf.
            (make_synchronized_device, True, "eels_spectrum", make_multi_acquisition_method,
             [
                 ((4, 6, 4), DataAndMetadata.DataDescriptor(True, 0, 2), None),
                 ((4, 6, 4, 512), DataAndMetadata.DataDescriptor(True, 2, 1), ensure_camera_metadata),
                 ((2, 6, 4), DataAndMetadata.DataDescriptor(True, 0, 2), None),
                 ((2, 6, 4, 512), DataAndMetadata.DataDescriptor(True, 2, 1), ensure_camera_metadata),
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None)
             ]),

            # five data items will be created. scan/si for each section + haadf.
            (make_synchronized_device, True, "eels_image", make_multi_acquisition_method,
             [
                 ((4, 6, 4), DataAndMetadata.DataDescriptor(True, 0, 2), None),
                 ((4, 6, 4, 128, 512), DataAndMetadata.DataDescriptor(True, 2, 2), ensure_camera_metadata),
                 ((2, 6, 4), DataAndMetadata.DataDescriptor(True, 0, 2), None),
                 ((2, 6, 4, 128, 512), DataAndMetadata.DataDescriptor(True, 2, 2), ensure_camera_metadata),
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None)
             ]),

            # two data items will be created: the series and the camera view.
            (make_camera_device, False, "ronchigram", make_tableau_acquisition_method,
             [((3, 3, 1024, 1024), DataAndMetadata.DataDescriptor(False, 2, 2), ensure_camera_metadata),
              ((1024, 1024), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata)]),

            # two data items will be created: the series and the scan view.
            (make_scan_device, False, None, make_sequence_acquisition_method,
             [((4, 256, 256), DataAndMetadata.DataDescriptor(True, 0, 2), None),
              ((256, 256), DataAndMetadata.DataDescriptor(False, 0, 2), None)]),

            # two data items will be created: the series and the scan view.
            (make_scan_device, False, None, make_series_acquisition_method,
             [((4, 256, 256), DataAndMetadata.DataDescriptor(True, 0, 2), None),
              ((256, 256), DataAndMetadata.DataDescriptor(False, 0, 2), None)]),

            # two data items will be created: the tableau and the scan view.
            (make_scan_device, False, None, make_tableau_acquisition_method,
             [((3, 3, 256, 256), DataAndMetadata.DataDescriptor(False, 2, 2), None),
              ((256, 256), DataAndMetadata.DataDescriptor(False, 0, 2), None)]),

            # single spectrum image
            # three data items will be created: the haadf, the spectrum image, amd the scan view.
            (make_synchronized_device, True, "eels_spectrum", make_basic_acquisition_method,
             [
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None),
                 ((6, 4, 512), DataAndMetadata.DataDescriptor(False, 2, 1), ensure_camera_metadata),
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None),
             ]),

            # single spectrum image
            # three data items will be created: the haadf, the spectrum image, amd the scan view.
            (make_slow_synchronized_device, True, "eels_spectrum", make_basic_acquisition_method,
             [
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None),
                 ((6, 4, 512), DataAndMetadata.DataDescriptor(False, 2, 1), ensure_camera_metadata),
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None),
             ]),

            # single spectrum image with 2d data
            # three data items will be created: the haadf, the spectrum image, amd the scan view.
            (make_synchronized_device, True, "eels_image", make_basic_acquisition_method,
             [
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None),
                 ((6, 4, 128, 512), DataAndMetadata.DataDescriptor(False, 2, 2), ensure_camera_metadata),
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None),
             ]),

            # sequence of spectrum image
            # three data items will be created: the haadf, the spectrum image, amd the scan view.
            (make_synchronized_device, True, "eels_spectrum", make_sequence_acquisition_method,
             [
                 ((4, 6, 4), DataAndMetadata.DataDescriptor(True, 0, 2), None),
                 ((4, 6, 4, 512), DataAndMetadata.DataDescriptor(True, 2, 1), ensure_camera_metadata),
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None),
             ]),

            # sequence of spectrum image with 2d data
            # three data items will be created: the camera series, the sync'd series, and the scan view.
            (make_synchronized_device, False, None, make_sequence_acquisition_method,
             [
                 ((4, 6, 4), DataAndMetadata.DataDescriptor(True, 0, 2), None),
                 ((4, 6, 4, 1024, 1024), DataAndMetadata.DataDescriptor(True, 2, 2), ensure_camera_metadata),
                 ((6, 4), DataAndMetadata.DataDescriptor(False, 0, 2), None)
             ]),

            # single camera image
            # two data items will be created: the result and the view.
            (make_camera_device, False, "ronchigram", make_basic_acquisition_method,
             [
                 ((1024, 1024), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
                 ((1024, 1024), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
             ]),

            # single eels image
            # three data items will be created: view image, camera image (image), camera image (spectrum)
            (make_camera_device, True, "eels_image", make_basic_acquisition_method,
             [
                 ((128, 512), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
                 ((128, 512), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
                 ((512,), DataAndMetadata.DataDescriptor(False, 0, 1), ensure_camera_metadata),
             ]),

            # single eels spectrum
            # three data items will be created: view image, camera image (image), camera image (spectrum)
            (make_camera_device, True, "eels_spectrum", make_basic_acquisition_method,
             [
                 ((512,), DataAndMetadata.DataDescriptor(False, 0, 1), ensure_camera_metadata),
                 ((128, 512), DataAndMetadata.DataDescriptor(False, 0, 2), ensure_camera_metadata),
                 ((512,), DataAndMetadata.DataDescriptor(False, 0, 1), ensure_camera_metadata),
             ]),

            # these acquisitions are not supported yet; there is no way to represent the results as data items.

            # make_synchronized_device, make_series_acquisition_method
            # make_synchronized_device, make_tableau_acquisition_method
        ]
        for acquisition_device_fn, is_eels, camera_channeL, acquisition_method_fn, expected_count in tc:
            with self.subTest(acquisition_device_fn=acquisition_device_fn, acquisition_method_fn=acquisition_method_fn, expected_count=expected_count):
                with self._test_context(is_eels=is_eels) as test_context:
                    self.__test_acq(test_context.document_controller, acquisition_device_fn(test_context, camera_channeL), acquisition_method_fn(), expected_count)

    def test_acquisition_panel_acquisition_restarts_view(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            acquisition_device = make_synchronized_device(test_context, None)
            acquisition_method = make_sequence_acquisition_method()
            try:
                # start hardware sources playing
                test_context.scan_hardware_source.start_playing(sync_timeout=3.0)
                test_context.camera_hardware_source.start_playing(sync_timeout=3.0)
                # run the acquisition procedure
                self.__test_acq(document_controller, acquisition_device, acquisition_method, [])
                # confirm the acquisition is still running
                self.assertTrue(test_context.scan_hardware_source.is_playing)
                self.assertTrue(test_context.camera_hardware_source.is_playing)
            finally:
                # stop the hardware sources
                test_context.scan_hardware_source.stop_playing(sync_timeout=3.0)
                test_context.camera_hardware_source.stop_playing(sync_timeout=3.0)

    def test_acquisition_panel_recovers_after_error_in_start_stream(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            acquisition_device = make_camera_device(test_context, "ronchigram")
            acquisition_method = make_series_acquisition_method()
            try:
                def raise_exception() -> None:
                    raise Exception("Error during acquisition")

                had_error = False

                def handle_error(e: Exception) -> None:
                    nonlocal had_error
                    had_error = True

                test_context.camera_hardware_source._test_start_hook = raise_exception
                test_context.camera_hardware_source._test_acquire_exception = lambda *args: None
                # run the acquisition procedure
                self.__test_acq(document_controller, acquisition_device, acquisition_method, [], True, handle_error)
                # confirm the acquisition is not running
                self.assertFalse(test_context.scan_hardware_source.is_playing)
                self.assertFalse(test_context.camera_hardware_source.is_playing)
                # confirm the error was handled
                self.assertTrue(had_error)
            finally:
                # stop the hardware sources
                test_context.scan_hardware_source.stop_playing(sync_timeout=3.0)
                test_context.camera_hardware_source.stop_playing(sync_timeout=3.0)

    def test_acquisition_panel_acquisition_runs_with_only_second_channel(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            test_context.scan_hardware_source.set_channel_enabled(0, False)
            test_context.scan_hardware_source.set_channel_enabled(1, True)
            acquisition_device = make_scan_device(test_context)
            acquisition_method = make_sequence_acquisition_method()
            # run the acquisition procedure
            self.__test_acq(document_controller, acquisition_device, acquisition_method, [])

    def test_acquisition_panel_scan_sequence_works_with_subscan(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            # this is an unintuitive way to set the subscan region, but use it until it is cleaned up.
            # the better way would be to be able to set it on the frame paramters and then set the current frame params.
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            acquisition_device = make_scan_device(test_context)
            acquisition_method = make_sequence_acquisition_method()
            # run the acquisition procedure
            self.__test_acq(document_controller, acquisition_device, acquisition_method, [])

    def test_acquisition_panel_scan_sequence_works_with_linescan(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            # this is an unintuitive way to set the subscan region, but use it until it is cleaned up.
            # the better way would be to be able to set it on the frame paramters and then set the current frame params.
            scan_hardware_source.line_scan_enabled = True
            acquisition_device = make_scan_device(test_context)
            acquisition_method = make_sequence_acquisition_method()
            # run the acquisition procedure
            self.__test_acq(document_controller, acquisition_device, acquisition_method, [])

    def test_exposure_string(self):
        t = (
            (1.2E-0,  0,  "1.2", "s"),
            (1.2E-0, -1,  "1.2", "s"),
            (1.2E-0, -2, "1.20", "s"),
            (1.2E-3, -3, "1.2", "ms"),
            (1.2E-3, -4, "1.2", "ms"),
            (1.2E-3, -5, "1.20", "ms"),
            (1.2E-6, -6, "1.2", "us"),
            (1.2E-6, -7, "1.2", "us"),
            (1.2E-6, -8, "1.20", "us"),
            (1.2E-9, -9, "1.2", "ns"),
            (1.2E-9, -10, "1.2", "ns"),
            (1.2E-9, -11, "1.20", "ns"),
        )
        for exposure, precision, exposure_s, units in t:
            self.assertEqual(exposure_s, CameraControlPanel.make_exposure_str(exposure, precision))
            self.assertEqual(units, CameraControlPanel.exposure_units[precision])

    def test_eels_summed_controller_is_recognized_when_dropping_data_into_display_panel(self):
        from nion.swift import DisplayPanel
        with self._test_context(is_eels=True) as test_context:
            hardware_source = test_context.camera_hardware_source
            display_panel_manager = DisplayPanel.DisplayPanelManager()
            display_panel_manager.register_display_panel_controller_factory("camera-live-" + hardware_source.hardware_source_id, CameraControlPanel.CameraDisplayPanelControllerFactory(hardware_source))
            try:
                document_controller = test_context.document_controller
                document_model = test_context.document_model
                state_controller = self.__create_state_controller(test_context)
                hardware_source.start_playing(sync_timeout=3.0)
                try:
                    hardware_source.get_next_xdatas_to_finish(5)
                    state_controller.use_processed_data = True
                finally:
                    hardware_source.stop_playing(sync_timeout=TIMEOUT)
                document_controller.periodic()
                display_panel = document_controller.selected_display_panel  # arbitrary
                c1 = display_panel_manager.detect_controller(document_model, document_model.data_items[0])
                c2 = display_panel_manager.detect_controller(document_model, document_model.data_items[1])
                controller1 = typing.cast(CameraControlPanel.CameraDisplayPanelController, display_panel_manager.make_display_panel_controller("camera-live", display_panel, c1))
                controller2 = typing.cast(CameraControlPanel.CameraDisplayPanelController, display_panel_manager.make_display_panel_controller("camera-live", display_panel, c2))
                self.assertIsInstance(controller1, CameraControlPanel.CameraDisplayPanelController)
                self.assertIsInstance(controller2, CameraControlPanel.CameraDisplayPanelController)
                self.assertFalse(controller1._show_processed_data)
                self.assertTrue(controller2._show_processed_data)
            finally:
                display_panel_manager.unregister_display_panel_controller_factory("camera-live-" + hardware_source.hardware_source_id)

    def test_camera_action_with_frame_parameters(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.exposure_ms = 14
            action_context = document_controller._get_action_context()
            action_context.parameters["hardware_source_id"] = hardware_source.hardware_source_id
            action_context.parameters["frame_parameters"] = frame_parameters.as_dict()
            document_controller.perform_action_in_context("acquisition.start_playing", action_context)
            start = time.time()
            while not hardware_source.is_playing:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < 3.0
            hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(14/1000, document_model.data_items[0].metadata["hardware_source"]["exposure"])

    def test_camera_action_with_no_frame_parameters(self) -> None:
        # essentially just testing that acquisition works.
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.camera_hardware_source
            action_context = document_controller._get_action_context()
            action_context.parameters["hardware_source_id"] = hardware_source.hardware_source_id
            document_controller.perform_action_in_context("acquisition.start_playing", action_context)
            start = time.time()
            while not hardware_source.is_playing:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < 3.0
            hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(1, len(document_model.data_items))

    def test_camera_calibrator(self) -> None:
        class InstrumentController:
            def __init__(self, values: dict[str, float]) -> None:
                self.values = values

            def TryGetVal(self, s: str) -> typing.Tuple[bool, typing.Optional[float]]:
                if s in self.values:
                    return True, self.values[s]
                else:
                    return False, None

        class CameraDevice:
            def __init__(self, camera_type: str) -> None:
                self.camera_type = camera_type

        class CameraFrameParameters:
            def __init__(self) -> None:
                self.binning = 2

        def calibration_equal(a: Calibration.Calibration, b: Calibration.Calibration) -> bool:
            a_scale_1000 = round(a.scale * 1000) if a.scale else None
            b_scale_1000 = round(b.scale * 1000) if b.scale else None
            a_offset_1000 = round(a.offset * 1000) if a.offset else None
            b_offset_1000 = round(b.offset * 1000) if b.offset else None
            return a_scale_1000 == b_scale_1000 and a_offset_1000 == b_offset_1000 and a.units == b.units

        instrument_controller = InstrumentController(
            {
                "angle": 0.01,
                "angle1": 0.011,
                "angle2": 0.012,
                "angle-old": 0.02,
                "angle-old1": 0.021,
                "angle-old2": 0.022,
                "intensity": 0.1,
                "intensity1": 0.11,
                "intensity2": 0.12,
                "intensity-old": 0.2,
                "intensity-old1": 0.21,
                "intensity-old2": 0.22,
                "index": 0,
                "index-old": 0
            }
        )

        # test simple ronchigram calibration, old style (lowercase)
        camera_device = CameraDevice("ronchigram")
        camera_frame_parameters = CameraFrameParameters()
        config = {
            "calibxscalecontrol": "angle-old",
            "calibyscalecontrol": "angle-old",
            "calibxoffsetcontrol": "",
            "calibyoffsetcontrol": "",
            "calibxunits": "rad-old",
            "calibyunits": "rad-old",
            "calibintensityscalecontrol": "intensity-old",
            "calibintensityoffsetcontrol": "",
            "calibintensityunits": "counts-old",
        }
        calibrator = camera_base.CalibrationControlsCalibrator2(instrument_controller, camera_device, config)
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.0, 0.04, "rad-old"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.0, 0.04, "rad-old"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.2, "counts-old"), intensity_calibration))

        # test simple ronchigram calibration, new style (capitalized)
        camera_device = CameraDevice("ronchigram")
        config = {
            "calibXScaleControl": "angle",
            "calibYScaleControl": "angle",
            "calibXOffsetControl": "",
            "calibYOffsetControl": "",
            "calibXUnits": "rad",
            "calibYUnits": "rad",
            "calibIntensityScaleControl": "intensity",
            "calibIntensityOffsetControl": "",
            "calibIntensityUnits": "counts",
        }
        calibrator = camera_base.CalibrationControlsCalibrator2(instrument_controller, camera_device, config)
        calibrations = calibrator.get_signal_calibrations(CameraFrameParameters(), (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.0, 0.02, "rad"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.0, 0.02, "rad"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(CameraFrameParameters())
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.1, "counts"), intensity_calibration))

        # test simple 1d/2d eels calibration, old style (lowercase)
        camera_device = CameraDevice("eels")
        config = {
            "calibxscalecontrol": "angle-old",
            "calibyscalecontrol": "angle-old",
            "calibxoffsetcontrol": "",
            "calibyoffsetcontrol": "",
            "calibxunits": "eV-old",
            "calibyunits": "y-old",
            "calibintensityscalecontrol": "intensity-old",
            "calibintensityoffsetcontrol": "",
            "calibintensityunits": "counts-old",
        }
        calibrator = camera_base.CalibrationControlsCalibrator2(instrument_controller, camera_device, config)
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.04, "y-old"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.04, "eV-old"), calibrations[1]))
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100,))
        self.assertEqual(1, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.04, "eV-old"), calibrations[0]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.2, "counts-old"), intensity_calibration))

        # test dynamic ronchigram calibration, new style (capitalized)
        camera_device = CameraDevice("eels")
        config = {
            "calibXScaleControl": "angle",
            "calibYScaleControl": "angle",
            "calibXOffsetControl": "",
            "calibYOffsetControl": "",
            "calibXUnits": "eV",
            "calibYUnits": "y",
            "calibIntensityScaleControl": "intensity",
            "calibIntensityOffsetControl": "",
            "calibIntensityUnits": "counts",
        }
        calibrator = camera_base.CalibrationControlsCalibrator2(instrument_controller, camera_device, config)
        camera_device.signal_type = None
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.02, "y"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.02, "eV"), calibrations[1]))
        camera_device.signal_type = "ronchigram"
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.0, 0.02, "y"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.0, 0.02, "eV"), calibrations[1]))
        camera_device.signal_type = None

        # test indexed ronchigram calibration, old style (lowercase)
        camera_device = CameraDevice("ronchigram")
        config = {
            "calibrationmodeindexcontrol": "index-old",
            "calibxscalecontrol": "angle-old",
            "calibyscalecontrol": "angle-old",
            "calibxscalecontrol1": "angle-old1",
            "calibyscalecontrol1": "angle-old1",
            "calibxscalecontrol2": "angle-old2",
            "calibyscalecontrol2": "angle-old2",
            "calibxoffsetcontrol": "",
            "calibyoffsetcontrol": "",
            "calibxoffsetcontrol1": "",
            "calibyoffsetcontrol1": "",
            "calibxoffsetcontrol2": "",
            "calibyoffsetcontrol2": "",
            "calibxunits": "rad-old",
            "calibyunits": "rad-old",
            "calibxunits1": "rad-old1",
            "calibyunits1": "rad-old1",
            "calibxunits2": "rad-old2",
            "calibyunits2": "rad-old2",
            "calibintensityscalecontrol": "intensity-old",
            "calibintensityscalecontrol1": "intensity-old1",
            "calibintensityscalecontrol2": "intensity-old2",
            "calibintensityoffsetcontrol": "",
            "calibintensityoffsetcontrol1": "",
            "calibintensityoffsetcontrol2": "",
            "calibintensityunits": "counts-old",
            "calibintensityunits1": "counts-old1",
            "calibintensityunits2": "counts-old2",
        }
        calibrator = camera_base.CalibrationControlsCalibrator2(instrument_controller, camera_device, config)
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.0, 0.04, "rad-old"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.0, 0.04, "rad-old"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.2, "counts-old"), intensity_calibration))
        instrument_controller.values["index-old"] = 1
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.1, 0.042, "rad-old1"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.1, 0.042, "rad-old1"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.21, "counts-old1"), intensity_calibration))
        instrument_controller.values["index-old"] = 2
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.2, 0.044, "rad-old2"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-2.2, 0.044, "rad-old2"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.22, "counts-old2"), intensity_calibration))
        instrument_controller.values["index-old"] = 0

        # test indexed ronchigram calibration, new style (capitalized)
        camera_device = CameraDevice("ronchigram")
        config = {
            "calibrationModeIndexControl": "index",
            "calibXScaleControl0": "angle",
            "calibYScaleControl0": "angle",
            "calibXScaleControl1": "angle1",
            "calibYScaleControl1": "angle1",
            "calibXScaleControl2": "angle2",
            "calibYScaleControl2": "angle2",
            "calibXOffsetControl0": "",
            "calibYOffsetControl0": "",
            "calibXOffsetControl1": "",
            "calibYOffsetControl1": "",
            "calibXOffsetControl2": "",
            "calibYOffsetControl2": "",
            "calibXUnits0": "rad",
            "calibYUnits0": "rad",
            "calibXUnits1": "rad1",
            "calibYUnits1": "rad1",
            "calibXUnits2": "rad2",
            "calibYUnits2": "rad2",
            "calibIntensityScaleControl0": "intensity",
            "calibIntensityScaleControl1": "intensity1",
            "calibIntensityScaleControl2": "intensity2",
            "calibIntensityOffsetControl0": "",
            "calibIntensityOffsetControl1": "",
            "calibIntensityOffsetControl2": "",
            "calibIntensityUnits0": "counts",
            "calibIntensityUnits1": "counts1",
            "calibIntensityUnits2": "counts2",
        }
        calibrator = camera_base.CalibrationControlsCalibrator2(instrument_controller, camera_device, config)
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.0, 0.02, "rad"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.0, 0.02, "rad"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.1, "counts"), intensity_calibration))
        instrument_controller.values["index"] = 1
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.1, 0.022, "rad1"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.1, 0.022, "rad1"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.11, "counts1"), intensity_calibration))
        instrument_controller.values["index"] = 2
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.2, 0.024, "rad2"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.2, 0.024, "rad2"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.12, "counts2"), intensity_calibration))
        instrument_controller.values["index"] = 0

        # test indexed ronchigram calibration, new style (capitalized), fallback
        camera_device = CameraDevice("ronchigram")
        config = {
            "calibrationModeIndexControl": "index_missing",
            "calibXScaleControl": "angle",
            "calibYScaleControl": "angle",
            "calibXScaleControl0": "angle1",
            "calibYScaleControl0": "angle1",
            "calibXOffsetControl0": "",
            "calibYOffsetControl0": "",
            "calibXUnits": "rad_fail",
            "calibYUnits": "rad_fail",
            "calibXUnits0": "rad1",
            "calibYUnits0": "rad1",
            "calibIntensityScaleControl": "intensity",
            "calibIntensityScaleControl0": "intensity1",
            "calibIntensityOffsetControl0": "",
            "calibIntensityUnits0": "counts1",
        }
        calibrator = camera_base.CalibrationControlsCalibrator2(instrument_controller, camera_device, config)
        calibrations = calibrator.get_signal_calibrations(camera_frame_parameters, (100, 100))
        self.assertEqual(2, len(calibrations))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.1, 0.022, "rad1"), calibrations[0]))
        self.assertTrue(calibration_equal(Calibration.Calibration(-1.1, 0.022, "rad1"), calibrations[1]))
        intensity_calibration = calibrator.get_intensity_calibration(camera_frame_parameters)
        self.assertTrue(calibration_equal(Calibration.Calibration(None, 0.11, "counts1"), intensity_calibration))

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
