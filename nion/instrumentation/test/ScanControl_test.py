import contextlib
import copy
import json
import logging
import math
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

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import ScanProfiles
from nion.instrumentation import stem_controller
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.instrumentation.test import HardwareSource_test
from nion.swift import DisplayPanel
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import ImportExportManager
from nion.swift.model import Metadata
from nion.swift.test import TestContext
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import TestUI
from nion.ui import UserInterface
from nion.ui import Window
from nion.utils import Geometry
from nion.utils import Observable
from nionswift_plugin.nion_instrumentation_ui import ScanControlPanel
from nionswift_plugin.nion_instrumentation_ui import ScanProfilesDialog

"""
# running in Swift
import sys, unittest
from superscan import SimulatorScanControl_test
suite = unittest.TestLoader().loadTestsFromTestCase(SimulatorScanControl_test.TestSimulatorScanControlClass)
result = unittest.TextTestResult(sys.stdout, True, True)
suite.run(result)
"""


class FrameParametersScanRecorder(Observable.Observable):
    """A scan recorder for the scan profile dialog which keeps the frame parameters it is asked to record with, instead of recording."""

    def __init__(self) -> None:
        super().__init__()
        self.record_button_enabled = True
        self.profile_activation_enabled = True
        self.recorded_frame_parameters_list: list[scan_base.ScanFrameParameters | None] = list()

    def handle_record_button_clicked(self, *, frame_parameters: scan_base.ScanFrameParameters | None = None) -> None:
        self.recorded_frame_parameters_list.append(frame_parameters)


class TestScanControlClass(unittest.TestCase):

    def setUp(self):
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def assertAlmostEqualPoint(self, p1, p2, e=0.00001):
        if not(Geometry.distance(p1, p2) < e):
            logging.info("%s != %s", p1, p2)
        self.assertTrue(Geometry.distance(p1, p2) < e)

    def _acquire_one(self, document_controller, hardware_source):
        hardware_source.start_playing(sync_timeout=3.0)
        hardware_source.stop_playing(sync_timeout=3.0)
        document_controller.periodic()

    def _record_one(self, document_controller: DocumentController.DocumentController, hardware_source: scan_base.ScanHardwareSource, scan_control_model: ScanControlPanel.ScanControlPanelModel, frame_parameters: scan_base.ScanFrameParameters | None = None) -> typing.Sequence[DataItem.DataItem]:
        frame_time = hardware_source.get_current_frame_time()

        enabled_channel_count = sum(hardware_source.get_channel_state(channel_index).enabled for channel_index in range(hardware_source.channel_count))

        recorded_data_items: list[DataItem.DataItem] = list()
        # the record queues each recorded data item to the main thread, so this runs from periodic on the main thread.
        def handle_record_data_item(data_item: DataItem.DataItem) -> None:
            recorded_data_items.append(data_item)

        scan_control_model.handle_record_button_clicked(frame_parameters=frame_parameters, completion_fn=handle_record_data_item)
        # the record starts on its own thread and its data items arrive only through periodic, so wait for both the
        # record to finish and every enabled channel's data item to arrive, running periodic while waiting.
        start_time = time.time()
        while hardware_source.is_recording or len(recorded_data_items) < enabled_channel_count:
            self.assertTrue(time.time() - start_time < 30.0)
            time.sleep(frame_time * 0.5)
            document_controller.periodic()
        return recorded_data_items

    def __create_scan_control_model(self, acquisition_test_context: AcquisitionTestContext.AcquisitionTestContext) -> ScanControlPanel.ScanControlPanelModel:
        scan_control_model = ScanControlPanel.ScanControlPanelModel(acquisition_test_context.scan_hardware_source, acquisition_test_context.document_controller)
        acquisition_test_context.push(scan_control_model)
        return scan_control_model

    def _test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        # subclasses may override this to provide a different configuration
        return AcquisitionTestContext.test_context()

    ## STANDARD ACQUISITION TESTS ##

    # Do not change the comment above as it is used to search for places needing updates when a new
    # standard acquisition test is added.

    def test_acquiring_frames_with_generator_produces_correct_frame_numbers(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_acquiring_frames_with_generator_produces_correct_frame_numbers(self, scan_hardware_source, document_controller)

    def test_acquire_multiple_frames_reuses_same_data_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_acquire_multiple_frames_reuses_same_data_item(self, scan_hardware_source, document_controller)

    def test_simple_hardware_start_and_stop_actually_stops_acquisition(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_stop_actually_stops_acquisition(self, scan_hardware_source, document_controller)

    def test_simple_hardware_start_and_abort_works_as_expected(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_simple_hardware_start_and_abort_works_as_expected(self, scan_hardware_source, document_controller)

    def test_record_only_acquires_one_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_record_only_acquires_one_item(self, scan_hardware_source, document_controller)

    def test_record_during_view_records_one_item_and_keeps_viewing(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_record_during_view_records_one_item_and_keeps_viewing(self, scan_hardware_source, document_controller)

    def test_abort_record_during_view_returns_to_view(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_abort_record_during_view_returns_to_view(self, scan_hardware_source, document_controller)

    def test_view_reuses_single_data_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_view_reuses_single_data_item(self, scan_hardware_source, document_controller)

    def test_get_next_data_elements_to_finish_returns_full_frames(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_get_next_data_elements_to_finish_returns_full_frames(self, scan_hardware_source, document_controller)

    def test_exception_during_view_halts_scan(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_exception_during_view_halts_playback(self, scan_hardware_source, scan_hardware_source.get_current_frame_time())

    def test_exception_during_record_halts_scan(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_selected_profile_index(2)
            HardwareSource_test._test_exception_during_record_halts_playback(self, scan_hardware_source, scan_hardware_source.get_current_frame_time())

    def test_able_to_restart_scan_after_exception_scan(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_able_to_restart_view_after_exception(self, scan_hardware_source, scan_hardware_source.get_current_frame_time())

    def test_record_starts_and_finishes_in_reasonable_time(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            HardwareSource_test._test_record_starts_and_finishes_in_reasonable_time(self, scan_hardware_source, scan_hardware_source.get_current_frame_time() * 16)

    # End of standard acquisition tests.

    def test_start_playing_with_frame_parameters(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            # ensure frame parameters can be passed as an object
            frame_parameters.size = Geometry.IntSize(16, 16)
            scan_hardware_source.start_playing(frame_parameters=frame_parameters, sync_timeout=3.0)
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(Geometry.IntSize(16, 16), document_controller.document_model.data_items[-1].data_shape)
            # ensure frame parameters can be passed as a dictionary
            frame_parameters.size = Geometry.IntSize(12, 12)
            scan_hardware_source.start_playing(frame_parameters=frame_parameters.as_dict(), sync_timeout=3.0)
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(Geometry.IntSize(12, 12), document_controller.document_model.data_items[-1].data_shape)

    def test_record_followed_by_view_updates_display(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            data_item_reference = scan_control_model.channels[0].data_item_reference
            displayed_data_item = [None]
            def display_data_item_changed():
                displayed_data_item[0] = data_item_reference.data_item
            data_item_reference_changed_listener = data_item_reference.data_item_reference_changed_event.listen(display_data_item_changed)
            self._record_one(document_controller, scan_hardware_source, scan_control_model)
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            frame_parameters = scan_hardware_source.get_frame_parameters(0)
            frame_parameters.pixel_time_us = 1
            scan_hardware_source.set_frame_parameters(0, frame_parameters)
            record_frame_parameters = scan_hardware_source.get_frame_parameters(2)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            self._record_one(document_controller, scan_hardware_source, scan_control_model, record_frame_parameters)
            self.assertEqual(len(document_model.data_items), 2)  # check assumptions
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(document_model.data_items[1].dimensional_shape, record_frame_parameters.size)
            self.assertNotEqual(document_model.data_items[0].dimensional_shape, document_model.data_items[1].dimensional_shape)

    def test_record_button_in_scan_control_panel_records_new_data_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            try:
                # acquire first, so that the live data item already exists and the record adds only its own data item.
                self._acquire_one(document_controller, scan_hardware_source)
                data_item_count = len(document_model.data_items)
                frame_time = scan_hardware_source.get_current_frame_time()
                scan_panel_controller.handle_record_button(widget)
                # the record runs on its own thread and adds its data item through periodic, so run periodic until the
                # record finishes and its data item arrives.
                start_time = time.time()
                while (scan_hardware_source.is_recording or len(document_model.data_items) == data_item_count) and time.time() - start_time < 30.0:
                    time.sleep(frame_time * 0.5)
                    document_controller.periodic()
                self.assertEqual(data_item_count + 1, len(document_model.data_items))
            finally:
                widget.close()

    def test_record_uses_current_frame_parameters(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            current_size = scan_hardware_source.get_current_frame_parameters().size
            self.assertNotEqual(current_size, scan_hardware_source.get_record_frame_parameters().size)  # check assumptions
            recorded_data_items = self._record_one(document_controller, scan_hardware_source, scan_control_model)
            self.assertEqual(current_size, Geometry.IntSize.make(recorded_data_items[0].dimensional_shape))

    def test_record_uses_current_subscan(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_tlhw(0.1, 0.2, 0.3, 0.4)
            subscan_pixel_size = scan_hardware_source.get_current_frame_parameters().subscan_pixel_size
            recorded_data_items = self._record_one(document_controller, scan_hardware_source, scan_control_model)
            self.assertEqual(subscan_pixel_size, Geometry.IntSize.make(recorded_data_items[0].dimensional_shape))

    def test_record_uses_current_subscan_under_one_pixel_high_with_pixel_width_override(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(1000, 1000)
            frame_parameters.subscan_pixel_width_override = 1000
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            scan_hardware_source.subscan_enabled = True
            # a region under one context pixel high, as dragging the subscan graphic can leave it
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_tlhw(0.6, 0.2, 0.0004, 0.3)
            subscan_pixel_size = scan_hardware_source.get_current_frame_parameters().subscan_pixel_size
            recorded_data_items = self._record_one(document_controller, scan_hardware_source, scan_control_model)
            self.assertEqual(subscan_pixel_size, Geometry.IntSize.make(recorded_data_items[0].dimensional_shape))

    def test_record_from_scan_profile_dialog_uses_profile_without_activating_it(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_index = profile_configuration_model.active_profile_index
            current_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            profile = profile_configuration_model.profiles[2]
            profile.size_linked = False
            profile.width = 64
            profile.height = 32
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            try:
                handler.show_profile_actions_menu(profile, Geometry.IntPoint())
                ui = document_controller.ui
                assert isinstance(ui, TestUI.UserInterface) and ui.popup
                record_callback = next(item.callback for item in ui.popup.items if item.title == "Record Once")
                assert record_callback
                record_callback()
                # the record adds its result as a data item apart from the scan channel's live data item, which appears
                # while the record runs. wait for the result and for the record to finish.
                live_data_item_reference = document_model.get_data_item_channel_reference(scan_hardware_source.hardware_source_id, scan_hardware_source.get_channel_id(0))
                start_time = time.time()
                while scan_hardware_source.is_recording or not (recorded_data_items := [data_item for data_item in document_model.data_items if data_item != live_data_item_reference.data_item]):
                    self.assertTrue(time.time() - start_time < 30.0)
                    time.sleep(0.01)
                    document_controller.periodic()
                self.assertEqual((32, 64), recorded_data_items[0].dimensional_shape)
                self.assertEqual(active_index, profile_configuration_model.active_profile_index)
                self.assertEqual(current_frame_parameters.size, scan_hardware_source.get_current_frame_parameters().size)
            finally:
                handler.close()

    def test_record_once_from_scan_profile_dialog_does_not_record_profile_removed_while_menu_is_open(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            profile_configuration_model = test_context.scan_hardware_source.scan_profiles
            scan_recorder = FrameParametersScanRecorder()
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_recorder, document_controller)
            try:
                handler.show_profile_actions_menu(profile_configuration_model.profiles[2], Geometry.IntPoint())
                ui = document_controller.ui
                assert isinstance(ui, TestUI.UserInterface) and ui.popup
                record_callback = next(item.callback for item in ui.popup.items if item.title == "Record Once")
                assert record_callback
                profile_configuration_model.remove_profile(2)
                record_callback()
                self.assertEqual(list(), scan_recorder.recorded_frame_parameters_list)
            finally:
                handler.close()

    def test_record_once_from_scan_profile_dialog_does_not_record_when_recording_becomes_impossible_while_menu_is_open(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            profile_configuration_model = test_context.scan_hardware_source.scan_profiles
            scan_recorder = FrameParametersScanRecorder()
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_recorder, document_controller)
            try:
                handler.show_profile_actions_menu(profile_configuration_model.profiles[2], Geometry.IntPoint())
                ui = document_controller.ui
                assert isinstance(ui, TestUI.UserInterface) and ui.popup
                record_callback = next(item.callback for item in ui.popup.items if item.title == "Record Once")
                assert record_callback
                # a recording started from the scan panel while the menu is open disables recording.
                scan_recorder.record_button_enabled = False
                record_callback()
                self.assertEqual(list(), scan_recorder.recorded_frame_parameters_list)
            finally:
                handler.close()

    def test_activate_button_in_scan_profile_configuration_dialog_makes_profile_active(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            active_index = profile_configuration_model.active_profile_index
            inactive_index = (active_index + 1) % len(profile_configuration_model.profiles)
            active_profile_handler = ScanProfilesDialog.ProfileHandler(profile_configuration_model, profile_configuration_model.profiles[active_index], scan_control_model, lambda profile, global_point: None)
            active_profile_widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, active_profile_handler)
            inactive_profile_handler = ScanProfilesDialog.ProfileHandler(profile_configuration_model, profile_configuration_model.profiles[inactive_index], scan_control_model, lambda profile, global_point: None)
            inactive_profile_widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, inactive_profile_handler)
            try:
                self.assertFalse(active_profile_handler.can_activate)
                self.assertTrue(inactive_profile_handler.can_activate)
                assert inactive_profile_handler.activate_button and inactive_profile_handler.activate_button.on_clicked
                inactive_profile_handler.activate_button.on_clicked()
                self.assertEqual(inactive_index, profile_configuration_model.active_profile_index)
                self.assertTrue(active_profile_handler.can_activate)
                self.assertFalse(inactive_profile_handler.can_activate)
            finally:
                active_profile_widget.close()
                inactive_profile_widget.close()

    def test_renaming_scan_profile_in_configuration_dialog_trims_the_name(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            profile_handler = ScanProfilesDialog.ProfileHandler(profile_configuration_model, profile_configuration_model.profiles[1], scan_control_model, lambda profile, global_point: None)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, profile_handler)
            try:
                name_line_edit = profile_handler.name_line_edit
                assert name_line_edit
                name_line_edit.editing_finished("  Survey  ")
                self.assertEqual("Survey", profile_configuration_model.profile_names[1])
                self.assertEqual("Survey", name_line_edit.text)
            finally:
                widget.close()

    def test_renaming_active_scan_profile_keeps_it_selected_in_scan_panel(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            try:
                active_index = profile_configuration_model.active_profile_index
                profile_configuration_model.rename_profile(profile_configuration_model.active_profile, "Survey")
                self.assertEqual(active_index, profile_configuration_model.active_profile_index)
                self.assertEqual(active_index, scan_panel_controller.profile_combo_box.current_index)
                self.assertEqual("Survey", scan_panel_controller.profile_combo_box.current_text)
            finally:
                widget.close()

    def test_scan_profile_configuration_button_reuses_open_dialog(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            try:
                scan_panel_controller.handle_profile_config_button(widget)
                dialog = scan_panel_controller.profile_configuration_dialog
                self.assertIsNotNone(dialog)
                scan_panel_controller.handle_profile_config_button(widget)
                self.assertIs(dialog, scan_panel_controller.profile_configuration_dialog)
                assert dialog
                dialog.request_close()
                self.assertIsNone(scan_panel_controller.profile_configuration_dialog)
                scan_panel_controller.handle_profile_config_button(widget)
                self.assertIsNotNone(scan_panel_controller.profile_configuration_dialog)
                self.assertIsNot(dialog, scan_panel_controller.profile_configuration_dialog)
            finally:
                if dialog := scan_panel_controller.profile_configuration_dialog:
                    dialog.request_close()
                widget.close()

    def test_escape_closes_scan_profile_configuration_dialog(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            try:
                scan_panel_controller.handle_profile_config_button(widget)
                dialog = scan_panel_controller.profile_configuration_dialog
                assert dialog
                self.assertTrue(dialog.key_pressed(document_controller.ui.create_key_by_id("escape")))
                self.assertIsNone(scan_panel_controller.profile_configuration_dialog)
            finally:
                if dialog := scan_panel_controller.profile_configuration_dialog:
                    dialog.request_close()
                widget.close()

    def test_closing_scan_panel_closes_its_scan_profile_configuration_dialog(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            scan_panel_controller.handle_profile_config_button(widget)
            dialog = scan_panel_controller.profile_configuration_dialog
            assert dialog
            closed_windows: list[Window.Window] = list()
            window_close_event_listener = dialog.window_close_event.listen(closed_windows.append)
            try:
                widget.close()
                self.assertEqual([dialog], closed_windows)
            finally:
                if not closed_windows:
                    dialog.request_close()
            window_close_event_listener = None

    def test_activating_scan_profile_in_one_scan_panel_activates_it_in_another(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_panel_controller1 = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget1 = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller1)
            scan_panel_controller2 = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget2 = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller2)
            try:
                self.assertEqual(0, scan_panel_controller2.profile_combo_box.current_index)
                scan_panel_controller1.profile_combo_box.current_index = 2
                self.assertEqual(2, scan_panel_controller2.profile_combo_box.current_index)
            finally:
                widget1.close()
                widget2.close()

    def test_previous_scan_profile_button_is_enabled_once_a_second_profile_is_activated(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            try:
                self.assertFalse(scan_panel_controller.previous_profile_button.enabled)
                scan_panel_controller.profile_combo_box.current_index = 1
                self.assertTrue(scan_panel_controller.previous_profile_button.enabled)
            finally:
                widget.close()

    def test_scan_profile_controls_in_scan_panel_are_disabled_during_sequence_acquisition(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            # activate a second profile so that the previous profile button is enabled outside the acquisition.
            scan_hardware_source.scan_profiles.active_profile_index = 1
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            # the acquisition thread waits in sequence mode until the controls have been checked.
            sequence_mode_started = threading.Event()
            controls_checked = threading.Event()

            def handle_property_changed(property_name: str) -> None:
                if property_name == "is_in_sequence_mode" and scan_hardware_source.is_in_sequence_mode:
                    sequence_mode_started.set()
                    controls_checked.wait(10.0)

            property_changed_event_listener = scan_hardware_source.property_changed_event.listen(handle_property_changed)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=16, w=16)
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(2)
            acquisition_thread = threading.Thread(target=Acquisition.acquire, args=(Acquisition.MakerDataStream(data_stream),))
            try:
                acquisition_thread.start()
                try:
                    self.assertTrue(sequence_mode_started.wait(10.0))
                    document_controller.periodic()
                    self.assertFalse(scan_panel_controller.profile_combo_box.enabled)
                    self.assertFalse(scan_panel_controller.previous_profile_button.enabled)
                finally:
                    controls_checked.set()
                    acquisition_thread.join(60.0)
                self.assertFalse(acquisition_thread.is_alive())
                document_controller.periodic()
                self.assertTrue(scan_panel_controller.profile_combo_box.enabled)
                self.assertTrue(scan_panel_controller.previous_profile_button.enabled)
            finally:
                property_changed_event_listener = None
                widget.close()

    def test_activate_button_in_scan_profile_configuration_dialog_is_disabled_during_sequence_acquisition(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            profile_handler = ScanProfilesDialog.ProfileHandler(profile_configuration_model, profile_configuration_model.profiles[1], scan_control_model, lambda profile, global_point: None)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, profile_handler)
            # the acquisition thread waits in sequence mode until the button has been checked.
            sequence_mode_started = threading.Event()
            button_checked = threading.Event()

            def handle_property_changed(property_name: str) -> None:
                if property_name == "is_in_sequence_mode" and scan_hardware_source.is_in_sequence_mode:
                    sequence_mode_started.set()
                    button_checked.wait(10.0)

            property_changed_event_listener = scan_hardware_source.property_changed_event.listen(handle_property_changed)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=16, w=16)
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(2)
            acquisition_thread = threading.Thread(target=Acquisition.acquire, args=(Acquisition.MakerDataStream(data_stream),))
            try:
                activate_button = profile_handler.activate_button
                assert activate_button
                acquisition_thread.start()
                try:
                    self.assertTrue(sequence_mode_started.wait(10.0))
                    document_controller.periodic()
                    self.assertFalse(activate_button.enabled)
                finally:
                    button_checked.set()
                    acquisition_thread.join(60.0)
                self.assertFalse(acquisition_thread.is_alive())
                document_controller.periodic()
                self.assertTrue(activate_button.enabled)
            finally:
                property_changed_event_listener = None
                widget.close()

    def test_scan_profile_combo_box_shows_active_profile_when_activation_during_sequence_acquisition_is_refused(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            active_profile_index = scan_hardware_source.scan_profiles.active_profile_index
            scan_panel_controller = ScanControlPanel.ScanPanelController(document_controller, scan_hardware_source)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller)
            combo_box_index_list: list[int | None] = list()

            # another profile is chosen in the combo box once the sequence acquisition has started. a user can't do this,
            # since the combo box is disabled during the acquisition, so this stands for a script which sets it.
            def handle_property_changed(property_name: str) -> None:
                if property_name == "is_in_sequence_mode" and scan_hardware_source.is_in_sequence_mode:
                    scan_panel_controller.profile_combo_box.current_index = active_profile_index + 1
                    combo_box_index_list.append(scan_panel_controller.profile_combo_box.current_index)

            property_changed_event_listener = scan_hardware_source.property_changed_event.listen(handle_property_changed)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=16, w=16)
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(2)
            try:
                Acquisition.acquire(Acquisition.MakerDataStream(data_stream))
                self.assertEqual([active_profile_index], combo_box_index_list)
            finally:
                property_changed_event_listener = None
                widget.close()

    def test_automatic_subscan_pixel_width_in_scan_profile_configuration_dialog_is_empty_text(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            profile = ScanProfiles.ProfileModel(str())
            profile.subscan_pixel_width = 64
            scan_control_model = self.__create_scan_control_model(test_context)
            profile_handler = ScanProfilesDialog.ProfileHandler(test_context.scan_hardware_source.scan_profiles, profile, scan_control_model, lambda profile, global_point: None)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, profile_handler)
            try:
                subscan_pixel_width_line_edit = profile_handler.subscan_pixel_width_line_edit
                assert subscan_pixel_width_line_edit
                subscan_pixel_width_line_edit.editing_finished(str())
                self.assertIsNone(profile.subscan_pixel_width)
                self.assertFalse(subscan_pixel_width_line_edit.text)
            finally:
                widget.close()

    def test_invalid_values_in_scan_profile_configuration_dialog_leave_values_unchanged(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            profile = ScanProfiles.ProfileModel(str())
            profile.field_width = 37.0
            profile.size_linked = False
            profile.width = 64
            profile.height = 32
            profile.subscan_field_width_nm = 10.0
            scan_control_model = self.__create_scan_control_model(test_context)
            profile_handler = ScanProfilesDialog.ProfileHandler(test_context.scan_hardware_source.scan_profiles, profile, scan_control_model, lambda profile, global_point: None)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, profile_handler)
            try:
                field_width_line_edit = profile_handler.field_width_line_edit
                assert field_width_line_edit
                for text in ("0", "-5", str()):
                    field_width_line_edit.editing_finished(text)
                    self.assertEqual(37.0, profile.field_width)
                    self.assertEqual("37.0", field_width_line_edit.text)
                width_line_edit = profile_handler.width_line_edit
                height_line_edit = profile_handler.height_line_edit
                assert width_line_edit and height_line_edit
                width_line_edit.editing_finished("0")
                height_line_edit.editing_finished("-1")
                self.assertEqual((64, 32, False), (profile.width, profile.height, profile.size_linked))
                self.assertEqual(("64", "32"), (width_line_edit.text, height_line_edit.text))
                # clearing the subscan width leaves it unchanged.
                subscan_field_width_line_edit = profile_handler.subscan_field_width_nm_line_edit
                assert subscan_field_width_line_edit
                subscan_field_width_line_edit.editing_finished(str())
                self.assertEqual(10.0, profile.subscan_field_width_nm)
                self.assertEqual("10.0", subscan_field_width_line_edit.text)
            finally:
                widget.close()

    def test_add_button_in_scan_profile_configuration_dialog_adds_profile_after_selected_profile(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                assert handler.add_button and handler.add_button.on_clicked
                profiles = list(profile_configuration_model.profiles)
                handler.current_index_model.value = 1
                handler.add_button.on_clicked()
                self.assertEqual(profiles[0:2], list(profile_configuration_model.profiles[0:2]))
                self.assertEqual(profiles[2:], list(profile_configuration_model.profiles[3:]))
                self.assertNotIn(profile_configuration_model.profiles[2], profiles)
                self.assertEqual(2, handler.current_index_model.value)
            finally:
                widget.close()

    def test_duplicate_button_in_scan_profile_configuration_dialog_adds_copy_after_selected_profile(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                assert handler.duplicate_button and handler.duplicate_button.on_clicked
                profile_count = len(profile_configuration_model.profiles)
                handler.current_index_model.value = 1
                handler.duplicate_button.on_clicked()
                self.assertEqual(profile_count + 1, len(profile_configuration_model.profiles))
                self.assertEqual(profile_configuration_model.profiles[1].write_values_to_dict(), profile_configuration_model.profiles[2].write_values_to_dict())
                self.assertEqual(2, handler.current_index_model.value)
            finally:
                widget.close()

    def test_move_down_button_in_scan_profile_configuration_dialog_moves_selected_profile(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                assert handler.move_down_button and handler.move_down_button.on_clicked
                profile = profile_configuration_model.profiles[1]
                handler.current_index_model.value = 1
                handler.move_down_button.on_clicked()
                self.assertIs(profile, profile_configuration_model.profiles[2])
                self.assertEqual(2, handler.current_index_model.value)
            finally:
                widget.close()

    def test_remove_button_in_scan_profile_configuration_dialog_removes_selected_profile(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                assert handler.remove_button and handler.remove_button.on_clicked
                profile_configuration_model.active_profile_index = 0
                profile = profile_configuration_model.profiles[1]
                handler.current_index_model.value = 1
                handler.remove_button.on_clicked()
                self.assertNotIn(profile, profile_configuration_model.profiles)
            finally:
                widget.close()

    def test_scan_profile_configuration_dialog_can_remove_selected_profile_only_when_it_is_inactive(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                profile_configuration_model.active_profile_index = 0
                handler.current_index_model.value = 0
                self.assertFalse(handler.can_remove)
                handler.current_index_model.value = 1
                self.assertTrue(handler.can_remove)
            finally:
                widget.close()

    def test_choosing_scan_profile_in_configuration_dialog_list_activates_it(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                profile_configuration_model.active_profile_index = 0
                list_view = handler.profile_list_view
                assert list_view and list_view.on_item_selected
                # choosing is a double click or return on the item.
                self.assertTrue(list_view.on_item_selected(2))
                self.assertIs(profile_configuration_model.profiles[2], profile_configuration_model.active_profile)
            finally:
                widget.close()

    def test_right_clicking_inactive_scan_profile_in_configuration_dialog_offers_to_activate_or_delete_it(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                profile_configuration_model.active_profile_index = 0
                list_view = handler.profile_list_view
                assert list_view and list_view.on_item_handle_context_menu
                ui = document_controller.ui
                assert isinstance(ui, TestUI.UserInterface)
                # right clicking a profile also selects it.
                list_view.on_item_handle_context_menu(0, 0, 0, 0, 2)
                self.assertEqual(2, handler.current_index_model.value)
                assert ui.popup
                menu_items = {item.title: item for item in ui.popup.items}
                assert menu_items["Activate"].action and menu_items["Delete"].action and menu_items["Activate"].callback
                self.assertTrue(menu_items["Activate"].action.enabled)
                self.assertTrue(menu_items["Delete"].action.enabled)
                menu_items["Activate"].callback()
                self.assertIs(profile_configuration_model.profiles[2], profile_configuration_model.active_profile)
            finally:
                widget.close()

    def test_right_clicking_active_scan_profile_in_configuration_dialog_offers_neither_to_activate_nor_delete_it(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_control_model = self.__create_scan_control_model(test_context)
            handler = ScanProfilesDialog.ProfileConfigurationHandler(profile_configuration_model, scan_control_model, document_controller)
            widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, handler)
            try:
                profile_configuration_model.active_profile_index = 2
                list_view = handler.profile_list_view
                assert list_view and list_view.on_item_handle_context_menu
                ui = document_controller.ui
                assert isinstance(ui, TestUI.UserInterface)
                list_view.on_item_handle_context_menu(0, 0, 0, 0, 2)
                assert ui.popup
                menu_items = {item.title: item for item in ui.popup.items}
                assert menu_items["Activate"].action and menu_items["Delete"].action
                self.assertFalse(menu_items["Activate"].action.enabled)
                self.assertFalse(menu_items["Delete"].action.enabled)
            finally:
                widget.close()

    def test_record_names_results_correctly(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            self._record_one(document_controller, scan_hardware_source, scan_control_model)
            self.assertTrue(document_model.data_items[2].title.startswith(document_model.data_items[0].title))
            self.assertTrue(len(document_model.data_items[2].title) > len(document_model.data_items[0].title))
            self.assertTrue(document_model.data_items[3].title.startswith(document_model.data_items[1].title))
            self.assertTrue(len(document_model.data_items[3].title) > len(document_model.data_items[1].title))

    def test_enable_positioned_after_one_frame_acquisition_should_add_graphic(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 0)
            scan_control_model.probe_position_enabled = True
            document_controller.periodic()
            self.assertEqual(len(display_item.graphics), 1)

    def test_disable_positioned_after_one_frame_acquisition_should_remove_graphic(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 1)
            scan_control_model.probe_position_enabled = False
            document_controller.periodic()
            self.assertEqual(len(display_item.graphics), 0)

    def test_deleting_probe_graphic_after_one_frame_acquisition_should_disable_positioned(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            display_item.remove_graphic(probe_graphic).close()
            self.assertEqual(len(display_item.graphics), 0)
            self.assertIsNone(scan_hardware_source.probe_position)

    def test_deleting_display_after_one_frame_acquisition_should_disable_positioned(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            data_item = document_model.data_items[0]
            document_model.remove_data_item(data_item)
            self.assertEqual(len(document_model.data_items), 0)
            self.assertIsNone(scan_hardware_source.probe_position)

    def test_deleting_data_item_during_acquisition_recovers_correctly(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.scan_hardware_source
            hardware_source.start_playing(sync_timeout=3.0)
            try:
                hardware_source.get_next_xdatas_to_finish()
                data_item_reference = document_model.get_data_item_reference("test_scan_device_a")
                data_item_reference._test_delay = 0.05
                document_controller.periodic()
                hardware_source.get_next_xdatas_to_finish()
                document_model.remove_data_item(document_model.data_items[0])
                document_controller.periodic()
                hardware_source.get_next_xdatas_to_start()
                document_controller.periodic()
            finally:
                hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()

    # TODO: test case where one of two displays is deleted, what happens to probe graphic on other one?
    # TODO: test case where probe graphic should be removed when a channel is disabled

    def test_start_playing_after_stopping_should_change_probe_position_graphic_to_gray(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(len(display_item.graphics), 1)
            self.assertEqual("#F80", display_item.graphics[-1].stroke_color)
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
            document_controller.periodic()
            self.assertEqual(len(display_item.graphics), 1)
            self.assertEqual("#888", display_item.graphics[-1].stroke_color)
            scan_hardware_source.stop_playing()

    def test_context_scan_attaches_required_metadata(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
                self.assertEqual(channel_index % 4, Metadata.get_metadata_value(metadata_source, "stem.scan.channel_index"))
                self.assertEqual(scan_hardware_source.get_channel_state(channel_index % 4).channel_id, Metadata.get_metadata_value(metadata_source, "stem.scan.channel_id"))
                self.assertEqual(scan_hardware_source.get_channel_state(channel_index % 4).name + " Subscan", Metadata.get_metadata_value(metadata_source, "stem.scan.channel_name"))
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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

    def test_start_recording_with_no_channels_enabled_does_nothing(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, False)
            frame_time = scan_hardware_source.get_current_frame_time()
            recording_task = scan_hardware_source.start_recording(sync_timeout=3.0)
            recording_task.grab_xdatas(timeout=3.0)
            scan_hardware_source.stop_recording(sync_timeout=3.0)
            self.assertFalse(scan_hardware_source.is_recording)
            self.assertFalse(scan_hardware_source.is_playing)

    def test_enabling_channel_during_acquisition_results_in_write_delayed_data_item(self):
        # this ensures that data items freshly created during acquisition don't write repeatedly to disk.
        # see also test_data_item_created_during_acquisition_is_write_delayed_during_and_not_after
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                # the frame's data channel update, which creates the data item, comes before the frame finishes.
                scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)  # throw out first scan, which can take longer due to initial conditions
            frame_time = scan_hardware_source.get_current_frame_time()
            scan_hardware_source.start_playing(sync_timeout=3.0)
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
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, False)
            scan_hardware_source.periodic()
            self.assertFalse(scan_control_model.scan_button_enabled)
            self.assertFalse(scan_control_model.record_button_enabled)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.periodic()
            self.assertTrue(scan_control_model.scan_button_enabled)
            self.assertTrue(scan_control_model.record_button_enabled)

    def test_record_generates_a_data_item(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            for i in range(3):
                self._record_one(document_controller, scan_hardware_source, scan_control_model)
            self.assertEqual(len(document_model.data_items), 4)  # 1 view image (always acquired), 3 records

    def test_ability_to_set_profile_parameters_is_reflected_in_acquisition(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            profiles = scan_hardware_source.scan_profiles.profiles
            profiles[0].width = 300
            profiles[0].height = 300
            profiles[1].width = 500
            profiles[1].height = 500
            scan_hardware_source.set_selected_profile_index(0)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (300, 300))
            scan_hardware_source.set_selected_profile_index(1)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (500, 500))

    def test_change_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            profiles = scan_hardware_source.scan_profiles.profiles
            profiles[0].width = 300
            profiles[0].height = 300
            profiles[1].width = 500
            profiles[1].height = 500
            scan_hardware_source.set_selected_profile_index(0)
            scan_hardware_source.start_playing()
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (300, 300))
            scan_hardware_source.set_selected_profile_index(1)
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (500, 500))
            scan_hardware_source.abort_playing()

    def test_mutating_current_profile_with_different_size_during_acquisition_should_produce_different_sized_data(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            # verify that the frame parameters are applied to the _next_ frame.
            frame_parameters_2 = scan_hardware_source.get_frame_parameters(2)
            frame_parameters_2.size = Geometry.IntSize(500, 500)
            frame_time = frame_parameters_2.pixel_time_us * 500 * 500 / 1000000.0
            recording_task = scan_hardware_source.start_recording(sync_timeout=3.0)
            time.sleep(frame_time * 0.1)
            scan_hardware_source.set_frame_parameters(2, frame_parameters_2)
            recorded_xdatas = recording_task.grab_xdatas(timeout=3.0)
            self.assertEqual(recorded_xdatas[0].data.shape, (1024, 1024))
            scan_hardware_source.stop_recording(sync_timeout=3.0)
            # confirm that recording is done, then verify that the frame parameters are actually applied to the _next_ frame.
            recording_task = scan_hardware_source.start_recording(sync_timeout=3.0)
            time.sleep(frame_time * 0.1)
            recorded_xdatas = recording_task.grab_xdatas(timeout=3.0)
            self.assertEqual(recorded_xdatas[0].data.shape, (500, 500))

    def test_changing_frame_parameters_during_sequence_does_not_affect_sequence(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(300, 300)
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_1 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_1.size = Geometry.IntSize(500, 500)
            scan_hardware_source.set_selected_profile_index(0)
            frame_count = 10
            scan_hardware_source.prepare_sequence_mode(frame_parameters_0, frame_count)
            scan_hardware_source.start_sequence_mode(frame_parameters_0, frame_count)
            try:
                self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (300, 300))
                scan_hardware_source.set_frame_parameters(0, frame_parameters_1)
                self.assertEqual(scan_hardware_source.get_next_xdatas_to_start()[0].data.shape, (300, 300))
            finally:
                scan_hardware_source.finish_sequence_mode()

    def test_capturing_during_view_captures_new_data_items(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)
            scan_control_model.handle_capture_button_clicked()
            # the capture listens before this wait does, so the frame this wait returns has also been given to the capture.
            scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 4)
            # check that the new data item titles are different
            self.assertNotEqual(document_model.data_items[2], document_model.data_items[3])
            # check that the new data item titles both have the same index
            self.assertTrue(document_model.data_items[2].title.endswith(" Capture 1"))
            self.assertTrue(document_model.data_items[3].title.endswith(" Capture 1"))

    def test_capturing_during_view_captures_session(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            ApplicationData.get_session_metadata_model().microscopist = "Ned Flanders"
            scan_hardware_source.set_channel_enabled(1, True)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
            document_controller.periodic()
            self.assertEqual(len(document_model.data_items), 2)
            scan_control_model.handle_capture_button_clicked()
            # the capture listens before this wait does, so the frame this wait returns has also been given to the capture.
            scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual("Ned Flanders", document_model.data_items[3].session_metadata["microscopist"])

    def test_capture_button_in_scan_display_panel_control_bar_captures_new_data_items(self):
        with self._test_context() as test_context:
            ScanControlPanel.run()
            try:
                document_controller = test_context.document_controller
                document_model = test_context.document_model
                scan_hardware_source = test_context.scan_hardware_source
                scan_hardware_source.set_channel_enabled(1, True)
                display_panel = document_controller.selected_display_panel
                display_panel.change_display_panel_content({"controller_type": "scan-live", "hardware_source_id": scan_hardware_source.hardware_source_id, "channel_id": scan_hardware_source.get_channel_state(0).channel_id})
                frame_parameters = scan_hardware_source.get_current_frame_parameters()
                frame_parameters.pixel_time_us = 4.0
                scan_hardware_source.start_playing(frame_parameters=frame_parameters, sync_timeout=3.0)
                scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
                document_controller.periodic()
                data_item_count = len(document_model.data_items)

                # find the Capture button in the display panel's control bar.
                def find_capture_button(canvas_item: CanvasItem.AbstractCanvasItem) -> CanvasItem.TextButtonCanvasItem | None:
                    if isinstance(canvas_item, CanvasItem.TextButtonCanvasItem) and canvas_item.text == "Capture":
                        return canvas_item
                    if isinstance(canvas_item, CanvasItem.CanvasItemComposition):
                        for child_canvas_item in canvas_item.canvas_items:
                            if capture_button := find_capture_button(child_canvas_item):
                                return capture_button
                    return None

                capture_button = find_capture_button(display_panel.footer_canvas_item)
                assert capture_button

                # hold the acquisition thread in a partial update of the next frame, so that the capture cannot finish
                # until the Capture button's state has been checked.
                frame_held_event = threading.Event()
                frame_released_event = threading.Event()

                # called from the acquisition thread
                def handle_data_channel_updated(data_channel_event_args: HardwareSource.DataChannelEventArgs, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
                    if data_channel_event_args.state == "partial" and not frame_held_event.is_set():
                        frame_held_event.set()
                        frame_released_event.wait(10.0)

                data_channel_updated_listener = scan_hardware_source.data_channel_updated_event.listen(handle_data_channel_updated)
                try:
                    self.assertTrue(frame_held_event.wait(10.0), "no partial frame update after the first frame finished")
                    # a click on the button reaches it as mouse_clicked.
                    capture_button.mouse_clicked(0, 0, CanvasItem.KeyboardModifiers())
                    document_controller.periodic()
                    self.assertFalse(capture_button.enabled)
                finally:
                    frame_released_event.set()

                # the capture listens before this wait does, so the capture has its frames when this wait returns.
                scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
                document_controller.periodic()
                self.assertEqual(data_item_count + 2, len(document_model.data_items))
                self.assertTrue(capture_button.enabled)
                scan_hardware_source.stop_playing(sync_timeout=3.0)
                document_controller.periodic()

                # keep the listener until acquisition stops
                data_channel_updated_listener = None
            finally:
                ScanControlPanel.stop()

    def test_changing_fov_during_capture_results_in_correct_calibration(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)

            # start playing and wait for one frame to finish
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.pixel_time_us = 4.0
            scan_hardware_source.start_playing(frame_parameters=scan_frame_parameters, sync_timeout=3.0)
            scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)
            document_controller.periodic()

            # hold the acquisition thread in the first partial frame update it gives this listener, so that the frame
            # cannot finish until the capture has been clicked and the fov changed. a partial update leaves the rest of
            # the frame to be acquired after the fov change. the acquisition thread alone writes held_frame_indexes, and
            # frame_held_event orders the write before this thread reads it.
            frame_held_event = threading.Event()
            frame_released_event = threading.Event()
            held_frame_indexes: list[int] = list()

            # called from the acquisition thread
            def handle_data_channel_updated(data_channel_event_args: HardwareSource.DataChannelEventArgs, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
                if data_channel_event_args.state == "partial" and not frame_held_event.is_set():
                    held_frame_indexes.append(data_and_metadata.metadata["hardware_source"]["frame_index"])
                    frame_held_event.set()
                    frame_released_event.wait(10.0)

            data_channel_updated_listener = scan_hardware_source.data_channel_updated_event.listen(handle_data_channel_updated)
            try:
                self.assertTrue(frame_held_event.wait(10.0), "no partial frame update after the first frame finished")

                # trigger capture. the capture takes the first frame to finish after the click, which is the held frame.
                scan_control_model.handle_capture_button_clicked()

                # change the fov during the capture
                scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
                self.assertAlmostEqual(100.0, scan_frame_parameters.fov_nm)
                scan_frame_parameters.fov_nm = 10.0
                scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
            finally:
                frame_released_event.set()

            # wait for a frame which starts after the fov change. the capture finishes first.
            next_xdatas = scan_hardware_source.get_next_xdatas_to_start(timeout=10.0)

            # stop playing
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()

            # keep the listener until acquisition stops
            data_channel_updated_listener = None

            # check that the fov change reached the acquisition
            next_xdata = next_xdatas[0]
            assert next_xdata
            self.assertEqual(10.0, next_xdata.metadata["scan"]["fov_nm"])

            # safety check that the capture is the frame which was in progress when the fov changed
            self.assertEqual(held_frame_indexes[0], document_model.data_items[1].metadata["hardware_source"]["frame_index"])

            # check fov
            self.assertAlmostEqual(100.0, document_model.data_items[1].dimensional_calibrations[0].convert_to_calibrated_size(document_model.data_items[1].data_shape[0]))
            self.assertEqual(100.0, document_model.data_items[1].metadata["scan"]["fov_nm"])

    def test_ability_to_start_playing_with_custom_parameters(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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

    def test_record_async_records_with_the_given_frame_parameters(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, test_context.scan_hardware_source)
            frame_parameters = typing.cast(scan_base.ScanFrameParameters, scan_hardware_source.get_record_frame_parameters())
            frame_parameters.pixel_size = Geometry.IntSize(100, 120)
            recorded_shapes: list[tuple[int, ...]] = list()
            recorded_event = threading.Event()

            def handle_record(data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:  # called from thread
                recorded_shapes.extend(data_promise.xdata.data_shape for data_promise in data_promises if data_promise.xdata)
                recorded_event.set()

            scan_hardware_source.record_async(handle_record, frame_parameters=frame_parameters)
            self.assertTrue(recorded_event.wait(10.0))
            self.assertEqual([(100, 120)], recorded_shapes)

    def test_record_async_records_the_subscan_which_the_given_frame_parameters_would_scan(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, test_context.scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            current_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(h=0.25, w=0.125)
            recorded_shapes: list[tuple[int, ...]] = list()
            recorded_event = threading.Event()

            def handle_record(data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:  # called from thread
                recorded_shapes.extend(data_promise.xdata.data_shape for data_promise in data_promises if data_promise.xdata)
                recorded_event.set()

            scan_hardware_source.record_async(handle_record, frame_parameters=frame_parameters)
            self.assertTrue(recorded_event.wait(10.0))
            self.assertEqual(current_frame_parameters.as_dict(), scan_hardware_source.get_current_frame_parameters().as_dict())
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            subscan_pixel_size = scan_hardware_source.get_current_frame_parameters().subscan_pixel_size
            assert subscan_pixel_size
            self.assertEqual([tuple(subscan_pixel_size)], recorded_shapes)

    def test_sequence_acquisition_data_and_metadata_come_from_its_own_frame_parameters(self) -> None:
        # a script's sequence acquisition passes its own frame parameters, and its data must be described by them
        # whatever the current frame parameters are.
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, test_context.scan_hardware_source)
            # the current frame parameters differ from the acquisition's in size, field of view and pixel time.
            current_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            current_frame_parameters.pixel_size = Geometry.IntSize(h=32, w=32)
            current_frame_parameters.fov_nm = 80.0
            current_frame_parameters.pixel_time_us = 2.0
            scan_hardware_source.set_current_frame_parameters(current_frame_parameters)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=24, w=24)
            frame_parameters.fov_nm = 50.0
            frame_parameters.pixel_time_us = 3.0
            # acquire a sequence of three frames the way a script does.
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(3)
            maker = Acquisition.MakerDataStream(data_stream)
            device_state = maker.prepare_device_state()
            Acquisition.acquire(maker)
            device_state.restore()
            # the shape, calibrations and metadata describe the acquired frames, so they match the acquisition's frame parameters.
            xdata, = [maker.get_data(channel) for channel in maker.channels]
            self.assertEqual((3, 24, 24), xdata.data_shape)
            self.assertAlmostEqual(50.0 / 24, xdata.dimensional_calibrations[1].scale)
            self.assertAlmostEqual(50.0 / 24, xdata.dimensional_calibrations[2].scale)
            scan_metadata = xdata.metadata["scan"]
            self.assertEqual(50.0, scan_metadata["fov_nm"])
            self.assertEqual((24, 24), tuple(scan_metadata["scan_size"]))
            self.assertEqual(3.0, scan_metadata["scan_device_parameters"]["pixel_time_us"])
            self.assertEqual(3.0, xdata.metadata["hardware_source"]["pixel_time_us"])

    def test_sequence_acquisition_with_its_own_subscan_has_data_and_metadata_of_that_subscan(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, test_context.scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            # the current frame parameters have a subscan which differs from the acquisition's in size and position.
            current_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            current_frame_parameters.pixel_size = Geometry.IntSize(h=32, w=32)
            current_frame_parameters.fov_nm = 80.0
            current_frame_parameters.pixel_time_us = 2.0
            current_frame_parameters.subscan_fractional_center = Geometry.FloatPoint(y=0.5, x=0.5)
            current_frame_parameters.subscan_fractional_size = Geometry.FloatSize(h=0.25, w=0.25)
            scan_hardware_source.set_current_frame_parameters(current_frame_parameters)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=24, w=24)
            frame_parameters.fov_nm = 50.0
            frame_parameters.pixel_time_us = 3.0
            frame_parameters.subscan_fractional_center = Geometry.FloatPoint(y=0.3, x=0.6)
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(h=0.4, w=0.4)
            # the scan device calculates a 9 x 9 subscan for this region, so set it here so that the data stream expects the
            # size the device scans.
            frame_parameters.subscan_pixel_size = Geometry.IntSize(h=9, w=9)
            # acquire a sequence of three frames the way a script does.
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(3)
            maker = Acquisition.MakerDataStream(data_stream)
            device_state = maker.prepare_device_state()
            Acquisition.acquire(maker)
            device_state.restore()
            # the shape, calibrations and metadata describe the acquired subscan, so they match the acquisition's frame parameters.
            xdata, = [maker.get_data(channel) for channel in maker.channels]
            self.assertEqual((3, 9, 9), xdata.data_shape)
            self.assertAlmostEqual(50.0 * 0.4 / 9, xdata.dimensional_calibrations[2].scale)
            scan_metadata = xdata.metadata["scan"]
            self.assertEqual(50.0, scan_metadata["fov_nm"])
            self.assertEqual((9, 9), tuple(scan_metadata["scan_size"]))
            self.assertEqual((24, 24), tuple(scan_metadata["scan_device_parameters"]["pixel_size"]))
            self.assertEqual(3.0, scan_metadata["scan_device_parameters"]["pixel_time_us"])
            self.assertEqual((9, 9), tuple(scan_metadata["scan_device_parameters"]["subscan_pixel_size"]))
            self.assertEqual(3.0, xdata.metadata["hardware_source"]["pixel_time_us"])

    def test_sequence_acquisition_with_its_own_subscan_leaves_subscan_region_and_rotation_unchanged(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, test_context.scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            # the user places the subscan, as the subscan graphic does.
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.5, x=0.5), Geometry.FloatSize(h=0.25, w=0.25))
            scan_hardware_source.subscan_rotation = 0.1
            subscan_region = scan_hardware_source.subscan_region
            subscan_rotation = scan_hardware_source.subscan_rotation
            # acquire with a subscan which differs from the user's in position, size and rotation.
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=24, w=24)
            frame_parameters.subscan_fractional_center = Geometry.FloatPoint(y=0.3, x=0.6)
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(h=0.4, w=0.4)
            frame_parameters.subscan_rotation = 0.3
            # the scan device calculates a 9 x 9 subscan for this region, so set it here so that the data stream expects the
            # size the device scans.
            frame_parameters.subscan_pixel_size = Geometry.IntSize(h=9, w=9)
            # acquire a sequence of three frames the way a script does.
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(3)
            maker = Acquisition.MakerDataStream(data_stream)
            device_state = maker.prepare_device_state()
            Acquisition.acquire(maker)
            device_state.restore()
            self.assertEqual(subscan_region, scan_hardware_source.subscan_region, "the acquisition moved the user's subscan")
            self.assertAlmostEqual(subscan_rotation, scan_hardware_source.subscan_rotation, msg="the acquisition rotated the user's subscan")

    def test_record_async_with_its_own_frame_parameters_leaves_current_frame_parameters_and_active_profile_unchanged(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, test_context.scan_hardware_source)
            active_profile = scan_hardware_source.scan_profiles.active_profile
            active_profile_index = scan_hardware_source.scan_profiles.active_profile_index
            active_profile_values = active_profile.write_values_to_dict()
            current_frame_parameters_dict = scan_hardware_source.get_current_frame_parameters().as_dict()
            # record with a size, field of view and pixel time which differ from the current frame parameters.
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=24, w=24)
            frame_parameters.fov_nm = 50.0
            frame_parameters.pixel_time_us = 3.0
            recorded_event = threading.Event()

            def handle_record(data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:  # called from thread
                recorded_event.set()

            scan_hardware_source.record_async(handle_record, frame_parameters=frame_parameters)
            self.assertTrue(recorded_event.wait(10.0))
            start_time = time.time()
            while scan_hardware_source.is_recording:
                self.assertTrue(time.time() - start_time < 30.0)
                time.sleep(0.01)
            document_controller.periodic()
            self.assertEqual(current_frame_parameters_dict, scan_hardware_source.get_current_frame_parameters().as_dict())
            self.assertEqual(active_profile_index, scan_hardware_source.scan_profiles.active_profile_index)
            self.assertIs(active_profile, scan_hardware_source.scan_profiles.active_profile)
            self.assertEqual(active_profile_values, active_profile.write_values_to_dict())

    def test_changing_profile_updates_frame_parameters_in_ui(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            profile = scan_hardware_source.scan_profiles.profiles[1]
            scan_hardware_source.set_selected_profile_index(1)
            frame_parameters = scan_control_model._scan_frame_parameters_for_testing
            self.assertIsNotNone(frame_parameters)
            self.assertEqual((profile.field_width, profile.width), (frame_parameters.fov_nm, frame_parameters.size.width))

    def test_scan_button_scans_with_active_scan_profile_other_than_first(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile_index = 1
            active_profile = profile_configuration_model.active_profile
            field_width, width = active_profile.field_width, active_profile.width
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.handle_scan_button_clicked()
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            self.assertEqual((field_width, width), (frame_parameters.fov_nm, frame_parameters.pixel_size.width), "the scan does not use the active profile")
            self.assertEqual((field_width, width), (active_profile.field_width, active_profile.width), "the active profile lost its values")

    def test_scan_button_scans_with_active_scan_profile_restored_from_saved_file(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.scan_profiles.active_profile_index = 1
            properties = scan_hardware_source.scan_profiles.write_to_dict()
            # a scan device as at the next launch, which loads the saved profiles file.
            scan_module = AcquisitionTestContextConfiguration.ScanModule(test_context.instrument, "test_restored_profiles_scan_device", AcquisitionTestContextConfiguration.ScanDataGenerator())
            configuration_location = pathlib.Path(test_context.configuration.configuration_location)
            (configuration_location / "test_restored_profiles_scan_device_scan_profiles.json").write_text(json.dumps(properties))
            restored_scan_hardware_source = scan_base.ConcreteScanHardwareSource(test_context.instrument, scan_module.device, scan_module.settings, configuration_location)
            # the scan button starts the scan through an action which finds the scan device by its id.
            HardwareSource.HardwareSourceManager().register_hardware_source(restored_scan_hardware_source)
            try:
                active_profile = restored_scan_hardware_source.scan_profiles.active_profile
                field_width, width = active_profile.field_width, active_profile.width
                scan_control_model = ScanControlPanel.ScanControlPanelModel(restored_scan_hardware_source, test_context.document_controller)
                try:
                    scan_control_model.handle_scan_button_clicked()
                    restored_scan_hardware_source.stop_playing(sync_timeout=3.0)
                finally:
                    scan_control_model.close()
                frame_parameters = restored_scan_hardware_source.get_current_frame_parameters()
                self.assertEqual((field_width, width), (frame_parameters.fov_nm, frame_parameters.pixel_size.width), "the scan does not use the restored active profile")
                self.assertEqual((field_width, width), (active_profile.field_width, active_profile.width), "the restored active profile lost its values")
            finally:
                HardwareSource.HardwareSourceManager().unregister_hardware_source(restored_scan_hardware_source)
                restored_scan_hardware_source.close()

    def test_ac_line_sync_check_box_keeps_active_scan_profile_other_than_first(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile_index = 1
            active_profile = profile_configuration_model.active_profile
            field_width, width = active_profile.field_width, active_profile.width
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.ac_line_sync_enabled = True
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            self.assertTrue(frame_parameters.ac_line_sync)
            self.assertEqual((field_width, width), (frame_parameters.fov_nm, frame_parameters.pixel_size.width), "the scan does not use the active profile")
            self.assertEqual((field_width, width), (active_profile.field_width, active_profile.width), "the active profile lost its values")

    def test_rotation_field_changes_only_active_scan_profile_other_than_first(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            first_profile_rotation = profile_configuration_model.profiles[0].rotation
            profile_configuration_model.active_profile_index = 1
            active_profile = profile_configuration_model.active_profile
            field_width, width = active_profile.field_width, active_profile.width
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.rotation_deg_str = "12.5"
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            self.assertAlmostEqual(12.5, math.degrees(frame_parameters.rotation_rad))
            self.assertAlmostEqual(12.5, active_profile.rotation)
            self.assertEqual((field_width, width), (frame_parameters.fov_nm, frame_parameters.pixel_size.width), "the scan does not use the active profile")
            self.assertEqual((field_width, width), (active_profile.field_width, active_profile.width), "the active profile lost its values")
            self.assertEqual(first_profile_rotation, profile_configuration_model.profiles[0].rotation, "an inactive profile took the rotation")

    def test_scan_panel_field_edit_leaves_hardware_profile_slots_unchanged(self) -> None:
        # a device such as SuperScan keeps its hardware profile slots as its own profiles, which a field edit must not mix
        # with the values of the active scan profile.
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.scan_profiles.active_profile_index = 1
            slot_frame_parameters_list = [scan_hardware_source.get_frame_parameters(slot_index).as_dict() for slot_index in range(3)]
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.pixel_time_str = "7.5"
            self.assertAlmostEqual(7.5, scan_hardware_source.get_current_frame_parameters().pixel_time_us)
            self.assertEqual(slot_frame_parameters_list, [scan_hardware_source.get_frame_parameters(slot_index).as_dict() for slot_index in range(3)])

    def test_scan_device_report_of_its_hardware_profiles_keeps_active_scan_profile(self) -> None:
        # a device such as SuperScan reports its hardware profiles whenever its own settings change.
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile_index = 1
            active_profile = profile_configuration_model.active_profile
            field_width, width = active_profile.field_width, active_profile.width
            reported_frame_parameters_list: list[scan_base.ScanFrameParameters] = list()
            for slot_index in range(3):
                frame_parameters = scan_hardware_source.get_frame_parameters(slot_index)
                frame_parameters.fov_nm = 7.0
                frame_parameters.pixel_size = Geometry.IntSize(h=64, w=64)
                reported_frame_parameters_list.append(frame_parameters)
            channel_states = [(scan_hardware_source.get_channel_name(channel_index) or str(), scan_hardware_source.get_channel_enabled(channel_index)) for channel_index in range(scan_hardware_source.channel_count)]
            device = test_context.configuration.scan_module.device
            assert callable(device.on_device_state_changed)
            device.on_device_state_changed(reported_frame_parameters_list, channel_states)
            scan_hardware_source.periodic()
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            self.assertEqual((field_width, width), (frame_parameters.fov_nm, frame_parameters.pixel_size.width), "the scan took the reported hardware profile")
            self.assertEqual((field_width, width), (active_profile.field_width, active_profile.width), "the active profile took the reported hardware profile")

    def test_scan_panel_field_edit_during_sequence_acquisition_is_ignored(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            active_profile = scan_hardware_source.scan_profiles.active_profile
            pixel_time_us = active_profile.pixel_time_us
            scan_control_model = self.__create_scan_control_model(test_context)
            # the acquisition thread waits in sequence mode until the field has been edited.
            sequence_mode_started = threading.Event()
            field_edited = threading.Event()

            def handle_property_changed(property_name: str) -> None:
                if property_name == "is_in_sequence_mode" and scan_hardware_source.is_in_sequence_mode:
                    sequence_mode_started.set()
                    field_edited.wait(10.0)

            property_changed_event_listener = scan_hardware_source.property_changed_event.listen(handle_property_changed)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=16, w=16)
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(2)
            maker = Acquisition.MakerDataStream(data_stream)
            # the acquisition saves the device state before it starts and restores it when it ends, as the acquisition panel does.
            device_state = maker.prepare_device_state()
            acquisition_thread = threading.Thread(target=Acquisition.acquire, args=(maker,))
            try:
                acquisition_thread.start()
                try:
                    self.assertTrue(sequence_mode_started.wait(10.0))
                    document_controller.periodic()
                    scan_control_model.pixel_time_str = str(pixel_time_us * 2)
                    self.assertAlmostEqual(pixel_time_us, scan_hardware_source.get_current_frame_parameters().pixel_time_us, msg="the edit changed the acquisition's frame parameters")
                finally:
                    field_edited.set()
                    acquisition_thread.join(60.0)
                self.assertFalse(acquisition_thread.is_alive())
                device_state.restore()
                document_controller.periodic()
                self.assertAlmostEqual(pixel_time_us, active_profile.pixel_time_us)
            finally:
                property_changed_event_listener = None

    def test_changing_current_profiles_frame_parameters_updates_frame_parameters_in_ui(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            active_profile = scan_hardware_source.scan_profiles.active_profile
            active_profile.width = 200
            active_profile.height = 200
            self.assertIsNotNone(scan_control_model._scan_frame_parameters_for_testing)
            self.assertEqual(scan_control_model._scan_frame_parameters_for_testing.size, (200, 200))

    def test_selecting_scan_profile_by_index_activates_it_and_keeps_previously_active_profile(self) -> None:
        # a script selects a profile by its position, through the scripting API's profile_index.
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile_index = 1
            previous_profile = profile_configuration_model.active_profile
            field_width, width = previous_profile.field_width, previous_profile.width
            third_profile = profile_configuration_model.profiles[2]
            scan_hardware_source.set_selected_profile_index(2)
            self.assertIs(third_profile, profile_configuration_model.active_profile)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            self.assertEqual((third_profile.field_width, third_profile.width), (frame_parameters.fov_nm, frame_parameters.pixel_size.width), "the scan does not use the selected profile")
            self.assertEqual((field_width, width), (previous_profile.field_width, previous_profile.width), "the previously active profile lost its values")

    def test_selecting_scan_profile_by_index_with_no_profile_leaves_active_profile(self) -> None:
        # a script selects a profile by a position which has no profile.
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.active_profile
            fov_nm = scan_hardware_source.get_current_frame_parameters().fov_nm
            scan_hardware_source.set_selected_profile_index(99)
            self.assertIs(active_profile, profile_configuration_model.active_profile, "selecting a missing profile changed the active profile")
            self.assertEqual(fov_nm, scan_hardware_source.get_current_frame_parameters().fov_nm, "selecting a missing profile changed the scan")

    def test_selected_scan_profile_index_follows_active_scan_profile(self) -> None:
        # a script reads the selected profile by its position, through the scripting API's profile_index.
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            selected_profile_indexes: list[int] = list()
            profile_changed_event_listener = scan_hardware_source.profile_changed_event.listen(selected_profile_indexes.append)
            scan_hardware_source.scan_profiles.active_profile_index = 1
            self.assertEqual(1, scan_hardware_source.selected_profile_index)
            self.assertEqual([1], selected_profile_indexes)
            profile_changed_event_listener = None

    def test_changing_fov_updates_only_that_profile(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_control_model.width_height_linked = False
            scan_control_model.decrease_width()
            scan_control_model.decrease_height()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (128, 128))

    def test_record_during_view_uses_record_mode(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.fov_nm = 10
            frame_parameters_0.pixel_time_us = 10
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            frame_parameters_2 = scan_hardware_source.get_frame_parameters(2)
            frame_parameters_2.fov_nm = 100
            scan_hardware_source.set_frame_parameters(2, frame_parameters_2)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            self.assertEqual(Metadata.get_metadata_value(scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)[0], "stem.scan.fov_nm"), 10)
            # the recording task gives the record data however late it is grabbed. the record is delivered before its
            # finished callback, so the next frame to finish after the grab comes from the view.
            recording_task = scan_hardware_source.start_recording()
            self.assertEqual(Metadata.get_metadata_value(recording_task.grab_xdatas(timeout=10.0)[0], "stem.scan.fov_nm"), 100)
            self.assertEqual(Metadata.get_metadata_value(scan_hardware_source.get_next_xdatas_to_finish(timeout=10.0)[0], "stem.scan.fov_nm"), 10)

    def test_consecutive_frames_have_unique_data(self):
        # this test will fail if the scan is saturated (or otherwise produces identical values naturally)
        numpy_random_state = numpy.random.get_state()
        random_state = random.getstate()
        numpy.random.seed(999)
        random.seed(999)
        try:
            with self._test_context() as test_context:
                scan_hardware_source = test_context.scan_hardware_source
                frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
                frame_parameters_0.size = Geometry.IntSize(16, 16)
                frame_parameters_0.pixel_time_us = 2
                scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
                scan_hardware_source.start_playing()
                data_list = list()
                for i in range(6):
                    data = scan_hardware_source.get_next_xdatas_to_finish()[0].data
                    data_list.append(data)
                for row in range(0, len(data_list)):
                    s = slice(row, row+1), slice(0, 16)
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
            with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_control_model.decrease_width()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (128, 128))

    def test_changing_width_when_unlinked_does_not_change_height(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 256))
            scan_control_model.width_height_linked = False
            scan_control_model.decrease_width()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].data_shape, (256, 128))

    def test_obscenely_large_scan_does_not_crash(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
            frame_parameters_0.size = Geometry.IntSize(32768, 32768)
            frame_parameters_0.pixel_time_us = 1
            scan_hardware_source.set_frame_parameters(0, frame_parameters_0)
            scan_hardware_source.start_playing()
            scan_hardware_source.abort_playing()

    def test_closing_display_panel_with_scan_controller_shuts_down_controller_correctly(self):
        # NOTE: this is a duplicate of test_closing_display_panel_with_display_controller_shuts_down_controller_correctly
        with self._test_context() as test_context:
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
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                frame_time = scan_hardware_source.get_current_frame_time()
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

    def test_scan_controller_matches(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            display_panel_manager = DisplayPanel.DisplayPanelManager()
            ScanControlPanel.register_scan_panel(scan_hardware_source)
            try:
                self._acquire_one(document_controller, scan_hardware_source)
                d = display_panel_manager.detect_controller(document_model, document_model.data_items[-1])
                self.assertEqual("a", d["channel_id"])
                self.assertEqual("scan-live", d["controller_type"])
                self.assertEqual(scan_hardware_source.hardware_source_id, d["hardware_source_id"])
                scan_hardware_source.subscan_enabled = True
                self._acquire_one(document_controller, scan_hardware_source)
                d = display_panel_manager.detect_controller(document_model, document_model.data_items[-1])
                self.assertEqual("a_subscan", d["channel_id"])
                self.assertEqual("scan-live", d["controller_type"])
                self.assertEqual(scan_hardware_source.hardware_source_id, d["hardware_source_id"])
                # synthesize the drift data item
                document_model.append_data_item(document_model.data_items[-1].snapshot())
                metadata = document_model.data_items[-1].metadata
                metadata["hardware_source"]["channel_id"] = "drift"
                document_model.data_items[-1].metadata = metadata
                document_model.data_items[-1].category = "temporary"
                d = display_panel_manager.detect_controller(document_model, document_model.data_items[-1])
                self.assertEqual("drift", d["channel_id"])
                self.assertEqual("scan-live", d["controller_type"])
                self.assertEqual(scan_hardware_source.hardware_source_id, d["hardware_source_id"])
            finally:
                ScanControlPanel.unregister_scan_panel(scan_hardware_source)

    def test_probe_graphic_gets_closed(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            self.assertFalse(probe_graphic._closed)
            scan_control_model.probe_position_enabled = False
            document_controller.periodic()
            self.assertIsNone(scan_hardware_source.probe_position)
            self.assertTrue(probe_graphic._closed)

    def test_probe_appears_on_all_channels(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            display_item0 = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item1 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertIsNotNone(display_item0.graphics[0])
            self.assertIsNotNone(display_item1.graphics[0])
            scan_control_model.probe_position_enabled = False
            document_controller.periodic()
            self.assertIsNone(scan_hardware_source.probe_position)
            self.assertEqual(len(display_item0.graphics), 0)
            self.assertEqual(len(display_item1.graphics), 0)

    def test_probe_on_multiple_channels_shuts_down_properly(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.graphics[0].position = 0.3, 0.4
            document_controller.periodic()
            # will output extraneous messages when it fails

    def test_probe_on_multiple_channels_deletes_properly(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_control_model.probe_position_enabled = True
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            scan_control_model.probe_position_enabled = False
            document_controller.periodic()
            display_item0 = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item1 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(len(display_item0.graphics), 0)
            self.assertEqual(len(display_item1.graphics), 0)
            self.assertIsNone(scan_hardware_source.probe_position)

    def test_setting_probe_position_updates_probe_graphic(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.45, x=0.65))
            document_controller.periodic()
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)

    def test_setting_probe_graphic_updates_probe_position(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_control_model.probe_position_enabled = True
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            probe_graphic = display_item.graphics[0]
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            probe_graphic.position = Geometry.FloatPoint(y=0.45, x=0.65)
            document_controller.periodic()
            self.assertEqual(stem_controller_.probe_position, probe_graphic.position)
            self.assertEqual(stem_controller_.probe_position, scan_hardware_source._get_last_idle_position_for_test())

    def test_disabling_and_reenabling_probe_should_return_to_the_same_position(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_control_model.probe_position_enabled = True
            self.assertEqual(stem_controller_.probe_position, Geometry.FloatPoint(y=0.5, x=0.5))
            self._acquire_one(document_controller, scan_hardware_source)
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(1, len(display_item.graphics))
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.4, x=0.6))
            scan_control_model.probe_position_enabled = False
            document_controller.periodic()
            self.assertEqual(0, len(display_item.graphics))
            scan_control_model.probe_position_enabled = True
            document_controller.periodic()
            self.assertEqual(1, len(display_item.graphics))
            self.assertEqual(stem_controller_.probe_position, Geometry.FloatPoint(y=0.4, x=0.6))

    def test_acquire_into_empty_scan_controlled_display_panel(self):
        with self._test_context() as test_context:
            ScanControlPanel.run()
            document_controller = test_context.document_controller
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            display_panel = document_controller.selected_display_panel
            display_panel.change_display_panel_content({"controller_type": "scan-live", "hardware_source_id": scan_hardware_source.hardware_source_id, "channel_id": scan_hardware_source.get_channel_state(0).channel_id})
            stem_controller_.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))
            scan_control_model.probe_position_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            ScanControlPanel.stop()

    def test_switch_windows_while_acquiring(self):
        # this intermittently would lead to dangling async routines. leaving this here in case it occurs again.
        with self._test_context() as test_context:
            ScanControlPanel.run()
            scan_hardware_source = test_context.scan_hardware_source
            document_controller = test_context.document_controller
            document_controller.workspace_controller.create_panels(["scan-control-panel-" + scan_hardware_source.hardware_source_id])
            display_panel = document_controller.selected_display_panel
            display_panel.change_display_panel_content({"controller_type": "scan-live", "hardware_source_id": scan_hardware_source.hardware_source_id, "channel_id": scan_hardware_source.get_channel_state(0).channel_id})
            scan_hardware_source.start_playing(sync_timeout=3.0)
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = True
            document_controller.close()
            test_context.document_controller = test_context.create_document_controller(auto_close=False)
            test_context.document_model = test_context.document_controller.document_model
            document_controller = test_context.document_controller
            document_controller.workspace_controller.create_panels(["scan-control-panel-" + scan_hardware_source.hardware_source_id])
            document_controller.periodic()
            scan_hardware_source.stop_playing(sync_timeout=3.0)
            ScanControlPanel.stop()

    def test_subscan_state_starts_hidden_upon_first_acquisition(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            self.assertEqual(stem_controller.SubscanState.DISABLED_HIDDEN, scan_hardware_source.subscan_state)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(stem_controller.SubscanState.DISABLED_HIDDEN, scan_hardware_source.subscan_state)

    def test_subscan_properties_are_observable_on_stem_controller_and_scan_hardware_source(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            instrument = scan_hardware_source.stem_controller
            scan_notifications: typing.List[str] = list()
            instrument_notifications: typing.List[str] = list()
            scan_listener = scan_hardware_source.property_changed_event.listen(scan_notifications.append)
            instrument_listener = instrument.property_changed_event.listen(instrument_notifications.append)
            try:
                scan_hardware_source.subscan_state = stem_controller.SubscanState.ENABLED_VISIBLE
                self.assertEqual(stem_controller.SubscanState.ENABLED_VISIBLE, instrument.subscan_state)
                self.assertEqual(scan_hardware_source.subscan_region, instrument.subscan_region)
                self.assertIn("subscan_state", scan_notifications)
                self.assertIn("subscan_state", instrument_notifications)

                scan_notifications.clear()
                instrument_notifications.clear()

                subscan_region = Geometry.FloatRect.from_tlhw(0.1, 0.2, 0.3, 0.4)
                instrument.subscan_region = subscan_region
                self.assertEqual(subscan_region, scan_hardware_source.subscan_region)
                self.assertIn("subscan_region", scan_notifications)
                self.assertIn("subscan_region", instrument_notifications)

                scan_notifications.clear()
                instrument_notifications.clear()

                instrument.subscan_rotation = 0.125
                self.assertEqual(0.125, scan_hardware_source.subscan_rotation)
                self.assertIn("subscan_rotation", scan_notifications)
                self.assertIn("subscan_rotation", instrument_notifications)
            finally:
                scan_listener.close()
                instrument_listener.close()

    def test_stem_controller_scan_properties_raise_without_scan_controller(self):
        instrument = stem_controller.STEMController()
        self.addCleanup(instrument.close)
        with self.assertRaises(RuntimeError):
            _ = instrument.subscan_state
        with self.assertRaises(RuntimeError):
            instrument.subscan_region = Geometry.FloatRect.from_tlhw(0.1, 0.2, 0.3, 0.4)

    def test_line_scan_properties_are_observable_on_stem_controller_and_scan_hardware_source(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            instrument = scan_hardware_source.stem_controller
            scan_notifications: typing.List[str] = list()
            instrument_notifications: typing.List[str] = list()
            scan_listener = scan_hardware_source.property_changed_event.listen(scan_notifications.append)
            instrument_listener = instrument.property_changed_event.listen(instrument_notifications.append)
            try:
                instrument.line_scan_state = stem_controller.LineScanState.ENABLED_VISIBLE
                self.assertEqual(stem_controller.LineScanState.ENABLED_VISIBLE, scan_hardware_source.line_scan_state)
                self.assertEqual(scan_hardware_source.line_scan_vector, instrument.line_scan_vector)
                self.assertIn("line_scan_state", scan_notifications)
                self.assertIn("line_scan_state", instrument_notifications)

                scan_notifications.clear()
                instrument_notifications.clear()

                vector = ((0.1, 0.2), (0.3, 0.4))
                scan_hardware_source.line_scan_vector = vector
                self.assertEqual(vector, instrument.line_scan_vector)
                self.assertIn("line_scan_vector", scan_notifications)
                self.assertIn("line_scan_vector", instrument_notifications)
            finally:
                scan_listener.close()
                instrument_listener.close()

    def test_sequence_mode_is_observable_on_scan_hardware_source(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            # record the sequence mode at each notification, which arrives on the thread running the acquisition.
            in_sequence_mode_list: list[bool] = list()

            def handle_property_changed(property_name: str) -> None:
                if property_name == "is_in_sequence_mode":
                    in_sequence_mode_list.append(scan_hardware_source.is_in_sequence_mode)

            property_changed_event_listener = scan_hardware_source.property_changed_event.listen(handle_property_changed)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=16, w=16)
            data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(2)
            Acquisition.acquire(Acquisition.MakerDataStream(data_stream))
            self.assertEqual([True, False], in_sequence_mode_list)
            property_changed_event_listener = None

    def test_enabling_subscan_during_initial_acquisition_puts_graphic_on_context(self):
        with self._test_context() as test_context:
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

    def test_subscan_graphic_updates_when_subscan_region_is_set(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            subscan_graphic = next(graphic for graphic in display_item.graphics if graphic.graphic_id == "subscan")
            self.assertEqual(scan_hardware_source.subscan_region, subscan_graphic.bounds)

            subscan_region = Geometry.FloatRect.from_tlhw(0.1, 0.2, 0.3, 0.4)
            scan_hardware_source.subscan_region = subscan_region
            document_controller.periodic()

            self.assertEqual(subscan_region, scan_hardware_source.subscan_region)
            self.assertEqual(subscan_region, subscan_graphic.bounds)

    def test_subscan_graphic_updates_when_current_frame_parameters_change_subscan_region(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            subscan_graphic = next(graphic for graphic in display_item.graphics if graphic.graphic_id == "subscan")

            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            subscan_fractional_center = typing.cast(Geometry.FloatPoint, frame_parameters.subscan_fractional_center)
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(0.3, 0.4)
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            document_controller.periodic()

            expected_subscan_region = Geometry.FloatRect.from_center_and_size(subscan_fractional_center, Geometry.FloatSize(0.3, 0.4))
            self.assertEqual(expected_subscan_region, scan_hardware_source.subscan_region)
            self.assertEqual(expected_subscan_region, subscan_graphic.bounds)

            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            subscan_fractional_size = typing.cast(Geometry.FloatSize, frame_parameters.subscan_fractional_size)
            frame_parameters.subscan_fractional_center = Geometry.FloatPoint(0.35, 0.65)
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            document_controller.periodic()

            expected_subscan_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(0.35, 0.65), subscan_fractional_size)
            self.assertEqual(expected_subscan_region, scan_hardware_source.subscan_region)
            self.assertEqual(expected_subscan_region, subscan_graphic.bounds)

    def test_subscan_with_pixel_width_override_keeps_shape_when_frame_parameters_are_set_again(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(1000, 1000)
            frame_parameters.subscan_pixel_width_override = 1000
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            scan_hardware_source.subscan_enabled = True
            # a region under one context pixel high, as dragging the subscan graphic can leave it
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_tlhw(0.6, 0.2, 0.0004, 0.3)
            subscan_pixel_size = scan_hardware_source.get_current_frame_parameters().subscan_pixel_size
            # an unrelated edit sets the current frame parameters again
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_time_us = frame_parameters.pixel_time_us * 2
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertEqual(subscan_pixel_size, scan_hardware_source.get_current_frame_parameters().subscan_pixel_size)

    def test_reenabling_subscan_after_deleting_region_restores_previous_region(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            instrument = test_context.instrument
            self._acquire_one(document_controller, scan_hardware_source)

            saved_subscan_region = Geometry.FloatRect.from_tlhw(0.1, 0.2, 0.3, 0.4)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = saved_subscan_region
            document_controller.periodic()

            instrument.subscan_state = stem_controller.SubscanState.DISABLED_HIDDEN
            document_controller.periodic()
            self.assertFalse(scan_hardware_source.subscan_enabled)
            self.assertEqual(stem_controller.SubscanState.DISABLED_HIDDEN, instrument.subscan_state)
            self.assertEqual(saved_subscan_region, scan_hardware_source.subscan_region)

            scan_hardware_source.subscan_enabled = True
            document_controller.periodic()
            self.assertEqual(saved_subscan_region, scan_hardware_source.subscan_region)

    def test_removing_subscan_graphic_disables_subscan_when_acquisition_running(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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

    def test_line_scan_graphic_updates_when_line_scan_vector_is_set(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            scan_hardware_source.line_scan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            line_scan_graphic = next(graphic for graphic in display_item.graphics if graphic.graphic_id == "line_scan")

            line_scan_vector = ((0.2, 0.3), (0.8, 0.7))
            scan_hardware_source.line_scan_vector = line_scan_vector
            document_controller.periodic()

            self.assertEqual(line_scan_vector, scan_hardware_source.line_scan_vector)
            line_scan_graphic_vector = line_scan_graphic.vector
            self.assertEqual(line_scan_vector, (line_scan_graphic_vector[0].as_tuple(), line_scan_graphic_vector[1].as_tuple()))

    def test_line_scan_graphic_updates_when_current_frame_parameters_change_line_scan_vector(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.start_playing()
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            scan_hardware_source.line_scan_enabled = True
            scan_hardware_source.get_next_xdatas_to_finish()
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            line_scan_graphic = next(graphic for graphic in display_item.graphics if graphic.graphic_id == "line_scan")

            line_scan_vector = ((0.15, 0.25), (0.75, 0.65))
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.line_scan_vector = line_scan_vector
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            document_controller.periodic()

            self.assertEqual(line_scan_vector, scan_hardware_source.line_scan_vector)
            line_scan_graphic_vector = line_scan_graphic.vector
            self.assertEqual(line_scan_vector, (line_scan_graphic_vector[0].as_tuple(), line_scan_graphic_vector[1].as_tuple()))

    def test_removing_subscan_graphic_disables_subscan(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_control_model.subscan_checkbox_checked = True
            document_controller.periodic()
            self.assertTrue(scan_hardware_source.subscan_enabled)
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.remove_graphic(display_item.graphics[0]).close()
            document_controller.periodic()
            self.assertFalse(scan_hardware_source.subscan_enabled)

    def test_subscan_appears_on_display_items_for_additional_channels(self):
        # scan once, enable subscan, disable subscan. should be a graphic.
        # add channel. scan once. should be a graphic on both displays.
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_control_model.subscan_checkbox_checked = True
            document_controller.periodic()
            scan_control_model.subscan_checkbox_checked = False
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            self.assertEqual(1, len(document_model.display_items))
            self.assertEqual(1, len(display_item.graphics))
            scan_hardware_source.set_channel_enabled(1, True)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(2, len(document_model.display_items))
            self.assertEqual(1, len(display_item.graphics))
            display_item2 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(1, len(display_item2.graphics))

    def test_subscan_appears_only_on_enabled_channel_display_items(self):
        # enable two channels, scan once, disable 2nd channel. enable subscan. should be a graphic on one, not the other.
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(2, len(document_model.display_items))
            scan_hardware_source.set_channel_enabled(1, False)
            document_controller.periodic()
            scan_control_model.subscan_checkbox_checked = True
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item2 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(1, len(display_item.graphics))
            self.assertEqual(0, len(display_item2.graphics))

    def test_subscan_is_removed_when_channels_are_disabled(self):
        # enable two channels, scan once, enable subscan, disable 2nd channel. should be a graphic on one, not the other.
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(2, len(document_model.display_items))
            scan_control_model.subscan_checkbox_checked = True
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item2 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(1, len(display_item.graphics))
            self.assertEqual(1, len(display_item2.graphics))
            scan_hardware_source.set_channel_enabled(1, False)
            document_controller.periodic()
            # self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(1, len(display_item.graphics))
            self.assertEqual(0, len(display_item2.graphics))

    def test_subscan_is_added_when_channels_is_enabled(self):
        # enable two channels, scan once, disable 2nd channel, enable subscan. should be a graphic on one, not the other.
        # now enable 2nd channel again. should be a graphics on both.
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.set_channel_enabled(1, True)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(2, len(document_model.display_items))
            scan_hardware_source.set_channel_enabled(1, False)
            document_controller.periodic()
            scan_control_model.subscan_checkbox_checked = True
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item2 = document_model.get_display_item_for_data_item(document_model.data_items[1])
            self.assertEqual(1, len(display_item.graphics))
            self.assertEqual(0, len(display_item2.graphics))
            # same as test_subscan_appears_only_on_enabled_channel_display_items up to here
            scan_hardware_source.set_channel_enabled(1, True)
            document_controller.periodic()
            self.assertEqual(1, len(display_item.graphics))
            self.assertEqual(1, len(display_item2.graphics))

    def test_subscan_not_allowed_with_width_or_height_of_zero(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_control_model.subscan_checkbox_checked = True
            document_controller.periodic()
            display_item = document_model.get_display_item_for_data_item(document_model.data_items[0])
            display_item.graphics[0].size = 0, 0.25
            document_controller.periodic()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual((1, 64), document_model.data_items[1].data_shape)

    def test_scan_context_updated_when_changing_parameters(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
            self.assertEqual(20, scan_context4.fov_nm)

    def test_scan_context_handled_after_changing_parameters_during_subscan_and_stopping(self):
        test_cases = [
            # fov_nm, rotation_rad, and center_nm are context parameters and should clear the context if changed.
            # changing the size should only clear the context if the aspect ratio changes.
            # changing pixel time should not clear the context.
            (True, "fov_nm", 20, False),
            (True, "rotation_rad", 1.0, False),
            (True, "center_nm", Geometry.FloatPoint(0.1, 0.1), False),
            (True, "size", Geometry.IntSize(24, 24), True),
            (True, "size", Geometry.IntSize(24, 28), False),
            (False, "pixel_time_us", 5, True),
        ]
        for param_in_context, param_name, param_value, expected_context_valid in test_cases:
            with self._test_context() as test_context:
                scan_hardware_source = test_context.scan_hardware_source
                stem_controller_ = test_context.instrument
                scan_hardware_source.start_playing(sync_timeout=3.0)
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                scan_hardware_source.subscan_enabled = True
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                scan_context_before = copy.deepcopy(stem_controller_.scan_context)
                frame_parameters_0 = scan_hardware_source.get_frame_parameters(0)
                setattr(frame_parameters_0, param_name, param_value)
                scan_hardware_source.set_current_frame_parameters(frame_parameters_0)
                scan_hardware_source.get_next_xdatas_to_finish()  # grab at least one frame
                scan_hardware_source.stop_playing(sync_timeout=3.0)
                scan_context_after = copy.deepcopy(stem_controller_.scan_context)
                if param_in_context:
                    self.assertNotEqual(param_value, getattr(scan_context_before, param_name))
                    if expected_context_valid:
                        self.assertEqual(param_value, getattr(scan_context_after, param_name))

    def test_changing_rotation_and_fov_update_device_parameters_immediately(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            self._acquire_one(document_controller, scan_hardware_source)
            frame_parameters = copy.deepcopy(scan_hardware_source.scan_device.current_frame_parameters)
            fov_nm = frame_parameters.fov_nm
            frame_parameters.fov_nm //= 2
            scan_hardware_source.set_frame_parameters(0, frame_parameters)
            self.assertEqual(fov_nm // 2, scan_hardware_source.scan_device.current_frame_parameters.fov_nm)

    def test_removing_subscan_graphic_disables_subscan_when_acquisition_stopped(self):
        with self._test_context() as test_context:
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

    def test_facade_view_task_with_multiple_channels(self):
        with self._test_context() as test_context:
            hardware_source = test_context.scan_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(hardware_source.hardware_source_id, "~1.0")
            scan_frame_parameters = hardware_source_facade.get_frame_parameters_for_profile_by_index(0)
            view_task = hardware_source_facade.create_view_task(scan_frame_parameters, channels_enabled=[True, True])
            with contextlib.closing(view_task):
                data_and_metadata_list1 = view_task.grab_next_to_finish()
                self.assertTrue(hardware_source_facade.is_playing)
                data_and_metadata_list2 = view_task.grab_immediate()
                data_and_metadata_list3 = view_task.grab_next_to_start()
                self.assertEqual(2, len(data_and_metadata_list1))
                self.assertIsNotNone(data_and_metadata_list1[0].data)
                self.assertIsNotNone(data_and_metadata_list1[1].data)
                self.assertIsNotNone(data_and_metadata_list2[0].data)
                self.assertIsNotNone(data_and_metadata_list3[0].data)

    def test_facade_record_data_with_immediate_close(self):
        with self._test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            api = Facade.get_api("~1.0", "~1.0")
            hardware_source_facade = api.get_hardware_source_by_id(scan_hardware_source.hardware_source_id, "~1.0")
            scan_frame_parameters = hardware_source_facade.get_frame_parameters_for_profile_by_index(2)
            scan_frame_parameters["external_clock_wait_time_ms"] = 20000 # int(camera_frame_parameters["exposure_ms"] * 1.5)
            scan_frame_parameters["external_clock_mode"] = 1
            scan_frame_parameters["ac_line_sync"] = False
            scan_frame_parameters["ac_frame_sync"] = False
            # this tests an issue for a race condition where thread for record task isn't started before the task
            # is canceled, resulting in the close waiting for the thread and the thread waiting to acquire.
            # this reduces the problem, but it's still possible that during external sync, the acquisition starts
            # before being canceled and must time out.
            with contextlib.closing(hardware_source_facade.create_record_task(scan_frame_parameters)) as task:
                pass

    def test_enabling_subscan_changes_output_data_item(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            stem_controller_ = test_context.instrument
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            stem_controller_ = typing.cast(stem_controller.STEMController, stem_controller_)
            self.assertIsNone(stem_controller_.drift_channel_id)
            self.assertIsNone(stem_controller_.drift_region)
            scan_control_model.drift_checkbox_checked = True
            document_controller.periodic()
            self.assertIsNotNone(stem_controller_.drift_channel_id)
            self.assertIsNotNone(stem_controller_.drift_region)
            scan_control_model.drift_checkbox_checked = False
            document_controller.periodic()
            self.assertIsNone(stem_controller_.drift_channel_id)
            self.assertIsNone(stem_controller_.drift_region)

    def test_subscan_has_proper_calibrations(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            scan_control_model = self.__create_scan_control_model(test_context)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_control_model.subscan_checkbox_checked = True
            document_controller.periodic()
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].scale, document_model.data_items[1].dimensional_calibrations[1].scale)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].units, document_model.data_items[1].dimensional_calibrations[1].units)

    def test_record_immediate_has_proper_calibrations(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            frame_parameters = copy.copy(scan_hardware_source.get_current_frame_parameters())
            frame_parameters.subscan_pixel_size = Geometry.IntSize(128, 128)
            frame_parameters.subscan_fractional_size = Geometry.FloatSize(0.5, 0.5)
            frame_parameters.subscan_fractional_center = Geometry.FloatPoint(0.5, 0.5)
            frame_parameters.enabled_channel_indexes = [0]
            xdata = scan_hardware_source.record_immediate(frame_parameters)[0]
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].scale, xdata.dimensional_calibrations[1].scale)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].units, xdata.dimensional_calibrations[1].units)
            frame_parameters.subscan_pixel_size = Geometry.IntSize(256, 256)
            xdata = scan_hardware_source.record_immediate(frame_parameters)[0]
            self.assertAlmostEqual(document_model.data_items[0].dimensional_calibrations[0].scale, xdata.dimensional_calibrations[1].scale * 2)
            self.assertEqual(document_model.data_items[0].dimensional_calibrations[0].units, xdata.dimensional_calibrations[1].units)

    def test_get_buffer_data_basic_functionality(self):
        with self._test_context() as test_context:
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
        with self._test_context() as test_context:
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
            test_context.instrument.subscan_state = stem_controller.SubscanState.DISABLED_HIDDEN
            scan_hardware_source.line_scan_enabled = False
            test_context.instrument.line_scan_state = stem_controller.LineScanState.DISABLED_HIDDEN
            scan_hardware_source.drift_enabled = False
            test_context.document_controller = test_context.create_document_controller(auto_close=False)
            test_context.document_model = test_context.document_controller.document_model
            document_model = test_context.document_model
            self.assertEqual(0, len(document_model.display_items[0].graphics))

    def test_reloading_document_cleans_only_acquisition_display_items(self):
        with self._test_context() as test_context:
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
            self.assertEqual(1, len(document_model.display_items))
            document_model.get_display_item_snapshot_new(document_model.display_items[0])
            self.assertEqual(4, len(document_model.display_items[1].graphics))
            self.assertEqual(2, len(document_model.display_items))
            document_controller.close()
            test_context.instrument.probe_position = None
            scan_hardware_source.subscan_enabled = False
            test_context.instrument.subscan_state = stem_controller.SubscanState.DISABLED_HIDDEN
            scan_hardware_source.line_scan_enabled = False
            test_context.instrument.line_scan_state = stem_controller.LineScanState.DISABLED_HIDDEN
            scan_hardware_source.drift_enabled = False
            test_context.document_controller = test_context.create_document_controller(auto_close=False)
            test_context.document_model = test_context.document_controller.document_model
            document_model = test_context.document_model
            self.assertEqual(2, len(document_model.display_items))
            self.assertEqual(0, len(document_model.display_items[0].graphics))
            self.assertEqual(4, len(document_model.display_items[1].graphics))

    def test_graphics_are_enabled_when_switching_project(self):
        with self._test_context() as test_context:
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

    def test_subscan_graphic_is_created_after_creating_new_project(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(1, len(document_model.display_items))
            scan_hardware_source.subscan_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = False
            document_controller.periodic()
            self.assertEqual(2, len(document_model.display_items))
            self.assertEqual(1, len(document_model.display_items[0].graphics))
            self.assertEqual(0, len(document_model.display_items[1].graphics))
            document_controller.close()
            test_context.reset_profile()
            test_context.document_controller = test_context.create_document_controller(auto_close=False)
            test_context.document_model = test_context.document_controller.document_model
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            self.assertEqual(0, len(document_model.display_items))
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual(1, len(document_model.display_items))
            document_controller.periodic()
            scan_hardware_source.subscan_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = False
            document_controller.periodic()
            self.assertEqual(2, len(document_model.display_items))
            self.assertEqual(1, len(document_model.display_items[0].graphics))
            self.assertEqual(0, len(document_model.display_items[1].graphics))

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
        with self._test_context() as test_context:
            hardware_source = test_context.scan_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            # ensure it is initially dict-like
            frame_parameters.fov_nm = 8.0
            self.assertEqual(frame_parameters.size, frame_parameters.size)
            self.assertEqual(frame_parameters.center_nm, frame_parameters.center_nm)
            self.assertEqual(frame_parameters.fov_nm, frame_parameters.fov_nm)
            self.assertEqual(frame_parameters.fov_size_nm, frame_parameters.fov_size_nm)
            # try setting values
            frame_parameters.size = Geometry.IntSize(4, 4)
            frame_parameters.center_nm = Geometry.FloatPoint(5.0, 5.0)
            frame_parameters.fov_nm = 16.0
            self.assertEqual(Geometry.IntSize(4, 4), frame_parameters.size)
            self.assertEqual(Geometry.FloatPoint(5.0, 5.0), frame_parameters.center_nm)
            self.assertEqual(16.0, frame_parameters.fov_nm)
            # test basic parameters take tuples
            frame_parameters.size = (40, 40)
            frame_parameters.center_nm = (50.0, 50.0)
            self.assertEqual(Geometry.IntSize(40, 40), frame_parameters.size)
            self.assertEqual(Geometry.FloatPoint(50.0, 50.0), frame_parameters.center_nm)
            frame_parameters.size = (41, 41)
            frame_parameters.center_nm = (51.0, 51.0)
            self.assertEqual(Geometry.IntSize(41, 41), frame_parameters.size)
            self.assertEqual(Geometry.FloatPoint(51.0, 51.0), frame_parameters.center_nm)
            # test optional parameters take tuples and None
            self.assertIsNone(frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = (20, 20)
            self.assertEqual(Geometry.IntSize(20, 20), frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = Geometry.IntSize(21, 21)
            self.assertEqual(Geometry.IntSize(21, 21), frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = Geometry.IntSize(22, 22)
            self.assertEqual(Geometry.IntSize(22, 22), frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = None
            self.assertIsNone(frame_parameters.subscan_pixel_size)
            frame_parameters.subscan_pixel_size = Geometry.IntSize(21, 21)
            frame_parameters.subscan_pixel_size = None
            # test dict is writeable to json
            json.dumps(frame_parameters.as_dict())

    def test_scan_immediate(self) -> None:
        with self._test_context() as test_context:
            hardware_source = test_context.scan_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.pixel_size = Geometry.IntSize(256, 256)
            frame_parameters.pixel_time_us = 16.0
            frame_parameters.center_nm = Geometry.FloatPoint()
            frame_parameters.ac_line_sync = False
            frame_parameters.fov_nm = 1.0
            frame_parameters.subscan_pixel_size = None
            frame_parameters.subscan_fractional_size = None
            frame_parameters.subscan_fractional_center = None
            frame_parameters.channel_override = "miniscan"  # ensure it doesn't get sent to a display
            # set the following superscan-specific parameters using the general set_parameter function.
            frame_parameters.set_parameter("ac_frame_sync", True)
            frame_parameters.set_parameter("external_clock_mode", 0)
            frame_parameters.set_parameter("external_scan_mode", 0)
            hardware_source.scan_immediate(frame_parameters)

    def test_scan_action_with_frame_parameters(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.scan_hardware_source
            frame_parameters = hardware_source.get_frame_parameters(0)
            frame_parameters.pixel_size = Geometry.IntSize(30, 40)
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
            self.assertEqual((30, 40), document_model.data_items[0].data_shape)

    def test_scan_action_with_no_frame_parameters(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            hardware_source = test_context.scan_hardware_source
            action_context = document_controller._get_action_context()
            action_context.parameters["hardware_source_id"] = hardware_source.hardware_source_id
            document_controller.perform_action_in_context("acquisition.start_playing", action_context)
            start = time.time()
            while not hardware_source.is_playing:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < 3.0
            hardware_source.stop_playing(sync_timeout=3.0)
            document_controller.periodic()
            self.assertEqual((256, 256), document_model.data_items[0].data_shape)

    def test_scan_frame_parameters_are_updated_when_channels_are_changed(self) -> None:
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_channel_enabled(0, True)
            scan_hardware_source.start_playing(sync_timeout=3.0)
            try:
                self.assertEqual([0], scan_hardware_source.get_enabled_channel_indexes())
                self.assertEqual([0], scan_hardware_source.get_current_frame_parameters().enabled_channel_indexes)
                scan_hardware_source.get_next_xdatas_to_start()
                document_controller.periodic()
                self.assertEqual(1, len(document_controller.document_model.data_items))
                scan_hardware_source.set_channel_enabled(1, True)
                self.assertEqual([0, 1], scan_hardware_source.get_enabled_channel_indexes())
                self.assertEqual([0, 1], scan_hardware_source.get_current_frame_parameters().enabled_channel_indexes)
                scan_hardware_source.get_next_xdatas_to_start()
                document_controller.periodic()
                self.assertEqual(2, len(document_controller.document_model.data_items))
                scan_hardware_source.set_channel_enabled(0, False)
                self.assertEqual([1], scan_hardware_source.get_enabled_channel_indexes())
                self.assertEqual([1], scan_hardware_source.get_current_frame_parameters().enabled_channel_indexes)
                scan_hardware_source.get_next_xdatas_to_start()
                document_controller.periodic()
            finally:
                scan_hardware_source.stop_playing(sync_timeout=3.0)
                scan_hardware_source.set_channel_enabled(0, True)
                scan_hardware_source.set_channel_enabled(1, False)

    def test_shift_click_with_rotation(self):
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            stem_controller_ = test_context.instrument
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.pixel_size = Geometry.IntSize(100, 100)
            scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertAlmostEqualPoint(Geometry.FloatPoint(), stem_controller_.GetVal2D("stage_position_m"), e=1e-9)
            logger = logging.getLogger("shift-click-test")
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
            scan_hardware_source.shift_click(Geometry.FloatPoint(53, 54), (100, 100), logger)
            self.assertAlmostEqualPoint(Geometry.FloatPoint(y=-3e-9, x=-4e-9), stem_controller_.GetVal2D("stage_position_m"), e=1e-9)
            scan_hardware_source.shift_click(Geometry.FloatPoint(47, 46), (100, 100), logger)
            self.assertAlmostEqualPoint(Geometry.FloatPoint(), stem_controller_.GetVal2D("stage_position_m"), e=1e-9)
            # now test rotated
            scan_frame_parameters.rotation_rad = math.radians(45)
            scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.shift_click(Geometry.FloatPoint(60, 50), (100, 100), logger)
            self.assertAlmostEqualPoint(Geometry.FloatPoint(y=-math.sqrt(2)/2*1e-8, x=math.sqrt(2)/2*1e-8), stem_controller_.GetVal2D("stage_position_m"), e=1e-9)
            scan_hardware_source.shift_click(Geometry.FloatPoint(40, 50), (100, 100), logger)
            self.assertAlmostEqualPoint(Geometry.FloatPoint(), stem_controller_.GetVal2D("stage_position_m"), e=1e-9)

    def test_scan_size_consistency(self):
        # NOTE: the hardware (and simulator) always map the FoV to the largest dimension
        with self._test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.fov_nm = 100.0
            # test the square case
            scan_frame_parameters.pixel_size = Geometry.IntSize(h=200, w=200)
            scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual((200, 200), document_controller.document_model.data_items[0].data_shape)
            self.assertEqual((100, 100), tuple(scan_hardware_source.get_current_frame_parameters().fov_size_nm))
            self.assertEqual((100, 100), tuple(scan_hardware_source.scan_context.fov_size_nm))
            self.assertEqual((200, 200), tuple(scan_hardware_source.scan_context.size))
            self.assertEqual(100/200, document_controller.document_model.data_items[0].dimensional_calibrations[0].scale)
            self.assertEqual(100/200, document_controller.document_model.data_items[0].dimensional_calibrations[1].scale)
            self.assertEqual(Calibration.Calibration(None, 100/200, "nm"), scan_hardware_source.scan_context.calibration)
            # test the w > h case
            scan_frame_parameters.pixel_size = Geometry.IntSize(h=150, w=300)
            scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual((150, 300), document_controller.document_model.data_items[0].data_shape)
            self.assertEqual((50, 100), tuple(scan_hardware_source.get_current_frame_parameters().fov_size_nm))
            self.assertEqual((100, 200), tuple(scan_hardware_source.scan_context.fov_size_nm))
            self.assertEqual((150, 300), tuple(scan_hardware_source.scan_context.size))
            self.assertEqual(100/300, document_controller.document_model.data_items[0].dimensional_calibrations[0].scale)
            self.assertEqual(100/300, document_controller.document_model.data_items[0].dimensional_calibrations[1].scale)
            self.assertEqual(Calibration.Calibration(None, 100/300, "nm"), scan_hardware_source.scan_context.calibration)
            # test the h > w case
            scan_frame_parameters.pixel_size = Geometry.IntSize(h=300, w=150)
            scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
            self._acquire_one(document_controller, scan_hardware_source)
            self.assertEqual((300, 150), document_controller.document_model.data_items[0].data_shape)
            self.assertEqual((100, 50), tuple(scan_hardware_source.get_current_frame_parameters().fov_size_nm))
            self.assertEqual((100, 50), tuple(scan_hardware_source.scan_context.fov_size_nm))
            self.assertEqual((300, 150), tuple(scan_hardware_source.scan_context.size))
            self.assertEqual(100/300, document_controller.document_model.data_items[0].dimensional_calibrations[0].scale)
            self.assertEqual(100/300, document_controller.document_model.data_items[0].dimensional_calibrations[1].scale)
            self.assertEqual(Calibration.Calibration(None, 100/300, "nm"), scan_hardware_source.scan_context.calibration)

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
