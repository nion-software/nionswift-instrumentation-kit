from __future__ import annotations

import json
import math
import pathlib
import threading
import typing
import unittest
import uuid

from nion.instrumentation import Acquisition
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base as ScanBase
from nion.instrumentation import ScanProfiles
from nion.instrumentation.test import AcquisitionTestContext
from nion.instrumentation.test import AcquisitionTestContextConfiguration
from nion.swift.test import TestContext
from nion.utils import Geometry


# the tests which check every parameter of a profile compare all of its values, or all of its included flags, at once,
# so that a parameter missed by one of the methods acting on every parameter fails them.

def _make_scan_profile_with_every_parameter_changed() -> ScanProfiles.ProfileModel:
    """Return a profile whose values and included flags all differ from those of a new profile."""
    profile = ScanProfiles.ProfileModel("Survey")
    profile.field_width = 37.5
    profile.size_linked = False
    profile.width = 512
    profile.height = 256
    profile.pixel_time_us = 2.5
    profile.rotation = 15.0
    profile.ac_line_sync = True
    profile.subscan_field_width_nm = 10.0
    profile.subscan_field_height_nm = 5.0
    profile.subscan_position_x_percent = 30.0
    profile.subscan_position_y_percent = 40.0
    profile.subscan_pixel_width = 64
    profile.subscan_rotation = 5.0
    profile.field_width_included = False
    profile.size_included = False
    profile.pixel_time_us_included = False
    profile.rotation_included = False
    profile.ac_line_sync_included = False
    profile.subscan_field_size_included = False
    profile.subscan_fractional_position_included = False
    profile.subscan_pixel_width_included = False
    profile.subscan_rotation_included = False
    return profile


def _get_scan_profile_values(profile: ScanProfiles.ProfileModel) -> typing.Sequence[float | int | bool | None]:
    """Return every value of the profile."""
    return (profile.field_width, profile.width, profile.height, profile.size_linked, profile.pixel_time_us, profile.rotation, profile.ac_line_sync,
            profile.subscan_field_width_nm, profile.subscan_field_height_nm, profile.subscan_position_x_percent, profile.subscan_position_y_percent,
            profile.subscan_pixel_width, profile.subscan_rotation)


def _get_scan_profile_included_flags(profile: ScanProfiles.ProfileModel) -> typing.Sequence[bool]:
    """Return every included flag of the profile."""
    return (profile.field_width_included, profile.size_included, profile.pixel_time_us_included, profile.rotation_included, profile.ac_line_sync_included,
            profile.subscan_field_size_included, profile.subscan_fractional_position_included, profile.subscan_pixel_width_included, profile.subscan_rotation_included)


def _make_scan_device_with_saved_scan_profiles(test_context: AcquisitionTestContext.AcquisitionTestContext, properties: object) -> ScanBase.ConcreteScanHardwareSource:
    """Return a new scan device which loads the properties from its saved scan profiles file. The caller closes it."""
    scan_module = AcquisitionTestContextConfiguration.ScanModule(test_context.instrument, "test_restored_profiles_scan_device", AcquisitionTestContextConfiguration.ScanDataGenerator())
    configuration_location = pathlib.Path(test_context.configuration.configuration_location)
    (configuration_location / "test_restored_profiles_scan_device_scan_profiles.json").write_text(json.dumps(properties))
    return ScanBase.ConcreteScanHardwareSource(test_context.instrument, scan_module.device, scan_module.settings, configuration_location)


class TestScanProfilesClass(unittest.TestCase):

    def setUp(self) -> None:
        AcquisitionTestContext.begin_leaks()
        self._test_setup = TestContext.TestSetup()

    def tearDown(self) -> None:
        self._test_setup = typing.cast(typing.Any, None)
        AcquisitionTestContext.end_leaks(self)

    def _test_context(self) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.test_context()

    def test_activating_scan_profile_applies_it_to_current_frame_parameters(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            fov_nm = scan_hardware_source.get_frame_parameters(1).fov_nm
            changed_property_names: list[str] = list()
            property_changed_event_listener = profile_configuration_model.property_changed_event.listen(changed_property_names.append)
            profile_configuration_model.active_profile_index = 1
            property_changed_event_listener = None
            self.assertIn("active_profile_index", changed_property_names)
            self.assertEqual(1, profile_configuration_model.active_profile_index)
            self.assertAlmostEqual(fov_nm, scan_hardware_source.get_current_frame_parameters().fov_nm)

    def test_previous_scan_profile_toggles_between_last_two_active_profiles(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            self.assertFalse(profile_configuration_model.has_previous_profile)
            profile_configuration_model.active_profile_index = 2
            self.assertTrue(profile_configuration_model.has_previous_profile)
            profile_configuration_model.activate_previous_profile()
            self.assertEqual(0, profile_configuration_model.active_profile_index)
            profile_configuration_model.activate_previous_profile()
            self.assertEqual(2, profile_configuration_model.active_profile_index)

    def test_removing_scan_profile_keeps_active_profile(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile_index = 2
            active_profile = profile_configuration_model.profiles[2]
            with self.assertRaises(ValueError):
                profile_configuration_model.remove_profile(2)
            profile_configuration_model.remove_profile(0)
            self.assertIs(active_profile, profile_configuration_model.profiles[profile_configuration_model.active_profile_index])
            self.assertFalse(profile_configuration_model.has_previous_profile)

    def test_editing_current_frame_parameters_is_kept_by_active_scan_profile_across_switching(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            other_profile_field_width = profile_configuration_model.profiles[1].field_width
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.fov_nm = 37.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertEqual(37, profile_configuration_model.profiles[0].field_width)
            self.assertEqual(other_profile_field_width, profile_configuration_model.profiles[1].field_width)
            profile_configuration_model.active_profile_index = 1
            self.assertNotAlmostEqual(37.0, scan_hardware_source.get_current_frame_parameters().fov_nm)
            profile_configuration_model.activate_previous_profile()
            self.assertAlmostEqual(37.0, scan_hardware_source.get_current_frame_parameters().fov_nm)

    def test_changing_current_frame_parameters_saves_scan_profiles_once(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            saved_properties_list: list[typing.Mapping[str, typing.Any]] = list()
            profiles_changed_event_listener = profile_configuration_model.profiles_changed_event.listen(saved_properties_list.append)
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.fov_nm += 17.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertEqual(1, len(saved_properties_list))
            profiles_changed_event_listener = None

    def test_sequence_acquisition_leaves_active_scan_profile_unchanged(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.active_profile
            pixel_time_us = active_profile.pixel_time_us
            saved_properties_list: list[typing.Mapping[str, typing.Any]] = list()
            profiles_changed_event_listener = profile_configuration_model.profiles_changed_event.listen(saved_properties_list.append)
            # acquire a sequence of scans with a size and pixel time which differ from those of the active profile.
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=24, w=24)
            frame_parameters.pixel_time_us = pixel_time_us + 1.0
            data_stream = ScanBase.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(2)
            maker = Acquisition.MakerDataStream(data_stream)
            Acquisition.acquire(maker)
            self.assertEqual(pixel_time_us, active_profile.pixel_time_us)
            # the acquisition saves nothing, since a crash during the acquisition would leave its frame parameters saved.
            self.assertEqual(list(), saved_properties_list)
            profiles_changed_event_listener = None

    def test_activating_scan_profile_during_sequence_acquisition_leaves_profiles_unchanged(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.active_profile
            other_profile = profile_configuration_model.profiles[1]
            other_profile_values = other_profile.write_values_to_dict()
            self.assertNotEqual(active_profile.write_values_to_dict(), other_profile_values)

            # the user chooses the other profile once the sequence acquisition has started.
            def handle_property_changed(property_name: str) -> None:
                if property_name == "is_in_sequence_mode" and scan_hardware_source.is_in_sequence_mode:
                    profile_configuration_model.active_profile_index = 1

            property_changed_event_listener = scan_hardware_source.property_changed_event.listen(handle_property_changed)
            # the acquisition saves the device state before it starts and restores it when it ends, as the acquisition panel does.
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=24, w=24)
            data_stream = ScanBase.ScanFrameDataStream(scan_hardware_source, frame_parameters).wrap_in_sequence(2)
            maker = Acquisition.MakerDataStream(data_stream)
            device_state = maker.prepare_device_state()
            Acquisition.acquire(maker)
            device_state.restore()
            property_changed_event_listener = None
            self.assertIs(active_profile, profile_configuration_model.active_profile)
            self.assertEqual(other_profile_values, other_profile.write_values_to_dict())

    def test_active_scan_profile_keeps_fractional_and_small_field_width(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.active_profile
            for fov_nm in (37.25, 0.25):
                frame_parameters = scan_hardware_source.get_current_frame_parameters()
                frame_parameters.fov_nm = fov_nm
                scan_hardware_source.set_current_frame_parameters(frame_parameters)
                self.assertAlmostEqual(fov_nm, active_profile.field_width)
            profile_configuration_model.active_profile_index = 1
            profile_configuration_model.activate_previous_profile()
            self.assertAlmostEqual(0.25, scan_hardware_source.get_current_frame_parameters().fov_nm)

    def test_active_scan_profile_keeps_automatic_subscan_pixel_width(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_hardware_source.subscan_enabled = True
            profile_configuration_model.profiles[1].subscan_pixel_width = 64
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.subscan_pixel_width_override = None
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertIsNone(profile_configuration_model.active_profile.subscan_pixel_width)
            profile_configuration_model.active_profile_index = 1
            self.assertEqual(64, scan_hardware_source.get_current_frame_parameters().subscan_pixel_width_override)
            profile_configuration_model.activate_previous_profile()
            self.assertIsNone(scan_hardware_source.get_current_frame_parameters().subscan_pixel_width_override)
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, profile_configuration_model.write_to_dict())
            try:
                self.assertIsNone(restored_scan_hardware_source.scan_profiles.active_profile.subscan_pixel_width)
            finally:
                restored_scan_hardware_source.close()

    def test_active_scan_profile_keeps_subscan_values_while_subscan_is_off(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            active_profile = scan_hardware_source.scan_profiles.active_profile
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_tlhw(0.1, 0.2, 0.3, 0.4)
            scan_hardware_source.subscan_rotation = math.radians(20)
            subscan_values = (active_profile.subscan_field_width_nm, active_profile.subscan_field_height_nm, active_profile.subscan_position_x_percent, active_profile.subscan_position_y_percent, active_profile.subscan_rotation)
            self.assertAlmostEqual(20.0, active_profile.subscan_rotation)
            scan_hardware_source.subscan_enabled = False
            self.assertEqual(subscan_values, (active_profile.subscan_field_width_nm, active_profile.subscan_field_height_nm, active_profile.subscan_position_x_percent, active_profile.subscan_position_y_percent, active_profile.subscan_rotation))

    def test_scan_profile_keeps_subscan_size_when_reading_frame_parameters_with_invalid_field_width(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            profile = ScanProfiles.ProfileModel("Profile")
            profile.read_from_frame_parameters(frame_parameters)
            subscan_field_size = (profile.subscan_field_width_nm, profile.subscan_field_height_nm)
            frame_parameters.fov_nm = 0.0
            profile.read_from_frame_parameters(frame_parameters)
            self.assertEqual(subscan_field_size, (profile.subscan_field_width_nm, profile.subscan_field_height_nm))

    def test_excluded_subscan_value_saved_while_subscan_is_off_is_restored(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile.subscan_fractional_position_included = False
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.3, x=0.6), Geometry.FloatSize(0.2, 0.2))
            scan_hardware_source.subscan_enabled = False
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, profile_configuration_model.write_to_dict())
            try:
                self.assertAlmostEqual(0.3, restored_scan_hardware_source.subscan_region.center.y)
                self.assertAlmostEqual(0.6, restored_scan_hardware_source.subscan_region.center.x)
            finally:
                restored_scan_hardware_source.close()

    def test_editing_current_frame_parameters_leaves_excluded_scan_profile_value_unchanged(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile = profile_configuration_model.profiles[0]
            profile.field_width_included = False
            field_width = profile.field_width
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.fov_nm = field_width + 11.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertEqual(field_width, profile.field_width)

    def test_editing_active_scan_profile_changes_current_frame_parameters(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.profiles[profile_configuration_model.active_profile_index]
            active_profile.field_width = scan_hardware_source.get_current_frame_parameters().fov_nm + 13.0
            self.assertAlmostEqual(active_profile.field_width, scan_hardware_source.get_current_frame_parameters().fov_nm)

    def test_including_parameter_in_active_scan_profile_keeps_current_value(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.profiles[profile_configuration_model.active_profile_index]
            active_profile.field_width_included = False
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.fov_nm = active_profile.field_width + 11.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            active_profile.field_width_included = True
            self.assertAlmostEqual(frame_parameters.fov_nm, scan_hardware_source.get_current_frame_parameters().fov_nm)
            self.assertAlmostEqual(frame_parameters.fov_nm, active_profile.field_width)

    def test_editing_previously_active_scan_profile_leaves_current_frame_parameters_unchanged(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            first_index = profile_configuration_model.active_profile_index
            profile_configuration_model.active_profile_index = (first_index + 1) % len(profile_configuration_model.profiles)
            fov_nm = scan_hardware_source.get_current_frame_parameters().fov_nm
            profile_configuration_model.profiles[first_index].field_width = fov_nm + 13.0
            self.assertAlmostEqual(fov_nm, scan_hardware_source.get_current_frame_parameters().fov_nm)

    def test_new_scan_profile_includes_current_values_without_becoming_active(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.profiles[profile_configuration_model.active_profile_index]
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.fov_nm += 17.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            profile = profile_configuration_model.insert_new_profile(0)
            self.assertIs(profile, profile_configuration_model.profiles[0])
            self.assertTrue(all((profile.field_width_included, profile.size_included, profile.pixel_time_us_included, profile.rotation_included, profile.ac_line_sync_included,
                                 profile.subscan_field_size_included, profile.subscan_fractional_position_included, profile.subscan_pixel_width_included, profile.subscan_rotation_included)))
            self.assertAlmostEqual(frame_parameters.fov_nm, profile.field_width)
            self.assertIs(active_profile, profile_configuration_model.profiles[profile_configuration_model.active_profile_index])

    def test_new_scan_profile_includes_current_subscan_values_while_subscan_is_off(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_enabled = False
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.6, x=0.3), Geometry.FloatSize(h=0.2, w=0.2))
            profile = profile_configuration_model.insert_new_profile(0)
            self.assertAlmostEqual(30.0, profile.subscan_position_x_percent)
            self.assertAlmostEqual(60.0, profile.subscan_position_y_percent)

    def test_added_scan_profiles_have_unique_names(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.insert_new_profile(len(profile_configuration_model.profiles))
            profile_configuration_model.insert_new_profile(len(profile_configuration_model.profiles))
            profile_configuration_model.insert_profile(0, ScanProfiles.ProfileModel(profile_configuration_model.profile_names[1]))
            self.assertEqual(len(profile_configuration_model.profile_names), len(set(profile_configuration_model.profile_names)))

    def test_closed_scan_profiles_no_longer_follow_the_scan_device(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            # a model of its own on the scan device, so that closing it tests that close releases the model's listeners on the scan device.
            profile_configuration_model = ScanProfiles.ProfileConfigurationModel(scan_hardware_source)
            field_width = profile_configuration_model.active_profile.field_width
            profile_configuration_model.close()
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.fov_nm += 17.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertEqual(field_width, profile_configuration_model.active_profile.field_width)

    def test_duplicated_scan_profile_is_independent_copy_with_distinct_name(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            source_profile = profile_configuration_model.profiles[1]
            source_profile.rotation_included = False
            copy_profile = profile_configuration_model.duplicate_profile(1)
            self.assertIs(copy_profile, profile_configuration_model.profiles[2])
            self.assertNotEqual(source_profile.name, copy_profile.name)
            self.assertNotEqual(source_profile.uuid, copy_profile.uuid)
            self.assertEqual(source_profile.write_values_to_dict(), copy_profile.write_values_to_dict())
            profile_configuration_model.duplicate_profile(1)
            self.assertEqual(len(profile_configuration_model.profile_names), len(set(profile_configuration_model.profile_names)))
            copy_profile.field_width = source_profile.field_width + 5.0
            self.assertNotEqual(source_profile.field_width, copy_profile.field_width)

    def test_renaming_scan_profile_updates_profile_names(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            changed_property_names: list[str] = list()
            property_changed_event_listener = profile_configuration_model.property_changed_event.listen(changed_property_names.append)
            profile_configuration_model.rename_profile(profile_configuration_model.profiles[1], "Survey")
            property_changed_event_listener = None
            self.assertIn("profile_names", changed_property_names)
            self.assertEqual("Survey", profile_configuration_model.profile_names[1])

    def test_renaming_scan_profile_to_blank_name_or_name_of_another_profile_is_rejected(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            original_names = list(profile_configuration_model.profile_names)
            for name in (str(), "   ", original_names[0]):
                profile_configuration_model.rename_profile(profile_configuration_model.profiles[1], name)
                self.assertEqual(original_names, list(profile_configuration_model.profile_names))

    def test_moving_scan_profile_keeps_same_active_and_previous_profiles(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile_index = 0
            profile_configuration_model.active_profile_index = 2
            first_profile, second_profile, third_profile = profile_configuration_model.profiles[0:3]
            profile_configuration_model.move_profile(2, 0)
            self.assertEqual([third_profile, first_profile, second_profile], list(profile_configuration_model.profiles[0:3]))
            self.assertIs(third_profile, profile_configuration_model.profiles[profile_configuration_model.active_profile_index])
            profile_configuration_model.activate_previous_profile()
            self.assertIs(first_profile, profile_configuration_model.profiles[profile_configuration_model.active_profile_index])

    def test_frame_parameters_made_for_scan_profile_match_activating_it(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.5, x=0.5), Geometry.FloatSize(0.5, 0.5))
            profile = profile_configuration_model.profiles[1]
            profile.subscan_field_width_nm = profile.field_width * 0.3
            profile.subscan_field_height_nm = profile.field_width * 0.3
            profile.subscan_pixel_width = None
            frame_parameters = profile_configuration_model.make_frame_parameters(profile)
            # activate the profile before recording, since the scan device ignores a profile change while it is recording.
            profile_configuration_model.active_profile_index = 1
            subscan_pixel_size = scan_hardware_source.get_current_frame_parameters().subscan_pixel_size
            assert subscan_pixel_size
            recorded_shapes: list[tuple[int, ...]] = list()
            recorded_event = threading.Event()

            def handle_record(data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:  # called from thread
                recorded_shapes.extend(data_promise.xdata.data_shape for data_promise in data_promises if data_promise.xdata)
                recorded_event.set()

            scan_hardware_source.record_async(handle_record, frame_parameters=frame_parameters)
            self.assertTrue(recorded_event.wait(10.0))
            self.assertEqual([tuple(subscan_pixel_size)], recorded_shapes)

    def test_scan_profiles_are_restored_from_saved_file(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.profiles[1].name = "Survey"
            profile_configuration_model.profiles[2].pixel_time_us_included = False
            profile_configuration_model.insert_new_profile(3)
            profile_configuration_model.active_profile_index = 2
            profile_configuration_model.active_profile_index = 3
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.fov_nm = 37.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            scan_profiles_file = pathlib.Path(test_context.configuration.configuration_location) / (scan_hardware_source.hardware_source_id + "_scan_profiles.json")
            with scan_profiles_file.open() as f:
                properties = json.load(f)
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                restored_model = restored_scan_hardware_source.scan_profiles
                self.assertEqual([profile.uuid for profile in profile_configuration_model.profiles], [profile.uuid for profile in restored_model.profiles])
                self.assertEqual(profile_configuration_model.profile_names, restored_model.profile_names)
                self.assertEqual(3, restored_model.active_profile_index)
                self.assertEqual(37, restored_model.active_profile.field_width)
                self.assertFalse(restored_model.profiles[2].pixel_time_us_included)
                restored_model.activate_previous_profile()
                self.assertEqual(2, restored_model.active_profile_index)
            finally:
                restored_scan_hardware_source.close()

    def test_activating_scan_profile_is_saved(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            # renaming saves the file first, so that the saved active profile is checked rather than whether the file exists.
            profile_configuration_model.profiles[1].name = "Survey"
            profile_configuration_model.active_profile_index = 2
            scan_profiles_file = pathlib.Path(test_context.configuration.configuration_location) / (scan_hardware_source.hardware_source_id + "_scan_profiles.json")
            with scan_profiles_file.open() as f:
                properties = json.load(f)
            self.assertEqual(profile_configuration_model.profiles[2].uuid, properties["active_profile_uuid"])

    def test_excluded_value_is_restored_from_saved_file(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile_configuration_model.active_profile.rotation_included = False
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.rotation_rad = math.radians(30)
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            scan_profiles_file = pathlib.Path(test_context.configuration.configuration_location) / (scan_hardware_source.hardware_source_id + "_scan_profiles.json")
            with scan_profiles_file.open() as f:
                properties = json.load(f)
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                self.assertAlmostEqual(30.0, math.degrees(restored_scan_hardware_source.get_current_frame_parameters().rotation_rad))
            finally:
                restored_scan_hardware_source.close()

    def test_restored_active_scan_profile_is_applied_to_current_frame_parameters(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            properties = scan_hardware_source.scan_profiles.write_to_dict()
            self.assertNotEqual(91, round(scan_hardware_source.get_current_frame_parameters().fov_nm))
            properties["profiles"][1]["field_width"] = 91
            properties["active_profile_uuid"] = properties["profiles"][1]["uuid"]
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                self.assertEqual(91, round(restored_scan_hardware_source.get_current_frame_parameters().fov_nm))
            finally:
                restored_scan_hardware_source.close()

    def test_restoring_unknown_active_scan_profile_makes_first_profile_active(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            scan_hardware_source.scan_profiles.active_profile_index = 2
            properties = scan_hardware_source.scan_profiles.write_to_dict()
            properties["active_profile_uuid"] = str(uuid.uuid4())
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                self.assertEqual(0, restored_scan_hardware_source.scan_profiles.active_profile_index)
            finally:
                restored_scan_hardware_source.close()

    def test_restoring_scan_profile_with_invalid_value_keeps_the_other_values(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            properties = scan_hardware_source.scan_profiles.write_to_dict()
            properties["profiles"][1]["field_width"] = 0
            properties["profiles"][1]["width"] = "wide"
            properties["profiles"][1]["pixel_time_us"] = 7.0
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                restored_profile = restored_scan_hardware_source.scan_profiles.profiles[1]
                self.assertGreater(restored_profile.field_width, 0)
                self.assertGreater(restored_profile.width, 0)
                self.assertAlmostEqual(7.0, restored_profile.pixel_time_us)
            finally:
                restored_scan_hardware_source.close()

    def test_restoring_scan_profile_with_out_of_range_value_keeps_the_other_values(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            properties = scan_hardware_source.scan_profiles.write_to_dict()
            properties["profiles"][1]["width"] = float("inf")
            properties["profiles"][1]["field_width"] = 10 ** 400
            properties["profiles"][1]["pixel_time_us"] = 7.0
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                restored_profile = restored_scan_hardware_source.scan_profiles.profiles[1]
                self.assertGreater(restored_profile.width, 0)
                self.assertTrue(math.isfinite(restored_profile.field_width))
                self.assertAlmostEqual(7.0, restored_profile.pixel_time_us)
            finally:
                restored_scan_hardware_source.close()

    def test_restoring_scan_profiles_with_duplicate_names_makes_names_unique(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            properties = scan_hardware_source.scan_profiles.write_to_dict()
            properties["profiles"][1]["name"] = properties["profiles"][0]["name"]
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                restored_model = restored_scan_hardware_source.scan_profiles
                self.assertEqual(properties["profiles"][0]["name"], restored_model.profile_names[0])
                self.assertEqual(len(restored_model.profile_names), len(set(restored_model.profile_names)))
            finally:
                restored_scan_hardware_source.close()

    def test_generated_scan_profiles_use_the_scan_device_subscan_region(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            subscan_region = scan_hardware_source.subscan_region
            for profile_index, profile in enumerate(scan_hardware_source.scan_profiles.profiles):
                fov_nm = scan_hardware_source.get_frame_parameters(profile_index).fov_nm
                self.assertAlmostEqual(subscan_region.width * fov_nm, profile.subscan_field_width_nm)
                self.assertAlmostEqual(subscan_region.height * fov_nm, profile.subscan_field_height_nm)

    def test_turning_subscan_on_uses_the_subscan_of_the_active_scan_profile_edited_while_it_is_off(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            active_profile = scan_hardware_source.scan_profiles.active_profile
            active_profile.subscan_field_width_nm = 10.0
            active_profile.subscan_field_height_nm = 10.0
            active_profile.subscan_position_x_percent = 30.0
            active_profile.subscan_position_y_percent = 40.0
            active_profile.subscan_rotation = 10.0
            scan_hardware_source.subscan_enabled = True
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            subscan_fractional_center = frame_parameters.subscan_fractional_center
            assert subscan_fractional_center
            self.assertAlmostEqual(0.3, subscan_fractional_center.x)
            self.assertAlmostEqual(0.4, subscan_fractional_center.y)
            self.assertAlmostEqual(10.0, math.degrees(frame_parameters.subscan_rotation))
            self.assertAlmostEqual(10.0, active_profile.subscan_field_width_nm)
            self.assertAlmostEqual(30.0, active_profile.subscan_position_x_percent)

    def test_turning_subscan_on_uses_the_subscan_size_of_a_scan_profile_which_includes_only_the_size(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            profile = profile_configuration_model.profiles[1]
            profile.field_width_included = False
            profile.size_included = False
            profile.pixel_time_us_included = False
            profile.rotation_included = False
            profile.ac_line_sync_included = False
            profile.subscan_fractional_position_included = False
            profile.subscan_pixel_width_included = False
            profile.subscan_rotation_included = False
            fov_nm = scan_hardware_source.get_current_frame_parameters().fov_nm
            profile.subscan_field_width_nm = fov_nm * 0.2
            profile.subscan_field_height_nm = fov_nm * 0.1
            profile_configuration_model.active_profile_index = 1
            scan_hardware_source.subscan_enabled = True
            subscan_fractional_size = scan_hardware_source.get_current_frame_parameters().subscan_fractional_size
            assert subscan_fractional_size
            self.assertAlmostEqual(0.2, subscan_fractional_size.width)
            self.assertAlmostEqual(0.1, subscan_fractional_size.height)

    def test_moving_subscan_while_it_is_off_changes_the_active_scan_profile(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            active_profile = scan_hardware_source.scan_profiles.active_profile
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_enabled = False
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.6, x=0.3), Geometry.FloatSize(h=0.2, w=0.2))
            self.assertAlmostEqual(30.0, active_profile.subscan_position_x_percent)
            # an edit to another parameter leaves the moved subscan in place.
            active_profile.pixel_time_us = 3.0
            self.assertAlmostEqual(0.3, scan_hardware_source.subscan_region.center.x)

    def test_scan_device_loads_with_invalid_scan_profiles_file(self) -> None:
        with self._test_context() as test_context:
            scan_module = AcquisitionTestContextConfiguration.ScanModule(test_context.instrument, "test_invalid_profiles_scan_device", AcquisitionTestContextConfiguration.ScanDataGenerator())
            configuration_location = pathlib.Path(test_context.configuration.configuration_location)
            scan_profiles_file = configuration_location / "test_invalid_profiles_scan_device_scan_profiles.json"
            scan_profiles_file.write_text("{\"profiles\": [")
            scan_hardware_source = ScanBase.ConcreteScanHardwareSource(test_context.instrument, scan_module.device, scan_module.settings, configuration_location)
            try:
                self.assertEqual(list(ScanProfiles.hardware_profile_slot_names), list(scan_hardware_source.scan_profiles.profile_names))
            finally:
                scan_hardware_source.close()

    def test_unreadable_scan_profiles_file_is_kept_when_scan_device_loads(self) -> None:
        with self._test_context() as test_context:
            scan_module = AcquisitionTestContextConfiguration.ScanModule(test_context.instrument, "test_unreadable_profiles_scan_device", AcquisitionTestContextConfiguration.ScanDataGenerator())
            configuration_location = pathlib.Path(test_context.configuration.configuration_location)
            scan_profiles_file = configuration_location / "test_unreadable_profiles_scan_device_scan_profiles.json"
            bad_scan_profiles_file = configuration_location / "test_unreadable_profiles_scan_device_scan_profiles.json.bad"
            # a file kept from an earlier unreadable file is replaced by the latest one.
            bad_scan_profiles_file.write_text("{\"profiles\": [")
            # a file with a comma missing, as editing it by hand might leave it.
            unreadable_text = "{\"version\": 1 \"profiles\": [{\"name\": \"Survey\", \"field_width\": 12.0}]}"
            scan_profiles_file.write_text(unreadable_text)
            scan_hardware_source = ScanBase.ConcreteScanHardwareSource(test_context.instrument, scan_module.device, scan_module.settings, configuration_location)
            try:
                self.assertEqual(unreadable_text, bad_scan_profiles_file.read_text())
                # loading writes no profiles file, so the frame parameters are changed to make the scan device save the profiles.
                frame_parameters = scan_hardware_source.get_current_frame_parameters()
                frame_parameters.pixel_time_us += 1.0
                scan_hardware_source.set_current_frame_parameters(frame_parameters)
                self.assertEqual(list(ScanProfiles.hardware_profile_slot_names), [profile_properties["name"] for profile_properties in json.loads(scan_profiles_file.read_text())["profiles"]])
            finally:
                scan_hardware_source.close()

    def test_scan_profiles_file_of_unusable_structure_is_kept_when_scan_device_loads(self) -> None:
        with self._test_context() as test_context:
            configuration_location = pathlib.Path(test_context.configuration.configuration_location)
            # files which are valid JSON, as editing them by hand might leave them: one with the profiles key
            # misspelled, so that no profile can be used, and one with an entry which is not a profile.
            for device_id, text in (("test_misspelled_profiles_scan_device", json.dumps({"version": 1, "profile": [{"name": "Survey", "field_width": 12.0}]})),
                                    ("test_string_profile_scan_device", json.dumps({"version": 1, "profiles": [{"name": "Survey", "field_width": 12.0}, "Slow"]}))):
                with self.subTest(device_id=device_id):
                    scan_module = AcquisitionTestContextConfiguration.ScanModule(test_context.instrument, device_id, AcquisitionTestContextConfiguration.ScanDataGenerator())
                    scan_profiles_file = configuration_location / f"{device_id}_scan_profiles.json"
                    bad_scan_profiles_file = configuration_location / f"{device_id}_scan_profiles.json.bad"
                    scan_profiles_file.write_text(text)
                    scan_hardware_source = ScanBase.ConcreteScanHardwareSource(test_context.instrument, scan_module.device, scan_module.settings, configuration_location)
                    try:
                        self.assertEqual(text, bad_scan_profiles_file.read_text())
                    finally:
                        scan_hardware_source.close()

    def test_restoring_scan_profiles_of_wrong_shape_uses_the_hardware_profiles(self) -> None:
        with self._test_context() as test_context:
            for properties in (["profiles"], {"profiles": "Survey"}, {"profiles": [1, "Survey"]}):
                restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
                try:
                    self.assertEqual(list(ScanProfiles.hardware_profile_slot_names), list(restored_scan_hardware_source.scan_profiles.profile_names))
                finally:
                    restored_scan_hardware_source.close()

    def test_restoring_scan_profiles_with_invalid_current_values_keeps_the_profiles(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            scan_hardware_source.scan_profiles.profiles[1].name = "Survey"
            properties = scan_hardware_source.scan_profiles.write_to_dict()
            properties["current_values"] = 3
            restored_scan_hardware_source = _make_scan_device_with_saved_scan_profiles(test_context, properties)
            try:
                self.assertEqual("Survey", restored_scan_hardware_source.scan_profiles.profile_names[1])
            finally:
                restored_scan_hardware_source.close()

    def test_editing_linked_scan_profile_size_changes_width_and_height_together(self) -> None:
        profile = ScanProfiles.ProfileModel(str())
        profile.width = 64
        self.assertEqual((64, 64, True), (profile.width, profile.height, profile.size_linked))
        profile.width = 128
        self.assertEqual((128, 128, True), (profile.width, profile.height, profile.size_linked))
        profile.height = 32
        self.assertEqual((32, 32, True), (profile.width, profile.height, profile.size_linked))
        profile.size_linked = False
        profile.width = 256
        self.assertEqual((256, 32, False), (profile.width, profile.height, profile.size_linked))
        profile.height = 16
        self.assertEqual((256, 16, False), (profile.width, profile.height, profile.size_linked))

    def test_editing_linked_size_of_active_scan_profile_saves_scan_profiles_once(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.active_profile
            active_profile.size_linked = True
            saved_properties_list: list[typing.Mapping[str, typing.Any]] = list()
            profiles_changed_event_listener = profile_configuration_model.profiles_changed_event.listen(saved_properties_list.append)
            active_profile.width = active_profile.width + 16
            self.assertEqual(1, len(saved_properties_list))
            self.assertEqual(Geometry.IntSize(active_profile.height, active_profile.width), scan_hardware_source.get_current_frame_parameters().pixel_size)
            profiles_changed_event_listener = None

    def test_unlinked_square_size_of_active_scan_profile_stays_unlinked_when_current_frame_parameters_change(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            profile_configuration_model = scan_hardware_source.scan_profiles
            active_profile = profile_configuration_model.active_profile
            active_profile.size_linked = False
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_time_us += 1.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertFalse(active_profile.size_linked)
            # switching away and back applies the profile and reads the current frame parameters back into it.
            profile_configuration_model.active_profile_index = 1
            profile_configuration_model.activate_previous_profile()
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_time_us += 1.0
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertFalse(active_profile.size_linked)
            active_profile.width = active_profile.width + 16
            self.assertNotEqual(active_profile.width, active_profile.height)

    def test_linked_size_of_active_scan_profile_is_unlinked_by_current_frame_parameters_with_different_width_and_height(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            active_profile = scan_hardware_source.scan_profiles.active_profile
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            frame_parameters.pixel_size = Geometry.IntSize(h=128, w=256)
            scan_hardware_source.set_current_frame_parameters(frame_parameters)
            self.assertEqual((256, 128, False), (active_profile.width, active_profile.height, active_profile.size_linked))

    def test_scan_profile_saved_by_version_1_is_restored_and_saved_unchanged(self) -> None:
        properties = {
            "uuid": str(uuid.uuid4()), "name": "Survey",
            "field_width": 37.5, "width": 512, "height": 256, "width_height_linked": False, "pixel_time_us": 2.5, "rotation": 15.0, "ac_line_sync": True,
            "subscan_field_width_nm": 10.0, "subscan_field_height_nm": 5.0, "subscan_fractional_position_x_percent": 30.0, "subscan_fractional_position_y_percent": 40.0,
            "subscan_pixel_width": None, "subscan_rotation": 5.0,
            "field_width_included": True, "size_included": False, "pixel_time_us_included": True, "rotation_included": False, "ac_line_sync_included": True,
            "subscan_field_size_included": False, "subscan_fractional_position_included": True, "subscan_pixel_width_included": False, "subscan_rotation_included": True,
        }
        profile = ScanProfiles.ProfileModel.from_dict(properties)
        self.assertEqual((512, 256, False), (profile.width, profile.height, profile.size_linked))
        self.assertEqual((10.0, 5.0), (profile.subscan_field_width_nm, profile.subscan_field_height_nm))
        self.assertFalse(profile.size_included)
        self.assertEqual(properties, profile.to_dict())

    def test_copying_scan_profile_values_copies_every_value(self) -> None:
        source_profile = _make_scan_profile_with_every_parameter_changed()
        profile = ScanProfiles.ProfileModel(str())
        profile.copy_values_from(source_profile)
        self.assertEqual(_get_scan_profile_values(source_profile), _get_scan_profile_values(profile))

    def test_copying_scan_profile_values_copies_only_the_values_which_a_profile_includes(self) -> None:
        source_profile = _make_scan_profile_with_every_parameter_changed()
        # a new profile includes every value and the source profile includes none.
        profile = ScanProfiles.ProfileModel(str())
        profile.copy_values_from(source_profile, included_by=ScanProfiles.ProfileModel(str()))
        self.assertEqual(_get_scan_profile_values(source_profile), _get_scan_profile_values(profile))
        profile = ScanProfiles.ProfileModel(str())
        profile.copy_values_from(source_profile, included_by=source_profile)
        self.assertEqual(_get_scan_profile_values(ScanProfiles.ProfileModel(str())), _get_scan_profile_values(profile))

    def test_copying_scan_profile_included_flags_copies_every_flag(self) -> None:
        source_profile = _make_scan_profile_with_every_parameter_changed()
        profile = ScanProfiles.ProfileModel(str())
        profile.copy_included_from(source_profile)
        self.assertEqual(_get_scan_profile_included_flags(source_profile), _get_scan_profile_included_flags(profile))

    def test_scan_profile_restored_from_saved_properties_has_every_value_and_flag(self) -> None:
        source_profile = _make_scan_profile_with_every_parameter_changed()
        profile = ScanProfiles.ProfileModel.from_dict(json.loads(json.dumps(source_profile.to_dict())))
        self.assertEqual(_get_scan_profile_values(source_profile), _get_scan_profile_values(profile))
        self.assertEqual(_get_scan_profile_included_flags(source_profile), _get_scan_profile_included_flags(profile))

    def test_scan_profile_written_to_frame_parameters_reads_back_every_value(self) -> None:
        with self._test_context() as test_context:
            scan_hardware_source = typing.cast(ScanBase.ScanHardwareSource, test_context.scan_hardware_source)
            # the frame parameters do not carry the size link. the source profile's size is unlinked, and its width
            # differs from its height, so reading unlinks the size of the new profile too.
            source_profile = _make_scan_profile_with_every_parameter_changed()
            frame_parameters = scan_hardware_source.get_current_frame_parameters()
            source_profile.write_to_frame_parameters(frame_parameters)
            profile = ScanProfiles.ProfileModel(str())
            profile.read_from_frame_parameters(frame_parameters)
            # the angles and the subscan size are converted, so they read back only nearly equal.
            for source_value, value in zip(_get_scan_profile_values(source_profile), _get_scan_profile_values(profile), strict=True):
                self.assertAlmostEqual(source_value, value)


if __name__ == '__main__':
    unittest.main()
