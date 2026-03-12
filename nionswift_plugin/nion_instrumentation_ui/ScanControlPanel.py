from __future__ import annotations

# standard libraries
import asyncio
import copy
import dataclasses
import functools
import gettext
import logging
import logging.handlers
import math
import pkgutil
import sys
import threading
import typing

# third party libraries
# None

# local libraries
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation import AcquisitionPreferences
from nion.swift import DataItemThumbnailWidget
from nion.swift import DisplayPanel
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.swift.model import PlugInManager
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import DrawingContext
from nion.ui import PreferencesDialog
from nion.ui import UserInterface
from nion.ui import Widgets
from nion.ui import Window
from nion.utils import Binding
from nion.utils import Converter
from nion.utils import Geometry
from nion.utils import ListModel
from nion.utils import Model
from nion.utils import Observable
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.swift import DocumentController
    from nion.swift.model import DisplayItem
    from nion.swift.model import Persistence
    from nion.utils import Event
    from nion.swift.model import Schema


_ = gettext.gettext

map_channel_state_to_text = {
    "stopped": _("Stopped"),
    "complete": _("Acquiring"),
    "partial": _("Acquiring"),
    "marked": _("Stopping"),
    "error": _("Error"),
}


class ScanControlStateController:
    """
    Track the state of a scan controller, as it relates to the UI. This object does not hold any state itself.

    scan_controller may be None

    Scan controller should support the following API:
        (acquisition)
            (event) acquisition_state_changed_event(is_acquiring)
            (read-only property) hardware_source_id
            (read-only property) is_playing
            (read-only property) is_recording
            (read-only property) display_name
            (method) start_playing()
            (method) stop_playing()
            (method) abort_playing()
            (method) start_recording()
            (method) abort_recording()
        (event) profile_changed_event(profile_index)
        (event) frame_parameters_changed_event(profile_index, frame_parameters)
        (event) probe_state_changed_event(probe_state, probe_position): "parked", "scanning"
        (event) channel_state_changed_event(channel_index, channel_id, channel_name, enabled)
        (read-only property) selected_profile_index: return current profile index
        (read-only property) probe_state
        (read/write property) probe_position may be None
        (read-only property) channel_count
        (read-only property) data_channel_count
        (read-only property) subscan_state
        (method) set_selected_profile_index(profile_index): change the profile index
        (method) get_frame_parameters(profile_index)
        (method) set_frame_parameters(profile_index, frame_parameters)
        (method) get_channel_state(channel_index): channel_id, channel_name, enabled
        (method) set_channel_enabled(channel_index, enabled)
        (method) set_current_frame_parameters(frame_parameters)
        (method) get_current_frame_parameters()
        (method) set_record_frame_parameters(frame_parameters)
        (method) open_configuration_interface(api_broker)
        (method) start_simulator()
        (method) shift_click(mouse_position, camera_shape)
        (method) validate_probe_position()
        (method) increase_pmt(channel_index)
        (method) decrease_pmt(channel_index)
        (method) periodic()

    Clients should call:
        handle_change_profile(profile_label)
        handle_play_pause_clicked()
        handle_abort_clicked()
        handle_record_clicked()
        handle_subscan_enabled(enabled)
        handle_enable_channel(channel_index, enabled)
        handle_settings_button_clicked(api_broker)
        handle_shift_click(hardware_source_id, mouse_position, image_dimensions)
        handle_positioned_check_box(checked)
        handle_width_changed(width_str)
        handle_increase_width()
        handle_decrease_width()
        handle_height_changed(height_str)
        handle_increase_height()
        handle_decrease_height()
        handle_time_changed(time_str)
        handle_increase_time()
        handle_decrease_time()
        handle_fov_changed(fov_str)
        handle_increase_fov()
        handle_decrease_fov()
        handle_rotation_changed(rotation_str)
        handle_capture_clicked()
        handle_increase_pmt_clicked()
        handle_decrease_pmt_clicked()
        handle_periodic()

    Clients can respond to:
        on_display_name_changed(display_name)
        on_subscan_state_changed(subscan_state)
        on_profiles_changed(profile_label_list)
        on_profile_changed(profile_label)
        on_frame_parameters_changed(frame_parameters)
        on_linked_changed(linked)
        on_scan_button_state_changed(enabled, play_button_state)  scan, stop
        on_abort_button_state_changed(visible, enabled)
        on_record_button_state_changed(visible, enabled)
        on_record_abort_button_state_changed(visible, enabled)
        on_capture_button_state_changed(visible, enabled)
        on_display_new_data_item(data_item)
        (thread) on_channel_state_changed(channel_index, channel_id, channel_name, enabled)
        (thread) on_data_channel_state_changed(data_channel_index, data_channel_id, data_channel_name, enabled)
        (thread) on_probe_state_changed(probe_state, probe_position)  parked, scanning
        (thread) on_positioned_check_box_changed(checked)
        (thread) on_ac_line_sync_check_box_changed(checked)

    States
        View Machine: stopped, partial, marked, complete, suspended
        Recording Machine: stopped, partial, paused, complete
    """

    profiles = { _("Puma"): 0, _("Rabbit"): 1, _("Frame"): 2 }

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, document_controller: DocumentController.DocumentController, channel_id: typing.Optional[str]) -> None:
        self.__document_controller = document_controller
        self.__scan_hardware_source = scan_hardware_source
        self.queue_task = document_controller.queue_task
        self.__channel_id = channel_id
        self.__linked = True
        self.__profile_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__frame_parameters_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__acquisition_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__probe_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__channel_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__subscan_state_changed_listener: typing.Optional[Event.EventListener] = None
        self.__line_scan_state_changed_listener: typing.Optional[Event.EventListener] = None
        self.__drift_channel_id_listener: typing.Optional[Event.EventListener] = None
        self.__drift_region_listener: typing.Optional[Event.EventListener] = None
        self.__drift_settings_listener: typing.Optional[Event.EventListener] = None
        self.__data_channel_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__scan_frame_parameters_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__max_fov_stream_listener: typing.Optional[Event.EventListener] = None
        self.on_display_name_changed : typing.Optional[typing.Callable[[str], None]] = None
        self.on_subscan_state_changed : typing.Optional[typing.Callable[[stem_controller.SubscanState, stem_controller.LineScanState], None]] = None
        self.on_drift_state_changed : typing.Optional[typing.Callable[[typing.Optional[str], typing.Optional[Geometry.FloatRect], stem_controller.DriftCorrectionSettings, stem_controller.SubscanState], None]] = None
        self.on_profiles_changed : typing.Optional[typing.Callable[[typing.Sequence[str]], None]] = None
        self.on_profile_changed : typing.Optional[typing.Callable[[str], None]] = None
        self.on_frame_parameters_changed : typing.Optional[typing.Callable[[scan_base.ScanFrameParameters], None]] = None
        self.on_scan_frame_parameters_changed: typing.Optional[typing.Callable[[scan_base.ScanFrameParameters], None]] = None
        self.on_max_fov_changed: typing.Optional[typing.Callable[[float], None]] = None
        self.on_linked_changed : typing.Optional[typing.Callable[[bool], None]] = None
        self.on_channel_state_changed : typing.Optional[typing.Callable[[int, bool, bool], None]] = None
        self.on_data_channel_state_changed : typing.Optional[typing.Callable[[int, str, str, bool], None]] = None
        self.on_scan_button_state_changed : typing.Optional[typing.Callable[[bool, str], None]] = None
        self.on_abort_button_state_changed : typing.Optional[typing.Callable[[bool, bool], None]] = None
        self.on_record_button_state_changed : typing.Optional[typing.Callable[[bool, bool], None]] = None
        self.on_record_abort_button_state_changed : typing.Optional[typing.Callable[[bool, bool], None]] = None
        self.on_probe_state_changed : typing.Optional[typing.Callable[[str, typing.Optional[Geometry.FloatPoint]], None]] = None
        self.on_positioned_check_box_changed : typing.Optional[typing.Callable[[bool], None]] = None
        self.on_ac_line_sync_check_box_changed : typing.Optional[typing.Callable[[bool], None]] = None
        self.on_capture_button_state_changed : typing.Optional[typing.Callable[[bool, bool], None]] = None
        self.on_display_new_data_item : typing.Optional[typing.Callable[[DataItem.DataItem], None]] = None

        self.acquisition_state_model = Model.PropertyModel[typing.Dict[str, typing.Optional[str]]](dict())

        self.__captured_xdatas_available_listener: typing.Optional[Event.EventListener] = None

        document_model = document_controller.document_model

        self.data_item_reference = document_model.get_data_item_reference(document_model.make_data_item_reference_key(self.__scan_hardware_source.hardware_source_id, self.__channel_id))

    def close(self) -> None:
        if self.__captured_xdatas_available_listener:
            self.__captured_xdatas_available_listener.close()
            self.__captured_xdatas_available_listener = None
        if self.__profile_changed_event_listener:
            self.__profile_changed_event_listener.close()
            self.__profile_changed_event_listener = None
        if self.__frame_parameters_changed_event_listener:
            self.__frame_parameters_changed_event_listener.close()
            self.__frame_parameters_changed_event_listener = None
        self.__scan_frame_parameters_changed_event_listener = None
        self.__max_fov_stream_listener = None
        self.__data_channel_state_changed_event_listener = None
        if self.__acquisition_state_changed_event_listener:
            self.__acquisition_state_changed_event_listener.close()
            self.__acquisition_state_changed_event_listener = None
        if self.__probe_state_changed_event_listener:
            self.__probe_state_changed_event_listener.close()
            self.__probe_state_changed_event_listener = None
        if self.__channel_state_changed_event_listener:
            self.__channel_state_changed_event_listener.close()
            self.__channel_state_changed_event_listener = None
        if self.__subscan_state_changed_listener:
            self.__subscan_state_changed_listener.close()
            self.__subscan_state_changed_listener = None
        if self.__line_scan_state_changed_listener:
            self.__line_scan_state_changed_listener.close()
            self.__line_scan_state_changed_listener = None
        if self.__drift_channel_id_listener:
            self.__drift_channel_id_listener.close()
            self.__drift_channel_id_listener = None
        if self.__drift_region_listener:
            self.__drift_region_listener.close()
            self.__drift_region_listener = None
        if self.__drift_settings_listener:
            self.__drift_settings_listener.close()
            self.__drift_settings_listener = None
        self.on_display_name_changed = None
        self.on_subscan_state_changed = None
        self.on_drift_state_changed = None
        self.on_profiles_changed = None
        self.on_profile_changed = None
        self.on_frame_parameters_changed = None
        self.on_scan_frame_parameters_changed = None
        self.on_max_fov_changed = None
        self.on_linked_changed = None
        self.on_channel_state_changed = None
        self.on_data_channel_state_changed = None
        self.on_scan_button_state_changed = None
        self.on_abort_button_state_changed = None
        self.on_record_button_state_changed = None
        self.on_record_abort_button_state_changed = None
        self.on_probe_state_changed = None
        self.on_positioned_check_box_changed = None
        self.on_ac_line_sync_check_box_changed = None
        self.on_capture_button_state_changed = None
        self.on_display_new_data_item = None
        self.__scan_hardware_source = typing.cast(typing.Any, None)

    def __update_scan_button_state(self) -> None:
        if self.on_scan_button_state_changed:
            is_any_channel_enabled = any(self.__channel_enabled)
            enabled = self.__scan_hardware_source is not None and is_any_channel_enabled
            self.on_scan_button_state_changed(enabled, "stop" if self.is_playing else "scan")

    def __update_abort_button_state(self) -> None:
        if self.on_abort_button_state_changed:
            self.on_abort_button_state_changed(self.is_playing, self.is_playing)
        if self.on_capture_button_state_changed:
            self.on_capture_button_state_changed(self.is_playing, not self.__captured_xdatas_available_listener)

    def __update_record_button_state(self) -> None:
        if self.on_record_button_state_changed:
            is_any_channel_enabled = any(self.__channel_enabled)
            self.on_record_button_state_changed(True, not self.is_recording and is_any_channel_enabled)

    def __update_record_abort_button_state(self) -> None:
        if self.on_record_abort_button_state_changed:
            self.on_record_abort_button_state_changed(self.is_recording, self.is_recording)

    def __update_profile_state(self, profile_label: str) -> None:
        if callable(self.on_profile_changed):
            self.on_profile_changed(profile_label)

    def __update_frame_parameters(self, profile_index: int, frame_parameters: scan_base.ScanFrameParameters) -> None:
        if callable(self.on_frame_parameters_changed):
            if profile_index == self.__scan_hardware_source.selected_profile_index:
                self.on_frame_parameters_changed(frame_parameters)

    def __scan_frame_parameters_changed(self, frame_parameters_with_channels: scan_base.ScanFrameParameters) -> None:
        """ Called when the frame parameters change. """
        if self.on_scan_frame_parameters_changed:
            self.on_scan_frame_parameters_changed(frame_parameters_with_channels)

    def __max_fov_changed(self, max_fov_nm: float) -> None:
        """ Called when the maximum field of view changes. """
        if self.on_max_fov_changed:
            self.on_max_fov_changed(max_fov_nm)

    # received from the scan controller when the profile changes.
    # thread safe
    def __update_profile_index(self, profile_index: int) -> None:
        for (k,v) in ScanControlStateController.profiles.items():
            if v == profile_index:
                self.__update_profile_state(k)
                self.__update_frame_parameters(self.__scan_hardware_source.selected_profile_index, self.__scan_hardware_source.get_frame_parameters(profile_index))

    def __update_buttons(self) -> None:
        self.__update_scan_button_state()
        self.__update_abort_button_state()
        self.__update_record_button_state()
        self.__update_record_abort_button_state()

    def initialize_state(self) -> None:
        """ Call this to initialize the state of the UI after everything has been connected. """
        if self.__scan_hardware_source:
            stem_controller = self.__scan_hardware_source.stem_controller

            self.__profile_changed_event_listener = self.__scan_hardware_source.scan_settings.profile_changed_event.listen(self.__update_profile_index)
            self.__frame_parameters_changed_event_listener = self.__scan_hardware_source.scan_settings.frame_parameters_changed_event.listen(self.__update_frame_parameters)
            self.__data_channel_state_changed_event_listener = self.__scan_hardware_source.data_channel_state_changed_event.listen(self.__data_channel_state_changed)
            self.__acquisition_state_changed_event_listener = self.__scan_hardware_source.acquisition_state_changed_event.listen(self.__acquisition_state_changed)
            self.__probe_state_changed_event_listener = self.__scan_hardware_source.probe_state_changed_event.listen(self.__probe_state_changed)
            self.__channel_state_changed_event_listener = self.__scan_hardware_source.channel_state_changed_event.listen(self.__channel_state_changed)
            self.__scan_frame_parameters_changed_event_listener = self.__scan_hardware_source.scan_frame_parameters_changed_event.listen(self.__scan_frame_parameters_changed)
            self.__max_fov_stream_listener = self.__scan_hardware_source.max_field_of_view_nm_stream.value_stream.listen(self.__max_fov_changed)

            def drift_state_changed(name: str) -> None:
                if name in ("drift_channel_id", "drift_region", "drift_settings"):
                    if callable(self.on_drift_state_changed):
                        self.on_drift_state_changed(stem_controller.drift_channel_id,
                                                    stem_controller.drift_region,
                                                    stem_controller.drift_settings,
                                                    stem_controller.subscan_state)

            def subscan_state_changed(name: str) -> None:
                if name == "subscan_state":
                    if callable(self.on_subscan_state_changed):
                        self.on_subscan_state_changed(stem_controller.subscan_state, stem_controller.line_scan_state)
                    if callable(self.on_drift_state_changed):
                        self.on_drift_state_changed(stem_controller.drift_channel_id,
                                                    stem_controller.drift_region,
                                                    stem_controller.drift_settings,
                                                    stem_controller.subscan_state)

            self.__subscan_state_changed_listener = stem_controller.property_changed_event.listen(subscan_state_changed)
            subscan_state_changed("subscan_state")

            def line_scan_state_changed(name: str) -> None:
                if name == "line_scan_state":
                    if callable(self.on_subscan_state_changed):
                        self.on_subscan_state_changed(stem_controller.subscan_state, stem_controller.line_scan_state)
                    if callable(self.on_drift_state_changed):
                        # note: passing subscan state -- subscan state being not invalid is the same
                        # as line scan state being not invalid. ugh.
                        self.on_drift_state_changed(stem_controller.drift_channel_id,
                                                    stem_controller.drift_region,
                                                    stem_controller.drift_settings,
                                                    stem_controller.subscan_state)

            self.__line_scan_state_changed_listener = stem_controller.property_changed_event.listen(line_scan_state_changed)
            line_scan_state_changed("line_scan_state")

            self.__drift_channel_id_listener = stem_controller.property_changed_event.listen(drift_state_changed)
            self.__drift_region_listener = stem_controller.property_changed_event.listen(drift_state_changed)
            self.__drift_settings_listener = stem_controller.property_changed_event.listen(drift_state_changed)
            drift_state_changed("value")

        if self.on_display_name_changed:
            self.on_display_name_changed(self.display_name)
        if self.on_subscan_state_changed:
            self.on_subscan_state_changed(self.__scan_hardware_source.subscan_state, self.__scan_hardware_source.line_scan_state)
        if callable(self.on_drift_state_changed):
            self.on_drift_state_changed(self.__scan_hardware_source.drift_channel_id, self.__scan_hardware_source.drift_region, self.__scan_hardware_source.drift_settings, self.__scan_hardware_source.subscan_state)
        channel_count = self.__scan_hardware_source.channel_count
        self.__channel_enabled = [False] * channel_count
        for channel_index in range(channel_count):
            channel_state = self.__scan_hardware_source.get_channel_state(channel_index)
            self.__channel_state_changed(channel_index, channel_state.channel_id, channel_state.name, channel_state.enabled)
            self.__channel_enabled[channel_index] = channel_state.enabled
        self.__update_buttons()
        if self.on_profiles_changed:
            profile_items = list(ScanControlStateController.profiles.items())
            profile_items.sort(key=lambda k_v: k_v[1])
            profiles = list(map(lambda k_v: k_v[0], profile_items))
            self.on_profiles_changed(profiles)
            self.__update_profile_index(self.__scan_hardware_source.selected_profile_index)
        if self.on_linked_changed:
            self.on_linked_changed(self.__linked)
        probe_state = self.__scan_hardware_source.probe_state
        probe_position = self.__scan_hardware_source.probe_position
        self.__probe_state_changed(probe_state, probe_position)

    # must be called on ui thread
    def handle_change_profile(self, profile_label: str) -> None:
        if profile_label in ScanControlStateController.profiles:
            self.__scan_hardware_source.set_selected_profile_index(ScanControlStateController.profiles[profile_label])

    # must be called on ui thread
    def handle_play_pause_clicked(self) -> None:
        """ Call this when the user clicks the play/pause button. """
        if self.__scan_hardware_source:
            if self.is_playing:
                action_context = self.__document_controller._get_action_context()
                action_context.parameters["hardware_source_id"] = self.__scan_hardware_source.hardware_source_id
                self.__document_controller.perform_action_in_context("acquisition.stop_playing", action_context)
            else:
                # the 'enabled channel indexes' implementation is incomplete, so, for now, explicitly add them to the
                # frame parameters so that they can be recorded when logging the start playing action.
                frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
                frame_parameters.enabled_channel_indexes = self.__scan_hardware_source.get_enabled_channel_indexes()
                action_context = self.__document_controller._get_action_context()
                action_context.parameters["hardware_source_id"] = self.__scan_hardware_source.hardware_source_id
                action_context.parameters["frame_parameters"] = frame_parameters.as_dict()
                self.__document_controller.perform_action_in_context("acquisition.start_playing", action_context)

    # must be called on ui thread
    def handle_abort_clicked(self) -> None:
        """ Call this when the user clicks the abort button. """
        if self.__scan_hardware_source:
            action_context = self.__document_controller._get_action_context()
            action_context.parameters["hardware_source_id"] = self.__scan_hardware_source.hardware_source_id
            self.__document_controller.perform_action_in_context("acquisition.abort_playing", action_context)

    # must be called on ui thread
    def handle_record_clicked(self, callback_fn: typing.Callable[[DataItem.DataItem], None]) -> None:
        """ Call this when the user clicks the record button. """
        assert callable(callback_fn)
        if self.__scan_hardware_source:

            def finish_record(data_promise_list: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:
                record_index = self.__scan_hardware_source.record_index
                for data_promise in data_promise_list:
                    data_and_metadata = data_promise.xdata
                    if data_and_metadata:
                        data_item = DataItem.DataItem()
                        display_name = data_and_metadata.metadata.get("hardware_source", dict()).get("hardware_source_name")
                        display_name = display_name if display_name else _("Record")
                        channel_name = data_and_metadata.metadata.get("hardware_source", dict()).get("channel_name")
                        title_base = "{} ({})".format(display_name, channel_name) if channel_name else display_name
                        data_item.title = "{} {}".format(title_base, record_index)
                        data_item.set_xdata(data_and_metadata)
                        callback_fn(data_item)
                self.__scan_hardware_source.record_index += 1

            self.__scan_hardware_source.record_async(finish_record)

    # must be called on ui thread
    def handle_record_abort_clicked(self) -> None:
        """ Call this when the user clicks the abort button. """
        if self.__scan_hardware_source:
            self.__scan_hardware_source.abort_recording()

    # must be called on ui thread
    def handle_subscan_enabled(self, enabled: bool) -> None:
        self.__scan_hardware_source.line_scan_enabled = False
        self.__scan_hardware_source.subscan_enabled = enabled

    # must be called on ui thread
    def handle_line_scan_enabled(self, enabled: bool) -> None:
        self.__scan_hardware_source.subscan_enabled = False
        self.__scan_hardware_source.line_scan_enabled = enabled

    # must be called on ui thread
    def handle_drift_enabled(self, enabled: bool) -> None:
        self.__scan_hardware_source.drift_enabled = enabled

    # must be called on ui thread
    def handle_enable_channel(self, channel_index: int, enabled: bool) -> None:
        self.__scan_hardware_source.set_channel_enabled(channel_index, enabled)

    # must be called on ui thread
    def handle_settings_button_clicked(self, api_broker: typing.Any) -> None:
        self.__scan_hardware_source.open_configuration_interface(api_broker)

    # must be called on ui thread
    def handle_shift_click(self, hardware_source_id: str, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> bool:
        if hardware_source_id == self.__scan_hardware_source.hardware_source_id:
            self.__scan_hardware_source.shift_click(mouse_position, camera_shape, logger)
            return True
        return False

    def handle_positioned_check_box(self, checked: bool) -> None:
        if checked:
            self.__scan_hardware_source.validate_probe_position()
        else:
            self.__scan_hardware_source.probe_position = None

    def handle_ac_line_sync_check_box(self, checked: bool) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.ac_line_sync = checked
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def __update_frame_size(self, frame_parameters: scan_base.ScanFrameParameters, field: str) -> None:
        size = frame_parameters.pixel_size
        if self.__linked:
            if field == "width":
                size = Geometry.IntSize(size.width, size.width)
            else:
                size = Geometry.IntSize(size.height, size.height)
        frame_parameters.pixel_size = size
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_linked_changed(self, linked: bool) -> None:
        self.__linked = bool(linked)
        if self.on_linked_changed:
            self.on_linked_changed(self.__linked)
        if self.__linked:
            frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
            self.__update_frame_size(frame_parameters, "height")

    def handle_width_changed(self, width_str: str) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_size = Geometry.IntSize(int(frame_parameters.pixel_size[0]), int(width_str))
        self.__update_frame_size(frame_parameters, "width")

    def handle_decrease_width(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_size = Geometry.IntSize(int(frame_parameters.pixel_size[0]), int(frame_parameters.pixel_size[1]/2))
        self.__update_frame_size(frame_parameters, "width")

    def handle_increase_width(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_size = Geometry.IntSize(int(frame_parameters.pixel_size[0]), int(frame_parameters.pixel_size[1]*2))
        self.__update_frame_size(frame_parameters, "width")

    def handle_height_changed(self, height_str: str) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_size = Geometry.IntSize(int(height_str), int(frame_parameters.pixel_size[1]))
        self.__update_frame_size(frame_parameters, "height")

    def handle_decrease_height(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_size = Geometry.IntSize(int(frame_parameters.pixel_size[0]/2), int(frame_parameters.pixel_size[1]))
        self.__update_frame_size(frame_parameters, "height")

    def handle_increase_height(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_size = Geometry.IntSize(int(frame_parameters.pixel_size[0]*2), int(frame_parameters.pixel_size[1]))
        self.__update_frame_size(frame_parameters, "height")

    def handle_time_changed(self, time_str: str) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_time_us = float(time_str)
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_decrease_time(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_time_us = frame_parameters.pixel_time_us * 0.5
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_increase_time(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.pixel_time_us = frame_parameters.pixel_time_us * 2.0
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_fov_changed(self, fov_str: str) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.fov_nm = float(fov_str)
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_decrease_fov(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.fov_nm = frame_parameters.fov_nm * 0.5
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_increase_fov(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.fov_nm = frame_parameters.fov_nm * 2.0
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_rotation_changed(self, rotation_str: str) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.rotation_rad = float(rotation_str) * math.pi / 180.0
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    def handle_increase_pmt_clicked(self, channel_index: int) -> None:
        try:
            self.__scan_hardware_source.increase_pmt(channel_index)
        except Exception as e:
            from nion.swift.model import Notification
            notification = Notification.Notification("notification", "\N{MICROSCOPE} STEM Controller",
                                                     "Unable to change PMT",
                                                     f"Exception: {e}.")
            Notification.notify(notification)

    def handle_decrease_pmt_clicked(self, channel_index: int) -> None:
        try:
            self.__scan_hardware_source.decrease_pmt(channel_index)
        except Exception as e:
            from nion.swift.model import Notification
            notification = Notification.Notification("notification", "\N{MICROSCOPE} STEM Controller",
                                                     "Unable to change PMT",
                                                     f"Exception: {e}.")
            Notification.notify(notification)

    def handle_capture_clicked(self) -> None:
        def receive_new_xdatas(data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:
            if self.__captured_xdatas_available_listener:
                self.__captured_xdatas_available_listener.close()
                self.__captured_xdatas_available_listener = None
            document_model = self.__document_controller.document_model
            Acquisition.session_manager.begin_acquisition(document_model)  # bump the index
            for data_promise in data_promises:
                def add_data_item(data_item: DataItem.DataItem) -> None:
                    if self.on_display_new_data_item:
                        self.on_display_new_data_item(data_item)
                xdata = data_promise.xdata
                if xdata:
                    data_item = DataItem.new_data_item(xdata)
                    display_name = xdata.metadata.get("hardware_source", dict()).get("hardware_source_name")
                    display_name = display_name if display_name else _("Capture")
                    channel_name = xdata.metadata.get("hardware_source", dict()).get("channel_name")
                    acquisition_number = Acquisition.session_manager.get_project_acquisition_index(document_model)
                    data_item_title = display_name
                    if channel_name:
                        data_item_title += f" ({channel_name})"
                    if acquisition_number:
                        data_item_title += f" Capture {acquisition_number}"
                    data_item.title = data_item_title
                    data_item.session_metadata = ApplicationData.get_session_metadata_dict()
                    self.queue_task(functools.partial(add_data_item, data_item))
            self.queue_task(self.__update_buttons)

        self.__captured_xdatas_available_listener = self.__scan_hardware_source.xdatas_available_event.listen(receive_new_xdatas)
        self.__update_buttons()

    # must be called on ui thread
    def handle_periodic(self) -> None:
        if self.__scan_hardware_source and getattr(self.__scan_hardware_source, "periodic", None):
            self.__scan_hardware_source.periodic()

    @property
    def is_playing(self) -> bool:
        """ Returns whether the hardware source is playing or not. """
        return self.__scan_hardware_source.is_playing if self.__scan_hardware_source else False

    @property
    def is_recording(self) -> bool:
        """ Returns whether the hardware source is playing or not. """
        return self.__scan_hardware_source.is_recording if self.__scan_hardware_source else False

    @property
    def display_name(self) -> str:
        """ Returns the display name for the hardware source. """
        return self.__scan_hardware_source.display_name if self.__scan_hardware_source else _("N/A")

    def get_channel_enabled(self, channel_index: int) -> bool:
        return self.__scan_hardware_source.get_channel_enabled(channel_index)

    def get_channel_id(self, channel_index: int) -> str:
        channel_id = self.__scan_hardware_source.get_channel_id(channel_index)
        assert channel_id
        return channel_id

    def get_channel_name(self, channel_index: int) -> str:
        channel_name = self.__scan_hardware_source.get_channel_name(channel_index)
        assert channel_name
        return channel_name

    # this message comes from the data buffer. it will always be invoked on a thread.
    def __acquisition_state_changed(self, is_playing: bool) -> None:
        if self.__captured_xdatas_available_listener:
            self.__captured_xdatas_available_listener.close()
            self.__captured_xdatas_available_listener = None
        self.queue_task(self.__update_buttons)

    def __data_channel_state_changed(self, data_channel_event_args: HardwareSource.DataChannelEventArgs) -> None:
        # the value (dict) does not get copied; so copy it here.
        acquisition_states = copy.deepcopy(self.acquisition_state_model.value) or dict()
        channel_id = data_channel_event_args.channel_id or "unknown"
        acquisition_states[channel_id] = data_channel_event_args.data_channel_state
        self.acquisition_state_model.value = acquisition_states

    def __probe_state_changed(self, probe_state: str, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        if self.on_probe_state_changed:
            self.on_probe_state_changed(probe_state, probe_position)
        if self.on_positioned_check_box_changed:
            self.on_positioned_check_box_changed(probe_position is not None)

    def __channel_state_changed(self, channel_index: int, channel_id: str, name: str, enabled: bool) -> None:
        if self.on_channel_state_changed:
            is_subscan_channel = self.__scan_hardware_source.subscan_state == stem_controller.SubscanState.ENABLED or self.__scan_hardware_source.line_scan_state == stem_controller.LineScanState.ENABLED
            self.on_channel_state_changed(channel_index, enabled, is_subscan_channel)
        data_channel_state_changed = self.on_data_channel_state_changed
        if callable(data_channel_state_changed):
            data_channel_state_changed(channel_index, channel_id, name, enabled)
            data_channel_state_changed(
                channel_index + self.__scan_hardware_source.channel_count,
                channel_id + "_subscan",
                name + " " + _("Subscan"),
                enabled)
        was_any_channel_enabled = any(self.__channel_enabled)
        self.__channel_enabled[channel_index] = enabled
        is_any_channel_enabled = any(self.__channel_enabled)
        if was_any_channel_enabled != is_any_channel_enabled:
            self.__update_scan_button_state()
            self.__update_record_button_state()


class ButtonCell(CanvasItem.Cell):
    def __init__(self) -> None:
        super().__init__()
        self.fill_style = "rgb(128, 128, 128)"
        self.fill_style_pressed = "rgb(64, 64, 64)"
        self.fill_style_disabled = "rgb(192, 192, 192)"
        self.border_style: typing.Optional[str] = None
        self.border_style_pressed: typing.Optional[str] = None
        self.border_style_disabled: typing.Optional[str] = None
        self.stroke_style = "#FFF"
        self.stroke_width = 3.0


class IconCell(ButtonCell):
    def __init__(self, icon_id: str) -> None:
        super().__init__()
        self.__icon_id = icon_id

    def _paint_cell(self, drawing_context: DrawingContext.DrawingContext, rect: Geometry.FloatRect, style: typing.Set[str]) -> None:
        canvas_size = rect.size
        enabled = "disabled" not in style
        active = "active" in style
        with drawing_context.saver():
            drawing_context.translate(rect.left, rect.top)
            center_x = canvas_size.width * 0.5
            center_y = canvas_size.height * 0.5
            drawing_context.begin_path()
            drawing_context.move_to(center_x + 7.0, center_y)
            drawing_context.arc(center_x, center_y, 7.0, 0, 2 * math.pi)
            if enabled:
                if active:
                    if self.fill_style_pressed:
                        drawing_context.fill_style = self.fill_style_pressed
                        drawing_context.fill()
                    if self.border_style_pressed:
                        drawing_context.stroke_style = self.border_style_pressed
                        drawing_context.stroke()
                else:
                    if self.fill_style:
                        drawing_context.fill_style = self.fill_style
                        drawing_context.fill()
                    if self.border_style:
                        drawing_context.stroke_style = self.border_style
                        drawing_context.stroke()
            else:
                if self.fill_style_disabled:
                    drawing_context.fill_style = self.fill_style_disabled
                    drawing_context.fill()
                if self.border_style_disabled:
                    drawing_context.stroke_style = self.border_style_disabled
                    drawing_context.stroke()
            drawing_context.begin_path()
            if self.__icon_id == "plus":
                drawing_context.move_to(center_x, center_y - 4.0)
                drawing_context.line_to(center_x, center_y + 4.0)
                drawing_context.move_to(center_x - 4.0, center_y)
                drawing_context.line_to(center_x + 4.0, center_y)
            elif self.__icon_id == "minus":
                drawing_context.move_to(center_x - 3.0, center_y)
                drawing_context.line_to(center_x + 3.0, center_y)
            drawing_context.stroke_style = self.stroke_style
            drawing_context.line_width = self.stroke_width
            drawing_context.stroke()
            drawing_context.begin_path()


class CharCell(ButtonCell):
    def __init__(self, char: str) -> None:
        super().__init__()
        self.padding = Geometry.IntSize()
        self.__char = char

    def _paint_cell(self, drawing_context: DrawingContext.DrawingContext, rect: Geometry.FloatRect, style: typing.Set[str]) -> None:
        enabled = "disabled" not in style
        active = "active" in style
        with drawing_context.saver():
            drawing_context.translate(rect.left, rect.top)
            drawing_context.begin_path()
            drawing_context.round_rect(1, 1, rect.width - 2, rect.height - 2, 2.0)
            if enabled:
                if active:
                    if self.fill_style_pressed:
                        drawing_context.fill_style = self.fill_style_pressed
                        drawing_context.fill()
                    if self.border_style_pressed:
                        drawing_context.stroke_style = self.border_style_pressed
                        drawing_context.stroke()
                else:
                    if self.fill_style:
                        drawing_context.fill_style = self.fill_style
                        drawing_context.fill()
                    if self.border_style:
                        drawing_context.stroke_style = self.border_style
                        drawing_context.stroke()
            else:
                if self.fill_style_disabled:
                    drawing_context.fill_style = self.fill_style_disabled
                    drawing_context.fill()
                if self.border_style_disabled:
                    drawing_context.stroke_style = self.border_style_disabled
                    drawing_context.stroke()
            drawing_context.begin_path()
            drawing_context.text_align = "center"
            drawing_context.text_baseline = "bottom"
            drawing_context.fill_style = self.stroke_style
            drawing_context.fill_text(self.__char, rect.width / 2, rect.height / 2 + 5.5)


class ButtonCellCanvasItem(CanvasItem.CellCanvasItem):

    def __init__(self, button_cell: ButtonCell) -> None:
        super().__init__(button_cell)
        self.__button_cell = button_cell

    @property
    def fill_style(self) -> str:
        return self.__button_cell.fill_style

    @fill_style.setter
    def fill_style(self, value: str) -> None:
        if value != self.__button_cell.fill_style:
            self.__button_cell.fill_style = value
            self.update()

    @property
    def fill_style_pressed(self) -> str:
        return self.__button_cell.fill_style_pressed

    @fill_style_pressed.setter
    def fill_style_pressed(self, value: str) -> None:
        if value != self.__button_cell.fill_style_pressed:
            self.__button_cell.fill_style_pressed = value
            self.update()

    @property
    def fill_style_disabled(self) -> str:
        return self.__button_cell.fill_style_disabled

    @fill_style_disabled.setter
    def fill_style_disabled(self, value: str) -> None:
        if value != self.__button_cell.fill_style_disabled:
            self.__button_cell.fill_style_disabled = value
            self.update()

    @property
    def border_style(self) -> typing.Optional[str]:
        return self.__button_cell.border_style

    @border_style.setter
    def border_style(self, value: typing.Optional[str]) -> None:
        if value != self.__button_cell.border_style:
            self.__button_cell.border_style = value
            self.update()

    @property
    def border_style_pressed(self) -> typing.Optional[str]:
        return self.__button_cell.border_style_pressed

    @border_style_pressed.setter
    def border_style_pressed(self, value: typing.Optional[str]) -> None:
        if value != self.__button_cell.border_style_pressed:
            self.__button_cell.border_style_pressed = value
            self.update()

    @property
    def border_style_disabled(self) -> typing.Optional[str]:
        return self.__button_cell.border_style_disabled

    @border_style_disabled.setter
    def border_style_disabled(self, value: typing.Optional[str]) -> None:
        if value != self.__button_cell.border_style_disabled:
            self.__button_cell.border_style_disabled = value
            self.update()

    @property
    def stroke_style(self) -> str:
        return self.__button_cell.stroke_style

    @stroke_style.setter
    def stroke_style(self, value: str) -> None:
        if value != self.__button_cell.stroke_style:
            self.__button_cell.stroke_style = value
            self.update()

    @property
    def stroke_width(self) -> float:
        return self.__button_cell.stroke_width

    @stroke_width.setter
    def stroke_width(self, value: float) -> None:
        if value != self.__button_cell.stroke_width:
            self.__button_cell.stroke_width = value
            self.update()


class IconCanvasItem(ButtonCellCanvasItem):

    def __init__(self, icon_id: str) -> None:
        super().__init__(IconCell(icon_id))
        self.wants_mouse_events = True
        self.update_sizing(self.sizing.with_fixed_size(Geometry.IntSize(18, 18)))


class CharButtonCanvasItem(ButtonCellCanvasItem):

    def __init__(self, char: str) -> None:
        super().__init__(CharCell(char))
        self.wants_mouse_events = True
        self.fill_style = "rgb(255, 255, 255)"
        self.fill_style_pressed = "rgb(128, 128, 128)"
        self.fill_style_disabled = "rgb(192, 192, 192)"
        self.border_style = "rgb(192, 192, 192)"
        self.border_style_pressed = "rgb(128, 128, 128)"
        self.border_style_disabled = "rgb(192, 192, 192)"
        self.stroke_style = "#000"
        self.border_enabled = False


class CharButtonWidget(UserInterface.Widget):
    def __init__(self, ui: UserInterface.UserInterface, text: str):
        fm = ui.get_font_metrics("", "M")
        height = fm.height + 4
        width = (fm.width + 3) if sys.platform == "win32" else (fm.width + 4)  # less padding on Windows to look right
        column_widget = ui.create_column_widget(properties={"height": height, "width": width})
        super().__init__(Widgets.CompositeWidgetBehavior(column_widget))

        self.__text = text

        self.on_clicked = None

        def button_clicked() -> None:
            if callable(self.on_clicked):
                self.on_clicked()

        canvas_item = CharButtonCanvasItem(text)
        canvas_item.on_clicked = button_clicked

        canvas_widget = ui.create_canvas_widget()
        canvas_widget.canvas_item.add_canvas_item(canvas_item)

        # ugh. this is a partially working stop-gap when a canvas item is in a widget it will not get mouse exited reliably
        root_container = canvas_item.root_container
        if root_container and root_container.canvas_widget:
            canvas_widget.on_mouse_exited = root_container.canvas_widget.on_mouse_exited

        self.__canvas_item = canvas_item

        column_widget.add(canvas_widget)

        def get_text() -> str:
            return self.__text

        def set_text(value: str) -> None:
            self.__text = str(value)

        self.__text_binding_helper = UserInterface.BindablePropertyHelper[str](get_text, set_text)

        self.text = text

    def close(self) -> None:
        self.__text_binding_helper.close()
        self.__text_binding_helper = typing.cast(typing.Any, None)
        self.on_clicked = None
        super().close()

    @property
    def text(self) -> str:
        return self.__text_binding_helper.value

    @text.setter
    def text(self, text: str) -> None:
        self.__text_binding_helper.value = text

    def bind_text(self, binding: Binding.Binding) -> None:
        self.__text_binding_helper.bind_value(binding)

    def unbind_text(self) -> None:
        self.__text_binding_helper.unbind_value()


class LinkedFieldsCheckBoxCanvasItemComposer(CanvasItem.BaseComposer):
    def __init__(self, canvas_item: CanvasItem.AbstractCanvasItem, layout_sizing: CanvasItem.Sizing, composer_cache: CanvasItem.ComposerCache, check_state: str, mouse_inside: bool, mouse_checked: bool) -> None:
        super().__init__(canvas_item, layout_sizing, composer_cache)
        self.__check_state = check_state
        self.__mouse_inside = mouse_inside
        self.__mouse_pressed = mouse_checked

    def _repaint(self, drawing_context: DrawingContext.DrawingContext, canvas_bounds: Geometry.IntRect, composer_cache: CanvasItem.ComposerCache) -> None:
        with drawing_context.saver():
            drawing_context.translate(canvas_bounds.left, canvas_bounds.top)
            drawing_context.begin_path()
            canvas_width = canvas_bounds.width
            canvas_height = canvas_bounds.height
            check_state = self.__check_state
            if check_state == "checked":
                drawing_context.move_to(2, 2)
                drawing_context.line_to(canvas_width - 2, 2)
                drawing_context.line_to(canvas_width - 2, canvas_height - 2)
                drawing_context.line_to(2, canvas_height - 2)
            else:
                # draw the top bar and down stroke
                drawing_context.move_to(2, 2)
                drawing_context.line_to(canvas_width - 2, 2)
                drawing_context.line_to(canvas_width - 2, canvas_height * 0.5 - 3)
                # draw the top crossbar
                drawing_context.move_to(canvas_width, canvas_height * 0.5 - 3)
                drawing_context.line_to(canvas_width - 4, canvas_height * 0.5 - 3)
                # draw the bottom crossbar
                drawing_context.move_to(canvas_width, canvas_height * 0.5 + 3)
                drawing_context.line_to(canvas_width - 4, canvas_height * 0.5 + 3)
                # draw the down stroke and bottom bar
                drawing_context.move_to(canvas_width - 2, canvas_height * 0.5 + 3)
                drawing_context.line_to(canvas_width - 2, canvas_height - 2)
                drawing_context.line_to(2, canvas_height - 2)
            if self.__mouse_pressed:
                drawing_context.stroke_style = "#CCC"
            else:
                drawing_context.stroke_style = "#888"
            if self.__mouse_inside:
                drawing_context.line_width = 3.0
            else:
                drawing_context.line_width = 1.5
            drawing_context.stroke()


class LinkedFieldsCheckBoxCanvasItem(CanvasItem.CheckBoxCanvasItem):
    def __init__(self) -> None:
        super().__init__()
        self.update_sizing(self.sizing.with_fixed_size(Geometry.IntSize(w=10, h=30)))

    def _get_composer(self, composer_cache: CanvasItem.ComposerCache) -> typing.Optional[CanvasItem.BaseComposer]:
        return LinkedFieldsCheckBoxCanvasItemComposer(self, self.layout_sizing, composer_cache, self.check_state, self._mouse_inside, self._mouse_pressed)


class LinkedFieldsCheckBoxWidget(UserInterface.Widget):
    """A linked checkbox widget.

    Presents a line/broken-line icon that can be toggled between checked and unchecked to represent linking between
    two vertically stacked text fields.
    """

    def __init__(self, ui: UserInterface.UserInterface) -> None:
        canvas_item = LinkedFieldsCheckBoxCanvasItem()
        properties = {"height": canvas_item.sizing.preferred_height, "width": canvas_item.sizing.preferred_width}
        widget = ui.create_canvas_widget(properties=properties)
        widget.canvas_item.add_canvas_item(canvas_item)
        super().__init__(Widgets.CompositeWidgetBehavior(widget))
        self.on_checked_changed: typing.Callable[[bool], None] | None = None

        def set_checked(value: bool) -> None:
            canvas_item.checked = value

        self.__checked_binding_helper = UserInterface.BindablePropertyHelper[bool](None, set_checked)

        def handle_checked_changed(checked: bool) -> None:
            if checked != self.checked:
                self.checked = checked
                if callable(self.on_checked_changed):
                    self.on_checked_changed(checked)

        canvas_item.on_checked_changed = handle_checked_changed

    def close(self) -> None:
        self.__checked_binding_helper.close()
        self.__checked_binding_helper = typing.cast(typing.Any, None)  # help catch use after close
        super().close()

    @property
    def checked(self) -> bool:
        return self.__checked_binding_helper.value

    @checked.setter
    def checked(self, value: bool) -> None:
        self.__checked_binding_helper.value = value

    def bind_checked(self, binding: Binding.Binding) -> None:
        self.__checked_binding_helper.bind_value(binding)

    def unbind_checked(self) -> None:
        self.__checked_binding_helper.unbind_value()


class LinkedFieldsCheckBoxFactory:
    """Declarative factory for the linked checkbox control.

    Clients should create declarative descriptions via the create_linked_fields_check_box() static method.
    """

    WIDGET_TYPE = "widget.acquisition.linked-check-box"

    def construct(self, d_type: str, ui: UserInterface.UserInterface, window: Window.Window | None,
                  d: Declarative.UIDescription, handler: Declarative.HandlerLike,
                  finishes: list[typing.Callable[[], None]]) -> UserInterface.Widget | None:
        """Construct a linked checkbox widget.

        This is a callback from the declarative UI engine.

        Returns the constructed widget or None if the d_type does not match.
        """
        if d_type == LinkedFieldsCheckBoxFactory.WIDGET_TYPE:
            widget = LinkedFieldsCheckBoxWidget(ui)
            Declarative.connect_name(widget, d, handler)
            Declarative.connect_reference_value(widget, d, handler, "checked", finishes, value_type=bool)
            Declarative.connect_event(widget, widget, d, handler, "on_checked_changed", ["checked"])
            Declarative.connect_attributes(widget, d, handler, finishes)
            return widget
        return None

    @staticmethod
    def create_linked_fields_check_box(*, name: Declarative.UIIdentifier | None = None,
                                checked: str | None = None,
                                on_checked_changed: Declarative.UICallableIdentifier | None = None,
                                **kwargs: typing.Any) -> Declarative.UIDescriptionResult:
        """Create a declarative description for a linked checkbox.

        Returns the declarative description.
        """
        d = {"type": LinkedFieldsCheckBoxFactory.WIDGET_TYPE}
        if name is not None:
            d["name"] = name
        if checked is not None:
            d["checked"] = checked
        if on_checked_changed is not None:
            d["on_checked_changed"] = on_checked_changed
        return d | kwargs


Registry.register_component(LinkedFieldsCheckBoxFactory(), {"declarative_constructor"})


class ThreadHelper:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.__event_loop = event_loop
        self.__pending_calls: typing.Dict[str, asyncio.Handle] = dict()

    def close(self) -> None:
        for handle in self.__pending_calls.values():
            handle.cancel()
        self.__pending_calls = dict()

    def call_on_main_thread(self, key: str, func: typing.Callable[[], None]) -> None:
        if threading.current_thread() != threading.main_thread():
            handle = self.__pending_calls.pop(key, None)
            if handle:
                handle.cancel()
            self.__pending_calls[key] = self.__event_loop.call_soon_threadsafe(func)
        else:
            func()


class DataItemReferenceThumbnailWidget(DataItemThumbnailWidget.ThumbnailWidget):
    """Thumbnail widget to display the thumbnail for a data item reference.

    The data_item_reference property is used to set the data item reference to display and supports binding.

    The document_controller property also needs to be set, but is not bindable.
    """

    def __init__(self, ui: UserInterface.UserInterface, size: Geometry.IntSize, properties: Persistence.PersistentDictType | None = None) -> None:
        super().__init__(ui, None, size, properties)
        self.__data_item_reference: DocumentModel.DocumentModel.DataItemReference | None = None
        self.__document_controller: DocumentController.DocumentController | None = None

        def get_data_item_reference() -> DocumentModel.DocumentModel.DataItemReference | None:
            return self.__data_item_reference

        def set_data_item_reference(value: DocumentModel.DocumentModel.DataItemReference | None) -> None:
            self.__data_item_reference = value
            self.__update()

        self.__data_item_reference_binding_helper = UserInterface.BindablePropertyHelper[DocumentModel.DocumentModel.DataItemReference | None](get_data_item_reference, set_data_item_reference)

        self.on_drag = self.drag

    def close(self) -> None:
        self.__data_item_reference_binding_helper.close()
        self.__data_item_reference_binding_helper = typing.cast(typing.Any, None)
        super().close()

    def __update(self) -> None:
        # when the document controller (window) or data item reference changes, update the thumbnail source.
        if self.__document_controller and self.__data_item_reference:
            data_item_reference = self.__data_item_reference
            thumbnail_source = DataItemThumbnailWidget.DataItemReferenceThumbnailSource(self.__document_controller, data_item_reference)
            self.set_thumbnail_source(thumbnail_source)

    @property
    def data_item_reference(self) -> DocumentModel.DocumentModel.DataItemReference | None:
        return self.__data_item_reference_binding_helper.value

    @data_item_reference.setter
    def data_item_reference(self, data_item_reference: DocumentModel.DocumentModel.DataItemReference | None) -> None:
        self.__data_item_reference_binding_helper.value = data_item_reference

    def bind_data_item_reference(self, binding: Binding.Binding) -> None:
        self.__data_item_reference_binding_helper.bind_value(binding)

    def unbind_data_item_reference(self) -> None:
        self.__data_item_reference_binding_helper.unbind_value()

    @property
    def document_controller(self) -> DocumentController.DocumentController | None:
        return self.__document_controller

    @document_controller.setter
    def document_controller(self, document_controller: DocumentController.DocumentController | None) -> None:
        self.__document_controller = document_controller
        self.__update()


class DeclarativeDataItemReferenceThumbnailFactory:
    """Declarative factory for the DataItemReferenceThumbnailWidget.

    Clients should create instances via the create_data_item_reference_thumbnail() static method.
    """

    WIDGET_TYPE = "widget.acquisition.data-item-reference-thumbnail"

    def construct(self, d_type: str, ui: UserInterface.UserInterface, window: Window.Window | None, d: Declarative.UIDescription, handler: Declarative.HandlerLike, finishes: typing.List[typing.Callable[[], None]]) -> UserInterface.Widget | None:
        if d_type == DeclarativeDataItemReferenceThumbnailFactory.WIDGET_TYPE:
            properties = Declarative.construct_sizing_properties(d)

            widget = DataItemReferenceThumbnailWidget(ui, size=Geometry.IntSize(properties.get("height", 48), properties.get("width", 48)), properties=properties)

            if handler:
                Declarative.connect_name(widget, d, handler)
                Declarative.connect_reference_value(widget, d, handler, "data_item_reference", finishes)
                Declarative.connect_reference_value(widget, d, handler, "document_controller", finishes)
                Declarative.connect_attributes(widget, d, handler, finishes)

            return widget

        return None

    @staticmethod
    def create_data_item_reference_thumbnail(*,
                                             window: typing.Optional[Declarative.UIIdentifier],
                                             data_item_reference: typing.Optional[Declarative.UIIdentifier],
                                             **kwargs: typing.Any) -> Declarative.UIDescriptionResult:
        return {
            "type": DeclarativeDataItemReferenceThumbnailFactory.WIDGET_TYPE,
            "document_controller": window,
            "data_item_reference": data_item_reference,
            "width": 48,
            "height": 48,
        }


# register the declarative factory
Registry.register_component(DeclarativeDataItemReferenceThumbnailFactory(), {"declarative_constructor"})


class CharButtonFactory:
    """Declarative factory for the character button (scan control panel specific).

    Clients should create instances via the create_char_button() static method.
    """

    WIDGET_TYPE = "widget.acquisition.char-button"
    DEFAULT_WIDTH = 15
    DEFAULT_HEIGHT = 20
    TEXT_OFFSET = 4.5

    def construct(self, d_type: str, ui: UserInterface.UserInterface, window: typing.Optional[Window.Window],
                  d: Declarative.UIDescription, handler: Declarative.HandlerLike,
                  finishes: list[typing.Callable[[], None]]) -> UserInterface.Widget | None:
        """Construct a character button widget.

        This is a callback from the declarative UI engine.

        Returns the constructed widget or None if the d_type does not match.
        """
        if d_type == CharButtonFactory.WIDGET_TYPE:
            text = d["text"]
            character = text[0] if text else " "
            widget = CharButtonWidget(ui, character)
            Declarative.connect_name(widget, d, handler)
            Declarative.connect_event(widget, widget, d, handler, "on_clicked", [])
            Declarative.connect_attributes(widget, d, handler, finishes)
            return widget
        return None

    @staticmethod
    def create_char_button(*, text: Declarative.UILabel | None = None, name: Declarative.UIIdentifier | None = None,
                           on_clicked: Declarative.UICallableIdentifier | None = None,
                           **kwargs: typing.Any) -> Declarative.UIDescriptionResult:
        """Create a declarative description for a character button with the given text, name, and on_clicked handler.

        Returns the description.
        """
        return {
            "type": CharButtonFactory.WIDGET_TYPE,
            "text": text,
            "on_clicked": on_clicked,
            "width": CharButtonFactory.DEFAULT_WIDTH,
            "height": CharButtonFactory.DEFAULT_HEIGHT
        } | kwargs


Registry.register_component(CharButtonFactory(), {"declarative_constructor"})


class ChannelModel(Observable.Observable):
    """Model for a scan channel.

    This model observes both scan hardware source channel_state_changed_event and stem_controller
    property_changed_event and updates its properties when either changes. The stem_controller property_changed_event
    is observed to track subscan and line scan state changes.

    The channel model provides the observable properties:
    - channel_id
    - name
    - enabled
    - data_item_reference

    The data item reference is updated when the subscan or line scan state changes and will reflect the
    appropriate data item for the channel and state. The document model is required to get the data item reference.

    State changes from the scan hardware source may occur on a different thread so the event loop is used to marshal
    property change notifications to the main thread.
    """
    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, channel_index: int, document_model: DocumentModel.DocumentModel, event_loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.__scan_hardware_source = scan_hardware_source
        self.__document_model = document_model
        self.__event_loop = event_loop
        self.__channel_state_changed_event_listener = self.__scan_hardware_source.channel_state_changed_event.listen(self.__channel_state_changed)
        self.__property_changed_event_listener = self.__scan_hardware_source.stem_controller.property_changed_event.listen(self.__handle_property_changed)
        self.__name = str()
        self.__channel_index = channel_index
        self.__enabled = False
        self.__channel_id = self.__scan_hardware_source.get_channel_id(channel_index) or str()
        self.__actual_channel_id: str | None = None
        self.__data_item_reference: DocumentModel.DocumentModel.DataItemReference | None = None
        self.__pending_call_lock = threading.RLock()
        self.__pending_call: typing.Optional[asyncio.Handle] = None
        self.__handle_state_changed_on_ui_thread()

    def close(self) -> None:
        with self.__pending_call_lock:
            if self.__pending_call is not None:
                self.__pending_call.cancel()
            self.__pending_call = None
            self.__channel_state_changed_event_listener = typing.cast(typing.Any, None)
            self.__property_changed_event_listener = typing.cast(typing.Any, None)

    def __handle_state_changed_on_ui_thread(self) -> None:
        channel_name = self.__scan_hardware_source.get_channel_name(self.__channel_index) or str()
        channel_enabled = self.__scan_hardware_source.get_channel_enabled(self.__channel_index)
        if self.__name != channel_name:
            self.__name = channel_name
            self.notify_property_changed("name")
        if self.__enabled != channel_enabled:
            self.__enabled = channel_enabled
            self.notify_property_changed("enabled")
        subscan_state = self.__scan_hardware_source.stem_controller.subscan_state
        line_scan_state = self.__scan_hardware_source.stem_controller.line_scan_state
        is_subscan_channel = subscan_state == stem_controller.SubscanState.ENABLED or line_scan_state == stem_controller.LineScanState.ENABLED
        actual_channel_id = self.__channel_id if not is_subscan_channel else self.__channel_id + "_subscan"
        if actual_channel_id != self.__actual_channel_id:
            self.__actual_channel_id = actual_channel_id
            self.__data_item_reference = self.__document_model.get_data_item_reference(self.__document_model.make_data_item_reference_key(self.__scan_hardware_source.hardware_source_id, actual_channel_id))
            self.notify_property_changed("data_item_reference")

    def __handle_state_changed(self) -> None:
        # use async to call handle_property_changed on the main thread if an existing call is not pending.
        if threading.current_thread() != threading.main_thread():
            with self.__pending_call_lock:
                if not self.__pending_call:
                    self.__pending_call = self.__event_loop.call_soon_threadsafe(self.__handle_state_changed_on_ui_thread)
        else:
            self.__handle_state_changed_on_ui_thread()

    def __channel_state_changed(self, channel_index: int, channel_id: str, name: str, enabled: bool) -> None:
        if channel_index == self.__channel_index:
            self.__handle_state_changed()

    def __handle_property_changed(self, property_name: str) -> None:
        self.__handle_state_changed()

    @property
    def channel_id(self) -> str:
        return self.__channel_id

    @property
    def name(self) -> str:
        return self.__name

    @property
    def enabled(self) -> bool:
        return self.__enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self.__scan_hardware_source.set_channel_enabled(self.__channel_index, enabled)

    @property
    def data_item_reference(self) -> DocumentModel.DocumentModel.DataItemReference | None:
        return self.__data_item_reference


class ChannelHandler(Declarative.Handler):
    """Declarative handler for a scan channel.

    Displays a thumbnail and a checkbox for enabling/disabling the channel.
    """
    def __init__(self, channel: ChannelModel, document_controller: DocumentController.DocumentController, scan_hardware_source: scan_base.ScanHardwareSource) -> None:
        super().__init__()

        self.document_controller = document_controller
        self._channel = channel

        thumbnail = DeclarativeDataItemReferenceThumbnailFactory.create_data_item_reference_thumbnail(window="@binding(document_controller)", data_item_reference="@binding(_channel.data_item_reference)")

        u = Declarative.DeclarativeUI()

        self.ui_view = u.create_column(
            thumbnail,
            u.create_check_box(text="@binding(_channel.name)", checked="@binding(_channel.enabled)"),
            u.create_stretch()
        )


class ScanControlPanelModel(Observable.Observable):
    """Model for the scan control panel.

    The scan hardware source and stem controller directly represent the hardware.

    The scan settings represent saved sets of scan settings (profiles).

    Observes the scan hardware source for:
    - scan settings profile changes (profile_index)
    - scan settings current frame parameters changes (to track width, height, pixel time, fov, rotation)
    - stem controller property changes (to track subscan and line scan state changes)
    - scan hardware source data channel state changes (to track acquisition state for enabling/disabling buttons)
    - scan hardware source acquisition state changes (to track acquisition state for enabling/disabling buttons)
    - stem controller probe state changes (to track probe state text and position)
    - scan hardware source channel state changes (to track channel enabled state)
    - scan hardware source scan frame parameters changes (to track max fov color changes)
    - scan hardware source max field of view stream changes (to track max fov color changes)

    All observed changes are marshaled to the main thread using the document controller event loop through
    __handle_state_changed which eventually calls __handle_state_changed_on_ui_thread.

    __handle_state_changed_on_ui_thread updates all observable properties as needed and notifies property changes.

    The following observable properties are provided:
    - profiles_model (read only)
    - profile_index (read/write)
    - width_str (read/write)
    - height_str (read/write)
    - width_height_linked (read/write)
    - pixel_time_str (read/write)
    - fov_str (read/write)
    - rotation_deg_str (read/write)
    - subscan_checkbox_enabled (read only)
    - subscan_checkbox_checked (read/write)
    - line_scan_checkbox_enabled (read only)
    - line_scan_checkbox_checked (read/write)
    - drift_controls_enabled (read only)
    - drift_checkbox_checked (read/write)
    - drift_settings_interval_str (read/write)
    - drift_settings_interval_units_index (read/write)
    - scan_button_enabled (read only)
    - scan_button_title (read only)
    - scan_abort_button_enabled (read only)
    - record_button_enabled (read only)
    - record_abort_button_enabled (read only)
    - play_state_text (read only)
    - probe_state_text (read only)
    - probe_position_enabled (read/write)
    - ac_line_sync_enabled (read/write)
    - fov_label_color (read only)
    - fov_label_tool_tip (read only)
    - channels (read only)

    The following methods are provided for modifying the model:
    - increase_width()
    - decrease_width()
    - increase_height()
    - decrease_height()
    - increase_pixel_time()
    - decrease_pixel_time()
    - increase_fov()
    - decrease_fov()
    - handle_scan_button_clicked()
    - handle_abort_button_clicked()
    - handle_record_button_clicked()
    - handle_record_abort_button_clicked()
    """

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, document_controller: DocumentController.DocumentController) -> None:
        super().__init__()
        self.__scan_hardware_source = scan_hardware_source
        self.__document_controller = document_controller
        self.__event_loop = document_controller.event_loop

        stem_controller_ = self.__scan_hardware_source

        self.__profile_changed_event_listener = self.__scan_hardware_source.scan_settings.profile_changed_event.listen(self.__update_profile_index)
        self.__property_changed_event_listener = self.__scan_hardware_source.stem_controller.property_changed_event.listen(self.__handle_property_changed)
        self.__frame_parameters_changed_event_listener = self.__scan_hardware_source.scan_settings.current_frame_parameters_changed_event.listen(self.__update_frame_parameters)
        self.__data_channel_state_changed_event_listener = self.__scan_hardware_source.data_channel_state_changed_event.listen(self.__data_channel_state_changed)
        self.__acquisition_state_changed_event_listener = self.__scan_hardware_source.acquisition_state_changed_event.listen(self.__acquisition_state_changed)
        self.__probe_state_changed_event_listener = stem_controller_.probe_state_changed_event.listen(self.__probe_state_changed)
        self.__channel_state_changed_event_listener = self.__scan_hardware_source.channel_state_changed_event.listen(self.__channel_state_changed)
        self.__scan_frame_parameters_changed_event_listener = self.__scan_hardware_source.scan_frame_parameters_changed_event.listen(self.__scan_frame_parameters_changed)
        self.__max_fov_stream_listener = self.__scan_hardware_source.max_field_of_view_nm_stream.value_stream.listen(self.__max_fov_changed)

        self.__pending_call_lock = threading.RLock()
        self.__pending_call: typing.Optional[asyncio.Handle] = None

        self.profiles_model = ListModel.ListModel[str]()
        self.profiles_model.append_item(_("Puma"))
        self.profiles_model.append_item(_("Rabbit"))
        self.profiles_model.append_item(_("Frame"))

        self.channels = [ChannelModel(scan_hardware_source, i, document_controller.document_model, self.__event_loop) for i in range(scan_hardware_source.channel_count)]

        self.__profile_index = 0
        self.__frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__profile_index)
        self.__width = 0
        self.__height = 0
        self.__width_height_linked = self.__frame_parameters.pixel_size.width == self.__frame_parameters.pixel_size.height
        self.__pixel_time_str = str()
        self.__fov_str = str()
        self.__rotation_deg_str = str()
        self.__subscan_checkbox_enabled = True
        self.__subscan_checkbox_checked = False
        self.__line_scan_checkbox_enabled = True
        self.__line_scan_checkbox_checked = False
        self.__drift_controls_enabled = False
        self.__drift_checkbox_checked = False
        self.__drift_settings_interval_str: str | None = None
        self.__drift_settings_interval_units_index = 0
        self.__scan_button_enabled = False
        self.__scan_button_title = _("Scan")
        self.__scan_abort_button_enabled = False
        self.__record_button_enabled = False
        self.__record_abort_button_enabled = False
        self.__play_state_text = map_channel_state_to_text["stopped"]
        self.__probe_state_text = _("Parked")
        self.__probe_position_enabled = False
        self.__ac_line_sync_enabled = False
        self.__fov_label_color = "black"
        self.__fov_label_tool_tip = str()

        self.__handle_state_changed_on_ui_thread()

    def close(self) -> None:
        while self.channels:
            channel = self.channels.pop()
            channel.close()
        with self.__pending_call_lock:
            if self.__pending_call is not None:
                self.__pending_call.cancel()
            self.__pending_call = None
            self.__profile_changed_event_listener = typing.cast(typing.Any, None)
            self.__property_changed_event_listener = typing.cast(typing.Any, None)
            self.__frame_parameters_changed_event_listener = typing.cast(typing.Any, None)
            self.__data_channel_state_changed_event_listener = typing.cast(typing.Any, None)
            self.__acquisition_state_changed_event_listener = typing.cast(typing.Any, None)
            self.__probe_state_changed_event_listener = typing.cast(typing.Any, None)
            self.__channel_state_changed_event_listener = typing.cast(typing.Any, None)
            self.__scan_frame_parameters_changed_event_listener = typing.cast(typing.Any, None)
            self.__max_fov_stream_listener = typing.cast(typing.Any, None)

    def __handle_state_changed_on_ui_thread(self) -> None:
        with self.__pending_call_lock:
            self.__pending_call = None
        profile_index = self.__scan_hardware_source.scan_settings.selected_profile_index
        if profile_index != self.__profile_index:
            self.__profile_index = profile_index
            self.notify_property_changed("profile_index")
        frame_parameters = self.__scan_hardware_source.get_current_frame_parameters()
        self.__frame_parameters = frame_parameters
        if self.__width != frame_parameters.pixel_size.width:
            self.__width = frame_parameters.pixel_size.width
            self.notify_property_changed("width_str")
        if self.__height != frame_parameters.pixel_size.height:
            self.__height = frame_parameters.pixel_size.height
            self.notify_property_changed("height_str")
        pixel_time_str = f"{frame_parameters.pixel_time_us:.2f}"
        if pixel_time_str != self.__pixel_time_str:
            self.__pixel_time_str = pixel_time_str
            self.notify_property_changed("pixel_time_str")
        fov_str = f"{frame_parameters.fov_nm:.1f}"
        if fov_str != self.__fov_str:
            self.__fov_str = fov_str
            self.notify_property_changed("fov_str")
        rotation_deg_str = f"{frame_parameters.rotation_rad * 180.0 / math.pi:.1f}"
        if rotation_deg_str != self.__rotation_deg_str:
            self.__rotation_deg_str = rotation_deg_str
            self.notify_property_changed("rotation_deg_str")
        ac_line_sync_enabled = frame_parameters.ac_line_sync
        if ac_line_sync_enabled != self.__ac_line_sync_enabled:
            self.__ac_line_sync_enabled = ac_line_sync_enabled
            self.notify_property_changed("ac_line_sync_enabled")
        subscan_state = self.__scan_hardware_source.stem_controller.subscan_state
        subscan_checkbox_enabled = subscan_state != stem_controller.SubscanState.INVALID
        if subscan_checkbox_enabled != self.__subscan_checkbox_enabled:
            self.__subscan_checkbox_enabled = subscan_checkbox_enabled
            self.notify_property_changed("subscan_checkbox_enabled")
        subscan_checkbox_checked = subscan_state == stem_controller.SubscanState.ENABLED
        if subscan_checkbox_checked != self.__subscan_checkbox_checked:
            self.__subscan_checkbox_checked = subscan_checkbox_checked
            self.notify_property_changed("subscan_checkbox_checked")
        line_scan_state = self.__scan_hardware_source.stem_controller.line_scan_state
        line_scan_checkbox_enabled = line_scan_state != stem_controller.LineScanState.INVALID
        if line_scan_checkbox_enabled != self.__line_scan_checkbox_enabled:
            self.__line_scan_checkbox_enabled = line_scan_checkbox_enabled
            self.notify_property_changed("line_scan_checkbox_enabled")
        line_scan_checkbox_checked = line_scan_state == stem_controller.LineScanState.ENABLED
        if line_scan_checkbox_checked != self.__line_scan_checkbox_checked:
            self.__line_scan_checkbox_checked = line_scan_checkbox_checked
            self.notify_property_changed("line_scan_checkbox_checked")
        drift_controls_enabled = subscan_state != stem_controller.SubscanState.INVALID
        if drift_controls_enabled != self.__drift_controls_enabled:
            self.__drift_controls_enabled = drift_controls_enabled
            self.notify_property_changed("drift_controls_enabled")
        drift_channel_id = self.__scan_hardware_source.stem_controller.drift_channel_id
        drift_region = self.__scan_hardware_source.stem_controller.drift_region
        drift_checkbox_checked = drift_channel_id is not None and drift_region is not None
        if drift_checkbox_checked != self.__drift_checkbox_checked:
            self.__drift_checkbox_checked = drift_checkbox_checked
            self.notify_property_changed("drift_checkbox_checked")
        drift_settings = self.__scan_hardware_source.stem_controller.drift_settings
        drift_settings_interval_str = Converter.IntegerToStringConverter().convert(drift_settings.interval)
        if drift_settings_interval_str != self.__drift_settings_interval_str:
            self.__drift_settings_interval_str = drift_settings_interval_str
            self.notify_property_changed("drift_settings_interval_str")
        drift_settings_interval_units_index = min(1, max(0, drift_settings.interval_units - 2))
        if drift_settings_interval_units_index != self.__drift_settings_interval_units_index:
            self.__drift_settings_interval_units_index = drift_settings_interval_units_index
            self.notify_property_changed("drift_settings_interval_units_index")
        channel_states = self.__scan_hardware_source.channel_states
        is_playing = self.__scan_hardware_source.is_playing
        is_recording = self.__scan_hardware_source.is_recording
        is_any_channel_enabled = any(channel_state.enabled for channel_state in channel_states)
        scan_button_enabled = is_any_channel_enabled
        if scan_button_enabled != self.__scan_button_enabled:
            self.__scan_button_enabled = scan_button_enabled
            self.notify_property_changed("scan_button_enabled")
        scan_button_title = _("Stop") if is_playing else _("Scan")
        if scan_button_title != self.__scan_button_title:
            self.__scan_button_title = scan_button_title
            self.notify_property_changed("scan_button_title")
        scan_abort_button_enabled = is_playing
        if scan_abort_button_enabled != self.__scan_abort_button_enabled:
            self.__scan_abort_button_enabled = scan_abort_button_enabled
            self.notify_property_changed("scan_abort_button_enabled")
        record_button_enabled = not is_recording and is_any_channel_enabled
        if record_button_enabled != self.__record_button_enabled:
            self.__record_button_enabled = record_button_enabled
            self.notify_property_changed("record_button_enabled")
        record_abort_button_enabled = is_recording
        if record_abort_button_enabled != self.__record_abort_button_enabled:
            self.__record_abort_button_enabled = record_abort_button_enabled
            self.notify_property_changed("record_abort_button_enabled")
        acquisition_state: str | None = None
        data_channel_states = self.__scan_hardware_source.data_channel_states
        for data_channel_state in data_channel_states:
            if data_channel_state.data_channel_state != "stopped":
                acquisition_state = data_channel_state.data_channel_state
                break
        play_state_text = map_channel_state_to_text[acquisition_state or "stopped"]
        if play_state_text != self.__play_state_text:
            self.__play_state_text = play_state_text
            self.notify_property_changed("play_state_text")
        stem_controller_ = self.__scan_hardware_source.stem_controller
        probe_state = stem_controller_.probe_state
        probe_position = stem_controller_.probe_position
        map_probe_state_to_text = {"scanning": _("Scanning"), "parked": _("Parked")}
        if probe_state != "scanning":
            if probe_position is not None:
                probe_position_str = " " + str(int(probe_position.x * 100)) + "%" + ", " + str(int(probe_position.y * 100)) + "%"
            else:
                probe_position_str = " Default"
        else:
            probe_position_str = ""
        probe_state_text = map_probe_state_to_text.get(probe_state, "") + probe_position_str
        if probe_state_text != self.__probe_state_text:
            self.__probe_state_text = probe_state_text
            self.notify_property_changed("probe_state_text")
        probe_position_enabled = self.__scan_hardware_source.stem_controller.probe_position is not None
        if probe_position_enabled != self.__probe_position_enabled:
            self.__probe_position_enabled = probe_position_enabled
            self.notify_property_changed("probe_position_enabled")
        max_fov_nm = self.__scan_hardware_source.max_field_of_view_nm_stream.value or 100000.0
        fov_nm = self.__scan_hardware_source.get_current_frame_parameters().fov_nm
        if fov_nm > max_fov_nm:
            fov_label_color = "red"
            fov_label_tool_tip = _("Exceeds maximum field of view:") + f" {int(max_fov_nm)}nm"
        elif fov_nm > max_fov_nm * 0.9:
            fov_label_color = "orange"
            fov_label_tool_tip = _("Near maximum field of view:") + f" {int(max_fov_nm)}nm"
        else:
            fov_label_color = "black"
            fov_label_tool_tip = _("Maximum field of view:") + f" {int(max_fov_nm)}nm"
        if fov_label_color != self.__fov_label_color:
            self.__fov_label_color = fov_label_color
            self.notify_property_changed("fov_label_color")
        if fov_label_tool_tip != self.__fov_label_tool_tip:
            self.__fov_label_tool_tip = fov_label_tool_tip
            self.notify_property_changed("fov_label_tool_tip")

    def __handle_state_changed(self) -> None:
        # use async to call handle_property_changed on the main thread if an existing call is not pending.
        if threading.current_thread() != threading.main_thread():
            with self.__pending_call_lock:
                if not self.__pending_call:
                    self.__pending_call = self.__event_loop.call_soon_threadsafe(self.__handle_state_changed_on_ui_thread)
        else:
            self.__handle_state_changed_on_ui_thread()

    def __update_profile_index(self, profile_index: int) -> None:
        self.__handle_state_changed()

    def __update_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters) -> None:
        self.__handle_state_changed()

    def __handle_property_changed(self, property_name: str) -> None:
        self.__handle_state_changed()

    def __data_channel_state_changed(self, data_channel_event_args: HardwareSource.DataChannelEventArgs) -> None:
        self.__handle_state_changed()

    def __acquisition_state_changed(self, acquisition_state: str) -> None:
        self.__handle_state_changed()

    def __probe_state_changed(self, probe_state: str, probe_position: Geometry.FloatPoint | None) -> None:
        self.__handle_state_changed()

    def __channel_state_changed(self, channel_index: int, channel_id: str, name: str, enabled: bool) -> None:
        self.__handle_state_changed()

    def __scan_frame_parameters_changed(self, scan_frame_parameters: scan_base.ScanFrameParameters) -> None:
        self.__handle_state_changed()

    def __max_fov_changed(self, max_fov: float) -> None:
        self.__handle_state_changed()

    @property
    def profile_index(self) -> int:
        return self.__profile_index

    @profile_index.setter
    def profile_index(self, value: int) -> None:
        self.__scan_hardware_source.set_selected_profile_index(value)

    @property
    def width_str(self) -> str:
        return str(self.__width)

    @width_str.setter
    def width_str(self, value_str: str) -> None:
        value = max(1, Converter.IntegerToStringConverter().convert_back(value_str) or 1)
        frame_parameters = copy.copy(self.__frame_parameters)
        if self.__width_height_linked:
            pixel_size = Geometry.IntSize(value, value)
        else:
            pixel_size = Geometry.IntSize(self.__frame_parameters.pixel_size.height, value)
        frame_parameters.pixel_size = pixel_size
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    @property
    def height_str(self) -> str:
        return str(self.__height)

    @height_str.setter
    def height_str(self, value_str: str) -> None:
        value = max(1, Converter.IntegerToStringConverter().convert_back(value_str) or 1)
        frame_parameters = copy.copy(self.__frame_parameters)
        if self.__width_height_linked:
            pixel_size = Geometry.IntSize(value, value)
        else:
            pixel_size = Geometry.IntSize(value, self.__frame_parameters.pixel_size.width)
        frame_parameters.pixel_size = pixel_size
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    @property
    def width_height_linked(self) -> bool:
        """Whether width and height are linked. If linking is enabled, changing one will change the other to match."""
        return self.__width_height_linked

    @width_height_linked.setter
    def width_height_linked(self, value: bool) -> None:
        self.__width_height_linked = value
        frame_parameters = copy.copy(self.__frame_parameters)
        if value:
            # if enabling linking, ensure the height matches the width immediately.
            # notifications and internal values are updated in __handle_state_changed_on_ui_thread
            pixel_size = Geometry.IntSize(self.__width, self.__width)
            frame_parameters.pixel_size = pixel_size
            self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def increase_width(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        if self.__width_height_linked:
            pixel_size = Geometry.IntSize(frame_parameters.pixel_size.height * 2, frame_parameters.pixel_size.width * 2)
        else:
            pixel_size = Geometry.IntSize(frame_parameters.pixel_size.height, frame_parameters.pixel_size.width * 2)
        frame_parameters.pixel_size = pixel_size
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def decrease_width(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        if self.__width_height_linked:
            pixel_size = Geometry.IntSize(max(1, frame_parameters.pixel_size.height // 2), max(1, frame_parameters.pixel_size.width // 2))
        else:
            pixel_size = Geometry.IntSize(frame_parameters.pixel_size.height, max(1, frame_parameters.pixel_size.width // 2))
        frame_parameters.pixel_size = pixel_size
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def increase_height(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        if self.__width_height_linked:
            pixel_size = Geometry.IntSize(frame_parameters.pixel_size.height * 2, frame_parameters.pixel_size.width * 2)
        else:
            pixel_size = Geometry.IntSize(frame_parameters.pixel_size.height * 2, frame_parameters.pixel_size.width)
        frame_parameters.pixel_size = pixel_size
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def decrease_height(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        if self.__width_height_linked:
            pixel_size = Geometry.IntSize(max(1, frame_parameters.pixel_size.height // 2), max(1, frame_parameters.pixel_size.width // 2))
        else:
            pixel_size = Geometry.IntSize(max(1, frame_parameters.pixel_size.height // 2), frame_parameters.pixel_size.width)
        frame_parameters.pixel_size = pixel_size
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    @property
    def pixel_time_str(self) -> str:
        return self.__pixel_time_str

    @pixel_time_str.setter
    def pixel_time_str(self, value_str: str) -> None:
        value = max(0.01, Converter.FloatToStringConverter().convert_back(value_str) or 0.01)
        frame_parameters = copy.copy(self.__frame_parameters)
        frame_parameters.pixel_time_us = value
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def increase_pixel_time(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        frame_parameters.pixel_time_us *= 2.0
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def decrease_pixel_time(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        frame_parameters.pixel_time_us = max(0.01, frame_parameters.pixel_time_us / 2.0)
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    @property
    def fov_str(self) -> str:
        return self.__fov_str

    @fov_str.setter
    def fov_str(self, value: str) -> None:
        fov_nm = max(1.0, Converter.FloatToStringConverter().convert_back(value) or 1.0)
        frame_parameters = copy.copy(self.__frame_parameters)
        frame_parameters.fov_nm = fov_nm
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def increase_fov(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        frame_parameters.fov_nm *= 2.0
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    def decrease_fov(self) -> None:
        frame_parameters = copy.copy(self.__frame_parameters)
        frame_parameters.fov_nm = max(1.0, frame_parameters.fov_nm / 2.0)
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    @property
    def rotation_deg_str(self) -> str:
        return self.__rotation_deg_str

    @rotation_deg_str.setter
    def rotation_deg_str(self, value: str) -> None:
        rotation_deg = Converter.FloatToStringConverter().convert_back(value) or 0.0
        frame_parameters = copy.copy(self.__frame_parameters)
        frame_parameters.rotation_rad = rotation_deg * math.pi / 180.0
        self.__scan_hardware_source.set_frame_parameters(self.__profile_index, frame_parameters)

    @property
    def subscan_checkbox_enabled(self) -> bool:
        return self.__subscan_checkbox_enabled

    @property
    def subscan_checkbox_checked(self) -> bool:
        return self.__subscan_checkbox_checked

    @subscan_checkbox_checked.setter
    def subscan_checkbox_checked(self, value: bool) -> None:
        self.__scan_hardware_source.line_scan_enabled = False
        self.__scan_hardware_source.subscan_enabled = value

    @property
    def line_scan_checkbox_enabled(self) -> bool:
        return self.__line_scan_checkbox_enabled

    @property
    def line_scan_checkbox_checked(self) -> bool:
        return self.__line_scan_checkbox_checked

    @line_scan_checkbox_checked.setter
    def line_scan_checkbox_checked(self, value: bool) -> None:
        self.__scan_hardware_source.subscan_enabled = False
        self.__scan_hardware_source.line_scan_enabled = value

    @property
    def drift_controls_enabled(self) -> bool:
        return self.__drift_controls_enabled

    @property
    def drift_checkbox_checked(self) -> bool:
        return self.__drift_checkbox_checked

    @drift_checkbox_checked.setter
    def drift_checkbox_checked(self, value: bool) -> None:
        self.__scan_hardware_source.drift_enabled = value

    @property
    def drift_settings_interval_str(self) -> str | None:
        return self.__drift_settings_interval_str

    @drift_settings_interval_str.setter
    def drift_settings_interval_str(self, value: str | None) -> None:
        if value is not None:
            interval = max(0, Converter.IntegerToStringConverter().convert_back(value) or 0)
            drift_settings = copy.copy(self.__scan_hardware_source.stem_controller.drift_settings)
            drift_settings.interval = interval
            self.__scan_hardware_source.stem_controller.drift_settings = drift_settings

    @property
    def drift_settings_interval_units_index(self) -> int:
        return self.__drift_settings_interval_units_index

    @drift_settings_interval_units_index.setter
    def drift_settings_interval_units_index(self, value: int) -> None:
        drift_settings = copy.copy(self.__scan_hardware_source.stem_controller.drift_settings)
        drift_settings.interval_units = stem_controller.DriftIntervalUnit(value + 2 if value else stem_controller.DriftIntervalUnit.LINE)
        self.__scan_hardware_source.stem_controller.drift_settings = drift_settings

    @property
    def scan_button_enabled(self) -> bool:
        return self.__scan_button_enabled

    @property
    def scan_button_title(self) -> str:
        return self.__scan_button_title

    @property
    def scan_abort_button_enabled(self) -> bool:
        return self.__scan_abort_button_enabled

    @property
    def record_button_enabled(self) -> bool:
        return self.__record_button_enabled

    @property
    def record_abort_button_enabled(self) -> bool:
        return self.__record_abort_button_enabled

    @property
    def play_state_text(self) -> str:
        return self.__play_state_text

    @property
    def probe_state_text(self) -> str:
        return self.__probe_state_text

    @property
    def probe_position_enabled(self) -> bool:
        return self.__probe_position_enabled

    @probe_position_enabled.setter
    def probe_position_enabled(self, value: bool) -> None:
        if value:
            self.__scan_hardware_source.validate_probe_position()
        else:
            self.__scan_hardware_source.probe_position = None

    @property
    def ac_line_sync_enabled(self) -> bool:
        return self.__ac_line_sync_enabled

    @ac_line_sync_enabled.setter
    def ac_line_sync_enabled(self, value: bool) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.ac_line_sync = value
        self.__scan_hardware_source.set_frame_parameters(self.__scan_hardware_source.selected_profile_index, frame_parameters)

    @property
    def fov_label_color(self) -> str:
        return self.__fov_label_color

    @property
    def fov_label_tool_tip(self) -> str:
        return self.__fov_label_tool_tip

    # must be called on ui thread
    def handle_scan_button_clicked(self) -> None:
        """ Call this when the user clicks the play/pause button. """
        if self.__scan_hardware_source.is_playing:
            action_context = self.__document_controller._get_action_context()
            action_context.parameters["hardware_source_id"] = self.__scan_hardware_source.hardware_source_id
            self.__document_controller.perform_action_in_context("acquisition.stop_playing", action_context)
        else:
            # the 'enabled channel indexes' implementation is incomplete, so, for now, explicitly add them to the
            # frame parameters so that they can be recorded when logging the start playing action.
            frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
            frame_parameters.enabled_channel_indexes = self.__scan_hardware_source.get_enabled_channel_indexes()
            action_context = self.__document_controller._get_action_context()
            action_context.parameters["hardware_source_id"] = self.__scan_hardware_source.hardware_source_id
            action_context.parameters["frame_parameters"] = frame_parameters.as_dict()
            self.__document_controller.perform_action_in_context("acquisition.start_playing", action_context)

    # must be called on ui thread
    def handle_scan_abort_button_clicked(self) -> None:
        """ Call this when the user clicks the abort button. """
        action_context = self.__document_controller._get_action_context()
        action_context.parameters["hardware_source_id"] = self.__scan_hardware_source.hardware_source_id
        self.__document_controller.perform_action_in_context("acquisition.abort_playing", action_context)

    # must be called on ui thread
    def handle_record_button_clicked(self) -> None:
        """ Call this when the user clicks the record button. """
        def finish_record(data_promise_list: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:
            record_index = self.__scan_hardware_source.record_index
            for data_promise in data_promise_list:
                data_and_metadata = data_promise.xdata
                if data_and_metadata:
                    document_controller = self.__document_controller
                    data_item = DataItem.DataItem()
                    display_name = data_and_metadata.metadata.get("hardware_source", dict()).get("hardware_source_name")
                    display_name = display_name if display_name else _("Record")
                    channel_name = data_and_metadata.metadata.get("hardware_source", dict()).get("channel_name")
                    title_base = "{} ({})".format(display_name, channel_name) if channel_name else display_name
                    data_item.title = "{} {}".format(title_base, record_index)
                    data_item.set_xdata(data_and_metadata)

                    def handle_record_data_item(data_item: DataItem.DataItem) -> None:
                        document_controller.document_model.append_data_item(data_item)
                        result_display_panel = document_controller.next_result_display_panel()
                        if result_display_panel:
                            result_display_panel.set_display_panel_data_item(data_item)
                            result_display_panel.request_focus()

                    document_controller.queue_task(functools.partial(handle_record_data_item, data_item))
            self.__scan_hardware_source.record_index += 1

        self.__scan_hardware_source.record_async(finish_record)

    # must be called on ui thread
    def handle_record_abort_clicked(self) -> None:
        """ Call this when the user clicks the abort button. """
        if self.__scan_hardware_source:
            self.__scan_hardware_source.abort_recording()


class ScanPanelController(Declarative.Handler):

    def __init__(self, document_controller: DocumentController.DocumentController, scan_hardware_source: scan_base.ScanHardwareSource) -> None:
        super().__init__()

        self.__document_controller = document_controller
        self.__scan_hardware_source = scan_hardware_source

        self._model = ScanControlPanelModel(scan_hardware_source, document_controller)

        self.__shift_click_state: str | None = None
        self.__mouse_pressed = False

        self.__key_pressed_event_listener = DisplayPanel.DisplayPanelManager().key_pressed_event.listen(self.__image_panel_key_pressed)
        self.__key_released_event_listener = DisplayPanel.DisplayPanelManager().key_released_event.listen(self.__image_panel_key_released)
        self.__image_display_mouse_pressed_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_pressed_event.listen(self.__image_panel_mouse_pressed)
        self.__image_display_mouse_released_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_released_event.listen(self.__image_panel_mouse_released)

        sliders_icon_24_png = pkgutil.get_data(__name__, "resources/sliders_icon_24.png")
        assert sliders_icon_24_png is not None
        self._config_icon = CanvasItem.load_rgba_data_from_bytes(sliders_icon_24_png, "png")

        u = Declarative.DeclarativeUI()

        @dataclasses.dataclass
        class KeyAndAction:
            key: str
            action: str

        def create_line_edit_row(label: Declarative.UILabel,
                                 text_binding: str,
                                 lower: KeyAndAction | None = None,
                                 upper: KeyAndAction | None = None,
                                 color_binding: str | None = None,
                                 tool_tip_binding: str | None = None,
                                 text_width: int | None = None) -> Declarative.UIDescription:
            return u.create_row(
                u.create_label(text=label, color=color_binding, tool_tip=tool_tip_binding, width=text_width, text_alignment_vertical="vcenter", text_alignment_horizontal="right"),
                u.create_row(
                    CharButtonFactory.create_char_button(text=lower.key, on_clicked=lower.action) if lower else u.create_spacing(CharButtonFactory.DEFAULT_WIDTH),
                    u.create_line_edit(text=text_binding, width=44),
                    CharButtonFactory.create_char_button(text=upper.key, on_clicked=upper.action) if upper else u.create_spacing(CharButtonFactory.DEFAULT_WIDTH),
                    u.create_stretch(),
                    spacing=2
                ),
                spacing=6
            )

        pixel_time_row = create_line_edit_row(_("Time (\N{MICRO SIGN}s)"), "@binding(_model.pixel_time_str)", KeyAndAction("F", "handle_decrease_time"), KeyAndAction("S", "handle_increase_time"), text_width=68)
        fov_row = create_line_edit_row(_("FoV (nm)"), "@binding(_model.fov_str)", KeyAndAction("I", "handle_decrease_fov"), KeyAndAction("O", "handle_increase_fov"), text_width=68, color_binding="@binding(_model.fov_label_color)", tool_tip_binding="@binding(_model.fov_label_tool_tip)")
        rotation_row = create_line_edit_row(_("Rot. (deg)"), "@binding(_model.rotation_deg_str)", text_width=68)
        width_row = create_line_edit_row(_("Width"), "@binding(_model.width_str)", KeyAndAction("L", "handle_decrease_width"), KeyAndAction("H", "handle_increase_width"), text_width=48)
        height_row = create_line_edit_row(_("Height"), "@binding(_model.height_str)", KeyAndAction("L", "handle_decrease_height"), KeyAndAction("H", "handle_increase_height"), text_width=48)

        size_row = u.create_row(
            u.create_column(width_row, height_row, spacing=2),
            LinkedFieldsCheckBoxFactory.create_linked_fields_check_box(checked="@binding(_model.width_height_linked)"),
            u.create_stretch()
        )

        scan_profile_row = u.create_row(
            u.create_label(text=_("Scan Profile")),
            u.create_combo_box(items_ref="@binding(_model.profiles_model.items)", current_index="@binding(_model.profile_index)"), u.create_stretch(),
            u.create_image(image="@binding(_config_icon)", width=24, height=24, on_clicked="handle_config_button"), spacing=8, margin=4
        )

        subscan_checkbox = u.create_check_box(text=_("Subscan"), checked="@binding(_model.subscan_checkbox_checked)", enabled="@binding(_model.subscan_checkbox_enabled)")
        line_scan_checkbox = u.create_check_box(text=_("Line Scan"), checked="@binding(_model.line_scan_checkbox_checked)", enabled="@binding(_model.line_scan_checkbox_enabled)")

        drift_correction_checkbox = u.create_check_box(text=_("Drift Correct Every"), checked="@binding(_model.drift_checkbox_checked)", enabled="@binding(_model.drift_controls_enabled)")
        drift_correction_edit = u.create_line_edit(width=44, text="@binding(_model.drift_settings_interval_str)", enabled="@binding(_model.drift_controls_enabled)")
        drift_correction_units = u.create_combo_box(items=[_("Scan Lines"), _("Scan Frames")], current_index="@binding(_model.drift_settings_interval_units_index)", enabled="@binding(_model.drift_controls_enabled)")

        play_button = u.create_push_button(text="@binding(_model.scan_button_title)", enabled="@binding(_model.scan_button_enabled)", on_clicked="handle_scan_button")
        abort_button = u.create_push_button(text=_("Abort"), enabled="@binding(_model.scan_abort_button_enabled)", on_clicked="handle_scan_abort_button")
        play_state_label = u.create_label(text="@binding(_model.play_state_text)")
        record_button = u.create_push_button(text=_("Record"), enabled="@binding(_model.record_button_enabled)", on_clicked="handle_record_button")
        record_abort_button = u.create_push_button(text=_("Abort"), enabled="@binding(_model.record_abort_button_enabled)", on_clicked="handle_record_abort_button")

        probe_state_label = u.create_label(text="@binding(_model.probe_state_text)")

        positioned_check_box = u.create_check_box(text=_("Positioned"), checked="@binding(_model.probe_position_enabled)")
        ac_line_sync_check_box = u.create_check_box(text=_("AC Line Sync"), checked="@binding(_model.ac_line_sync_enabled)")

        self.display_item: DisplayItem.DisplayItem | None = document_controller.document_model.display_items[10] if len(document_controller.document_model.display_items) > 10 else None

        self.ui_view = u.create_column(
            scan_profile_row,
            # region_row,
            # region2_row,
            u.create_row(
                u.create_column(pixel_time_row, fov_row, rotation_row, u.create_stretch(), spacing=4),
                u.create_column(
                    u.create_column(size_row, subscan_checkbox, line_scan_checkbox, u.create_stretch(), spacing=4),
                    u.create_stretch()
                ),
                spacing=8,
                margin_horizontal=4
            ),
            u.create_spacing(4),
            u.create_row(
                drift_correction_checkbox,
                u.create_spacing(8),
                drift_correction_edit,
                drift_correction_units,
                u.create_stretch(),
                spacing=4,
                margin_horizontal=8
            ),
            u.create_spacing(4),
            u.create_row(
                play_button,
                abort_button,
                u.create_stretch(),
                play_state_label,
                spacing=6,
                margin_horizontal=8
            ),
            u.create_row(
                record_button,
                record_abort_button,
                u.create_stretch(),
                spacing=6,
                margin_horizontal=8
            ),
            u.create_spacing(4),
            u.create_row(u.create_stretch(), u.create_row(items="_model.channels", item_component_id="channel-component", spacing=4), u.create_stretch(), margin_horizontal=8),
            u.create_spacing(4),
            u.create_row(probe_state_label, u.create_stretch(), margin_horizontal=8),
            u.create_spacing(4),
            u.create_row(positioned_check_box, u.create_stretch(), ac_line_sync_check_box, margin_horizontal=8),
            u.create_stretch(),
            margin=2
        )

    def close(self) -> None:
        self._model.close()
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> Declarative.HandlerLike | None:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "channel-component":
            assert container is not None
            assert item is not None
            return ChannelHandler(typing.cast(ChannelModel, item), self.__document_controller, self.__scan_hardware_source)
        return None

    def handle_config_button(self, widget: UserInterface.Widget) -> None:
        self.__scan_hardware_source.scan_settings.open_configuration_interface(PlugInManager.APIBroker())

    def handle_increase_time(self, widget: UserInterface.Widget) -> None:
        self._model.increase_pixel_time()

    def handle_decrease_time(self, widget: UserInterface.Widget) -> None:
        self._model.decrease_pixel_time()

    def handle_increase_fov(self, widget: UserInterface.Widget) -> None:
        self._model.increase_fov()

    def handle_decrease_fov(self, widget: UserInterface.Widget) -> None:
        self._model.decrease_fov()

    def handle_increase_width(self, widget: UserInterface.Widget) -> None:
        self._model.increase_width()

    def handle_decrease_width(self, widget: UserInterface.Widget) -> None:
        self._model.decrease_width()

    def handle_increase_height(self, widget: UserInterface.Widget) -> None:
        self._model.increase_height()

    def handle_decrease_height(self, widget: UserInterface.Widget) -> None:
        self._model.decrease_height()

    def handle_scan_button(self, widget: UserInterface.Widget) -> None:
        self._model.handle_scan_button_clicked()

    def handle_scan_abort_button(self, widget: UserInterface.Widget) -> None:
        self._model.handle_scan_abort_button_clicked()

    def handle_record_button(self, widget: UserInterface.Widget) -> None:
        self._model.handle_record_button_clicked()

    def handle_record_abort_button(self, widget: UserInterface.Widget) -> None:
        self._model.handle_record_abort_clicked()

    # this gets called from the DisplayPanelManager. pass on the message to the state controller.
    # must be called on ui thread
    def __image_panel_mouse_pressed(self, display_panel: DisplayPanel.DisplayPanel, display_item: DisplayItem.DisplayItem, image_position: Geometry.FloatPoint, modifiers: CanvasItem.KeyboardModifiers) -> bool:
        data_item = display_panel.data_item if display_panel else None
        hardware_source_id = data_item.metadata.get("hardware_source", dict()).get("hardware_source_id") if data_item else str()
        logger = logging.getLogger("camera_control_ui")
        logger.propagate = False  # do not send messages to root logger
        if not logger.handlers:
            logger.addHandler(logging.handlers.BufferingHandler(4))
        data_metadata = data_item.data_metadata if data_item else None
        camera_shape = data_metadata.dimensional_shape if data_metadata else None
        if hardware_source_id == self.__scan_hardware_source.hardware_source_id and camera_shape and len(camera_shape) == 2 and self.__shift_click_state == "shift":
            mouse_position = image_position
            self.__scan_hardware_source.shift_click(mouse_position, typing.cast(DataAndMetadata.Shape2dType, camera_shape), logger)
            logger_buffer = typing.cast(logging.handlers.BufferingHandler, logger.handlers[0])
            for record in logger_buffer.buffer:
                display_panel.document_controller.display_log_record(record)
            logger_buffer.flush()
            self.__shift_click_state = None
            self.__mouse_pressed = True
            return self.__mouse_pressed
        return False

    def __image_panel_mouse_released(self, display_panel: DisplayPanel.DisplayPanel, display_item: DisplayItem.DisplayItem, image_position: Geometry.FloatPoint, modifiers: CanvasItem.KeyboardModifiers) -> bool:
        mouse_pressed = self.__mouse_pressed
        self.__mouse_pressed = False
        return mouse_pressed

    def __image_panel_key_pressed(self, display_panel: DisplayPanel.DisplayPanel, key: UserInterface.Key) -> bool:
        if key.text.lower() == "s":
            self.__shift_click_state = "shift"
        else:
            self.__shift_click_state = None
        return False

    def __image_panel_key_released(self, display_panel: DisplayPanel.DisplayPanel, key: UserInterface.Key) -> bool:
        self.__shift_click_state = None
        return False


class ScanControlPanel(Panel.Panel):

    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> None:
        super().__init__(document_controller, panel_id, "scan-control-panel")
        ui = document_controller.ui
        self.__column_widget = ui.create_column_widget()
        self.widget = self.__column_widget
        self.__hardware_source_id = properties["hardware_source_id"]
        # listen for any hardware added or removed messages, and refresh the list
        self.__build_widget()
        HardwareSource.HardwareSourceManager().aliases_updated.append(self.__build_widget)

    def __build_widget(self) -> None:
        scan_hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(self.__hardware_source_id)
        if isinstance(scan_hardware_source, scan_base.ScanHardwareSource):
            document_controller = self.document_controller
            scan_panel_controller = ScanPanelController(document_controller, scan_hardware_source)
            self.__column_widget.add(Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, scan_panel_controller))
            self.__column_widget.add_stretch()

    def close(self) -> None:
        HardwareSource.HardwareSourceManager().aliases_updated.remove(self.__build_widget)
        super().close()


class ScanDisplayPanelController:
    """
        Represents a controller for the content of an image panel.
    """

    type = "scan-live"

    def __init__(self, display_panel: DisplayPanel.DisplayPanel, hardware_source_id: str, data_channel_id: str) -> None:
        assert hardware_source_id is not None
        hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
        assert isinstance(hardware_source, scan_base.ScanHardwareSource)
        self.type = ScanDisplayPanelController.type

        self.__thread_helper = ThreadHelper(display_panel.document_controller.event_loop)

        self.__hardware_source_id = hardware_source_id
        self.__data_channel_id = data_channel_id

        # configure the user interface
        self.__display_name = str()
        self.__channel_index = 0
        self.__channel_name = str()
        self.__channel_enabled = False
        self.__scan_button_enabled = False
        self.__scan_button_play_button_state = "scan"
        self.__abort_button_visible = False
        self.__abort_button_enabled = False
        self.__display_panel = display_panel
        self.__display_panel.header_canvas_item.end_header_color = "#98FB98"
        self.__playback_controls_composition = CanvasItem.CanvasItemComposition()
        self.__playback_controls_composition.layout = CanvasItem.CanvasItemLayout()
        self.__playback_controls_composition.update_sizing(self.__playback_controls_composition.sizing.with_fixed_height(30))
        playback_controls_row = CanvasItem.CanvasItemComposition()
        playback_controls_row.layout = CanvasItem.CanvasItemRowLayout()
        scan_button_canvas_item = CanvasItem.TextButtonCanvasItem()
        scan_button_canvas_item.border_enabled = False
        abort_button_canvas_item = CanvasItem.TextButtonCanvasItem()
        abort_button_canvas_item.border_enabled = False
        status_text_canvas_item = CanvasItem.StaticTextCanvasItem(str())
        hardware_source_display_name_canvas_item = CanvasItem.StaticTextCanvasItem(str())
        playback_controls_row.add_canvas_item(scan_button_canvas_item)
        playback_controls_row.add_canvas_item(abort_button_canvas_item)
        playback_controls_row.add_canvas_item(status_text_canvas_item)
        playback_controls_row.add_stretch()
        capture_button = CanvasItem.TextButtonCanvasItem(_("Capture"))
        capture_button.border_enabled = False
        playback_controls_row.add_canvas_item(capture_button)
        pmt_group = CanvasItem.CanvasItemComposition()
        pmt_group.layout = CanvasItem.CanvasItemRowLayout()
        decrease_pmt_button = IconCanvasItem("minus")
        decrease_pmt_button.fill_style = "#FFF"
        decrease_pmt_button.fill_style_pressed = "rgb(128, 128, 128)"
        decrease_pmt_button.border_style = "#000"
        decrease_pmt_button.border_style_pressed = "#000"
        decrease_pmt_button.stroke_style = "#000"
        decrease_pmt_button.stroke_width = 1.5
        pmt_group.add_canvas_item(decrease_pmt_button)
        pmt_label = CanvasItem.StaticTextCanvasItem(_("PMT"))
        pmt_label.size_to_content(display_panel.image_panel_get_font_metrics)
        pmt_group.add_canvas_item(pmt_label)
        increase_pmt_button = IconCanvasItem("plus")
        increase_pmt_button.fill_style = "#FFF"
        increase_pmt_button.fill_style_pressed = "rgb(128, 128, 128)"
        increase_pmt_button.border_style = "#000"
        increase_pmt_button.border_style_pressed = "#000"
        increase_pmt_button.stroke_style = "#000"
        increase_pmt_button.stroke_width = 1.5
        pmt_group.add_canvas_item(increase_pmt_button)
        channel_enabled_check_box = CanvasItem.CheckBoxCanvasItem()
        playback_controls_row.add_canvas_item(pmt_group)
        playback_controls_row.add_canvas_item(channel_enabled_check_box)
        playback_controls_row.add_canvas_item(hardware_source_display_name_canvas_item)
        self.__playback_controls_composition.add_canvas_item(CanvasItem.BackgroundCanvasItem("#98FB98"))
        self.__playback_controls_composition.add_canvas_item(playback_controls_row)
        self.__display_panel.footer_canvas_item.insert_canvas_item(0, self.__playback_controls_composition)

        # configure the hardware source state controller
        self.__state_controller = ScanControlStateController(hardware_source, display_panel.document_controller, data_channel_id)

        def update_display_name() -> None:
            new_text = "%s (%s)" % (self.__display_name, self.__channel_name)
            if hardware_source_display_name_canvas_item.text != new_text:
                hardware_source_display_name_canvas_item.text = new_text
                hardware_source_display_name_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_play_button() -> None:
            map_play_button_state_to_text = {"scan": _("Scan"), "stop": _("Stop")}
            scan_button_text = map_play_button_state_to_text[self.__scan_button_play_button_state]
            new_enabled = self.__channel_enabled and self.__scan_button_enabled
            new_text = scan_button_text if self.__channel_enabled else str()
            if scan_button_canvas_item.enabled != new_enabled or scan_button_canvas_item.text != new_text:
                scan_button_canvas_item.enabled = new_enabled
                scan_button_canvas_item.text = new_text
                scan_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_abort_button() -> None:
            abort_button_visible = self.__channel_enabled and self.__abort_button_visible
            abort_button_enabled = self.__channel_enabled and self.__abort_button_enabled
            new_text = _("Abort") if abort_button_visible else str()
            if abort_button_canvas_item.enabled != abort_button_enabled or abort_button_canvas_item.text != new_text:
                abort_button_canvas_item.text = new_text
                abort_button_canvas_item.enabled = abort_button_enabled
                abort_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_channel_enabled_check_box() -> None:
            channel_enabled_check_box.check_state = "checked" if self.__channel_enabled else "unchecked"

        def display_name_changed(display_name: str) -> None:
            self.__display_name = display_name
            update_display_name()

        def scan_button_state_changed(enabled: bool, play_button_state: str) -> None:
            self.__scan_button_enabled = enabled
            self.__scan_button_play_button_state = play_button_state
            update_play_button()

        def abort_button_state_changed(visible: bool, enabled: bool) -> None:
            self.__abort_button_visible = visible
            self.__abort_button_enabled = enabled
            update_abort_button()

        def acquisition_state_changed(key: str) -> None:
            # this may be called on a thread. ensure it runs on the main thread.
            def update_acquisition_state_label(acquisition_states: typing.Mapping[str, typing.Optional[str]]) -> None:
                acquisition_state = acquisition_states.get(self.__data_channel_id, None) or "stopped"
                status_text_canvas_item.text = map_channel_state_to_text[acquisition_state]
                status_text_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

            acquisition_states = self.__state_controller.acquisition_state_model.value or dict()
            self.__thread_helper.call_on_main_thread("acquisition_state_changed", functools.partial(update_acquisition_state_label, acquisition_states))

        def data_channel_state_changed(data_channel_index: int, data_channel_id: str, channel_name: str, enabled: bool) -> None:
            if data_channel_id == self.__data_channel_id:
                assert isinstance(hardware_source, scan_base.ScanHardwareSource)
                self.__channel_index = hardware_source.get_channel_index_for_data_channel_index(data_channel_index)
                self.__channel_name = channel_name
                self.__channel_enabled = enabled
                update_display_name()
                update_play_button()
                update_abort_button()
                update_channel_enabled_check_box()

        def update_capture_button(visible: bool, enabled: bool) -> None:
            if visible:
                capture_button.enabled = enabled
                capture_button.text = _("Capture")
                capture_button.size_to_content(display_panel.image_panel_get_font_metrics)
            else:
                capture_button.enabled = False
                capture_button.text = str()
                capture_button.size_to_content(display_panel.image_panel_get_font_metrics)

        def display_new_data_item(data_item: DataItem.DataItem) -> None:
            document_controller = display_panel.document_controller
            document_controller.document_model.append_data_item(data_item)
            result_display_panel = document_controller.next_result_display_panel()
            if result_display_panel:
                result_display_panel.set_display_panel_data_item(data_item)
                result_display_panel.request_focus()

        self.__state_controller.on_display_name_changed = display_name_changed
        self.__state_controller.on_scan_button_state_changed = scan_button_state_changed
        self.__state_controller.on_abort_button_state_changed = abort_button_state_changed
        self.__state_controller.on_data_channel_state_changed = data_channel_state_changed
        self.__state_controller.on_capture_button_state_changed = update_capture_button
        self.__state_controller.on_display_new_data_item = display_new_data_item

        display_panel.set_data_item_reference(self.__state_controller.data_item_reference)

        def channel_enabled_check_box_check_state_changed(check_state: str) -> None:
            self.__state_controller.handle_enable_channel(self.__channel_index, check_state == "checked")

        scan_button_canvas_item.on_button_clicked = self.__state_controller.handle_play_pause_clicked
        abort_button_canvas_item.on_button_clicked = self.__state_controller.handle_abort_clicked
        channel_enabled_check_box.on_check_state_changed = channel_enabled_check_box_check_state_changed
        capture_button.on_button_clicked = self.__state_controller.handle_capture_clicked

        self.__acquisition_state_changed_listener = self.__state_controller.acquisition_state_model.property_changed_event.listen(acquisition_state_changed)

        self.__state_controller.initialize_state()

        acquisition_state_changed("value")

        # put these after initialize state so that channel index is initialized.
        decrease_pmt_button.on_button_clicked = functools.partial(self.__state_controller.handle_decrease_pmt_clicked, self.__channel_index)
        increase_pmt_button.on_button_clicked = functools.partial(self.__state_controller.handle_increase_pmt_clicked, self.__channel_index)

    def close(self) -> None:
        self.__thread_helper.close()
        self.__thread_helper = typing.cast(typing.Any, None)
        self.__display_panel.footer_canvas_item.remove_canvas_item(self.__playback_controls_composition)
        self.__display_panel = typing.cast(typing.Any, None)
        self.__acquisition_state_changed_listener = typing.cast(typing.Any, None)
        self.__state_controller.close()
        self.__state_controller = typing.cast(typing.Any, None)

    def save(self, d: typing.MutableMapping[str, typing.Any]) -> None:
        d["hardware_source_id"] = self.__hardware_source_id
        if self.__data_channel_id is not None:
            d["channel_id"] = self.__data_channel_id

    def key_pressed(self, key: UserInterface.Key) -> bool:
        if key.text == " ":
            self.__state_controller.handle_play_pause_clicked()
            return True
        elif key.key == 0x1000000:  # escape
            self.__state_controller.handle_abort_clicked()
            return True
        return False

    def key_released(self, key: UserInterface.Key) -> bool:
        return False

    @property
    def channel_id(self) -> str:
        return self.__data_channel_id

    @property
    def hardware_source_id(self) -> str:
        return self.__hardware_source_id


class DriftScanPreferencesPanel:
    """Define a drift scan preferences panel.

    This panel allows the user to change the size and dwell time of the drift scan used in
    DriftTracker.DriftCorrectionBehavior
    """

    def __init__(self) -> None:
        self.identifier = "nion.drift-scan-customization"
        self.label = _("Drift scan settings")

    def build(self, ui: UserInterface.UserInterface, event_loop: typing.Optional[asyncio.AbstractEventLoop] = None, **kwargs: typing.Any) -> Declarative.DeclarativeWidget:
        u = Declarative.DeclarativeUI()

        class Handler(Declarative.Handler):
            def __init__(self, drift_frame_parameters: Schema.EntityType) -> None:
                super().__init__()
                self.drift_frame_parameters = drift_frame_parameters
                self.scan_width_converter = Converter.IntegerToStringConverter()
                self.dwell_time_converter = Converter.PhysicalValueToStringConverter("us", format="{:.1f}")
                self.ui_view = u.create_column(
                    u.create_row(u.create_label(text="Drift scan width (pixels)"), u.create_line_edit(text="@binding(drift_frame_parameters.scan_width_pixels, converter=scan_width_converter)", width=40), u.create_stretch(), spacing=8),
                    u.create_row(u.create_label(text="Drift scan dwell time"), u.create_line_edit(text="@binding(drift_frame_parameters.dwell_time_us, converter=dwell_time_converter)", width=60), u.create_stretch(), spacing=8),
                    u.create_row(u.create_push_button(text="Restore defaults", on_clicked="restore_defaults_clicked"), u.create_stretch(), spacing=8),
                    u.create_stretch(),
                    spacing=8)

            def restore_defaults_clicked(self, widget: Declarative.UIWidget) -> None:
                for key, value in AcquisitionPreferences.default_drift_frame_parameters.items():
                    setattr(self.drift_frame_parameters, key, value)

        return Declarative.DeclarativeWidget(ui, event_loop or asyncio.get_event_loop(), Handler(getattr(AcquisitionPreferences.acquisition_preferences, "drift_scan_customization")))


def create_scan_panel(document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> Panel.Panel:
    """Create a custom scan panel.

    The panel type is specified in the 'panel_type' key in the properties dict.

    The panel type must match the 'panel_type' of a 'scan_panel' factory in the Registry.

    The matching panel factory must return a ui_handler for the panel which is used to produce the UI.
    """
    panel_type = properties.get("panel_type")
    for component in Registry.get_components_by_type("scan_panel"):
        if component.panel_type == panel_type:
            hardware_source_id = properties["hardware_source_id"]
            hardware_source = typing.cast(scan_base.ScanHardwareSource, HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id))
            scan_device = hardware_source.scan_device
            scan_settings = hardware_source.scan_settings
            ui_handler = component.get_ui_handler(api_broker=PlugInManager.APIBroker(), event_loop=document_controller.event_loop, hardware_source_id=hardware_source_id, scan_device=scan_device, scan_settings=scan_settings)
            panel = Panel.Panel(document_controller, panel_id, "scan-control-panel")
            panel.widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, ui_handler)
            return panel
    raise Exception(f"Unable to create scan panel: {panel_id}")


# Register the preferences panel.
PreferencesDialog.PreferencesManager().register_preference_pane(DriftScanPreferencesPanel())

hardware_source_added_event_listener: typing.Optional[Event.EventListener] = None
hardware_source_removed_event_listener: typing.Optional[Event.EventListener] = None
scan_control_panels = dict()


def register_scan_panel(hardware_source: HardwareSource.HardwareSource) -> None:
    # NOTE: if scan control panel is not appearing, stop here and make sure aliases.ini is present in the workspace
    if hardware_source.features.get("is_scanning", False):
        panel_id = "scan-control-panel-" + hardware_source.hardware_source_id
        scan_control_panels[hardware_source.hardware_source_id] = panel_id

        class ScanDisplayPanelControllerFactory:
            def __init__(self) -> None:
                self.priority = 2

            def build_menu(self, display_type_menu: UserInterface.Menu, selected_display_panel: typing.Optional[DisplayPanel.DisplayPanel]) -> typing.Sequence[UserInterface.MenuAction]:
                # return a list of actions that have been added to the menu.
                assert isinstance(hardware_source, scan_base.ScanHardwareSource)
                maybe_view_data_channel_specifiers = hardware_source.get_view_data_channel_specifiers()
                view_data_channel_specifiers: typing.List[scan_base.ScanDataChannelSpecifier]
                if maybe_view_data_channel_specifiers:
                    view_data_channel_specifiers = list(maybe_view_data_channel_specifiers)
                else:
                    view_data_channel_specifiers = list()
                    for channel_index in range(hardware_source.channel_count):
                        channel_id_ = hardware_source.get_channel_id(channel_index) or str()
                        channel_name_ = hardware_source.get_channel_name(channel_index) or str()
                        view_data_channel_specifiers.append(scan_base.ScanDataChannelSpecifier(channel_id_, None, channel_name_))
                    for channel_index in range(hardware_source.channel_count):
                        channel_id_ = hardware_source.get_channel_id(channel_index) or str()
                        channel_name_ = hardware_source.get_channel_name(channel_index) or str()
                        view_data_channel_specifiers.append(scan_base.ScanDataChannelSpecifier(channel_id_, "subscan", channel_name_ + " " + _("Subscan")))
                    view_data_channel_specifiers.append(scan_base.ScanDataChannelSpecifier("drift", None, _("Drift")))
                actions = list()
                for view_data_channel_specifier in view_data_channel_specifiers:
                    def switch_to_live_controller(hardware_source: scan_base.ScanHardwareSource, channel_id: str) -> None:
                        if selected_display_panel:
                            d = {"type": "image", "controller_type": ScanDisplayPanelController.type,
                                 "hardware_source_id": hardware_source.hardware_source_id, "channel_id": channel_id}
                            selected_display_panel.change_display_panel_content(d)

                    channel_name = view_data_channel_specifier.channel_name or _("Scan Data")
                    display_name = f"{hardware_source.display_name} ({channel_name})"
                    assert view_data_channel_specifier.channel_id
                    data_channel_id = view_data_channel_specifier.channel_id + ("_" + view_data_channel_specifier.channel_variant if view_data_channel_specifier.channel_variant else "")
                    action = display_type_menu.add_menu_item(display_name, functools.partial(switch_to_live_controller, hardware_source, data_channel_id))
                    display_panel_controller = selected_display_panel.display_panel_controller if selected_display_panel else None
                    action.checked = isinstance(display_panel_controller, ScanDisplayPanelController) and display_panel_controller.channel_id == data_channel_id and display_panel_controller.hardware_source_id == hardware_source.hardware_source_id
                    actions.append(action)
                return actions

            def make_new(self, controller_type: str, display_panel: DisplayPanel.DisplayPanel, d: Persistence.PersistentDictType) -> typing.Optional[ScanDisplayPanelController]:
                # make a new display panel controller, typically called to restore contents of a display panel.
                # controller_type will match the type property of the display panel controller when it was saved.
                # d is the dictionary that is saved when the display panel controller closes.
                hardware_source_id = typing.cast(str, d.get("hardware_source_id"))
                channel_id = typing.cast(str, d.get("channel_id"))
                if controller_type == ScanDisplayPanelController.type and hardware_source_id == hardware_source.hardware_source_id:
                    return ScanDisplayPanelController(display_panel, hardware_source_id, channel_id)
                return None

            def match(self, document_model: DocumentModel.DocumentModel, data_item: DataItem.DataItem) -> typing.Optional[Persistence.PersistentDictType]:
                # determine whether the given data item represents a live view item that could be controlled by this display
                # panel controller. if so, return a dictionary that can be used to restore the display panel controller.
                # checks for the scan device channel and whether it is a subscan or drift channel.
                assert isinstance(hardware_source, scan_base.ScanHardwareSource)
                for channel_index in range(hardware_source.channel_count):
                    channel_id_ = hardware_source.get_channel_id(channel_index) or str()
                    for channel_id in (channel_id_, channel_id_ + "_subscan", "drift"):
                        if HardwareSource.matches_hardware_source(hardware_source.hardware_source_id, channel_id, document_model, data_item):
                            if frame_parameters := hardware_source.get_frame_parameters_from_metadata(data_item.metadata):
                                if frame_parameters.subscan_pixel_size and channel_id != "drift":
                                    channel_id = channel_id_ + "_subscan"
                            return {"controller_type": ScanDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id, "channel_id": channel_id}
                return None

        factory_id = "scan-live-" + hardware_source.hardware_source_id
        DisplayPanel.DisplayPanelManager().register_display_panel_controller_factory(factory_id, ScanDisplayPanelControllerFactory())

        name = hardware_source.display_name + " " + _("Scan Control")
        panel_properties = {"hardware_source_id": hardware_source.hardware_source_id}

        panel_type = hardware_source.features.get("panel_type")
        if not panel_type:
            Workspace.WorkspaceManager().register_panel(ScanControlPanel, panel_id, name, ["left", "right"], "left", panel_properties)
        else:
            panel_properties["panel_type"] = panel_type
            Workspace.WorkspaceManager().register_panel(typing.cast(typing.Type[typing.Any], create_scan_panel), panel_id, name, ["left", "right"], "left", panel_properties)


def unregister_scan_panel(hardware_source: HardwareSource.HardwareSource) -> None:
    if hardware_source.features.get("is_scanning", False):
        factory_id = "scan-live-" + hardware_source.hardware_source_id
        DisplayPanel.DisplayPanelManager().unregister_display_panel_controller_factory(factory_id)
        panel_id = scan_control_panels.pop(hardware_source.hardware_source_id)
        if panel_id:
            Workspace.WorkspaceManager().unregister_panel(panel_id)


def run() -> None:
    global hardware_source_added_event_listener, hardware_source_removed_event_listener, scan_control_panels
    hardware_source_added_event_listener = HardwareSource.HardwareSourceManager().hardware_source_added_event.listen(register_scan_panel)
    hardware_source_removed_event_listener = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(unregister_scan_panel)
    for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
        register_scan_panel(hardware_source)


def stop() -> None:
    global hardware_source_added_event_listener, hardware_source_removed_event_listener, scan_control_panels
    if hardware_source_added_event_listener:
        hardware_source_added_event_listener.close()
        hardware_source_added_event_listener = None
    if hardware_source_removed_event_listener:
        hardware_source_removed_event_listener.close()
        hardware_source_removed_event_listener = None
    for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
        unregister_scan_panel(hardware_source)
