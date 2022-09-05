from __future__ import annotations

# standard libraries
import asyncio
import copy
import functools
import gettext
import logging
import logging.handlers
import math
import pkgutil
import sys
import typing

# third party libraries
# None

# local libraries
from nion.data import DataAndMetadata
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation import AcquisitionPreferences
from nion.swift import DataItemThumbnailWidget
from nion.swift import DisplayPanel
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem
from nion.swift.model import PlugInManager
from nion.ui import CanvasItem
from nion.ui import MouseTrackingCanvasItem
from nion.ui import UserInterface
from nion.ui import Declarative
from nion.ui import Widgets
from nion.ui import PreferencesDialog
from nion.utils import Converter
from nion.utils import Geometry
from nion.utils import Model

if typing.TYPE_CHECKING:
    from nion.swift import DocumentController
    from nion.swift.model import DisplayItem
    from nion.swift.model import DocumentModel
    from nion.swift.model import Persistence
    from nion.ui import DrawingContext
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
        (read-only property) use_hardware_simulator: return True to use hardware simulator
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

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, queue_task: typing.Callable[[typing.Callable[[], None]], None], document_model: DocumentModel.DocumentModel, channel_id: typing.Optional[str]) -> None:
        self.__scan_hardware_source = scan_hardware_source
        self.queue_task = queue_task
        self.__document_model = document_model
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
        self.__scan_context_changed_listener: typing.Optional[Event.EventListener] = None
        self.__data_channel_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.on_display_name_changed : typing.Optional[typing.Callable[[str], None]] = None
        self.on_subscan_state_changed : typing.Optional[typing.Callable[[stem_controller.SubscanState, stem_controller.LineScanState], None]] = None
        self.on_drift_state_changed : typing.Optional[typing.Callable[[typing.Optional[str], typing.Optional[Geometry.FloatRect], stem_controller.DriftCorrectionSettings, stem_controller.SubscanState], None]] = None
        self.on_profiles_changed : typing.Optional[typing.Callable[[typing.Sequence[str]], None]] = None
        self.on_profile_changed : typing.Optional[typing.Callable[[str], None]] = None
        self.on_frame_parameters_changed : typing.Optional[typing.Callable[[scan_base.ScanFrameParameters], None]] = None
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
        if self.__scan_context_changed_listener:
            self.__scan_context_changed_listener.close()
            self.__scan_context_changed_listener = None
        self.on_display_name_changed = None
        self.on_subscan_state_changed = None
        self.on_drift_state_changed = None
        self.on_profiles_changed = None
        self.on_profile_changed = None
        self.on_frame_parameters_changed = None
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

            self.__profile_changed_event_listener = self.__scan_hardware_source.profile_changed_event.listen(self.__update_profile_index)
            self.__frame_parameters_changed_event_listener = self.__scan_hardware_source.frame_parameters_changed_event.listen(self.__update_frame_parameters)
            self.__data_channel_state_changed_event_listener = self.__scan_hardware_source.data_channel_state_changed_event.listen(self.__data_channel_state_changed)
            self.__acquisition_state_changed_event_listener = self.__scan_hardware_source.acquisition_state_changed_event.listen(self.__acquisition_state_changed)
            self.__probe_state_changed_event_listener = self.__scan_hardware_source.probe_state_changed_event.listen(self.__probe_state_changed)
            self.__channel_state_changed_event_listener = self.__scan_hardware_source.channel_state_changed_event.listen(self.__channel_state_changed)

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

            def scan_context_changed() -> None:
                # when the scan context changes, if it is no longer valid, turn off probe position
                if not stem_controller.scan_context.is_valid:
                    self.__scan_hardware_source.probe_position = None

            self.__scan_context_changed_listener = stem_controller.scan_context_changed_event.listen(scan_context_changed)

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
                self.__scan_hardware_source.stop_playing()
            else:
                self.__scan_hardware_source.start_playing()

    # must be called on ui thread
    def handle_abort_clicked(self) -> None:
        """ Call this when the user clicks the abort button. """
        if self.__scan_hardware_source:
            self.__scan_hardware_source.abort_playing()

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
        size = frame_parameters.size
        if self.__linked:
            if field == "width":
                size = Geometry.IntSize(size.width, size.width)
            else:
                size = Geometry.IntSize(size.height, size.height)
        frame_parameters.size = size
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
        frame_parameters.size = Geometry.IntSize(int(frame_parameters.size[0]), int(width_str))
        self.__update_frame_size(frame_parameters, "width")

    def handle_decrease_width(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.size = Geometry.IntSize(int(frame_parameters.size[0]), int(frame_parameters.size[1]/2))
        self.__update_frame_size(frame_parameters, "width")

    def handle_increase_width(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.size = Geometry.IntSize(int(frame_parameters.size[0]), int(frame_parameters.size[1]*2))
        self.__update_frame_size(frame_parameters, "width")

    def handle_height_changed(self, height_str: str) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.size = Geometry.IntSize(int(height_str), int(frame_parameters.size[1]))
        self.__update_frame_size(frame_parameters, "height")

    def handle_decrease_height(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.size = Geometry.IntSize(int(frame_parameters.size[0]/2), int(frame_parameters.size[1]))
        self.__update_frame_size(frame_parameters, "height")

    def handle_increase_height(self) -> None:
        frame_parameters = self.__scan_hardware_source.get_frame_parameters(self.__scan_hardware_source.selected_profile_index)
        frame_parameters.size = Geometry.IntSize(int(frame_parameters.size[0]*2), int(frame_parameters.size[1]))
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
                    data_item.title = "%s (%s)" % (display_name, channel_name) if channel_name else display_name
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

    @property
    def data_channels(self) -> typing.Sequence[HardwareSource.DataChannel]:
        return self.__scan_hardware_source.data_channels

    def get_channel_enabled(self, channel_index: int) -> bool:
        return self.__scan_hardware_source.get_channel_enabled(channel_index)

    # this message comes from the data buffer. it will always be invoked on a thread.
    def __acquisition_state_changed(self, is_playing: bool) -> None:
        if self.__captured_xdatas_available_listener:
            self.__captured_xdatas_available_listener.close()
            self.__captured_xdatas_available_listener = None
        self.queue_task(self.__update_buttons)

    def __data_channel_state_changed(self, data_channel: HardwareSource.DataChannel) -> None:
        # the value (dict) does not get copied; so copy it here.
        acquisition_states = copy.deepcopy(self.acquisition_state_model.value) or dict()
        channel_id = data_channel.channel_id or "unknown"
        if data_channel.is_started and data_channel.state:
            acquisition_states[channel_id] = data_channel.state
        else:
            acquisition_states[channel_id] = "error" if data_channel.is_error else "stopped"
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
            subscan_channel_index, subscan_channel_id, subscan_channel_name = self.__scan_hardware_source.get_subscan_channel_info(channel_index, channel_id, name)
            data_channel_state_changed(subscan_channel_index, subscan_channel_id, subscan_channel_name, enabled)
        was_any_channel_enabled = any(self.__channel_enabled)
        self.__channel_enabled[channel_index] = enabled
        is_any_channel_enabled = any(self.__channel_enabled)
        if was_any_channel_enabled != is_any_channel_enabled:
            self.__update_scan_button_state()
            self.__update_record_button_state()


class IconCanvasItem(CanvasItem.TextButtonCanvasItem):

    def __init__(self, icon_id: str) -> None:
        super().__init__()
        self.__icon_id = icon_id
        self.wants_mouse_events = True
        self.__mouse_inside = False
        self.__mouse_pressed = False
        self.fill_style = "rgb(128, 128, 128)"
        self.fill_style_pressed = "rgb(64, 64, 64)"
        self.fill_style_disabled = "rgb(192, 192, 192)"
        self.border_style: typing.Optional[str] = None
        self.border_style_pressed: typing.Optional[str] = None
        self.border_style_disabled: typing.Optional[str] = None
        self.stroke_style = "#FFF"
        self.stroke_width = 3.0
        self.on_button_clicked: typing.Optional[typing.Callable[[], None]] = None
        self.__size_to_content()

    def close(self) -> None:
        self.on_button_clicked = None
        super().close()

    def __size_to_content(self, horizontal_padding: typing.Optional[int] = None, vertical_padding: typing.Optional[int] = None) -> None:
        """ Size the canvas item to the text content. """

        if horizontal_padding is None:
            horizontal_padding = 0

        if vertical_padding is None:
            vertical_padding = 0

        self.update_sizing(self.sizing.with_fixed_size(Geometry.IntSize(18 + 2 * horizontal_padding, 18 + 2 * vertical_padding)))

    def mouse_entered(self) -> bool:
        self.__mouse_inside = True
        self.update()
        return True

    def mouse_exited(self) -> bool:
        self.__mouse_inside = False
        self.update()
        return True

    def mouse_pressed(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        self.__mouse_pressed = True
        self.update()
        return True

    def mouse_released(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        self.__mouse_pressed = False
        self.update()
        return True

    def mouse_clicked(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        if self.enabled:
            if self.on_button_clicked:
                self.on_button_clicked()
        return True

    def _repaint(self, drawing_context: DrawingContext.DrawingContext) -> None:
        canvas_size = self.canvas_size
        if not canvas_size:
            return
        with drawing_context.saver():
            center_x = canvas_size.width * 0.5
            center_y = canvas_size.height * 0.5
            drawing_context.begin_path()
            drawing_context.move_to(center_x + 7.0, center_y)
            drawing_context.arc(center_x, center_y, 7.0, 0, 2 * math.pi)
            if self.enabled:
                if self.__mouse_inside and self.__mouse_pressed:
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
            # drawing_context.rect(0, 0, canvas_size.height, canvas_size.width)
            # drawing_context.stroke_style = "#F00"
            # drawing_context.line_width = 1.0
            # drawing_context.stroke()


class CharButtonCanvasItem(CanvasItem.TextButtonCanvasItem):

    def __init__(self, char: str) -> None:
        super().__init__()
        self.__char = char
        self.wants_mouse_events = True
        self.__mouse_inside = False
        self.__mouse_pressed = False
        self.fill_style = "rgb(255, 255, 255)"
        self.fill_style_pressed = "rgb(128, 128, 128)"
        self.fill_style_disabled = "rgb(192, 192, 192)"
        self.border_style = "rgb(192, 192, 192)"
        self.border_style_pressed = "rgb(128, 128, 128)"
        self.border_style_disabled = "rgb(192, 192, 192)"
        self.stroke_style = "#000"
        self.border_enabled = False
        self.on_button_clicked : typing.Optional[typing.Callable[[], None]] = None

    def close(self) -> None:
        self.on_button_clicked = None
        super().close()

    def mouse_entered(self) -> bool:
        self.__mouse_inside = True
        self.update()
        return True

    def mouse_exited(self) -> bool:
        self.__mouse_inside = False
        self.update()
        return True

    def mouse_pressed(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        self.__mouse_pressed = True
        self.update()
        return True

    def mouse_released(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        self.__mouse_pressed = False
        self.update()
        return True

    def mouse_clicked(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        if self.enabled:
            if callable(self.on_button_clicked):
                self.on_button_clicked()
        return True

    def _repaint(self, drawing_context: DrawingContext.DrawingContext) -> None:
        canvas_size = self.canvas_size
        if not canvas_size:
            return
        with drawing_context.saver():
            center_x = int(canvas_size.width * 0.5)
            center_y = int(canvas_size.height * 0.5)
            drawing_context.begin_path()
            height = 18 if sys.platform == "win32" else 20
            text_base = 4 if sys.platform == "win32" else 6
            drawing_context.round_rect(center_x - 7.5, center_y - 9.5, 14.0, height, 2.0)
            if self.enabled:
                if self.__mouse_inside and self.__mouse_pressed:
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
            drawing_context.fill_text(self.__char, center_x, center_y + text_base)


class ArrowSliderCanvasItem(CanvasItem.AbstractCanvasItem):

    def __init__(self, ui: UserInterface.UserInterface, event_loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.ui = ui
        self.__event_loop = event_loop
        self.wants_mouse_events = True
        self.__mouse_inside = False
        self.__mouse_pressed = False
        self.fill_style = "rgb(255, 255, 255)"
        self.fill_style_pressed = "rgb(128, 128, 128)"
        self.fill_style_disabled = "rgb(192, 192, 192)"
        self.border_style = "rgb(192, 192, 192)"
        self.border_style_pressed = "rgb(128, 128, 128)"
        self.border_style_disabled = "rgb(192, 192, 192)"
        self.stroke_style = "#000"
        self.border_enabled = False
        self.enabled = True
        self.on_mouse_delta : typing.Optional[typing.Callable[[Geometry.IntPoint], None]] = None
        self.__label_canvas_item: typing.Optional[CanvasItem.StaticTextCanvasItem] = None

    @property
    def text(self) -> str:
        if self.__label_canvas_item:
            return self.__label_canvas_item.text
        return str()

    @text.setter
    def text(self, value: str) -> None:
        if self.__label_canvas_item:
            self.__label_canvas_item.text = value

    def mouse_entered(self) -> bool:
        self.__mouse_inside = True
        self.update()
        return True

    def mouse_exited(self) -> bool:
        self.__mouse_inside = False
        self.update()
        return True

    def mouse_pressed(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        self.__mouse_pressed = True
        self.update()
        return True

    def mouse_released(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        self.__mouse_pressed = False
        self.update()
        return True

    def mouse_clicked(self, x: int, y: int, modifiers: UserInterface.KeyboardModifiers) -> bool:
        if self.enabled:
            # create the popup window content
            background_canvas_item = CanvasItem.BackgroundCanvasItem("#00FA9A")
            self.__label_canvas_item = CanvasItem.StaticTextCanvasItem()

            def mouse_position_changed_by(mouse_delta: Geometry.IntPoint) -> None:
                if callable(self.on_mouse_delta):
                    self.on_mouse_delta(mouse_delta)

            canvas_item = CanvasItem.CanvasItemComposition()
            canvas_item.add_canvas_item(background_canvas_item)
            canvas_item.add_canvas_item(self.__label_canvas_item)

            global_pos = self.map_to_global(Geometry.IntPoint(x=x, y=y))
            MouseTrackingCanvasItem.start_mouse_tracker(self.ui, self.__event_loop, canvas_item, mouse_position_changed_by, global_pos, Geometry.IntSize(20, 80))
        return True

    def _repaint(self, drawing_context: DrawingContext.DrawingContext) -> None:
        canvas_size = self.canvas_size
        if not canvas_size:
            return
        with drawing_context.saver():
            center_x = int(canvas_size.width * 0.5)
            center_y = int(canvas_size.height * 0.5)
            drawing_context.begin_path()
            height = 18 if sys.platform == "win32" else 20
            drawing_context.round_rect(center_x - 7.5, center_y - 9.5, 14.0, height, 2.0)
            if self.enabled:
                if self.__mouse_inside and self.__mouse_pressed:
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
            arrow_length = 10.0
            feather_length = 2.0
            half = arrow_length / 2
            drawing_context.begin_path()
            drawing_context.move_to(center_x - (half - feather_length), center_y - feather_length)
            drawing_context.line_to(center_x - half, center_y)
            drawing_context.line_to(center_x - (half - feather_length), center_y + feather_length)
            drawing_context.move_to(center_x + (half - feather_length), center_y - feather_length)
            drawing_context.line_to(center_x + half, center_y)
            drawing_context.line_to(center_x + (half - feather_length), center_y + feather_length)
            drawing_context.move_to(center_x - half, center_y)
            drawing_context.line_to(center_x + half, center_y)
            drawing_context.stroke_style = self.stroke_style
            drawing_context.stroke()


class LinkedCheckBoxCanvasItem(CanvasItem.CheckBoxCanvasItem):

    def __init__(self) -> None:
        super().__init__()
        self.update_sizing(self.sizing.with_fixed_size(Geometry.IntSize(w=10, h=30)))

    def _repaint(self, drawing_context: DrawingContext.DrawingContext) -> None:
        canvas_size = self.canvas_size
        if not canvas_size:
            return
        with drawing_context.saver():
            drawing_context.begin_path()
            if self.check_state == "checked":
                drawing_context.move_to(canvas_size.width - 2, 0)
                drawing_context.line_to(2, 0)
                drawing_context.line_to(2, canvas_size.height)
                drawing_context.line_to(canvas_size.width - 2, canvas_size.height)
            else:
                drawing_context.move_to(canvas_size.width - 2, 0)
                drawing_context.line_to(2, 0)
                drawing_context.line_to(2, canvas_size.height * 0.5 - 4)
                drawing_context.move_to(0, canvas_size.height * 0.5 - 4)
                drawing_context.line_to(4, canvas_size.height * 0.5 - 4)
                drawing_context.move_to(0, canvas_size.height * 0.5 + 4)
                drawing_context.line_to(4, canvas_size.height * 0.5 + 4)
                drawing_context.move_to(2, canvas_size.height * 0.5 + 4)
                drawing_context.line_to(2, canvas_size.height)
                drawing_context.line_to(canvas_size.width - 2, canvas_size.height)
            drawing_context.stroke_style = "#000"
            drawing_context.line_width = 1.0
            drawing_context.stroke()


class ScanControlWidget(Widgets.CompositeWidgetBase):

    """A controller for the scan control widget.

    Pass in the document controller and the scan controller.

    This widget presents the UI for the scan controller. All display logic and control is done through the scan
    controller.
    """

    def __init__(self, document_controller: DocumentController.DocumentController, scan_controller: scan_base.ScanHardwareSource) -> None:
        column_widget = document_controller.ui.create_column_widget(properties={"margin": 6, "spacing": 2})
        super().__init__(column_widget)

        self.document_controller = document_controller

        self.__state_controller = ScanControlStateController(scan_controller, document_controller.queue_task, document_controller.document_model, None)

        self.__shift_click_state: typing.Optional[str] = None

        ui = document_controller.ui

        self.__key_pressed_event_listener = DisplayPanel.DisplayPanelManager().key_pressed_event.listen(self.image_panel_key_pressed)
        self.__key_released_event_listener = DisplayPanel.DisplayPanelManager().key_released_event.listen(self.image_panel_key_released)
        self.__image_display_mouse_pressed_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_pressed_event.listen(self.image_panel_mouse_pressed)
        self.__image_display_mouse_released_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_released_event.listen(self.image_panel_mouse_released)
        self.__mouse_pressed = False

        def handle_record_data_item(data_item: DataItem.DataItem) -> None:
            def perform() -> None:
                document_controller.document_model.append_data_item(data_item)
                result_display_panel = document_controller.next_result_display_panel()
                if result_display_panel:
                    result_display_panel.set_display_panel_data_item(data_item)
                    result_display_panel.request_focus()
            document_controller.queue_task(perform)

        sliders_icon_24_png = pkgutil.get_data(__name__, "resources/sliders_icon_24.png")
        assert sliders_icon_24_png is not None
        open_controls_button = CanvasItem.BitmapButtonCanvasItem(CanvasItem.load_rgba_data_from_bytes(sliders_icon_24_png, "png"))
        open_controls_widget = ui.create_canvas_widget(properties={"height": 24, "width": 24})
        open_controls_widget.canvas_item.add_canvas_item(open_controls_button)
        profile_label = ui.create_label_widget(_("Scan profile: "), properties={"margin": 4})
        profile_combo = ui.create_combo_box_widget(properties={"min-width": 72})
        play_state_label = ui.create_label_widget()
        play_button = ui.create_push_button_widget(_("Scan"))
        play_button.on_clicked = self.__state_controller.handle_play_pause_clicked
        abort_button = ui.create_push_button_widget(_("Abort"))
        abort_button.on_clicked = self.__state_controller.handle_abort_clicked
        record_state_label = ui.create_label_widget()
        record_button = ui.create_push_button_widget(_("Record"))
        record_button.on_clicked = functools.partial(self.__state_controller.handle_record_clicked, handle_record_data_item)
        record_abort_button = ui.create_push_button_widget(_("Abort"))
        record_abort_button.on_clicked = self.__state_controller.handle_record_abort_clicked
        probe_state_label = ui.create_label_widget(_("Unknown"))
        positioned_check_box = ui.create_check_box_widget(_("Positioned"))
        ac_line_sync_check_box = ui.create_check_box_widget(_("AC Line Sync"))

        button_row1 = ui.create_row_widget(properties={"spacing": 2})
        button_row1.add(profile_label)
        button_row1.add(profile_combo)
        button_row1.add_stretch()
        button_row1.add(open_controls_widget)

        def handle_width_changed(text: str) -> None:
            self.__state_controller.handle_width_changed(text)
            width_field.request_refocus()

        def handle_decrease_width() -> None:
            self.__state_controller.handle_decrease_width()
            width_field.request_refocus()

        def handle_increase_width() -> None:
            self.__state_controller.handle_increase_width()
            width_field.request_refocus()

        width_field = ui.create_line_edit_widget(properties={"width": 44, "stylesheet": "qproperty-alignment: AlignRight"})  # note: this alignment technique will not work in future
        width_field.on_editing_finished = handle_width_changed

        def handle_height_changed(text: str) -> None:
            self.__state_controller.handle_height_changed(text)
            height_field.request_refocus()

        def handle_decrease_height() -> None:
            self.__state_controller.handle_decrease_height()
            height_field.request_refocus()

        def handle_increase_height() -> None:
            self.__state_controller.handle_increase_height()
            height_field.request_refocus()

        height_field = ui.create_line_edit_widget(properties={"width": 44, "stylesheet": "qproperty-alignment: AlignRight"})  # note: this alignment technique will not work in future
        height_field.on_editing_finished = handle_height_changed

        def handle_time_changed(text: str) -> None:
            self.__state_controller.handle_time_changed(text)
            time_field.request_refocus()

        def handle_decrease_time() -> None:
            self.__state_controller.handle_decrease_time()
            time_field.request_refocus()

        def handle_increase_time() -> None:
            self.__state_controller.handle_increase_time()
            time_field.request_refocus()

        time_field = ui.create_line_edit_widget(properties={"width": 44, "stylesheet": "qproperty-alignment: AlignRight"})  # note: this alignment technique will not work in future
        time_field.on_editing_finished = handle_time_changed

        def handle_fov_changed(text: str) -> None:
            self.__state_controller.handle_fov_changed(text)
            fov_field.request_refocus()

        def handle_decrease_fov() -> None:
            self.__state_controller.handle_decrease_fov()
            time_field.request_refocus()

        def handle_increase_fov() -> None:
            self.__state_controller.handle_increase_fov()
            time_field.request_refocus()

        fov_field = ui.create_line_edit_widget(properties={"width": 44, "stylesheet": "qproperty-alignment: AlignRight"})  # note: this alignment technique will not work in future
        fov_field.on_editing_finished = handle_fov_changed

        def handle_rotation_changed(text: str) -> None:
            self.__state_controller.handle_rotation_changed(text)
            rotation_field.request_refocus()

        rotation_field = ui.create_line_edit_widget(properties={"width": 44, "stylesheet": "qproperty-alignment: AlignRight"})  # note: this alignment technique will not work in future
        rotation_field.on_editing_finished = handle_rotation_changed

        time_row = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        time_row.add_stretch()
        time_row.add(ui.create_label_widget(_(u"Time (\u00b5s)"), properties={"width": 68, "stylesheet": "qproperty-alignment: 'AlignVCenter | AlignRight'"}))  # note: this alignment technique will not work in future
        time_row.add_spacing(4)
        time_group = ui.create_row_widget(properties={"width": 84})
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        decrease_button = CharButtonCanvasItem("F")
        canvas_widget.canvas_item.add_canvas_item(decrease_button)
        time_group.add(canvas_widget)
        time_group.add(time_field)
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        increase_button = CharButtonCanvasItem("S")
        canvas_widget.canvas_item.add_canvas_item(increase_button)
        time_group.add(canvas_widget)
        time_row.add(time_group)

        decrease_button.on_button_clicked = handle_decrease_time
        increase_button.on_button_clicked = handle_increase_time

        width_row = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        width_row.add_stretch()
        width_row.add(ui.create_label_widget(_("Width"), properties={"stylesheet": "qproperty-alignment: 'AlignVCenter | AlignRight'"}))  # note: this alignment technique will not work in future
        width_row.add_spacing(4)
        width_group = ui.create_row_widget(properties={"width": 84})
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        decrease_button = CharButtonCanvasItem("L")
        canvas_widget.canvas_item.add_canvas_item(decrease_button)
        width_group.add(canvas_widget)
        width_group.add(width_field)
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        increase_button = CharButtonCanvasItem("H")
        canvas_widget.canvas_item.add_canvas_item(increase_button)
        width_group.add(canvas_widget)
        width_row.add(width_group)

        decrease_button.on_button_clicked = handle_decrease_width
        increase_button.on_button_clicked = handle_increase_width

        height_row = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        height_row.add_stretch()
        height_row.add(ui.create_label_widget(_("Height"), properties={"stylesheet": "qproperty-alignment: 'AlignVCenter | AlignRight'"}))  # note: this alignment technique will not work in future
        height_row.add_spacing(4)
        height_group = ui.create_row_widget(properties={"width": 84})
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        decrease_button = CharButtonCanvasItem("L")
        canvas_widget.canvas_item.add_canvas_item(decrease_button)
        height_group.add(canvas_widget)
        height_group.add(height_field)
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        increase_button = CharButtonCanvasItem("H")
        canvas_widget.canvas_item.add_canvas_item(increase_button)
        height_group.add(canvas_widget)
        height_row.add(height_group)

        decrease_button.on_button_clicked = handle_decrease_height
        increase_button.on_button_clicked = handle_increase_height

        link_canvas_widget = ui.create_canvas_widget(properties={"height": 18, "width": 18})
        link_checkbox = LinkedCheckBoxCanvasItem()
        link_canvas_widget.canvas_item.add_canvas_item(link_checkbox)

        def handle_linked_changed(linked: bool) -> None:
            self.__state_controller.handle_linked_changed(linked)
            if linked:
                width_field.request_refocus()

        link_checkbox.on_checked_changed = handle_linked_changed

        fov_row = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        fov_row.add_stretch()
        fov_row.add(ui.create_label_widget(_("FOV (nm)"), properties={"width": 68, "stylesheet": "qproperty-alignment: 'AlignVCenter | AlignRight'"}))  # note: this alignment technique will not work in future
        fov_row.add_spacing(4)
        fov_group = ui.create_row_widget(properties={"width": 84})
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        decrease_button = CharButtonCanvasItem("I")
        canvas_widget.canvas_item.add_canvas_item(decrease_button)
        fov_group.add(canvas_widget)
        fov_group.add(fov_field)
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        increase_button = CharButtonCanvasItem("O")
        canvas_widget.canvas_item.add_canvas_item(increase_button)
        fov_group.add(canvas_widget)
        fov_row.add(fov_group)

        decrease_button.on_button_clicked = handle_decrease_fov
        increase_button.on_button_clicked = handle_increase_fov

        rotation_row = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        rotation_row.add_stretch()
        rotation_row.add(ui.create_label_widget(_("Rot. (deg)"), properties={"width": 68, "stylesheet": "qproperty-alignment: 'AlignVCenter | AlignRight'"}))  # note: this alignment technique will not work in future
        rotation_row.add_spacing(4)
        rotation_group = ui.create_row_widget(properties={"width": 84})
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        rotation_tracker = ArrowSliderCanvasItem(ui, document_controller.event_loop)
        canvas_widget.canvas_item.add_canvas_item(rotation_tracker)
        rotation_group.add(canvas_widget)
        rotation_group.add(rotation_field)
        rotation_group.add_spacing(18)
        rotation_row.add(rotation_group)

        def rotation_tracker_mouse_delta(mouse_delta: Geometry.IntPoint) -> None:
            float_to_string_converter = Converter.FloatToStringConverter()
            rotation = float_to_string_converter.convert_back(rotation_field.text) or 0.0
            text = str(rotation + mouse_delta.x / 20)
            self.__state_controller.handle_rotation_changed(text)

        rotation_tracker.on_mouse_delta = rotation_tracker_mouse_delta

        subscan_checkbox = ui.create_check_box_widget(_("Subscan"))
        subscan_checkbox.on_checked_changed = self.__state_controller.handle_subscan_enabled

        line_scan_checkbox = ui.create_check_box_widget(_("Line Scan"))
        line_scan_checkbox.on_checked_changed = self.__state_controller.handle_line_scan_enabled

        drift_checkbox = ui.create_check_box_widget(_("Drift Correct Every"))
        drift_checkbox.on_checked_changed = self.__state_controller.handle_drift_enabled

        drift_settings_value = ui.create_line_edit_widget(properties={"width": 44, "stylesheet": "qproperty-alignment: AlignRight"})
        drift_settings_unit = ui.create_combo_box_widget([_("Scan Lines"), _("Scan Frames")])

        def drift_value_edited(text: str) -> None:
            drift_settings = copy.copy(scan_controller.drift_settings)
            drift_settings.interval = Converter.IntegerToStringConverter().convert_back(text) or 0
            scan_controller.drift_settings = drift_settings
            drift_settings_value.request_refocus()

        def drift_unit_changed(index: typing.Optional[int]) -> None:
            if index is not None:
                drift_settings = copy.copy(scan_controller.drift_settings)
                drift_settings.interval_units = stem_controller.DriftIntervalUnit(index + 2)
                scan_controller.drift_settings = drift_settings

        drift_settings_value.on_editing_finished = drift_value_edited
        drift_settings_unit.on_current_index_changed = drift_unit_changed

        drift_row = ui.create_row_widget()
        drift_row.add(drift_checkbox)
        drift_row.add_spacing(12)
        drift_row.add(drift_settings_value)
        drift_row.add_spacing(4)
        drift_row.add(drift_settings_unit)
        drift_row.add_stretch()

        time_column = ui.create_column_widget()
        time_column.add(time_row)
        time_column.add(fov_row)
        time_column.add(rotation_row)
        time_column.add_stretch()

        size_column = ui.create_column_widget()
        size_column.add(width_row)
        size_column.add(height_row)

        link_column = ui.create_column_widget()
        link_column.add_stretch()
        link_column.add(link_canvas_widget)
        link_column.add_stretch()

        sizes_row = ui.create_row_widget()
        sizes_row.add_stretch()
        sizes_row.add(link_column)
        sizes_row.add(size_column)

        subscan_column = ui.create_column_widget()
        subscan_column.add(subscan_checkbox)
        subscan_column.add_spacing(4)
        subscan_column.add(line_scan_checkbox)

        sizes_row_column = ui.create_column_widget()
        sizes_row_column.add(sizes_row)
        sizes_row_column.add(subscan_column)
        sizes_row_column.add_stretch()

        parameters_group1 = ui.create_row_widget()
        parameters_group2 = ui.create_row_widget()

        parameters_group1.add(time_column)
        parameters_group1.add_stretch()
        parameters_group1.add(sizes_row_column)
        parameters_group2.add_stretch()

        thumbnail_row = ui.create_row_widget()
        thumbnail_group = ui.create_row_widget(properties={"spacing": 4})
        thumbnail_row.add_stretch()
        thumbnail_row.add(thumbnail_group)
        thumbnail_row.add_stretch()

        scan_row = ui.create_row_widget()
        scan_row.add(play_button, alignment="left")
        scan_row.add_spacing(6)
        scan_row.add(abort_button, alignment="left")
        scan_row.add_stretch()
        scan_row.add(play_state_label)

        record_row = ui.create_row_widget()
        record_row.add(record_button, alignment="right")
        record_row.add_spacing(6)
        record_row.add(record_abort_button, alignment="right")
        record_row.add_stretch()
        record_row.add(record_state_label)

        row5 = ui.create_row_widget(properties={"spacing": 2})
        row5.add(probe_state_label)
        row5.add_stretch()

        row7 = ui.create_row_widget(properties={"spacing": 2})
        row7.add(positioned_check_box)
        row7.add_stretch()
        row7.add(ac_line_sync_check_box, alignment="right")

        column_widget.add(button_row1)
        column_widget.add(parameters_group1)
        column_widget.add(parameters_group2)
        column_widget.add(drift_row)
        column_widget.add(scan_row)
        column_widget.add(record_row)
        column_widget.add(thumbnail_row)
        column_widget.add(row5)
        column_widget.add(row7)
        column_widget.add_stretch()

        def positioned_check_state_changed(check_state: str) -> None:
            self.__state_controller.handle_positioned_check_box(check_state == "checked")

        def ac_line_sync_check_state_changed(check_state: str) -> None:
            self.__state_controller.handle_ac_line_sync_check_box(check_state == "checked")

        open_controls_button.on_button_clicked = functools.partial(self.__state_controller.handle_settings_button_clicked, PlugInManager.APIBroker())
        profile_combo.on_current_text_changed = self.__state_controller.handle_change_profile
        positioned_check_box.on_check_state_changed = positioned_check_state_changed
        ac_line_sync_check_box.on_check_state_changed = ac_line_sync_check_state_changed

        def profiles_changed(items: typing.Sequence[typing.Any]) -> None:
            profile_combo.items = items

        # thread safe
        def profile_changed(profile_label: str) -> None:
            # the current_text must be set on ui thread
            self.document_controller.queue_task(lambda: setattr(profile_combo, "current_text", profile_label))

        def frame_parameters_changed(frame_parameters: scan_base.ScanFrameParameters) -> None:
            width_field.text = str(int(frame_parameters.size[1]))
            height_field.text = str(int(frame_parameters.size[0]))
            time_field.text = str("{0:.2f}".format(float(frame_parameters.pixel_time_us)))
            fov_field.text = str("{0:.1f}".format(float(frame_parameters.fov_nm)))
            rotation_field.text = str("{0:.1f}".format(float(frame_parameters.rotation_rad) * 180.0 / math.pi))
            rotation_tracker.text = rotation_field.text
            ac_line_sync_check_box.check_state = "checked" if frame_parameters.ac_line_sync else "unchecked"
            if width_field.focused:
                width_field.request_refocus()
            if height_field.focused:
                height_field.request_refocus()
            if time_field.focused:
                time_field.request_refocus()
            if fov_field.focused:
                fov_field.request_refocus()
            if rotation_field.focused:
                rotation_field.request_refocus()

        def linked_changed(linked: bool) -> None:
            link_checkbox.checked = linked

        def scan_button_state_changed(enabled: bool, play_button_state: str) -> None:
            play_button_text = {"scan": _("Scan"), "stop": _("Stop")}
            play_button.enabled = enabled
            play_button.text = play_button_text[play_button_state]

        def abort_button_state_changed(visible: bool, enabled: bool) -> None:
            # abort_button.visible = visible
            abort_button.enabled = enabled

        def record_button_state_changed(visible: bool, enabled: bool) -> None:
            record_button.visible = visible
            record_button.enabled = enabled

        def record_abort_button_state_changed(visible: bool, enabled: bool) -> None:
            # record_abort_button.visible = visible
            record_abort_button.enabled = enabled

        def acquisition_state_changed(key: str) -> None:
            # this may be called on a thread. create an async method (guaranteed to run on the main thread)
            # and add it to the window event loop.
            async def update_acquisition_state_label(acquisition_states: typing.Mapping[str, typing.Optional[str]]) -> None:
                acquisition_state: typing.Optional[str] = None
                for acquisition_state in acquisition_states.values():
                    if acquisition_state != "stopped":
                        break
                acquisition_state = acquisition_state or "stopped"
                play_state_label.text = map_channel_state_to_text[acquisition_state]

            acquisition_states = self.__state_controller.acquisition_state_model.value or dict()
            self.document_controller.event_loop.create_task(update_acquisition_state_label(acquisition_states))

        def probe_state_changed(probe_state: str, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
            map_probe_state_to_text = {"scanning": _("Scanning"), "parked": _("Parked")}
            if probe_state != "scanning":
                if probe_position is not None:
                    probe_position_str = " " + str(int(probe_position.x * 100)) + "%" + ", " + str(int(probe_position.y * 100)) + "%"
                else:
                    probe_position_str = " Default"
            else:
                probe_position_str = ""
            probe_state_label.text = map_probe_state_to_text.get(probe_state, "") + probe_position_str

        def positioned_check_box_changed(checked: bool) -> None:
            positioned_check_box.check_state = "checked" if checked else "unchecked"

        def ac_line_sync_check_box_changed(checked: bool) -> None:
            ac_line_sync_check_box.check_state = "checked" if checked else "unchecked"

        def channel_state_changed(channel_index: int, enabled: bool, is_subscan_channel: bool) -> None:
            # then rebuild thumbnail for the channel_index, setting up the thumbnail widget, a drag
            # handler, and checkbox handler.

            data_channel = self.__state_controller.data_channels[channel_index]
            channel_id = data_channel.channel_id or str()
            name = data_channel.name

            thumbnail_column = typing.cast(UserInterface.BoxWidget, thumbnail_group.children[channel_index])
            thumbnail_column.remove_all()

            actual_channel_id = channel_id if not is_subscan_channel else channel_id + "_subscan"

            document_model = document_controller.document_model
            data_item_reference = document_model.get_data_item_reference(document_model.make_data_item_reference_key(scan_controller.hardware_source_id, actual_channel_id))
            data_item_thumbnail_source = DataItemThumbnailWidget.DataItemReferenceThumbnailSource(ui, document_model, data_item_reference)
            thumbnail_widget = DataItemThumbnailWidget.DataItemThumbnailWidget(ui, data_item_thumbnail_source, Geometry.IntSize(width=48, height=48))

            def thumbnail_widget_drag(mime_data: UserInterface.MimeData, thumbnail: typing.Optional[DrawingContext.RGBA32Type], hot_spot_x: int, hot_spot_y: int) -> None:
                column_widget.drag(mime_data, thumbnail, hot_spot_x, hot_spot_y)

            thumbnail_widget.on_drag = thumbnail_widget_drag

            thumbnail_column.add(thumbnail_widget)
            channel_enabled_check_box_widget = ui.create_check_box_widget(name)
            channel_enabled_check_box_widget.checked = enabled

            def checked_changed(checked: bool) -> None:
                self.__state_controller.handle_enable_channel(channel_index, checked)

            channel_enabled_check_box_widget.on_checked_changed = checked_changed
            thumbnail_column.add(channel_enabled_check_box_widget)

        async def update_subscan_state(subscan_state: stem_controller.SubscanState, line_scan_state: stem_controller.LineScanState) -> None:
            for channel_index in range(scan_controller.channel_count):
                is_subscan_channel = subscan_state == stem_controller.SubscanState.ENABLED or line_scan_state == stem_controller.LineScanState.ENABLED
                channel_state_changed(channel_index, self.__state_controller.get_channel_enabled(channel_index), is_subscan_channel)
            subscan_checkbox.enabled = subscan_state != stem_controller.SubscanState.INVALID
            subscan_checkbox.checked = subscan_state == stem_controller.SubscanState.ENABLED
            line_scan_checkbox.enabled = line_scan_state != stem_controller.LineScanState.INVALID
            line_scan_checkbox.checked = line_scan_state == stem_controller.LineScanState.ENABLED

        def subscan_state_changed(subscan_state: stem_controller.SubscanState, line_scan_state: stem_controller.LineScanState) -> None:
            # handle subscan state changes from the low level
            self.document_controller.event_loop.create_task(update_subscan_state(subscan_state, line_scan_state))

        async def update_drift_state(drift_channel_id: typing.Optional[str], drift_region: typing.Optional[Geometry.FloatRect], drift_settings: stem_controller.DriftCorrectionSettings, subscan_state: stem_controller.SubscanState) -> None:
            enabled = subscan_state != stem_controller.SubscanState.INVALID
            drift_checkbox.enabled = enabled
            drift_settings_value.enabled = enabled
            drift_settings_unit.enabled = enabled
            drift_checkbox.checked = drift_channel_id is not None and drift_region is not None
            drift_settings_value.text = Converter.IntegerToStringConverter().convert(drift_settings.interval)
            drift_settings_unit.current_item = drift_settings.interval_units

        def drift_state_changed(drift_channel_id: typing.Optional[str], drift_region: typing.Optional[Geometry.FloatRect], drift_settings: stem_controller.DriftCorrectionSettings, subscan_state: stem_controller.SubscanState) -> None:
            self.document_controller.event_loop.create_task(update_drift_state(drift_channel_id, drift_region, drift_settings, subscan_state))

        self.__state_controller.on_display_name_changed = None
        self.__state_controller.on_profiles_changed = profiles_changed
        self.__state_controller.on_profile_changed = profile_changed
        self.__state_controller.on_frame_parameters_changed = frame_parameters_changed
        self.__state_controller.on_linked_changed = linked_changed
        self.__state_controller.on_scan_button_state_changed = scan_button_state_changed
        self.__state_controller.on_abort_button_state_changed = abort_button_state_changed
        self.__state_controller.on_record_button_state_changed = record_button_state_changed
        self.__state_controller.on_record_abort_button_state_changed = record_abort_button_state_changed
        self.__state_controller.on_probe_state_changed = lambda a, b: self.document_controller.queue_task(lambda: probe_state_changed(a, b))
        self.__state_controller.on_positioned_check_box_changed = lambda a: self.document_controller.queue_task(lambda: positioned_check_box_changed(a))
        self.__state_controller.on_ac_line_sync_check_box_changed = lambda a: self.document_controller.queue_task(lambda: ac_line_sync_check_box_changed(a))
        self.__state_controller.on_channel_state_changed = channel_state_changed
        self.__state_controller.on_subscan_state_changed = subscan_state_changed
        self.__state_controller.on_drift_state_changed = drift_state_changed

        # before state controller gets initialized
        for i in range(scan_controller.channel_count):
            thumbnail_group.add(ui.create_column_widget())

        self.__acquisition_state_changed_listener = self.__state_controller.acquisition_state_model.property_changed_event.listen(acquisition_state_changed)

        self.__state_controller.initialize_state()

        acquisition_state_changed("value")

    def close(self) -> None:
        self.__key_pressed_event_listener.close()
        self.__key_pressed_event_listener = typing.cast(typing.Any, None)
        self.__key_released_event_listener.close()
        self.__key_released_event_listener = typing.cast(typing.Any, None)
        self.__image_display_mouse_pressed_event_listener.close()
        self.__image_display_mouse_pressed_event_listener= typing.cast(typing.Any, None)
        self.__image_display_mouse_released_event_listener.close()
        self.__image_display_mouse_released_event_listener= typing.cast(typing.Any, None)
        self.__state_controller.close()
        self.__state_controller = typing.cast(typing.Any, None)
        self.__acquisition_state_changed_listener = typing.cast(typing.Any, None)
        super().close()

    def periodic(self) -> None:
        self.__state_controller.handle_periodic()
        super().periodic()

    # this gets called from the DisplayPanelManager. pass on the message to the state controller.
    # must be called on ui thread
    def image_panel_mouse_pressed(self, display_panel: DisplayPanel.DisplayPanel, display_item: DisplayItem.DisplayItem, image_position: Geometry.FloatPoint, modifiers: CanvasItem.KeyboardModifiers) -> bool:
        data_item = display_panel.data_item if display_panel else None
        hardware_source_id = data_item.metadata.get("hardware_source", dict()).get("hardware_source_id") if data_item else str()
        logger = logging.getLogger("camera_control_ui")
        logger.propagate = False  # do not send messages to root logger
        if not logger.handlers:
            logger.addHandler(logging.handlers.BufferingHandler(4))
        camera_shape = data_item.dimensional_shape if data_item else ()
        if data_item and hardware_source_id and len(camera_shape) == 2 and self.__shift_click_state == "shift":
            mouse_position = image_position
            self.__mouse_pressed = self.__state_controller.handle_shift_click(hardware_source_id, mouse_position, typing.cast(DataAndMetadata.Shape2dType, camera_shape), logger)
            logger_buffer = typing.cast(logging.handlers.BufferingHandler, logger.handlers[0])
            for record in logger_buffer.buffer:
                display_panel.document_controller.display_log_record(record)
            logger_buffer.flush()
            return self.__mouse_pressed
        return False

    def image_panel_mouse_released(self, display_panel: DisplayPanel.DisplayPanel, display_item: DisplayItem.DisplayItem, image_position: Geometry.FloatPoint, modifiers: CanvasItem.KeyboardModifiers) -> bool:
        mouse_pressed = self.__mouse_pressed
        self.__mouse_pressed = False
        return mouse_pressed

    def image_panel_key_pressed(self, display_panel: DisplayPanel.DisplayPanel, key: UserInterface.Key) -> bool:
        if key.text.lower() == "s":
            self.__shift_click_state = "shift"
        else:
            self.__shift_click_state = None
        return False

    def image_panel_key_released(self, display_panel: DisplayPanel.DisplayPanel, key: UserInterface.Key) -> bool:
        self.__shift_click_state = None
        return False


class ScanControlPanel(Panel.Panel):

    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> None:
        super().__init__(document_controller, panel_id, "scan-control-panel")
        ui = document_controller.ui
        self.__column_widget = ui.create_column_widget()
        self.widget = self.__column_widget
        self.__scan_control_widget: typing.Optional[ScanControlWidget] = None
        self.__hardware_source_id = properties["hardware_source_id"]
        # listen for any hardware added or removed messages, and refresh the list
        self.__build_widget()
        HardwareSource.HardwareSourceManager().aliases_updated.append(self.__build_widget)

    def __build_widget(self) -> None:
        if not self.__scan_control_widget:
            scan_controller = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(self.__hardware_source_id)
            if isinstance(scan_controller, scan_base.ScanHardwareSource):
                self.__scan_control_widget = ScanControlWidget(self.document_controller, scan_controller)
                self.__column_widget.add(self.__scan_control_widget)
                self.__column_widget.add_spacing(12)
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
        self.__state_controller = ScanControlStateController(hardware_source, display_panel.document_controller.queue_task, display_panel.document_controller.document_model, data_channel_id)

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
            # this may be called on a thread. create an async method (guaranteed to run on the main thread)
            # and add it to the window event loop.
            async def update_acquisition_state_label(acquisition_states: typing.Mapping[str, typing.Optional[str]]) -> None:
                acquisition_state = acquisition_states.get(self.__data_channel_id, None) or "stopped"
                status_text_canvas_item.text = map_channel_state_to_text[acquisition_state]
                status_text_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

            acquisition_states = self.__state_controller.acquisition_state_model.value or dict()
            self.__display_panel.document_controller.event_loop.create_task(update_acquisition_state_label(acquisition_states))

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
        self.__display_panel.footer_canvas_item.remove_canvas_item(self.__playback_controls_composition)
        self.__display_panel = typing.cast(typing.Any, None)
        self.__state_controller.close()
        self.__state_controller = typing.cast(typing.Any, None)
        self.__acquisition_state_changed_listener = typing.cast(typing.Any, None)

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
                actions = list()
                for channel_index in range(hardware_source.data_channel_count):
                    channel_id, channel_name, __ = hardware_source.get_data_channel_state(channel_index)  # hack since there is no get_channel_info call
                    def switch_to_live_controller(hardware_source: scan_base.ScanHardwareSource, channel_id: str) -> None:
                        if selected_display_panel:
                            d = {"type": "image", "controller_type": ScanDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id, "channel_id": channel_id}
                            selected_display_panel.change_display_panel_content(d)

                    display_name = "%s (%s)" % (hardware_source.display_name, channel_name)
                    action = display_type_menu.add_menu_item(display_name, functools.partial(switch_to_live_controller, hardware_source, channel_id))
                    display_panel_controller = selected_display_panel.display_panel_controller if selected_display_panel else None
                    action.checked = isinstance(display_panel_controller, ScanDisplayPanelController) and display_panel_controller.channel_id == channel_id and display_panel_controller.hardware_source_id == hardware_source.hardware_source_id
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
                assert isinstance(hardware_source, scan_base.ScanHardwareSource)
                for channel_index in range(hardware_source.data_channel_count):
                    channel_id, channel_name, __ = hardware_source.get_data_channel_state(channel_index)  # hack since there is no get_channel_info call
                    if HardwareSource.matches_hardware_source(hardware_source.hardware_source_id, channel_id, document_model, data_item):
                        return {"controller_type": ScanDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id, "channel_id": channel_id}
                return None

        factory_id = "scan-live-" + hardware_source.hardware_source_id
        DisplayPanel.DisplayPanelManager().register_display_panel_controller_factory(factory_id, ScanDisplayPanelControllerFactory())
        name = hardware_source.display_name + " " + _("Scan Control")
        properties = {"hardware_source_id": hardware_source.hardware_source_id}
        Workspace.WorkspaceManager().register_panel(ScanControlPanel, panel_id, name, ["left", "right"], "left", properties)

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
