from __future__ import annotations

# standard libraries
import asyncio
import functools
import gettext
import logging
import logging.handlers
import math
import numpy
import pkgutil
import sys
import time
import typing

# local libraries
from nion.data import DataAndMetadata
from nion.instrumentation import camera_base
from nion.instrumentation import HardwareSource
from nion.swift import DataItemThumbnailWidget
from nion.swift import DisplayPanel
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem
from nion.swift.model import PlugInManager
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import Dialog
from nion.ui import Widgets
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.swift.model import DisplayItem
    from nion.swift.model import DocumentModel
    from nion.swift.model import Persistence
    from nion.swift import DocumentController
    from nion.ui import DrawingContext
    from nion.ui import UserInterface
    from nion.ui import Window
    from nion.utils import Event

_ = gettext.gettext

map_channel_state_to_text = {
    "stopped": _("Stopped"),
    "complete": _("Acquiring"),
    "partial": _("Acquiring"),
    "marked": _("Stopping"),
    "error": _("Error"),
}


class CameraControlStateController:
    """Track the state of a camera controller, as it relates to the UI. This object does not hold any state itself.

    Clients can query:
        (property) has_processed_data

    Clients should call:
        handle_change_profile(profile_label)
        handle_play_pause_clicked(workspace_controller)
        handle_abort_clicked()
        handle_shift_click(hardware_source_id, mouse_position, image_dimensions)
        handle_tilt_click(hardware_source_id, mouse_position, image_dimensions)
        handle_binning_changed(binning_str)
        handle_exposure_changed(exposure)
        handle_increase_exposure()
        handle_decrease_exposure()
        handle_settings_button_clicked(api_broker)
        handle_periodic()
        handle_capture_clicked()

    Clients can respond to:
        on_display_name_changed(display_name)
        on_binning_values_changed(binning_values)
        on_profiles_changed(profile_label_list)
        on_profile_changed(profile_label)
        on_frame_parameters_changed(frame_parameters)
        on_play_button_state_changed(enabled, play_button_state)  play, pause
        on_abort_button_state_changed(visible, enabled)
        on_capture_button_state_changed(visible, enabled)
        on_display_new_data_item(data_item)
        on_log_messages(messages, data_elements)
    """

    def __init__(self, camera_hardware_source: camera_base.CameraHardwareSource, queue_task: typing.Callable[[typing.Callable[[], None]], None], document_model: DocumentModel.DocumentModel) -> None:
        self.__camera_hardware_source = camera_hardware_source
        self.__has_processed_channel = typing.cast(bool, camera_hardware_source.features.get("has_processed_channel", False))
        self.use_processed_data = False
        self.queue_task = queue_task
        self.__document_model = document_model
        self.__profile_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__frame_parameters_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__acquisition_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__log_messages_event_listener: typing.Optional[Event.EventListener] = None
        self.__data_channel_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.on_display_name_changed: typing.Optional[typing.Callable[[str], None]] = None
        self.on_binning_values_changed: typing.Optional[typing.Callable[[typing.Sequence[int]], None]] = None
        self.on_profiles_changed: typing.Optional[typing.Callable[[typing.Sequence[str]], None]] = None
        self.on_profile_changed: typing.Optional[typing.Callable[[str], None]] = None
        self.on_frame_parameters_changed: typing.Optional[typing.Callable[[camera_base.CameraFrameParameters], None]] = None
        self.on_play_button_state_changed: typing.Optional[typing.Callable[[bool, str], None]] = None
        self.on_abort_button_state_changed: typing.Optional[typing.Callable[[bool, bool], None]] = None
        self.acquisition_state_model = Model.PropertyModel[str]("stopped")
        self.on_capture_button_state_changed: typing.Optional[typing.Callable[[bool, bool], None]] = None
        self.on_display_new_data_item: typing.Optional[typing.Callable[[DataItem.DataItem], None]] = None
        self.on_camera_current_changed: typing.Optional[typing.Callable[[typing.Optional[float]], None]] = None
        self.on_log_messages: typing.Optional[typing.Callable[[typing.List[str], typing.List[typing.Dict[str, typing.Any]]], None]] = None

        self.__captured_xdatas_available_event: typing.Optional[Event.EventListener] = None

        self.__camera_current = None
        self.__last_camera_current_time = 0.0
        self.__xdatas_available_event = self.__camera_hardware_source.xdatas_available_event.listen(self.__receive_new_xdatas)

        self.data_item_reference = document_model.get_data_item_reference(self.__camera_hardware_source.hardware_source_id)
        self.processed_data_item_reference = document_model.get_data_item_reference(document_model.make_data_item_reference_key(self.__camera_hardware_source.hardware_source_id, "summed"))

    def close(self) -> None:
        if self.__captured_xdatas_available_event:
            self.__captured_xdatas_available_event.close()
            self.__captured_xdatas_available_event = None
        if self.__xdatas_available_event:
            self.__xdatas_available_event.close()
            self.__xdatas_available_event = typing.cast(typing.Any, None)
        if self.__profile_changed_event_listener:
            self.__profile_changed_event_listener.close()
            self.__profile_changed_event_listener = None
        if self.__frame_parameters_changed_event_listener:
            self.__frame_parameters_changed_event_listener.close()
            self.__frame_parameters_changed_event_listener = None
        if self.__acquisition_state_changed_event_listener:
            self.__acquisition_state_changed_event_listener.close()
            self.__acquisition_state_changed_event_listener = None
        self.__data_channel_state_changed_event_listener = None
        if self.__log_messages_event_listener:
            self.__log_messages_event_listener.close()
            self.__log_messages_event_listener = None
        self.on_display_name_changed = None
        self.on_binning_values_changed = None
        self.on_profiles_changed = None
        self.on_profile_changed = None
        self.on_frame_parameters_changed = None
        self.on_play_button_state_changed = None
        self.on_abort_button_state_changed = None
        self.on_capture_button_state_changed = None
        self.on_display_new_data_item = None
        self.on_log_messages = None
        self.__camera_hardware_source = typing.cast(typing.Any, None)

    def _reset_camera_current(self) -> None:
        self.__last_camera_current_time = 0.0

    def __update_play_button_state(self) -> None:
        enabled = self.__camera_hardware_source is not None
        if callable(self.on_play_button_state_changed):
            self.on_play_button_state_changed(enabled, "pause" if self.is_playing else "play")
        if enabled and not self.is_playing and callable(self.on_camera_current_changed):
            self.on_camera_current_changed(None)

    def __update_abort_button_state(self) -> None:
        if callable(self.on_abort_button_state_changed):
            self.on_abort_button_state_changed(self.is_playing, self.is_playing)
        if callable(self.on_capture_button_state_changed):
            self.on_capture_button_state_changed(self.is_playing, not self.__captured_xdatas_available_event)

    def __update_buttons(self) -> None:
        self.__update_play_button_state()
        self.__update_abort_button_state()

    def __update_profile_state(self, profile_label: str) -> None:
        if callable(self.on_profile_changed):
            self.on_profile_changed(profile_label)

    def __update_frame_parameters(self, profile_index: int, frame_parameters: camera_base.CameraFrameParameters) -> None:
        if callable(self.on_frame_parameters_changed):
            if profile_index == self.__camera_hardware_source.selected_profile_index:
                self.on_frame_parameters_changed(frame_parameters)

    # received from the camera controller when the profile changes.
    # thread safe
    def __update_profile_index(self, profile_index: int) -> None:
        for index, name in enumerate(self.__camera_hardware_source.modes):
            if index == profile_index:
                def update_profile_state_and_frame_parameters(name: str) -> None:
                    if self.__camera_hardware_source:  # check to see if close has been called.
                        self.__update_profile_state(name)
                        self.__update_frame_parameters(self.__camera_hardware_source.selected_profile_index, self.__camera_hardware_source.get_frame_parameters(profile_index))
                self.queue_task(functools.partial(update_profile_state_and_frame_parameters, name))

    @property
    def has_processed_data(self) -> bool:
        return self.__has_processed_channel

    def __receive_new_xdatas(self, data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:
        current_time = time.time()
        if current_time - self.__last_camera_current_time > 5.0 and len(data_promises) > 0 and callable(self.on_camera_current_changed):
            xdata = data_promises[0].xdata
            if xdata:
                counts_per_electron = xdata.metadata.get("hardware_source", dict()).get("counts_per_electron")
                exposure = xdata.metadata.get("hardware_source", dict()).get("exposure")
                if xdata.intensity_calibration.units == "counts" and counts_per_electron and exposure:
                    sum_counts = xdata.intensity_calibration.convert_to_calibrated_value(numpy.sum(xdata._data_ex))
                    detector_current = sum_counts / exposure / counts_per_electron / 6.242e18 if exposure > 0 and counts_per_electron > 0 else 0.0
                    if detector_current != self.__camera_current:
                        self.__camera_current = detector_current

                        def update_camera_current() -> None:
                            if callable(self.on_camera_current_changed):
                                self.on_camera_current_changed(self.__camera_current)

                        self.queue_task(update_camera_current)
                self.__last_camera_current_time = current_time

    def initialize_state(self) -> None:
        """ Call this to initialize the state of the UI after everything has been connected. """
        if self.__camera_hardware_source:
            self.__profile_changed_event_listener = self.__camera_hardware_source.profile_changed_event.listen(self.__update_profile_index)
            self.__frame_parameters_changed_event_listener = self.__camera_hardware_source.frame_parameters_changed_event.listen(self.__update_frame_parameters)
            self.__data_channel_state_changed_event_listener = self.__camera_hardware_source.data_channel_state_changed_event.listen(self.__data_channel_state_changed)
            self.__acquisition_state_changed_event_listener = self.__camera_hardware_source.acquisition_state_changed_event.listen(self.__acquisition_state_changed)
            self.__log_messages_event_listener = self.__camera_hardware_source.log_messages_event.listen(self.__log_messages)
        if callable(self.on_display_name_changed):
            self.on_display_name_changed(self.display_name)
        if callable(self.on_binning_values_changed):
            self.on_binning_values_changed(self.__camera_hardware_source.binning_values)
        self.__update_buttons()
        if callable(self.on_profiles_changed):
            profile_items = self.__camera_hardware_source.modes
            self.on_profiles_changed(profile_items)
            self.__update_profile_index(self.__camera_hardware_source.selected_profile_index)

    # must be called on ui thread
    def handle_change_profile(self, profile_label: str) -> None:
        if profile_label in self.__camera_hardware_source.modes:
            self.__camera_hardware_source.set_selected_profile_index(self.__camera_hardware_source.modes.index(profile_label))

    def handle_play_pause_clicked(self) -> None:
        """ Call this when the user clicks the play/pause button. """
        if self.__camera_hardware_source:
            if self.is_playing:
                self.__camera_hardware_source.stop_playing()
            else:
                self.__camera_hardware_source.start_playing()

    def handle_abort_clicked(self) -> None:
        """ Call this when the user clicks the abort button. """
        if self.__camera_hardware_source:
            self.__camera_hardware_source.abort_playing()

    # must be called on ui thread
    def handle_settings_button_clicked(self, api_broker: typing.Any) -> None:
        if self.__camera_hardware_source:
            self.__camera_hardware_source.open_configuration_interface(api_broker)

    # must be called on ui thread
    def handle_shift_click(self, hardware_source_id: str, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> bool:
        if hardware_source_id == self.__camera_hardware_source.hardware_source_id:
            self.__camera_hardware_source.shift_click(mouse_position, camera_shape, logger)
            return True
        return False

    # must be called on ui thread
    def handle_tilt_click(self, hardware_source_id: str, mouse_position: Geometry.FloatPoint, camera_shape: DataAndMetadata.Shape2dType, logger: logging.Logger) -> bool:
        if hardware_source_id == self.__camera_hardware_source.hardware_source_id:
            self.__camera_hardware_source.tilt_click(mouse_position, camera_shape, logger)
            return True
        return False

    # must be called on ui thread
    def handle_binning_changed(self, binning_str: str) -> None:
        frame_parameters = self.__camera_hardware_source.get_frame_parameters(self.__camera_hardware_source.selected_profile_index)
        frame_parameters.binning = max(int(binning_str), 1)
        frame_parameters = self.__camera_hardware_source.validate_frame_parameters(frame_parameters)
        self.__camera_hardware_source.set_frame_parameters(self.__camera_hardware_source.selected_profile_index, frame_parameters)

    # must be called on ui thread
    def handle_exposure_changed(self, exposure: float) -> None:
        frame_parameters = self.__camera_hardware_source.get_frame_parameters(self.__camera_hardware_source.selected_profile_index)
        try:
            frame_parameters.exposure_ms = exposure * 1000
        except ValueError:
            pass
        frame_parameters = self.__camera_hardware_source.validate_frame_parameters(frame_parameters)
        self.__camera_hardware_source.set_frame_parameters(self.__camera_hardware_source.selected_profile_index, frame_parameters)

    def handle_decrease_exposure(self) -> None:
        frame_parameters = self.__camera_hardware_source.get_frame_parameters(self.__camera_hardware_source.selected_profile_index)
        frame_parameters.exposure_ms = frame_parameters.exposure_ms * 0.5
        frame_parameters = self.__camera_hardware_source.validate_frame_parameters(frame_parameters)
        self.__camera_hardware_source.set_frame_parameters(self.__camera_hardware_source.selected_profile_index, frame_parameters)

    def handle_increase_exposure(self) -> None:
        frame_parameters = self.__camera_hardware_source.get_frame_parameters(self.__camera_hardware_source.selected_profile_index)
        frame_parameters.exposure_ms = frame_parameters.exposure_ms * 2.0
        frame_parameters = self.__camera_hardware_source.validate_frame_parameters(frame_parameters)
        self.__camera_hardware_source.set_frame_parameters(self.__camera_hardware_source.selected_profile_index, frame_parameters)

    def handle_capture_clicked(self) -> None:
        def capture_xdatas(data_promises: typing.Sequence[HardwareSource.DataAndMetadataPromise]) -> None:
            if self.__captured_xdatas_available_event:
                self.__captured_xdatas_available_event.close()
                self.__captured_xdatas_available_event = None
            for index, data_promise in enumerate(data_promises):
                def add_data_item(data_item: DataItem.DataItem) -> None:
                    self.__document_model.append_data_item(data_item)
                    if callable(self.on_display_new_data_item):
                        self.on_display_new_data_item(data_item)

                if index == (1 if self.use_processed_data else 0):
                    xdata = data_promise.xdata
                    if xdata:
                        data_item = DataItem.new_data_item(xdata)
                        display_name = xdata.metadata.get("hardware_source", dict()).get("hardware_source_name")
                        display_name = display_name if display_name else _("Capture")
                        data_item.title = display_name
                        self.queue_task(functools.partial(add_data_item, data_item))
            self.queue_task(self.__update_buttons)

        self.__captured_xdatas_available_event = self.__camera_hardware_source.xdatas_available_event.listen(capture_xdatas)
        self.__update_buttons()

    # must be called on ui thread
    def handle_periodic(self) -> None:
        if self.__camera_hardware_source and getattr(self.__camera_hardware_source, "periodic", None):
            self.__camera_hardware_source.periodic()

    @property
    def is_playing(self) -> bool:
        """ Returns whether the hardware source is playing or not. """
        return self.__camera_hardware_source.is_playing if self.__camera_hardware_source else False

    @property
    def display_name(self) -> str:
        """ Returns the display name for the hardware source. """
        return self.__camera_hardware_source.display_name if self.__camera_hardware_source else _("N/A")

    # this message comes from the data buffer. it will always be invoked on the UI thread.
    def __acquisition_state_changed(self, is_acquiring: bool) -> None:
        if self.__captured_xdatas_available_event:
            self.__captured_xdatas_available_event.close()
            self.__captured_xdatas_available_event = None
        self.queue_task(self.__update_buttons)

    def __log_messages(self, messages: typing.List[str], data_elements: typing.List[typing.Dict[str, typing.Any]]) -> None:
        if callable(self.on_log_messages):
            self.on_log_messages(messages, data_elements)

    def __data_channel_state_changed(self, data_channel: HardwareSource.DataChannel) -> None:
        if data_channel.is_started and data_channel.state:
            self.acquisition_state_model.value = data_channel.state
        else:
            self.acquisition_state_model.value = "error" if data_channel.is_error else "stopped"


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


class CameraPanelDelegate:

    def has_feature(self, feature_flag: str) -> bool:
        return False

    def open_help(self, *, api_broker: typing.Optional[PlugInManager.APIBroker] = None) -> bool:
        return False

    def get_configuration_ui_handler(self, *, api_broker: typing.Optional[PlugInManager.APIBroker] = None,
                                     event_loop: typing.Optional[asyncio.AbstractEventLoop] = None,
                                     hardware_source_id: typing.Optional[str] = None,
                                     camera_device: typing.Optional[camera_base.CameraDevice] = None,
                                     camera_settings: typing.Optional[camera_base.CameraSettings] = None,
                                     **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        return None

    def open_configuration(self, *, api_broker: PlugInManager.APIBroker,
                           hardware_source_id: typing.Optional[str] = None,
                           camera_device: typing.Optional[camera_base.CameraDevice] = None,
                           camera_settings: typing.Optional[camera_base.CameraSettings] = None) -> bool:
        return False


exposure_units = {0: "s", -1: "s", -2: "s", -3: "ms", -4: "ms", -5: "ms", -6: "us", -7: "us", -8: "us", -9: "ns", -10: "ns", -11: "ns"}
exposure_format = {0: ".1", -1: ".1", -2: ".2", -3: ".1", -4: ".1", -5: ".2", -6: ".1", -7: ".1", -8: ".2", -9: ".1", -10: ".1", -11: ".2"}

def make_exposure_str(exposure: float, exposure_precision: int) -> str:
    format_str = f"{{0:{exposure_format[exposure_precision]}f}}"
    return str(format_str.format(exposure / math.pow(10, math.trunc(exposure_precision / 3) * 3)))


class CameraControlWidget(Widgets.CompositeWidgetBase):

    def __init__(self, document_controller: DocumentController.DocumentController, camera_hardware_source: camera_base.CameraHardwareSource) -> None:
        column_widget = document_controller.ui.create_column_widget(properties={"margin": 6, "spacing": 2})
        super().__init__(column_widget)

        self.document_controller = document_controller

        self.__state_controller = CameraControlStateController(camera_hardware_source, document_controller.queue_task, document_controller.document_model)

        self.__delegate: typing.Optional[CameraPanelDelegate] = None

        camera_panel_delegate_type = camera_hardware_source.features.get("camera_panel_delegate_type")
        for component in Registry.get_components_by_type("camera_panel_delegate"):
            if component.camera_panel_delegate_type == camera_panel_delegate_type:
                self.__delegate = component

        self.__shift_click_state: typing.Optional[str] = None

        self.__changes_blocked = False

        ui = document_controller.ui

        self.__key_pressed_event_listener = DisplayPanel.DisplayPanelManager().key_pressed_event.listen(self.image_panel_key_pressed)
        self.__key_released_event_listener = DisplayPanel.DisplayPanelManager().key_released_event.listen(self.image_panel_key_released)
        self.__image_display_mouse_pressed_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_pressed_event.listen(self.image_panel_mouse_pressed)
        self.__image_display_mouse_released_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_released_event.listen(self.image_panel_mouse_released)
        self.__mouse_pressed = False

        help_widget = None
        if self.__delegate and self.__delegate.has_feature("help"):

            def help_button_clicked() -> None:
                api_broker = PlugInManager.APIBroker()
                if self.__delegate:
                    self.__delegate.open_help(api_broker=api_broker)

            help_icon_24_png = pkgutil.get_data(__name__, "resources/help_icon_24.png")
            assert help_icon_24_png is not None
            help_button = CanvasItem.BitmapButtonCanvasItem(CanvasItem.load_rgba_data_from_bytes(help_icon_24_png, "png"))
            help_button.on_button_clicked = help_button_clicked
            help_widget = ui.create_canvas_widget(properties={"height": 24, "width": 24})
            help_widget.canvas_item.add_canvas_item(help_button)

        open_controls_widget = None
        self.__configuration_dialog_close_listener = None
        if not self.__delegate or self.__delegate.has_feature("configuration"):

            def configuration_button_clicked() -> None:
                api_broker = PlugInManager.APIBroker()
                if self.__delegate:
                    if self.__configuration_dialog_close_listener:
                        return
                    # if not already open, see if delegate wants to open it via a ui handler.
                    ui_handler = self.__delegate.get_configuration_ui_handler(api_broker=api_broker,
                                                                              event_loop=document_controller.event_loop,
                                                                              hardware_source_id=camera_hardware_source.hardware_source_id,
                                                                              camera_device=camera_hardware_source.camera,
                                                                              camera_settings=camera_hardware_source.camera_settings)
                    if ui_handler:
                        dialog = Dialog.ActionDialog(ui, camera_hardware_source.display_name)
                        dialog.content.add(Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, ui_handler))

                        def handle_window_close(window: Window.Window) -> None:
                            self.__configuration_dialog_close_listener = None

                        self.__configuration_dialog_close_listener = dialog._window_close_event.listen(handle_window_close)
                        dialog.show()
                        return
                    # fall through means there is no declarative configuration dialog
                    if self.__delegate.open_configuration(api_broker=api_broker,
                                                          hardware_source_id=camera_hardware_source.hardware_source_id,
                                                          camera_device=camera_hardware_source.camera,
                                                          camera_settings=camera_hardware_source.camera_settings):
                        return
                # fall through: no ui handler or direct handler
                self.__state_controller.handle_settings_button_clicked(api_broker)

            sliders_icon_24_png = pkgutil.get_data(__name__, "resources/sliders_icon_24.png")
            assert sliders_icon_24_png is not None
            open_controls_button = CanvasItem.BitmapButtonCanvasItem(CanvasItem.load_rgba_data_from_bytes(sliders_icon_24_png, "png"))
            open_controls_button.on_button_clicked = configuration_button_clicked
            open_controls_widget = ui.create_canvas_widget(properties={"height": 24, "width": 24})
            open_controls_widget.canvas_item.add_canvas_item(open_controls_button)

        camera_current_label = ui.create_label_widget()
        profile_label = ui.create_label_widget(_("Mode: "), properties={"margin": 4})
        profile_combo = ui.create_combo_box_widget(properties={"min-width": 72})
        play_state_label = ui.create_label_widget()
        play_button = ui.create_push_button_widget(_("Play"))
        play_button.on_clicked = self.__state_controller.handle_play_pause_clicked
        abort_button = ui.create_push_button_widget(_("Abort"))
        abort_button.on_clicked = self.__state_controller.handle_abort_clicked

        document_model = self.document_controller.document_model
        data_item_reference = document_model.get_data_item_reference(camera_hardware_source.hardware_source_id)
        data_item_thumbnail_source = DataItemThumbnailWidget.DataItemReferenceThumbnailSource(ui, document_model, data_item_reference)
        thumbnail_widget = DataItemThumbnailWidget.DataItemThumbnailWidget(ui, data_item_thumbnail_source, Geometry.IntSize(width=48, height=48))

        def thumbnail_widget_drag(mime_data: UserInterface.MimeData, thumbnail: typing.Optional[DrawingContext.RGBA32Type], hot_spot_x: int, hot_spot_y: int) -> None:
            self.drag(mime_data, thumbnail, hot_spot_x, hot_spot_y)

        thumbnail_widget.on_drag = thumbnail_widget_drag

        button_row1 = ui.create_row_widget(properties={"spacing": 2})
        button_row1.add(profile_label)
        button_row1.add(profile_combo)
        button_row1.add_stretch()
        if help_widget:
            button_row1.add(help_widget)
        if open_controls_widget:
            button_row1.add(open_controls_widget)

        def binning_combo_text_changed(text: str) -> None:
            if not self.__changes_blocked:
                self.__state_controller.handle_binning_changed(text)
                binning_combo.request_refocus()

        binning_combo = ui.create_combo_box_widget(properties={"min-width": 72})
        binning_combo.on_current_text_changed = binning_combo_text_changed

        def handle_exposure_changed(text: str) -> None:
            self.__state_controller.handle_exposure_changed(float(text) * math.pow(10, math.trunc(camera_hardware_source.exposure_precision / 3) * 3))
            exposure_field.request_refocus()

        def handle_decrease_exposure() -> None:
            self.__state_controller.handle_decrease_exposure()
            exposure_field.request_refocus()

        def handle_increase_exposure() -> None:
            self.__state_controller.handle_increase_exposure()
            exposure_field.request_refocus()

        exposure_field = ui.create_line_edit_widget(properties={"width": 72})
        exposure_field.on_editing_finished = handle_exposure_changed

        parameters_group1 = ui.create_row_widget()
        parameters_group2 = ui.create_row_widget()

        binning_row = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        binning_row.add(ui.create_label_widget(_("Binning")))
        binning_row.add_spacing(4)
        binning_row.add(binning_combo)
        parameters_group2.add(binning_row)
        parameters_group2.add_stretch()

        parameters_row2 = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        colx = ui.create_column_widget()
        colx.add_spacing(2)
        units = exposure_units[camera_hardware_source.exposure_precision]
        label = _("Time")
        colx.add(ui.create_label_widget(f"{label} ({units})"))
        colx.add_stretch()
        parameters_row2.add(colx)
        parameters_row2.add_spacing(4)
        group = ui.create_row_widget()
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        decrease_button = CharButtonCanvasItem("F")
        canvas_widget.canvas_item.add_canvas_item(decrease_button)
        group.add(canvas_widget)
        group.add(exposure_field)
        canvas_widget = ui.create_canvas_widget(properties={"height": 21, "width": 18})
        increase_button = CharButtonCanvasItem("S")
        canvas_widget.canvas_item.add_canvas_item(increase_button)
        group.add(canvas_widget)
        colx = ui.create_column_widget()
        colx.add(group)
        colx.add_stretch()
        parameters_row2.add(colx)
        parameters_group1.add(parameters_row2)
        parameters_group1.add_stretch()
        parameters_group1.add(camera_current_label)

        decrease_button.on_button_clicked = handle_decrease_exposure
        increase_button.on_button_clicked = handle_increase_exposure

        status_row = ui.create_row_widget(properties={"spacing": 2})
        status_row.add(play_state_label)
        status_row.add_stretch()

        button_column = ui.create_column_widget()
        button_column.add(play_button)
        button_column.add(abort_button)

        thumbnail_column = ui.create_column_widget()
        thumbnail_column.add(thumbnail_widget)
        thumbnail_column.add_stretch()

        button_row = ui.create_row_widget()
        button_row.add(button_column)
        button_row.add_stretch()
        button_row.add(thumbnail_column)

        column_widget.add(button_row1)
        column_widget.add(parameters_group1)
        column_widget.add(parameters_group2)
        column_widget.add(status_row)
        column_widget.add(button_row)
        column_widget.add_stretch()

        def profile_combo_text_changed(text: str) -> None:
            if not self.__changes_blocked:
                self.__state_controller.handle_change_profile(text)
                profile_combo.request_refocus()

        profile_combo.on_current_text_changed = profile_combo_text_changed

        def binning_values_changed(binning_values: typing.Sequence[int]) -> None:
            binning_combo.items = [str(binning_value) for binning_value in binning_values]

        def profiles_changed(items: typing.Sequence[str]) -> None:
            profile_combo.items = list(items)

        def change_profile_combo(profile_label: str) -> None:
            blocked = self.__changes_blocked
            self.__changes_blocked = True
            try:
                profile_combo.current_text = profile_label
                profile_combo.request_refocus()
            finally:
                self.__changes_blocked = blocked

        # thread safe
        def profile_changed(profile_label: str) -> None:
            # the current_text must be set on ui thread
            self.document_controller.queue_task(functools.partial(change_profile_combo, profile_label))

        def frame_parameters_changed(frame_parameters: camera_base.CameraFrameParameters) -> None:
            blocked = self.__changes_blocked
            self.__changes_blocked = True
            try:
                exposure_text = make_exposure_str(float(frame_parameters.exposure_ms) / 1000, camera_hardware_source.exposure_precision)
                exposure_field.text = exposure_text
                if exposure_field.focused:
                    exposure_field.request_refocus()
                binning_combo.current_text = str(frame_parameters.binning)
            finally:
                self.__changes_blocked = blocked

        def play_button_state_changed(enabled: bool, play_button_state: str) -> None:
            play_button_text = { "play": _("Play"), "pause": _("Pause") }
            play_button.enabled = enabled
            play_button.text = play_button_text[play_button_state]

        def abort_button_state_changed(visible: bool, enabled: bool) -> None:
            # abort_button.visible = visible
            abort_button.enabled = enabled

        def acquisition_state_changed(key: str) -> None:
            # this may be called on a thread. create an async method (guaranteed to run on the main thread)
            # and add it to the window event loop.
            async def update_acquisition_state_label(acquisition_state: typing.Optional[str]) -> None:
                acquisition_state = acquisition_state or "stopped"
                play_state_label.text = map_channel_state_to_text[acquisition_state]

            self.document_controller.event_loop.create_task(update_acquisition_state_label(self.__state_controller.acquisition_state_model.value))

        def camera_current_changed(camera_current: typing.Optional[float]) -> None:
            if camera_current:
                camera_current_int = int(camera_current * 1e12) if math.isfinite(camera_current) else 0
                camera_current_label.text = str(camera_current_int) + _("pA")
                camera_current_label.text_color = "black"
            else:
                camera_current_label.text_color = "gray"

        def log_messages(messages: typing.List[str], data_elements: typing.List[typing.Dict[str, typing.Any]]) -> None:
            while len(messages) > 0:
                message = messages.pop(0)
                logging.info(message)
            while len(data_elements) > 0:
                data_element = data_elements.pop(0)
                document_controller.add_data_element(data_element)

        self.__state_controller.on_display_name_changed = None
        self.__state_controller.on_binning_values_changed = binning_values_changed
        self.__state_controller.on_profiles_changed = profiles_changed
        self.__state_controller.on_profile_changed = profile_changed
        self.__state_controller.on_frame_parameters_changed = frame_parameters_changed
        self.__state_controller.on_play_button_state_changed = play_button_state_changed
        self.__state_controller.on_abort_button_state_changed = abort_button_state_changed
        self.__state_controller.on_camera_current_changed = camera_current_changed
        self.__state_controller.on_log_messages = log_messages

        self.__acquisition_state_changed_listener = self.__state_controller.acquisition_state_model.property_changed_event.listen(acquisition_state_changed)

        self.__state_controller.initialize_state()

        acquisition_state_changed("value")

    # HACK: this is used to dump log messages to Swift.
    def periodic(self) -> None:
        self.__state_controller.handle_periodic()
        super().periodic()

    def close(self) -> None:
        self.__configuration_dialog_close_listener = None
        self.__key_pressed_event_listener.close()
        self.__key_pressed_event_listener = typing.cast(typing.Any, None)
        self.__key_released_event_listener.close()
        self.__key_released_event_listener = typing.cast(typing.Any, None)
        self.__image_display_mouse_pressed_event_listener.close()
        self.__image_display_mouse_pressed_event_listener = typing.cast(typing.Any, None)
        self.__image_display_mouse_released_event_listener.close()
        self.__image_display_mouse_released_event_listener = typing.cast(typing.Any, None)
        self.__state_controller.close()
        self.__state_controller = typing.cast(typing.Any, None)
        self.__acquisition_state_changed_listener = typing.cast(typing.Any, None)
        super().close()

    # this gets called from the DisplayPanelManager. pass on the message to the state controller.
    # must be called on ui thread
    def image_panel_mouse_pressed(self, display_panel: DisplayPanel.DisplayPanel, display_item: DisplayItem.DisplayItem, image_position: Geometry.FloatPoint, modifiers: CanvasItem.KeyboardModifiers) -> bool:
        data_item = display_panel.data_item if display_panel else None
        hardware_source_id = data_item.metadata.get("hardware_source", dict()).get("hardware_source_id") if data_item else str()
        logger = logging.getLogger("camera_control_ui")
        logger.propagate = False  # do not send messages to root logger
        if not logger.handlers:
            logger.addHandler(logging.handlers.BufferingHandler(4))
        if data_item and hardware_source_id and self.__shift_click_state == "shift":
            mouse_position = image_position
            camera_shape = data_item.dimensional_shape
            self.__mouse_pressed = self.__state_controller.handle_shift_click(hardware_source_id, mouse_position, typing.cast(DataAndMetadata.Shape2dType, camera_shape), logger)
            logger_buffer = typing.cast(logging.handlers.BufferingHandler, logger.handlers[0])
            for record in logger_buffer.buffer:
                display_panel.document_controller.display_log_record(record)
            logger_buffer.flush()
            return self.__mouse_pressed
        if data_item and hardware_source_id and self.__shift_click_state == "tilt":
            mouse_position = image_position
            camera_shape = data_item.dimensional_shape
            self.__mouse_pressed = self.__state_controller.handle_tilt_click(hardware_source_id, mouse_position, typing.cast(DataAndMetadata.Shape2dType, camera_shape), logger)
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
        elif key.text.lower() == "t":
            self.__shift_click_state = "tilt"
        else:
            self.__shift_click_state = None
        return False

    def image_panel_key_released(self, display_panel: DisplayPanel.DisplayPanel, key: UserInterface.Key) -> bool:
        self.__shift_click_state = None
        return False

    @property
    def state_controller(self) -> CameraControlStateController:
        return self.__state_controller


class CameraControlPanel(Panel.Panel):

    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> None:
        super().__init__(document_controller, panel_id, "camera-control-panel")
        ui = document_controller.ui
        self.widget = ui.create_column_widget()
        self.hardware_source_id = properties["hardware_source_id"]
        hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(self.hardware_source_id)
        if hardware_source:
            camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, hardware_source)
            camera_control_widget = CameraControlWidget(self.document_controller, camera_hardware_source)
            self.widget.add(camera_control_widget)
            self.widget.add_spacing(12)
            self.widget.add_stretch()


def create_camera_panel(document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> Panel.Panel:
    """Create a custom camera panel.

    The camera panel type is specified in the 'camera_panel_type' key in the properties dict.

    The camera panel type must match a the 'camera_panel_type' of a camera panel factory in the Registry.

    The matching camera panel factory must return a ui_handler for the panel which is used to produce the UI.
    """
    camera_panel_type = properties.get("camera_panel_type")
    for component in Registry.get_components_by_type("camera_panel"):
        if component.camera_panel_type == camera_panel_type:
            hardware_source_id = properties["hardware_source_id"]
            hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
            camera_device = getattr(hardware_source, "camera", None)
            camera_settings = getattr(hardware_source, "camera_settings", None)
            ui_handler = component.get_ui_handler(api_broker=PlugInManager.APIBroker(), event_loop=document_controller.event_loop, hardware_source_id=hardware_source_id, camera_device=camera_device, camera_settings=camera_settings)
            panel = Panel.Panel(document_controller, panel_id, "camera-control-panel")
            panel.widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, ui_handler)
            return panel
    raise Exception(f"Unable to create camera panel: {panel_id}")


class CameraDisplayPanelController:
    """
        Represents a controller for the content of an image panel.
    """

    type = "camera-live"

    def __init__(self, display_panel: DisplayPanel.DisplayPanel, hardware_source_id: str, show_processed_data: bool) -> None:
        assert hardware_source_id is not None
        hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
        camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, hardware_source)
        self.type = CameraDisplayPanelController.type

        self.__hardware_source_id = hardware_source_id

        # configure the hardware source state controller
        self.__state_controller = CameraControlStateController(camera_hardware_source, display_panel.document_controller.queue_task, display_panel.document_controller.document_model)

        # configure the user interface
        self.__display_name = str()
        self.__play_button_enabled = False
        self.__play_button_play_button_state = "play"
        self.__abort_button_visible = False
        self.__abort_button_enabled = False
        self.__display_panel = display_panel
        self.__display_panel.header_canvas_item.end_header_color = "#98FB98"
        self.__playback_controls_composition = CanvasItem.CanvasItemComposition()
        self.__playback_controls_composition.layout = CanvasItem.CanvasItemLayout()
        self.__playback_controls_composition.update_sizing(self.__playback_controls_composition.sizing.with_fixed_height(30))
        playback_controls_row = CanvasItem.CanvasItemComposition()
        playback_controls_row.layout = CanvasItem.CanvasItemRowLayout()
        play_button_canvas_item = CanvasItem.TextButtonCanvasItem()
        play_button_canvas_item.border_enabled = False
        abort_button_canvas_item = CanvasItem.TextButtonCanvasItem()
        abort_button_canvas_item.border_enabled = False
        status_text_canvas_item = CanvasItem.StaticTextCanvasItem(str())
        hardware_source_display_name_canvas_item = CanvasItem.StaticTextCanvasItem(str())
        playback_controls_row.add_canvas_item(play_button_canvas_item)
        playback_controls_row.add_canvas_item(abort_button_canvas_item)
        playback_controls_row.add_canvas_item(status_text_canvas_item)
        playback_controls_row.add_stretch()
        capture_button = CanvasItem.TextButtonCanvasItem(_("Capture"))
        capture_button.border_enabled = False
        playback_controls_row.add_canvas_item(capture_button)
        self.__show_processed_checkbox = None
        if self.__state_controller.has_processed_data:
            self.__show_processed_checkbox = CanvasItem.CheckBoxCanvasItem()
            self.__show_processed_checkbox.check_state = "checked" if show_processed_data else "unchecked"
            self.__state_controller.use_processed_data = show_processed_data
            playback_controls_row.add_canvas_item(self.__show_processed_checkbox)
        playback_controls_row.add_canvas_item(hardware_source_display_name_canvas_item)
        self.__playback_controls_composition.add_canvas_item(CanvasItem.BackgroundCanvasItem("#98FB98"))
        self.__playback_controls_composition.add_canvas_item(playback_controls_row)
        self.__display_panel.footer_canvas_item.insert_canvas_item(0, self.__playback_controls_composition)

        def update_display_name() -> None:
            new_text = self.__display_name
            if hardware_source_display_name_canvas_item.text != new_text:
                hardware_source_display_name_canvas_item.text = new_text
                hardware_source_display_name_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_play_button() -> None:
            map_play_button_state_to_text = {"play": _("Play"), "pause": _("Pause")}
            play_button_text = map_play_button_state_to_text[self.__play_button_play_button_state]
            new_enabled = self.__play_button_enabled
            new_text = play_button_text
            if play_button_canvas_item.enabled != new_enabled or play_button_canvas_item.text != new_text:
                play_button_canvas_item.enabled = new_enabled
                play_button_canvas_item.text = new_text
                play_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_abort_button() -> None:
            abort_button_visible = self.__abort_button_visible
            abort_button_enabled = self.__abort_button_enabled
            new_text = _("Abort") if abort_button_visible else str()
            if abort_button_canvas_item.enabled != abort_button_enabled or abort_button_canvas_item.text != new_text:
                abort_button_canvas_item.text = new_text
                abort_button_canvas_item.enabled = abort_button_enabled
                abort_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def acquisition_state_changed(key: str) -> None:
            # this may be called on a thread. create an async method (guaranteed to run on the main thread)
            # and add it to the window event loop.
            async def update_acquisition_state_label(acquisition_state: typing.Optional[str]) -> None:
                acquisition_state = acquisition_state or "stopped"
                status_text_canvas_item.text = map_channel_state_to_text[acquisition_state]
                status_text_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

            self.__display_panel.document_controller.event_loop.create_task(update_acquisition_state_label(self.__state_controller.acquisition_state_model.value))

        def display_name_changed(display_name: str) -> None:
            self.__display_name = display_name
            update_display_name()

        def play_button_state_changed(enabled: bool, play_button_state: str) -> None:
            self.__play_button_enabled = enabled
            self.__play_button_play_button_state = play_button_state
            update_play_button()

        def abort_button_state_changed(visible: bool, enabled: bool) -> None:
            self.__abort_button_visible = visible
            self.__abort_button_enabled = enabled
            update_abort_button()

        def update_capture_button(visible: bool, enabled: bool) -> None:
            if visible:
                capture_button.enabled = enabled
                capture_button.text = _("Capture")
                capture_button.size_to_content(display_panel.image_panel_get_font_metrics)
            else:
                capture_button.enabled = False
                capture_button.text = str()
                capture_button.size_to_content(display_panel.image_panel_get_font_metrics)

        def show_processed_checkbox_changed(check_state: str) -> None:
            if check_state == "checked":
                display_panel.set_data_item_reference(self.__state_controller.processed_data_item_reference)
                self.__state_controller.use_processed_data = True  # for capture
            else:
                display_panel.set_data_item_reference(self.__state_controller.data_item_reference)
                self.__state_controller.use_processed_data = False  # for capture

        def display_new_data_item(data_item: DataItem.DataItem) -> None:
            result_display_panel = display_panel.document_controller.next_result_display_panel()
            if result_display_panel:
                result_display_panel.set_display_panel_data_item(data_item)
                result_display_panel.request_focus()

        if self.__show_processed_checkbox:
            self.__show_processed_checkbox.on_check_state_changed = show_processed_checkbox_changed

        self.__state_controller.on_display_name_changed = display_name_changed
        self.__state_controller.on_binning_values_changed = None
        self.__state_controller.on_play_button_state_changed = play_button_state_changed
        self.__state_controller.on_abort_button_state_changed = abort_button_state_changed
        self.__state_controller.on_capture_button_state_changed = update_capture_button
        self.__state_controller.on_display_new_data_item = display_new_data_item

        self.__acquisition_state_changed_listener = self.__state_controller.acquisition_state_model.property_changed_event.listen(acquisition_state_changed)

        play_button_canvas_item.on_button_clicked = self.__state_controller.handle_play_pause_clicked
        abort_button_canvas_item.on_button_clicked = self.__state_controller.handle_abort_clicked
        capture_button.on_button_clicked = self.__state_controller.handle_capture_clicked

        self.__state_controller.initialize_state()

        acquisition_state_changed("value")

        checkstate = self.__show_processed_checkbox.check_state if self.__show_processed_checkbox else "unchecked"

        show_processed_checkbox_changed(checkstate)

    def close(self) -> None:
        self.__display_panel.footer_canvas_item.remove_canvas_item(self.__playback_controls_composition)
        self.__display_panel = typing.cast(typing.Any, None)
        self.__state_controller.close()
        self.__state_controller = typing.cast(typing.Any, None)
        self.__acquisition_state_changed_listener = typing.cast(typing.Any, None)

    def save(self, d: typing.MutableMapping[str, typing.Any]) -> None:
        d["hardware_source_id"] = self.__hardware_source_id
        if self.__show_processed_checkbox:
            d["show_processed_data"] = self.__show_processed_checkbox.check_state == "checked"

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
    def hardware_source_id(self) -> str:
        return self.__hardware_source_id


hardware_source_added_event_listener: typing.Optional[Event.EventListener] = None
hardware_source_removed_event_listener: typing.Optional[Event.EventListener] = None


_component_registered_listener: typing.Optional[Event.EventListener] = None
_component_unregistered_listener: typing.Optional[Event.EventListener] = None


def run() -> None:
    global hardware_source_added_event_listener, hardware_source_removed_event_listener
    camera_control_panels = dict()

    def register_camera_panel(hardware_source: HardwareSource.HardwareSource) -> None:
        """Called when a hardware source is added to the hardware source manager."""

        # check to see if we handle this hardware source.
        is_camera = hardware_source.features.get("is_camera", False)
        if is_camera:

            panel_id = "camera-control-panel-" + hardware_source.hardware_source_id
            name = hardware_source.display_name + " " + _("Camera Control")
            camera_control_panels[hardware_source.hardware_source_id] = panel_id

            class CameraDisplayPanelControllerFactory:
                def __init__(self) -> None:
                    self.priority = 2

                def build_menu(self, display_type_menu: UserInterface.Menu, selected_display_panel: typing.Optional[DisplayPanel.DisplayPanel]) -> typing.Sequence[UserInterface.MenuAction]:
                    # return a list of actions that have been added to the menu.
                    assert isinstance(hardware_source, camera_base.CameraHardwareSource)
                    def switch_to_live_controller(hardware_source: camera_base.CameraHardwareSource) -> None:
                        d = {"type": "image", "controller_type": CameraDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                        if selected_display_panel:
                            selected_display_panel.change_display_panel_content(d)

                    action = display_type_menu.add_menu_item(hardware_source.display_name, functools.partial(switch_to_live_controller, hardware_source))
                    display_panel_controller = selected_display_panel.display_panel_controller if selected_display_panel else None
                    action.checked = isinstance(display_panel_controller, CameraDisplayPanelController) and display_panel_controller.hardware_source_id == hardware_source.hardware_source_id
                    return [action]

                def make_new(self, controller_type: str, display_panel: DisplayPanel.DisplayPanel, d: Persistence.PersistentDictType) -> typing.Optional[CameraDisplayPanelController]:
                    # make a new display panel controller, typically called to restore contents of a display panel.
                    # controller_type will match the type property of the display panel controller when it was saved.
                    # d is the dictionary that is saved when the display panel controller closes.
                    hardware_source_id = d.get("hardware_source_id")
                    show_processed_data = d.get("show_processed_data", False)
                    if controller_type == CameraDisplayPanelController.type and hardware_source_id == hardware_source.hardware_source_id:
                        return CameraDisplayPanelController(display_panel, hardware_source_id, show_processed_data)
                    return None

                def match(self, document_model: DocumentModel.DocumentModel, data_item: DataItem.DataItem) -> typing.Optional[Persistence.PersistentDictType]:
                    if HardwareSource.matches_hardware_source(hardware_source.hardware_source_id, None, document_model, data_item):
                        return {"controller_type": CameraDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                    return None

            DisplayPanel.DisplayPanelManager().register_display_panel_controller_factory("camera-live-" + hardware_source.hardware_source_id, CameraDisplayPanelControllerFactory())

            panel_properties = {"hardware_source_id": hardware_source.hardware_source_id}

            camera_panel_type = hardware_source.features.get("camera_panel_type")
            if not camera_panel_type:
                Workspace.WorkspaceManager().register_panel(CameraControlPanel, panel_id, name, ["left", "right"], "left", panel_properties)
            else:
                panel_properties["camera_panel_type"] = camera_panel_type
                Workspace.WorkspaceManager().register_panel(typing.cast(typing.Type[typing.Any], create_camera_panel), panel_id, name, ["left", "right"], "left", panel_properties)

    def unregister_camera_panel(hardware_source: HardwareSource.HardwareSource) -> None:
        """Called when a hardware source is removed from the hardware source manager."""
        is_camera = hardware_source.features.get("is_camera", False)
        if is_camera:
            DisplayPanel.DisplayPanelManager().unregister_display_panel_controller_factory("camera-live-" + hardware_source.hardware_source_id)
            panel_id = camera_control_panels.get(hardware_source.hardware_source_id)
            if panel_id:
                Workspace.WorkspaceManager().unregister_panel(panel_id)

    hardware_source_added_event_listener = HardwareSource.HardwareSourceManager().hardware_source_added_event.listen(register_camera_panel)
    hardware_source_removed_event_listener = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(unregister_camera_panel)
    for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
        register_camera_panel(hardware_source)
