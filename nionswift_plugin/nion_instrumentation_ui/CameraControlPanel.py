# standard libraries
import functools
import gettext
import logging
import numpy
import sys
import time

# local libraries
from nion.swift import DataItemThumbnailWidget
from nion.swift import Decorators
from nion.swift import DisplayPanel
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem
from nion.swift.model import HardwareSource
from nion.swift.model import PlugInManager
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import Widgets
from nion.utils import Geometry
from nion.utils import Registry

_ = gettext.gettext


class CameraControlStateController:
    """
    Track the state of a camera controller, as it relates to the UI. This object does not hold any state itself.

    Camera controller should support the following API:
        (acquisition)
            (event) data_item_states_changed_event(data_item_states)
            (event) acquisition_state_changed_event(is_acquiring)
            (read-only property) hardware_source_id
            (read-only property) is_playing
            (read-only property) display_name
            (read-only property) features
            (method) start_playing()
            (method) stop_playing()
            (method) abort_playing()
        (event) profile_changed_event(profile_index)
        (event) frame_parameters_changed_event(profile_index, frame_parameters)
        (event) log_messages_event(messages, data_elements)
        (read-only property) selected_profile_index: return current profile index
        (method) set_selected_profile_index(profile_index): change the profile index
        (method) get_frame_parameters(profile_index)
        (method) set_frame_parameters(profile_index, frame_parameters)
        (method) set_current_frame_parameters(frame_parameters)
        (method) get_current_frame_parameters()
        (method) shift_click(mouse_position, camera_shape)
        (method) open_configuration_interface(api_broker)
        (method) open_monitor()
        (method) periodic()

    Clients can query:
        (property) has_processed_data

    Clients should call:
        handle_change_profile(profile_label)
        handle_play_pause_clicked(workspace_controller)
        handle_abort_clicked()
        handle_shift_click(hardware_source_id, mouse_position, image_dimensions)
        handle_tilt_click(hardware_source_id, mouse_position, image_dimensions)
        handle_binning_changed(binning_str)
        handle_exposure_changed(exposure_str)
        handle_increase_exposure()
        handle_decrease_exposure()
        handle_settings_button_clicked(api_broker)
        handle_monitor_button_clicked()
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
        on_monitor_button_state_changed(visible, enabled)
        on_capture_button_state_changed(visible, enabled)
        on_display_data_item_changed(data_item)
        on_display_new_data_item(data_item)
        on_processed_data_item_changed(data_item)
        on_log_messages(messages, data_elements)
        (thread) on_data_item_states_changed(data_item_states)
    """

    def __init__(self, camera_hardware_source, queue_task, document_model):
        self.__hardware_source = camera_hardware_source
        self.__is_eels_camera = self.__hardware_source and self.__hardware_source.features.get("is_eels_camera", False)
        self.use_processed_data = False
        self.queue_task = queue_task
        self.__document_model = document_model
        self.__profile_changed_event_listener = None
        self.__frame_parameters_changed_event_listener = None
        self.__data_item_states_changed_event_listener = None
        self.__acquisition_state_changed_event_listener = None
        self.__log_messages_event_listener = None
        self.on_display_name_changed = None
        self.on_binning_values_changed = None
        self.on_profiles_changed = None
        self.on_profile_changed = None
        self.on_frame_parameters_changed = None
        self.on_play_button_state_changed = None
        self.on_abort_button_state_changed = None
        self.on_monitor_button_state_changed = None
        self.on_data_item_states_changed = None
        self.on_capture_button_state_changed = None
        self.on_display_data_item_changed = None
        self.on_display_new_data_item = None
        self.on_processed_data_item_changed = None
        self.on_camera_current_changed = None
        self.on_log_messages = None

        self.__captured_xdatas_available_event = None

        self.__camera_current = None
        self.__last_camera_current_time = 0
        self.__xdatas_available_event = self.__hardware_source.xdatas_available_event.listen(self.__receive_new_xdatas)

        # this function is threadsafe
        # it queues the threadsafe call to the UI thread, checking to make sure the
        # hardware source wasn't closed before being called (mainly to make tests run).
        def handle_data_item_changed():
            def update_display_data_item():
                if self.__hardware_source:
                    self.__update_display_data_item()
            self.queue_task(update_display_data_item)

        self.__data_item = None
        data_item_reference = document_model.get_data_item_reference(self.__hardware_source.hardware_source_id)
        self.__data_item_changed_event_listener = data_item_reference.data_item_changed_event.listen(handle_data_item_changed)

        self.__eels_data_item = None
        eels_data_item_reference = document_model.get_data_item_reference(document_model.make_data_item_reference_key(self.__hardware_source.hardware_source_id, "summed"))
        self.__eels_data_item_changed_event_listener = eels_data_item_reference.data_item_changed_event.listen(handle_data_item_changed)

    def close(self):
        if self.__captured_xdatas_available_event:
            self.__captured_xdatas_available_event.close()
            self.__captured_xdatas_available_event = None
        if self.__profile_changed_event_listener:
            self.__profile_changed_event_listener.close()
            self.__profile_changed_event_listener = None
        if self.__frame_parameters_changed_event_listener:
            self.__frame_parameters_changed_event_listener.close()
            self.__frame_parameters_changed_event_listener = None
        if self.__acquisition_state_changed_event_listener:
            self.__acquisition_state_changed_event_listener.close()
            self.__acquisition_state_changed_event_listener = None
        if self.__data_item_states_changed_event_listener:
            self.__data_item_states_changed_event_listener.close()
            self.__data_item_states_changed_event_listener = None
        if self.__log_messages_event_listener:
            self.__log_messages_event_listener.close()
            self.__log_messages_event_listener = None
        self.__data_item_changed_event_listener.close()
        self.__data_item_changed_event_listener = None
        self.__eels_data_item_changed_event_listener.close()
        self.__eels_data_item_changed_event_listener = None
        self.on_display_name_changed = None
        self.on_binning_values_changed = None
        self.on_profiles_changed = None
        self.on_profile_changed = None
        self.on_frame_parameters_changed = None
        self.on_play_button_state_changed = None
        self.on_abort_button_state_changed = None
        self.on_monitor_button_state_changed = None
        self.on_data_item_states_changed = None
        self.on_capture_button_state_changed = None
        self.on_display_data_item_changed = None
        self.on_display_new_data_item = None
        self.on_processed_data_item_changed = None
        self.on_log_messages = None
        self.__hardware_source = None

    def __update_play_button_state(self):
        enabled = self.__hardware_source is not None
        if self.on_play_button_state_changed:
            self.on_play_button_state_changed(enabled, "pause" if self.is_playing else "play")
        if enabled and not self.is_playing and callable(self.on_camera_current_changed):
            self.on_camera_current_changed(None)

    def __update_abort_button_state(self):
        if self.on_abort_button_state_changed:
            self.on_abort_button_state_changed(self.is_playing, self.is_playing)
        if self.on_capture_button_state_changed:
            self.on_capture_button_state_changed(self.is_playing, not self.__captured_xdatas_available_event)

    def __update_buttons(self):
        self.__update_play_button_state()
        self.__update_abort_button_state()

    def __update_profile_state(self, profile_label):
        if self.on_profile_changed:
            self.on_profile_changed(profile_label)

    def __update_frame_parameters(self, profile_index, frame_parameters):
        if self.on_frame_parameters_changed:
            if profile_index == self.__hardware_source.selected_profile_index:
                self.on_frame_parameters_changed(frame_parameters)

    # received from the camera controller when the profile changes.
    # thread safe
    def __update_profile_index(self, profile_index):
        for index, name in enumerate(self.__hardware_source.modes):
            if index == profile_index:
                def update_profile_state_and_frame_parameters(name):
                    if self.__hardware_source:  # check to see if close has been called.
                        self.__update_profile_state(name)
                        self.__update_frame_parameters(self.__hardware_source.selected_profile_index, self.__hardware_source.get_frame_parameters(profile_index))
                self.queue_task(functools.partial(update_profile_state_and_frame_parameters, name))

    @property
    def has_processed_data(self):
        return self.__is_eels_camera

    # not thread safe
    def __update_display_data_item(self):
        data_item_reference = self.__document_model.get_data_item_reference(self.__hardware_source.hardware_source_id)
        with data_item_reference.mutex:
            self.__data_item = data_item_reference.data_item
            if self.on_display_data_item_changed:
                self.on_display_data_item_changed(self.__data_item)
        eels_data_item_reference = self.__document_model.get_data_item_reference(self.__document_model.make_data_item_reference_key(self.__hardware_source.hardware_source_id, "summed"))
        with eels_data_item_reference.mutex:
            self.__eels_data_item = eels_data_item_reference.data_item
            if self.on_processed_data_item_changed:
                self.on_processed_data_item_changed(self.__eels_data_item)

    def __receive_new_xdatas(self, xdatas):
        current_time = time.time()
        if current_time - self.__last_camera_current_time > 5.0 and len(xdatas) > 0 and callable(self.on_camera_current_changed):
            xdata = xdatas[0]
            counts_per_electron = xdata.metadata.get("hardware_source", dict()).get("counts_per_electron")
            exposure = xdata.metadata.get("hardware_source", dict()).get("exposure")
            if xdata.intensity_calibration.units == "counts" and counts_per_electron and exposure:
                sum_counts = xdata.intensity_calibration.convert_to_calibrated_value(numpy.sum(xdatas[0].data))
                detector_current = sum_counts / exposure / counts_per_electron / 6.242e18
                if detector_current != self.__camera_current:
                    self.__camera_current = detector_current
                    def update_camera_current():
                        self.on_camera_current_changed(self.__camera_current)
                    self.queue_task(update_camera_current)
            self.__last_camera_current_time = current_time

    def initialize_state(self):
        """ Call this to initialize the state of the UI after everything has been connected. """
        if self.__hardware_source:
            self.__profile_changed_event_listener = self.__hardware_source.profile_changed_event.listen(self.__update_profile_index)
            self.__frame_parameters_changed_event_listener = self.__hardware_source.frame_parameters_changed_event.listen(self.__update_frame_parameters)
            self.__data_item_states_changed_event_listener = self.__hardware_source.data_item_states_changed_event.listen(self.__data_item_states_changed)
            self.__acquisition_state_changed_event_listener = self.__hardware_source.acquisition_state_changed_event.listen(self.__acquisition_state_changed)
            self.__log_messages_event_listener = self.__hardware_source.log_messages_event.listen(self.__log_messages)
        if self.on_display_name_changed:
            self.on_display_name_changed(self.display_name)
        if self.on_binning_values_changed:
            self.on_binning_values_changed(self.__hardware_source.binning_values)
        if self.on_monitor_button_state_changed:
            has_monitor = self.__hardware_source and self.__hardware_source.features.get("has_monitor", False)
            self.on_monitor_button_state_changed(has_monitor, has_monitor)
        self.__update_buttons()
        if self.on_profiles_changed:
            profile_items = self.__hardware_source.modes
            self.on_profiles_changed(profile_items)
            self.__update_profile_index(self.__hardware_source.selected_profile_index)
        if self.on_data_item_states_changed:
            self.on_data_item_states_changed(list())
        self.__update_display_data_item()

    # must be called on ui thread
    def handle_change_profile(self, profile_label):
        if profile_label in self.__hardware_source.modes:
            self.__hardware_source.set_selected_profile_index(self.__hardware_source.modes.index(profile_label))

    def handle_play_pause_clicked(self):
        """ Call this when the user clicks the play/pause button. """
        if self.__hardware_source:
            if self.is_playing:
                self.__hardware_source.stop_playing()
            else:
                self.__hardware_source.start_playing()

    def handle_abort_clicked(self):
        """ Call this when the user clicks the abort button. """
        if self.__hardware_source:
            self.__hardware_source.abort_playing()

    # must be called on ui thread
    def handle_settings_button_clicked(self, api_broker):
        if self.__hardware_source:
            self.__hardware_source.open_configuration_interface(api_broker)

    # must be called on ui thread
    def handle_monitor_button_clicked(self):
        if self.__hardware_source:
            self.__hardware_source.open_monitor()

    # must be called on ui thread
    def handle_shift_click(self, hardware_source_id, mouse_position, camera_shape):
        if hardware_source_id == self.__hardware_source.hardware_source_id:
            self.__hardware_source.shift_click(mouse_position, camera_shape)
            return True
        return False

    # must be called on ui thread
    def handle_tilt_click(self, hardware_source_id, mouse_position, camera_shape):
        if hardware_source_id == self.__hardware_source.hardware_source_id:
            self.__hardware_source.tilt_click(mouse_position, camera_shape)
            return True
        return False

    # must be called on ui thread
    def handle_binning_changed(self, binning_str):
        frame_parameters = self.__hardware_source.get_frame_parameters(self.__hardware_source.selected_profile_index)
        frame_parameters.binning = max(int(binning_str), 1)
        self.__hardware_source.set_frame_parameters(self.__hardware_source.selected_profile_index, frame_parameters)

    # must be called on ui thread
    def handle_exposure_changed(self, exposure):
        frame_parameters = self.__hardware_source.get_frame_parameters(self.__hardware_source.selected_profile_index)
        try:
            frame_parameters.exposure_ms = float(exposure)
        except ValueError:
            pass
        self.__hardware_source.set_frame_parameters(self.__hardware_source.selected_profile_index, frame_parameters)

    def handle_decrease_exposure(self):
        frame_parameters = self.__hardware_source.get_frame_parameters(self.__hardware_source.selected_profile_index)
        frame_parameters.exposure_ms = frame_parameters.exposure_ms * 0.5
        self.__hardware_source.set_frame_parameters(self.__hardware_source.selected_profile_index, frame_parameters)

    def handle_increase_exposure(self):
        frame_parameters = self.__hardware_source.get_frame_parameters(self.__hardware_source.selected_profile_index)
        frame_parameters.exposure_ms = frame_parameters.exposure_ms * 2.0
        self.__hardware_source.set_frame_parameters(self.__hardware_source.selected_profile_index, frame_parameters)

    def handle_capture_clicked(self):
        def capture_xdatas(xdatas):
            if self.__captured_xdatas_available_event:
                self.__captured_xdatas_available_event.close()
                self.__captured_xdatas_available_event = None
            for index, xdata in enumerate(xdatas):
                def add_data_item(data_item):
                    self.__document_model.append_data_item(data_item)
                    if self.on_display_new_data_item:
                        self.on_display_new_data_item(data_item)

                if index == (1 if self.use_processed_data else 0):
                    data_item = DataItem.new_data_item(xdata)
                    display_name = xdata.metadata.get("hardware_source", dict()).get("hardware_source_name")
                    display_name = display_name if display_name else _("Capture")
                    data_item.title = display_name
                    self.queue_task(functools.partial(add_data_item, data_item))
            self.queue_task(self.__update_buttons)

        self.__captured_xdatas_available_event = self.__hardware_source.xdatas_available_event.listen(capture_xdatas)
        self.__update_buttons()

    # must be called on ui thread
    def handle_periodic(self):
        if self.__hardware_source and getattr(self.__hardware_source, "periodic", None):
            self.__hardware_source.periodic()

    @property
    def is_playing(self):
        """ Returns whether the hardware source is playing or not. """
        return self.__hardware_source.is_playing if self.__hardware_source else False

    @property
    def display_name(self):
        """ Returns the display name for the hardware source. """
        return self.__hardware_source.display_name if self.__hardware_source else _("N/A")

    # this message comes from the data buffer. it will always be invoked on the UI thread.
    def __acquisition_state_changed(self, is_acquiring):
        if self.__captured_xdatas_available_event:
            self.__captured_xdatas_available_event.close()
            self.__captured_xdatas_available_event = None
        self.queue_task(self.__update_buttons)

    def __log_messages(self, messages, data_elements):
        if self.on_log_messages:
            self.on_log_messages(messages, data_elements)

    # this message comes from the hardware source. may be called from thread.
    def __data_item_states_changed(self, data_item_states):
        if self.on_data_item_states_changed:
            self.on_data_item_states_changed(data_item_states)


class IconCanvasItem(CanvasItem.TextButtonCanvasItem):

    def __init__(self, icon_id):
        super(IconCanvasItem, self).__init__()
        self.__icon_id = icon_id
        self.wants_mouse_events = True
        self.__mouse_inside = False
        self.__mouse_pressed = False
        self.fill_style = "rgb(128, 128, 128)"
        self.fill_style_pressed = "rgb(64, 64, 64)"
        self.fill_style_disabled = "rgb(192, 192, 192)"
        self.border_style = None
        self.border_style_pressed = None
        self.border_style_disabled = None
        self.stroke_style = "#FFF"
        self.stroke_width = 3.0
        self.on_button_clicked = None

    def close(self):
        self.on_button_clicked = None
        super(IconCanvasItem, self).close()

    def mouse_entered(self):
        self.__mouse_inside = True
        self.update()

    def mouse_exited(self):
        self.__mouse_inside = False
        self.update()

    def mouse_pressed(self, x, y, modifiers):
        self.__mouse_pressed = True
        self.update()

    def mouse_released(self, x, y, modifiers):
        self.__mouse_pressed = False
        self.update()

    def mouse_clicked(self, x, y, modifiers):
        if self.enabled:
            if self.on_button_clicked:
                self.on_button_clicked()
        return True

    def _repaint(self, drawing_context):
        with drawing_context.saver():
            import math
            canvas_size = self.canvas_size
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

    def __init__(self, char: str):
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
        self.on_button_clicked = None
        self.border_enabled = False

    def close(self):
        self.on_button_clicked = None
        super().close()

    def mouse_entered(self):
        self.__mouse_inside = True
        self.update()

    def mouse_exited(self):
        self.__mouse_inside = False
        self.update()

    def mouse_pressed(self, x, y, modifiers):
        self.__mouse_pressed = True
        self.update()

    def mouse_released(self, x, y, modifiers):
        self.__mouse_pressed = False
        self.update()

    def mouse_clicked(self, x, y, modifiers):
        if self.enabled:
            if callable(self.on_button_clicked):
                self.on_button_clicked()
        return True

    def _repaint(self, drawing_context):
        with drawing_context.saver():
            canvas_size = self.canvas_size
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


class CameraControlWidget(Widgets.CompositeWidgetBase):

    def __init__(self, document_controller, camera_controller):
        super().__init__(document_controller.ui.create_column_widget(properties={"margin": 6, "spacing": 2}))

        self.document_controller = document_controller

        self.__state_controller = CameraControlStateController(camera_controller, document_controller.queue_task, document_controller.document_model)

        self.__shift_click_state = None

        self.__changes_blocked = False

        ui = document_controller.ui

        self.__key_pressed_event_listener = DisplayPanel.DisplayPanelManager().key_pressed_event.listen(self.image_panel_key_pressed)
        self.__key_released_event_listener = DisplayPanel.DisplayPanelManager().key_released_event.listen(self.image_panel_key_released)
        self.__image_display_mouse_pressed_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_pressed_event.listen(self.image_panel_mouse_pressed)
        self.__image_display_mouse_released_event_listener = DisplayPanel.DisplayPanelManager().image_display_mouse_released_event.listen(self.image_panel_mouse_released)
        self.__mouse_pressed = False

        open_controls_button = CanvasItem.BitmapButtonCanvasItem(document_controller.ui.load_rgba_data_from_file(Decorators.relative_file(__file__, "resources/sliders_icon_24.png")))
        open_controls_widget = ui.create_canvas_widget(properties={"height": 24, "width": 24})
        open_controls_widget.canvas_item.add_canvas_item(open_controls_button)
        monitor_button = ui.create_push_button_widget(_("Monitor View..."))
        camera_current_label = ui.create_label_widget()
        profile_label = ui.create_label_widget(_("Mode: "), properties={"margin":4})
        profile_combo = ui.create_combo_box_widget(properties={"min-width":72})
        play_state_label = ui.create_label_widget()
        play_button = ui.create_push_button_widget(_("Play"))
        play_button.on_clicked = self.__state_controller.handle_play_pause_clicked
        abort_button = ui.create_push_button_widget(_("Abort"))
        abort_button.on_clicked = self.__state_controller.handle_abort_clicked

        document_model = self.document_controller.document_model
        data_item_reference = document_model.get_data_item_reference(camera_controller.hardware_source_id)
        data_item_thumbnail_source = DataItemThumbnailWidget.DataItemReferenceThumbnailSource(ui, data_item_reference)
        thumbnail_widget = DataItemThumbnailWidget.DataItemThumbnailWidget(ui, data_item_thumbnail_source, Geometry.IntSize(width=48, height=48))

        def thumbnail_widget_drag(mime_data, thumbnail, hot_spot_x, hot_spot_y):
            self.drag(mime_data, thumbnail, hot_spot_x, hot_spot_y)

        thumbnail_widget.on_drag = thumbnail_widget_drag

        button_row1 = ui.create_row_widget(properties={"spacing": 2})
        button_row1.add(profile_label)
        button_row1.add(profile_combo)
        button_row1.add_stretch()
        button_row1.add(open_controls_widget)

        button_row1a = ui.create_row_widget(properties={"spacing": 2})
        button_row1a.add(monitor_button)
        button_row1a.add_stretch()
        button_row1a.add(camera_current_label)

        def monitor_button_state_changed(visible, enabled):
            monitor_button.visible = visible
            monitor_button.enabled = enabled
            button_row1a.visible = visible

        def binning_combo_text_changed(text):
            if not self.__changes_blocked:
                self.__state_controller.handle_binning_changed(text)
                binning_combo.request_refocus()

        binning_combo = ui.create_combo_box_widget(properties={"min-width":72})
        binning_combo.on_current_text_changed = binning_combo_text_changed

        def handle_exposure_changed(text):
            self.__state_controller.handle_exposure_changed(text)
            exposure_field.request_refocus()

        def handle_decrease_exposure():
            self.__state_controller.handle_decrease_exposure()
            exposure_field.request_refocus()

        def handle_increase_exposure():
            self.__state_controller.handle_increase_exposure()
            exposure_field.request_refocus()

        exposure_field = ui.create_line_edit_widget(properties={"width": 44, "stylesheet": "qproperty-alignment: AlignRight"})  # note: this alignment technique will not work in future
        exposure_field.on_editing_finished = handle_exposure_changed

        parameters_group1 = ui.create_row_widget()

        parameters_row1 = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        parameters_row1.add_stretch()
        parameters_row1.add(ui.create_label_widget(_("Binning"), properties={"width": 68, "stylesheet": "qproperty-alignment: AlignRight"}))  # note: this alignment technique will not work in future
        parameters_row1.add_spacing(4)
        parameters_row1.add(binning_combo)
        parameters_group1.add(parameters_row1)
        parameters_group1.add_stretch()

        parameters_row2 = ui.create_row_widget(properties={"margin": 4, "spacing": 2})
        parameters_row2.add_stretch()
        colx = ui.create_column_widget()
        colx.add_spacing(2)
        colx.add(ui.create_label_widget(_("Time (ms)"), properties={"width": 68, "stylesheet": "qproperty-alignment: 'AlignBottom | AlignRight'"}))  # note: this alignment technique will not work in future
        colx.add_stretch()
        parameters_row2.add(colx)
        parameters_row2.add_spacing(4)
        group = ui.create_row_widget(properties={"width": 84})
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

        column = self.content_widget

        column.add(button_row1)
        column.add(button_row1a)
        column.add(parameters_group1)
        column.add(status_row)
        column.add(button_row)
        column.add_stretch()

        def profile_combo_text_changed(text):
            if not self.__changes_blocked:
                self.__state_controller.handle_change_profile(text)
                profile_combo.request_refocus()

        open_controls_button.on_button_clicked = functools.partial(self.__state_controller.handle_settings_button_clicked, PlugInManager.APIBroker())
        monitor_button.on_clicked = self.__state_controller.handle_monitor_button_clicked
        profile_combo.on_current_text_changed = profile_combo_text_changed

        def binning_values_changed(binning_values):
            binning_combo.items = [str(binning_value) for binning_value in binning_values]

        def profiles_changed(items):
            profile_combo.items = items

        def change_profile_combo(profile_label):
            blocked = self.__changes_blocked
            self.__changes_blocked = True
            try:
                profile_combo.current_text = profile_label
                profile_combo.request_refocus()
            finally:
                self.__changes_blocked = blocked

        # thread safe
        def profile_changed(profile_label):
            # the current_text must be set on ui thread
            self.document_controller.queue_task(functools.partial(change_profile_combo, profile_label))

        def frame_parameters_changed(frame_parameters):
            blocked = self.__changes_blocked
            self.__changes_blocked = True
            try:
                exposure_field.text = str("{0:.1f}".format(float(frame_parameters.exposure_ms)))
                if exposure_field.focused:
                    exposure_field.request_refocus()
                binning_combo.current_text = str(frame_parameters.binning)
            finally:
                self.__changes_blocked = blocked

        def play_button_state_changed(enabled, play_button_state):
            play_button_text = { "play": _("Play"), "pause": _("Pause") }
            play_button.enabled = enabled
            play_button.text = play_button_text[play_button_state]

        def abort_button_state_changed(visible, enabled):
            # abort_button.visible = visible
            abort_button.enabled = enabled

        def data_item_states_changed(data_item_states):
            map_channel_state_to_text = {"stopped": _("Stopped"), "complete": _("Acquiring"),
                "partial": _("Acquiring"), "marked": _("Stopping")}
            if len(data_item_states) > 0:
                data_item_state = data_item_states[0]
                channel_state = data_item_state["channel_state"]
                play_state_label.text = map_channel_state_to_text[channel_state]
            else:
                play_state_label.text = map_channel_state_to_text["stopped"]

        def camera_current_changed(camera_current):
            if camera_current:
                camera_current_label.text = str(int(camera_current * 1e12)) + _("pA")
                camera_current_label.text_color = "black"
            else:
                camera_current_label.text_color = "gray"

        def log_messages(messages, data_elements):
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
        self.__state_controller.on_data_item_states_changed = lambda a: self.document_controller.queue_task(lambda: data_item_states_changed(a))
        self.__state_controller.on_monitor_button_state_changed = monitor_button_state_changed
        self.__state_controller.on_camera_current_changed = camera_current_changed
        self.__state_controller.on_log_messages = log_messages

        self.__state_controller.initialize_state()

    # HACK: this is used to dump log messages to Swift.
    def periodic(self):
        self.__state_controller.handle_periodic()
        super().periodic()

    def close(self):
        self.__key_pressed_event_listener.close()
        self.__key_pressed_event_listener = None
        self.__key_released_event_listener.close()
        self.__key_released_event_listener = None
        self.__image_display_mouse_pressed_event_listener.close()
        self.__image_display_mouse_pressed_event_listener= None
        self.__image_display_mouse_released_event_listener.close()
        self.__image_display_mouse_released_event_listener= None
        self.__state_controller.close()
        self.__state_controller = None
        super().close()

    # this gets called from the DisplayPanelManager. pass on the message to the state controller.
    # must be called on ui thread
    def image_panel_mouse_pressed(self, display_panel, display_specifier, image_position, modifiers):
        data_item = display_specifier.data_item if display_specifier else None
        hardware_source_id = data_item and data_item.metadata.get("hardware_source", dict()).get("hardware_source_id")
        if self.__shift_click_state == "shift":
            mouse_position = image_position
            camera_shape = data_item.dimensional_shape
            self.__mouse_pressed = self.__state_controller.handle_shift_click(hardware_source_id, mouse_position, camera_shape)
            return self.__mouse_pressed
        if self.__shift_click_state == "tilt":
            mouse_position = image_position
            camera_shape = data_item.dimensional_shape
            self.__mouse_pressed = self.__state_controller.handle_tilt_click(hardware_source_id, mouse_position, camera_shape)
            return self.__mouse_pressed
        return False

    def image_panel_mouse_released(self, display_panel, display_specifier, image_position, modifiers):
        mouse_pressed = self.__mouse_pressed
        self.__mouse_pressed = False
        return mouse_pressed

    def image_panel_key_pressed(self, display_panel, key):
        if key.text.lower() == "s":
            self.__shift_click_state = "shift"
        elif key.text.lower() == "t":
            self.__shift_click_state = "tilt"
        else:
            self.__shift_click_state = None
        return False

    def image_panel_key_released(self, display_panel, key):
        self.__shift_click_state = None
        return False


class CameraControlPanel(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, "camera-control-panel")
        ui = document_controller.ui
        self.widget = ui.create_column_widget()
        self.hardware_source_id = properties["hardware_source_id"]
        camera_controller = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(self.hardware_source_id)
        if camera_controller:
            camera_control_widget = CameraControlWidget(self.document_controller, camera_controller)
            self.widget.add(camera_control_widget)
            self.widget.add_spacing(12)
            self.widget.add_stretch()


def create_camera_panel(document_controller, panel_id, properties):
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
            panel = Panel.Panel(document_controller, panel_id, properties)
            panel.widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, ui_handler)
            return panel
    return None


class CameraDisplayPanelController:
    """
        Represents a controller for the content of an image panel.
    """

    type = "camera-live"

    def __init__(self, display_panel, hardware_source_id, show_processed_data):
        assert hardware_source_id is not None
        hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
        self.type = CameraDisplayPanelController.type

        self.__hardware_source_id = hardware_source_id

        # configure the hardware source state controller
        self.__state_controller = CameraControlStateController(hardware_source, display_panel.document_controller.queue_task, display_panel.document_controller.document_model)

        # configure the user interface
        self.__display_name = str()
        self.__play_button_enabled = False
        self.__play_button_play_button_state = "play"
        self.__abort_button_visible = False
        self.__abort_button_enabled = False
        self.__data_item_states = list()
        self.__display_panel = display_panel
        self.__display_panel.header_canvas_item.end_header_color = "#98FB98"
        self.__playback_controls_composition = CanvasItem.CanvasItemComposition()
        self.__playback_controls_composition.layout = CanvasItem.CanvasItemLayout()
        self.__playback_controls_composition.sizing.set_fixed_height(30)
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
        self.__last_data_item = None
        self.__last_processed_data_item = None
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

        def update_display_name():
            new_text = "%s" % (self.__display_name)
            if hardware_source_display_name_canvas_item.text != new_text:
                hardware_source_display_name_canvas_item.text = new_text
                hardware_source_display_name_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_play_button():
            map_play_button_state_to_text = {"play": _("Play"), "pause": _("Pause")}
            play_button_text = map_play_button_state_to_text[self.__play_button_play_button_state]
            new_enabled = self.__play_button_enabled
            new_text = play_button_text
            if play_button_canvas_item.enabled != new_enabled or play_button_canvas_item.text != new_text:
                play_button_canvas_item.enabled = new_enabled
                play_button_canvas_item.text = new_text
                play_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_abort_button():
            abort_button_visible = self.__abort_button_visible
            abort_button_enabled = self.__abort_button_enabled
            new_text = _("Abort") if abort_button_visible else str()
            if abort_button_canvas_item.enabled != abort_button_enabled or abort_button_canvas_item.text != new_text:
                abort_button_canvas_item.text = new_text
                abort_button_canvas_item.enabled = abort_button_enabled
                abort_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def update_status_text():
            map_channel_state_to_text = {"stopped": _("Stopped"), "complete": _("Acquiring"),
                "partial": _("Acquiring"), "marked": _("Stopping")}
            for data_item_state in self.__data_item_states:
                channel_state = data_item_state["channel_state"]
                new_text = map_channel_state_to_text[channel_state]
                if status_text_canvas_item.text != new_text:
                    status_text_canvas_item.text = new_text
                    status_text_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)
                return

        def display_name_changed(display_name):
            self.__display_name = display_name
            update_display_name()

        def play_button_state_changed(enabled, play_button_state):
            self.__play_button_enabled = enabled
            self.__play_button_play_button_state = play_button_state
            update_play_button()

        def abort_button_state_changed(visible, enabled):
            self.__abort_button_visible = visible
            self.__abort_button_enabled = enabled
            update_abort_button()

        def data_item_states_changed(data_item_states):
            self.__data_item_states = data_item_states
            update_status_text()

        def update_capture_button(visible, enabled):
            if visible:
                capture_button.enabled = enabled
                capture_button.text = _("Capture")
                capture_button.size_to_content(display_panel.image_panel_get_font_metrics)
            else:
                capture_button.enabled = False
                capture_button.text = str()
                capture_button.size_to_content(display_panel.image_panel_get_font_metrics)

        def display_data_item_changed(data_item):
            if not self.__show_processed_checkbox or not self.__show_processed_checkbox.check_state == "checked":
                self.__state_controller.use_processed_data = False  # for capture
                display_panel.set_displayed_data_item(data_item)
            self.__last_data_item = data_item

        def processed_data_item_changed(data_item):
            if self.__show_processed_checkbox and self.__show_processed_checkbox.check_state == "checked":
                self.__state_controller.use_processed_data = True  # for capture
                display_panel.set_displayed_data_item(data_item)
            self.__last_processed_data_item = data_item

        def show_processed_checkbox_changed(check_state):
            if check_state == "checked":
                processed_data_item_changed(self.__last_processed_data_item)
            else:
                display_data_item_changed(self.__last_data_item)

        def display_new_data_item(data_item):
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
        self.__state_controller.on_data_item_states_changed = data_item_states_changed
        self.__state_controller.on_capture_button_state_changed = update_capture_button
        self.__state_controller.on_display_data_item_changed = display_data_item_changed
        self.__state_controller.on_display_new_data_item = display_new_data_item
        self.__state_controller.on_processed_data_item_changed = processed_data_item_changed

        play_button_canvas_item.on_button_clicked = self.__state_controller.handle_play_pause_clicked
        abort_button_canvas_item.on_button_clicked = self.__state_controller.handle_abort_clicked
        capture_button.on_button_clicked = self.__state_controller.handle_capture_clicked

        self.__state_controller.initialize_state()

    def close(self):
        self.__display_panel.footer_canvas_item.remove_canvas_item(self.__playback_controls_composition)
        self.__display_panel = None
        self.__state_controller.close()
        self.__state_controller = None

    def save(self, d):
        d["hardware_source_id"] = self.__hardware_source_id
        if self.__show_processed_checkbox:
            d["show_processed_data"] = self.__show_processed_checkbox.check_state == "checked"

    def key_pressed(self, key):
        if key.text == " ":
            self.__state_controller.handle_play_pause_clicked()
            return True
        elif key.key == 0x1000000:  # escape
            self.__state_controller.handle_abort_clicked()
            return True
        return False

    def key_released(self, key):
        return False


hardware_source_added_event_listener = None
hardware_source_removed_event_listener = None


_component_registered_listener = None
_component_unregistered_listener = None


def run():
    global hardware_source_added_event_listener, hardware_source_removed_event_listener
    camera_control_panels = dict()

    def register_camera_panel(hardware_source):
        """Called when a hardware source is added to the hardware source manager."""

        # check to see if we handle this hardware source.
        is_ronchigram_camera = hardware_source.features.get("is_ronchigram_camera", False)
        is_eels_camera = hardware_source.features.get("is_eels_camera", False)
        if is_ronchigram_camera or is_eels_camera:

            panel_id = "camera-control-panel-" + hardware_source.hardware_source_id
            name = hardware_source.display_name + " " + _("Camera Control")
            camera_control_panels[hardware_source.hardware_source_id] = panel_id

            class CameraDisplayPanelControllerFactory:
                def __init__(self):
                    self.priority = 2

                def build_menu(self, display_type_menu, selected_display_panel):
                    # return a list of actions that have been added to the menu.
                    def switch_to_live_controller(hardware_source):
                        d = {"type": "image", "controller_type": CameraDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                        selected_display_panel.change_display_panel_content(d)

                    action = display_type_menu.add_menu_item(hardware_source.display_name, functools.partial(switch_to_live_controller, hardware_source))
                    action.checked = False
                    return [action]

                def make_new(self, controller_type, display_panel, d):
                    # make a new display panel controller, typically called to restore contents of a display panel.
                    # controller_type will match the type property of the display panel controller when it was saved.
                    # d is the dictionary that is saved when the display panel controller closes.
                    hardware_source_id = d.get("hardware_source_id")
                    show_processed_data = d.get("show_processed_data", False)
                    if controller_type == CameraDisplayPanelController.type and hardware_source_id == hardware_source.hardware_source_id:
                        return CameraDisplayPanelController(display_panel, hardware_source_id, show_processed_data)
                    return None

                def match(self, document_model, data_item: DataItem.DataItem) -> dict:
                    if HardwareSource.matches_hardware_source(hardware_source.hardware_source_id, None, document_model, data_item):
                        return {"controller_type": CameraDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                    return None

            DisplayPanel.DisplayPanelManager().register_display_panel_controller_factory("camera-live-" + hardware_source.hardware_source_id, CameraDisplayPanelControllerFactory())

            panel_properties = {"hardware_source_id": hardware_source.hardware_source_id}

            camera_panel_type = hardware_source.features.get("camera_panel_type")
            if not camera_panel_type or camera_panel_type in ("ronchigram", "eels"):
                Workspace.WorkspaceManager().register_panel(CameraControlPanel, panel_id, name, ["left", "right"], "left", panel_properties)
            else:
                panel_properties["camera_panel_type"] = camera_panel_type
                Workspace.WorkspaceManager().register_panel(create_camera_panel, panel_id, name, ["left", "right"], "left", panel_properties)

    def unregister_camera_panel(hardware_source):
        """Called when a hardware source is removed from the hardware source manager."""
        is_ronchigram_camera = hardware_source.features.get("is_ronchigram_camera", False)
        is_eels_camera = hardware_source.features.get("is_eels_camera", False)
        if is_ronchigram_camera or is_eels_camera:
            DisplayPanel.DisplayPanelManager().unregister_display_panel_controller_factory("camera-live-" + hardware_source.hardware_source_id)
            panel_id = camera_control_panels.get(hardware_source.hardware_source_id)
            if panel_id:
                Workspace.WorkspaceManager().unregister_panel(panel_id)

    hardware_source_added_event_listener = HardwareSource.HardwareSourceManager().hardware_source_added_event.listen(register_camera_panel)
    hardware_source_removed_event_listener = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(unregister_camera_panel)
    for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
        register_camera_panel(hardware_source)
