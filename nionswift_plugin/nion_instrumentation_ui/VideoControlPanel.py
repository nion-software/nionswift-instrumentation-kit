# would like to create a panel with a combobox containing
# all registered HWSources, and a play/stop button.

import functools
import gettext

from nion.swift import DataItemThumbnailWidget
from nion.swift import DisplayPanel
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import HardwareSource
from nion.ui import CanvasItem
from nion.ui import Widgets
from nion.utils import Geometry

from nion.instrumentation import video_base


_ = gettext.gettext


class VideoSourceStateController:

    """
    Track the state of a hardware source, as it relates to the UI.

    hardware_source may be None

    Hardware source should support the following API:
        (acquisition)
            (event) data_item_states_changed_event(data_item_states)
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

    Clients should call:
        handle_play_clicked(workspace_controller)
        handle_abort_clicked()

    Clients can respond to:
        on_display_name_changed(display_name)
        on_play_button_state_changed(enabled, play_button_state)  play, pause
        on_abort_button_state_changed(visible, enabled)
    """

    def __init__(self, hardware_source, queue_task, document_model):
        self.__hardware_source = hardware_source
        self.queue_task = queue_task
        self.__document_model = document_model
        self.__data_item_states_changed_event_listener = None
        self.__acquisition_state_changed_event_listener = None
        self.on_display_name_changed = None
        self.on_play_button_state_changed = None
        self.on_abort_button_state_changed = None
        self.on_data_item_states_changed = None
        self.on_display_data_item_changed = None
        self.on_display_new_data_item = None

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

    def close(self):
        if self.__acquisition_state_changed_event_listener:
            self.__acquisition_state_changed_event_listener.close()
            self.__acquisition_state_changed_event_listener = None
        if self.__data_item_states_changed_event_listener:
            self.__data_item_states_changed_event_listener.close()
            self.__data_item_states_changed_event_listener = None
        self.__data_item_changed_event_listener.close()
        self.__data_item_changed_event_listener = None
        self.on_display_name_changed = None
        self.on_play_button_state_changed = None
        self.on_abort_button_state_changed = None
        self.on_data_item_states_changed = None
        self.on_display_data_item_changed = None
        self.on_display_new_data_item = None
        self.__hardware_source = None

    def __update_play_button_state(self):
        enabled = self.__hardware_source is not None
        if self.on_play_button_state_changed:
            self.on_play_button_state_changed(enabled, "pause" if self.is_playing else "play")

    def __update_abort_button_state(self):
        if self.on_abort_button_state_changed:
            self.on_abort_button_state_changed(self.is_playing, self.is_playing)

    def __update_buttons(self):
        self.__update_play_button_state()
        self.__update_abort_button_state()

    # not thread safe
    def __update_display_data_item(self):
        data_item_reference = self.__document_model.get_data_item_reference(self.__hardware_source.hardware_source_id)
        with data_item_reference.mutex:
            self.__data_item = data_item_reference.data_item
            if self.on_display_data_item_changed:
                self.on_display_data_item_changed(self.__data_item)

    def initialize_state(self):
        """ Call this to initialize the state of the UI after everything has been connected. """
        if self.__hardware_source:
            self.__data_item_states_changed_event_listener = self.__hardware_source.data_item_states_changed_event.listen(self.__data_item_states_changed)
            self.__acquisition_state_changed_event_listener = self.__hardware_source.acquisition_state_changed_event.listen(self.__acquisition_state_changed)
        if self.on_display_name_changed:
            self.on_display_name_changed(self.display_name)
        self.__update_buttons()
        if self.on_data_item_states_changed:
            self.on_data_item_states_changed(list())
        self.__update_display_data_item()

    def handle_play_clicked(self):
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

    @property
    def is_playing(self):
        """ Returns whether the hardware source is playing or not. """
        return self.__hardware_source.is_playing if self.__hardware_source else False

    @property
    def display_name(self):
        """ Returns the display name for the hardware source. """
        return self.__hardware_source.display_name if self.__hardware_source else _("N/A")

    # must be called on ui thread
    def handle_periodic(self):
        if self.__hardware_source and getattr(self.__hardware_source, "periodic", None):
            self.__hardware_source.periodic()

    # this message comes from the data buffer. it will always be invoked on the UI thread.
    def __acquisition_state_changed(self, is_playing):
        self.queue_task(self.__update_buttons)

    # this message comes from the hardware source. may be called from thread.
    def __data_item_states_changed(self, data_item_states):
        if self.on_data_item_states_changed:
            self.on_data_item_states_changed(data_item_states)


class VideoDisplayPanelController:
    """
        Represents a controller for the content of an image panel.
    """

    type = "video-live"

    def __init__(self, display_panel, hardware_source_id):
        assert hardware_source_id is not None
        hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
        self.type = VideoDisplayPanelController.type

        self.__hardware_source_id = hardware_source_id

        # configure the hardware source state controller
        self.__state_controller = VideoSourceStateController(hardware_source, display_panel.document_controller.queue_task, display_panel.document_controller.document_model)

        # configure the user interface
        self.__play_button_enabled = False
        self.__play_button_play_button_state = "play"
        self.__data_item_states = list()
        self.__display_panel = display_panel
        self.__display_panel.header_canvas_item.end_header_color = "#DAA520"
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
        playback_controls_row.add_canvas_item(hardware_source_display_name_canvas_item)
        self.__playback_controls_composition.add_canvas_item(CanvasItem.BackgroundCanvasItem("#DAA520"))
        self.__playback_controls_composition.add_canvas_item(playback_controls_row)
        self.__display_panel.footer_canvas_item.insert_canvas_item(0, self.__playback_controls_composition)

        def display_name_changed(display_name):
            hardware_source_display_name_canvas_item.text = display_name
            hardware_source_display_name_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def play_button_state_changed(enabled, play_button_state):
            play_button_canvas_item.enabled = enabled
            map_play_button_state_to_text = {"play": _("Play"), "pause": _("Pause")}
            play_button_canvas_item.text = map_play_button_state_to_text[play_button_state]
            play_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def abort_button_state_changed(visible, enabled):
            abort_button_canvas_item.text = _("Abort") if visible else str()
            abort_button_canvas_item.enabled = enabled
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

        def data_item_states_changed(data_item_states):
            self.__data_item_states = data_item_states
            update_status_text()

        def display_data_item_changed(data_item):
            display_panel.set_displayed_data_item(data_item)

        def display_new_data_item(data_item):
            result_display_panel = display_panel.document_controller.next_result_display_panel()
            if result_display_panel:
                result_display_panel.set_display_panel_data_item(data_item)
                result_display_panel.request_focus()

        self.__state_controller.on_display_name_changed = display_name_changed
        self.__state_controller.on_play_button_state_changed = play_button_state_changed
        self.__state_controller.on_abort_button_state_changed = abort_button_state_changed
        self.__state_controller.on_data_item_states_changed = data_item_states_changed
        self.__state_controller.on_display_data_item_changed = display_data_item_changed
        self.__state_controller.on_display_new_data_item = display_new_data_item

        play_button_canvas_item.on_button_clicked = self.__state_controller.handle_play_clicked
        abort_button_canvas_item.on_button_clicked = self.__state_controller.handle_abort_clicked

        self.__state_controller.initialize_state()

        document_model = self.__display_panel.document_controller.document_model

        # def update_display_data_item():
        #     data_item_reference = document_model.get_data_item_reference(self.__hardware_source_id)
        #     with data_item_reference.mutex:
        #         data_item = data_item_reference.data_item
        #         if data_item:
        #             self.__display_panel.set_displayed_data_item(data_item)
        #         else:
        #             self.__display_panel.set_displayed_data_item(None)
        #
        # self.__data_item_reference = document_model.get_data_item_reference(self.__hardware_source_id)
        # self.__data_item_changed_event_listener = self.__data_item_reference.data_item_changed_event.listen(update_display_data_item)
        #
        # update_display_data_item()

    def close(self):
        # self.__data_item_changed_event_listener.close()
        # self.__data_item_changed_event_listener = None
        self.__display_panel.footer_canvas_item.remove_canvas_item(self.__playback_controls_composition)
        self.__display_panel = None
        self.__state_controller.close()
        self.__state_controller = None

    def save(self, d):
        d["hardware_source_id"] = self.__hardware_source_id

    def key_pressed(self, key):
        if key.text == " ":
            self.__state_controller.handle_play_clicked()
            return True
        elif key.key == 0x1000000:  # escape
            self.__state_controller.handle_abort_clicked()
            return True
        return False

    def key_released(self, key):
        return False


class VideoSourceWidget(Widgets.CompositeWidgetBase):

    def __init__(self, document_controller, hardware_source):
        super().__init__(document_controller.ui.create_column_widget(properties={"margin": 0, "spacing": 2}))

        self.document_controller = document_controller

        self.__state_controller = VideoSourceStateController(hardware_source, document_controller.queue_task, document_controller.document_model)

        ui = document_controller.ui

        # top row, source selection, play button
        top_row = ui.create_row_widget(properties={"spacing": 8, "margin-left": 8, "margin-top": 4})
        top_row.add(ui.create_label_widget(hardware_source.display_name))
        top_row.add_stretch()
        # next row, prev and next buttons and status text
        next_row = ui.create_row_widget(properties={"spacing": 8, "margin-left": 8, "margin-top": 0})
        play_pause_button = ui.create_push_button_widget("Play")
        next_row.add(play_pause_button)
        next_row.add_stretch()
        document_model = self.document_controller.document_model
        data_item_reference = document_model.get_data_item_reference(hardware_source.hardware_source_id)
        data_item_thumbnail_source = DataItemThumbnailWidget.DataItemReferenceThumbnailSource(ui, data_item_reference)
        thumbnail_widget = DataItemThumbnailWidget.ThumbnailWidget(ui, data_item_thumbnail_source, Geometry.IntSize(width=36, height=36))
        next_row.add(thumbnail_widget)
        next_row.add_spacing(12)
        # build the main column
        column = self.content_widget
        column.add(top_row)
        column.add(next_row)

        def play_button_state_changed(enabled, play_button_state):
            play_button_text = { "play": _("Play"), "pause": _("Pause") }
            play_pause_button.enabled = enabled
            play_pause_button.text = play_button_text[play_button_state]

        def thumbnail_widget_drag(mime_data, thumbnail, hot_spot_x, hot_spot_y):
            self.drag(mime_data, thumbnail, hot_spot_x, hot_spot_y)

        thumbnail_widget.on_drag = thumbnail_widget_drag

        # connections
        play_pause_button.on_clicked = self.__state_controller.handle_play_clicked

        self.__state_controller.on_play_button_state_changed = play_button_state_changed

        self.__state_controller.initialize_state()

    def close(self):
        self.__state_controller.close()
        self.__state_controller = None
        super().close()

    # HACK: this is used to dump log messages to Swift.
    def periodic(self):
        self.__state_controller.handle_periodic()
        super().periodic()


class VideoSourcePanel(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, _("Video Source"))

        ui = document_controller.ui

        hardware_column = ui.create_column_widget()

        self.hardware_source_widgets = []

        hardware_sources = HardwareSource.HardwareSourceManager().hardware_sources
        hardware_sources.sort(key=lambda hardware_source: hardware_source.display_name)

        for hardware_source in hardware_sources:
            hardware_source_widget = VideoSourceWidget(document_controller, hardware_source)
            hardware_column.add(hardware_source_widget)

        hardware_column.add_stretch()

        self.widget = hardware_column


workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(VideoSourcePanel, "video-source-control-panel", _("Video Source"), ["left", "right"], "left")

hardware_source_added_event_listener = None
hardware_source_removed_event_listener = None

def run():
    global hardware_source_added_event_listener, hardware_source_removed_event_listener
    hardware_control_panels = dict()

    def register_hardware_panel(hardware_source):
        if hardware_source.features.get("is_video", False):
            panel_id = "video-control-panel-" + hardware_source.hardware_source_id
            hardware_control_panels[hardware_source.hardware_source_id] = panel_id

            class HardwareDisplayPanelControllerFactory:
                def __init__(self):
                    self.priority = 1

                def build_menu(self, display_type_menu, selected_display_panel):
                    # return a list of actions that have been added to the menu.
                    def switch_to_live_controller(hardware_source):
                        d = {"type": "image", "controller_type": VideoDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                        selected_display_panel.change_display_panel_content(d)

                    action = display_type_menu.add_menu_item(hardware_source.display_name, functools.partial(switch_to_live_controller, hardware_source))
                    action.checked = False
                    return [action]

                def make_new(self, controller_type, display_panel, d):
                    # make a new display panel controller, typically called to restore contents of a display panel.
                    # controller_type will match the type property of the display panel controller when it was saved.
                    # d is the dictionary that is saved when the display panel controller closes.
                    hardware_source_id = d.get("hardware_source_id")
                    if controller_type == VideoDisplayPanelController.type and hardware_source_id == hardware_source.hardware_source_id:
                        return VideoDisplayPanelController(display_panel, hardware_source_id)
                    return None

                def match(self, document_model, data_item):
                    if HardwareSource.matches_hardware_source(hardware_source.hardware_source_id, None, document_model, data_item):
                        return {"controller_type": VideoDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                    return None

            DisplayPanel.DisplayPanelManager().register_display_panel_controller_factory("video-live-" + hardware_source.hardware_source_id, HardwareDisplayPanelControllerFactory())

            video_base.video_configuration.video_sources.append_item(hardware_source)

    def unregister_hardware_panel(hardware_source):
        if hardware_source.features.get("is_video", False):
            DisplayPanel.DisplayPanelManager().unregister_display_panel_controller_factory("video-live-" + hardware_source.hardware_source_id)
            video_sources = video_base.video_configuration.video_sources
            video_sources.remove_item(video_sources.items.index(hardware_source))

    hardware_source_added_event_listener = HardwareSource.HardwareSourceManager().hardware_source_added_event.listen(register_hardware_panel)
    hardware_source_removed_event_listener = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(unregister_hardware_panel)
    for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
        register_hardware_panel(hardware_source)
