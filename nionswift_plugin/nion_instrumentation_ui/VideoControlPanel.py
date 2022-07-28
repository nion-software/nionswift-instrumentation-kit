# would like to create a panel with a combobox containing
# all registered HWSources, and a play/stop button.

from __future__ import annotations

import asyncio
import copy
import functools
import gettext
import typing

from nion.instrumentation import HardwareSource
from nion.swift import DataItemThumbnailWidget
from nion.swift import DisplayPanel
from nion.swift import Panel
from nion.swift import Workspace
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import PreferencesDialog
from nion.ui import Widgets
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Registry

from nion.instrumentation import video_base

if typing.TYPE_CHECKING:
    from nion.swift.model import DataItem
    from nion.swift.model import DocumentModel
    from nion.swift.model import Persistence
    from nion.swift import DocumentController
    from nion.ui import DrawingContext
    from nion.ui import UserInterface
    from nion.utils import ListModel

_ = gettext.gettext

map_channel_state_to_text = {
    "stopped": _("Stopped"),
    "complete": _("Acquiring"),
    "partial": _("Acquiring"),
    "marked": _("Stopping"),
    "error": _("Error"),
}


class VideoSourceStateController:

    """
    Track the state of a hardware source, as it relates to the UI.

    hardware_source may be None

    Hardware source should support the following API:
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

    Clients should call:
        handle_play_clicked(workspace_controller)
        handle_abort_clicked()

    Clients can respond to:
        on_display_name_changed(display_name)
        on_play_button_state_changed(enabled, play_button_state)  play, pause
        on_abort_button_state_changed(visible, enabled)
    """

    def __init__(self, hardware_source: video_base.VideoHardwareSource, queue_task: typing.Callable[[typing.Callable[[], None]], None], document_model: DocumentModel.DocumentModel) -> None:
        self.__hardware_source = hardware_source
        self.queue_task = queue_task
        self.__document_model = document_model
        self.__acquisition_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.__data_channel_state_changed_event_listener: typing.Optional[Event.EventListener] = None
        self.on_display_name_changed: typing.Optional[typing.Callable[[str], None]] = None
        self.on_play_button_state_changed: typing.Optional[typing.Callable[[bool, str], None]] = None
        self.on_abort_button_state_changed: typing.Optional[typing.Callable[[bool, bool], None]] = None
        self.on_display_new_data_item: typing.Optional[typing.Callable[[DataItem.DataItem], None]] = None

        self.acquisition_state_model = Model.PropertyModel[str]("stopped")

        self.data_item_reference = document_model.get_data_item_reference(self.__hardware_source.hardware_source_id)

        def hardware_source_property_changed(property_name: str) -> None:
            if property_name == "display_name":
                if callable(self.on_display_name_changed):
                    self.on_display_name_changed(self.__hardware_source.display_name)

        self.__property_changed_event_listener: typing.Optional[Event.EventListener] = self.__hardware_source.property_changed_event.listen(hardware_source_property_changed)

        def hardware_source_removed(hardware_source: video_base.VideoHardwareSource) -> None:
            if self.__hardware_source == hardware_source:
                if self.__property_changed_event_listener:
                    self.__property_changed_event_listener.close()
                    self.__property_changed_event_listener = typing.cast(typing.Any, None)
                    self.__hardware_source = typing.cast(typing.Any, None)

        self.__hardware_source_removed_event = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(hardware_source_removed)

    def close(self) -> None:
        if self.__acquisition_state_changed_event_listener:
            self.__acquisition_state_changed_event_listener.close()
            self.__acquisition_state_changed_event_listener = None
        self.__data_channel_state_changed_event_listener = None
        self.on_display_name_changed = None
        self.on_play_button_state_changed = None
        self.on_abort_button_state_changed = None
        self.on_display_new_data_item = None
        if self.__property_changed_event_listener:
            self.__property_changed_event_listener.close()
            self.__property_changed_event_listener = None
        self.__hardware_source_removed_event.close()
        self.__hardware_source_removed_event = typing.cast(typing.Any, None)
        self.__hardware_source = typing.cast(typing.Any, None)

    def __update_play_button_state(self) -> None:
        enabled = self.__hardware_source is not None
        if self.on_play_button_state_changed:
            self.on_play_button_state_changed(enabled, "pause" if self.is_playing else "play")

    def __update_abort_button_state(self) -> None:
        if self.on_abort_button_state_changed:
            self.on_abort_button_state_changed(self.is_playing, self.is_playing)

    def __update_buttons(self) -> None:
        self.__update_play_button_state()
        self.__update_abort_button_state()

    def initialize_state(self) -> None:
        """ Call this to initialize the state of the UI after everything has been connected. """
        if self.__hardware_source:
            self.__data_channel_state_changed_event_listener = self.__hardware_source.data_channel_state_changed_event.listen(self.__data_channel_state_changed)
            self.__acquisition_state_changed_event_listener = self.__hardware_source.acquisition_state_changed_event.listen(self.__acquisition_state_changed)
        if self.on_display_name_changed:
            self.on_display_name_changed(self.display_name)
        self.__update_buttons()

    def handle_play_clicked(self) -> None:
        """ Call this when the user clicks the play/pause button. """
        if self.__hardware_source:
            if self.is_playing:
                self.__hardware_source.stop_playing()
            else:
                self.__hardware_source.start_playing()

    def handle_abort_clicked(self) -> None:
        """ Call this when the user clicks the abort button. """
        if self.__hardware_source:
            self.__hardware_source.abort_playing()

    @property
    def is_playing(self) -> bool:
        """ Returns whether the hardware source is playing or not. """
        return self.__hardware_source.is_playing if self.__hardware_source else False

    @property
    def display_name(self) -> str:
        """ Returns the display name for the hardware source. """
        return self.__hardware_source.display_name if self.__hardware_source else _("N/A")

    # this message comes from the data buffer. it will always be invoked on the UI thread.
    def __acquisition_state_changed(self, is_playing: bool) -> None:
        self.queue_task(self.__update_buttons)

    def __data_channel_state_changed(self, data_channel: HardwareSource.DataChannel) -> None:
        if data_channel.is_started and data_channel.state:
            self.acquisition_state_model.value = data_channel.state
        else:
            self.acquisition_state_model.value = "error" if data_channel.is_error else "stopped"


class VideoDisplayPanelController:
    """
        Represents a controller for the content of an image panel.
    """

    type = "video-live"

    def __init__(self, display_panel: DisplayPanel.DisplayPanel, hardware_source_id: str) -> None:
        assert hardware_source_id is not None
        hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
        assert isinstance(hardware_source, video_base.VideoHardwareSource)
        self.type = VideoDisplayPanelController.type

        self.__hardware_source_id = hardware_source_id

        # configure the hardware source state controller
        self.__state_controller = VideoSourceStateController(hardware_source, display_panel.document_controller.queue_task, display_panel.document_controller.document_model)

        # configure the user interface
        self.__play_button_enabled = False
        self.__play_button_play_button_state = "play"
        self.__display_panel = display_panel
        self.__display_panel.header_canvas_item.end_header_color = "#DAA520"
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
        playback_controls_row.add_canvas_item(hardware_source_display_name_canvas_item)
        self.__playback_controls_composition.add_canvas_item(CanvasItem.BackgroundCanvasItem("#DAA520"))
        self.__playback_controls_composition.add_canvas_item(playback_controls_row)
        self.__display_panel.footer_canvas_item.insert_canvas_item(0, self.__playback_controls_composition)

        def display_name_changed(display_name: str) -> None:
            hardware_source_display_name_canvas_item.text = display_name
            hardware_source_display_name_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def play_button_state_changed(enabled: bool, play_button_state: str) -> None:
            play_button_canvas_item.enabled = enabled
            map_play_button_state_to_text = {"play": _("Play"), "pause": _("Pause")}
            play_button_canvas_item.text = map_play_button_state_to_text[play_button_state]
            play_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def abort_button_state_changed(visible: bool, enabled: bool) -> None:
            abort_button_canvas_item.text = _("Abort") if visible else str()
            abort_button_canvas_item.enabled = enabled
            abort_button_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

        def acquisition_state_changed(key: str) -> None:
            # this may be called on a thread. create an async method (guaranteed to run on the main thread)
            # and add it to the window event loop.
            async def update_acquisition_state_label(acquisition_state: typing.Optional[str]) -> None:
                acquisition_state = acquisition_state or "stopped"
                status_text_canvas_item.text = map_channel_state_to_text[acquisition_state]
                status_text_canvas_item.size_to_content(display_panel.image_panel_get_font_metrics)

            self.__display_panel.document_controller.event_loop.create_task(update_acquisition_state_label(self.__state_controller.acquisition_state_model.value))

        def display_new_data_item(data_item: DataItem.DataItem) -> None:
            result_display_panel = display_panel.document_controller.next_result_display_panel()
            if result_display_panel:
                result_display_panel.set_display_panel_data_item(data_item)
                result_display_panel.request_focus()

        self.__state_controller.on_display_name_changed = display_name_changed
        self.__state_controller.on_play_button_state_changed = play_button_state_changed
        self.__state_controller.on_abort_button_state_changed = abort_button_state_changed
        self.__state_controller.on_display_new_data_item = display_new_data_item

        display_panel.set_data_item_reference(self.__state_controller.data_item_reference)

        play_button_canvas_item.on_button_clicked = self.__state_controller.handle_play_clicked
        abort_button_canvas_item.on_button_clicked = self.__state_controller.handle_abort_clicked

        self.__acquisition_state_changed_listener = self.__state_controller.acquisition_state_model.property_changed_event.listen(acquisition_state_changed)

        self.__state_controller.initialize_state()

        acquisition_state_changed("value")

    def close(self) -> None:
        self.__display_panel.footer_canvas_item.remove_canvas_item(self.__playback_controls_composition)
        self.__display_panel = typing.cast(typing.Any, None)
        self.__state_controller.close()
        self.__state_controller = typing.cast(typing.Any, None)
        self.__acquisition_state_changed_listener = typing.cast(typing.Any, None)

    def save(self, d: typing.MutableMapping[str, typing.Any]) -> None:
        d["hardware_source_id"] = self.__hardware_source_id

    def key_pressed(self, key: UserInterface.Key) -> bool:
        if key.text == " ":
            self.__state_controller.handle_play_clicked()
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


class VideoSourceWidget(Widgets.CompositeWidgetBase):

    def __init__(self, document_controller: DocumentController.DocumentController, hardware_source: video_base.VideoHardwareSource) -> None:
        content_widget = document_controller.ui.create_column_widget(properties={"margin": 0, "spacing": 2})
        super().__init__(content_widget)

        self.document_controller = document_controller

        self.__state_controller = VideoSourceStateController(hardware_source, document_controller.queue_task, document_controller.document_model)

        ui = document_controller.ui

        # top row, source selection, play button
        hardware_source_display_name_label = ui.create_label_widget(hardware_source.display_name)
        top_row = ui.create_row_widget(properties={"spacing": 8, "margin-left": 8, "margin-top": 4})
        top_row.add(hardware_source_display_name_label)
        top_row.add_stretch()
        # next row, prev and next buttons and status text
        next_row = ui.create_row_widget(properties={"spacing": 8, "margin-left": 8, "margin-top": 0})
        play_pause_button = ui.create_push_button_widget("Play")
        next_row.add(play_pause_button)
        next_row.add_stretch()
        document_model = self.document_controller.document_model
        data_item_reference = document_model.get_data_item_reference(hardware_source.hardware_source_id)
        data_item_thumbnail_source = DataItemThumbnailWidget.DataItemReferenceThumbnailSource(ui, document_model, data_item_reference)
        thumbnail_widget = DataItemThumbnailWidget.ThumbnailWidget(ui, data_item_thumbnail_source, Geometry.IntSize(width=36, height=36))
        next_row.add(thumbnail_widget)
        next_row.add_spacing(12)
        # build the main column
        content_widget.add(top_row)
        content_widget.add(next_row)

        def play_button_state_changed(enabled: bool, play_button_state: str) -> None:
            play_button_text = { "play": _("Play"), "pause": _("Pause") }
            play_pause_button.enabled = enabled
            play_pause_button.text = play_button_text[play_button_state]

        def thumbnail_widget_drag(mime_data: UserInterface.MimeData, thumbnail: typing.Optional[DrawingContext.RGBA32Type], hot_spot_x: int, hot_spot_y: int) -> None:
            self.drag(mime_data, thumbnail, hot_spot_x, hot_spot_y)

        def display_name_changed(display_name: str) -> None:
            hardware_source_display_name_label.text = display_name

        thumbnail_widget.on_drag = thumbnail_widget_drag

        # connections
        play_pause_button.on_clicked = self.__state_controller.handle_play_clicked

        self.__state_controller.on_display_name_changed = display_name_changed
        self.__state_controller.on_play_button_state_changed = play_button_state_changed

        self.__state_controller.initialize_state()

    def close(self) -> None:
        self.__state_controller.close()
        self.__state_controller = typing.cast(typing.Any, None)
        super().close()


class VideoSourcePanel(Panel.Panel):

    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> None:
        super().__init__(document_controller, panel_id, _("Video Source"))

        ui = document_controller.ui

        hardware_column = ui.create_column_widget()

        self.__hardware_source_widgets: typing.Dict[str, VideoSourceWidget] = dict()

        hardware_sources = HardwareSource.HardwareSourceManager().hardware_sources
        hardware_sources.sort(key=lambda hardware_source: hardware_source.display_name or _("Unknown"))

        def hardware_source_added(hardware_source: HardwareSource.HardwareSource) -> None:
            if hasattr(hardware_source, "video_device") and isinstance(hardware_source, video_base.VideoHardwareSource):
                hardware_source_widget = VideoSourceWidget(document_controller, hardware_source)
                hardware_column.add(hardware_source_widget)
                self.__hardware_source_widgets[hardware_source.hardware_source_id] = hardware_source_widget

        def hardware_source_removed(hardware_source: HardwareSource.HardwareSource) -> None:
            hardware_source_widget = self.__hardware_source_widgets.get(hardware_source.hardware_source_id)
            if hardware_source_widget:
                hardware_column.remove(hardware_source_widget)

        for hardware_source in hardware_sources:
            hardware_source_added(hardware_source)

        hardware_column.add_stretch()

        self.__hardware_source_added_event = HardwareSource.HardwareSourceManager().hardware_source_added_event.listen(hardware_source_added)

        self.__hardware_source_removed_event = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(hardware_source_removed)

        self.widget = hardware_column

    def close(self) -> None:
        self.__hardware_source_added_event.close()
        self.__hardware_source_added_event = typing.cast(typing.Any, None)
        self.__hardware_source_removed_event.close()
        self.__hardware_source_removed_event = typing.cast(typing.Any, None)
        super().close()


class VideoPreferencePanel:

    def __init__(self, video_configuration: video_base.VideoConfiguration) -> None:
        self.identifier = "video_sources"
        self.label = _("Video Sources")
        self.__video_configuration = video_configuration

    def build(self, ui: UserInterface.UserInterface, event_loop: asyncio.AbstractEventLoop, **kwargs: typing.Any) -> Declarative.DeclarativeWidget:
        u = Declarative.DeclarativeUI()

        video_device_factories: typing.List[video_base.VideoDeviceFactoryLike] = list(Registry.get_components_by_type("video_device_factory"))

        class Handler(Declarative.Handler):
            def __init__(self, ui_view: Declarative.UIDescription, video_sources: ListModel.ListModel[video_base.VideoHardwareSource]) -> None:
                super().__init__()
                self.ui_view = ui_view
                self.video_sources = video_sources
                self.video_source_type_index = Model.PropertyModel(0)

            def create_new_video_device(self, widget: UserInterface.Widget) -> None:
                video_base.video_configuration.create_hardware_source(video_device_factories[self.video_source_type_index.value or 0])

            def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:

                class SectionHandler(Declarative.Handler):

                    def __init__(self, container: typing.Any, hardware_source: video_base.VideoHardwareSource) -> None:
                        super().__init__()
                        self.container = container
                        self.hardware_source = hardware_source
                        self.settings = video_base.video_configuration.get_settings_model(hardware_source)
                        self.settings_original = copy.deepcopy(self.settings)
                        self.needs_saving_model = Model.PropertyModel(False)
                        self.property_changed_event = Event.Event()
                        # these will be assigned by declarative setup
                        self.apply_button: typing.Optional[UserInterface.PushButtonWidget] = None
                        self.revert_button: typing.Optional[UserInterface.PushButtonWidget] = None

                        def settings_changed(property_name: str) -> None:
                            self.needs_saving_model.value = True

                        self.__settings_changed_event_listener = self.settings.property_changed_event.listen(settings_changed) if self.settings else None

                        def needs_saving_model_changed(property_name: str) -> None:
                            if self.apply_button:
                                self.apply_button.enabled = self.needs_saving_model.value or False
                            if self.revert_button:
                                self.revert_button.enabled = self.needs_saving_model.value or False

                        self.__needs_saving_changed_event_listener = self.needs_saving_model.property_changed_event.listen(needs_saving_model_changed)

                    def close(self) -> None:
                        if self.__settings_changed_event_listener:
                            self.__settings_changed_event_listener.close()
                            self.__settings_changed_event_listener = None
                        super().close()

                    def init_handler(self) -> None:
                        if self.apply_button:
                            self.apply_button.enabled = self.needs_saving_model.value or False
                        if self.revert_button:
                            self.revert_button.enabled = self.needs_saving_model.value or False

                    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
                        if component_id == "edit_group":
                            if self.__video_device_factory:
                                return typing.cast(Declarative.HandlerLike, self.__video_device_factory.create_editor_handler(self.settings))
                            else:
                                return self
                        raise ValueError("Unable to create component.")

                    @property
                    def resources(self) -> typing.Mapping[str, typing.Any]:
                        if self.__video_device_factory:
                            content = self.__video_device_factory.get_editor_description()
                        else:
                            content = u.create_label(text=_("Not Available"))

                        component = u.define_component(content=content, component_id="edit_group")

                        return {"edit_group": component}

                    @property
                    def __video_device_factory(self) -> typing.Optional[video_base.VideoDeviceFactoryLike]:
                        for video_device_factory in video_device_factories:
                            if video_device_factory.factory_id == getattr(self.settings, "driver", None):
                                return video_device_factory
                        return None

                    @property
                    def driver_display_name(self) -> str:
                        video_device_factory = self.__video_device_factory
                        return video_device_factory.display_name if video_device_factory else _("Unknown")

                    def apply(self, widget: UserInterface.Widget) -> None:
                        settings = self.settings
                        if settings:
                            video_base.video_configuration.set_settings_model(self.hardware_source, settings)
                            self.needs_saving_model.value = False

                    def revert(self, widget: UserInterface.Widget) -> None:
                        if self.settings:
                            self.settings.copy_from(self.settings_original)
                            self.needs_saving_model.value = False

                    def remove(self, widget: UserInterface.Widget) -> None:
                        video_base.video_configuration.remove_hardware_source(self.hardware_source)

                if component_id == "section":
                    return SectionHandler(container, item)

                raise ValueError("Unable to create component.")

            @property
            def resources(self) -> typing.Mapping[str, typing.Any]:
                u = Declarative.DeclarativeUI()

                driver_display_name_label = u.create_label(text="@binding(driver_display_name)")

                device_id_field = u.create_line_edit(text="@binding(settings.device_id)", width=180)

                display_name_field = u.create_line_edit(text="@binding(settings.name)", width=240)

                edit_group_content = u.create_component_instance("edit_group")

                edit_group = u.create_group(edit_group_content, margin=8)

                edit_row = u.create_row(edit_group, u.create_stretch())

                apply_button = u.create_push_button(name="apply_button", text=_("Apply"), on_clicked="apply")
                revert_button = u.create_push_button(name="revert_button", text=_("Revert"), on_clicked="revert")
                remove_button = u.create_push_button(text=_("Remove"), on_clicked="remove")
                remove_row = u.create_row(apply_button, revert_button, remove_button, u.create_stretch())

                label_column = u.create_column(u.create_label(text=_("Driver:")), u.create_label(text=_("Device ID:")), u.create_label(text=_("Display Name:")), spacing=4)
                field_column = u.create_column(driver_display_name_label, device_id_field, display_name_field, spacing=4)

                content_row = u.create_row(label_column, field_column, u.create_stretch(), spacing=12)

                content = u.create_column(content_row, edit_row, remove_row, spacing=8)

                component = u.define_component(content=content, component_id="section")

                return {"section": component}

        sources_column = u.create_column(items="video_sources.items", item_component_id="section", spacing=8)

        sources_content = u.create_scroll_area(u.create_column(sources_column, u.create_stretch()))

        video_source_types = [video_device_factory.display_name for video_device_factory in video_device_factories]

        video_source_type_combo = u.create_combo_box(items=video_source_types, current_index="@binding(video_source_type_index.value)")

        button_row = u.create_row(u.create_stretch(), video_source_type_combo, u.create_push_button(text=_("New"), on_clicked="create_new_video_device"), spacing=8)

        content = u.create_column(sources_content, button_row)

        return Declarative.DeclarativeWidget(ui, event_loop, Handler(content, self.__video_configuration.video_sources))


workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(VideoSourcePanel, "video-source-control-panel", _("Video Source"), ["left", "right"], "left")

video_preference_panel = VideoPreferencePanel(video_base.video_configuration)
PreferencesDialog.PreferencesManager().register_preference_pane(video_preference_panel)

hardware_source_added_event_listener: typing.Optional[Event.EventListener] = None
hardware_source_removed_event_listener: typing.Optional[Event.EventListener] = None

def run() -> None:
    global hardware_source_added_event_listener, hardware_source_removed_event_listener
    hardware_control_panels = dict()

    def register_hardware_panel(hardware_source: HardwareSource.HardwareSource) -> None:
        if hardware_source.features.get("is_video", False):
            panel_id = "video-control-panel-" + hardware_source.hardware_source_id
            hardware_control_panels[hardware_source.hardware_source_id] = panel_id

            class HardwareDisplayPanelControllerFactory:
                def __init__(self) -> None:
                    self.priority = 1

                def build_menu(self, display_type_menu: UserInterface.Menu, selected_display_panel: typing.Optional[DisplayPanel.DisplayPanel]) -> typing.Sequence[UserInterface.MenuAction]:
                    # return a list of actions that have been added to the menu.
                    def switch_to_live_controller(hardware_source: video_base.VideoHardwareSource) -> None:
                        d = {"type": "image", "controller_type": VideoDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                        if selected_display_panel:
                            selected_display_panel.change_display_panel_content(d)

                    action = display_type_menu.add_menu_item(hardware_source.display_name or _("Unknown"), functools.partial(switch_to_live_controller, hardware_source))
                    display_panel_controller = selected_display_panel.display_panel_controller if selected_display_panel else None
                    action.checked = isinstance(display_panel_controller, VideoDisplayPanelController) and display_panel_controller.hardware_source_id == hardware_source.hardware_source_id
                    return [action]

                def make_new(self, controller_type: str, display_panel: DisplayPanel.DisplayPanel, d: Persistence.PersistentDictType) -> typing.Optional[VideoDisplayPanelController]:
                    # make a new display panel controller, typically called to restore contents of a display panel.
                    # controller_type will match the type property of the display panel controller when it was saved.
                    # d is the dictionary that is saved when the display panel controller closes.
                    hardware_source_id = d.get("hardware_source_id")
                    if controller_type == VideoDisplayPanelController.type and hardware_source_id == hardware_source.hardware_source_id:
                        return VideoDisplayPanelController(display_panel, hardware_source_id)
                    return None

                def match(self, document_model: DocumentModel.DocumentModel, data_item: DataItem.DataItem) -> typing.Optional[Persistence.PersistentDictType]:
                    if HardwareSource.matches_hardware_source(hardware_source.hardware_source_id, None, document_model, data_item):
                        return {"controller_type": VideoDisplayPanelController.type, "hardware_source_id": hardware_source.hardware_source_id}
                    return None

            DisplayPanel.DisplayPanelManager().register_display_panel_controller_factory("video-live-" + hardware_source.hardware_source_id, HardwareDisplayPanelControllerFactory())

    def unregister_hardware_panel(hardware_source: HardwareSource.HardwareSource) -> None:
        if hardware_source.features.get("is_video", False):
            DisplayPanel.DisplayPanelManager().unregister_display_panel_controller_factory("video-live-" + hardware_source.hardware_source_id)

    hardware_source_added_event_listener = HardwareSource.HardwareSourceManager().hardware_source_added_event.listen(register_hardware_panel)
    hardware_source_removed_event_listener = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(unregister_hardware_panel)
    for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
        register_hardware_panel(hardware_source)
