"""The dialog which configures the scan profiles of a scan device."""

from __future__ import annotations

# standard libraries
import asyncio
import functools
import gettext
import typing

# local libraries
from nion.instrumentation import scan_base
from nion.instrumentation import ScanProfiles
from nion.ui import Declarative
from nion.ui import Dialog
from nion.ui import UserInterface
from nion.ui import Widgets
from nion.ui import Window
from nion.utils import Converter
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import ReferenceCounting
from nionswift_plugin.nion_instrumentation_ui import InstrumentationWidgets


_ = gettext.gettext


class ScanRecorderLike(typing.Protocol):
    """The scan recording which the dialog uses to record once with a profile, such as that of the scan control panel."""

    @property
    def record_button_enabled(self) -> bool: ...

    def handle_record_button_clicked(self, *, frame_parameters: scan_base.ScanFrameParameters | None = None) -> None: ...


class ScanProfileIncludedColorConverter(Converter.ConverterLike[bool, str]):
    """Convert whether a scan profile includes a parameter to the color of the parameter's label, gray when not included."""

    def convert(self, value: bool | None) -> str | None:
        return "black" if value else "gray"

    def convert_back(self, formatted_value: str | None) -> bool | None:
        return formatted_value == "black"


class ProfileHandler(Declarative.Handler):
    """Display and edit one profile in the detail pane of the profile configuration dialog.

    Each parameter is a row of a check box for whether the profile includes it, its label, gray when it is not included,
    and an editor bound to the profile. The profile ignores an invalid or empty value typed into an editor, which then
    shows the unchanged value again.
    """

    INCLUDED_WIDTH: typing.Final[int] = 20
    PARAMETER_COLUMN_WIDTH: typing.Final[int] = 160
    VALUE_WIDTH: typing.Final[int] = 120
    LINE_EDIT_WIDTH: typing.Final[int] = 60

    def __init__(self, profile_configuration_model: ScanProfiles.ProfileConfigurationModel, profile_model: ScanProfiles.ProfileModel,
                 show_actions_menu: typing.Callable[[ScanProfiles.ProfileModel, Geometry.IntPoint], None]) -> None:
        super().__init__()
        self._profile_model = profile_model
        self.__profile_configuration_model = profile_configuration_model
        self.__show_actions_menu = show_actions_menu
        self.activate_button: UserInterface.PushButtonWidget | None = None
        self.name_line_edit: UserInterface.LineEditWidget | None = None
        self.field_width_line_edit: UserInterface.LineEditWidget | None = None
        self.width_line_edit: UserInterface.LineEditWidget | None = None
        self.height_line_edit: UserInterface.LineEditWidget | None = None
        self.pixel_time_us_line_edit: UserInterface.LineEditWidget | None = None
        self.rotation_line_edit: UserInterface.LineEditWidget | None = None
        self.subscan_field_width_nm_line_edit: UserInterface.LineEditWidget | None = None
        self.subscan_field_height_nm_line_edit: UserInterface.LineEditWidget | None = None
        self.subscan_position_x_percent_line_edit: UserInterface.LineEditWidget | None = None
        self.subscan_position_y_percent_line_edit: UserInterface.LineEditWidget | None = None
        self.subscan_pixel_width_line_edit: UserInterface.LineEditWidget | None = None
        self.subscan_rotation_line_edit: UserInterface.LineEditWidget | None = None
        self.__profile_configuration_model_listener = profile_configuration_model.property_changed_event.listen(
            ReferenceCounting.weak_partial(ProfileHandler.__handle_profile_configuration_model_property_changed, self)
        )

        # the converters used by the bindings of the editors and the labels below.
        self.included_color_converter = ScanProfileIncludedColorConverter()
        self.field_width_converter = Converter.FloatToStringConverter(format="{:.1f}")
        self.size_converter = Converter.IntegerToStringConverter()
        self.pixel_time_us_converter = Converter.FloatToStringConverter(format="{:.2f}")
        self.rotation_converter = Converter.FloatToStringConverter(format="{:.1f}")
        self.subscan_field_size_converter = Converter.FloatToStringConverter(format="{:.1f}")
        self.subscan_position_converter = Converter.FloatToStringConverter(format="{:.2f}")
        # an empty line edit gives None, which is an automatic subscan pixel width.
        self.subscan_pixel_width_converter = Converter.IntegerToStringConverter(pass_none=True)
        self.subscan_rotation_converter = Converter.FloatToStringConverter(format="{:.1f}")

        u = Declarative.DeclarativeUI()

        name_row = u.create_row(
            u.create_label(text=_("Name"), text_alignment_vertical="vcenter"),
            u.create_line_edit(name="name_line_edit", text="@binding(profile_name)", width=self.PARAMETER_COLUMN_WIDTH, tool_tip=_("The name of the scan profile.")),
            u.create_stretch(),
            u.create_push_button(text=_("Activate"), name="activate_button", tool_tip=_("Make this the active scan profile."), enabled="@binding(can_activate)", on_clicked="handle_activate_button"),
            u.create_push_button(text="\N{MIDLINE HORIZONTAL ELLIPSIS}", style="minimal", width=ProfileConfigurationHandler.LIST_BUTTON_TEXT_WIDTH,
                                 tool_tip=_("More actions for this scan profile."), on_clicked="handle_actions_button"),
            spacing=6,
        )

        parameters_table_header = u.create_row(
            u.create_spacing(self.INCLUDED_WIDTH),
            u.create_label(text=_("Parameter"), width=self.PARAMETER_COLUMN_WIDTH, text_alignment_horizontal="left"),
            u.create_label(text=_("Value"), width=self.VALUE_WIDTH, text_alignment_horizontal="left"),
            u.create_stretch(),
            spacing=6,
        )

        field_width_row = self.__create_parameter_row(
            u, "@binding(_profile_model.field_width_included)",
            self.__create_label(u, _("Field Width (nm)"), "@binding(_profile_model.field_width_included, converter=included_color_converter)"),
            self.__create_line_edit(u, "field_width_line_edit", "@binding(_profile_model.field_width, converter=field_width_converter)"),
        )

        # the width and height with the check box which links them.
        size_row = self.__create_parameter_row(
            u, "@binding(_profile_model.size_included)",
            self.__create_label_pair(u, _("Width (px)"), _("Height (px)"), "@binding(_profile_model.size_included, converter=included_color_converter)"),
            u.create_row(
                u.create_column(
                    self.__create_line_edit(u, "width_line_edit", "@binding(_profile_model.width, converter=size_converter)"),
                    self.__create_line_edit(u, "height_line_edit", "@binding(_profile_model.height, converter=size_converter)"),
                    spacing=2,
                ),
                InstrumentationWidgets.LinkedFieldsCheckBoxFactory.create_linked_fields_check_box(checked="@binding(_profile_model.size_linked)"),
                spacing=4,
            ),
        )

        pixel_time_us_row = self.__create_parameter_row(
            u, "@binding(_profile_model.pixel_time_us_included)",
            self.__create_label(u, _("Pixel Time (\N{MICRO SIGN}s)"), "@binding(_profile_model.pixel_time_us_included, converter=included_color_converter)"),
            self.__create_line_edit(u, "pixel_time_us_line_edit", "@binding(_profile_model.pixel_time_us, converter=pixel_time_us_converter)"),
        )

        rotation_row = self.__create_parameter_row(
            u, "@binding(_profile_model.rotation_included)",
            self.__create_label(u, _("Rotation (deg)"), "@binding(_profile_model.rotation_included, converter=included_color_converter)"),
            self.__create_line_edit(u, "rotation_line_edit", "@binding(_profile_model.rotation, converter=rotation_converter)"),
        )

        ac_line_sync_row = self.__create_parameter_row(
            u, "@binding(_profile_model.ac_line_sync_included)",
            self.__create_label(u, _("AC Line Sync"), "@binding(_profile_model.ac_line_sync_included, converter=included_color_converter)"),
            u.create_check_box(checked="@binding(_profile_model.ac_line_sync)"),
        )

        subscan_field_size_row = self.__create_parameter_row(
            u, "@binding(_profile_model.subscan_field_size_included)",
            self.__create_label_pair(u, _("Subscan Width (nm)"), _("Subscan Height (nm)"), "@binding(_profile_model.subscan_field_size_included, converter=included_color_converter)"),
            u.create_column(
                self.__create_line_edit(u, "subscan_field_width_nm_line_edit", "@binding(_profile_model.subscan_field_width_nm, converter=subscan_field_size_converter)"),
                self.__create_line_edit(u, "subscan_field_height_nm_line_edit", "@binding(_profile_model.subscan_field_height_nm, converter=subscan_field_size_converter)"),
                spacing=2,
            ),
        )

        subscan_fractional_position_row = self.__create_parameter_row(
            u, "@binding(_profile_model.subscan_fractional_position_included)",
            self.__create_label_pair(u, _("Subscan Position X (%)"), _("Subscan Position Y (%)"), "@binding(_profile_model.subscan_fractional_position_included, converter=included_color_converter)"),
            u.create_column(
                self.__create_line_edit(u, "subscan_position_x_percent_line_edit", "@binding(_profile_model.subscan_position_x_percent, converter=subscan_position_converter)"),
                self.__create_line_edit(u, "subscan_position_y_percent_line_edit", "@binding(_profile_model.subscan_position_y_percent, converter=subscan_position_converter)"),
                spacing=2,
            ),
        )

        subscan_pixel_width_row = self.__create_parameter_row(
            u, "@binding(_profile_model.subscan_pixel_width_included)",
            self.__create_label(u, _("Subscan Pixel Width (px)"), "@binding(_profile_model.subscan_pixel_width_included, converter=included_color_converter)",
                                tool_tip=_("Subscan pixel height is derived from subscan fractional size shape.")),
            self.__create_line_edit(u, "subscan_pixel_width_line_edit", "@binding(_profile_model.subscan_pixel_width, converter=subscan_pixel_width_converter)", placeholder_text=_("Auto")),
        )

        subscan_rotation_row = self.__create_parameter_row(
            u, "@binding(_profile_model.subscan_rotation_included)",
            self.__create_label(u, _("Subscan Rotation (deg)"), "@binding(_profile_model.subscan_rotation_included, converter=included_color_converter)"),
            self.__create_line_edit(u, "subscan_rotation_line_edit", "@binding(_profile_model.subscan_rotation, converter=subscan_rotation_converter)"),
        )

        parameters_table = u.create_column(
            parameters_table_header,
            u.create_divider(orientation="horizontal"),
            field_width_row,
            size_row,
            pixel_time_us_row,
            rotation_row,
            ac_line_sync_row,
            subscan_field_size_row,
            subscan_fractional_position_row,
            subscan_pixel_width_row,
            subscan_rotation_row,
            spacing=4,
        )

        self.ui_view = u.create_column(
            name_row,
            u.create_divider(orientation="horizontal"),
            parameters_table,
            u.create_stretch(),
            spacing=4,
            margin=4,
            margin_right=12,
        )

    def __create_parameter_row(self, u: Declarative.DeclarativeUI, included_binding: str, label: Declarative.UIDescriptionResult, editor: Declarative.UIDescriptionResult) -> Declarative.UIDescriptionResult:
        # the included check box stays at the top of a row with two lines.
        included_column = u.create_column(
            u.create_check_box(checked=included_binding, tool_tip=_("Include this parameter in the profile. A parameter which is not included keeps its current value when the profile is activated.")),
            u.create_stretch(),
            width=self.INCLUDED_WIDTH,
        )
        return u.create_row(included_column, label, u.create_row(editor, u.create_stretch(), width=self.VALUE_WIDTH, spacing=4), u.create_stretch(), spacing=6)

    def __create_label(self, u: Declarative.DeclarativeUI, text: str, color_binding: str, tool_tip: str | None = None) -> Declarative.UIDescriptionResult:
        return u.create_label(text=text, color=color_binding, tool_tip=tool_tip, width=self.PARAMETER_COLUMN_WIDTH, text_alignment_vertical="vcenter", text_alignment_horizontal="left")

    def __create_label_pair(self, u: Declarative.DeclarativeUI, first_text: str, second_text: str, color_binding: str) -> Declarative.UIDescriptionResult:
        return u.create_column(
            u.create_row(u.create_label(text=first_text, color=color_binding, text_alignment_vertical="vcenter", text_alignment_horizontal="left"), u.create_stretch(), width=self.PARAMETER_COLUMN_WIDTH, spacing=4),
            u.create_row(u.create_label(text=second_text, color=color_binding, text_alignment_vertical="vcenter", text_alignment_horizontal="left"), u.create_stretch(), width=self.PARAMETER_COLUMN_WIDTH, spacing=4),
            spacing=2,
        )

    def __create_line_edit(self, u: Declarative.DeclarativeUI, name: str, text_binding: str, placeholder_text: str | None = None) -> Declarative.UIDescriptionResult:
        return u.create_line_edit(name=name, text=text_binding, placeholder_text=placeholder_text, width=self.LINE_EDIT_WIDTH)

    def __handle_profile_configuration_model_property_changed(self, property_name: str) -> None:
        if property_name == "active_profile_index":
            self.notify_property_changed("can_activate")

    @property
    def can_activate(self) -> bool:
        return not self.__profile_configuration_model.is_active_profile(self._profile_model)

    @property
    def profile_name(self) -> str:
        return self._profile_model.name

    @profile_name.setter
    def profile_name(self, value: str) -> None:
        # the line edit shows the resulting name, which is the existing name if the new one is rejected.
        self.__profile_configuration_model.rename_profile(self._profile_model, value.strip())
        self.notify_property_changed("profile_name")

    def handle_activate_button(self, widget: UserInterface.Widget) -> None:
        self.__profile_configuration_model.active_profile_index = self.__profile_configuration_model.profiles.index(self._profile_model)

    def handle_actions_button(self, widget: UserInterface.Widget) -> None:
        # the menu opens below the button.
        self.__show_actions_menu(self._profile_model, widget.map_to_global(Geometry.IntPoint(y=widget.size.height, x=0)))


class ProfileRowHandler(Declarative.Handler):
    """Display one profile in the master list by its name, marking the active profile."""

    def __init__(self, profile_configuration_model: ScanProfiles.ProfileConfigurationModel, profile_model: ScanProfiles.ProfileModel) -> None:
        super().__init__()
        u = Declarative.DeclarativeUI()
        self._profile_model = profile_model
        self.__profile_configuration_model = profile_configuration_model
        self.__profile_configuration_model_listener = profile_configuration_model.property_changed_event.listen(
            ReferenceCounting.weak_partial(ProfileRowHandler.__handle_profile_configuration_model_property_changed, self)
        )
        row = u.create_row(
            u.create_label(text="@binding(active_marker)", width=14, tool_tip=_("The active scan profile.")),
            u.create_label(text="@binding(_profile_model.name)"),
            u.create_stretch(),
            spacing=4,
            margin=1,
        )
        # the list gives each row a fixed height taller than the row's content; the stretches center it vertically.
        self.ui_view = u.create_column(u.create_stretch(), row, u.create_stretch())

    def __handle_profile_configuration_model_property_changed(self, property_name: str) -> None:
        if property_name == "active_profile_index":
            self.notify_property_changed("active_marker")

    @property
    def active_marker(self) -> str:
        return "\N{CHECK MARK}" if self.__profile_configuration_model.is_active_profile(self._profile_model) else str()


class ProfileConfigurationHandler(Declarative.Handler):
    """Display the list of scan profiles, with buttons to manage the list, beside the detail pane of the selected one."""

    # the width of a minimal push button is the width of its text; the button adds its own margins around it.
    LIST_BUTTON_TEXT_WIDTH = 12

    def __init__(self, profile_configuration_model: ScanProfiles.ProfileConfigurationModel, scan_recorder: ScanRecorderLike, window: Window.Window) -> None:
        super().__init__()

        self._profile_configuration_model = profile_configuration_model
        self.__scan_recorder = scan_recorder
        self.__window = window
        self.profile_list_view: Widgets.ListViewWidget | None = None
        self.add_button: UserInterface.PushButtonWidget | None = None
        self.remove_button: UserInterface.PushButtonWidget | None = None
        self.duplicate_button: UserInterface.PushButtonWidget | None = None
        self.move_up_button: UserInterface.PushButtonWidget | None = None
        self.move_down_button: UserInterface.PushButtonWidget | None = None

        self.current_index_model = Model.PropertyModel(profile_configuration_model.active_profile_index)
        self.__current_index_listener = self.current_index_model.property_changed_event.listen(
            ReferenceCounting.weak_partial(ProfileConfigurationHandler.__handle_list_state_changed, self)
        )
        self.__profile_configuration_model_listener = profile_configuration_model.property_changed_event.listen(
            ReferenceCounting.weak_partial(ProfileConfigurationHandler.__handle_list_state_changed, self)
        )

        u = Declarative.DeclarativeUI()

        # choosing a profile in the list, by double click or return, activates it. a right click opens its actions menu.
        master = u.create_list_view(items="_profile_configuration_model.profiles",
                                     item_component_id="row",
                                     item_height=22,
                                     name="profile_list_view",
                                     current_index="@binding(current_index_model.value)",
                                     on_item_selected="handle_profile_chosen",
                                     on_item_handle_context_menu="handle_profile_context_menu",
                                     width=180, min_height=320, size_policy_vertical="expanding")

        LIST_BUTTON_TEXT_WIDTH = ProfileConfigurationHandler.LIST_BUTTON_TEXT_WIDTH
        list_buttons = u.create_row(
            u.create_push_button(text="+", name="add_button", style="minimal", width=LIST_BUTTON_TEXT_WIDTH, tool_tip=_("Add a scan profile using the current frame parameters."), on_clicked="handle_add_button"),
            u.create_push_button(text="\N{MINUS SIGN}", name="remove_button", style="minimal", width=LIST_BUTTON_TEXT_WIDTH, tool_tip=_("Delete the selected scan profile. The active scan profile cannot be deleted."), enabled="@binding(can_remove)", on_clicked="handle_remove_button"),
            u.create_push_button(text="\N{TWO JOINED SQUARES}", name="duplicate_button", style="minimal", width=LIST_BUTTON_TEXT_WIDTH, tool_tip=_("Add a copy of the selected scan profile."), enabled="@binding(has_selection)", on_clicked="handle_duplicate_button"),
            u.create_stretch(),
            u.create_push_button(text="\N{UPWARDS ARROW}", name="move_up_button", style="minimal", width=LIST_BUTTON_TEXT_WIDTH, tool_tip=_("Move the selected scan profile up."), enabled="@binding(can_move_up)", on_clicked="handle_move_up_button"),
            u.create_push_button(text="\N{DOWNWARDS ARROW}", name="move_down_button", style="minimal", width=LIST_BUTTON_TEXT_WIDTH, tool_tip=_("Move the selected scan profile down."), enabled="@binding(can_move_down)", on_clicked="handle_move_down_button"),
            spacing=4,
        )

        # the detail panes are constructed as they are first displayed, rather than all of them up front.
        detail = u.create_stack(items="_profile_configuration_model.profiles",
                                 item_component_id="detail",
                                 current_index="@binding(current_index_model.value)",
                                 item_construction="deferred")

        self.ui_view = u.create_row(u.create_column(master, list_buttons, spacing=4), u.create_column(detail), spacing=12)

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> Declarative.HandlerLike | None:
        if component_id == "row":
            return ProfileRowHandler(self._profile_configuration_model, typing.cast(ScanProfiles.ProfileModel, item))
        if component_id == "detail":
            return ProfileHandler(self._profile_configuration_model, typing.cast(ScanProfiles.ProfileModel, item), self.show_profile_actions_menu)
        return None

    def __handle_list_state_changed(self, property_name: str) -> None:
        for name in ("has_selection", "can_remove", "can_move_up", "can_move_down"):
            self.notify_property_changed(name)

    @property
    def __selected_index(self) -> int | None:
        index = self.current_index_model.value
        return index if index is not None and 0 <= index < len(self._profile_configuration_model.profiles) else None

    @property
    def has_selection(self) -> bool:
        return self.__selected_index is not None

    @property
    def can_remove(self) -> bool:
        index = self.__selected_index
        return index is not None and self._profile_configuration_model.can_remove_profile(index)

    @property
    def can_move_up(self) -> bool:
        index = self.__selected_index
        return index is not None and index > 0

    @property
    def can_move_down(self) -> bool:
        index = self.__selected_index
        return index is not None and index < len(self._profile_configuration_model.profiles) - 1

    def handle_profile_chosen(self, widget: UserInterface.Widget, current_index: int) -> bool:
        if 0 <= current_index < len(self._profile_configuration_model.profiles):
            self._profile_configuration_model.active_profile_index = current_index
            return True
        return False

    def handle_profile_context_menu(self, widget: UserInterface.Widget, x: int, y: int, gx: int, gy: int, index: int | None) -> bool:
        if index is not None and 0 <= index < len(self._profile_configuration_model.profiles):
            self.current_index_model.value = index
            self.show_profile_actions_menu(self._profile_configuration_model.profiles[index], Geometry.IntPoint(y=gy, x=gx))
            return True
        return False

    def show_profile_actions_menu(self, profile: ScanProfiles.ProfileModel, global_point: Geometry.IntPoint) -> None:
        """Show the menu of actions for the profile at the global point."""
        profiles = self._profile_configuration_model.profiles
        menu = self.__window.create_context_menu()
        activate_action = menu.add_menu_item(_("Activate"), functools.partial(self.__activate_profile, profile))
        activate_action.enabled = not self._profile_configuration_model.is_active_profile(profile)
        record_action = menu.add_menu_item(_("Record Once"), functools.partial(self.__record_profile, profile))
        record_action.enabled = self.__scan_recorder.record_button_enabled
        menu.add_separator()
        menu.add_menu_item(_("Duplicate"), functools.partial(self.__duplicate_profile, profile))
        remove_action = menu.add_menu_item(_("Delete"), functools.partial(self.__remove_profile, profile))
        remove_action.enabled = self._profile_configuration_model.can_remove_profile(profiles.index(profile))
        menu.popup(global_point.x, global_point.y)

    # the menu actions look up the profile index when chosen, since the list may change while the menu is open.

    def __activate_profile(self, profile: ScanProfiles.ProfileModel) -> None:
        if profile in self._profile_configuration_model.profiles:
            self._profile_configuration_model.active_profile_index = self._profile_configuration_model.profiles.index(profile)

    def __record_profile(self, profile: ScanProfiles.ProfileModel) -> None:
        # records once with the profile without activating it. a recording may have started while the menu was open.
        if profile in self._profile_configuration_model.profiles and self.__scan_recorder.record_button_enabled:
            self.__scan_recorder.handle_record_button_clicked(frame_parameters=self._profile_configuration_model.make_frame_parameters(profile))

    def __duplicate_profile(self, profile: ScanProfiles.ProfileModel) -> None:
        if profile in self._profile_configuration_model.profiles:
            index = self._profile_configuration_model.profiles.index(profile)
            self._profile_configuration_model.duplicate_profile(index)
            self.current_index_model.value = index + 1

    def __remove_profile(self, profile: ScanProfiles.ProfileModel) -> None:
        if profile in self._profile_configuration_model.profiles:
            index = self._profile_configuration_model.profiles.index(profile)
            if self._profile_configuration_model.can_remove_profile(index):
                self._profile_configuration_model.remove_profile(index)
                self.current_index_model.value = min(index, len(self._profile_configuration_model.profiles) - 1)

    def handle_add_button(self, widget: UserInterface.Widget) -> None:
        index = self.__selected_index
        new_index = index + 1 if index is not None else len(self._profile_configuration_model.profiles)
        self._profile_configuration_model.insert_new_profile(new_index)
        self.current_index_model.value = new_index

    def handle_duplicate_button(self, widget: UserInterface.Widget) -> None:
        if (index := self.__selected_index) is not None:
            self.__duplicate_profile(self._profile_configuration_model.profiles[index])

    def handle_remove_button(self, widget: UserInterface.Widget) -> None:
        if (index := self.__selected_index) is not None:
            self.__remove_profile(self._profile_configuration_model.profiles[index])

    def handle_move_up_button(self, widget: UserInterface.Widget) -> None:
        if (index := self.__selected_index) is not None and index > 0:
            self._profile_configuration_model.move_profile(index, index - 1)
            self.current_index_model.value = index - 1

    def handle_move_down_button(self, widget: UserInterface.Widget) -> None:
        if (index := self.__selected_index) is not None and index < len(self._profile_configuration_model.profiles) - 1:
            self._profile_configuration_model.move_profile(index, index + 1)
            self.current_index_model.value = index + 1


class ProfileConfigurationDialog(Dialog.ActionDialog):
    """The dialog which configures the scan profiles of a scan device, and which closes when escape is pressed."""

    def __init__(self, ui: UserInterface.UserInterface, event_loop: asyncio.AbstractEventLoop, title: str,
                 profile_configuration_model: ScanProfiles.ProfileConfigurationModel, scan_recorder: ScanRecorderLike, *,
                 hardware_source_id: str, parent_window: Window.Window | None = None) -> None:
        # the persistent id gives each scan device its own remembered position and size.
        super().__init__(ui, title, parent_window=parent_window, persistent_id="ScanProfilesDialog/" + hardware_source_id)
        self.content.add(Declarative.DeclarativeWidget(ui, event_loop, ProfileConfigurationHandler(profile_configuration_model, scan_recorder, self)))

    def key_pressed(self, key: UserInterface.Key) -> bool:
        if key.is_escape:
            self.request_close()
            return True
        return super().key_pressed(key)
