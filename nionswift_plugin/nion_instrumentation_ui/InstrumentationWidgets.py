"""Widgets shared by the instrumentation control panels."""

from __future__ import annotations

# standard libraries
import typing

# local libraries
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import DrawingContext
from nion.ui import UserInterface
from nion.ui import Widgets
from nion.ui import Window
from nion.utils import Binding
from nion.utils import Geometry
from nion.utils import Registry


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
