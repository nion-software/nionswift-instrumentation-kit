# standard libraries
import asyncio
import copy
import gettext
import logging
import threading

# third party libraries
# None

# local libraries
from nion.swift.model import Graphics
from nion.utils import Binding
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Event


_ = gettext.gettext


class PropertyToGraphicBinding(Binding.PropertyBinding):

    """
        Binds a property of an operation item to a property of a graphic item.
    """

    def __init__(self, region, region_property_name, graphic, graphic_property_name):
        super().__init__(region, region_property_name)
        self.__graphic = graphic
        self.__graphic_property_changed_listener = graphic.property_changed_event.listen(self.__property_changed)
        self.__graphic_property_name = graphic_property_name
        self.__region_property_name = region_property_name
        def set_target_value(value):
            setattr(self.__graphic, graphic_property_name, value)
        self.target_setter = set_target_value

    def close(self):
        self.__graphic_property_changed_listener.close()
        self.__graphic_property_changed_listener = None
        self.__graphic = None
        super().close()

    # watch for property changes on the graphic.
    def __property_changed(self, property_name):
        if property_name == self.__graphic_property_name:
            old_property_value = getattr(self.source, self.__region_property_name)
            # to prevent message loops, check to make sure it changed
            property_value = getattr(self.__graphic, property_name)
            if property_value != old_property_value:
                self.update_source(property_value)


class ProbeGraphicConnection:
    """Manage the connection between the hardware and the graphics representing the probe on a display."""
    def __init__(self, display, probe_position_value, hide_probe_graphics_fn):
        self.display = display
        self.probe_position_value = probe_position_value
        self.graphic = None
        self.binding = None
        self.remove_region_graphic_event_listener = None
        self.hide_probe_graphics_fn = hide_probe_graphics_fn

    def close(self):
        graphic = self.graphic
        self.hide_probe_graphic()
        if graphic:
            self.display.remove_graphic(graphic)

    def update_probe_state(self, probe_position, static_probe_state):
        if probe_position is not None:
            self.probe_position_value.value = probe_position
            self.show_probe_graphic()
        else:
            graphic = self.graphic
            self.hide_probe_graphic()
            if graphic:
                self.display.remove_graphic(graphic)
        if self.graphic:
            self.graphic.color = "#FF0" if static_probe_state == "blanked" else "#F80"

    def show_probe_graphic(self):
        if not self.graphic:
            self.graphic = Graphics.PointGraphic()
            self.graphic.graphic_id = "probe"
            self.graphic.label = _("Probe")
            self.graphic.position = self.probe_position_value.value
            self.graphic.is_bounds_constrained = True
            self.display.add_graphic(self.graphic)
            self.binding = PropertyToGraphicBinding(self.probe_position_value, "value", self.graphic, "position")
            def graphic_removed():
                self.hide_probe_graphic()
                # next make sure all other probe graphics get hidden so that setting the probe_position_value
                # doesn't set graphics positions to None
                self.hide_probe_graphics_fn()
                self.probe_position_value.value = None
            def display_removed():
                self.hide_probe_graphic()
            self.remove_region_graphic_event_listener = self.graphic.about_to_be_removed_event.listen(graphic_removed)
            self.display_about_to_be_removed_listener = self.display.about_to_be_removed_event.listen(display_removed)

    def hide_probe_graphic(self):
        if self.graphic:
            self.binding.close()
            self.binding = None
            self.remove_region_graphic_event_listener.close()
            self.remove_region_graphic_event_listener = None
            self.display_about_to_be_removed_listener.close()
            self.display_about_to_be_removed_listener = None
            self.graphic = None


class STEMController:
    """An interface to a STEM microscope.

    Methods and properties starting with a single underscore are called internally and shouldn't be called by general
    clients.

    Methods starting with double underscores are private.

    Probe
    -----
    probe_state (parked, blanked, scanning)
    probe_position (fractional coordinates, optional)
    set_probe_position(probe_position)
    validate_probe_position()
    static_probe_state (parked, blanked)
    set_static_probe_state(static_probe_state)

    probe_state_changed_event (probe_state, probe_position, static_probe_state)
    """

    def __init__(self):
        self.__probe_position_value = Model.PropertyModel()
        self.__probe_position_value.on_value_changed = self.set_probe_position
        self.__probe_position = None
        self.__probe_state_stack = list()
        self.__probe_state_stack.append("parked")
        self.probe_state_changed_event = Event.Event()
        self.probe_data_items_changed_event = Event.Event()

    def close(self):
        pass

    def _enter_scanning_state(self):
        self.__probe_state_stack.append("scanning")
        self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)

    def _exit_scanning_state(self):
        self.__probe_state_stack.pop()
        self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)

    @property
    def probe_position(self):
        """ Return the probe position, in normalized coordinates with origin at top left. """
        return self.__probe_position

    def set_probe_position(self, probe_position):
        """ Set the probe position, in normalized coordinates with origin at top left. """
        if probe_position is not None:
            # convert the probe position to a FloatPoint and limit it to the 0.0 to 1.0 range in both axes.
            probe_position = Geometry.FloatPoint.make(probe_position)
            probe_position = Geometry.FloatPoint(y=max(min(probe_position.y, 1.0), 0.0),
                                                 x=max(min(probe_position.x, 1.0), 0.0))
        if ((self.__probe_position is None) != (probe_position is None)) or (self.__probe_position != probe_position):
            self.__probe_position = probe_position
            # update the probe position for listeners and also explicitly update for probe_graphic_connections.
            self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)

    def validate_probe_position(self):
        """Validate the probe position.

        This is called when the user switches from not controlling to controlling the position."""
        self.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5))

    @property
    def probe_position_value(self):
        return self.__probe_position_value

    def disconnect_probe_connections(self):
        self.probe_data_items_changed_event.fire(list())

    def _data_item_states_changed(self, data_item_states):
        if len(data_item_states) > 0:
            self.probe_data_items_changed_event.fire([data_item_state.get("data_item") for data_item_state in data_item_states])

    def set_static_probe_state(self, value: str) -> None:
        """Set the static probe state.

        Static probe state is the state of the probe when not scanning. Valid values are 'parked' or 'blanked'.
        """
        if self.probe_state != value:
            if value != "parked" and value != "blanked":
                logging.warning("static_probe_state must be 'parked' or 'blanked'")
                value = "parked"
            self.__probe_state_stack[0] = value
            self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)

    @property
    def static_probe_state(self):
        """Static probe state is the default when not scanning and can be 'blanked' or 'parked'."""
        return self.__probe_state_stack[0]

    @property
    def probe_state(self) -> str:
        """Probe state is the current probe state and can be 'blanked', 'parked', or 'scanning'."""
        return self.__probe_state_stack[-1]


class ProbeView:
    """Observes the probe (STEM controller) and updates data items and graphics."""

    def __init__(self, stem_controller: STEMController, event_loop: asyncio.AbstractEventLoop):
        assert event_loop is not None
        self.__event_loop = event_loop
        self.__last_data_items_lock = threading.RLock()
        self.__last_data_items = list()
        self.__probe_state = None
        self.__probe_graphic_connections = list()
        self.__probe_position_value = stem_controller.probe_position_value
        self.__probe_state_changed_listener = stem_controller.probe_state_changed_event.listen(self.__probe_state_changed)
        self.__probe_data_items_changed_listener = stem_controller.probe_data_items_changed_event.listen(self.__probe_data_items_changed)

    def close(self):
        self.__probe_data_items_changed_listener.close()
        self.__probe_data_items_changed_listener = None
        self.__probe_state_changed_listener.close()
        self.__probe_state_changed_listener = None
        self.__last_data_items = list()
        self.__event_loop = None

    def __probe_data_items_changed(self, data_items):
        # thread safe.
        with self.__last_data_items_lock:
            self.__last_data_items = copy.copy(data_items)

    def __probe_state_changed(self, probe_state, probe_position, static_probe_state):
        # thread safe. move actual call to main thread using the event loop.
        self.__event_loop.create_task(self.__update_probe_state(probe_state, probe_position, static_probe_state))

    async def __update_probe_state(self, probe_state, probe_position, static_probe_state):
        # thread unsafe. always called on main thread (via event loop).
        if probe_state != self.__probe_state:
            if probe_state == "scanning":
                self.__hide_probe_graphics()
            else:
                self.__show_probe_graphics(probe_position, static_probe_state)
            self.__probe_state = probe_state
        self.__update_probe_graphics(probe_position, static_probe_state)

    def __hide_probe_graphics(self):
        # thread unsafe.
        probe_graphic_connections = copy.copy(self.__probe_graphic_connections)
        self.__probe_graphic_connections = list()
        for probe_graphic_connection in probe_graphic_connections:
            probe_graphic_connection.close()

    def __show_probe_graphics(self, probe_position, static_probe_state):
        # thread unsafe.
        with self.__last_data_items_lock:
            data_items = self.__last_data_items
            # self.__last_data_items = list()
        for data_item in data_items:
            # scanning has stopped, figure out the displays that might be used to display the probe position
            # then watch for changes to that list. changes will include the display being removed by the user
            # or newer more appropriate displays becoming available.
            display = data_item.primary_display_specifier.display
            if display:
                # the probe position value object gives the ProbeGraphicConnection the ability to
                # get, set, and watch for changes to the probe position.
                probe_graphic_connection = ProbeGraphicConnection(display, self.__probe_position_value, self.__hide_probe_graphics)
                probe_graphic_connection.update_probe_state(probe_position, static_probe_state)
                self.__probe_graphic_connections.append(probe_graphic_connection)

    def __update_probe_graphics(self, probe_position, static_probe_state):
        # thread unsafe.
        for probe_graphic_connection in self.__probe_graphic_connections:
            probe_graphic_connection.update_probe_state(probe_position, static_probe_state)


from nion.swift.model import HardwareSource


class ProbeViewController:
    """Manage a ProbeView for each instrument (STEMController) that gets registered."""

    def __init__(self, event_loop):
        assert event_loop is not None
        self.__event_loop = event_loop
        # be sure to keep a reference or it will be closed immediately.
        self.__instrument_added_event_listener = None
        self.__instrument_removed_event_listener = None
        self.__instrument_added_event_listener = HardwareSource.HardwareSourceManager().instrument_added_event.listen(self.register_instrument)
        self.__instrument_removed_event_listener = HardwareSource.HardwareSourceManager().instrument_removed_event.listen(self.unregister_instrument)
        for instrument in HardwareSource.HardwareSourceManager().instruments:
            self.register_instrument(instrument)

    def close(self):
        # close will be called when the extension is unloaded. in turn, close any references so they get closed. this
        # is not strictly necessary since the references will be deleted naturally when this object is deleted.
        self.__instrument_added_event_listener.close()
        self.__instrument_added_event_listener = None
        self.__instrument_removed_event_listener.close()
        self.__instrument_removed_event_listener = None

    def register_instrument(self, instrument):
        if hasattr(instrument, "probe_position_value"):
            instrument._probe_view = ProbeView(instrument, self.__event_loop)

    def unregister_instrument(self, instrument):
        if hasattr(instrument, "_probe_view"):
            instrument._probe_view.close()
            instrument._probe_view = None


class STEMControllerExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.stem_controller"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        self.__probe_view_controller = ProbeViewController(api.application._application.event_loop)

    def close(self):
        self.__probe_view_controller.close()
        self.__probe_view_controller = None
