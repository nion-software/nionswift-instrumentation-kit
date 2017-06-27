# standard libraries
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
        self.target_setter = lambda value: setattr(self.__graphic, graphic_property_name, value)

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
    def __init__(self, display, probe_position_value):
        self.display = display
        self.probe_position_value = probe_position_value
        self.graphic = None
        self.binding = None
        self.remove_region_graphic_event_listener = None

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
                self.probe_position_value.value = None
            self.remove_region_graphic_event_listener = self.graphic.about_to_be_removed_event.listen(graphic_removed)
            self.display_about_to_be_removed_listener = self.display.about_to_be_removed_event.listen(graphic_removed)

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
    probe_state
    probe_position
    set_probe_position
    validate_probe_position
    static_probe_state
    set_static_probe_state

    probe_state_changed_event (probe_state, probe_position, static_probe_state)
    """

    def __init__(self):
        self.__last_data_items = list()
        self.__probe_position = None
        self.__probe_state_stack = list()
        self.__probe_state_stack.append("parked")
        self.__probe_state_updates = list()
        self.__probe_state_updates_lock = threading.RLock()
        self.__probe_graphic_connections = list()
        self.probe_state_changed_event = Event.Event()

    def close(self):
        self.__last_data_items = list()

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

    def set_probe_position(self, probe_position, call_soon_fn):
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
            with self.__probe_state_updates_lock:
                for probe_graphic_connection in self.__probe_graphic_connections:
                    self.__probe_state_updates.append((probe_graphic_connection, self.probe_position, self.static_probe_state))
            call_soon_fn(self.__update_probe_states)

    def validate_probe_position(self, call_soon_fn):
        """Validate the probe position.

        This is called when the user switches from not controlling to controlling the position."""
        self.set_probe_position(Geometry.FloatPoint(y=0.5, x=0.5), call_soon_fn)

    def disconnect_probe_connections(self, call_soon_fn):
        probe_graphic_connections = copy.copy(self.__probe_graphic_connections)
        self.__probe_graphic_connections = list()
        for probe_graphic_connection in probe_graphic_connections:
            call_soon_fn(probe_graphic_connection.close)

    def _data_item_states_changed(self, data_item_states, call_soon_fn):
        if len(data_item_states) == 0:
            for data_item in self.__last_data_items:
                # scanning has stopped, figure out the displays that might be used to display the probe position
                # then watch for changes to that list. changes will include the display being removed by the user
                # or newer more appropriate displays becoming available.
                display = data_item.primary_display_specifier.display
                if display:
                    # the probe position value object gives the ProbeGraphicConnection the ability to
                    # get, set, and watch for changes to the probe position.
                    probe_position_value = Model.PropertyModel()
                    probe_position_value.on_value_changed = lambda value: self.set_probe_position(value, call_soon_fn)
                    probe_graphic_connection = ProbeGraphicConnection(display, probe_position_value)
                    with self.__probe_state_updates_lock:
                        self.__probe_state_updates.append((probe_graphic_connection, self.probe_position, self.static_probe_state))
                    call_soon_fn(self.__update_probe_states)
                    self.__probe_graphic_connections.append(probe_graphic_connection)
        else:
            # scanning, remove all known probe graphic connections.
            self.disconnect_probe_connections(call_soon_fn)

        self.__last_data_items = [data_item_state.get("data_item") for data_item_state in data_item_states]

    def set_static_probe_state(self, value: str, call_soon_fn) -> None:
        """Set the static probe state.

        Static probe state is the state of the probe when not scanning. Valid values are 'parked' or 'blanked'.
        """
        if self.probe_state != value:
            if value != "parked" and value != "blanked":
                logging.warning("static_probe_state must be 'parked' or 'blanked'")
                value = "parked"
            self.__probe_state_stack[0] = value
            self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)
            with self.__probe_state_updates_lock:
                for probe_graphic_connection in self.__probe_graphic_connections:
                    self.__probe_state_updates.append((probe_graphic_connection, self.probe_position, self.static_probe_state))
            call_soon_fn(self.__update_probe_states)

    @property
    def static_probe_state(self):
        """Static probe state is the default when not scanning and can be 'blanked' or 'parked'."""
        return self.__probe_state_stack[0]

    @property
    def probe_state(self) -> str:
        """Probe state is the current probe state and can be 'blanked', 'parked', or 'scanning'."""
        return self.__probe_state_stack[-1]

    def __update_probe_states(self):
        with self.__probe_state_updates_lock:
            probe_state_updates = self.__probe_state_updates
            self.__probe_state_updates = list()
        for probe_graphic_connection, probe_position, static_probe_state in probe_state_updates:
            probe_graphic_connection.update_probe_state(probe_position, static_probe_state)
