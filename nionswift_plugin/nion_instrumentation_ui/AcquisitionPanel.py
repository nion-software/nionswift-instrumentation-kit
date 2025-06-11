"""Acquisition panel.

The acquisition panel allows the user to select an acquisition method and an associated acquisition device,
which may be a direct device (camera or scan) or a virtual device (synchronized scan/camera).

All choices in the UI are persistent. The code supporting persistence is written so that it is easy to change
the persistent behavior to be profile- or project- based. It is currently file based. The persistence code is
based on schema/entity from nionswift. This closely ties this panel to nionswift schema/entity behavior, which
is still evolving.

The time/space usage is calculated based on the acquisition method and the device used. The device method watches for
changed parameters and fires an event when the time/space usage changes. The acquisition method listens to the device
time/space usage changed event and its own parameters, and then calculates its own time/space usage based on the
device time/space usage, and then fires its event. The UI watches for the acquisition method time/space usage changed
event and updates the UI accordingly. The acquisition method must be able to ask the device for its time/space usage
based on exposure time since the acquisition method may modify the low level exposure time as it implements its
method."""

from __future__ import annotations

# system imports
import asyncio
import copy
import dataclasses
import functools
import gettext
import logging
import math
import operator
import pathlib
import pkgutil
import typing
import weakref

import numpy.typing

# local libraries
from nion.instrumentation import Acquisition
from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import camera_base
from nion.instrumentation import DataChannel
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller as STEMController
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import Schema
from nion.ui import Application
from nion.ui import CanvasItem
from nion.ui import Declarative
from nion.ui import PreferencesDialog
from nion.ui import UserInterface as UserInterfaceModule
from nion.utils import Converter
from nion.utils import Event
from nion.utils import ListModel
from nion.utils import Model
from nion.utils import Observable
from nion.utils.ReferenceCounting import weak_partial
from nion.utils import Recorder
from nion.utils import Registry
from nion.utils import Stream

from . import HardwareSourceChoice

_ = gettext.gettext

# define SI units used in this module
units_multiplier = {
    "": 1,
    "nm": 1E9,
    "um": 1E6,
    "mm": 1E3,
    "m": 1,
    "eV": 1,
    "meV": 1E3,
    "ns": 1E9,
    "us": 1E6,
    "ms": 1E3,
    "s": 1,
}


class ComponentHandler(Declarative.Handler):
    """Define an interface for component handler to be used with component combo box handler."""

    def __init__(self, display_name: str):
        super().__init__()
        self.display_name = display_name

    def __str__(self) -> str:
        return self.display_name


class ComboBoxHandler(Declarative.Handler):
    """Declarative component handler for combo box based on observable list.

    Also facilitate reading/writing the selected item identifier from/to a property model.

    container is the containing object; items_key is the key within that object. The property associated with
    the key should be an observable list.

    sort_key and filter are used to order and limit the entries.

    id_getter and selection_storage_model are used to read/write the selected item identifier.
    """

    def __init__(self, container: Observable.Observable, items_key: str, sort_key: ListModel.OptionalSortKeyCallable,
                 filter: typing.Optional[ListModel.Filter], id_getter: typing.Callable[[typing.Any], str],
                 selection_storage_model: Model.PropertyModel[str]) -> None:
        super().__init__()
        # create a filtered list model with the sort key and filter key.
        self.sorted_items = ListModel.FilteredListModel(container=container, items_key=items_key)
        self.sorted_items.sort_key = sort_key
        if filter:
            self.sorted_items.filter = filter

        # create an observable property model based on the sorted items.
        self.item_list = ListModel.ListPropertyModel(self.sorted_items)

        # create an index model for the combo box.
        self.selected_index_model = Model.PropertyModel[int](0)

        # create a value stream for the selected item. this is useful in cases where other UI items
        # need to adjust themselves based on the selected value.
        self.selected_item_value_stream = Stream.ValueStream[typing.Any]().add_ref()

        # update the selected item. this function should not refer to self.
        def update_selected_item(c: ListModel.ListPropertyModel, index_model: Model.PropertyModel[int],
                                 v: Stream.ValueStream[int], k: str) -> None:
            index = index_model.value or 0
            item = c.value[index] if 0 <= index < len(c.value) else None
            v.value = item
            if item:
                selection_storage_model.value = id_getter(item)

        self.__selected_component_index_listener = self.selected_index_model.property_changed_event.listen(
            functools.partial(update_selected_item, self.item_list, self.selected_index_model,
                              self.selected_item_value_stream))

        # read the identifier from storage and match it to an item and make that item the selected one
        selected_item_id = selection_storage_model.value
        for index, item in enumerate(self.sorted_items.items or list()):
            if id_getter(item) == selected_item_id:
                self.selected_index_model.value = index

        update_selected_item(self.item_list, self.selected_index_model, self.selected_item_value_stream, "value")

        # build the UI
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_combo_box(items_ref="@binding(item_list.value)",
                                          current_index="@binding(selected_index_model.value)")

    def close(self) -> None:
        self.__selected_component_index_listener.close()
        self.__selected_component_index_listener = typing.cast(typing.Any, None)
        self.selected_item_value_stream.remove_ref()
        self.selected_item_value_stream = typing.cast(typing.Any, None)
        self.selected_index_model.close()
        self.selected_index_model = typing.cast(typing.Any, None)
        self.item_list.close()
        self.item_list = typing.cast(typing.Any, None)
        self.sorted_items.close()
        self.sorted_items = typing.cast(typing.Any, None)
        super().close()

    @property
    def current_item(self) -> typing.Any:
        index = self.selected_index_model.value or 0
        return self.sorted_items.items[index]


class ComponentComboBoxHandler(Declarative.Handler):
    """Declarative component handler for a set of registry components.

    Also facilitate reading/writing the component instances and selected item identifier from/to an entity.

    component-base is a string to use for locating the registry components. the component-base + "-factory"
    is used to find the components. individual components are identified by have component-base as their
    prefix.

    title is used to label the combo box.

    configuration is a schema entity with properties of component_id_key (a string used to read/write the current
    component identifier) and components_key (a string used to maintain the list of component instances).
    """

    def __init__(self, component_base: str, title: str, configuration: Schema.Entity, preferences: Observable.Observable, component_id_key: str, components_key: str, extra_top_right: typing.Optional[Declarative.HandlerLike] = None) -> None:
        super().__init__()

        # store these values for bookkeeping
        self.__component_name = component_base
        self.__component_factory_name = f"{component_base}-factory"

        self.__extra_top_right = extra_top_right

        # create a list model for the component handlers
        self.__components = ListModel.ListModel[ComponentHandler]()

        # create a map of component id's to component handlers. used for looking up components by id for the
        # acquisition method component stream, used in time/space usage calculation.
        self._component_handler_map = dict[str, AcquisitionMethodComponentHandler]()

        # and a model to store the selected component id. this is a property model based on observing
        # the component_id_key of the configuration entity.
        self.__selected_component_id_model = Model.PropertyChangedPropertyModel[str](configuration, component_id_key)

        # make a map from each component id to the associated component.
        component_map: typing.Dict[str, Schema.Entity] = dict()
        for component_entity in configuration._get_array_items(components_key):
            component_map[component_entity.entity_type.entity_id] = component_entity

        # construct the components.
        # put this above combo box so the selection gets made correctly from existing items
        # iterate through the components, creating them. look for a corresponding entry
        # in the component_map, creating a new entity if it doesn't exist. then assign the
        # entity to the component.
        for component_factory in Registry.get_components_by_type(self.__component_factory_name):
            component_id = f"{component_base}-{component_factory.component_id}".replace("-", "_")
            component_entity = component_map.get(component_id)
            if not component_entity:
                entity_type = Schema.get_entity_type(component_id)
                component_entity = entity_type.create() if entity_type else None
                if component_entity:
                    configuration._append_item(components_key, component_entity)
            component = component_factory(component_entity, preferences)
            self.__components.append_item(component)
            self._component_handler_map[component.component_id] = component

        def sort_key(o: typing.Any) -> typing.Any:
            return "Z" + o.display_name if o.display_name != "None" else "A"

        # make a combo box handler.
        # this gets closed by the declarative machinery.
        self._combo_box_handler = ComboBoxHandler(self.__components, "items", sort_key,
                                                  None, operator.attrgetter("component_id"),
                                                  self.__selected_component_id_model)
        # must delete the components if they are not added to another widget.
        self.__combo_box_handler_to_delete: typing.Optional[ComboBoxHandler] = self._combo_box_handler

        # create a value stream for the selected item. this is useful in cases where other UI items
        # need to adjust themselves based on the selected value.
        # this is merely a reference and does not need to be closed.
        self.selected_item_value_stream = self._combo_box_handler.selected_item_value_stream

        # TODO: listen for components being registered/unregistered

        u = Declarative.DeclarativeUI()

        extras = list()
        if extra_top_right:
            extras.append(u.create_component_instance(identifier="extra"))

        component_type_row = u.create_row(
            u.create_label(text=title),
            u.create_component_instance(identifier="combo_box"),
            u.create_stretch(),
            *extras,
            spacing=8
        )
        component_page = u.create_stack(
            items="_combo_box_handler.sorted_items.items",
            item_component_id=self.__component_name,
            current_index="@binding(_combo_box_handler.selected_index_model.value)",
            size_policy_vertical="preferred"
        )
        self.ui_view = u.create_column(component_type_row, component_page, spacing=8, size_policy_vertical="maximum")

    def close(self) -> None:
        self.__selected_component_id_model.close()
        self.__selected_component_id_model = typing.cast(typing.Any, None)
        if self.__combo_box_handler_to_delete:
            self.__combo_box_handler_to_delete.close()
            self.__combo_box_handler_to_delete = None
            for component in self.__components.items:
                component.close()
        self.__components.close()
        self.__components = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == self.__component_name:
            return typing.cast(Declarative.HandlerLike, item)
        if component_id == "combo_box":
            self.__combo_box_handler_to_delete = None
            return self._combo_box_handler
        if component_id == "extra":
            return self.__extra_top_right
        return None

    @property
    def current_item(self) -> typing.Any:
        return self._combo_box_handler.current_item


class AcquisitionMethodComponentHandler(ComponentHandler):
    """Define methods for acquisition method components.

    Also, define a time/space usage event that can be used to notify the UI of changes in the time/space usage.

    Track the device component, listen to its time space usage changed event, and fire the time space usage event
    for this object when it changes.

    Define a _fire_time_space_usage_changed_event that can be fired by subclasses when they detect a change in
    time/space usage.

    Define a _get_time_space_usage method that subclasses can override to return the time/space usage specific to this
    acquisition method.
    """

    def __init__(self, display_name: str, configuration: Schema.Entity) -> None:
        super().__init__(display_name)
        self.__device_component: AcquisitionDeviceComponentHandler | None = None
        self.__time_space_usage_changed_event_listener: Event.EventListener | None = None
        self.time_space_usage_changed_event = Event.Event()

    def set_device_component(self, device_component: AcquisitionDeviceComponentHandler | None) -> None:
        # set the device component for this acquisition method. called by TimeSpaceUsageStream, which tracks the
        # method/device required for time/space usage. this is needed because the acquisition method may use the
        # device to know the unit of time/space usage being rolled into the time/space usage for the method.
        self.__device_component = device_component
        if device_component:
            self.__time_space_usage_changed_event_listener = device_component.time_space_usage_changed_event.listen(weak_partial(AcquisitionMethodComponentHandler.__time_space_usage_changed, self))
            self.time_space_usage_changed_event.fire()
        else:
            self.__time_space_usage_changed_event_listener = None
            self.time_space_usage_changed_event.fire()

    def __time_space_usage_changed(self) -> None:
        # when the device's time space usage changes, fire the same event from this handler.
        self.time_space_usage_changed_event.fire()

    def get_time_space_usage(self) -> TimeSpaceUsage:
        device_component = self.__device_component
        if device_component:
            return self._get_time_space_usage(device_component)
        else:
            return TimeSpaceUsage(None, None)

    def _fire_time_space_usage_changed_event(self) -> None:
        self.time_space_usage_changed_event.fire()

    def _get_time_space_usage(self, device_component: AcquisitionDeviceComponentHandler) -> TimeSpaceUsage:
        return device_component.get_time_space_usage()

    def build_acquisition_method(self) -> Acquisition.AcquisitionMethodLike:
        # build the device.
        raise NotImplementedError()


class BasicAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Basic acquisition method - single acquire from acquisition device with no options.

    Produces a data stream directly from the acquisition device.
    """

    component_id = "basic-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("None"), configuration)
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            spacing=8
        )

    def build_acquisition_method(self) -> Acquisition.AcquisitionMethodLike:
        return Acquisition.BasicAcquisitionMethod()


class SequenceAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Sequence acquisition method - a sequence of acquires from acquisition device with no options.

    Produces a data stream that is a sequence of the acquisition device data stream.

    The configuration entity should have an integer 'count' field.
    """

    component_id = "sequence-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("Sequence"), configuration)
        self.configuration = configuration

        # the sequence listens for the count property of the configuration and fires a time space usage changed event
        # when the count changes.
        self.__configuration_property_changed_event_listener = configuration.property_changed_event.listen(weak_partial(SequenceAcquisitionMethodComponentHandler.__handle_configuration_property_changed, self))

        self.count_converter = Converter.IntegerToStringConverter()

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(
                u.create_label(text=_("Count")),
                u.create_line_edit(text="@binding(configuration.count, converter=count_converter)", width=90),
                u.create_stretch(),
                spacing=8
            ),
            spacing=8
        )

    def __handle_configuration_property_changed(self, name: str) -> None:
        if name == "count":
            self._fire_time_space_usage_changed_event()

    def _get_time_space_usage(self, device_component: AcquisitionDeviceComponentHandler) -> TimeSpaceUsage:
        unit_time_space_usage = device_component.get_time_space_usage()
        return TimeSpaceUsage(unit_time_space_usage.time * self.configuration.count,
                              unit_time_space_usage.space * self.configuration.count)

    def build_acquisition_method(self) -> Acquisition.AcquisitionMethodLike:
        length = max(1, self.configuration.count) if self.configuration.count else 1
        return Acquisition.SequenceAcquisitionMethod(length)


class SeriesControlHandler(Declarative.Handler):
    """Declarative component handler for count/start/step control UI.

    control_customization is the control being controlled.

    control_values is an entity with fields for control_id, count, start_value, and step_value.

    label is an optional UI label for the row.
    """

    def __init__(self, control_customization: AcquisitionPreferences.ControlCustomization, control_values: Schema.Entity, label: typing.Optional[str]):
        super().__init__()
        self.__control_customization = control_customization
        self.control_values = control_values
        self.count_converter = Converter.IntegerToStringConverter()
        control_description = control_customization.control_description
        assert control_description
        self.value_converter = Converter.PhysicalValueToStringConverter(control_description.units,
                                                                        units_multiplier[control_description.units],
                                                                        "{:.1f}")
        u = Declarative.DeclarativeUI()
        row_items = list()
        # the widths and spacing are typically duplicated in the header; so take care when changing to change the
        # headers too.
        row_items.append(u.create_spacing(20))
        if label is not None:
            row_items.append(u.create_label(text=label, width=28))
        row_items.append(u.create_line_edit(text="@binding(control_values.count, converter=count_converter)", width=90))
        row_items.append(u.create_line_edit(text="@binding(control_values.start_value, converter=value_converter)", width=90))
        row_items.append(u.create_line_edit(text="@binding(control_values.step_value, converter=value_converter)", width=90))
        row_items.append(u.create_stretch())
        self.ui_view = u.create_row(*row_items, spacing=8)

    def get_control_values(self) -> numpy.typing.NDArray[typing.Any]:
        control_description = self.__control_customization.control_description
        assert control_description
        count = max(1, self.control_values.count) if self.control_values.count else 1
        start = (self.control_values.start_value or 0)
        step = (self.control_values.step_value or 0)
        multiplier = control_description.multiplier
        return numpy.stack([numpy.fromfunction(lambda x: (start + step * x) * multiplier, (count,))], axis=-1)


def get_control_values(configuration: Schema.Entity, control_values_list_key: str, control_customization: AcquisitionPreferences.ControlCustomization, value_index: typing.Optional[int] = None) -> Schema.Entity:
    """Extract the control values for the control customization from the configuration under control_values_list_key.

    The value_index can be used to index into the default value for the control, if required.
    """
    control_id = control_customization.control_id
    # make a reverse map from control_id's in the configuration to control_values.
    m = dict()
    for c in configuration._get_array_items(control_values_list_key):
        m[c.control_id] = c
    # lookup existing control values in the map matching the control customization.
    control_values = m.get(control_id)
    # if no control values, create a new set.
    if not control_values:
        control_description = control_customization.control_description
        assert control_description
        value = control_description.default_value
        control_values = ControlValuesSchema.create(None, {"control_id": control_id, "count": 1,
                                                           "start_value": value[value_index] if value_index is not None else value,
                                                           "step_value": 0.0})
        configuration._append_item(control_values_list_key, control_values)
    return control_values


class SeriesAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Series acquisition method - a sequence of acquires from acquisition device with a changing parameter.

    Produces a data stream that is a sequence of the acquisition device data stream.

    The configuration entity should have a field for control_id and control_values_list. The control values list
    should have entities with fields for control_id, count, start_value, step_value. There should be one entry in
    the control values list for each possible control.
    """

    component_id = "series-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("1D Ramp"), configuration)
        self.configuration = configuration
        # the control UI is constructed as a stack with one item for each control_id.
        # the control_handlers is a map from the control_id to the SeriesControlHandler
        # for the control.
        self.__control_handlers = dict[str, SeriesControlHandler]()
        # listen to the control handler property changed events so that we can fire the time space usage changed event
        self.__control_handler_property_changed_event_listeners = dict[str, Event.EventListener]()
        # the selection storage model is a property model made by observing the control_id in the configuration.
        self.__selection_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "control_id")
        # update time space usage when the selection changes
        self.__selection_storage_model_listener = self.__selection_storage_model.property_changed_event.listen(weak_partial(SeriesAcquisitionMethodComponentHandler.__handle_control_values_property_changed, self))
        # the control combo box handler gives a choice of which control to use. in this case, the controls are iterated
        # by looking at control customizations. only 1d controls are presented.
        def filter_1d_control_customizations(control_customization: AcquisitionPreferences.ControlCustomization) -> bool:
            return control_customization.control_description is not None and (str(control_customization.control_description.control_type) == "1d")

        self._control_combo_box_handler = ComboBoxHandler(preferences,
                                                          "control_customizations",
                                                          operator.attrgetter("name"),
                                                          ListModel.PredicateFilter(filter_1d_control_customizations),
                                                          operator.attrgetter("control_id"),
                                                          self.__selection_storage_model)
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(
                u.create_label(text=_("Control")),
                u.create_component_instance(identifier="control_combo_box"),
                u.create_stretch(),
                spacing=8
            ),
            u.create_row(
                u.create_spacing(20),
                u.create_label(text=_("Count"), width=90),
                u.create_label(text=_("Start"), width=90),
                u.create_label(text=_("Step"), width=90),
                u.create_stretch(),
                spacing=8
            ),
            u.create_stack(
                items="_control_combo_box_handler.sorted_items.items",
                item_component_id="series-control",
                current_index="@binding(_control_combo_box_handler.selected_index_model.value)",
                size_policy_vertical="preferred"
            ),
            spacing=8
        )

    def close(self) -> None:
        self.__selection_storage_model_listener = typing.cast(typing.Any, None)
        self.__selection_storage_model = typing.cast(typing.Any, None)
        self.__control_handlers = typing.cast(typing.Any, None)
        self.__control_handler_property_changed_event_listeners = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "control_combo_box":
            return self._control_combo_box_handler
        if component_id == "series-control":
            # make a SeriesControlHandler for each control and store it into the control_handlers map.
            # control_customization is the dynamic customization of a static control_description.
            # control_values specify a series of values to be used during acquisition: count, start, step.
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_id = control_customization.control_id
            assert control_id not in self.__control_handlers
            control_values = get_control_values(self.configuration, "control_values_list", control_customization)
            self.__control_handlers[control_id] = SeriesControlHandler(control_customization, control_values, None)
            # messy bit to listen to control_values
            self.__control_handler_property_changed_event_listeners[control_id] = control_values.property_changed_event.listen(weak_partial(SeriesAcquisitionMethodComponentHandler.__handle_control_values_property_changed, self))
            return self.__control_handlers[control_id]
        return None

    def __handle_control_values_property_changed(self, name: str) -> None:
        self._fire_time_space_usage_changed_event()

    def _get_time_space_usage(self, device_component: AcquisitionDeviceComponentHandler) -> TimeSpaceUsage:
        unit_time_space_usage = device_component.get_time_space_usage()
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_values = get_control_values(self.configuration, "control_values_list", control_customization)
            count = control_values.count
            unit_time = unit_time_space_usage.time
            unit_space = unit_time_space_usage.space
            if count > 0 and unit_time is not None and unit_time > 0.0 and unit_space is not None and unit_space > 0:
                OVERHEAD: typing.Final[float] = 0.2  # overhead time for setting the control values
                unit_time = unit_time + OVERHEAD
                return TimeSpaceUsage(unit_time * count, unit_space * count)
        return TimeSpaceUsage(None, None)

    def build_acquisition_method(self) -> Acquisition.AcquisitionMethodLike:
        # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
        # start by getting the selected control customization from the UI.
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            # get the associated control handler that was created in create_handler and used within the stack
            # of control handlers declarative components.
            control_handler = self.__control_handlers.get(control_customization.control_id)
            if control_handler:
                # get the control values range from the control handler.
                return Acquisition.SeriesAcquisitionMethod(control_customization, control_handler.get_control_values())
        return Acquisition.BasicAcquisitionMethod(_("Series"))


class TableauAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Tableau acquisition method - a grid of acquires from acquisition device with a changing 2d parameter.

    Produces a data stream that is a 2d collection of the acquisition device data stream.

    The configuration entity should have a field for control_id, axis_id, x_control_values_list, and
    y_control_values_list. The control values lists should have entities with fields for control_id, count, start_value,
    step_value. There should be one entry in each control values list for each possible control.
    """

    component_id = "tableau-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("2D Ramp"), configuration)
        self.configuration = configuration
        # the control UIs are constructed as a stack with one item for each control_id.
        # the control_handlers is a map from the control_id to the SeriesControlHandler
        # for the control.
        self.__x_control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        self.__y_control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        self.__x_control_handler_property_changed_event_listeners = dict[str, Event.EventListener]()
        self.__y_control_handler_property_changed_event_listeners = dict[str, Event.EventListener]()
        # the selection storage model is a property model made by observing the control_id in the configuration.
        self.__selection_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "control_id")
        # update time space usage when the selection changes
        self.__selection_storage_model_listener = self.__selection_storage_model.property_changed_event.listen(weak_partial(TableauAcquisitionMethodComponentHandler.__handle_control_values_property_changed, self))
        # the control combo box handler gives a choice of which control to use. in this case, the controls are iterated
        # by looking at control customizations. only 2d controls are presented.
        def filter_2d_control_customizations(control_customization: AcquisitionPreferences.ControlCustomization) -> bool:
            return control_customization.control_description is not None and (str(control_customization.control_description.control_type) == "2d")

        self._control_combo_box_handler = ComboBoxHandler(preferences,
                                                          "control_customizations",
                                                          operator.attrgetter("name"),
                                                          ListModel.PredicateFilter(filter_2d_control_customizations),
                                                          operator.attrgetter("control_id"),
                                                          self.__selection_storage_model)
        # the axis storage model is a property model made by observing the axis_id in the configuration.
        self.__axis_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "axis_id")
        stem_controller = Registry.get_component("stem_controller")
        assert stem_controller
        # the axis combo box handler gives a choice of which axis to use. the axes are sourced from the stem controller.
        self._axis_combo_box_handler = ComboBoxHandler(stem_controller, "axis_descriptions",
                                                       operator.attrgetter("display_name"), None,
                                                       operator.attrgetter("axis_id"), self.__axis_storage_model)
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(
                u.create_label(text=_("Control")),
                u.create_component_instance(identifier="control_combo_box"),
                u.create_stretch(),
                spacing=8),
            u.create_row(
                u.create_label(text=_("Axis")),
                u.create_component_instance(identifier="axis_combo_box"),
                u.create_stretch(),
                spacing=8),
            u.create_row(
                u.create_spacing(20),
                u.create_spacing(28),
                u.create_label(text=_("Count"), width=90),
                u.create_label(text=_("Start"), width=90),
                u.create_label(text=_("Step"), width=90),
                u.create_stretch(),
                spacing=8
            ),
            u.create_stack(
                items="_control_combo_box_handler.sorted_items.items",
                item_component_id="x-control",
                current_index="@binding(_control_combo_box_handler.selected_index_model.value)",
                size_policy_vertical="preferred"
            ),
            u.create_stack(
                items="_control_combo_box_handler.sorted_items.items",
                item_component_id="y-control",
                current_index="@binding(_control_combo_box_handler.selected_index_model.value)",
                size_policy_vertical="preferred"
            ),
            spacing=8
        )

    def close(self) -> None:
        self.__selection_storage_model_listener = typing.cast(typing.Any, None)
        self.__selection_storage_model = typing.cast(typing.Any, None)
        self.__x_control_handlers = typing.cast(typing.Any, None)
        self.__y_control_handlers = typing.cast(typing.Any, None)
        self.__x_control_handler_property_changed_event_listeners = typing.cast(typing.Any, None)
        self.__y_control_handler_property_changed_event_listeners = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "control_combo_box":
            return self._control_combo_box_handler
        if component_id == "axis_combo_box":
            return self._axis_combo_box_handler
        if component_id == "y-control":
            # make a SeriesControlHandler for each y-control and store it into the control_handlers map.
            # control_customization is the dynamic customization of a static control_description.
            # control_values specify a series of values to be used during acquisition: count, start, step.
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_id = control_customization.control_id
            assert control_id not in self.__y_control_handlers
            control_values = get_control_values(self.configuration, "y_control_values_list", control_customization, 0)
            self.__y_control_handlers[control_id] = SeriesControlHandler(control_customization, control_values, "Y")
            # messy bit to listen to control_values
            self.__y_control_handler_property_changed_event_listeners[control_id] = control_values.property_changed_event.listen(weak_partial(TableauAcquisitionMethodComponentHandler.__handle_control_values_property_changed, self))
            return self.__y_control_handlers[control_id]
        if component_id == "x-control":
            # make a SeriesControlHandler for each x-control and store it into the control_handlers map.
            # control_customization is the dynamic customization of a static control_description.
            # control_values specify a series of values to be used during acquisition: count, start, step.
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_id = control_customization.control_id
            assert control_id not in self.__x_control_handlers
            control_values = get_control_values(self.configuration, "x_control_values_list", control_customization, 1)
            self.__x_control_handlers[control_id] = SeriesControlHandler(control_customization, control_values, "X")
            # messy bit to listen to control_values
            self.__x_control_handler_property_changed_event_listeners[control_id] = control_values.property_changed_event.listen(weak_partial(TableauAcquisitionMethodComponentHandler.__handle_control_values_property_changed, self))
            return self.__x_control_handlers[control_id]
        return None

    def __handle_control_values_property_changed(self, name: str) -> None:
        self._fire_time_space_usage_changed_event()

    def _get_time_space_usage(self, device_component: AcquisitionDeviceComponentHandler) -> TimeSpaceUsage:
        unit_time_space_usage = device_component.get_time_space_usage()
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            y_control_values = get_control_values(self.configuration, "y_control_values_list", control_customization, 0)
            x_control_values = get_control_values(self.configuration, "x_control_values_list", control_customization, 1)
            x_count = x_control_values.count
            y_count = y_control_values.count
            unit_time = unit_time_space_usage.time
            unit_space = unit_time_space_usage.space
            if x_count > 0 and y_count > 0 and unit_time is not None and unit_time > 0.0 and unit_space is not None and unit_space > 0:
                OVERHEAD: typing.Final[float] = 0.2  # overhead time for setting the control values
                unit_time = unit_time + OVERHEAD
                return TimeSpaceUsage(unit_time * x_count * y_count, unit_space * x_count * y_count)
        return TimeSpaceUsage(None, None)

    def build_acquisition_method(self) -> Acquisition.AcquisitionMethodLike:
        # given an acquisition data stream, wrap this acquisition method around the acquisition data stream.
        # start by getting the selected control customization from the UI.
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            # get the associated control handlers that were created in create_handler and used within the stack
            # of control handlers declarative components.
            x_control_handler = self.__x_control_handlers.get(control_customization.control_id)
            y_control_handler = self.__y_control_handlers.get(control_customization.control_id)
            if x_control_handler and y_control_handler:
                # get the axis and control values ranges from the control handlers.
                axis_id = self.__axis_storage_model.value
                y_control_values = y_control_handler.get_control_values()
                x_control_values = x_control_handler.get_control_values()
                control_values = numpy.stack(numpy.meshgrid(y_control_values, x_control_values, indexing='ij'), axis=-1)
                return Acquisition.TableAcquisitionMethod(control_customization, axis_id, control_values)
        return Acquisition.BasicAcquisitionMethod(_("Tableau"))


class MultiAcquireEntryHandler(Declarative.Handler):
    """Declarative component handler for a section in a multiple acquire method component."""

    def __init__(self, container: typing.Any, item: Schema.Entity):
        super().__init__()
        self.offset_converter = Converter.PhysicalValueToStringConverter("eV", units_multiplier["eV"], "{:.0f}")
        self.exposure_converter = Converter.PhysicalValueToStringConverter("ms", units_multiplier["ms"], "{:.1f}")
        self.count_converter = Converter.IntegerToStringConverter()
        self.item = item
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_row(
            u.create_line_edit(text="@binding(item.offset, converter=offset_converter)", width=86),
            u.create_line_edit(text="@binding(item.exposure, converter=exposure_converter)", width=86),
            u.create_line_edit(text="@binding(item.count, converter=count_converter)", width=86),
            u.create_check_box(checked="@binding(item.include_sum)", width=42),
            u.create_stretch(),
            spacing=8
        )


class MultipleAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Multiple acquisition method - a sequential set of acquires from acquisition device with a control and exposure.

    Currently, the control is always energy offset.

    Produces multiple data streams that are sequences of the acquisition device data stream.

    The configuration entity should have a list of sections where each section is an entity with offset, exposure,
    and count fields.
    """

    component_id = "multiple-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("Multiple"), configuration)
        self.configuration = configuration
        # ensure that there are always a few example sections.
        if len(self.configuration.sections) == 0:
            self.configuration._append_item("sections", MultipleAcquireEntrySchema.create(None, {"offset": 0.0, "exposure": 0.001, "count": 2}))
            self.configuration._append_item("sections", MultipleAcquireEntrySchema.create(None, {"offset": 10.0, "exposure": 0.01, "count": 3}))

        # track sections inserted/removed for time/space usage calculation
        self.__item_inserted_listener = self.configuration.item_inserted_event.listen(weak_partial(MultipleAcquisitionMethodComponentHandler.__handle_section_inserted, self))
        self.__item_removed_listener = self.configuration.item_removed_event.listen(weak_partial(MultipleAcquisitionMethodComponentHandler.__handle_section_removed, self))

        self.__section_property_changed_listeners = list[Event.EventListener]()

        for index, section in enumerate(self.configuration.sections):
            self.__handle_section_inserted("sections", section, index)

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(
                u.create_label(text=_("Offset"), width=86),
                u.create_label(text=_("Exposure"), width=86),
                u.create_label(text=_("Frames"), width=86),
                u.create_label(text=_("Sum"), width=42),
                u.create_stretch(),
                spacing=8
            ),
            u.create_column(items="configuration.sections", item_component_id="section", spacing=8),
            u.create_row(
                u.create_push_button(text="+", on_clicked="add"),
                u.create_push_button(text="-", on_clicked="remove"),
                u.create_stretch(),
                spacing=8
            ),
        )

    def __handle_section_inserted(self, key: str, value: Schema.Entity, index: int) -> None:
        # this is called when a section is inserted into the configuration.
        # it is used to update the time/space usage.
        if key == "sections":
            self.__section_property_changed_listeners.insert(index, self.configuration.sections[index].property_changed_event.listen(weak_partial(MultipleAcquisitionMethodComponentHandler.__handle_section_property_changed, self)))
            self._fire_time_space_usage_changed_event()

    def __handle_section_removed(self, key: str, value: Schema.Entity, index: int) -> None:
        # this is called when a section is inserted into the configuration.
        # it is used to update the time/space usage.
        if key == "sections":
            self.__section_property_changed_listeners.pop(index)
            self._fire_time_space_usage_changed_event()

    def __handle_section_property_changed(self, name: str) -> None:
        # this is called when a section is changed in the configuration.
        # it is used to update the time/space usage.
        if name == "include_sum" or name == "count":
            self._fire_time_space_usage_changed_event()

    def _get_time_space_usage(self, device_component: AcquisitionDeviceComponentHandler) -> TimeSpaceUsage:
        unit_time_space_usage = device_component.get_time_space_usage()
        unit_time = unit_time_space_usage.time
        unit_space = unit_time_space_usage.space
        if unit_time is not None and unit_time > 0.0 and unit_space is not None and unit_space > 0:
            time = 0.0
            space = 0
            for section in self.configuration.sections:
                count = section.count
                if count > 0:
                    time += unit_time * count
                    space += unit_space * count
                if section.include_sum:
                    space += unit_space
            return TimeSpaceUsage(time, space)
        return TimeSpaceUsage(None, None)

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "section":
            assert container is not None
            assert item is not None
            return MultiAcquireEntryHandler(container, item)
        return None

    def add(self, widget: UserInterfaceModule.Widget) -> None:
        # handle add section request. always adds a new section with offset=0, exposure=1ms, count=2.
        self.configuration._append_item("sections", MultipleAcquireEntrySchema.create(None, {"offset": 0.0, "exposure": 0.001, "count": 2}))

    def remove(self, widget: UserInterfaceModule.Widget) -> None:
        # handle remove section request. always removes the last section.
        self.configuration._remove_item("sections", self.configuration._get_array_item("sections", -1))

    def build_acquisition_method(self) -> Acquisition.AcquisitionMethodLike:
        # given an acquisition data stream, wrap this acquisition method around the acquisition data stream.
        # start by getting the selected control customization from the UI.
        return Acquisition.MultipleAcquisitionMethod(self.configuration.sections)


# register each component as an acquisition method component factory.
Registry.register_component(BasicAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(SequenceAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(SeriesAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(TableauAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(MultipleAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})


class HardwareSourceChoiceModel(Observable.Observable):
    def __init__(self, configuration: Schema.Entity, device_id_key: str, filter: typing.Optional[typing.Callable[[HardwareSource.HardwareSource], bool]] = None, *, force_enabled: bool = False) -> None:
        super().__init__()

        # the hardware source choice model is a property model made by observing the device_id_key in the configuration.
        self.__hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, device_id_key)

        # the hardware source choice associates a device id with a hardware source and also facilitates a combo box.
        # it will not be presented in the UI unless multiple choices exist.
        self.__hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__hardware_source_choice_model, filter)

        # a model that indicates whether the hardware source choice should be presented in the UI.
        def is_enabled(x: typing.Optional[typing.List[typing.Optional[HardwareSource.HardwareSource]]]) -> bool:
            return force_enabled or len(x or list()) > 1

        self.ui_enabled_model = Model.StreamValueModel(Stream.MapStream(
            Stream.PropertyChangedEventStream(self.__hardware_source_choice.hardware_sources_model, "value"),
            is_enabled))

        def property_changed(handler: HardwareSourceChoiceHandler, property: str) -> None:
            handler.notify_property_changed("hardware_source_display_names")

        # use weak_partial to avoid self reference and facilitate no-close.
        self.__listener = self.__hardware_source_choice.hardware_sources_model.property_changed_event.listen(weak_partial(property_changed, self))

    def close(self) -> None:
        self.__listener = typing.cast(typing.Any, None)
        self.ui_enabled_model = typing.cast(typing.Any, None)
        self.__hardware_source_choice = typing.cast(typing.Any, None)
        self.__hardware_source_choice_model = typing.cast(typing.Any, None)

    @property
    def hardware_source_choice(self) -> HardwareSourceChoice.HardwareSourceChoice:
        return self.__hardware_source_choice

    @property
    def hardware_source(self) -> typing.Optional[HardwareSource.HardwareSource]:
        return self.__hardware_source_choice.hardware_source

    @property
    def hardware_source_display_names(self) -> typing.List[str]:
        return [hardware_source.display_name if hardware_source else _("None") for hardware_source in self.hardware_source_choice.hardware_sources]


class HardwareSourceChoiceHandler(Declarative.Handler):
    """A declarative component handler for a hardware source choice.

    Includes a combo box to configure the hardware source. The combo box is hidden if there is only one choice and
    the hardware source is not forced to be enabled.

    hardware_source_display_names is a read-only list of strings. It is an observable property.
    """

    def __init__(self, hardware_choice_model: HardwareSourceChoiceModel, *, title: typing.Optional[str] = None) -> None:
        super().__init__()

        self._hardware_choice_model = hardware_choice_model

        u = Declarative.DeclarativeUI()
        if title:
            self.ui_view = u.create_row(
                u.create_label(text=title),
                u.create_combo_box(items_ref="@binding(_hardware_choice_model.hardware_source_display_names)", current_index="@binding(_hardware_choice_model.hardware_source_choice.hardware_source_index_model.value)"),
                u.create_stretch(),
                spacing=8,
                visible="@binding(_hardware_choice_model.ui_enabled_model.value)"
            )
        else:
            self.ui_view = u.create_row(
                u.create_combo_box(items_ref="@binding(_hardware_choice_model.hardware_source_display_names)", current_index="@binding(_hardware_choice_model.hardware_source_choice.hardware_source_index_model.value)"),
                visible="@binding(_hardware_choice_model.ui_enabled_model.value)"
            )

    def close(self) -> None:
        self._hardware_choice_model = typing.cast(typing.Any, None)
        super().close()


@dataclasses.dataclass
class TimeSpaceUsage:
    """A class to define the time and space requirements of a component."""
    time: float | None
    space: int | None


class AcquisitionDeviceComponentHandler(ComponentHandler):
    """Define methods for acquisition device components."""

    component_id: str

    acquire_valid_value_stream: Stream.AbstractStream[bool]

    # subclasses must fire this event when the time or space requirements of the device component change.
    time_space_usage_changed_event: Event.Event

    def get_time_space_usage(self, camera_exposure_time: float | None = None) -> TimeSpaceUsage:
        # get the time and space requirements of the device component.
        raise NotImplementedError()

    def build_acquisition_device(self) -> Acquisition.AcquisitionDeviceLike:
        # build the device.
        raise NotImplementedError()


class ScanSpecifierValueStream(Stream.ValueStream[STEMController.ScanSpecifier]):
    """A value stream of the synchronized scan description of the latest values of a camera hardware source stream and
    scan hardware source stream.

    Listens to both hardware source streams for changes and updates the synchronized scan description as required.
    """
    def __init__(self, camera_hardware_source_stream: Stream.AbstractStream[HardwareSource.HardwareSource], scan_hardware_source_stream: Stream.AbstractStream[HardwareSource.HardwareSource], scan_width_model: Model.PropertyModel[int], event_loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.__event_loop = event_loop
        self.__camera_hardware_source_stream = camera_hardware_source_stream.add_ref()
        self.__scan_hardware_source_stream = scan_hardware_source_stream.add_ref()
        self.__scan_width_model = scan_width_model
        self.__hardware_source_stream_listener = self.__scan_hardware_source_stream.value_stream.listen(
            weak_partial(ScanSpecifierValueStream.__hardware_source_stream_changed, self))

        def property_changed(vs: ScanSpecifierValueStream, property: str) -> None:
            vs.__update_context()

        self.__scan_width_changed_listener = self.__scan_width_model.property_changed_event.listen(
            weak_partial(property_changed, self))
        self.__stem_controller = typing.cast(STEMController.STEMController, Registry.get_component("stem_controller"))
        self.__stem_controller_property_listener: typing.Optional[Event.EventListener] = None
        self.__scan_context_changed_listener: typing.Optional[Event.EventListener] = None
        if self.__stem_controller:
            self.__stem_controller_property_listener = self.__stem_controller.property_changed_event.listen(
                weak_partial(ScanSpecifierValueStream.__stem_controller_property_changed, self))
            self.__scan_context_changed_listener = self.__stem_controller.scan_context_changed_event.listen(
                weak_partial(ScanSpecifierValueStream.__scan_context_changed, self))
        self.__update_context()

    def about_to_delete(self) -> None:
        if self.__stem_controller_property_listener:
            self.__stem_controller_property_listener.close()
            self.__stem_controller_property_listener = None
        if self.__scan_context_changed_listener:
            self.__scan_context_changed_listener.close()
            self.__scan_context_changed_listener = None
        self.__hardware_source_stream_listener.close()
        self.__hardware_source_stream_listener = typing.cast(typing.Any, None)
        self.__scan_width_changed_listener.close()
        self.__scan_width_changed_listener = typing.cast(typing.Any, None)
        self.__scan_hardware_source_stream.remove_ref()
        self.__scan_hardware_source_stream = typing.cast(typing.Any, None)
        self.__camera_hardware_source_stream.remove_ref()
        self.__camera_hardware_source_stream = typing.cast(typing.Any, None)
        super().about_to_delete()

    def __hardware_source_stream_changed(self, hardware_source: HardwareSource.HardwareSource) -> None:
        if hardware_source and hardware_source.features.get("is_scanning"):
            self.__update_context()

    def __stem_controller_property_changed(self, key: str) -> None:
        # this can be triggered from a thread, so use call soon to transfer it to the UI thread.
        if key in ("subscan_state", "subscan_region", "subscan_rotation", "line_scan_state", "line_scan_vector", "drift_channel_id", "drift_region", "drift_settings"):
            self.__event_loop.call_soon_threadsafe(self.__update_context)

    def __scan_context_changed(self) -> None:
        # this can be triggered from a thread, so use call soon to transfer it to the UI thread.
        self.__event_loop.call_soon_threadsafe(self.__update_context)

    def __update_context(self) -> None:
        maybe_camera_hardware_source = self.__camera_hardware_source_stream.value
        maybe_scan_hardware_source = self.__scan_hardware_source_stream.value
        if maybe_scan_hardware_source:
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, maybe_scan_hardware_source)

            if maybe_camera_hardware_source:
                camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, maybe_camera_hardware_source)
                exposure_time = camera_hardware_source.get_frame_parameters(0).exposure_ms / 1000
            else:
                exposure_time = scan_hardware_source.get_frame_parameters(0).pixel_time_us / 1E6
            scan_width = self.__scan_width_model.value or 32
            assert scan_width is not None

            scan_specifier = STEMController.ScanSpecifier()
            scan_specifier.update(scan_hardware_source, exposure_time * 1000, scan_width, 1, False)
            self.send_value(scan_specifier)


class DeviceSettingsModel(Observable.Observable):
    """A base model of the device settings and related values of a hardware source stream.

    Listens to the hardware_source_stream for changes. And then listens to the current hardware source
    for parameter changes.
    """
    def __init__(self, configuration: Schema.Entity, device_id_key: str, filter: typing.Optional[typing.Callable[[HardwareSource.HardwareSource], bool]]) -> None:
        super().__init__()
        # the device hardware source choice model.
        self.__hardware_source_choice_model = HardwareSourceChoiceModel(configuration, device_id_key, filter, force_enabled=True)
        self.__hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self.__hardware_source_choice_model.hardware_source_choice)
        self.__hardware_source_stream.add_ref()  # separate so hardware_source_stream has correct type

        # use weak_partial to avoid self reference and facilitate no-close.
        self.__hardware_source_stream_listener = self.__hardware_source_stream.value_stream.listen(weak_partial(DeviceSettingsModel.__hardware_source_stream_changed, self))
        self.__frame_parameters_changed_listener: typing.Optional[Event.EventListener] = None

        hardware_source_stream = self.__hardware_source_stream

        def finalize(hardware_source_choice_model: HardwareSourceChoiceModel) -> None:
            hardware_source_stream.remove_ref()
            hardware_source_choice_model.close()

        weakref.finalize(self, functools.partial(finalize, self.__hardware_source_choice_model))

    def _finish_init(self) -> None:
        self.__hardware_source_stream_changed(self.hardware_source)

    def __hardware_source_stream_changed(self, hardware_source: typing.Optional[HardwareSource.HardwareSource]) -> None:
        # when the hardware source choice changes, update the frame parameters listener. close the old one.
        if self.__frame_parameters_changed_listener:
            self.__frame_parameters_changed_listener.close()
            self.__frame_parameters_changed_listener = None
        if hardware_source and hardware_source.features.get("is_camera"):
            # reconfigure listener. use weak_partial to avoid self reference and facilitate no-close.
            self.__frame_parameters_changed_listener = hardware_source.current_frame_parameters_changed_event.listen(
                weak_partial(DeviceSettingsModel.__frame_parameters_changed, self, hardware_source))
            self._hardware_source_changed()
            self._update_settings(hardware_source)

    def _hardware_source_changed(self) -> None:
        # subclass should override to handle hardware source change.
        # subclass should not call.
        pass

    def _update_settings(self, hardware_source: HardwareSource.HardwareSource) -> None:
        # subclass should override to handle settings change, but no hardware source change.
        # subclass can call if a configuration value changes.
        pass

    def __frame_parameters_changed(self, hardware_source: HardwareSource.HardwareSource, frame_parameters: HardwareSource.FrameParameters) -> None:
        self._update_settings(hardware_source)

    @property
    def hardware_source(self) -> typing.Optional[HardwareSource.HardwareSource]:
        return self.__hardware_source_stream.value

    @property
    def hardware_source_choice(self) -> HardwareSourceChoice.HardwareSourceChoice:
        return self.__hardware_source_choice_model.hardware_source_choice

    @property
    def hardware_source_choice_model(self) -> HardwareSourceChoiceModel:
        return self.__hardware_source_choice_model

    @property
    def hardware_source_choice_stream(self) -> HardwareSourceChoice.HardwareSourceChoiceStream:
        return self.__hardware_source_stream


exposure_units = {0: "s", -1: "s", -2: "s", -3: "ms", -4: "ms", -5: "ms", -6: "us", -7: "us", -8: "us", -9: "ns", -10: "ns", -11: "ns"}
exposure_format = {0: ".1", -1: ".1", -2: ".2", -3: ".1", -4: ".1", -5: ".2", -6: ".1", -7: ".1", -8: ".2", -9: ".1", -10: ".1", -11: ".2"}

def make_exposure_str(exposure: float, exposure_precision: int) -> str:
    format_str = f"{{0:{exposure_format[exposure_precision]}f}}"
    return str(format_str.format(exposure / math.pow(10, math.trunc(exposure_precision / 3) * 3)))


class CameraSettingsModel(DeviceSettingsModel):
    """A model of the camera exposure and related values of a hardware source stream.

    Listens to the hardware_source_stream for changes. And then listens to the current hardware source
    for parameter changes. Sends out new exposure_time, exposure_precision, exposure_units_str, and exposure_str
    values when changed.

    channel_descriptions is a list of channel descriptions. It is a read-only observable property.
    channel_index is the selected index. It is a read/write observable property.
    """
    def __init__(self, configuration: Schema.Entity) -> None:
        super().__init__(configuration, "camera_device_id", lambda hardware_source: hardware_source.features.get("is_camera", False))

        # the camera exposure model is a property model made by observing camera_exposure in the configuration.
        self.__camera_config_exposure_model = Model.PropertyChangedPropertyModel[float](configuration, "camera_exposure")

        # the camera hardware source channel model is a property model made by observing the camera_channel_id in the configuration.
        self.__camera_hardware_source_channel_model = Model.PropertyChangedPropertyModel[str](configuration, "camera_channel_id")

        # internal values
        self.__exposure_time = 0.0
        self.__exposure_precision = 0
        self.__exposure_str: typing.Optional[str] = None
        self.__exposure_placeholder_str = str()
        self.__exposure_units_str = str()
        self.__channel_descriptions: typing.List[Acquisition.HardwareSourceChannelDescription] = list()

        # use weak_partial to avoid self reference and facilitate no-close.
        self.__exposure_changed_listener = self.__camera_config_exposure_model.property_changed_event.listen(weak_partial(CameraSettingsModel.__handle_exposure_changed, self))

        self._finish_init()

        self.__handle_exposure_changed("value")
        self.__update_channel_descriptions()

    def __handle_exposure_changed(self, property: str) -> None:
        if property == "value":
            if self.hardware_source:
                self._update_settings(self.hardware_source)

    @property
    def camera_frame_parameters(self) -> typing.Optional[camera_base.CameraFrameParameters]:
        camera_hardware_source = self.camera_hardware_source
        if camera_hardware_source:
            # create a minimal camera frame parameters by creating a default one, then copying the exposure and
            # binning. the exposure may come from the exposure model that is presented to the user in the UI;
            # otherwise it is copied from the current parameters. the binning is always copied from the current
            # parameters since it is not presented in the acquisition panel UI currently.
            camera_frame_parameters = camera_hardware_source.validate_frame_parameters(camera_base.CameraFrameParameters(dict()))
            camera_frame_parameters.exposure = self.__camera_config_exposure_model.value or camera_hardware_source.get_current_frame_parameters().exposure
            camera_frame_parameters.binning = camera_hardware_source.get_current_frame_parameters().binning
            return camera_frame_parameters
        return None

    def _hardware_source_changed(self) -> None:
        self.__update_channel_descriptions()

    def _update_settings(self, hardware_source: HardwareSource.HardwareSource) -> None:
        camera_hardware_source = self.camera_hardware_source
        if camera_hardware_source:
            frame_parameters = camera_hardware_source.get_current_frame_parameters()
            has_exposure = self.__camera_config_exposure_model.value is not None
            self.__exposure_time = (self.__camera_config_exposure_model.value if has_exposure else frame_parameters.exposure) or 0.0
            self.__exposure_precision = camera_hardware_source.exposure_precision
            self.__exposure_str = make_exposure_str(self.__exposure_time, self.__exposure_precision) if has_exposure else None
            self.__exposure_placeholder_str = make_exposure_str(frame_parameters.exposure, self.__exposure_precision)
            self.__exposure_units_str = exposure_units[self.__exposure_precision]
            self.notify_property_changed("exposure_time")
            self.notify_property_changed("exposure_precision")
            self.notify_property_changed("exposure_str")
            self.notify_property_changed("exposure_placeholder_str")
            self.notify_property_changed("exposure_units_str")

    @property
    def channel_descriptions(self) -> typing.List[Acquisition.HardwareSourceChannelDescription]:
        return self.__channel_descriptions

    @channel_descriptions.setter
    def channel_descriptions(self, value: typing.List[Acquisition.HardwareSourceChannelDescription]) -> None:
        # hack to work around lack of read-only binding
        pass

    @property
    def channel_index(self) -> typing.Optional[int]:
        # map from the channel model (channel identifier string) to a channel index.
        m = {o.channel_id: o for o in self.__channel_descriptions}
        return self.__channel_descriptions.index(m[self.__camera_hardware_source_channel_model.value]) if self.__camera_hardware_source_channel_model.value in m else None

    @channel_index.setter
    def channel_index(self, value: typing.Optional[int]) -> None:
        # map from the channel index to the channel model (channel identifier string).
        channel_id = self.__channel_descriptions[value].channel_id if (value is not None and 0 <= value < len(self.__channel_descriptions)) else (self.__channel_descriptions[0].channel_id if self.__channel_descriptions else "image")
        if channel_id != self.__camera_hardware_source_channel_model.value:
            self.__camera_hardware_source_channel_model.value = channel_id
            self.notify_property_changed("channel_index")

    def __update_channel_descriptions(self) -> None:
        # when the list of hardware sources changes or the selected hardware source changes, the list of available
        # channels needs to be updated. the selected channel may also be updated if it is no longer available.
        camera_hardware_source = self.camera_hardware_source
        if camera_hardware_source:
            if getattr(camera_hardware_source.camera, "camera_type") == "ronchigram":
                channel_descriptions = [Acquisition.hardware_source_channel_descriptions["ronchigram"]]
            elif getattr(camera_hardware_source.camera, "camera_type") == "eels":
                channel_descriptions = [Acquisition.hardware_source_channel_descriptions["eels_spectrum"], Acquisition.hardware_source_channel_descriptions["eels_image"]]
            else:
                channel_descriptions = [Acquisition.hardware_source_channel_descriptions["image"]]
        else:
            channel_descriptions = [Acquisition.hardware_source_channel_descriptions["image"]]
        output = Acquisition.hardware_source_channel_descriptions.get(self.__camera_hardware_source_channel_model.value or str())
        if not output or output not in channel_descriptions:
            output = channel_descriptions[0]
        channel_descriptions_changed = channel_descriptions != self.__channel_descriptions
        self.__channel_descriptions = channel_descriptions
        self.__camera_hardware_source_channel_model.value = output.channel_id
        if channel_descriptions_changed:
            self.notify_property_changed("channel_descriptions")

    def get_byte_dimensions(self, camera_size: tuple[int, int]) -> tuple[int, ...]:
        camera_hardware_source = self.camera_hardware_source
        item_size = numpy.dtype(numpy.float32).itemsize
        if camera_hardware_source:
            if getattr(camera_hardware_source.camera, "camera_type") == "ronchigram":
                return (camera_size[0], camera_size[1], item_size)
            elif getattr(camera_hardware_source.camera, "camera_type") == "eels":
                return (camera_size[1], item_size) if self.channel_index == 0 else (camera_size[0], camera_size[1], item_size)
            else:
                return (camera_size[0], camera_size[1], item_size)
        else:
            return (camera_size[0], camera_size[1], item_size)

    @property
    def exposure_time(self) -> typing.Optional[float]:
        return self.__exposure_time

    @exposure_time.setter
    def exposure_time(self, exposure_time: typing.Optional[float]) -> None:
        self.__camera_config_exposure_model.value = exposure_time

    @property
    def exposure_precision(self) -> int:
        return self.__exposure_precision

    @property
    def exposure_str(self) -> typing.Optional[str]:
        return self.__exposure_str

    @exposure_str.setter
    def exposure_str(self, exposure_str: typing.Optional[str]) -> None:
        converter = Converter.FloatToStringConverter()
        scaled_exposure = converter.convert_back(exposure_str) if exposure_str else None
        if scaled_exposure:
            self.exposure_time = scaled_exposure * math.pow(10, math.trunc(self.__exposure_precision / 3) * 3)
        else:
            self.exposure_time = None

    @property
    def exposure_placeholder_str(self) -> str:
        return self.__exposure_placeholder_str

    @property
    def exposure_units_str(self) -> str:
        return self.__exposure_units_str

    @property
    def camera_hardware_source(self) -> typing.Optional[camera_base.CameraHardwareSource]:
        return typing.cast(typing.Optional[camera_base.CameraHardwareSource], self.hardware_source)

    @property
    def camera_channel(self) -> typing.Optional[str]:
        return self.__camera_hardware_source_channel_model.value


class CameraDetailsHandler(Declarative.Handler):
    """A declarative component handler for a row describing a camera device.

    The hardware_source_choice parameter is the associated hardware source choice.
    """

    def __init__(self, camera_settings_model: CameraSettingsModel) -> None:
        super().__init__()

        # the exposure model converts the exposure value stream to a property model that supports binding.
        self.camera_settings_model = camera_settings_model

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_row(
            u.create_stack(
                u.create_row(
                    u.create_row(
                        u.create_label(text=_("Camera Exposure Time")),
                        u.create_label(text=_(" (")),
                        u.create_label(text="@binding(camera_settings_model.exposure_units_str)"),
                        u.create_label(text=_(")")),
                    ),
                    u.create_line_edit(text="@binding(camera_settings_model.exposure_str)", placeholder_text="@binding(camera_settings_model.exposure_placeholder_str)", width=80),
                    u.create_stretch(),
                    spacing=8
                )
            ),
            u.create_stretch()
        )

    def close(self) -> None:
        self.camera_settings_model = typing.cast(typing.Any, None)
        super().close()


class SynchronizedScanAcquisitionDeviceComponentHandler(AcquisitionDeviceComponentHandler):
    """A declarative component handler for a synchronized scan/camera virtual device.

    Produces data streams from the camera and scan.

    The configuration should contain a camera_device_id, camera_channel_id, scan_device_id, and scan_width. The
    camera_device_id, camera_channel_id, and scan_device_id are used to associate a camera hardware source,
    a camera channel, and a scan hardware source with this virtual device. The scan_width will override the
    width used in the context. The subscan may be used depending on the settings in the scan control panel.
    """

    component_id = "synchronized-scan"
    display_name = _("Synchronized Scan")

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(SynchronizedScanAcquisitionDeviceComponentHandler.display_name)

        # the camera settings represent the settings for the camera. in addition to handling the model updates,
        # it can provide the current camera_hardware_source and camera_frame_parameters.
        self._camera_settings_model = CameraSettingsModel(configuration)

        # the scan hardware source handler is a declarative component handler that facilitates a combo box.
        self.__scan_hardware_source_choice_model = HardwareSourceChoiceModel(configuration, "scan_device_id",
                                                                             lambda hardware_source: hardware_source.features.get("is_scanning", False),
                                                                             force_enabled=True)

        # the scan width model is a property model made by observing the scan_width property in the configuration.
        self.scan_width = Model.PropertyChangedPropertyModel[int](configuration, "scan_width")

        # the scan context description value stream observes the scan and camera hardware sources to produce
        # a description of the upcoming scan. it also supplies a scan_context_valid flag which can be used to enable
        # the acquire button in the UI.
        self.__scan_context_description_value_stream = ScanSpecifierValueStream(
            self._camera_settings_model.hardware_source_choice_stream,
            HardwareSourceChoice.HardwareSourceChoiceStream(self.__scan_hardware_source_choice_model.hardware_source_choice),
            self.scan_width,
            asyncio.get_event_loop_policy().get_event_loop()).add_ref()

        # the scan context value model is the text description of the scan context extracted from the value stream.
        self.scan_context_value_model = Model.StreamValueModel(Stream.MapStream(
            self.__scan_context_description_value_stream,
            lambda x: x.context_description if x is not None else str()
        ))

        # the scan context value model is the text description of the upcoming scan extracted from the value stream.
        self.scan_value_model = Model.StreamValueModel(Stream.MapStream(
            self.__scan_context_description_value_stream,
            lambda x: x.scan_description if x is not None else str()
        ))

        # a converter for the scan width.
        self.scan_width_converter = Converter.IntegerToStringConverter()

        # the acquire_valid_value_stream is a value stream of bool values extracted from the scan context description.
        # it is used to enable the acquire button. but to do so, this stream must be read from the enclosing
        # declarative component handler. this stream is not used within this class. perhaps there is a better way to
        # do this.
        def is_scan_specifier_valid(scan_specifier: STEMController.ScanSpecifier | None) -> bool:
            return scan_specifier.scan_context is not None if scan_specifier is not None else False

        self.acquire_valid_value_stream = Stream.MapStream[STEMController.ScanSpecifier, bool](self.__scan_context_description_value_stream, is_scan_specifier_valid).add_ref()

        # for this component, the time space usage changed event is fired when the extended camera frame parameters (
        # includes readout area) or the extended scan frame parameters (includes enabled channels) change.
        self.time_space_usage_changed_event = Event.Event()

        camera_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self._camera_settings_model.hardware_source_choice_model.hardware_source_choice)

        scan_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self.__scan_hardware_source_choice_model.hardware_source_choice)

        self.__camera_hardware_source_stream = camera_hardware_source_stream

        self.__camera_frame_parameters_stream = CameraFrameParametersStream(camera_hardware_source_stream)

        def camera_frame_parameters_changed(camera_frame_parameters: CameraFrameParametersAndReadoutArea | None) -> None:
            self.time_space_usage_changed_event.fire()

        self.__camera_frame_parameters_stream_listener = self.__camera_frame_parameters_stream.value_stream.listen(camera_frame_parameters_changed)

        def camera_settings_property_changed(key: str) -> None:
            if key in ("exposure_time", "channel_index"):
                self.time_space_usage_changed_event.fire()

        self.__camera_settings_listener = self._camera_settings_model.property_changed_event.listen(camera_settings_property_changed)

        def scan_frame_parameters_changed(scan_frame_parameters: scan_base.ScanFrameParameters | None) -> None:
            self.time_space_usage_changed_event.fire()

        self.__scan_hardware_source_stream = scan_hardware_source_stream

        self.__scan_frame_parameters_stream = ScanFrameParametersStream(scan_hardware_source_stream)

        self.__scan_frame_parameters_stream_listener = self.__scan_frame_parameters_stream.value_stream.listen(scan_frame_parameters_changed)

        scan_frame_parameters_changed(self.__scan_frame_parameters_stream.value)

        def scan_width_property_changed(key: str) -> None:
            if key in ("value"):
                self.time_space_usage_changed_event.fire()

        self.__scan_width_listener = self.scan_width.property_changed_event.listen(scan_width_property_changed)

        u = Declarative.DeclarativeUI()

        column_items = list()
        column_items.append(
            u.create_row(
                u.create_component_instance(identifier="camera-hardware-source-choice-component"),
                u.create_combo_box(items_ref="@binding(_camera_settings_model.channel_descriptions)", current_index="@binding(_camera_settings_model.channel_index)"),
                u.create_stretch(),
                spacing=8
            )
        )
        column_items.append(
            u.create_row(
                u.create_component_instance(identifier="acquisition-device-component-details"),
                u.create_stretch(),
                spacing=8
            )
        )
        column_items.append(
            u.create_component_instance(identifier="scan-hardware-source-choice-component"),
        )
        column_items.append(
            u.create_row(
                # TODO: height not necessary if all-rows-same-height available on columns
                u.create_label(text="@binding(scan_context_value_model.value)", height=24),
                u.create_stretch(),
                spacing=8
            )
        )
        column_items.append(
            u.create_row(
                u.create_label(text=_("Scan Width (pixels)")),
                u.create_line_edit(text="@binding(scan_width.value, converter=scan_width_converter)", width=60),
                u.create_label(text="@binding(scan_value_model.value)"),
                u.create_stretch(),
                spacing=8
            )
        )
        self.ui_view = u.create_column(
            *column_items,
            spacing=8,
        )

    def close(self) -> None:
        self.acquire_valid_value_stream.remove_ref()
        self.acquire_valid_value_stream = typing.cast(typing.Any, None)
        self.scan_value_model.close()
        self.scan_value_model = typing.cast(typing.Any, None)
        self.__scan_context_description_value_stream.remove_ref()
        self.__scan_context_description_value_stream = typing.cast(typing.Any, None)
        self.scan_context_value_model.close()
        self.scan_context_value_model = typing.cast(typing.Any, None)
        self.scan_width.close()
        self.scan_width = typing.cast(typing.Any, None)
        self.__scan_hardware_source_choice_model.close()
        self.__scan_hardware_source_choice_model = typing.cast(typing.Any, None)
        self._camera_settings_model = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "camera-hardware-source-choice-component":
            return HardwareSourceChoiceHandler(self._camera_settings_model.hardware_source_choice_model)
        elif component_id == "scan-hardware-source-choice-component":
            return HardwareSourceChoiceHandler(self.__scan_hardware_source_choice_model, title=_("Scan Detector"))
        elif component_id == "acquisition-device-component-details":
            return CameraDetailsHandler(self._camera_settings_model)
        return None

    def get_time_space_usage(self, camera_exposure_time: float | None = None) -> TimeSpaceUsage:
        camera_hardware_source_stream = self.__camera_hardware_source_stream
        camera_hardware_source = camera_hardware_source_stream.value
        camera_frame_parameters = self._camera_settings_model.camera_frame_parameters
        total_frame_bytes: int | None = None
        camera_frame_time: float | None = None
        camera_frame_bytes: int | None = None
        if isinstance(camera_hardware_source, camera_base.CameraHardwareSource) and camera_frame_parameters:
            camera_size = camera_hardware_source.get_expected_dimensions(camera_frame_parameters.binning)
            camera_dimensions = self._camera_settings_model.get_byte_dimensions(camera_size)
            camera_frame_bytes = int(numpy.prod(camera_dimensions, dtype=numpy.int64))
            camera_frame_time = camera_exposure_time if camera_exposure_time is not None else camera_frame_parameters.exposure
        scan_hardware_source_stream = self.__scan_hardware_source_stream
        scan_frame_parameters = copy.copy(self.__scan_frame_parameters_stream.value)  # use a copy since they will be modified during calculations.
        scan_hardware_source = scan_hardware_source_stream.value
        scan_frame_time: float | None = None
        if isinstance(scan_hardware_source,
                      scan_base.ScanHardwareSource) and scan_frame_parameters and camera_frame_parameters:
            scan_context_description = self.__scan_context_description_value_stream.value
            assert scan_context_description
            scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters, typing.cast(typing.Tuple[int, int],
                                                                                               scan_context_description.scan_size))
            scan_size = scan_frame_parameters.scan_size
            scan_frame_parameters_with_camera_exposure = copy.copy(scan_frame_parameters)
            scan_frame_parameters_with_camera_exposure.pixel_time_us = camera_frame_parameters.exposure * 1E6 if camera_frame_parameters.exposure is not None else scan_frame_parameters.pixel_time_us
            flyback_pixels = scan_hardware_source.calculate_flyback_pixels(scan_frame_parameters_with_camera_exposure)
            assert camera_frame_time is not None
            assert camera_frame_bytes is not None
            scan_frame_time = scan_size.height * (scan_size.width + flyback_pixels) * camera_frame_time
            channel_bytes = len(scan_frame_parameters.enabled_channel_indexes or list()) * numpy.dtype(
                numpy.float32).itemsize
            scan_frame_bytes = scan_size.height * scan_size.width * channel_bytes
            total_frame_bytes = camera_frame_bytes * scan_size.height * scan_size.width + scan_frame_bytes
        return TimeSpaceUsage(scan_frame_time, total_frame_bytes)

    def build_acquisition_device(self) -> Acquisition.AcquisitionDeviceLike:
        # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.
        camera_hardware_source = self._camera_settings_model.camera_hardware_source
        camera_frame_parameters = self._camera_settings_model.camera_frame_parameters
        camera_channel = self._camera_settings_model.camera_channel
        assert camera_hardware_source
        assert camera_frame_parameters
        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source_choice_model.hardware_source)
        scan_context_description = self.__scan_context_description_value_stream.value
        assert scan_context_description
        scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
        scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters, typing.cast(typing.Tuple[int, int], scan_context_description.scan_size))
        return scan_base.SynchronizedScanAcquisitionDevice(scan_hardware_source,
                                                           scan_frame_parameters,
                                                           camera_hardware_source,
                                                           camera_frame_parameters,
                                                           camera_channel,
                                                           scan_context_description.drift_correction_enabled,
                                                           scan_context_description.drift_interval_lines,
                                                           scan_context_description.drift_interval_scans,
                                                           scan_hardware_source.drift_channel_id,
                                                           scan_hardware_source.drift_region,
                                                           scan_hardware_source.drift_rotation
                                                           )


@dataclasses.dataclass
class CameraFrameParametersAndReadoutArea:
    """A class to define the frame parameters and readout area of a camera device."""
    frame_parameters: camera_base.CameraFrameParameters
    readout_area_TLBR: tuple[int, int, int, int]


class CameraFrameParametersStream(Stream.ValueStream[CameraFrameParametersAndReadoutArea]):
    """Define a stream of combination of camera frame parameters and readout area.

    The CameraHardwareSource camera_frame_parameters_changed_event will get fired if the device provides readout area
    changed event and fires it. This is a hack since eventually the readout area should be part of the frame parameters.
    """

    def __init__(self, hardware_source_stream: HardwareSourceChoice.HardwareSourceChoiceStream) -> None:
        super().__init__()
        self.__frame_parameters_changed_listener: typing.Optional[Event.EventListener] = None
        self.__camera_hardware_source_stream_listener = hardware_source_stream.value_stream.listen(weak_partial(CameraFrameParametersStream.__hardware_source_stream_changed, self))
        self.__hardware_source_stream_changed(hardware_source_stream.value)

    def __hardware_source_stream_changed(self, hardware_source: typing.Optional[HardwareSource.HardwareSource]) -> None:
        # when the hardware source choice changes, update the frame parameters listener. close the old one.
        self.__frame_parameters_changed_listener = None
        assert isinstance(hardware_source, camera_base.CameraHardwareSource)
        self.__frame_parameters_changed_listener = hardware_source.current_frame_parameters_changed_event.listen(weak_partial(CameraFrameParametersStream.__camera_frame_parameters_changed, self, hardware_source))
        self.__camera_frame_parameters_changed(hardware_source, hardware_source.get_current_frame_parameters())
        self.__camera_frame_parameters_changed_listener = hardware_source.camera_frame_parameters_changed_event.listen(weak_partial(CameraFrameParametersStream.__camera_frame_parameters_changed, self, hardware_source))

    def __camera_frame_parameters_changed(self, camera_hardware_source: camera_base.CameraHardwareSource, frame_parameters: camera_base.CameraFrameParameters) -> None:
        self.value = CameraFrameParametersAndReadoutArea(copy.copy(frame_parameters), camera_hardware_source.camera.readout_area)


class CameraAcquisitionDeviceComponentHandler(AcquisitionDeviceComponentHandler):
    """A declarative component handler for a camera device.

    Produces a data stream from the camera in the form of individual frames.

    The configuration should contain a camera_device_id and camera_channel_id. The camera_device_id and
    camera_channel_id are used to associate a camera hardware source and a camera channel.
    """

    component_id = "camera"
    display_name = _("Camera")

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(CameraAcquisitionDeviceComponentHandler.display_name)

        # the camera settings represent the settings for the camera. in addition to handling the model updates,
        # it can provide the current camera_hardware_source and camera_frame_parameters.
        self._camera_settings_model = CameraSettingsModel(configuration)

        # a camera is always valid.
        self.acquire_valid_value_stream = Stream.ConstantStream(True)

        # for this component, the time space usage changed event is fired when the extended camera frame parameters (
        # includes readout area) change.
        self.time_space_usage_changed_event = Event.Event()

        camera_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self._camera_settings_model.hardware_source_choice_model.hardware_source_choice)

        self.__camera_hardware_source_stream = camera_hardware_source_stream

        self.__camera_frame_parameters_stream = CameraFrameParametersStream(camera_hardware_source_stream)

        def camera_frame_parameters_changed(camera_frame_parameters: CameraFrameParametersAndReadoutArea | None) -> None:
            self.time_space_usage_changed_event.fire()

        self.__camera_frame_parameters_stream_listener = self.__camera_frame_parameters_stream.value_stream.listen(camera_frame_parameters_changed)

        def camera_settings_property_changed(key: str) -> None:
            if key in ("exposure_time", "channel_index"):
                self.time_space_usage_changed_event.fire()

        self.__camera_settings_listener = self._camera_settings_model.property_changed_event.listen(camera_settings_property_changed)

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(
                u.create_component_instance(identifier="acquisition-device-component"),
                u.create_combo_box(items_ref="@binding(_camera_settings_model.channel_descriptions)", current_index="@binding(_camera_settings_model.channel_index)"),
                u.create_stretch(),
                spacing=8
            ),
            u.create_row(
                u.create_component_instance(identifier="acquisition-device-component-details"),
                u.create_stretch(),
                spacing=8
            ),
            spacing=8,
        )

    def close(self) -> None:
        self._camera_settings_model = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "acquisition-device-component":
            return HardwareSourceChoiceHandler(self._camera_settings_model.hardware_source_choice_model)
        elif component_id == "acquisition-device-component-details":
            return CameraDetailsHandler(self._camera_settings_model)
        return None

    def get_time_space_usage(self, camera_exposure_time: float | None = None) -> TimeSpaceUsage:
        camera_hardware_source_stream = self.__camera_hardware_source_stream
        camera_hardware_source = camera_hardware_source_stream.value
        camera_frame_parameters = self._camera_settings_model.camera_frame_parameters
        camera_frame_time: float | None = None
        camera_frame_bytes: int | None = None
        if isinstance(camera_hardware_source, camera_base.CameraHardwareSource) and camera_frame_parameters:
            camera_size = camera_hardware_source.get_expected_dimensions(camera_frame_parameters.binning)
            camera_dimensions = self._camera_settings_model.get_byte_dimensions(camera_size)
            camera_frame_bytes = int(numpy.prod(camera_dimensions, dtype=numpy.int64))
            camera_frame_time = camera_exposure_time if camera_exposure_time is not None else camera_frame_parameters.exposure
        return TimeSpaceUsage(camera_frame_time, camera_frame_bytes)

    def build_acquisition_device(self) -> Acquisition.AcquisitionDeviceLike:
        # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.

        # grab the camera hardware source and frame parameters
        camera_hardware_source = self._camera_settings_model.camera_hardware_source
        camera_frame_parameters = self._camera_settings_model.camera_frame_parameters
        camera_channel = self._camera_settings_model.camera_channel
        assert camera_hardware_source
        assert camera_frame_parameters

        return camera_base.CameraAcquisitionDevice(camera_hardware_source, camera_frame_parameters, camera_channel)


class ScanFrameParametersStream(Stream.ValueStream[scan_base.ScanFrameParameters]):
    """Define a stream of scan frame parameters (extended with enabled channels)."""

    def __init__(self, scan_hardware_source_stream: HardwareSourceChoice.HardwareSourceChoiceStream) -> None:
        super().__init__()
        self.__frame_parameters_changed_listener: typing.Optional[Event.EventListener] = None
        self.__scan_hardware_source_stream_listener = scan_hardware_source_stream.value_stream.listen(weak_partial(ScanFrameParametersStream.__hardware_source_stream_changed, self))
        self.__hardware_source_stream_changed(scan_hardware_source_stream.value)

    def __hardware_source_stream_changed(self, hardware_source: typing.Optional[HardwareSource.HardwareSource]) -> None:
        # when the hardware source choice changes, update the frame parameters listener. close the old one.
        self.__frame_parameters_changed_listener = None
        assert isinstance(hardware_source, scan_base.ScanHardwareSource)
        self.__frame_parameters_changed_listener = hardware_source.scan_frame_parameters_changed_event.listen(weak_partial(ScanFrameParametersStream.__frame_parameters_changed, self))
        scan_frame_parameters = hardware_source.get_current_frame_parameters()
        scan_frame_parameters.enabled_channel_indexes = hardware_source.get_enabled_channel_indexes()
        self.__frame_parameters_changed(scan_frame_parameters)

    def __frame_parameters_changed(self, frame_parameters: scan_base.ScanFrameParameters) -> None:
        self.value = copy.copy(frame_parameters)


class ScanAcquisitionDeviceComponentHandler(AcquisitionDeviceComponentHandler):
    """A declarative component handler for a scan device.

    Produces a data stream from scan device.

    The configuration should contain a scan_device_id, used to associate a scan hardware source. The subscan may be
    used depending on the settings in the scan control panel.
    """

    component_id = "scan"
    display_name = _("Scan")

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(ScanAcquisitionDeviceComponentHandler.display_name)

        # the scan hardware source handler is a declarative component handler that facilitates a combo box.
        self.__scan_hardware_source_choice_model = HardwareSourceChoiceModel(configuration, "scan_device_id",
                                                                             lambda hardware_source: hardware_source.features.get("is_scanning", False),
                                                                             force_enabled=True)

        # the scan is always valid.
        self.acquire_valid_value_stream = Stream.ConstantStream(True)

        # for this component, the time space usage changed event is fired when the extended scan frame parameters (
        # includes enabled channels) change.
        self.time_space_usage_changed_event = Event.Event()

        scan_hardware_source_stream = HardwareSourceChoice.HardwareSourceChoiceStream(self.__scan_hardware_source_choice_model.hardware_source_choice)

        def scan_frame_parameters_changed(scan_frame_parameters: scan_base.ScanFrameParameters | None) -> None:
            self.time_space_usage_changed_event.fire()

        self.__scan_hardware_source_stream = scan_hardware_source_stream

        self.__scan_frame_parameters_stream = ScanFrameParametersStream(scan_hardware_source_stream)

        self.__scan_frame_parameters_stream_listener = self.__scan_frame_parameters_stream.value_stream.listen(scan_frame_parameters_changed)

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_component_instance(identifier="scan-hardware-source-choice-component"),
            u.create_stretch(),
            spacing=8
        )

    def close(self) -> None:
        self.__scan_frame_parameters_stream_listener = typing.cast(typing.Any, None)
        self.__scan_frame_parameters_stream = typing.cast(typing.Any, None)
        self.__scan_hardware_source_choice_model.close()
        self.__scan_hardware_source_choice_model = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "scan-hardware-source-choice-component":
            return HardwareSourceChoiceHandler(self.__scan_hardware_source_choice_model, title=_("Scan Detector"))
        return None

    def get_time_space_usage(self, camera_exposure_time: float | None = None) -> TimeSpaceUsage:
        scan_hardware_source_stream = self.__scan_hardware_source_stream
        scan_frame_parameters = self.__scan_frame_parameters_stream.value
        scan_hardware_source = scan_hardware_source_stream.value
        scan_frame_time: float | None = None
        scan_frame_bytes: int | None = None
        if isinstance(scan_hardware_source, scan_base.ScanHardwareSource) and scan_frame_parameters:
            size = scan_frame_parameters.scan_size
            flyback_pixels = scan_hardware_source.calculate_flyback_pixels(scan_frame_parameters)
            pixel_time_us = scan_frame_parameters.pixel_time_us
            scan_frame_time = size.height * (size.width + flyback_pixels) * pixel_time_us / 1000000.0
            channel_bytes = len(scan_frame_parameters.enabled_channel_indexes or list()) * numpy.dtype(numpy.float32).itemsize
            scan_frame_bytes = size.height * size.width * channel_bytes
        return TimeSpaceUsage(scan_frame_time, scan_frame_bytes)

    def build_acquisition_device(self) -> Acquisition.AcquisitionDeviceLike:
        # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.

        # first get the scan hardware source.
        scan_hardware_source = typing.cast(typing.Optional[scan_base.ScanHardwareSource], self.__scan_hardware_source_choice_model.hardware_source)
        assert scan_hardware_source is not None

        scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
        scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters)

        return scan_base.ScanAcquisitionDevice(scan_hardware_source, scan_frame_parameters)


# register each component as an acquisition device component factory.
Registry.register_component(SynchronizedScanAcquisitionDeviceComponentHandler, {"acquisition-device-component-factory"})
Registry.register_component(CameraAcquisitionDeviceComponentHandler, {"acquisition-device-component-factory"})
Registry.register_component(ScanAcquisitionDeviceComponentHandler, {"acquisition-device-component-factory"})

# define the entities used for persistent storage.

AcquisitionDeviceComponentSchema = Schema.entity("acquisition_device_component", None, None, {
})

# SynchronizedScanAcquisitionDeviceComponentHandler
Schema.entity("acquisition_device_component_synchronized_scan", AcquisitionDeviceComponentSchema, None, {
    "camera_device_id": Schema.prop(Schema.STRING),
    "camera_channel_id": Schema.prop(Schema.STRING),
    "scan_device_id": Schema.prop(Schema.STRING),
    "scan_width": Schema.prop(Schema.INT, default=32, optional=True),
    "camera_exposure": Schema.prop(Schema.FLOAT, optional=True),
})

# ScanAcquisitionDeviceComponentHandler
Schema.entity("acquisition_device_component_scan", AcquisitionDeviceComponentSchema, None, {
    "scan_device_id": Schema.prop(Schema.STRING),
})

# CameraAcquisitionDeviceComponentHandler
Schema.entity("acquisition_device_component_camera", AcquisitionDeviceComponentSchema, None, {
    "camera_device_id": Schema.prop(Schema.STRING),
    "camera_channel_id": Schema.prop(Schema.STRING),
    "camera_exposure": Schema.prop(Schema.FLOAT, optional=True),
})

AcquisitionMethodSchema = Schema.entity("acquisition_method_component", None, None, {
})

# BasicAcquisitionMethodComponentHandler
Schema.entity("acquisition_method_component_basic_acquire", AcquisitionMethodSchema, None, {
})

# SequenceAcquisitionMethodComponentHandler
Schema.entity("acquisition_method_component_sequence_acquire", AcquisitionMethodSchema, None, {
    "count": Schema.prop(Schema.INT, default=1),
})

ControlValuesSchema = Schema.entity("control_values", None, None, {
    "control_id": Schema.prop(Schema.STRING),
    "count": Schema.prop(Schema.INT),
    "start_value": Schema.prop(Schema.FLOAT),
    "step_value": Schema.prop(Schema.FLOAT),
})

# SeriesAcquisitionMethodComponentHandler
Schema.entity("acquisition_method_component_series_acquire", AcquisitionMethodSchema, None, {
    "control_id": Schema.prop(Schema.STRING),
    "control_values_list": Schema.array(Schema.component(ControlValuesSchema))
})

# TableauAcquisitionMethodComponentHandler
Schema.entity("acquisition_method_component_tableau_acquire", AcquisitionMethodSchema, None, {
    "control_id": Schema.prop(Schema.STRING),
    "axis_id": Schema.prop(Schema.STRING),
    "x_control_values_list": Schema.array(Schema.component(ControlValuesSchema)),
    "y_control_values_list": Schema.array(Schema.component(ControlValuesSchema)),
})

# MultipleAcquisitionMethodComponentHandler
MultipleAcquireEntrySchema = Schema.entity("multi_acquire_entry", None, None, {
    "offset": Schema.prop(Schema.FLOAT),
    "exposure": Schema.prop(Schema.FLOAT),
    "count": Schema.prop(Schema.INT),
    "include_sum": Schema.prop(Schema.BOOLEAN),
})

Schema.entity("acquisition_method_component_multiple_acquire", AcquisitionMethodSchema, None, {
    "sections": Schema.array(Schema.component(MultipleAcquireEntrySchema)),
})

AcquisitionConfigurationSchema = Schema.entity("acquisition_configuration", None, None, {
    "acquisition_device_component_id": Schema.prop(Schema.STRING),
    "acquisition_method_component_id": Schema.prop(Schema.STRING),
    "acquisition_device_components": Schema.array(Schema.component(AcquisitionDeviceComponentSchema)),
    "acquisition_method_components": Schema.array(Schema.component(AcquisitionMethodSchema)),
})


class AcquisitionConfiguration(Schema.Entity):
    """A configuration object that is synchronized to a file path.

    The logger/recorder observes changes to the entity. The changes get written to the file.
    """

    def __init__(self, app_data: AcquisitionPreferences.DictRecorderLoggerDictInterface) -> None:
        super().__init__(AcquisitionConfigurationSchema)
        self.read_from_dict(app_data.get_data_dict())
        field = Schema.ComponentField(None, self.entity_type.entity_id)
        field.set_field_value(None, self)
        self.__logger = AcquisitionPreferences.DictRecorderLogger(field, app_data)
        self.__recorder = Recorder.Recorder(self, None, self.__logger)

    def close(self) -> None:
        self.__recorder.close()
        self.__recorder = typing.cast(typing.Any, None)
        super().close()

    def _create(self, context: typing.Optional[Schema.EntityContext]) -> Schema.Entity:
        raise NotImplementedError()


# define the global acquisition configuration object. this object is created/destroyed when the
# global application object changes. this is observed using the registry.
acquisition_configuration: typing.Optional[AcquisitionConfiguration] = None


# when the registry gets a "application" object, call this function. this function configures
# the file path and creates/destroys the acquisition_configuration global variable.
def handle_application_changed(is_register: bool, component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
    if "application" in component_types:
        application: typing.Optional[Application.BaseApplication] = component if is_register else None
        global acquisition_configuration
        if application:
            file_path = application.ui.get_configuration_location() / pathlib.Path("nion_acquisition_preferences.json")
            logging.info("Acquisition preferences: " + str(file_path))
            AcquisitionPreferences.init_acquisition_preferences(file_path)
            file_path = application.ui.get_configuration_location() / pathlib.Path("nion_acquisition_configuration.json")
            logging.info("Acquisition configuration: " + str(file_path))
            acquisition_configuration = AcquisitionConfiguration(ApplicationData.ApplicationData(file_path))
        else:
            AcquisitionPreferences.deinit_acquisition_preferences()
            acquisition_configuration = None


component_registered_event_listener = Registry.listen_component_registered_event(functools.partial(handle_application_changed, True))
component_unregistered_event_listener = Registry.listen_component_unregistered_event(functools.partial(handle_application_changed, False))
Registry.fire_existing_component_registered_events("application")


class PreferencesButtonHandler(Declarative.Handler):

    def __init__(self, document_controller: DocumentController.DocumentController):
        super().__init__()
        self.document_controller = document_controller
        sliders_icon_24_png = pkgutil.get_data(__name__, "resources/sliders_icon_24.png")
        assert sliders_icon_24_png is not None
        self._sliders_icon_24_png = CanvasItem.load_rgba_data_from_bytes(sliders_icon_24_png, "png")
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_image(image="@binding(_sliders_icon_24_png)", width=24, height=24, on_clicked="handle_preferences")

    def handle_preferences(self, widget: UserInterfaceModule.Widget) -> None:
        self.document_controller.open_preferences()


T = typing.TypeVar('T')


class StreamStreamer(Stream.ValueStream[T], typing.Generic[T]):
    """A utility stream for stream a set of streams. There must be a better way!"""

    def __init__(self, streams_stream: Stream.AbstractStream[Stream.AbstractStream[T]]) -> None:
        super().__init__()
        self.__streams_stream = streams_stream.add_ref()
        self.__sub_stream_listener: typing.Optional[Event.EventListener] = None
        self.__sub_stream: typing.Optional[Stream.AbstractStream[T]] = None
        self.__listener = self.__streams_stream.value_stream.listen(weak_partial(StreamStreamer.__attach_stream, self))
        self.__attach_stream(self.__streams_stream.value)

    def close(self) -> None:
        self.__streams_stream.remove_ref()
        self.__listener.close()
        self.__listener = typing.cast(typing.Any, None)
        if self.__sub_stream_listener:
            self.__sub_stream_listener.close()
            self.__sub_stream_listener = None
        if self.__sub_stream:
            self.__sub_stream.remove_ref()
            self.__sub_stream = None

    def __attach_stream(self, value_stream: typing.Optional[Stream.AbstractStream[T]]) -> None:
        # watching the stream of streams, this gets called with a new stream when it changes.
        if self.__sub_stream_listener:
            self.__sub_stream_listener.close()
            self.__sub_stream_listener = None
        if self.__sub_stream:
            self.__sub_stream.remove_ref()
            self.__sub_stream = None
        # create the stream for the value
        self.__sub_stream = value_stream.add_ref() if value_stream else None
        # watch the new stream, sending it's value to this value stream when it changes
        if self.__sub_stream:
            self.__sub_stream_listener = self.__sub_stream.value_stream.listen(weak_partial(StreamStreamer.send_value, self))
            self.send_value(self.__sub_stream.value)
        else:
            # initialize the first value
            self.send_value(None)


class TimeSpaceUsageStream(Stream.ValueStream[TimeSpaceUsage]):
    """A time/space usage stream based on the method and device components.

    When the selected method or device component changes, this object informs the method component of the new device
    so it can get the unit time/space usage from the device and use it in its own time/space usage calculation. It
    also listens for the method to fire time space usage changed events, updating the value of this stream when that
    happens.
    """

    def __init__(self, method_component_stream: Stream.AbstractStream[AcquisitionMethodComponentHandler], device_component_stream: Stream.AbstractStream[AcquisitionDeviceComponentHandler]) -> None:
        super().__init__()
        self.__method_component_stream = method_component_stream
        self.__device_component_stream = device_component_stream
        self.__method_component_stream_listener = method_component_stream.value_stream.listen(weak_partial(TimeSpaceUsageStream.__method_component_changed, self))
        self.__device_component_stream_listener = device_component_stream.value_stream.listen(weak_partial(TimeSpaceUsageStream.__device_component_changed, self))
        self.__time_space_usage_changed_event_listener: Event.EventListener | None = None
        self.__device_component_changed(device_component_stream.value)
        self.__method_component_changed(method_component_stream.value)

    def __device_component_changed(self, device_component: AcquisitionDeviceComponentHandler | None) -> None:
        method_component = self.__method_component_stream.value
        if method_component:
            method_component.set_device_component(device_component)

    def __method_component_changed(self, method_component: AcquisitionMethodComponentHandler | None) -> None:
        method_component = self.__method_component_stream.value
        if method_component:
            method_component.set_device_component(self.__device_component_stream.value)
            self.__time_space_usage_changed_event_listener = method_component.time_space_usage_changed_event.listen(weak_partial(TimeSpaceUsageStream.__time_space_usage_changed, self, method_component))
            self.value = method_component.get_time_space_usage()
        else:
            self.__time_space_usage_changed_event_listener = None
            self.value = TimeSpaceUsage(None, None)

    def __time_space_usage_changed(self, method_component: AcquisitionMethodComponentHandler) -> None:
        self.value = method_component.get_time_space_usage()


class BytesToStringConverter(Converter.ConverterLike[float, str]):
    """A converter that converts bytes to a human-readable string."""

    def convert(self, value: float | None) -> str | None:
        if value is not None:
            bytes = float(value)
            kbytes = float(1024)
            mbytes = float(kbytes ** 2)  # 1,048,576
            gbytes = float(kbytes ** 3)  # 1,073,741,824
            tbytes = float(kbytes ** 4)  # 1,099,511,627,776

            if bytes < kbytes:
                return '{0} {1}'.format(bytes, 'Bytes' if bytes != 1 else 'Byte')
            elif kbytes <= bytes < mbytes:
                return '{0:.2f} KB'.format(bytes / kbytes)
            elif mbytes <= bytes < gbytes:
                return '{0:.2f} MB'.format(bytes / mbytes)
            elif gbytes <= bytes < tbytes:
                return '{0:.2f} GB'.format(bytes / gbytes)
            elif tbytes <= bytes:
                return '{0:.2f} TB'.format(bytes / tbytes)
        return None

    def convert_back(self, formatted_value: str | None) -> float | None:
        return None


class TimeToStringConverter(Converter.ConverterLike[float, str]):
    """A converter that converts time to a human-readable string."""

    def convert(self, value: float | None) -> str | None:
        if value is not None:
            if value > 3600:
                return "{0:.1f} hours".format(int(value) / 3600)
            elif value > 90:
                return "{0:.1f} minutes".format(int(value) / 60)
            elif value > 1:
                return "{0:.1f} seconds".format(value)
            elif value > 0.001:
                return "{0:.1f} milliseconds".format(value * 1000)
            elif value > 0.000001:
                return "{0:.1f} microseconds".format(value * 1000000)
        return None

    def convert_back(self, formatted_value: str | None) -> float | None:
        return None


@dataclasses.dataclass
class HandlerEntry:
    handler: Declarative.HandlerLike
    used: bool = False


class AcquisitionController(Declarative.Handler):
    """The acquisition controller is the top level declarative component handler for the acquisition panel UI.

    The acquisition controller allows the user to select an acquisition method (such as basic, sequence, serial, etc.)
    and an acquisition device (synchronized scan/camera, scan, camera, etc.).

    It also implements the basic acquisition start button, progress bar, drift logger (optional), and data display.
    """

    def __init__(self, document_controller: DocumentController.DocumentController, acquisition_configuration: AcquisitionConfiguration, acquisition_preferences: Observable.Observable) -> None:
        super().__init__()

        self.document_controller = document_controller

        # create two component combo box declarative components for handling the method and device.
        # pass the configuration and desired accessor strings for each.
        # these get closed by the declarative machinery
        self.__acquisition_method_component = ComponentComboBoxHandler("acquisition-method-component",
                                                                       _("Repeat Mode"),
                                                                       acquisition_configuration,
                                                                       acquisition_preferences,
                                                                       "acquisition_method_component_id",
                                                                       "acquisition_method_components",
                                                                       PreferencesButtonHandler(document_controller))

        def find_component_configuration(component_type: str) -> Schema.Entity:
            components_key = "acquisition_device_components"
            component_id = "acquisition_device_component_" + component_type

            # make a map from each component id to the associated component.
            component_map: typing.Dict[str, Schema.Entity] = dict()

            for component_entity_ in typing.cast(typing.Sequence[Schema.Entity], acquisition_configuration._get_array_items(components_key)):
                component_map[component_entity_.entity_type.entity_id] = component_entity_

            component_entity = typing.cast(typing.Optional[Schema.Entity], component_map.get(component_id))
            if not component_entity:
                entity_type = Schema.get_entity_type(component_id)
                component_entity = entity_type.create() if entity_type else None
                if component_entity:
                    acquisition_configuration._append_item(components_key, component_entity)

            assert component_entity
            return component_entity

        # create components for the scan, synchronized scan, and camera controls
        device_component_handlers = [
            ScanAcquisitionDeviceComponentHandler(find_component_configuration("scan"), acquisition_preferences),
            SynchronizedScanAcquisitionDeviceComponentHandler(find_component_configuration("synchronized_scan"), acquisition_preferences),
            CameraAcquisitionDeviceComponentHandler(find_component_configuration("camera"), acquisition_preferences)
        ]

        # create a list of handlers to be returned from create_handler. this is an experimental system for components.
        self.__handlers: typing.Dict[str, HandlerEntry] = dict()
        self.__handlers["acquisition-method-component"] = HandlerEntry(self.__acquisition_method_component)
        self.__handlers["scan-control-component"] = HandlerEntry(device_component_handlers[0])
        self.__handlers["scan-synchronized-control-component"] = HandlerEntry(device_component_handlers[1])
        self.__handlers["camera-control-component"] = HandlerEntry(device_component_handlers[2])

        # define whether this controller is in an error state
        self.is_error = False

        # define the list of methods
        method_component_ids = tuple(self.__acquisition_method_component._component_handler_map.keys())

        # define the method model
        self.method_model = Model.PropertyChangedPropertyModel[str](acquisition_configuration, "acquisition_method_component_id")
        self.method_model.value = self.method_model.value if self.method_model.value in method_component_ids else "basic-acquire"

        # define the list of acquisition modes
        acquisition_modes = [device_component_handler.component_id for device_component_handler in device_component_handlers]

        # define the acquisition mode - this determines which page is shown for further configuration
        self.acquisition_mode_model = Model.PropertyChangedPropertyModel[str](acquisition_configuration, "acquisition_device_component_id")
        self.acquisition_mode_model.value = self.acquisition_mode_model.value if self.acquisition_mode_model.value in acquisition_modes else acquisition_modes[0]

        # define a converter from the acquisition mode to an index for the page control
        self.acquisition_mode_to_index_converter = Converter.ValuesToIndexConverter(acquisition_modes)

        # define the progress value model, a simple bool 'is_acquiring' model, and a button text model that
        # updates according to whether acquire is running or not.
        self.progress_value_model = Model.PropertyModel[int](0)
        self.is_acquiring_model = Model.PropertyModel[bool](False)

        def is_enabled(b: typing.Optional[bool]) -> str:
            return _("Acquire") if not b else _("Cancel")

        self.button_text_model = Model.StreamValueModel(Stream.MapStream(
            Stream.PropertyChangedEventStream(self.is_acquiring_model, "value"),
            is_enabled))

        # create the button enabled property model.
        # the device_component_stream is a stream that maps the acquisition mode to the device component handler.
        # the get_acquire_valid_value_stream function maps the device component handler its
        # associated acquire_valid_value_stream, which is a stream of bools.
        # finally, the button_enabled_model turns the stream of bools into a property model.

        def find_method_component_handler(component_id: str | None) -> AcquisitionMethodComponentHandler | None:
            return self.__acquisition_method_component._component_handler_map.get(component_id, None) if component_id else None

        # create a stream of AcquisitionMethodComponentHandler based on the selected acquisition method.
        self.__method_component_stream = Stream.MapStream[str, AcquisitionMethodComponentHandler](Stream.PropertyChangedEventStream(self.method_model, "value"), find_method_component_handler)

        # create a stream of AcquisitionDeviceComponentHandler based on the selected acquisition mode.
        self.__device_component_stream = Stream.MapStream[str, AcquisitionDeviceComponentHandler](Stream.PropertyChangedEventStream(self.acquisition_mode_model, "value"), lambda component_id: device_component_handlers[acquisition_modes.index(component_id) if component_id in acquisition_modes else 0])

        def get_acquire_valid_value_stream(device_component: typing.Optional[AcquisitionDeviceComponentHandler]) -> Stream.AbstractStream[bool]:
            return device_component.acquire_valid_value_stream if device_component else Stream.ConstantStream(False)

        self.button_enabled_model = Model.StreamValueModel(StreamStreamer[bool](Stream.MapStream[AcquisitionDeviceComponentHandler, Stream.AbstractStream[bool]](self.__device_component_stream, get_acquire_valid_value_stream)))

        # create the stream of time/space usage with method/device component streams as inputs.
        time_space_usage_stream = TimeSpaceUsageStream(self.__method_component_stream, self.__device_component_stream)

        # break out the time and space for use in the UI.
        self.time_usage_model = Model.StreamValueModel[float](Stream.MapStream(time_space_usage_stream, operator.attrgetter("time")))
        self.space_usage_model = Model.StreamValueModel[int](Stream.MapStream(time_space_usage_stream, operator.attrgetter("space")))

        # define a progress task and acquisition. these are ephemeral and get closed after use in _acquire_data_stream.
        self.__progress_task: typing.Optional[asyncio.Task[None]] = None
        self.__acquisition: typing.Optional[Acquisition.Acquisition] = None

        self.bytes_converter = BytesToStringConverter()
        self.time_converter = TimeToStringConverter()

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_component_instance(identifier="acquisition-method-component"),
            u.create_spacing(8),
            u.create_divider(orientation="horizontal", height=8),
            u.create_row(
                u.create_label(text="Mode"),
                u.create_radio_button(text="Scan", value=acquisition_modes[0], group_value="@binding(acquisition_mode_model.value)"),
                u.create_radio_button(text="Scan Synchronized", value=acquisition_modes[1], group_value="@binding(acquisition_mode_model.value)"),
                u.create_radio_button(text="Camera", value=acquisition_modes[2], group_value="@binding(acquisition_mode_model.value)"),
                u.create_stretch(),
                spacing=8
            ),
            u.create_stack(
                u.create_column(u.create_component_instance(identifier="scan-control-component"), u.create_stretch(), spacing=8),
                u.create_column(u.create_component_instance(identifier="scan-synchronized-control-component"), u.create_stretch(), spacing=8),
                u.create_column(u.create_component_instance(identifier="camera-control-component"), u.create_stretch(), spacing=8),
                current_index="@binding(acquisition_mode_model.value, converter=acquisition_mode_to_index_converter)",
            ),
            u.create_divider(orientation="horizontal", height=8),
            u.create_row(u.create_label(text="@binding(time_usage_model.value, converter=time_converter)"), u.create_label(text="@binding(space_usage_model.value, converter=bytes_converter)"), u.create_stretch(), spacing=8),
            u.create_row(
                u.create_push_button(text="@binding(button_text_model.value)",
                                     enabled="@binding(button_enabled_model.value)", on_clicked="handle_button",
                                     width=80),
                u.create_progress_bar(value="@binding(progress_value_model.value)", width=180),
                u.create_stretch(),
                spacing=8,
            ),
            u.create_stretch(),
            spacing=8,
            margin=8
        )

    def close(self) -> None:
        self.button_enabled_model.close()
        self.button_enabled_model = typing.cast(typing.Any, None)
        self.is_acquiring_model.close()
        self.is_acquiring_model = typing.cast(typing.Any, None)
        self.progress_value_model.close()
        self.progress_value_model = typing.cast(typing.Any, None)
        self.button_text_model.close()
        self.button_text_model = typing.cast(typing.Any, None)
        for handler_entry in self.__handlers.values():
            if not handler_entry.used:
                handler_entry.handler.close()
        self.__handlers.clear()
        super().close()

    def handle_button(self, widget: UserInterfaceModule.Widget) -> None:
        # handle acquire button, which can either start or stop acquisition.
        if self.__acquisition and not self.__acquisition.is_finished:
            self.__acquisition.abort_acquire()
        else:
            device_component = self.__device_component_stream.value
            assert device_component

            method_component = self.__acquisition_method_component.current_item
            assert method_component

            self.start_acquisition(device_component, method_component)

    def start_acquisition(self, device_component: AcquisitionDeviceComponentHandler, method_component: AcquisitionMethodComponentHandler) -> None:

        # starting acquisition means building the device data stream using the acquisition device component and
        # then wrapping the device data stream using the acquisition method component.

        class DataChannelProvider(Acquisition.DataChannelProviderLike):
            def __init__(self, document_controller: DocumentController.DocumentController) -> None:
                self.__document_controller = document_controller

            def get_data_channel(self, title_base: str, channel_names: typing.Mapping[Acquisition.Channel, str], **kwargs: typing.Any) -> Acquisition.DataChannel:
                # create a data item data channel for converting data streams to data items, using partial updates and
                # minimizing extra copies where possible.

                # define a callback method to display the data item.
                def display_data_item(document_controller: DocumentController.DocumentController, data_item: DataItem.DataItem) -> None:
                    Facade.DocumentWindow(document_controller).display_data_item(Facade.DataItem(data_item))

                data_item_data_channel = DataChannel.DataItemDataChannel(self.__document_controller.document_model, title_base, channel_names)
                data_item_data_channel.on_display_data_item = weak_partial(display_data_item, self.__document_controller)

                return data_item_data_channel

        stem_device_controller = STEMController.STEMDeviceController()

        device_map: typing.Dict[str, STEMController.DeviceController] = dict()
        device_map["stem"] = stem_device_controller

        device_data_stream = device_component.build_acquisition_device().build_acquisition_device_data_stream(device_map)
        data_stream = method_component.build_acquisition_method().wrap_acquisition_device_data_stream(device_data_stream, device_map)
        drift_tracker = stem_device_controller.stem_controller.drift_tracker
        drift_logger = DriftTracker.DriftLogger(self.document_controller.document_model, drift_tracker, self.document_controller.event_loop) if drift_tracker else None

        def handle_acquire_finished() -> None:
            self.__acquisition = None

        Acquisition.session_manager.begin_acquisition(self.document_controller.document_model)

        self.__acquisition = Acquisition.start_acquire(data_stream,
                                                       data_stream.title or _("Acquire"),
                                                       data_stream.channel_names,
                                                       DataChannelProvider(self.document_controller),
                                                       drift_logger,
                                                       self.progress_value_model,
                                                       self.is_acquiring_model,
                                                       self.document_controller.event_loop,
                                                       handle_acquire_finished)


    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id in self.__handlers:
            self.__handlers[component_id].used = True
            return self.__handlers[component_id].handler
        return None


class AcquisitionPanel(Panel.Panel):
    """The acquisition panel holds the declarative component acquisition controller."""

    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: typing.Mapping[str, typing.Any]) -> None:
        super().__init__(document_controller, panel_id, "acquisition-panel")
        if Registry.get_component("stem_controller"):
            assert acquisition_configuration
            acquisition_preferences = AcquisitionPreferences.acquisition_preferences
            assert acquisition_preferences
            self.widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, AcquisitionController(document_controller, acquisition_configuration, acquisition_preferences))
        else:
            self.widget = document_controller.ui.create_column_widget()


class AcquisitionPreferencePanel:
    """Define a acquisition preference panel.

    This preference panel allows the user to customize the various controls.
    """

    def __init__(self) -> None:
        self.identifier = "nion.acquisition-panel"
        self.label = _("Acquisition")

    def build(self, ui: UserInterfaceModule.UserInterface, event_loop: typing.Optional[asyncio.AbstractEventLoop] = None, **kwargs: typing.Any) -> Declarative.DeclarativeWidget:
        u = Declarative.DeclarativeUI()

        class ControlDescriptionHandler(Declarative.Handler):
            def __init__(self, item: AcquisitionPreferences.ControlDescription) -> None:
                super().__init__()
                self.delay_converter = Converter.PhysicalValueToStringConverter("ms", 1000, "{:.0f}")
                self.item = item
                self.ui_view = u.create_column(
                    u.create_row(
                        u.create_label(text="@binding(item.name)", width=120),
                        u.create_line_edit(text="@binding(item.device_control_id)", width=160),
                        u.create_line_edit(text="@binding(item.delay, converter=delay_converter)", width=80),
                        u.create_stretch(),
                        spacing=8
                    ),
                    spacing=8
                )

        class Handler(Declarative.Handler):
            def __init__(self) -> None:
                super().__init__()
                self.sorted_controls = ListModel.FilteredListModel(container=AcquisitionPreferences.acquisition_preferences, items_key="control_customizations")
                self.sorted_controls.sort_key = operator.attrgetter("name")
                self.sorted_controls.filter = ListModel.PredicateFilter(lambda x: bool(x.is_customizable))
                self.ui_view = u.create_column(
                    u.create_row(
                        u.create_label(text="Name", width=120),
                        u.create_label(text="Control", width=160),
                        u.create_label(text="Delay", width=80),
                        u.create_stretch(),
                        spacing=8
                    ),
                    u.create_divider(orientation="horizontal"),
                    u.create_scroll_area(content=u.create_column(items="sorted_controls.items", item_component_id="control-component", spacing=4)),
                    spacing=8
                )

            def close(self) -> None:
                self.sorted_controls.close()
                self.sorted_controls = typing.cast(typing.Any, None)
                super().close()

            def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
                # this is called to construct contained declarative component handlers within this handler.
                if component_id == "control-component":
                    assert container is not None
                    assert item is not None
                    return ControlDescriptionHandler(item)
                return None

        return Declarative.DeclarativeWidget(ui, event_loop or asyncio.get_event_loop(), Handler())


# register the preference panel.
PreferencesDialog.PreferencesManager().register_preference_pane(AcquisitionPreferencePanel())


class AcquisitionPanelExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.instrumentation-kit.acquisition-panel"

    def __init__(self, api_broker: typing.Any) -> None:
        Workspace.WorkspaceManager().register_panel(AcquisitionPanel, "acquisition-panel", _("Acquisition"), ["left", "right"], "right", {"min-width": 320, "height": 60})

    def close(self) -> None:
        pass
