"""Acquisition panel.

The acquisition panel allows the user to select an acquisition method and an associated acquisition device,
which may be a direct device (camera or scan) or a virtual device (synchronized scan/camera).

All choices in the UI are persistent. The code supporting persistence is written so that it is easy to change
the persistent behavior to be profile- or project- based. It is currently file based. The persistence code is
based on schema/entity from nionswift. This closely ties this panel to nionswift schema/entity behavior, which
is still evolving.
"""

from __future__ import annotations

# system imports
import abc
import asyncio
import dataclasses
import functools
import gettext
import logging
import operator
import pathlib
import pkgutil
import numpy
import time
import typing
import uuid

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import camera_base
from nion.instrumentation import DataChannel
from nion.instrumentation import DriftTracker
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
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
from nion.utils import Geometry
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


@dataclasses.dataclass
class AcquisitionMethodResult:
    """Define result values for acquire handler apply function.

    data_stream is the result data stream.
    title_base is the base name for titles describing the acquire handler. may be empty, but not None.
    channel_names is a mapping from each acquisition channel to a display name.
    """
    data_stream: Acquisition.DataStream
    title_base: str
    channel_names: typing.Dict[Acquisition.Channel, str]


class AcquisitionMethodComponentHandler(ComponentHandler):
    """Define methods for acquisition method components."""

    def wrap_acquisition_device_data_stream(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
        # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
        raise NotImplementedError()


class BasicAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Basic acquisition method - single acquire from acquisition device with no options.

    Produces a data stream directly from the acquisition device.
    """

    component_id = "basic-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("None"))
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            spacing=8
        )

    def wrap_acquisition_device_data_stream(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
        # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
        return AcquisitionMethodResult(data_stream.add_ref(), str(), channel_names)


def wrap_acquisition_device_data_stream_for_sequence(data_stream: Acquisition.DataStream, count: int, channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
    # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
    if count > 1:
        # special case for framed-data-stream with sum operator; in the frame-by-frame case, the camera device
        # has the option of doing the processing itself and the operator will not be applied to the result. in
        # this case, the framed-data-stream-with-sum-operator is wrapped so that the processing can be performed
        # on the entire sequence. there is probably a better way to abstract this in the future.
        if isinstance(data_stream, Acquisition.FramedDataStream) and isinstance(data_stream.operator, Acquisition.SumOperator):
            return AcquisitionMethodResult(data_stream.data_stream.wrap_in_sequence(count).add_ref(), _("Sequence"), channel_names)
        else:
            return AcquisitionMethodResult(data_stream.wrap_in_sequence(count).add_ref(), _("Sequence"), channel_names)
    else:
        return AcquisitionMethodResult(data_stream.add_ref(), _("Sequence"), channel_names)


class SequenceAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Sequence acquisition method - a sequence of acquires from acquisition device with no options.

    Produces a data stream that is a sequence of the acquisition device data stream.

    The configuration entity should have an integer 'count' field.
    """

    component_id = "sequence-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("Sequence"))
        self.configuration = configuration
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

    def wrap_acquisition_device_data_stream(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
        length = max(1, self.configuration.count) if self.configuration.count else 1
        return wrap_acquisition_device_data_stream_for_sequence(data_stream, length, channel_names)


@dataclasses.dataclass
class ControlValuesRange:
    """Description of the range of values to apply to a control."""
    count: int
    start: float
    step: float


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

    def get_control_values_range(self) -> ControlValuesRange:
        """Return control info (count, start, step)."""
        control_description = self.__control_customization.control_description
        assert control_description
        count = max(1, self.control_values.count) if self.control_values.count else 1
        start = (self.control_values.start_value or 0)
        step = (self.control_values.step_value or 0)
        return ControlValuesRange(count, start * control_description.multiplier, step * control_description.multiplier)


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


def wrap_acquisition_device_data_stream_for_series(data_stream: Acquisition.DataStream, control_customization: AcquisitionPreferences.ControlCustomization, control_values_range: ControlValuesRange, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
    # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
    # get the associated control handler that was created in created_handler and used within the stack
    # of control handlers declarative components.
    assert data_stream
    if control_values_range.count > 1:
        class ActionDelegate(Acquisition.ActionDataStreamDelegate):
            def __init__(self, control_customization: AcquisitionPreferences.ControlCustomization,
                         device_map: typing.Mapping[str, DeviceController], starts: typing.Sequence[float],
                         steps: typing.Sequence[float]) -> None:
                self.control_customization = control_customization
                self.device_map = device_map
                self.starts = starts
                self.steps = steps
                self.original_values: typing.Sequence[float] = list()

            def start(self) -> None:
                control_description = self.control_customization.control_description
                assert control_description
                device_controller = self.device_map.get(control_description.device_id)
                if device_controller:
                    self.original_values = device_controller.get_values(self.control_customization)

            # define an action function to apply control values during acquisition
            def perform(self, index: Acquisition.ShapeType) -> None:
                # look up the device controller in the device_map using the device_id in the control description
                control_description = self.control_customization.control_description
                assert control_description
                device_controller = self.device_map.get(control_description.device_id)
                if device_controller:
                    # calculate the current value (in each dimension) and send the result to the
                    # device controller. the device controller may be a camera, scan, or stem device
                    # controller.
                    values = [start + step * i for start, step, i in zip(self.starts, self.steps, index)]
                    device_controller.update_values(self.control_customization, self.original_values, values)

            def finish(self) -> None:
                control_description = self.control_customization.control_description
                assert control_description
                device_controller = self.device_map.get(control_description.device_id)
                if device_controller:
                    device_controller.set_values(self.control_customization, self.original_values)

        # configure the action function and data stream using weak_partial to carefully control ref counts
        action_delegate = ActionDelegate(control_customization, device_map, [control_values_range.start], [control_values_range.step])
        data_stream = Acquisition.SequenceDataStream(Acquisition.ActionDataStream(data_stream, action_delegate), control_values_range.count)
    return AcquisitionMethodResult(data_stream.add_ref(), _("Series"), channel_names)


class SeriesAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Series acquisition method - a sequence of acquires from acquisition device with a changing parameter.

    Produces a data stream that is a sequence of the acquisition device data stream.

    The configuration entity should have a field for control_id and control_values_list. The control values list
    should have entities with fields for control_id, count, start_value, step_value. There should be one entry in
    the control values list for each possible control.
    """

    component_id = "series-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("1D Ramp"))
        self.configuration = configuration
        # the control UI is constructed as a stack with one item for each control_id.
        # the control_handlers is a map from the control_id to the SeriesControlHandler
        # for the control.
        self.__control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        # the selection storage model is a property model made by observing the control_id in the configuration.
        self.__selection_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "control_id")
        # the control combo box handler gives a choice of which control to use. in this case, the controls are iterated
        # by looking at control customizations. only 1d controls are presented.
        self._control_combo_box_handler = ComboBoxHandler(preferences,
                                                          "control_customizations",
                                                          operator.attrgetter("name"),
                                                          ListModel.PredicateFilter(lambda x: str(x.control_description.control_type) == "1d"),
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
        self.__selection_storage_model.close()
        self.__selection_storage_model = typing.cast(typing.Any, None)
        self.__control_handlers = typing.cast(typing.Any, None)
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
            return self.__control_handlers[control_id]
        return None

    def wrap_acquisition_device_data_stream(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
        # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
        # start by getting the selected control customization from the UI.
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            # get the associated control handler that was created in created_handler and used within the stack
            # of control handlers declarative components.
            control_handler = self.__control_handlers.get(control_customization.control_id)
            if control_handler and data_stream:
                # get the control values range from the control handler.
                control_values_range = control_handler.get_control_values_range()
                return wrap_acquisition_device_data_stream_for_series(
                    data_stream,
                    control_customization,
                    control_values_range,
                    device_map,
                    channel_names
                )
        return AcquisitionMethodResult(data_stream.add_ref(), _("Series"), channel_names)


def wrap_acquisition_device_data_stream_for_tableau(data_stream: Acquisition.DataStream,
                                                    control_customization: AcquisitionPreferences.ControlCustomization,
                                                    axis_id: typing.Optional[str],
                                                    x_control_values_range: ControlValuesRange,
                                                    y_control_values_range: ControlValuesRange,
                                                    device_map: typing.Mapping[str, DeviceController],
                                                    channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
    # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
    assert data_stream
    if x_control_values_range.count > 1 or y_control_values_range.count > 1:
        class ActionDelegate(Acquisition.ActionDataStreamDelegate):
            def __init__(self, control_customization: AcquisitionPreferences.ControlCustomization,
                         device_map: typing.Mapping[str, DeviceController], starts: typing.Sequence[float],
                         steps: typing.Sequence[float], axis_id: typing.Optional[str]) -> None:
                self.control_customization = control_customization
                self.device_map = device_map
                self.starts = starts
                self.steps = steps
                self.axis_id = axis_id
                self.original_values: typing.Sequence[float] = list()

            def start(self) -> None:
                control_description = self.control_customization.control_description
                assert control_description
                device_controller = self.device_map.get(control_description.device_id)
                if device_controller:
                    self.original_values = device_controller.get_values(self.control_customization, self.__resolve_axis())

            # define an action function to apply control values during acquisition
            def perform(self, index: Acquisition.ShapeType) -> None:
                # look up the device controller in the device_map using the device_id in the control description
                control_description = self.control_customization.control_description
                assert control_description
                device_controller = self.device_map.get(control_description.device_id)
                if device_controller:
                    # calculate the current value (in each dimension) and send the result to the
                    # device controller. the device controller may be a camera, scan, or stem device
                    # controller.
                    values = [start + step * i for start, step, i in zip(self.starts, self.steps, index)]
                    device_controller.update_values(self.control_customization, self.original_values, values, self.__resolve_axis())

            def finish(self) -> None:
                control_description = self.control_customization.control_description
                assert control_description
                device_controller = self.device_map.get(control_description.device_id)
                if device_controller:
                    device_controller.set_values(self.control_customization, self.original_values, self.__resolve_axis())

            def __resolve_axis(self) -> stem_controller.AxisType:
                # resolve the axis for the 2d control
                axis: stem_controller.AxisType = ("x", "y")
                for axis_description in typing.cast(STEMDeviceController,
                                                    self.device_map[
                                                        "stem"]).stem_controller.axis_descriptions:
                    if self.axis_id == axis_description.axis_id:
                        axis = axis_description.axis_type
                return axis

        # configure the action function and data stream using weak_partial to carefully control ref counts
        action_delegate = ActionDelegate(control_customization, device_map, [y_control_values_range.start, x_control_values_range.start], [y_control_values_range.step, x_control_values_range.step], axis_id)
        data_stream = Acquisition.CollectedDataStream(
            Acquisition.ActionDataStream(data_stream, action_delegate),
            (y_control_values_range.count, x_control_values_range.count),
            (Calibration.Calibration(), Calibration.Calibration()))
    return AcquisitionMethodResult(data_stream.add_ref(), _("Tableau"), channel_names)


class TableauAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Tableau acquisition method - a grid of acquires from acquisition device with a changing 2d parameter.

    Produces a data stream that is a 2d collection of the acquisition device data stream.

    The configuration entity should have a field for control_id, axis_id, x_control_values_list, and
    y_control_values_list. The control values lists should have entities with fields for control_id, count, start_value,
    step_value. There should be one entry in each control values list for each possible control.
    """

    component_id = "tableau-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("2D Ramp"))
        self.configuration = configuration
        # the control UIs are constructed as a stack with one item for each control_id.
        # the control_handlers is a map from the control_id to the SeriesControlHandler
        # for the control.
        self.__x_control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        self.__y_control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        # the selection storage model is a property model made by observing the control_id in the configuration.
        self.__selection_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "control_id")
        # the control combo box handler gives a choice of which control to use. in this case, the controls are iterated
        # by looking at control customizations. only 2d controls are presented.
        self._control_combo_box_handler = ComboBoxHandler(preferences,
                                                          "control_customizations",
                                                          operator.attrgetter("name"),
                                                          ListModel.PredicateFilter(lambda x: str(x.control_description.control_type) == "2d"),
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
        self.__selection_storage_model.close()
        self.__selection_storage_model = typing.cast(typing.Any, None)
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
            return self.__x_control_handlers[control_id]
        return None

    def wrap_acquisition_device_data_stream(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
        # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
        # start by getting the selected control customization from the UI.
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            # get the associated control handlers that were created in created_handler and used within the stack
            # of control handlers declarative components.
            x_control_handler = self.__x_control_handlers.get(control_customization.control_id)
            y_control_handler = self.__y_control_handlers.get(control_customization.control_id)
            if x_control_handler and y_control_handler and data_stream:
                # get the axis and control values ranges from the control handlers.
                axis_id = self.__axis_storage_model.value
                y_control_values_range = y_control_handler.get_control_values_range()
                x_control_values_range = x_control_handler.get_control_values_range()
                return wrap_acquisition_device_data_stream_for_tableau(data_stream, control_customization, axis_id,
                                                                       x_control_values_range, y_control_values_range,
                                                                       device_map, channel_names)
        return AcquisitionMethodResult(data_stream.add_ref(), _("Tableau"), channel_names)


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
            u.create_line_edit(text="@binding(item.offset, converter=offset_converter)", width=100),
            u.create_line_edit(text="@binding(item.exposure, converter=exposure_converter)", width=100),
            u.create_line_edit(text="@binding(item.count, converter=count_converter)", width=100),
            u.create_stretch(),
            spacing=8
        )


class MultipleAcquisitionMethodComponentHandler(AcquisitionMethodComponentHandler):
    """Multiple acquisition method - a sequential set of acquires from acquisition device with a control and exposure.

    Currently the control is always energy offset.

    Produces multiple data streams that are sequences of the acquisition device data stream.

    The configuration entity should have a list of sections where each section is an entity with offset, exposure,
    and count fields.
    """

    component_id = "multiple-acquire"

    def __init__(self, configuration: Schema.Entity, preferences: Observable.Observable):
        super().__init__(_("Multiple"))
        self.configuration = configuration
        # ensure that there are always a few example sections.
        if len(self.configuration.sections) == 0:
            self.configuration._append_item("sections", MultipleAcquireEntrySchema.create(None, {"offset": 0.0, "exposure": 0.001, "count": 2}))
            self.configuration._append_item("sections", MultipleAcquireEntrySchema.create(None, {"offset": 10.0, "exposure": 0.01, "count": 3}))
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(
                u.create_label(text=_("Offset"), width=100),
                u.create_label(text=_("Exposure"), width=100),
                u.create_label(text=_("Frames"), width=100),
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

    def wrap_acquisition_device_data_stream(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> AcquisitionMethodResult:
        # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
        assert AcquisitionPreferences.acquisition_preferences
        # define a list of data streams that will be acquired sequentially.
        data_streams: typing.List[Acquisition.DataStream] = list()
        # create a map from control_id to the control customization.
        control_customizations_map = {control_customization.control_id: control_customization for
                                      control_customization in
                                      AcquisitionPreferences.acquisition_preferences.control_customizations}
        # grab the stem and camera device controllers from the device map.
        stem_value_controller = device_map.get("stem")
        camera_value_controller = device_map.get("camera")
        # grab the control customizations and descriptions for energy offset and exposure
        control_customization_energy_offset = control_customizations_map["energy_offset"]
        control_customization_exposure = control_customizations_map["exposure"]
        assert control_customization_energy_offset
        assert control_customization_exposure
        control_description_energy_offset = control_customization_energy_offset.control_description
        control_description_exposure = control_customization_exposure.control_description
        assert control_description_energy_offset
        assert control_description_exposure
        # for each section, build the data stream.
        for item in self.configuration.sections:
            class ActionDelegate(Acquisition.ActionDataStreamDelegate):
                def __init__(self, offset_value: float, exposure_value: float) -> None:
                    self.offset_value = offset_value
                    self.exposure_value = exposure_value
                    self.original_energy_offset_values: typing.Sequence[float] = list()
                    self.original_exposure_values: typing.Sequence[float] = list()

                def start(self) -> None:
                    if stem_value_controller:
                        self.original_energy_offset_values = stem_value_controller.get_values(control_customization_energy_offset)
                    if camera_value_controller:
                        self.original_exposure_values = camera_value_controller.get_values(control_customization_exposure)

                # define an action function to apply control values during acquisition
                def perform(self, index: Acquisition.ShapeType) -> None:
                    if stem_value_controller:
                        stem_value_controller.set_values(control_customization_energy_offset, [self.offset_value])
                    if camera_value_controller:
                        camera_value_controller.set_values(control_customization_exposure, [self.exposure_value])

                def finish(self) -> None:
                    if stem_value_controller:
                        stem_value_controller.set_values(control_customization_energy_offset, self.original_energy_offset_values)
                    if camera_value_controller:
                        camera_value_controller.set_values(control_customization_exposure, self.original_exposure_values)

            # configure the action function and data stream using weak_partial to carefully control ref counts
            multi_acquire_entry = typing.cast(Schema.Entity, item)
            action_delegate = ActionDelegate(multi_acquire_entry.offset * control_description_energy_offset.multiplier,
                                             multi_acquire_entry.exposure * control_description_exposure.multiplier)
            data_streams.append(
                Acquisition.SequenceDataStream(Acquisition.ActionDataStream(data_stream, action_delegate),
                                               max(1, multi_acquire_entry.count)))

        # create a sequential data stream from the section data streams.
        sequential_data_stream = Acquisition.SequentialDataStream(data_streams)
        # the sequential data stream will emit channels of the form n.sub-channel. add a name for each of those
        # channels. do this by getting the name of the sub channel and constructing a new name for n.sub_channel
        # for each index.
        for channel in sequential_data_stream.channels:
            channel_names[channel] = " ".join((f"{int(channel.segments[0]) + 1} / {str(len(data_streams))}",
                                               channel_names[Acquisition.Channel(*channel.segments[1:])]))
        return AcquisitionMethodResult(sequential_data_stream.add_ref(), _("Multiple"), channel_names)


# register each component as an acquisition method component factory.
Registry.register_component(BasicAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(SequenceAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(SeriesAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(TableauAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})
Registry.register_component(MultipleAcquisitionMethodComponentHandler, {"acquisition-method-component-factory"})


class HardwareSourceHandler(Declarative.Handler):
    """A declarative component handler for a hardware source choice combo box.

    hardware_source_display_names is a read-only list of strings. It is an observable property.
    """

    def __init__(self, hardware_source_choice: HardwareSourceChoice.HardwareSourceChoice):
        super().__init__()
        self.hardware_source_choice = hardware_source_choice

        def property_changed(handler: HardwareSourceHandler, property: str) -> None:
            handler.notify_property_changed("hardware_source_display_names")

        # use weak_partial to avoid self reference and facilitate no-close.
        self.__listener = self.hardware_source_choice.hardware_sources_model.property_changed_event.listen(weak_partial(property_changed, self))

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_combo_box(items_ref="@binding(hardware_source_display_names)", current_index="@binding(hardware_source_choice.hardware_source_index_model.value)")

    def close(self) -> None:
        self.__listener.close()
        self.__listener = typing.cast(typing.Any, None)
        super().close()

    @property
    def hardware_source_display_names(self) -> typing.List[str]:
        return [x.display_name for x in self.hardware_source_choice.hardware_sources]


@dataclasses.dataclass
class HardwareSourceChannelDescription:
    """Describes a channel available on a camera.

    channel_id is unique for this channel (for persistence).
    processing_id is an optional processing identifier describing how to go from the native channel to this channel.
    display_name is the display name for the channel. it is displayed in the UI combo box.
    data_descriptor is the descriptor for the channel. it is used to provide downstream processing options.
    """
    channel_id: str
    processing_id: typing.Optional[str]
    display_name: str
    data_descriptor: DataAndMetadata.DataDescriptor

    def __str__(self) -> str:
        return self.display_name


# hardcoded list of channel descriptions. this list should be dynamically constructed from the devices eventually.
hardware_source_channel_descriptions = {
    "ronchigram": HardwareSourceChannelDescription("ronchigram", None, _("Ronchigram"), DataAndMetadata.DataDescriptor(False, 0, 2)),
    "eels_spectrum": HardwareSourceChannelDescription("eels_spectrum", "sum_project", _("Spectra"), DataAndMetadata.DataDescriptor(False, 0, 1)),
    "eels_image": HardwareSourceChannelDescription("eels_image", None, _("Image"), DataAndMetadata.DataDescriptor(False, 0, 2)),
    "image": HardwareSourceChannelDescription("image", None, _("Image"), DataAndMetadata.DataDescriptor(False, 0, 2)),
}


class HardwareSourceChannelChooserHandler(Declarative.Handler):
    """A declarative component handler for a hardware source choice channel combo box.

    The hardware_source_choice parameter is the associated hardware source choice from which to build the available
    channels.

    The channel_model parameter is a model representing the persistent channel description id. It will be read and
    written as the UI is presented, as the user changes the hardware source, and as the user makes an different
    channel selection.

    channel_descriptions is a list of channel descriptions. It is a read-only observable property.
    channel_index is the selected index. It is a read/write observable property.
    """

    def __init__(self, hardware_source_choice: HardwareSourceChoice.HardwareSourceChoice, channel_model: Model.PropertyModel[str]):
        super().__init__()
        self.__hardware_source_choice = hardware_source_choice
        self.__channel_descriptions: typing.List[HardwareSourceChannelDescription] = list()
        self.__channel_model = channel_model
        # use weak_partial to avoid self reference and facilitate no-close.
        self.__hardware_sources_list_changed_listener = hardware_source_choice.hardware_sources_model.property_changed_event.listen(
            weak_partial(HardwareSourceChannelChooserHandler.__update_channel_descriptions, self))
        self.__hardware_source_changed_listener = hardware_source_choice.hardware_source_index_model.property_changed_event.listen(
            weak_partial(HardwareSourceChannelChooserHandler.__update_channel_descriptions, self))
        self.__update_channel_descriptions("value")
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_combo_box(items_ref="@binding(channel_descriptions)", current_index="@binding(channel_index)")

    def close(self) -> None:
        self.__hardware_sources_list_changed_listener.close()
        self.__hardware_sources_list_changed_listener = typing.cast(typing.Any, None)
        self.__hardware_source_changed_listener.close()
        self.__hardware_source_changed_listener = typing.cast(typing.Any, None)
        self.__channel_model = typing.cast(typing.Any, None)
        self.__hardware_source_choice = typing.cast(typing.Any, None)
        super().close()

    @property
    def channel_descriptions(self) -> typing.List[HardwareSourceChannelDescription]:
        return self.__channel_descriptions

    @channel_descriptions.setter
    def channel_descriptions(self, value: typing.List[HardwareSourceChannelDescription]) -> None:
        # hack to work around lack of read-only binding
        pass

    @property
    def channel_index(self) -> int:
        # map from the channel model (channel identifier string) to a channel index.
        m = {o.channel_id: o for o in self.__channel_descriptions}
        return self.__channel_descriptions.index(m[self.__channel_model.value]) if self.__channel_model.value in m else 0

    @channel_index.setter
    def channel_index(self, value: int) -> None:
        # map from the channel index to the channel model (channel identifier string).
        channel_id = self.__channel_descriptions[value].channel_id if 0 <= value < len(self.__channel_descriptions) else "image"
        if channel_id != self.__channel_model.value:
            self.__channel_model.value = channel_id
            self.notify_property_changed("channel_index")

    def __update_channel_descriptions(self, k: str) -> None:
        # when the list of hardware sources changes or the selected hardware source changes, the list of available
        # channels needs to be updated. the selected channel may also be updated if it is no longer available.
        if k == "value":
            hardware_source = self.__hardware_source_choice.hardware_source
            if hardware_source and hardware_source.features.get("is_camera", False):
                camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, hardware_source)
                if getattr(camera_hardware_source.camera, "camera_type") == "ronchigram":
                    channel_descriptions = [hardware_source_channel_descriptions["ronchigram"]]
                elif getattr(camera_hardware_source.camera, "camera_type") == "eels":
                    channel_descriptions = [hardware_source_channel_descriptions["eels_spectrum"], hardware_source_channel_descriptions["eels_image"]]
                else:
                    channel_descriptions = [hardware_source_channel_descriptions["image"]]
            else:
                channel_descriptions = [hardware_source_channel_descriptions["image"]]
            output = hardware_source_channel_descriptions.get(self.__channel_model.value or str())
            if not output or output not in channel_descriptions:
                output = channel_descriptions[0]
            channel_descriptions_changed = channel_descriptions != self.__channel_descriptions
            self.__channel_descriptions = channel_descriptions
            self.__channel_model.value = output.channel_id
            if channel_descriptions_changed:
                self.notify_property_changed("channel_descriptions")


@dataclasses.dataclass
class AcquisitionDeviceResult:
    """Define result values for acquisition device component build function.

    data_stream is the result data stream.
    channel_names is a mapping from each acquisition channel to a display name.
    drift_tracker is an optional drift tracker, if used.
    device_map is a mapping from device_id to a DeviceController.
    """
    data_stream: Acquisition.DataStream
    channel_names: typing.Dict[Acquisition.Channel, str]
    drift_tracker: typing.Optional[DriftTracker.DriftTracker]
    device_map: typing.Dict[str, DeviceController]


class AcquisitionDeviceComponentHandler(ComponentHandler):
    """Define methods for acquisition device components."""

    def build_acquisition_device_data_stream(self) -> AcquisitionDeviceResult:
        # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.
        raise NotImplementedError()


@dataclasses.dataclass
class SynchronizedScanDescription:
    """Describes the parameters for a synchronized scan.

    context_text is a string explaining the current scan context.
    context_valid is True if the context is valid.
    scan_text is a string explaining the planned scan.
    scan_size is an IntSize of the scan size.
    drift_interval_lines is an int of how often to do drift correction.
    drift_interval_scans is an int of how often to do drift correction.
    enable_drift_correction is a bool of whether to do drift correction between scans.
    """
    context_text: str
    context_valid: bool
    scan_text: str
    scan_size: Geometry.IntSize
    drift_interval_lines: int
    drift_interval_scans: int
    enable_drift_correction: bool


class SynchronizedScanDescriptionValueStream(Stream.ValueStream[SynchronizedScanDescription]):
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
            weak_partial(SynchronizedScanDescriptionValueStream.__hardware_source_stream_changed, self))

        def property_changed(vs: SynchronizedScanDescriptionValueStream, property: str) -> None:
            vs.__update_context()

        self.__scan_width_changed_listener = self.__scan_width_model.property_changed_event.listen(
            weak_partial(property_changed, self))
        self.__stem_controller = typing.cast(stem_controller.STEMController, Registry.get_component("stem_controller"))
        self.__stem_controller_property_listener: typing.Optional[Event.EventListener] = None
        self.__scan_context_changed_listener: typing.Optional[Event.EventListener] = None
        if self.__stem_controller:
            self.__stem_controller_property_listener = self.__stem_controller.property_changed_event.listen(
                weak_partial(SynchronizedScanDescriptionValueStream.__stem_controller_property_changed, self))
            self.__scan_context_changed_listener = self.__stem_controller.scan_context_changed_event.listen(
                weak_partial(SynchronizedScanDescriptionValueStream.__scan_context_changed, self))
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
        if maybe_camera_hardware_source and maybe_scan_hardware_source:
            camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, maybe_camera_hardware_source)
            scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, maybe_scan_hardware_source)
            scan_context = scan_hardware_source.scan_context

            exposure_time = camera_hardware_source.get_frame_parameters(0).exposure_ms / 1000
            scan_width = self.__scan_width_model.value
            assert scan_width is not None

            scan_context_size = scan_context.size
            if scan_context.is_valid and scan_hardware_source.line_scan_enabled and scan_hardware_source.line_scan_vector:
                assert scan_context_size
                calibration = scan_context.calibration
                start = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[0])
                end = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[1])
                length = int(Geometry.distance(start, end) * scan_context_size.height)
                max_dim = max(scan_context_size.width, scan_context_size.height)
                length_str = calibration.convert_to_calibrated_size_str(length, value_range=(0, max_dim), samples=max_dim)
                line_str = _("Line Scan")
                context_text = f"{line_str} {length_str}"
                scan_length = max(scan_width or 0, 1)
                scan_text = f"{scan_length} px"
                scan_size = Geometry.IntSize(height=1, width=scan_length)
                drift_interval_lines = 0
                drift_interval_scans = scan_hardware_source.calculate_drift_scans()
                enable_drift_correction = False
                self.send_value(SynchronizedScanDescription(context_text, True, scan_text, scan_size, drift_interval_lines, drift_interval_scans, enable_drift_correction))
            elif scan_context.is_valid and scan_hardware_source.subscan_enabled and scan_hardware_source.subscan_region:
                assert scan_context_size
                calibration = scan_context.calibration
                width = scan_hardware_source.subscan_region.width * scan_context_size.width
                height = scan_hardware_source.subscan_region.height * scan_context_size.height
                width_str = calibration.convert_to_calibrated_size_str(width,
                                                                       value_range=(0, scan_context_size.width),
                                                                       samples=scan_context_size.width)
                height_str = calibration.convert_to_calibrated_size_str(height,
                                                                        value_range=(0, scan_context_size.height),
                                                                        samples=scan_context_size.height)
                rect_str = _("Subscan")
                context_text = f"{rect_str} {width_str} x {height_str}"
                scan_height = int(scan_width * height / width)
                scan_text = f"{scan_width} x {scan_height}"
                scan_size = Geometry.IntSize(height=scan_height, width=scan_width)
                drift_interval_lines = scan_hardware_source.calculate_drift_lines(scan_width, exposure_time)
                drift_interval_scans = scan_hardware_source.calculate_drift_scans()
                enable_drift_correction = False
                self.send_value(SynchronizedScanDescription(context_text, True, scan_text, scan_size, drift_interval_lines, drift_interval_scans, enable_drift_correction))
            elif scan_context.is_valid:
                assert scan_context_size
                calibration = scan_context.calibration
                width = scan_context_size.width
                height = scan_context_size.height
                width_str = calibration.convert_to_calibrated_size_str(width,
                                                                       value_range=(0, scan_context_size.width),
                                                                       samples=scan_context_size.width)
                height_str = calibration.convert_to_calibrated_size_str(height,
                                                                        value_range=(0, scan_context_size.height),
                                                                        samples=scan_context_size.height)
                data_str = _("Context Scan")
                context_text = f"{data_str} {width_str} x {height_str}"
                scan_height = int(scan_width * height / width)
                scan_text = f"{scan_width} x {scan_height}"
                scan_size = Geometry.IntSize(height=scan_height, width=scan_width)
                drift_interval_lines = scan_hardware_source.calculate_drift_lines(scan_width, exposure_time)
                drift_interval_scans = scan_hardware_source.calculate_drift_scans()
                enable_drift_correction = False
                self.send_value(SynchronizedScanDescription(context_text, True, scan_text, scan_size, drift_interval_lines, drift_interval_scans, enable_drift_correction))
            else:
                context_text = _("No scan context")
                scan_text = str()
                self.send_value(SynchronizedScanDescription(context_text, False, scan_text, Geometry.IntSize(), 0, 0, False))


class CameraExposureValueStream(Stream.ValueStream[float]):
    """A value stream of the camera exposure of the latest values of a hardware source stream.

    Listens to the hardware_source_stream for changes. And then listens to the current hardware source
    for parameter changes. Sends out new exposure_time_ms values when changed.

    Always uses profile 0 for camera exposure.
    """
    def __init__(self, hardware_source_stream: Stream.AbstractStream[HardwareSource.HardwareSource]):
        super().__init__()
        self.__hardware_source_stream = hardware_source_stream.add_ref()
        # use weak_partial to avoid self reference and facilitate no-close.
        self.__hardware_source_stream_listener = self.__hardware_source_stream.value_stream.listen(
            weak_partial(CameraExposureValueStream.__hardware_source_stream_changed, self))
        self.__frame_parameters_changed_listener: typing.Optional[Event.EventListener] = None
        hardware_source = hardware_source_stream.value
        assert hardware_source
        self.__hardware_source_stream_changed(hardware_source)

    def about_to_delete(self) -> None:
        if self.__frame_parameters_changed_listener:
            self.__frame_parameters_changed_listener.close()
            self.__frame_parameters_changed_listener = None
        self.__hardware_source_stream_listener.close()
        self.__hardware_source_stream_listener = typing.cast(typing.Any, None)
        self.__hardware_source_stream.remove_ref()
        super().about_to_delete()

    def __hardware_source_stream_changed(self, hardware_source: HardwareSource.HardwareSource) -> None:
        # when the hardware source choice changes, update the frame parameters listener.
        if self.__frame_parameters_changed_listener:
            self.__frame_parameters_changed_listener.close()
            self.__frame_parameters_changed_listener = None
        if hardware_source and hardware_source.features.get("is_camera"):
            camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, hardware_source)
            self.send_value(camera_hardware_source.get_frame_parameters(0).exposure_ms)
            # use weak_partial to avoid self reference and facilitate no-close.
            self.__frame_parameters_changed_listener = camera_hardware_source.frame_parameters_changed_event.listen(
                weak_partial(CameraExposureValueStream.__frame_parameters_changed, self, camera_hardware_source))

    def __frame_parameters_changed(self, camera_hardware_source: camera_base.CameraHardwareSource, profile_index: int, frame_parameters: camera_base.CameraFrameParameters) -> None:
        if profile_index == 0:
            self.send_value(camera_hardware_source.get_frame_parameters(0).exposure_ms)

    @property
    def exposure_time_ms(self) -> float:
        return self.value if self.value else 0.0

    @exposure_time_ms.setter
    def exposure_time_ms(self, exposure_time_ms: float) -> None:
        if exposure_time_ms and exposure_time_ms > 0:
            # cast to typing.Any until HardwareSource protocols are implemented sanely.
            hardware_source = typing.cast(typing.Any, self.__hardware_source_stream.value)
            if hardware_source:
                frame_parameters = hardware_source.get_frame_parameters(0)
                frame_parameters.exposure_ms = exposure_time_ms
                hardware_source.set_frame_parameters(0, frame_parameters)


class CameraDetailsHandler(Declarative.Handler):
    """A declarative component handler for a row describing a camera device.

    The hardware_source_choice parameter is the associated hardware source choice.
    """

    def __init__(self, hardware_source_choice: HardwareSourceChoice.HardwareSourceChoice):
        super().__init__()

        # the exposure value stream gives the stream of exposure values from the hardware source choice
        self.exposure_value_stream = typing.cast(CameraExposureValueStream, CameraExposureValueStream(HardwareSourceChoice.HardwareSourceChoiceStream(hardware_source_choice)).add_ref())
        # the exposure model converts the exposure value stream to a property model that supports binding.
        self.exposure_model = Model.StreamValueModel(self.exposure_value_stream)
        # the exposure value converter converts the exposure value to a string and back in the line edit.
        self.exposure_value_converter = Converter.PhysicalValueToStringConverter("ms", 1, "{:.4f}")

        # need to explicitly watch the exposure model for a value change from the UI so that it can update the exposure
        # value stream. this is a hack; check whether there is a better way when encountering this code in the future -
        # something like standardized support for setting values in the value streams.
        self.__exposure_model_listener = self.exposure_model.property_changed_event.listen(weak_partial(CameraDetailsHandler.__exposure_changed, self))

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_row(
            u.create_stack(
                u.create_row(
                    u.create_label(text=_("Camera Exposure Time")),
                    u.create_line_edit(text="@binding(exposure_model.value, converter=exposure_value_converter)", width=80),
                    u.create_stretch(),
                    spacing=8
                )
            ),
            u.create_stretch()
        )

    def close(self) -> None:
        self.__exposure_model_listener.close()
        self.__exposure_model_listener = typing.cast(typing.Any, None)
        self.exposure_model.close()
        self.exposure_model = typing.cast(typing.Any, None)
        self.exposure_value_stream.remove_ref()
        self.exposure_value_stream = typing.cast(typing.Any, None)
        super().close()

    def __exposure_changed(self, k: str) -> None:
        if k == "value":
            self.exposure_value_stream.exposure_time_ms = self.exposure_model.value if self.exposure_model.value else 0.0


def build_synchronized_device_data_stream(scan_hardware_source: scan_base.ScanHardwareSource, scan_context_description: SynchronizedScanDescription, camera_hardware_source: camera_base.CameraHardwareSource, channel: typing.Optional[str] = None) -> AcquisitionDeviceResult:
    # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.

    # first get the camera hardware source and the camera channel description.
    if channel in hardware_source_channel_descriptions:
        camera_channel_description = hardware_source_channel_descriptions[channel]
    else:
        camera_channel_description = hardware_source_channel_descriptions["image"]
    assert camera_hardware_source is not None
    assert camera_channel_description is not None

    assert scan_hardware_source is not None
    assert scan_context_description is not None

    # configure the camera hardware source processing. always use camera parameters at index 0.
    camera_frame_parameters = camera_hardware_source.get_frame_parameters(0)
    if camera_channel_description.processing_id:
        camera_frame_parameters.processing = camera_channel_description.processing_id
    else:
        camera_frame_parameters.processing = None

    # configure the scan uuid and scan frame parameters.
    scan_count = 1
    scan_size = scan_context_description.scan_size
    scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
    scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters, typing.cast(typing.Tuple[int, int], scan_size))
    scan_frame_parameters.scan_id = uuid.uuid4()

    # set up drift correction, if enabled in the scan control panel. this can be used for intra-scan drift
    # correction. the synchronized acquisition can also utilize the drift tracker associated with the scan
    # hardware source directly, which watches the first channel for drift in sequences of scans. the drift
    # tracker is separate at the moment.
    drift_correction_functor: typing.Optional[Acquisition.DataStreamFunctor] = None
    section_height: typing.Optional[int] = None
    drift_tracker = scan_hardware_source.drift_tracker
    if drift_tracker and scan_context_description.drift_interval_lines > 0:
        drift_correction_functor = DriftTracker.DriftCorrectionDataStreamFunctor(scan_hardware_source, scan_frame_parameters, drift_tracker, scan_context_description.drift_interval_scans)
        section_height = scan_context_description.drift_interval_lines
    enable_drift_tracker = drift_tracker is not None and scan_context_description.enable_drift_correction

    # build the synchronized data stream. this will also automatically include scan-channel drift correction.
    synchronized_scan_data_stream = scan_base.make_synchronized_scan_data_stream(
        scan_hardware_source=scan_hardware_source,
        scan_frame_parameters=scan_frame_parameters,
        camera_hardware_source=camera_hardware_source,
        camera_frame_parameters=camera_frame_parameters,
        scan_data_stream_functor=drift_correction_functor,
        section_height=section_height,
        scan_count=scan_count,
        include_raw=True,
        include_summed=False,
        enable_drift_tracker=enable_drift_tracker
    )

    # construct the channel names.
    op = _("Synchronized")
    channel_names: typing.Dict[Acquisition.Channel, str] = dict()
    for c in scan_hardware_source.get_enabled_channels():
        channel_state = scan_hardware_source.get_channel_state(c)
        channel_index_segment = str(scan_hardware_source.get_channel_index(channel_state.channel_id))
        channel_names[Acquisition.Channel(scan_hardware_source.hardware_source_id, channel_index_segment)] = f"{op} {channel_state.name}"
    channel_names[Acquisition.Channel(camera_hardware_source.hardware_source_id)] = f"{op} {camera_hardware_source.get_signal_name(camera_frame_parameters)}"

    drift_tracker = scan_hardware_source.drift_tracker

    # construct the device map for this acquisition device.
    device_map: typing.Dict[str, DeviceController] = dict()
    device_map["stem"] = STEMDeviceController()
    device_map["camera"] = CameraDeviceController(camera_hardware_source, camera_frame_parameters)
    device_map["magnification"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)
    device_map["scan"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)

    return AcquisitionDeviceResult(synchronized_scan_data_stream.add_ref(), channel_names, drift_tracker, device_map)


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

        # the camera hardware source choice model is a property model made by observing the camera_device_id in the configuration.
        self.__camera_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, "camera_device_id")

        # the camera hardware source choice associates a camera_device_id with a hardware source and also facilitates a combo box.
        self.__camera_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__camera_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_camera", False))

        # the camera hardware source channel model is a property model made by observing the camera_channel_id in the configuration.
        self.__camera_hardware_source_channel_model = Model.PropertyChangedPropertyModel[str](configuration, "camera_channel_id")

        # the scan hardware source choice model is a property model made by observing the scan_device_id in the configuration.
        self.__scan_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, "scan_device_id")

        # the scan hardware source choice associates a camera_device_id with a hardware source and also facilitates a combo box.
        # it will not be presented in the UI unless multiple choices exist.
        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__scan_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_scanning", False))

        # the scan width model is a property model made by observing the scan_width property in the configuration.
        self.scan_width = Model.PropertyChangedPropertyModel[int](configuration, "scan_width")

        # the scan context description value stream observes the scan and camera hardware sources to produce
        # a description of the upcoming scan. it also supplies a context_valid flag which can be used to enable
        # the acquire button in the UI.
        self.__scan_context_description_value_stream = SynchronizedScanDescriptionValueStream(
            HardwareSourceChoice.HardwareSourceChoiceStream(self.__camera_hardware_source_choice),
            HardwareSourceChoice.HardwareSourceChoiceStream(self.__scan_hardware_source_choice),
            self.scan_width,
            asyncio.get_event_loop_policy().get_event_loop()).add_ref()

        # the scan context value model is the text description of the scan context extracted from the value stream.
        self.scan_context_value_model = Model.StreamValueModel(Stream.MapStream(
            self.__scan_context_description_value_stream,
            lambda x: x.context_text if x is not None else str()
        ))

        # the scan context value model is the text description of the upcoming scan extracted from the value stream.
        self.scan_value_model = Model.StreamValueModel(Stream.MapStream(
            self.__scan_context_description_value_stream,
            lambda x: x.scan_text if x is not None else str()
        ))

        # a converter for the scan width.
        self.scan_width_converter = Converter.IntegerToStringConverter()

        # the acquire valid value stream is a value stream of bool values extracted from the scan context description.
        # it is used to enable the acquire button. but to do so, this stream must be read from the enclosing
        # declarative component handler. this stream is not used within this class. perhaps there is a better way to
        # do this.
        self.acquire_valid_value_stream = Stream.MapStream(self.__scan_context_description_value_stream,
                                                           lambda x: x.context_valid if x is not None else False).add_ref()

        u = Declarative.DeclarativeUI()

        column_items = list()
        column_items.append(
            u.create_row(
                u.create_component_instance(identifier="acquisition-device-component"),
                u.create_component_instance(identifier="acquisition-device-component-output"),
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
        if len(self.__scan_hardware_source_choice.hardware_sources) > 1:
            column_items.append(
                u.create_row(
                    u.create_label(text="Scan Device"),
                    u.create_component_instance(identifier="scan-component"),
                    u.create_stretch(),
                    spacing=8
                )
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
        self.__camera_hardware_source_choice.close()
        self.__camera_hardware_source_choice = typing.cast(typing.Any, None)
        self.__camera_hardware_source_choice_model.close()
        self.__camera_hardware_source_choice_model = typing.cast(typing.Any, None)
        self.__camera_hardware_source_channel_model.close()
        self.__camera_hardware_source_channel_model = typing.cast(typing.Any, None)
        self.__scan_hardware_source_choice.close()
        self.__scan_hardware_source_choice = typing.cast(typing.Any, None)
        self.__scan_hardware_source_choice_model.close()
        self.__scan_hardware_source_choice_model = typing.cast(typing.Any, None)
        self.scan_width.close()
        self.scan_width = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "acquisition-device-component":
            return HardwareSourceHandler(self.__camera_hardware_source_choice)
        elif component_id == "acquisition-device-component-output":
            return HardwareSourceChannelChooserHandler(self.__camera_hardware_source_choice, self.__camera_hardware_source_channel_model)
        elif component_id == "acquisition-device-component-details":
            return CameraDetailsHandler(self.__camera_hardware_source_choice)
        elif component_id == "scan-component":
            return HardwareSourceHandler(self.__scan_hardware_source_choice)
        return None

    def build_acquisition_device_data_stream(self) -> AcquisitionDeviceResult:
        # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.
        camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, self.__camera_hardware_source_choice.hardware_source)
        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source_choice.hardware_source)
        scan_context_description = self.__scan_context_description_value_stream.value
        assert scan_context_description
        return build_synchronized_device_data_stream(scan_hardware_source, scan_context_description, camera_hardware_source, self.__camera_hardware_source_channel_model.value)


def build_camera_device_data_stream(camera_hardware_source: camera_base.CameraHardwareSource, channel: typing.Optional[str] = None) -> AcquisitionDeviceResult:
    # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.

    # first get the camera hardware source and the camera channel description.
    if channel in hardware_source_channel_descriptions:
        camera_channel_description = hardware_source_channel_descriptions[channel]
    else:
        camera_channel_description = hardware_source_channel_descriptions["image"]
    assert camera_hardware_source is not None
    assert camera_channel_description is not None

    # configure the camera hardware source processing. always use camera parameters at index 0.
    camera_frame_parameters = camera_hardware_source.get_frame_parameters(0)
    if camera_channel_description.processing_id:
        camera_frame_parameters.processing = camera_channel_description.processing_id
    else:
        camera_frame_parameters.processing = None

    # gather the instrument metadata
    instrument_metadata: typing.Dict[str, typing.Any] = dict()
    stem_controller = Registry.get_component('stem_controller')
    assert stem_controller
    scan_base.update_instrument_properties(instrument_metadata, stem_controller, None)

    # construct the camera frame data stream. add processing.
    camera_data_stream = camera_base.CameraFrameDataStream(camera_hardware_source, camera_frame_parameters)
    processed_camera_data_stream: Acquisition.DataStream = camera_data_stream
    if camera_frame_parameters.processing == "sum_project":
        processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream,
                                                                    operator=Acquisition.SumOperator(axis=0))
    elif camera_frame_parameters.processing == "sum_masked":
        active_masks = camera_frame_parameters.active_masks
        if active_masks:
            operator = Acquisition.StackedDataStreamOperator(
                [Acquisition.MaskedSumOperator(active_mask) for active_mask in active_masks])
            processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream,
                                                                        operator=operator)
        else:
            operator = Acquisition.StackedDataStreamOperator([Acquisition.SumOperator()])
            processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream,
                                                                        operator=operator)

    # construct the channel names.
    channel_names: typing.Dict[Acquisition.Channel, str] = dict()
    channel_names[Acquisition.Channel(camera_hardware_source.hardware_source_id)] = camera_hardware_source.get_signal_name(camera_frame_parameters)

    # construct the device map for this acquisition device.
    device_map: typing.Dict[str, DeviceController] = dict()
    device_map["stem"] = STEMDeviceController()
    device_map["camera"] = CameraDeviceController(camera_hardware_source, camera_frame_parameters)

    return AcquisitionDeviceResult(processed_camera_data_stream.add_ref(), channel_names, None, device_map)


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

        # the camera hardware source choice model is a property model made by observing the camera_device_id in the configuration.
        self.__camera_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration,
                                                                                             "camera_device_id")

        # the camera hardware source choice associates a camera_device_id with a hardware source and also facilitates a combo box.
        self.__camera_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(
            self.__camera_hardware_source_choice_model,
            lambda hardware_source: hardware_source.features.get("is_camera", False))

        # the camera hardware source channel model is a property model made by observing the camera_channel_id in the configuration.
        self.__camera_hardware_source_channel_model = Model.PropertyChangedPropertyModel[str](configuration,
                                                                                              "camera_channel_id")

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_row(
                u.create_component_instance(identifier="acquisition-device-component"),
                u.create_component_instance(identifier="acquisition-device-component-output"),
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
        self.__camera_hardware_source_choice.close()
        self.__camera_hardware_source_choice = typing.cast(typing.Any, None)
        self.__camera_hardware_source_choice_model.close()
        self.__camera_hardware_source_choice_model = typing.cast(typing.Any, None)
        self.__camera_hardware_source_channel_model.close()
        self.__camera_hardware_source_channel_model = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "acquisition-device-component":
            return HardwareSourceHandler(self.__camera_hardware_source_choice)
        elif component_id == "acquisition-device-component-output":
            return HardwareSourceChannelChooserHandler(self.__camera_hardware_source_choice, self.__camera_hardware_source_channel_model)
        elif component_id == "acquisition-device-component-details":
            return CameraDetailsHandler(self.__camera_hardware_source_choice)
        return None

    def build_acquisition_device_data_stream(self) -> AcquisitionDeviceResult:
        # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.

        # first get the camera hardware source and the camera channel description.
        camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, self.__camera_hardware_source_choice.hardware_source)

        return build_camera_device_data_stream(camera_hardware_source, self.__camera_hardware_source_channel_model.value)


def build_scan_device_data_stream(scan_hardware_source: scan_base.ScanHardwareSource) -> AcquisitionDeviceResult:
    # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.

    assert scan_hardware_source is not None

    # configure the scan uuid and scan frame parameters.
    scan_uuid = uuid.uuid4()
    scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
    scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters)
    scan_frame_parameters.scan_id = scan_uuid

    # gather the scan metadata.
    scan_metadata: typing.Dict[str, typing.Any] = dict()
    scan_base.update_scan_metadata(scan_metadata, scan_hardware_source.hardware_source_id,
                                   scan_hardware_source.display_name, scan_frame_parameters, scan_uuid, dict())

    instrument_metadata: typing.Dict[str, typing.Any] = dict()
    scan_base.update_instrument_properties(instrument_metadata, scan_hardware_source.stem_controller,
                                           scan_hardware_source.scan_device)

    # build the scan frame data stream.
    scan_data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, scan_frame_parameters, scan_hardware_source.drift_tracker)

    # potentially break the scan into multiple sections; this is an unused capability currently.
    scan_size = scan_data_stream.scan_size
    section_height = scan_size.height
    section_count = (scan_size.height + section_height - 1) // section_height
    collectors: typing.List[Acquisition.CollectedDataStream] = list()
    for section in range(section_count):
        start = section * section_height
        stop = min(start + section_height, scan_size.height)
        collectors.append(Acquisition.CollectedDataStream(scan_data_stream, (stop - start, scan_size.width), scan_frame_parameters.get_scan_calibrations()))
    collector = Acquisition.StackedDataStream(collectors)
    # construct the channel names.
    channel_names: typing.Dict[Acquisition.Channel, str] = dict()
    for c in scan_hardware_source.get_enabled_channels():
        channel_state = scan_hardware_source.get_channel_state(c)
        channel_index_segment = str(scan_hardware_source.get_channel_index(channel_state.channel_id))
        channel_names[Acquisition.Channel(scan_hardware_source.hardware_source_id, channel_index_segment)] = channel_state.name

    # construct the device map for this acquisition device.
    device_map: typing.Dict[str, DeviceController] = dict()
    device_map["stem"] = STEMDeviceController()
    device_map["magnification"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)
    device_map["scan"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)

    return AcquisitionDeviceResult(collector.add_ref(), channel_names, None, device_map)


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

        # the scan hardware source choice model is a property model made by observing the scan_device_id in the configuration.
        self.__scan_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, "scan_device_id")

        # the scan hardware source choice associates a camera_device_id with a hardware source and also facilitates a combo box.
        # it will not be presented in the UI unless multiple choices exist.
        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__scan_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_scanning", False))

        u = Declarative.DeclarativeUI()
        if len(self.__scan_hardware_source_choice.hardware_sources) > 1:
            self.ui_view = u.create_column(
                u.create_row(
                    u.create_label(text="Scan Device"),
                    u.create_component_instance(identifier="scan-component"),
                    u.create_stretch(),
                    spacing=8
                ),
                spacing=8,
            )
        else:
            self.ui_view = u.create_column()

    def close(self) -> None:
        self.__scan_hardware_source_choice.close()
        self.__scan_hardware_source_choice = typing.cast(typing.Any, None)
        self.__scan_hardware_source_choice_model.close()
        self.__scan_hardware_source_choice_model = typing.cast(typing.Any, None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "scan-component":
            return HardwareSourceHandler(self.__scan_hardware_source_choice)
        return None

    def build_acquisition_device_data_stream(self) -> AcquisitionDeviceResult:
        # build the device data stream. return the data stream, channel names, drift tracker (optional), and device map.

        # first get the scan hardware source.
        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source_choice.hardware_source)
        assert scan_hardware_source is not None

        return build_scan_device_data_stream(scan_hardware_source)


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
    "scan_width": Schema.prop(Schema.INT, default=32),
})

# ScanAcquisitionDeviceComponentHandler
Schema.entity("acquisition_device_component_scan", AcquisitionDeviceComponentSchema, None, {
    "scan_device_id": Schema.prop(Schema.STRING),
})

# CameraAcquisitionDeviceComponentHandler
Schema.entity("acquisition_device_component_camera", AcquisitionDeviceComponentSchema, None, {
    "camera_device_id": Schema.prop(Schema.STRING),
    "camera_channel_id": Schema.prop(Schema.STRING),
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


class AcquisitionState:
    def __init__(self) -> None:
        self._acquisition: typing.Optional[Acquisition.Acquisition] = None
        self.is_error = False

    def _start(self, framed_data_stream: Acquisition.FramedDataStream) -> None:
        self._acquisition = Acquisition.Acquisition(framed_data_stream)
        self.is_error = False

    def _end(self) -> None:
        assert self._acquisition
        self._acquisition.close()
        self._acquisition = None

    @property
    def _acquisition_ex(self) -> Acquisition.Acquisition:
        assert self._acquisition
        return self._acquisition

    @property
    def is_active(self) -> bool:
        return self._acquisition is not None

    def abort_acquire(self) -> None:
        if self._acquisition:
            self._acquisition.abort_acquire()


def _acquire_data_stream(data_stream: Acquisition.DataStream,
                         document_controller: DocumentController.DocumentController,
                         acquisition_state: AcquisitionState,
                         progress_value_model: Model.PropertyModel[int],
                         is_acquiring_model: Model.PropertyModel[bool],
                         title_base: str,
                         channel_names: typing.Dict[Acquisition.Channel, str],
                         drift_tracker: typing.Optional[DriftTracker.DriftTracker]) -> None:
    """Perform acquisition of of the data stream."""

    # define a callback method to display the data item.
    def display_data_item(document_controller: DocumentController.DocumentController, data_item: DataItem.DataItem) -> None:
        Facade.DocumentWindow(document_controller).display_data_item(Facade.DataItem(data_item))

    # create a data item data channel for converting data streams to data items, using partial updates and
    # minimizing extra copies where possible.
    data_item_data_channel = DataChannel.DataItemDataChannel(document_controller.document_model, title_base, channel_names)
    data_item_data_channel.on_display_data_item = weak_partial(display_data_item, document_controller)
    framed_data_stream = Acquisition.FramedDataStream(data_stream, data_channel=data_item_data_channel).add_ref()

    # create the acquisition state/controller object based on the data item data channel data stream.
    acquisition_state._start(framed_data_stream)

    # configure the scan drift logger if required. the drift tracker here is only enabled if using the
    # scan hardware source drift tracker.
    scan_drift_logger = DriftTracker.DriftLogger(document_controller.document_model, drift_tracker,
                                                 document_controller.event_loop) if drift_tracker else None

    # define a method that gets called when the async acquisition method finished. this closes the various
    # objects and updates the UI as 'complete'.
    def finish_grab_async(framed_data_stream: Acquisition.FramedDataStream,
                          acquisition_state: AcquisitionState,
                          scan_drift_logger: typing.Optional[DriftTracker.DriftLogger],
                          progress_task: typing.Optional[asyncio.Task[None]],
                          progress_value_model: Model.PropertyModel[int],
                          is_acquiring_model: Model.PropertyModel[bool]) -> None:
        acquisition_state._end()
        acquisition_state.is_error = framed_data_stream.is_error
        framed_data_stream.remove_ref()
        if scan_drift_logger:
            scan_drift_logger.close()
        is_acquiring_model.value = False
        if progress_task:
            progress_task.cancel()
        progress_value_model.value = 100

    # manage the 'is_acquiring' state.
    is_acquiring_model.value = True

    # define a task to update progress every 250ms.
    async def update_progress(acquisition: Acquisition.Acquisition, progress_value_model: Model.PropertyModel[int]) -> None:
        while True:
            try:
                progress = acquisition.progress
                progress_value_model.value = int(100 * progress)
                await asyncio.sleep(0.25)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

    progress_task = asyncio.get_event_loop_policy().get_event_loop().create_task(update_progress(acquisition_state._acquisition_ex, progress_value_model))

    # start async acquire.
    acquisition_state._acquisition_ex.acquire_async(event_loop=document_controller.event_loop, on_completion=functools.partial(finish_grab_async, framed_data_stream, acquisition_state, scan_drift_logger, progress_task, progress_value_model, is_acquiring_model))


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
                                                                       _("Iterator Method"),
                                                                       acquisition_configuration,
                                                                       acquisition_preferences,
                                                                       "acquisition_method_component_id",
                                                                       "acquisition_method_components",
                                                                       PreferencesButtonHandler(document_controller))
        self.__acquisition_device_component = ComponentComboBoxHandler("acquisition-device-component",
                                                                       _("Detector"),
                                                                       acquisition_configuration,
                                                                       acquisition_preferences,
                                                                       "acquisition_device_component_id",
                                                                       "acquisition_device_components")
        # must delete the components if they are not added to another widget.
        self.__acquisition_method_component_to_delete: typing.Optional[ComponentComboBoxHandler] = self.__acquisition_method_component
        self.__acquisition_device_component_to_delete: typing.Optional[ComponentComboBoxHandler] = self.__acquisition_device_component

        # define whether this controller is in an error state
        self.is_error = False

        # define the progress value model, a simple bool 'is_acquiring' model, and a button text model that
        # updates according to whether acquire is running or not.
        self.progress_value_model = Model.PropertyModel[int](0)
        self.is_acquiring_model = Model.PropertyModel[bool](False)
        self.button_text_model = Model.StreamValueModel(Stream.MapStream(
            Stream.PropertyChangedEventStream(self.is_acquiring_model, "value"),
            lambda b: _("Acquire") if not b else _("Cancel")))

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

        # configure the button enabled. the selected_item_value_stream will give the selected acquisition frame
        # component. this may have a acquire_valid_value_stream property. the stream streamer listens to that stream
        # and sends its value as its own value when it changes. argh!
        self.button_enabled_model = Model.StreamValueModel(StreamStreamer(
            Stream.MapStream[typing.Any, typing.Any](self.__acquisition_device_component.selected_item_value_stream,
                             lambda c: getattr(c, "acquire_valid_value_stream", Stream.ConstantStream(True)))
        ))

        # define a progress task and acquisition. these are ephemeral and get closed after use in _acquire_data_stream.
        self.__progress_task: typing.Optional[asyncio.Task[None]] = None
        self.__acquisition_state = AcquisitionState()

        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            u.create_component_instance(identifier="acquisition-method-component"),
            u.create_spacing(8),
            u.create_divider(orientation="horizontal", height=8),
            u.create_component_instance(identifier="acquisition-device-component"),
            u.create_spacing(8),
            u.create_divider(orientation="horizontal", height=8),
            u.create_row(
                u.create_push_button(text="@binding(button_text_model.value)", enabled="@binding(button_enabled_model.value)", on_clicked="handle_button", width=80),
                u.create_progress_bar(value="@binding(progress_value_model.value)", width=180),
                u.create_stretch(),
                spacing=8,
            ),
            u.create_stretch(),
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
        if self.__acquisition_method_component_to_delete:
            self.__acquisition_method_component_to_delete.close()
            self.__acquisition_method_component_to_delete = typing.cast(typing.Any, None)
        if self.__acquisition_device_component_to_delete:
            self.__acquisition_device_component_to_delete.close()
            self.__acquisition_device_component_to_delete = typing.cast(typing.Any, None)
        super().close()

    def handle_button(self, widget: UserInterfaceModule.Widget) -> None:
        # handle acquire button, which can either start or stop acquisition.
        if self.__acquisition_state.is_active:
            self.__acquisition_state.abort_acquire()
        else:
            # starting acquisition means building the device data stream using the acquisition device component and
            # then wrapping the device data stream using the acquisition method component.
            build_result = self.__acquisition_device_component.current_item.build_acquisition_device_data_stream()
            try:
                apply_result = self.__acquisition_method_component.current_item.wrap_acquisition_device_data_stream(build_result.data_stream, build_result.device_map, build_result.channel_names)
                try:
                    # call the acquire data stream method to carry out the acquisition.
                    self._acquire_data_stream(apply_result.data_stream, apply_result.title_base, apply_result.channel_names, build_result.drift_tracker)
                finally:
                    apply_result.data_stream.remove_ref()
            finally:
                build_result.data_stream.remove_ref()

    def _acquire_data_stream(self,
                             data_stream: Acquisition.DataStream,
                             title_base: str,
                             channel_names: typing.Dict[Acquisition.Channel, str],
                             drift_tracker: typing.Optional[DriftTracker.DriftTracker]) -> None:
        """Perform acquisition of of the data stream."""
        _acquire_data_stream(data_stream,
                             self.document_controller,
                             self.__acquisition_state,
                             self.progress_value_model,
                             self.is_acquiring_model,
                             title_base,
                             channel_names,
                             drift_tracker)

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        # this is called to construct contained declarative component handlers within this handler.
        if component_id == "acquisition-method-component":
            self.__acquisition_method_component_to_delete = None
            return self.__acquisition_method_component
        if component_id == "acquisition-device-component":
            self.__acquisition_device_component_to_delete = None
            return self.__acquisition_device_component
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


class DeviceController(abc.ABC):
    @abc.abstractmethod
    def get_values(self, control_customization: AcquisitionPreferences.ControlCustomization, axis: typing.Optional[stem_controller.AxisType] = None) -> typing.Sequence[float]: ...

    @abc.abstractmethod
    def update_values(self, control_customization: AcquisitionPreferences.ControlCustomization, original_values: typing.Sequence[float], values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None: ...

    @abc.abstractmethod
    def set_values(self, control_customization: AcquisitionPreferences.ControlCustomization, values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None: ...


class STEMDeviceController(DeviceController):
    # NOTE: the STEM device controller treats all values as delta's from starting control value
    # This was decided in discussion with Nion engineers.

    def __init__(self) -> None:
        stem_controller_component = Registry.get_component('stem_controller')
        assert stem_controller_component
        self.stem_controller = typing.cast(stem_controller.STEMController, stem_controller_component)

    def get_values(self, control_customization: AcquisitionPreferences.ControlCustomization, axis: typing.Optional[stem_controller.AxisType] = None) -> typing.Sequence[float]:
        control_description = control_customization.control_description
        assert control_description
        if control_description.control_type == "1d":
            return [self.stem_controller.GetVal(control_customization.device_control_id)]
        elif control_description.control_type == "2d":
            assert axis is not None
            return self.stem_controller.GetVal2D(control_customization.device_control_id, axis=axis).as_tuple()
        raise ValueError()

    def update_values(self, control_customization: AcquisitionPreferences.ControlCustomization, original_values: typing.Sequence[float], values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        control_description = control_customization.control_description
        assert control_description
        if control_description.control_type == "1d":
            self.stem_controller.SetValAndConfirm(control_customization.device_control_id, original_values[0] + values[0], 1.0, 5000)
            time.sleep(control_customization.delay)
        elif control_description.control_type == "2d":
            assert axis is not None
            self.stem_controller.SetVal2DAndConfirm(control_customization.device_control_id,
                                                    Geometry.FloatPoint(y=original_values[0], x=original_values[1]) + Geometry.FloatPoint(y=values[0], x=values[1]),
                                                    1.0, 5000,
                                                    axis=axis)
            time.sleep(control_customization.delay)

    def set_values(self, control_customization: AcquisitionPreferences.ControlCustomization, values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        control_description = control_customization.control_description
        assert control_description
        if control_description.control_type == "1d":
            self.stem_controller.SetValAndConfirm(control_customization.device_control_id, values[0], 1.0, 5000)
            time.sleep(control_customization.delay)
        elif control_description.control_type == "2d":
            assert axis is not None
            self.stem_controller.SetVal2DAndConfirm(control_customization.device_control_id,
                                                    Geometry.FloatPoint(y=values[0], x=values[1]), 1.0, 5000,
                                                    axis=axis)
            time.sleep(control_customization.delay)


class CameraDeviceController(DeviceController):
    def __init__(self, camera_hardware_source: camera_base.CameraHardwareSource, camera_frame_parameters: camera_base.CameraFrameParameters):
        self.camera_hardware_source = camera_hardware_source
        self.camera_frame_parameters = camera_frame_parameters

    def get_values(self, control_customization: AcquisitionPreferences.ControlCustomization, axis: typing.Optional[stem_controller.AxisType] = None) -> typing.Sequence[float]:
        control_description = control_customization.control_description
        assert control_description
        if control_customization.control_id == "exposure":
            return [self.camera_frame_parameters.exposure_ms]
        raise ValueError()

    def update_values(self, control_customization: AcquisitionPreferences.ControlCustomization, original_values: typing.Sequence[float], values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        self.set_values(control_customization, values, axis)

    def set_values(self, control_customization: AcquisitionPreferences.ControlCustomization, values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        control_description = control_customization.control_description
        assert control_description
        if control_customization.control_id == "exposure":
            self.camera_frame_parameters.exposure_ms = values[0]


class ScanDeviceController(DeviceController):
    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, scan_frame_parameters: scan_base.ScanFrameParameters):
        self.scan_hardware_source = scan_hardware_source
        self.scan_frame_parameters = scan_frame_parameters

    def get_values(self, control_customization: AcquisitionPreferences.ControlCustomization, axis: typing.Optional[stem_controller.AxisType] = None) -> typing.Sequence[float]:
        control_description = control_customization.control_description
        assert control_description
        if control_customization.control_id == "field_of_view":
            return [self.scan_frame_parameters.fov_nm]
        raise ValueError()

    def update_values(self, control_customization: AcquisitionPreferences.ControlCustomization, original_values: typing.Sequence[float], values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        self.set_values(control_customization, values, axis)

    def set_values(self, control_customization: AcquisitionPreferences.ControlCustomization, values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        control_description = control_customization.control_description
        assert control_description
        if control_customization.control_id == "field_of_view":
            self.scan_frame_parameters.fov_nm = values[0]


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
                        u.create_line_edit(text="@binding(item.device_control_id)", width=180),
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
                        u.create_label(text="Control", width=180),
                        u.create_label(text="Delay", width=80),
                        u.create_stretch(),
                        spacing=8
                    ),
                    u.create_divider(orientation="horizontal"),
                    u.create_column(items="sorted_controls.items", item_component_id="control-component", spacing=8),
                    u.create_stretch(),
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
