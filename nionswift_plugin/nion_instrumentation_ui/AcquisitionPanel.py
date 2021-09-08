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
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import ApplicationData
from nion.swift.model import DataItem
from nion.swift.model import HardwareSource
from nion.swift.model import Schema
from nion.ui import Application
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


class HardwareSourceHandler(Observable.Observable):
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
        self.__listener = typing.cast(Event.EventListener, None)

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


class HardwareSourceChannelChooserHandler(Observable.Observable):
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
        self.__hardware_sources_list_changed_listener = None
        self.__hardware_source_changed_listener.close()
        self.__hardware_source_changed_listener = None
        self.__channel_model = typing.cast(Model.PropertyModel[str], None)
        self.__hardware_source_choice = typing.cast(HardwareSourceChoice.HardwareSourceChoice, None)

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
            if hardware_source.features.get("is_camera", False):
                if hardware_source.camera.camera_type == "ronchigram":
                    channel_descriptions = [hardware_source_channel_descriptions["ronchigram"]]
                elif hardware_source.camera.camera_type == "eels":
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
        self.__hardware_source_stream_changed(hardware_source_stream.value)

    def about_to_delete(self) -> None:
        if self.__frame_parameters_changed_listener:
            self.__frame_parameters_changed_listener.close()
            self.__frame_parameters_changed_listener = None
        self.__hardware_source_stream_listener.close()
        self.__hardware_source_stream_listener = None
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
            hardware_source = self.__hardware_source_stream.value
            if hardware_source:
                frame_parameters = hardware_source.get_frame_parameters(0)
                frame_parameters.exposure_ms = exposure_time_ms
                hardware_source.set_frame_parameters(0, frame_parameters)


class CameraDetailsHandler(Observable.Observable):
    """A declarative component handler for a row describing a camera device.

    The hardware_source_choice parameter is the associated hardware source choice.
    """

    def __init__(self, hardware_source_choice: HardwareSourceChoice.HardwareSourceChoice):
        super().__init__()

        # the exposure value stream gives the stream of exposure values from the hardware source choice
        self.exposure_value_stream = CameraExposureValueStream(HardwareSourceChoice.HardwareSourceChoiceStream(hardware_source_choice)).add_ref()
        # the exposure model converts the exposure value stream to a property model that supports binding.
        self.exposure_model = Model.StreamValueModel(self.exposure_value_stream)
        # the exposure value converter converts the exposure value to a string and back in the line edit.
        self.exposure_value_converter = Converter.PhysicalValueToStringConverter("ms", 1, "{:.1f}")

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
        self.__exposure_model_listener = typing.cast(Event.EventListener, None)
        self.exposure_model.close()
        self.exposure_model = typing.cast(Model.StreamValueModel, None)
        self.exposure_value_stream.remove_ref()
        self.exposure_value_stream = typing.cast(CameraExposureValueStream, None)

    def __exposure_changed(self, k: str) -> None:
        if k == "value":
            self.exposure_value_stream.exposure_time_ms = self.exposure_model.value if self.exposure_model.value else 0.0


@dataclasses.dataclass
class SynchronizedScanDescription:
    """Describes the parameters for a synchronized scan.

    context_text is a string explaining the current scan context.
    context_valid is True if the context is valid.
    scan_text is a string explaining the planned scan.
    scan_size is an IntSize of the scan size.
    drift_interval_lines is an int of how often to do drift correction.
    """
    context_text: str
    context_valid: bool
    scan_text: str
    scan_size: Geometry.IntSize
    drift_interval_lines: int


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
        self.__hardware_source_stream_listener = None
        self.__scan_width_changed_listener.close()
        self.__scan_width_changed_listener = None
        self.__scan_hardware_source_stream.remove_ref()
        self.__scan_hardware_source_stream = typing.cast(Stream.AbstractStream[HardwareSource.HardwareSource], None)
        self.__camera_hardware_source_stream.remove_ref()
        self.__camera_hardware_source_stream = typing.cast(Stream.AbstractStream[HardwareSource.HardwareSource], None)
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

            if scan_context.is_valid and scan_hardware_source.line_scan_enabled and scan_hardware_source.line_scan_vector:
                calibration = scan_context.calibration
                start = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[0])
                end = Geometry.FloatPoint.make(scan_hardware_source.line_scan_vector[1])
                length = int(Geometry.distance(start, end) * scan_context.size.height)
                max_dim = max(scan_context.size.width, scan_context.size.height)
                length_str = calibration.convert_to_calibrated_size_str(length, value_range=(0, max_dim), samples=max_dim)
                line_str = _("Line Scan")
                context_text = f"{line_str} {length_str}"
                scan_length = max(self.__scan_width_model.value or 0, 1)
                scan_text = f"{scan_length} px"
                scan_size = Geometry.IntSize(height=1, width=scan_length)
                drift_interval_lines = 0
                self.send_value(SynchronizedScanDescription(context_text, True, scan_text, scan_size, drift_interval_lines))
            elif scan_context.is_valid and scan_hardware_source.subscan_enabled and scan_hardware_source.subscan_region:
                calibration = scan_context.calibration
                width = scan_hardware_source.subscan_region.width * scan_context.size.width
                height = scan_hardware_source.subscan_region.height * scan_context.size.height
                width_str = calibration.convert_to_calibrated_size_str(width,
                                                                       value_range=(0, scan_context.size.width),
                                                                       samples=scan_context.size.width)
                height_str = calibration.convert_to_calibrated_size_str(height,
                                                                        value_range=(0, scan_context.size.height),
                                                                        samples=scan_context.size.height)
                rect_str = _("Subscan")
                context_text = f"{rect_str} {width_str} x {height_str}"
                scan_height = int(self.__scan_width_model.value * height / width)
                scan_text = f"{scan_width} x {scan_height}"
                scan_size = Geometry.IntSize(height=scan_height, width=scan_width)
                drift_lines = scan_hardware_source.calculate_drift_lines(scan_width, exposure_time)
                drift_interval_lines = drift_lines
                self.send_value(SynchronizedScanDescription(context_text, True, scan_text, scan_size, drift_interval_lines))
            elif scan_context.is_valid:
                calibration = scan_context.calibration
                width = scan_context.size.width
                height = scan_context.size.height
                width_str = calibration.convert_to_calibrated_size_str(width,
                                                                       value_range=(0, scan_context.size.width),
                                                                       samples=scan_context.size.width)
                height_str = calibration.convert_to_calibrated_size_str(height,
                                                                        value_range=(0, scan_context.size.height),
                                                                        samples=scan_context.size.height)
                data_str = _("Context Scan")
                context_text = f"{data_str} {width_str} x {height_str}"
                scan_height = int(self.__scan_width_model.value * height / width)
                scan_text = f"{scan_width} x {scan_height}"
                scan_size = Geometry.IntSize(height=scan_height, width=scan_width)
                drift_lines = scan_hardware_source.calculate_drift_lines(scan_width, exposure_time)
                drift_interval_lines = drift_lines
                self.send_value(SynchronizedScanDescription(context_text, True, scan_text, scan_size, drift_interval_lines))
            else:
                context_text = _("No scan context")
                scan_text = str()
                self.send_value(SynchronizedScanDescription(context_text, False, scan_text, Geometry.IntSize(), 0))


class ComponentHandler:
    def __init__(self, display_name: str):
        self.display_name = display_name

    def __str__(self) -> str:
        return self.display_name

    def close(self) -> None:
        pass


class ComboBoxHandler:
    """Combine a label and combo box; and facilitate storing the selected item using an item identifier."""

    def __init__(self, container: Observable.Observable, items_key: str, sort_key: ListModel.SortKeyCallable,
                 filter: typing.Optional[ListModel.Filter], id_getter: typing.Callable[[typing.Any], str],
                 selection_storage_model: Model.PropertyModel):
        self.sorted_items = ListModel.FilteredListModel(container=container, items_key=items_key)
        self.sorted_items.sort_key = sort_key
        if filter:
            self.sorted_items.filter = filter
        self.item_list = ListModel.ListPropertyModel(self.sorted_items)
        self.selected_index_model = Model.PropertyModel[int](0)
        self.selected_item_value_stream = Stream.ValueStream().add_ref()

        def update_selected_item(c: ListModel.ListPropertyModel, index_model: Model.PropertyModel[int],
                                 v: Stream.ValueStream, k: str) -> None:
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
        component_type_combo = u.create_combo_box(items_ref="@binding(item_list.value)",
                                                  current_index="@binding(selected_index_model.value)")
        self.ui_view = component_type_combo

    def close(self) -> None:
        self.__selected_component_index_listener.close()
        self.__selected_component_index_listener = typing.cast(Event.EventListener, None)
        self.selected_item_value_stream.remove_ref()
        self.selected_item_value_stream = typing.cast(Stream.ValueStream, None)
        self.selected_index_model.close()
        self.selected_index_model = typing.cast(Model.PropertyModel[int], None)
        self.item_list.close()
        self.item_list = typing.cast(ListModel.ListPropertyModel, None)
        self.sorted_items.close()
        self.sorted_items = typing.cast(ListModel.FilteredListModel, None)

    @property
    def current_item(self) -> typing.Any:
        index = self.selected_index_model.value or 0
        return self.sorted_items.items[index]


class ComponentComboBoxHandler:

    def __init__(self, component_base: str, title: str, configuration: AcquisitionConfiguration, component_id_key: str, components_key: str):
        self.__component_name = component_base
        self.__component_factory_name = f"{component_base}-factory"
        self.__components = ListModel.ListModel[ComponentHandler]()
        self.__selected_component_id_model = Model.PropertyChangedPropertyModel[str](configuration, component_id_key)

        component_map: typing.Dict[str, Schema.Entity] = dict()
        for component_entity in configuration._get_array_items(components_key):
            component_map[component_entity.entity_type.entity_id] = component_entity

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
            component = component_factory(component_entity)
            self.__components.append_item(component)

        # this gets closed by the declarative machinery
        self._combo_box_handler = ComboBoxHandler(self.__components, "items", operator.attrgetter("display_name"),
                                                  None, operator.attrgetter("component_id"),
                                                  self.__selected_component_id_model)

        # this is merely a reference and does not need to be closed
        self.selected_item_value_stream = self._combo_box_handler.selected_item_value_stream

        # TODO: listen for components being registered/unregistered

        u = Declarative.DeclarativeUI()
        component_type_row = u.create_row(
            u.create_label(text=title),
            u.create_component_instance(identifier="combo_box"),
            u.create_stretch(),
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
        self.__selected_component_id_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        self.__components.close()
        self.__components = typing.cast(ListModel.ListModel[ComponentHandler], None)

    def create_handler(self, component_id: str, container=None, item=None, **kwargs):
        if component_id == self.__component_name:
            return item
        if component_id == "combo_box":
            return self._combo_box_handler
        return None

    @property
    def current_item(self) -> typing.Any:
        return self._combo_box_handler.current_item


class AcquireHandler(ComponentHandler):

    def enclose(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> typing.Tuple[Acquisition.DataStream, str, typing.Dict[Acquisition.Channel, str]]:
        raise NotImplementedError()


class BasicAcquireHandler(AcquireHandler):
    component_id = "basic-acquire"

    def __init__(self, configuration: Schema.Entity):
        super().__init__(_("Basic Acquire"))
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_column(
            spacing=8
        )

    def enclose(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> typing.Tuple[Acquisition.DataStream, str, typing.Dict[Acquisition.Channel, str]]:
        return data_stream, str(), channel_names


class SequenceAcquireHandler(AcquireHandler):
    component_id = "sequence-acquire"

    def __init__(self, configuration: Schema.Entity):
        super().__init__(_("Sequence Acquire"))
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

    def enclose(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> typing.Tuple[Acquisition.DataStream, str, typing.Dict[Acquisition.Channel, str]]:
        width = max(1, self.configuration.count) if self.configuration.count else 1
        if width > 1:
            return Acquisition.SequenceDataStream(data_stream, width), _("Sequence"), channel_names
        else:
            return data_stream, _("Sequence"), channel_names


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

class SeriesControlHandler:

    def __init__(self, control_customization: AcquisitionPreferences.ControlCustomization, control_values: Schema.Entity, label: typing.Optional[str]):
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
        row_items.append(u.create_spacing(20))
        if label is not None:
            row_items.append(u.create_label(text=label, width=28))
        row_items.append(u.create_line_edit(text="@binding(control_values.count, converter=count_converter)", width=90))
        row_items.append(u.create_line_edit(text="@binding(control_values.start_value, converter=value_converter)", width=90))
        row_items.append(u.create_line_edit(text="@binding(control_values.step_value, converter=value_converter)", width=90))
        row_items.append(u.create_stretch())
        self.ui_view = u.create_row(*row_items, spacing=8)

    def get_control_info(self) -> typing.Tuple[float, float, int]:
        control_description = self.__control_customization.control_description
        assert control_description
        width = max(1, self.control_values.count) if self.control_values.count else 1
        start = (self.control_values.start_value or 0)
        step = (self.control_values.step_value or 0)
        return start * control_description.multiplier, step * control_description.multiplier, width


def get_control_values(configuration: Schema.Entity, control_values_list_key: str, control_customization: AcquisitionPreferences.ControlCustomization, value_index: typing.Optional[int] = None) -> Schema.Entity:
    """Extract the control values from the configuration."""
    control_id = control_customization.control_id
    m = dict()
    for c in configuration._get_array_items(control_values_list_key):
        m[c.control_id] = c
    control_values = m.get(control_id)
    if not control_values:
        control_description = control_customization.control_description
        assert control_description
        value = control_description.default_value
        control_values = ControlValuesSchema.create(None, {"control_id": control_id, "count": 1,
                                                           "start_value": value[value_index] if value_index is not None else value,
                                                           "step_value": 0.0})
        configuration._append_item(control_values_list_key, control_values)
    return control_values


class SeriesAcquireHandler(AcquireHandler):
    component_id = "series-acquire"

    def __init__(self, configuration: Schema.Entity):
        super().__init__(_("Series Acquire"))
        assert AcquisitionPreferences.acquisition_preferences
        self.configuration = configuration
        self.__control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        self.__selection_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "control_id")
        self._control_combo_box_handler = ComboBoxHandler(AcquisitionPreferences.acquisition_preferences,
                                                          "control_customizations",
                                                          operator.attrgetter("name"),
                                                          ListModel.PredicateFilter(lambda x: x.control_description.control_type == "1d"),
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
        self.__selection_storage_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        super().close()

    def create_handler(self, component_id: str, container=None, item=None, **kwargs):
        if component_id == "control_combo_box":
            return self._control_combo_box_handler
        if component_id == "series-control":
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_id = control_customization.control_id
            assert control_id not in self.__control_handlers
            control_values = get_control_values(self.configuration, "control_values_list", control_customization)
            self.__control_handlers[control_id] = SeriesControlHandler(control_customization, control_values, None)
            return self.__control_handlers[control_id]
        return None

    def enclose(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> typing.Tuple[Acquisition.DataStream, str, typing.Dict[Acquisition.Channel, str]]:
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_handler = self.__control_handlers.get(control_customization.control_id)
            if control_handler and data_stream:
                start, step, width = control_handler.get_control_info()
                if width > 1:
                    def action(control_customization: AcquisitionPreferences.ControlCustomization,
                               device_map: typing.Mapping[str, DeviceController],
                               starts: typing.Sequence[float], steps: typing.Sequence[float],
                               index: typing.Sequence[int]) -> None:
                        control_description = control_customization.control_description
                        assert control_description
                        device_controller = device_map.get(control_description.device_id)
                        if device_controller:
                            values = [start + step * i for start, step, i in zip(starts, steps, index)]
                            device_controller.set_values(control_customization, values)

                    action_fn = weak_partial(action, control_customization, device_map, [start], [step])
                    data_stream = Acquisition.SequenceDataStream(Acquisition.ActionDataStream(data_stream, action_fn), width)
        return data_stream, _("Series"), channel_names


class TableauAcquireHandler(AcquireHandler):
    component_id = "tableau-acquire"

    def __init__(self, configuration: Schema.Entity):
        super().__init__(_("Tableau Acquire"))
        assert AcquisitionPreferences.acquisition_preferences
        self.configuration = configuration
        self.__x_control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        self.__y_control_handlers: typing.Dict[str, SeriesControlHandler] = dict()
        self.__selection_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "control_id")
        self._control_combo_box_handler = ComboBoxHandler(AcquisitionPreferences.acquisition_preferences,
                                                          "control_customizations",
                                                          operator.attrgetter("name"),
                                                          ListModel.PredicateFilter(lambda x: x.control_description.control_type == "2d"),
                                                          operator.attrgetter("control_id"),
                                                          self.__selection_storage_model)
        self.__axis_storage_model = Model.PropertyChangedPropertyModel[str](self.configuration, "axis_id")
        stem_controller = Registry.get_component("stem_controller")
        assert stem_controller
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
        self.__selection_storage_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        super().close()

    def create_handler(self, component_id: str, container=None, item=None, **kwargs):
        if component_id == "control_combo_box":
            return self._control_combo_box_handler
        if component_id == "axis_combo_box":
            return self._axis_combo_box_handler
        if component_id == "y-control":
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_id = control_customization.control_id
            assert control_id not in self.__y_control_handlers
            control_values = get_control_values(self.configuration, "y_control_values_list", control_customization, 0)
            self.__y_control_handlers[control_id] = SeriesControlHandler(control_customization, control_values, "Y")
            return self.__y_control_handlers[control_id]
        if component_id == "x-control":
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            control_id = control_customization.control_id
            assert control_id not in self.__x_control_handlers
            control_values = get_control_values(self.configuration, "x_control_values_list", control_customization, 1)
            self.__x_control_handlers[control_id] = SeriesControlHandler(control_customization, control_values, "X")
            return self.__x_control_handlers[control_id]
        return None

    def enclose(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> typing.Tuple[Acquisition.DataStream, str, typing.Dict[Acquisition.Channel, str]]:
        item = self._control_combo_box_handler.selected_item_value_stream.value
        if item:
            control_customization = typing.cast(AcquisitionPreferences.ControlCustomization, item)
            x_control_handler = self.__x_control_handlers.get(control_customization.control_id)
            y_control_handler = self.__y_control_handlers.get(control_customization.control_id)
            if x_control_handler and y_control_handler and data_stream:
                axis_id = self.__axis_storage_model.value
                y_start, y_step, height = y_control_handler.get_control_info()
                x_start, x_step, width = x_control_handler.get_control_info()
                if width > 1 or height > 1:
                    def action(control_customization: AcquisitionPreferences.ControlCustomization,
                               device_map: typing.Mapping[str, DeviceController],
                               starts: typing.Sequence[float], steps: typing.Sequence[float],
                               axis_id: str, index: typing.Sequence[int]) -> None:
                        control_description = control_customization.control_description
                        assert control_description
                        device_controller = device_map.get(control_description.device_id)
                        if device_controller:
                            axis: typing.Optional[stem_controller.AxisType] = ("x", "y")
                            for axis_description in typing.cast(STEMDeviceController, device_map["stem"]).stem_controller.axis_descriptions:
                                if axis_id == axis_description.axis_id:
                                    axis = axis_description.axis_type
                            values = [start + step * i for start, step, i in zip(starts, steps, index)]
                            device_controller.set_values(control_customization, values, axis)

                    action_fn = weak_partial(action, control_customization, device_map, [y_start, x_start], [y_step, x_step], axis_id)
                    data_stream = Acquisition.CollectedDataStream(Acquisition.ActionDataStream(data_stream, action_fn),
                                                                 (height, width), (Calibration.Calibration(), Calibration.Calibration()))
        return data_stream, _("Tableau"), channel_names


class MultiAcquireEntryHandler:

    def __init__(self, container: typing.Any, item: Schema.Entity):
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


class MultipleAcquireHandler(AcquireHandler):
    component_id = "multiple-acquire"

    def __init__(self, configuration: Schema.Entity):
        super().__init__(_("Multiple Acquire"))
        self.configuration = configuration
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

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs):
        if component_id == "section":
            assert container is not None
            assert item is not None
            return MultiAcquireEntryHandler(container, item)
        return None

    def add(self, widget: UserInterfaceModule.Widget) -> None:
        self.configuration._append_item("sections", MultipleAcquireEntrySchema.create(None, {"offset": 0.0, "exposure": 0.001, "count": 2}))

    def remove(self, widget: UserInterfaceModule.Widget) -> None:
        self.configuration._remove_item("sections", self.configuration._get_array_item("sections", -1))

    def enclose(self, data_stream: Acquisition.DataStream, device_map: typing.Mapping[str, DeviceController], channel_names: typing.Dict[Acquisition.Channel, str]) -> typing.Tuple[Acquisition.DataStream, str, typing.Dict[Acquisition.Channel, str]]:
        assert AcquisitionPreferences.acquisition_preferences
        data_streams: typing.List[Acquisition.DataStream] = list()
        for item in self.configuration.sections:
            multi_acquire_entry = typing.cast(Schema.Entity, item)
            control_customizations_map = {control_customization.control_id: control_customization for
                                          control_customization in AcquisitionPreferences.acquisition_preferences.control_customizations}
            stem_value_controller = device_map.get("stem")
            camera_value_controller = device_map.get("camera")
            control_customization_energy_offset = control_customizations_map["energy_offset"]
            control_customization_exposure = control_customizations_map["exposure"]
            assert control_customization_energy_offset
            assert control_customization_exposure
            control_description_energy_offset = control_customization_energy_offset.control_description
            control_description_exposure = control_customization_exposure.control_description
            assert control_description_energy_offset
            assert control_description_exposure

            def action(offset_value: float, exposure_value: float, index: typing.Sequence[int]) -> None:
                if stem_value_controller:
                    stem_value_controller.set_values(control_customization_energy_offset, [offset_value])
                if camera_value_controller:
                    camera_value_controller.set_values(control_customization_exposure, [exposure_value])

            action_fn = functools.partial(action,
                                          multi_acquire_entry.offset * control_description_energy_offset.multiplier,
                                          multi_acquire_entry.exposure * control_description_exposure.multiplier)
            data_streams.append(Acquisition.SequenceDataStream(Acquisition.ActionDataStream(data_stream, action_fn), max(1, multi_acquire_entry.count)))

        sequential_data_stream = Acquisition.SequentialDataStream(data_streams)
        for index, channel in enumerate(sequential_data_stream.channels):
            channel_names[channel] = " ".join((f"{str(index + 1)} / {str(len(sequential_data_stream.channels))}", channel_names[Acquisition.Channel(*channel.segments[1:])]))
        return sequential_data_stream, _("Multiple"), channel_names


Registry.register_component(BasicAcquireHandler, {"acquisition-method-component-factory"})
Registry.register_component(SequenceAcquireHandler, {"acquisition-method-component-factory"})
Registry.register_component(SeriesAcquireHandler, {"acquisition-method-component-factory"})
Registry.register_component(TableauAcquireHandler, {"acquisition-method-component-factory"})
Registry.register_component(MultipleAcquireHandler, {"acquisition-method-component-factory"})


class AcquisitionComponentHandler(ComponentHandler):

    def _handle_acquire(self) -> typing.Tuple[Acquisition.DataStream, typing.Dict[Acquisition.Channel, str], typing.Optional[scan_base.DriftTracker], typing.Mapping[str, typing.Any]]:
        raise NotImplementedError()


class SynchronizedScanAcquisitionComponentHandler(AcquisitionComponentHandler):
    component_id = "synchronized-scan"
    display_name = _("Synchronized Scan")

    def __init__(self, configuration: Schema.Entity):
        super().__init__(SynchronizedScanAcquisitionComponentHandler.display_name)
        self.__camera_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, "camera_device_id")
        self.__camera_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__camera_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_camera"))
        self.__camera_hardware_source_channel_model = Model.PropertyChangedPropertyModel[str](configuration, "camera_channel_id")
        self.__scan_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, "scan_device_id")
        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__scan_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_scanning"))
        self.scan_width = Model.PropertyChangedPropertyModel[int](configuration, "scan_width")
        self.__scan_context_description_value_stream = SynchronizedScanDescriptionValueStream(
            HardwareSourceChoice.HardwareSourceChoiceStream(self.__camera_hardware_source_choice),
            HardwareSourceChoice.HardwareSourceChoiceStream(self.__scan_hardware_source_choice),
            self.scan_width,
            asyncio.get_event_loop()).add_ref()
        self.scan_context_value_model = Model.StreamValueModel(Stream.MapStream(
            self.__scan_context_description_value_stream,
            lambda x: x.context_text
        ))
        self.scan_value_model = Model.StreamValueModel(Stream.MapStream(
            self.__scan_context_description_value_stream,
            lambda x: x.scan_text
        ))
        self.scan_width_converter = Converter.IntegerToStringConverter()
        self.acquire_valid_value_stream = Stream.MapStream(self.__scan_context_description_value_stream,
                                                           lambda x: x.context_valid).add_ref()

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
        self.acquire_valid_value_stream = typing.cast(Stream.MapStream, None)
        self.scan_value_model.close()
        self.scan_value_model = typing.cast(Model.StreamValueModel, None)
        self.__scan_context_description_value_stream.remove_ref()
        self.__scan_context_description_value_stream = None
        self.scan_context_value_model.close()
        self.scan_context_value_model = typing.cast(Model.StreamValueModel, None)
        self.__camera_hardware_source_choice.close()
        self.__camera_hardware_source_choice = typing.cast(HardwareSourceChoice.HardwareSourceChoice, None)
        self.__camera_hardware_source_choice_model.close()
        self.__camera_hardware_source_choice_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        self.__camera_hardware_source_channel_model.close()
        self.__camera_hardware_source_channel_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        self.__scan_hardware_source_choice.close()
        self.__scan_hardware_source_choice = typing.cast(HardwareSourceChoice.HardwareSourceChoice, None)
        self.__scan_hardware_source_choice_model.close()
        self.__scan_hardware_source_choice_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        self.scan_width.close()
        self.scan_width = typing.cast(Model.PropertyChangedPropertyModel[int], None)
        super().close()

    def create_handler(self, component_id: str, **kwargs):
        if component_id == "acquisition-device-component":
            return HardwareSourceHandler(self.__camera_hardware_source_choice)
        elif component_id == "acquisition-device-component-output":
            return HardwareSourceChannelChooserHandler(self.__camera_hardware_source_choice, self.__camera_hardware_source_channel_model)
        elif component_id == "acquisition-device-component-details":
            return CameraDetailsHandler(self.__camera_hardware_source_choice)
        elif component_id == "scan-component":
            return HardwareSourceHandler(self.__scan_hardware_source_choice)
        return None

    def _handle_acquire(self) -> typing.Tuple[Acquisition.DataStream, typing.Dict[Acquisition.Channel, str], typing.Optional[scan_base.DriftTracker], typing.Mapping[str, typing.Any]]:
        camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, self.__camera_hardware_source_choice.hardware_source)
        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source_choice.hardware_source)
        if self.__camera_hardware_source_channel_model.value in hardware_source_channel_descriptions:
            camera_channel_description = hardware_source_channel_descriptions[self.__camera_hardware_source_channel_model.value]
        else:
            camera_channel_description = hardware_source_channel_descriptions["image"]
        scan_context_description = self.__scan_context_description_value_stream.value

        assert camera_hardware_source is not None
        assert scan_hardware_source is not None
        assert camera_channel_description is not None
        assert scan_context_description is not None

        camera_frame_parameters = camera_hardware_source.get_frame_parameters(0)
        if camera_channel_description.processing_id:
            camera_frame_parameters["processing"] = camera_channel_description.processing_id

        scan_count = 1
        scan_size = scan_context_description.scan_size
        scan_frame_parameters = scan_hardware_source.get_frame_parameters(2)
        scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters, scan_size)
        scan_frame_parameters["scan_id"] = str(uuid.uuid4())

        drift_correction_behavior: typing.Optional[DriftTracker.DriftCorrectionBehavior] = None
        section_height: typing.Optional[int] = None
        if scan_context_description.drift_interval_lines > 0:
            drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters)
            section_height = scan_context_description.drift_interval_lines

        synchronized_scan_data_stream = scan_base.make_synchronized_scan_data_stream(
            scan_hardware_source=scan_hardware_source,
            scan_frame_parameters=scan_frame_parameters,
            camera_hardware_source=camera_hardware_source,
            camera_frame_parameters=camera_frame_parameters,
            scan_behavior=drift_correction_behavior,
            section_height=section_height,
            scan_count=scan_count,
            include_raw=True,
            include_summed=False
        )

        op = _("Synchronized")
        channel_names: typing.Dict[Acquisition.Channel, str] = dict()
        for c in scan_hardware_source.get_enabled_channels():
            channel_state = scan_hardware_source.get_channel_state(c)
            channel_index_segment = str(scan_hardware_source.get_channel_index(channel_state.channel_id))
            channel_names[Acquisition.Channel(scan_hardware_source.hardware_source_id, channel_index_segment)] = f"{op} {channel_state.name}"
        channel_names[Acquisition.Channel(camera_hardware_source.hardware_source_id)] = f"{op} {camera_hardware_source.get_signal_name(camera_frame_parameters)}"

        drift_tracker = scan_hardware_source.drift_tracker

        device_map: typing.Dict[str, DeviceController] = dict()
        device_map["stem"] = STEMDeviceController()
        device_map["camera"] = CameraDeviceController(camera_hardware_source, camera_frame_parameters)
        device_map["magnification"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)
        device_map["scan"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)

        return synchronized_scan_data_stream, channel_names, drift_tracker, device_map


class CameraFrameDataStream(Acquisition.DataStream):
    # unsynchronized camera frames
    # TODO: use sequence acquisition if there are no "in between" actions

    def __init__(self, camera_hardware_source: camera_base.CameraHardwareSource, frame_parameters: camera_base.CameraFrameParameters):
        super().__init__()
        self.__hardware_source = camera_hardware_source
        self.__frame_parameters = frame_parameters
        self.__record_task = typing.cast(scan_base.RecordTask, None)
        self.__record_count = 0
        self.__frame_shape = camera_hardware_source.get_expected_dimensions(frame_parameters.binning)
        self.__channel = Acquisition.Channel(self.__hardware_source.hardware_source_id)

    def about_to_delete(self) -> None:
        if self.__record_task:
            self.__record_task = typing.cast(scan_base.RecordTask, None)
        super().about_to_delete()

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return (self.__channel,)

    def get_info(self, channel: Acquisition.Channel) -> Acquisition.DataStreamInfo:
        data_shape = tuple(self.__hardware_source.get_expected_dimensions(self.__frame_parameters.binning))
        data_metadata = DataAndMetadata.DataMetadata((data_shape, numpy.float32))
        return Acquisition.DataStreamInfo(data_metadata, self.__frame_parameters.exposure_ms / 1000)

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs) -> None:
        self.__hardware_source.abort_playing(sync_timeout=5.0)

    def _start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        self.__record_task = scan_base.RecordTask(self.__hardware_source, self.__frame_parameters)
        self.__record_count = numpy.product(stream_args.shape, dtype=numpy.uint64)

    def _finish_stream(self) -> None:
        if self.__record_task:
            self.__record_task.grab()  # ensure grab is finished
            self.__record_task = typing.cast(scan_base.RecordTask, None)

    def _abort_stream(self) -> None:
        self.__hardware_source.abort_recording()

    def _send_next(self) -> None:
        if self.__record_task.is_finished:
            # data metadata describes the data being sent from this stream: shape, data type, and descriptor
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape))
            data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape, numpy.float32), data_descriptor=data_descriptor)
            source_data_slice: typing.Tuple[slice, ...] = (slice(0, self.__frame_shape[0]), slice(None))
            state = Acquisition.DataStreamStateEnum.COMPLETE
            data = self.__record_task.grab()[0].data
            data_stream_event = Acquisition.DataStreamEventArgs(self, self.__channel, data_metadata, data, None,
                                                                source_data_slice, state)
            self.fire_data_available(data_stream_event)
            self._sequence_next(self.__channel)
            self.__record_count -= 1
            if self.__record_count > 0:
                self.__record_task = scan_base.RecordTask(self.__hardware_source, self.__frame_parameters)


class CameraAcquisitionComponentHandler(AcquisitionComponentHandler):
    component_id = "camera"
    display_name = _("Camera")

    def __init__(self, configuration: Schema.Entity):
        super().__init__(CameraAcquisitionComponentHandler.display_name)
        self.__camera_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, "camera_device_id")
        self.__camera_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__camera_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_camera"))
        self.__camera_hardware_source_channel_model = Model.PropertyChangedPropertyModel[str](configuration, "camera_channel_id")

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
        self.__camera_hardware_source_choice = typing.cast(HardwareSourceChoice.HardwareSourceChoice, None)
        self.__camera_hardware_source_choice_model.close()
        self.__camera_hardware_source_choice_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        self.__camera_hardware_source_channel_model.close()
        self.__camera_hardware_source_channel_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs):
        if component_id == "acquisition-device-component":
            return HardwareSourceHandler(self.__camera_hardware_source_choice)
        elif component_id == "acquisition-device-component-output":
            return HardwareSourceChannelChooserHandler(self.__camera_hardware_source_choice, self.__camera_hardware_source_channel_model)
        elif component_id == "acquisition-device-component-details":
            return CameraDetailsHandler(self.__camera_hardware_source_choice)
        return None

    def _handle_acquire(self) -> typing.Tuple[Acquisition.DataStream, typing.Dict[Acquisition.Channel, str], typing.Optional[scan_base.DriftTracker], typing.Mapping[str, typing.Any]]:
        camera_hardware_source = typing.cast(camera_base.CameraHardwareSource, self.__camera_hardware_source_choice.hardware_source)
        if self.__camera_hardware_source_channel_model.value in hardware_source_channel_descriptions:
            camera_channel_description = hardware_source_channel_descriptions[self.__camera_hardware_source_channel_model.value]
        else:
            camera_channel_description = hardware_source_channel_descriptions["image"]

        assert camera_hardware_source is not None
        assert camera_channel_description is not None

        camera_frame_parameters = camera_hardware_source.get_frame_parameters(0)
        if camera_channel_description.processing_id:
            camera_frame_parameters["processing"] = camera_channel_description.processing_id

        instrument_metadata: typing.Dict[str, typing.Any] = dict()
        stem_controller = Registry.get_component('stem_controller')
        assert stem_controller
        scan_base.update_instrument_properties(instrument_metadata, stem_controller, None)

        camera_data_stream = CameraFrameDataStream(camera_hardware_source, camera_frame_parameters)
        processed_camera_data_stream: Acquisition.DataStream = camera_data_stream
        if camera_frame_parameters.get("processing", None) == "sum_project":
            processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream,
                                                                        operator=Acquisition.SumOperator(axis=0))
        elif camera_frame_parameters.get("processing", None) == "sum_masked":
            active_masks = typing.cast(camera_base.CameraFrameParameters, camera_frame_parameters).active_masks
            if active_masks:
                operator = Acquisition.StackedDataStreamOperator(
                    [Acquisition.MaskedSumOperator(active_mask) for active_mask in active_masks])
                processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream,
                                                                            operator=operator)
            else:
                operator = Acquisition.StackedDataStreamOperator([Acquisition.SumOperator()])
                processed_camera_data_stream = Acquisition.FramedDataStream(processed_camera_data_stream,
                                                                            operator=operator)

        channel_names: typing.Dict[Acquisition.Channel, str] = dict()
        channel_names[Acquisition.Channel(camera_hardware_source.hardware_source_id)] = camera_hardware_source.get_signal_name(camera_frame_parameters)

        device_map: typing.Dict[str, DeviceController] = dict()
        device_map["stem"] = STEMDeviceController()
        device_map["camera"] = CameraDeviceController(camera_hardware_source, camera_frame_parameters)

        return processed_camera_data_stream, channel_names, None, device_map


class ScanAcquisitionComponentHandler(AcquisitionComponentHandler):
    component_id = "scan"
    display_name = _("Scan")

    def __init__(self, configuration: Schema.Entity):
        super().__init__(ScanAcquisitionComponentHandler.display_name)
        self.__scan_hardware_source_choice_model = Model.PropertyChangedPropertyModel[str](configuration, "scan_device_id")
        self.__scan_hardware_source_choice = HardwareSourceChoice.HardwareSourceChoice(self.__scan_hardware_source_choice_model, lambda hardware_source: hardware_source.features.get("is_scanning"))
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
        self.__scan_hardware_source_choice = typing.cast(HardwareSourceChoice.HardwareSourceChoice, None)
        self.__scan_hardware_source_choice_model.close()
        self.__scan_hardware_source_choice_model = typing.cast(Model.PropertyChangedPropertyModel[str], None)
        super().close()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs):
        if component_id == "scan-component":
            return HardwareSourceHandler(self.__scan_hardware_source_choice)
        return None

    def _handle_acquire(self) -> typing.Tuple[Acquisition.DataStream, typing.Dict[Acquisition.Channel, str], typing.Optional[scan_base.DriftTracker], typing.Mapping[str, typing.Any]]:
        scan_hardware_source = typing.cast(scan_base.ScanHardwareSource, self.__scan_hardware_source_choice.hardware_source)

        assert scan_hardware_source is not None

        scan_uuid = uuid.uuid4()
        scan_frame_parameters = scan_hardware_source.get_frame_parameters(2)
        scan_hardware_source.apply_scan_context_subscan(scan_frame_parameters)
        scan_frame_parameters["scan_id"] = str(scan_uuid)

        scan_metadata: typing.Dict[str, typing.Any] = dict()
        scan_base.update_scan_metadata(scan_metadata, scan_hardware_source.hardware_source_id,
                                       scan_hardware_source.display_name, scan_frame_parameters, scan_uuid, dict())

        instrument_metadata: typing.Dict[str, typing.Any] = dict()
        scan_base.update_instrument_properties(instrument_metadata, scan_hardware_source.stem_controller,
                                               scan_hardware_source.scan_device)

        scan_data_stream = scan_base.ScanFrameDataStream(scan_hardware_source, scan_frame_parameters)

        scan_size = scan_data_stream.scan_size
        section_height = scan_size.height
        section_count = (scan_size.height + section_height - 1) // section_height
        slice_list: typing.List[typing.Tuple[slice, slice]] = list()
        for section in range(section_count):
            start = section * section_height
            stop = min(start + section_height, scan_size.height)
            slice_list.append((slice(start, stop), slice(0, scan_size.width)))
        collector: Acquisition.DataStream = Acquisition.CollectedDataStream(scan_data_stream, tuple(scan_size),
                                                                            scan_frame_parameters.get_scan_calibrations(),
                                                                            slice_list)

        channel_names: typing.Dict[Acquisition.Channel, str] = dict()
        for c in scan_hardware_source.get_enabled_channels():
            channel_state = scan_hardware_source.get_channel_state(c)
            channel_index_segment = str(scan_hardware_source.get_channel_index(channel_state.channel_id))
            channel_names[Acquisition.Channel(scan_hardware_source.hardware_source_id, channel_index_segment)] = channel_state.name

        device_map: typing.Dict[str, DeviceController] = dict()
        device_map["stem"] = STEMDeviceController()
        device_map["magnification"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)
        device_map["scan"] = ScanDeviceController(scan_hardware_source, scan_frame_parameters)

        return collector, channel_names, None, device_map


Registry.register_component(SynchronizedScanAcquisitionComponentHandler, {"acquisition-device-component-factory"})
Registry.register_component(CameraAcquisitionComponentHandler, {"acquisition-device-component-factory"})
Registry.register_component(ScanAcquisitionComponentHandler, {"acquisition-device-component-factory"})

AcquisitionDeviceComponentSchema = Schema.entity("acquisition_device_component", None, None, {
})

# SynchronizedScanAcquisitionComponentHandler
Schema.entity("acquisition_device_component_synchronized_scan", AcquisitionDeviceComponentSchema, None, {
    "camera_device_id": Schema.prop(Schema.STRING),
    "camera_channel_id": Schema.prop(Schema.STRING),
    "scan_device_id": Schema.prop(Schema.STRING),
    "scan_width": Schema.prop(Schema.INT, default=32),
})

# ScanAcquisitionComponentHandler
Schema.entity("acquisition_device_component_scan", AcquisitionDeviceComponentSchema, None, {
    "scan_device_id": Schema.prop(Schema.STRING),
})

# CameraAcquisitionComponentHandler
Schema.entity("acquisition_device_component_camera", AcquisitionDeviceComponentSchema, None, {
    "camera_device_id": Schema.prop(Schema.STRING),
    "camera_channel_id": Schema.prop(Schema.STRING),
})

AcquisitionMethodSchema = Schema.entity("acquisition_method_component", None, None, {
})

# BasicAcquireHandler
Schema.entity("acquisition_method_component_basic_acquire", AcquisitionMethodSchema, None, {
})

# SequenceAcquireHandler
Schema.entity("acquisition_method_component_sequence_acquire", AcquisitionMethodSchema, None, {
    "count": Schema.prop(Schema.INT, default=1),
})

ControlValuesSchema = Schema.entity("control_values", None, None, {
    "control_id": Schema.prop(Schema.STRING),
    "count": Schema.prop(Schema.INT),
    "start_value": Schema.prop(Schema.FLOAT),
    "step_value": Schema.prop(Schema.FLOAT),
})

# SeriesAcquireHandler
Schema.entity("acquisition_method_component_series_acquire", AcquisitionMethodSchema, None, {
    "control_id": Schema.prop(Schema.STRING),
    "control_values_list": Schema.array(Schema.component(ControlValuesSchema))
})

# TableauAcquireHandler
Schema.entity("acquisition_method_component_tableau_acquire", AcquisitionMethodSchema, None, {
    "control_id": Schema.prop(Schema.STRING),
    "axis_id": Schema.prop(Schema.STRING),
    "x_control_values_list": Schema.array(Schema.component(ControlValuesSchema)),
    "y_control_values_list": Schema.array(Schema.component(ControlValuesSchema)),
})

# MultipleAcquireHandler
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
    def __init__(self, file_path: pathlib.Path):
        super().__init__(AcquisitionConfigurationSchema)
        self.file_path = file_path
        self.__app_data = ApplicationData.ApplicationData(file_path)
        self.read_from_dict(self.__app_data.get_data_dict())
        field = Schema.ComponentField(None, self.entity_type.entity_id)
        field.set_field_value(None, self)
        self.__logger = AcquisitionPreferences.DictRecorderLogger(field, typing.cast(AcquisitionPreferences.DictRecorderLoggerDictInterface, self.__app_data))
        self.__recorder = Recorder.Recorder(self, None, self.__logger)

    def close(self) -> None:
        self.__recorder.close()
        self.__recorder = typing.cast(Recorder.Recorder, None)
        super().close()

    def _create(self, context: typing.Optional[Schema.EntityContext]) -> Schema.Entity:
        entity = self.__class__(self.file_path)
        if context:
            entity._set_entity_context(context)
        return entity


acquisition_configuration: typing.Optional[AcquisitionConfiguration] = None


class AcquisitionController:

    def __init__(self, document_controller: DocumentController.DocumentController):
        self.document_controller = document_controller
        assert acquisition_configuration

        # these get closed by the declarative machinery
        self.__acquisition_method_component = ComponentComboBoxHandler("acquisition-method-component",
                                                                       _("Acquisition Method"),
                                                                       acquisition_configuration,
                                                                       "acquisition_method_component_id",
                                                                       "acquisition_method_components")
        self.__acquisition_device_component = ComponentComboBoxHandler("acquisition-device-component",
                                                                       _("Acquisition Device"),
                                                                       acquisition_configuration,
                                                                       "acquisition_device_component_id",
                                                                       "acquisition_device_components")

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
        self.progress_value_model = Model.PropertyModel[int](0)
        self.is_acquiring_model = Model.PropertyModel[bool](False)
        self.button_text_model = Model.StreamValueModel(Stream.MapStream(
            Stream.PropertyChangedEventStream(self.is_acquiring_model, "value"),
            lambda b: _("Acquire") if not b else _("Cancel")))

        class StreamStreamer(Stream.ValueStream):
            def __init__(self, streams_stream: Stream.AbstractStream[Stream.AbstractStream]):
                super().__init__()
                self.__streams_stream = streams_stream.add_ref()
                self.__sub_stream_listener: typing.Optional[Event.EventListener] = None
                self.__sub_stream: typing.Optional[Stream.AbstractStream] = None
                self.__listener = self.__streams_stream.value_stream.listen(weak_partial(StreamStreamer.__attach_stream, self))
                self.__attach_stream(self.__streams_stream.value)

            def close(self) -> None:
                self.__streams_stream.remove_ref()
                self.__listener.close()
                self.__listener = typing.cast(Event.EventListener, None)
                if self.__sub_stream_listener:
                    self.__sub_stream_listener.close()
                    self.__sub_stream_listener = None
                if self.__sub_stream:
                    self.__sub_stream.remove_ref()
                    self.__sub_stream = None

            def __attach_stream(self, value_stream: typing.Optional[Stream.AbstractStream]) -> None:
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
            Stream.MapStream(self.__acquisition_device_component.selected_item_value_stream,
                             lambda c: getattr(c, "acquire_valid_value_stream", Stream.ConstantStream(True)))
        ))
        # these get closed after use in _acquire_data_stream
        self.__progress_task = typing.cast(asyncio.Task, None)
        self.__acquisition = typing.cast(Acquisition.Acquisition, None)

    def close(self) -> None:
        self.button_enabled_model.close()
        self.button_enabled_model = typing.cast(Model.StreamValueModel, None)
        self.is_acquiring_model.close()
        self.is_acquiring_model = typing.cast(Model.PropertyModel[bool], None)
        self.progress_value_model.close()
        self.progress_value_model = typing.cast(Model.PropertyModel[int], None)
        self.button_text_model.close()
        self.button_text_model = typing.cast(Model.StreamValueModel, None)

    def handle_button(self, widget: UserInterfaceModule.Widget) -> None:
        if self.__acquisition:
            self.__acquisition.abort_acquire()
        else:
            data_stream, channel_names, drift_tracker, device_map = self.__acquisition_device_component.current_item._handle_acquire()
            data_stream, title_base, channel_names = self.__acquisition_method_component.current_item.enclose(data_stream, device_map, channel_names)
            self._acquire_data_stream(data_stream, title_base, channel_names, drift_tracker)

    def _acquire_data_stream(self,
                             data_stream: Acquisition.DataStream,
                             title_base: str,
                             channel_names: typing.Dict[Acquisition.Channel, str],
                             drift_tracker: typing.Optional[scan_base.DriftTracker]):
        def display_data_item(document_controller: DocumentController.DocumentController, data_item: DataItem.DataItem) -> None:
            Facade.DocumentWindow(document_controller).display_data_item(Facade.DataItem(data_item))

        data_item_data_channel = DataChannel.DataItemDataChannel(self.document_controller.document_model, title_base, channel_names)
        data_item_data_channel.on_display_data_item = weak_partial(display_data_item, self.document_controller)
        event_loop = self.document_controller.event_loop
        self.__data_stream = Acquisition.FramedDataStream(data_stream, data_channel=data_item_data_channel).add_ref()
        self.__acquisition = Acquisition.Acquisition(self.__data_stream)
        self.__scan_drift_logger = DriftTracker.DriftLogger(self.document_controller.document_model, drift_tracker, event_loop) if drift_tracker else None

        def finish_grab_async():
            self.__acquisition.close()
            self.__acquisition = typing.cast(Acquisition.Acquisition, None)
            self.__data_stream.remove_ref()
            self.__data_stream = typing.cast(Acquisition.FramedDataStream, None)
            if self.__scan_drift_logger:
                self.__scan_drift_logger.close()
                self.__scan_drift_logger = typing.cast(DriftTracker.DriftLogger, None)
            self.is_acquiring_model.value = False
            self.__progress_task.cancel()
            self.__progress_task = typing.cast(asyncio.Task, None)
            self.progress_value_model.value = 100

        self.is_acquiring_model.value = True
        if not self.__progress_task:
            async def update_progress():
                while True:
                    self.progress_value_model.value = int(100 * self.__acquisition.progress)
                    await asyncio.sleep(0.25)

            self.__progress_task = asyncio.get_event_loop().create_task(update_progress())
        self.__acquisition.acquire_async(event_loop=event_loop, on_completion=finish_grab_async)

    def create_handler(self, component_id: str, container=None, item=None, **kwargs):
        if component_id == "acquisition-device-component":
            return self.__acquisition_device_component
        elif component_id == "acquisition-method-component":
            return self.__acquisition_method_component
        return None


class AcquisitionPanel(Panel.Panel):
    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: dict):
        super().__init__(document_controller, panel_id, "acquisition-panel")
        self.widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, AcquisitionController(document_controller))


class DeviceController(abc.ABC):
    @abc.abstractmethod
    def set_values(self, control_customization: AcquisitionPreferences.ControlCustomization, values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None: ...


class STEMDeviceController(DeviceController):
    def __init__(self):
        stem_controller_component = Registry.get_component('stem_controller')
        assert stem_controller_component
        self.stem_controller = typing.cast(stem_controller.STEMController, stem_controller_component)

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

    def set_values(self, control_customization: AcquisitionPreferences.ControlCustomization, values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        control_description = control_customization.control_description
        assert control_description
        if control_customization.control_id == "exposure":
            self.camera_frame_parameters.exposure_ms = values[0]


class ScanDeviceController(DeviceController):
    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, scan_frame_parameters: scan_base.ScanFrameParameters):
        self.scan_hardware_source = scan_hardware_source
        self.scan_frame_parameters = scan_frame_parameters

    def set_values(self, control_customization: AcquisitionPreferences.ControlCustomization, values: typing.Sequence[float], axis: typing.Optional[stem_controller.AxisType] = None) -> None:
        control_description = control_customization.control_description
        assert control_description
        if control_customization.control_id == "field_of_view":
            self.scan_frame_parameters.fov_nm = values[0]


def handle_application_changed(application: typing.Optional[Application.BaseApplication]):
    global acquisition_configuration
    if application:
        file_path = application.ui.get_configuration_location() / pathlib.Path("nion_acquisition_preferences.json")
        logging.info("Acquisition preferences: " + str(file_path))
        AcquisitionPreferences.init_acquisition_preferences(file_path)
        file_path = application.ui.get_configuration_location() / pathlib.Path("nion_acquisition_configuration.json")
        logging.info("Acquisition configuration: " + str(file_path))
        acquisition_configuration = AcquisitionConfiguration(file_path)
    else:
        AcquisitionPreferences.deinit_acquisition_preferences()
        acquisition_configuration = None


Registry.register_component(handle_application_changed, {"application_changed"})


class AcquisitionPreferencePanel:

    def __init__(self):
        self.identifier = "nion.acquisition-panel"
        self.label = _("Acquisition")

    def build(self, ui: UserInterfaceModule.UserInterface, event_loop=None, **kwargs):
        u = Declarative.DeclarativeUI()

        class ControlDescriptionHandler:
            def __init__(self, item: AcquisitionPreferences.ControlDescription):
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

        class Handler:
            def __init__(self):
                self.sorted_controls = ListModel.FilteredListModel(container=AcquisitionPreferences.acquisition_preferences, items_key="control_customizations")
                self.sorted_controls.sort_key = operator.attrgetter("name")
                self.sorted_controls.filter = ListModel.PredicateFilter(lambda x: x.is_customizable)
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
                self.sorted_controls = typing.cast(ListModel.ListModel[AcquisitionPreferences.ControlDescription], None)

            def create_handler(self, component_id: str, container=None, item=None, **kwargs):
                if component_id == "control-component":
                    assert container is not None
                    assert item is not None
                    return ControlDescriptionHandler(item)
                return None

        return Declarative.DeclarativeWidget(ui, event_loop, Handler())


PreferencesDialog.PreferencesManager().register_preference_pane(AcquisitionPreferencePanel())


class AcquisitionPanelExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.instrumentation-kit.acquisition-panel"

    def __init__(self, api_broker):
        Workspace.WorkspaceManager().register_panel(AcquisitionPanel, "acquisition-panel", _("Acquisition"), ["left", "right"], "right", {"min-width": 320, "height": 60})

    def close(self) -> None:
        pass
