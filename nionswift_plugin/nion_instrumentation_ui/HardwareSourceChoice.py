# system imports
import operator
import typing

# local libraries
from nion.instrumentation import HardwareSource
from nion.ui import UserInterface
from nion.utils import Binding
from nion.utils import Event
from nion.utils import Model
from nion.utils import Stream
from nion.utils.ReferenceCounting import weak_partial


class PersistentStorageInterface:
    def get_persistent_string(self, key: str, default_value: typing.Optional[str] = None) -> str: ...
    def set_persistent_string(self, key: str, value: str) -> None: ...


class HardwareSourceChoice:
    def __init__(self, choice_model: Model.PropertyModel[str],
                 filter: typing.Optional[typing.Callable[[HardwareSource.HardwareSource], bool]] = None):
        self.__choice_model = choice_model
        self.hardware_sources_model = Model.PropertyModel[typing.List[HardwareSource.HardwareSource]](list())
        self.hardware_source_index_model = Model.PropertyModel[int](0)
        self.hardware_source_changed_event = Event.Event()
        self.__filter = filter or (lambda x: True)
        self.__hardware_source_added_event_listener = HardwareSource.HardwareSourceManager().hardware_source_added_event.listen(weak_partial(HardwareSourceChoice.__rebuild_hardware_source_list, self))
        self.__hardware_source_removed_event_listener = HardwareSource.HardwareSourceManager().hardware_source_removed_event.listen(weak_partial(HardwareSourceChoice.__rebuild_hardware_source_list, self))
        self.__rebuild_hardware_source_list(self.hardware_source)

        hardware_source_id = choice_model.value
        new_index = None
        hardware_sources = self.hardware_sources_model.value or list()
        for index, hardware_source in enumerate(hardware_sources):
            if hardware_source.hardware_source_id == hardware_source_id:
                new_index = index
                break
        new_index = new_index if new_index is not None else 0 if len(hardware_sources) > 0 else None
        self.hardware_source_index_model.value = new_index
        self.hardware_source_changed_event.fire(self.hardware_source)

        self.__property_changed_event_listener = self.hardware_source_index_model.property_changed_event.listen(weak_partial(HardwareSourceChoice.__update_current_hardware_source, self))

    def close(self) -> None:
        self.__hardware_source_added_event_listener.close()
        self.__hardware_source_added_event_listener = typing.cast(typing.Any, None)
        self.__hardware_source_removed_event_listener.close()
        self.__hardware_source_removed_event_listener = typing.cast(typing.Any, None)
        self.__property_changed_event_listener.close()
        self.__property_changed_event_listener = typing.cast(typing.Any, None)

    @property
    def hardware_source_count(self) -> int:
        hardware_sources = self.hardware_sources_model.value or list()
        return len(hardware_sources)

    @property
    def hardware_sources(self) -> typing.Sequence[HardwareSource.HardwareSource]:
        return self.hardware_sources_model.value or list()

    @property
    def hardware_source(self) -> typing.Optional[HardwareSource.HardwareSource]:
        index = self.hardware_source_index_model.value or 0
        hardware_sources = self.hardware_sources_model.value or list()
        return hardware_sources[index] if 0 <= index < len(hardware_sources) else None

    def create_combo_box(self, ui: UserInterface.UserInterface) -> UserInterface.ComboBoxWidget:
        combo_box = ui.create_combo_box_widget(self.hardware_sources_model.value, item_getter=operator.attrgetter("display_name"))
        combo_box.bind_items(Binding.PropertyBinding(self.hardware_sources_model, "value"))
        combo_box.bind_current_index(Binding.PropertyBinding(self.hardware_source_index_model, "value"))
        return combo_box

    def __rebuild_hardware_source_list(self, h: typing.Optional[HardwareSource.HardwareSource]) -> None:
        # keep selected item the same
        old_index = self.hardware_source_index_model.value or 0
        hardware_sources = self.hardware_sources_model.value or list()
        old_hardware_source = hardware_sources[old_index] if 0 <= old_index < len(hardware_sources) else None
        items = list()
        for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
            if self.__filter(hardware_source):
                items.append(hardware_source)
        self.hardware_sources_model.value = sorted(items, key=operator.attrgetter("display_name"))
        new_index = None
        for index, hardware_source in enumerate(self.hardware_sources_model.value):
            if hardware_source == old_hardware_source:
                new_index = index
                break
        new_index = new_index if new_index is not None else 0 if len(self.hardware_sources_model.value) > 0 else None
        self.hardware_source_index_model.value = new_index
        self.hardware_source_changed_event.fire(self.hardware_source)

    def __update_current_hardware_source(self, key: str) -> None:
        if key == "value":
            hardware_sources = self.hardware_sources_model.value or list()
            index = self.hardware_source_index_model.value or 0
            hardware_source_id = hardware_sources[index].hardware_source_id
            self.__choice_model.value = hardware_source_id
            self.hardware_source_changed_event.fire(self.hardware_source)


class HardwareSourceChoiceStream(Stream.AbstractStream[HardwareSource.HardwareSource]):

    def __init__(self, hardware_source_choice: HardwareSourceChoice):
        super().__init__()
        # outgoing messages
        self.value_stream = Event.Event()
        # initialize
        self.__value: typing.Optional[HardwareSource.HardwareSource] = None
        # listen for display changes
        self.__hardware_source_changed_listener = hardware_source_choice.hardware_source_changed_event.listen(self.__hardware_source_changed)
        self.__hardware_source_changed(hardware_source_choice.hardware_source)

    def about_to_delete(self) -> None:
        self.__hardware_source_changed_listener.close()
        self.__hardware_source_changed_listener = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def value(self) -> typing.Optional[HardwareSource.HardwareSource]:
        return self.__value

    def __hardware_source_changed(self, hardware_source: typing.Optional[HardwareSource.HardwareSource]) -> None:
        if hardware_source != self.__value:
            self.__value = hardware_source
            self.value_stream.fire(self.__value)
