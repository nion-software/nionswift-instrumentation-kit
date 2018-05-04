# system imports
import operator

# local libraries
from nion.swift.model import HardwareSource as HardwareSourceModule
from nion.utils import Binding
from nion.utils import Event
from nion.utils import Model


class HardwareSourceChoice:
    def __init__(self, ui, hardware_source_key, filter=None):

        self.hardware_sources_model = Model.PropertyModel(list())
        self.hardware_source_index_model = Model.PropertyModel()

        self.hardware_source_changed_event = Event.Event()

        filter = filter if filter is not None else lambda x: True

        def rebuild_hardware_source_list():
            # keep selected item the same
            old_index = self.hardware_source_index_model.value
            old_hardware_source = self.hardware_sources_model.value[old_index] if old_index is not None else None
            items = list()
            for hardware_source in HardwareSourceModule.HardwareSourceManager().hardware_sources:
                if filter(hardware_source):
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

        self.__hardware_source_added_event_listener = HardwareSourceModule.HardwareSourceManager().hardware_source_added_event.listen(lambda h: rebuild_hardware_source_list())
        self.__hardware_source_removed_event_listener = HardwareSourceModule.HardwareSourceManager().hardware_source_removed_event.listen(lambda h: rebuild_hardware_source_list())

        rebuild_hardware_source_list()

        hardware_source_id = ui.get_persistent_string(hardware_source_key)

        new_index = None
        for index, hardware_source in enumerate(self.hardware_sources_model.value):
            if hardware_source.hardware_source_id == hardware_source_id:
                new_index = index
                break
        new_index = new_index if new_index is not None else 0 if len(self.hardware_sources_model.value) > 0 else None
        self.hardware_source_index_model.value = new_index
        self.hardware_source_changed_event.fire(self.hardware_source)

        def update_current_hardware_source(key):
            if key == "value":
                hardware_source_id = self.hardware_sources_model.value[self.hardware_source_index_model.value].hardware_source_id
                ui.set_persistent_string(hardware_source_key, hardware_source_id)
                self.hardware_source_changed_event.fire(self.hardware_source)

        self.__property_changed_event_listener = self.hardware_source_index_model.property_changed_event.listen(update_current_hardware_source)

    def close(self):
        self.__hardware_source_added_event_listener.close()
        self.__hardware_source_added_event_listener = None
        self.__hardware_source_removed_event_listener.close()
        self.__hardware_source_removed_event_listener = None
        self.__property_changed_event_listener.close()
        self.__property_changed_event_listener = None

    @property
    def hardware_source_count(self) -> int:
        return len(self.hardware_sources_model.value)

    @property
    def hardware_source(self):
        index = self.hardware_source_index_model.value
        hardware_sources = self.hardware_sources_model.value
        return hardware_sources[index] if (index is not None and 0 <= index < len(hardware_sources)) else None

    def create_combo_box(self, ui):
        combo_box = ui.create_combo_box_widget(self.hardware_sources_model.value, item_getter=operator.attrgetter("display_name"))
        combo_box.bind_items(Binding.PropertyBinding(self.hardware_sources_model, "value"))
        combo_box.bind_current_index(Binding.PropertyBinding(self.hardware_source_index_model, "value"))
        return combo_box
