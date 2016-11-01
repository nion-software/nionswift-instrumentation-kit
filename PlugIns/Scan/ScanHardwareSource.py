# standard libraries
import copy
import gettext
import logging
import queue
import threading
import weakref

# third party libraries
# None

# local libraries
from nion.swift.model import DataItem
from nion.swift.model import Graphics
from nion.swift.model import HardwareSource
from nion.swift.model import Utility
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
    def __property_changed(self, property_name, property_value):
        if property_name == self.__graphic_property_name:
            old_property_value = getattr(self.source, self.__region_property_name)
            # to prevent message loops, check to make sure it changed
            if property_value != old_property_value:
                self.update_source(property_value)



class ProbeGraphicConnection(object):

    """Manage the connection between the hardware and the graphics representing the probe on a display.

    This object does not change any state; it only facilitates the connection between the graphic and
    the scan hardware source."""

    # TODO: translate position to physical position using display

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


class BaseScanHardwareSource(HardwareSource.HardwareSource):

    def __init__(self, hardware_source_id, hardware_source_name):
        super().__init__(hardware_source_id, hardware_source_name)
        self.features["is_scanning"] = True
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()
        self.__probe_position = None
        self.__probe_state_stack = list()
        self.__probe_state_stack.append("parked")
        self.probe_state_changed_event = Event.Event()
        self.__last_data_items = list()
        self.__probe_graphic_connections = list()

    def init_probe_state(self, state):
        self.__probe_state_stack[0] = state

    def close(self):
        # when overriding hardware source close, the acquisition loop may still be running
        # so nothing can be changed here that will make the acquisition loop fail.
        self.__last_data_items = list()
        super().close()

    def _enter_scanning_state(self):
        self.__probe_state_stack.append("scanning")
        self._set_blanker(self.probe_state == "blanked")
        self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)

    def _exit_scanning_state(self):
        self.__probe_state_stack.pop()
        self._set_blanker(self.probe_state == "blanked")
        self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)

    def _set_static_probe_state(self, value, propogate=True):
        # propogate is whether the value gets set back to hardware; pass False to prevent change loops
        if self.probe_state != value:
            if value != "parked" and value != "blanked":
                logging.warning("static_probe_state must be 'parked' or 'blanked'")
                value = "parked"
            self.__probe_state_stack[0] = value
            self._set_blanker(value == "blanked")
            self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)
            for probe_graphic_connection in self.__probe_graphic_connections:
                probe_graphic_connection.update_probe_state(self.probe_position, self.static_probe_state)

    @property
    def static_probe_state(self):
        return self.__probe_state_stack[0]

    @static_probe_state.setter
    def static_probe_state(self, value):
        self._set_static_probe_state(value)

    @property
    def probe_state(self):
        return self.__probe_state_stack[-1]

    def _set_probe_position(self, probe_position):
        """Subclasses should override this method. Called when probe position changes."""
        pass

    def _set_blanker(self, blanker_on):
        """Subclasses should override this method. Called when blanker state changes."""
        pass

    @property
    def _actual_blanker(self):
        raise NotImplementedError()

    @property
    def probe_position(self):
        """ Return the probe position, in normalized coordinates with origin at top left. """
        return self.__probe_position

    @probe_position.setter
    def probe_position(self, probe_position):
        """ Set the probe position, in normalized coordinates with origin at top left. """
        if probe_position is not None:
            # convert the probe position to a FloatPoint and limit it to the 0.0 to 1.0 range in both axes.
            probe_position = Geometry.FloatPoint.make(probe_position)
            probe_position = Geometry.FloatPoint(y=max(min(probe_position.y, 1.0), 0.0),
                                                 x=max(min(probe_position.x, 1.0), 0.0))
        self.__probe_position = probe_position
        # subclasses will override _set_probe_position
        self._set_probe_position(probe_position)
        # update the probe position for listeners and also explicitly update for probe_graphic_connections.
        self.probe_state_changed_event.fire(self.probe_state, self.probe_position, self.static_probe_state)
        for probe_graphic_connection in self.__probe_graphic_connections:
            probe_graphic_connection.update_probe_state(self.probe_position, self.static_probe_state)

    def validate_probe_position(self):
        """Validate the probe position.

        This is called when the user switches from not controlling to controlling the position."""
        self.probe_position = Geometry.FloatPoint(y=0.5, x=0.5)

    # override from the HardwareSource parent class.
    def data_item_states_changed(self, data_item_states):
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
                    probe_position_value.on_value_changed = lambda value: setattr(self, "probe_position", value)
                    probe_graphic_connection = ProbeGraphicConnection(display, probe_position_value)
                    probe_graphic_connection.update_probe_state(self.probe_position, self.static_probe_state)
                    self.__probe_graphic_connections.append(probe_graphic_connection)
        else:
            # scanning, remove all known probe graphic connections.
            probe_graphic_connections = copy.copy(self.__probe_graphic_connections)
            self.__probe_graphic_connections = list()
            for probe_graphic_connection in probe_graphic_connections:
                probe_graphic_connection.close()

        self.__last_data_items = [data_item_state.get("data_item") for data_item_state in data_item_states]


class ScanAcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, delegate, scan_hardware_source):
        super().__init__(delegate.is_continuous)
        self.__delegate = delegate
        self.__weak_scan_hardware_source = weakref.ref(scan_hardware_source)

    def set_frame_parameters(self, frame_parameters):
        self.__delegate.set_frame_parameters(frame_parameters)

    @property
    def frame_parameters(self):
        return self.__delegate.frame_parameters

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self.__weak_scan_hardware_source()._enter_scanning_state()
        if self.__delegate.start_acquisition():
            return True
        return False

    def _suspend_acquisition(self) -> None:
        super()._suspend_acquisition()
        return self.__delegate.suspend_acquisition()

    def _resume_acquisition(self) -> None:
        super()._resume_acquisition()
        self.__delegate.resume_acquisition()

    def _request_abort_acquisition(self):
        super()._request_abort_acquisition()
        self.__delegate.request_abort_acquisition()

    def _abort_acquisition(self) -> None:
        super()._abort_acquisition()
        self.__delegate.abort_acquisition()

    def _mark_acquisition(self) -> None:
        super()._mark_acquisition()
        self.__delegate.mark_acquisition()

    def _stop_acquisition(self) -> None:
        super()._stop_acquisition()
        self.__delegate.stop_acquisition()
        self.__weak_scan_hardware_source()._exit_scanning_state()

    def _acquire_data_elements(self):
        return self.__delegate.acquire_data_elements()


class ScanHardwareSource(BaseScanHardwareSource):

    def __init__(self, scan_adapter):
        super().__init__(scan_adapter.hardware_source_id, scan_adapter.display_name)
        self.__scan_adapter = scan_adapter
        self.__scan_adapter.on_selected_profile_index_changed = self.__selected_profile_index_changed
        self.__scan_adapter.on_profile_frame_parameters_changed = self.__profile_frame_parameters_changed
        self.__scan_adapter.on_channel_states_changed = self.__channel_states_changed
        self.__scan_adapter.on_static_probe_state_changed = self.__static_probe_state_changed
        self.features.update(self.__scan_adapter.features)
        for channel_info in self.__scan_adapter.channel_info_list:
            self.add_data_channel(channel_info.channel_id, channel_info.name)
        self.__profiles = list()
        self.__profiles.extend(self.__scan_adapter.get_initial_profiles())
        self.__current_profile_index = self.__scan_adapter.get_initial_profile_index()
        self.__frame_parameters = self.__profiles[0]
        self.__record_parameters = self.__profiles[2]
        self.__acquisition_task = None
        self.profile_changed_event = Event.Event()
        self.frame_parameters_changed_event = Event.Event()
        self.channel_state_changed_event = Event.Event()
        self.static_probe_state = scan_adapter.initial_state_probe_state
        # the task queue is a list of tasks that must be executed on the UI thread. items are added to the queue
        # and executed at a later time in the __handle_executing_task_queue method.
        self.__task_queue = queue.Queue()
        self.__latest_values_lock = threading.RLock()
        self.__latest_values = dict()

    def close(self):
        self.__scan_adapter.on_selected_profile_index_changed = None
        self.__scan_adapter.on_profile_frame_parameters_changed = None
        super().close()
        # keep the scan adapter around until super close is called, since super
        # may do something that requires the scan adapter (blanker).
        self.__scan_adapter.close()
        self.__scan_adapter = None

    def periodic(self):
        self.__handle_executing_task_queue()

    def __handle_executing_task_queue(self):
        try:
            task = self.__task_queue.get(False)
            try:
                task()
            except Exception as e:
                import traceback
                traceback.print_exc()
                traceback.print_stack()
            self.__task_queue.task_done()
        except queue.Empty as e:
            pass

    def _create_acquisition_view_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__frame_parameters is not None
        return ScanAcquisitionTask(self.__scan_adapter.create_acquisition_task(self.__frame_parameters), self)

    def _view_task_updated(self, view_task):
        self.__acquisition_task = view_task

    def _create_acquisition_record_task(self) -> HardwareSource.AcquisitionTask:
        assert self.__record_parameters is not None
        return ScanAcquisitionTask(self.__scan_adapter.create_record_task(self.__record_parameters), self)

    def __update_frame_parameters(self, profile_index, frame_parameters):
        # update the frame parameters as they are changed from the low level. no need to set them.
        frame_parameters = copy.copy(frame_parameters)
        self.__profiles[profile_index] = frame_parameters
        if profile_index == self.__current_profile_index:
            self.__frame_parameters = copy.copy(frame_parameters)
        if profile_index == 2:
            self.__record_parameters = copy.copy(frame_parameters)
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def set_frame_parameters(self, profile_index, frame_parameters):
        frame_parameters = copy.copy(frame_parameters)
        self.__profiles[profile_index] = frame_parameters
        self.__scan_adapter.set_profile_frame_parameters(profile_index, frame_parameters)
        if profile_index == self.__current_profile_index:
            self.set_current_frame_parameters(frame_parameters)
        if profile_index == 2:
            self.set_record_frame_parameters(frame_parameters)
        self.frame_parameters_changed_event.fire(profile_index, frame_parameters)

    def get_frame_parameters(self, profile):
        return copy.copy(self.__profiles[profile])

    def set_current_frame_parameters(self, frame_parameters):
        if self.__acquisition_task:
            self.__acquisition_task.set_frame_parameters(frame_parameters)
        self.__frame_parameters = copy.copy(frame_parameters)

    def get_current_frame_parameters(self):
        return self.__frame_parameters

    def set_record_frame_parameters(self, frame_parameters):
        self.__record_parameters = copy.copy(frame_parameters)

    def get_record_frame_parameters(self):
        return self.__record_parameters

    def get_channel_state(self, channel_index):
        return self.__scan_adapter.get_channel_state(channel_index)

    def set_channel_enabled(self, channel_index, enabled):
        changed = self.__scan_adapter.set_channel_enabled(channel_index, enabled)
        if changed:
            self.__channel_states_changed([self.get_channel_state(i_channel_index) for i_channel_index in range(self.channel_count)])

    def set_selected_profile_index(self, profile_index):
        self.__current_profile_index = profile_index
        self.__scan_adapter.set_selected_profile_index(profile_index)
        self.set_current_frame_parameters(self.__profiles[self.__current_profile_index])
        self.profile_changed_event.fire(profile_index)

    @property
    def selected_profile_index(self):
        return self.__current_profile_index

    def __selected_profile_index_changed(self, profile_index):
        self.__task_queue.put(lambda: self.set_selected_profile_index(profile_index))

    def __profile_frame_parameters_changed(self, profile_index, frame_parameters):
        # this method will be called when the device changes parameters (via a dialog or something similar).
        # it calls __update_frame_parameters instead of set_frame_parameters so that we do _not_ update the
        # current acquisition (which can cause a cycle in that it would again set the low level values, which
        # itself wouldn't be an issue unless the user makes multiple changes in quick succession). not setting
        # current values is different semantics than the scan control panel, which _does_ set current values if
        # the current profile is selected. Hrrmmm.
        with self.__latest_values_lock:
            self.__latest_values[profile_index] = frame_parameters
        def do_update_parameters():
            with self.__latest_values_lock:
                for profile_index in self.__latest_values.keys():
                    self.__update_frame_parameters(profile_index, self.__latest_values[profile_index])
                self.__latest_values = dict()
        self.__task_queue.put(do_update_parameters)

    def __channel_states_changed(self, channel_states):
        # this method will be called when the device changes channels enabled (via dialog or script).
        # it updates the channels internally but does not send out a message to set the channels to the
        # hardware, since they're already set, and doing so can cause strange change loops.
        assert len(channel_states) == self.channel_count
        def channel_states_changed():
            for channel_index, channel_state in enumerate(channel_states):
                self.channel_state_changed_event.fire(channel_index, channel_state.channel_id, channel_state.name, channel_state.enabled)
            at_least_one_enabled = False
            for channel_index in range(self.channel_count):
                if self.get_channel_state(channel_index).enabled:
                    at_least_one_enabled = True
                    break
            if not at_least_one_enabled:
                self.stop_playing()
        self.__task_queue.put(channel_states_changed)

    def __static_probe_state_changed(self, static_probe_state):
        # this method will be called when the device changes probe state (via dialog or script).
        # it updates the channels internally but does not send out a message to set the probe state
        # to the hardware, since it's already set, and doing so can cause strange change loops.
        def static_probe_state_changed():
            self._set_static_probe_state(static_probe_state, propogate=False)
        self.__task_queue.put(static_probe_state_changed)

    def get_frame_parameters_from_dict(self, d):
        return self.__scan_adapter.get_frame_parameters_from_dict(d)

    def get_current_frame_time(self):
        frame_parameters = self.get_current_frame_parameters()
        return frame_parameters.size[0] * frame_parameters.size[1] * frame_parameters.pixel_time_us / 1000000.0

    def clean_data_item(self, data_item: DataItem.DataItem, data_channel: HardwareSource.DataChannel) -> None:
        display = data_item.maybe_data_source.displays[0]
        for graphic in copy.copy(display.graphics):
            if graphic.graphic_id == "probe":
                display.remove_graphic(graphic)

    # override from BaseScanHardwareSource
    def _set_probe_position(self, probe_position):
        self.__scan_adapter.set_probe_position(probe_position)

    # override from BaseScanHardwareSource
    def _set_blanker(self, blanker_on):
        self.__scan_adapter.set_blanker(blanker_on)

    @property
    def _actual_blanker(self):
        return self.__scan_adapter.actual_blanker

    @property
    def use_hardware_simulator(self):
        return False

    def get_property(self, name):
        return self.__scan_adapter.get_property(name)

    def set_property(self, name, value):
        self.__scan_adapter.set_property(name, value)

    def open_configuration_interface(self):
        self.__scan_adapter.open_configuration_interface()

    def shift_click(self, mouse_position, scan_shape):
        if hasattr(self.__scan_adapter, "shift_click") and callable(self.__scan_adapter.shift_click):
            self.__scan_adapter.shift_click(mouse_position, scan_shape)

    def increase_pmt(self, channel_index):
        if hasattr(self.__scan_adapter, "increase_pmt") and callable(self.__scan_adapter.increase_pmt):
            self.__scan_adapter.increase_pmt(channel_index)

    def decrease_pmt(self, channel_index):
        if hasattr(self.__scan_adapter, "decrease_pmt") and callable(self.__scan_adapter.decrease_pmt):
            self.__scan_adapter.decrease_pmt(channel_index)

    def get_api(self, version):
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Camera API requested version %s is greater than %s." % (version, actual_version))

        class CameraFacade(object):

            def __init__(self):
                pass

        return CameraFacade()
