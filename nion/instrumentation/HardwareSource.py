"""
This module defines classes for handling live data sources.

A HardwareSource represents a source of data and metadata frames.

An AcquisitionTask tracks the state for a particular acquisition of a frame or sequence of frames.

The HardwareSourceManager allows callers to register and unregister hardware sources.

This module also defines individual functions that can be used to collect data from hardware sources.
"""
from __future__ import annotations

# system imports
import abc
import contextlib
import copy
import enum
import functools
import gettext
import logging
import threading
import time
import typing
import uuid

# library imports
import numpy

# local imports
from nion.data import Core
from nion.data import DataAndMetadata
from nion.swift.model import Activity
from nion.swift.model import DataItem
from nion.swift.model import DisplayItem
from nion.swift.model import Graphics
from nion.swift.model import ImportExportManager
from nion.swift.model import Metadata
from nion.swift.model import Utility
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Observable
from nion.utils import Registry
from nion.utils.ReferenceCounting import weak_partial

if typing.TYPE_CHECKING:
    from nion.swift.model import DocumentModel

_ = gettext.gettext


class DataAndMetadataPromise:
    def __init__(self, xdata: typing.Optional[DataAndMetadata.DataAndMetadata]) -> None:
        self.__xdata_unsafe = xdata
        self.__xdata: typing.Optional[DataAndMetadata.DataAndMetadata] = None

    @property
    def xdata(self) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
        if not self.__xdata and self.__xdata_unsafe:
            self.__xdata = copy.deepcopy(self.__xdata_unsafe)
            self.__xdata_unsafe = None
        return self.__xdata


DataElementType = typing.Dict[str, typing.Any]
_FinishedCallbackType = typing.Callable[[typing.Sequence[DataAndMetadataPromise]], None]
_NDArray = numpy.typing.NDArray[typing.Any]


class DocumentModelInterface(typing.Protocol):

    @abc.abstractmethod
    def _call_soon(self, fn: typing.Callable[[], None]) -> None: ...

    @abc.abstractmethod
    def update_data_item_session(self, data_item: DataItem.DataItem) -> None: ...

    @abc.abstractmethod
    def get_data_item_channel_reference(self, hardware_source_id: str, channel_id: typing.Optional[str]) -> DocumentModel.DocumentModel.DataItemReference: ...

    @abc.abstractmethod
    def _update_data_item_reference(self, key: str, data_item: DataItem.DataItem) -> None: ...

    @abc.abstractmethod
    def append_data_item(self, data_item: DataItem.DataItem, auto_display: bool = True) -> None: ...

    @abc.abstractmethod
    def _queue_data_item_update(self, data_item: DataItem.DataItem, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None: ...

    @abc.abstractmethod
    def get_display_items_for_data_item(self, data_item: typing.Optional[DataItem.DataItem]) -> typing.Set[DisplayItem.DisplayItem]: ...

    @property
    @abc.abstractmethod
    def project_loaded_event(self) -> Event.Event: ...


class HardwareSourceBridge:
    # NOTE: this will always be created before the first document is loaded.

    def __init__(self, document_model: DocumentModelInterface):
        self.__document_model = document_model
        self.__data_items_to_append_lock = threading.RLock()
        self.__data_items_to_append: typing.List[typing.Tuple[str, DataItem.DataItem]] = list()
        self.__data_channel_updated_listeners: typing.Dict[str, typing.List[Event.EventListener]] = dict()
        self.__data_channel_start_listeners: typing.Dict[str, typing.List[Event.EventListener]] = dict()
        self.__data_channel_stop_listeners: typing.Dict[str, typing.List[Event.EventListener]] = dict()
        self.__data_channel_states_updated_listeners: typing.Dict[str, Event.EventListener] = dict()
        self.__hardware_source_added_listener = HardwareSourceManager().hardware_source_added_event.listen(self.__hardware_source_added)
        self.__hardware_source_removed_listener = HardwareSourceManager().hardware_source_removed_event.listen(self.__hardware_source_removed)
        for hardware_source in HardwareSourceManager().hardware_sources:
            self.__hardware_source_added(hardware_source)

    def close(self) -> None:
        # close hardware source related stuff
        # close data items left to append that haven't been appended
        with self.__data_items_to_append_lock:
            for key, data_item in self.__data_items_to_append:
                data_item.close()
        self.__hardware_source_added_listener.close()
        self.__hardware_source_added_listener = typing.cast(typing.Any, None)
        self.__hardware_source_removed_listener.close()
        self.__hardware_source_removed_listener = typing.cast(typing.Any, None)
        for listener in self.__data_channel_states_updated_listeners.values():
            listener.close()
        self.__data_channel_states_updated_listeners = typing.cast(typing.Any, None)
        HardwareSourceManager()._hardware_source_bridge_closed()
        for listeners in self.__data_channel_updated_listeners.values():
            for listener in listeners:
                listener.close()
        for listeners in self.__data_channel_start_listeners.values():
            for listener in listeners:
                listener.close()
        for listeners in self.__data_channel_stop_listeners.values():
            for listener in listeners:
                listener.close()
        self.__data_channel_updated_listeners = typing.cast(typing.Any, None)
        self.__data_channel_start_listeners = typing.cast(typing.Any, None)
        self.__data_channel_stop_listeners = typing.cast(typing.Any, None)
        self.__document_model = typing.cast(typing.Any, None)

    def __call_soon(self, fn: typing.Callable[[], None]) -> None:
        assert self.__document_model
        self.__document_model._call_soon(fn)

    def __hardware_source_added(self, hardware_source: HardwareSource) -> None:
        self.__data_channel_states_updated_listeners[hardware_source.hardware_source_id] = hardware_source.data_channel_states_updated.listen(functools.partial(self.__data_channel_states_updated, hardware_source))
        for data_channel in hardware_source.data_channels:
            data_channel_updated_listener = data_channel.data_channel_updated_event.listen(functools.partial(self.__data_channel_updated, hardware_source, data_channel))
            self.__data_channel_updated_listeners.setdefault(hardware_source.hardware_source_id, list()).append(data_channel_updated_listener)
            data_channel_start_listener = data_channel.data_channel_start_event.listen(functools.partial(self.__data_channel_start, hardware_source, data_channel))
            self.__data_channel_start_listeners.setdefault(hardware_source.hardware_source_id, list()).append(data_channel_start_listener)
            data_channel_stop_listener = data_channel.data_channel_stop_event.listen(functools.partial(self.__data_channel_stop, hardware_source, data_channel))
            self.__data_channel_stop_listeners.setdefault(hardware_source.hardware_source_id, list()).append(data_channel_stop_listener)

    def __hardware_source_removed(self, hardware_source: HardwareSource) -> None:
        self.__data_channel_states_updated_listeners[hardware_source.hardware_source_id].close()
        del self.__data_channel_states_updated_listeners[hardware_source.hardware_source_id]
        for listener in self.__data_channel_updated_listeners.get(hardware_source.hardware_source_id, list()):
            listener.close()
        for listener in self.__data_channel_start_listeners.get(hardware_source.hardware_source_id, list()):
            listener.close()
        for listener in self.__data_channel_stop_listeners.get(hardware_source.hardware_source_id, list()):
            listener.close()
        self.__data_channel_updated_listeners.pop(hardware_source.hardware_source_id, None)
        self.__data_channel_start_listeners.pop(hardware_source.hardware_source_id, None)
        self.__data_channel_stop_listeners.pop(hardware_source.hardware_source_id, None)

    def __construct_data_item_reference(self, hardware_source: HardwareSource, data_channel: DataChannel) -> DocumentModel.DocumentModel.DataItemReference:
        """Construct a data item reference.

        Construct a data item reference and assign a data item to it. Update data item session id and session metadata.
        Also connect the data channel processor.

        This method is thread safe.
        """
        data_item_reference = self.__document_model.get_data_item_channel_reference(hardware_source.hardware_source_id, data_channel.channel_id)
        with data_item_reference.mutex:
            data_item = data_item_reference.data_item
            # if we still don't have a data item, create it.
            if data_item is None:
                data_item = DataItem.DataItem()
                data_item.title = "%s (%s)" % (hardware_source.display_name, data_channel.name) if data_channel.name else hardware_source.display_name
                data_item.category = "temporary"
                data_item_reference.data_item = data_item

                def append_data_items() -> None:
                    with self.__data_items_to_append_lock:
                        for key, data_item in self.__data_items_to_append:
                            self.__document_model.append_data_item(data_item)
                            # this next line may be redundant since the data_item_reference.data_item is set above.
                            # however, this line may ensure it is written to disk; needs a test before removing.
                            self.__document_model._update_data_item_reference(key, data_item)
                        self.__data_items_to_append.clear()

                with self.__data_items_to_append_lock:
                    self.__data_items_to_append.append((data_item_reference.key, data_item))

                self.__call_soon(append_data_items)

            def update_session(data_item: DataItem.DataItem) -> None:
                # since this is a delayed call, the data item might have disappeared. check it.
                if data_item._closed:
                    return
                self.__document_model.update_data_item_session(data_item)
                src_channel_index = data_channel.src_channel_index
                sum_processor = data_channel.processor
                if sum_processor and src_channel_index is not None:
                    src_data_channel = hardware_source.data_channels[src_channel_index]
                    src_data_item_reference = self.__document_model.get_data_item_channel_reference(hardware_source.hardware_source_id, src_data_channel.channel_id)
                    sum_processor.connect_data_item_reference(src_data_item_reference)

            assert data_item
            self.__call_soon(functools.partial(update_session, data_item))

            return data_item_reference

    def __data_channel_start(self, hardware_source: HardwareSource, data_channel: DataChannel) -> None:
        def data_channel_start() -> None:
            assert threading.current_thread() == threading.main_thread()
            data_item_reference = self.__document_model.get_data_item_channel_reference(hardware_source.hardware_source_id, data_channel.channel_id)
            data_item_reference.start()
        self.__call_soon(data_channel_start)

    def __data_channel_stop(self, hardware_source: HardwareSource, data_channel: DataChannel) -> None:
        def data_channel_stop() -> None:
            assert threading.current_thread() == threading.main_thread()
            data_item_reference = self.__document_model.get_data_item_channel_reference(hardware_source.hardware_source_id, data_channel.channel_id)
            data_item_reference.stop()
        self.__call_soon(data_channel_stop)

    def __data_channel_updated(self, hardware_source: HardwareSource, data_channel: DataChannel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
        data_item_reference = self.__construct_data_item_reference(hardware_source, data_channel)
        data_item = data_item_reference.data_item
        assert data_item
        self.__document_model._queue_data_item_update(data_item, data_and_metadata)

    def __data_channel_states_updated(self, hardware_source: HardwareSource, data_channels: typing.Sequence[DataChannel]) -> None:
        channel_map: typing.Dict[str, DataItem.DataItem] = dict()
        for data_channel in data_channels:
            data_item_reference = self.__document_model.get_data_item_channel_reference(hardware_source.hardware_source_id, data_channel.channel_id)
            data_item = data_item_reference.data_item
            channel_id = data_channel.channel_id
            if channel_id and data_item:
                channel_map[channel_id] = data_item
        hardware_source.data_channel_map_updated(channel_map)


# Keeps track of all registered hardware sources and instruments.
# Also keeps track of aliases between hardware sources and logical names.
class HardwareSourceManager(metaclass=Utility.Singleton):
    def __init__(self) -> None:
        super().__init__()
        self.hardware_sources: typing.List[HardwareSource] = list()
        self.instruments: typing.List[Instrument] = list()
        # we create a list of callbacks for when a hardware
        # source is added or removed
        self.hardware_source_added_event = Event.Event()
        self.hardware_source_removed_event = Event.Event()
        self.instrument_added_event = Event.Event()
        self.instrument_removed_event = Event.Event()
        self.aliases_updated: typing.List[typing.Callable[[], None]] = list()
        # aliases are shared between hardware sources and instruments
        self.__aliases: typing.Dict[str, typing.Tuple[str, str]] = dict()

    def close(self) -> None:
        self._close_hardware_sources()
        self._close_instruments()

    def _close_instruments(self) -> None:
        for instrument in self.instruments:
            if hasattr(instrument, "close"):
                instrument.close()
        self.instruments = []

    def _close_hardware_sources(self) -> None:
        for hardware_source in self.hardware_sources:
            if hasattr(hardware_source, "close"):
                hardware_source.close()
        self.hardware_sources = []

    def _reset(self) -> None:  # used for testing to start from scratch
        self.hardware_sources = []
        self.instruments = []
        self.hardware_source_added_event = Event.Event()
        self.hardware_source_removed_event = Event.Event()
        self.instrument_added_event = Event.Event()
        self.instrument_removed_event = Event.Event()
        self.__aliases = {}

    def register_hardware_source(self, hardware_source: HardwareSource) -> None:
        self.hardware_sources.append(hardware_source)
        self.hardware_source_added_event.fire(hardware_source)

    def unregister_hardware_source(self, hardware_source: HardwareSource) -> None:
        self.hardware_sources.remove(hardware_source)
        self.hardware_source_removed_event.fire(hardware_source)

    def register_instrument(self, instrument_id: str, instrument: Instrument) -> None:
        instrument.instrument_id = instrument_id
        self.instruments.append(instrument)
        self.instrument_added_event.fire(instrument)

    def unregister_instrument(self, instrument_id: str) -> None:
        for instrument in self.instruments:
            if instrument.instrument_id == instrument_id:
                instrument.instrument_id = None
                self.instruments.remove(instrument)
                self.instrument_removed_event.fire(instrument)
                break

    def abort_all_and_close(self) -> None:
        for hardware_source in copy.copy(self.hardware_sources):
            hardware_source.abort_playing()

    def get_all_instrument_ids(self) -> typing.Set[str]:
        instrument_ids: typing.Set[str] = set()
        instrument_ids.update(list(instrument.instrument_id for instrument in self.instruments if instrument.instrument_id))
        for alias in self.__aliases.keys():
            resolved_alias = self.get_instrument_by_id(alias)
            if resolved_alias:
                instrument_ids.add(alias)
        return instrument_ids

    def get_all_hardware_source_ids(self) -> typing.Set[str]:
        hardware_source_ids = set()
        hardware_source_ids.update(list(hardware_source.hardware_source_id for hardware_source in self.hardware_sources))
        for alias in self.__aliases.keys():
            resolved_alias = self.get_hardware_source_for_hardware_source_id(alias)
            if resolved_alias:
                hardware_source_ids.add(alias)
        return hardware_source_ids

    def __get_info_for_instrument_id(self, instrument_id: str) -> typing.Optional[typing.Tuple[Instrument, str]]:
        display_name = str()
        seen_instrument_ids = []  # prevent loops, just so we don't get into endless loop in case of user error
        while instrument_id in self.__aliases and instrument_id not in seen_instrument_ids:
            seen_instrument_ids.append(instrument_id)  # must go before next line
            instrument_id, display_name = self.__aliases[instrument_id]
        for instrument in self.instruments:
            if instrument.instrument_id == instrument_id:
                return instrument, display_name
        return None

    # may return None
    def get_instrument_by_id(self, instrument_id: str) -> typing.Optional[Instrument]:
        info = self.__get_info_for_instrument_id(instrument_id)
        if info:
            instrument, display_name = info
            return instrument
        return None

    def __get_info_for_hardware_source_id(self, hardware_source_id: str) -> typing.Optional[typing.Tuple[HardwareSource, str]]:
        display_name = str()
        seen_hardware_source_ids = []  # prevent loops, just so we don't get into endless loop in case of user error
        while hardware_source_id in self.__aliases and hardware_source_id not in seen_hardware_source_ids:
            seen_hardware_source_ids.append(hardware_source_id)  # must go before next line
            hardware_source_id, display_name = self.__aliases[hardware_source_id]
        for hardware_source in self.hardware_sources:
            if hardware_source.hardware_source_id == hardware_source_id:
                return hardware_source, display_name
        return None

    def get_hardware_source_for_hardware_source_id(self, hardware_source_id: str) -> typing.Optional[HardwareSource]:
        info = self.__get_info_for_hardware_source_id(hardware_source_id)
        if info:
            hardware_source, display_name = info
            return hardware_source
        return None

    def make_instrument_alias(self, instrument_id: str, alias_instrument_id: str, display_name: str) -> None:
        """ Configure an alias.

            Callers can use the alias to refer to the instrument or hardware source.
            The alias should be lowercase, no spaces. The display name may be used to display alias to
            the user. Neither the original instrument or hardware source id and the alias id should ever
            be visible to end users.

            :param str instrument_id: the hardware source id (lowercase, no spaces)
            :param str alias_instrument_id: the alias of the hardware source id (lowercase, no spaces)
            :param str display_name: the display name for the alias
        """
        self.__aliases[alias_instrument_id] = (instrument_id, display_name)
        for f in self.aliases_updated:
            f()

    def make_delegate_hardware_source(self, delegate: DelegateAcquisitionTaskProtocol, hardware_source_id: str, hardware_source_name: str) -> DelegateHardwareSource:
        return DelegateHardwareSource(delegate, hardware_source_id, hardware_source_name)

    def register_document_model(self, document_model: DocumentModelInterface) -> HardwareSourceBridge:
        return HardwareSourceBridge(document_model)

    def _hardware_source_bridge_closed(self) -> None:
        for hardware_source in copy.copy(self.hardware_sources):
            hardware_source.abort_playing()


class AcquisitionTask:
    """Basic acquisition task carries out acquisition repeatedly during an acquisition loop, keeping track of state.

    The caller controls the state of the task by calling the following methods:
        execute: start or continue acquisition, should be called repeatedly until is_finished is True
        suspend: suspend the state of acquisition
        resume: resume a suspended state of acquisition
        stop: notify that acquisition should stop after end of current frame
        abort: notify that acquisition should abort as soon as possible

    In addition the caller can query the state of acquisition using the following method:
        is_finished: whether acquisition has finished or not

    Finally, the caller can listen to the following events:
        data_elements_changed_event(data_elements, is_continuous, view_id, is_complete, is_stopping):
            fired when data elements change. the state of acquisition is passed too.

    Subclasses can override these methods to implement the acquisition:
        _start_acquisition: called once at the beginning of this task
        _abort_acquisition: called from thread when the caller has requested to abort acquisition; guaranteed to be called synchronously.
        _request_abort_acquisition: called from UI when the called has requested to abort acquisition; may be called asynchronously.
        _suspend_acquisition: called when the caller has requested to suspend acquisition
        _resume_acquisition: called when the caller has requested to resume a suspended acquisition
        _mark_acquisition: marks the acquisition to stop at end of current frame
        _acquire_data_elements: return list of data elements, with metadata indicating completion status
        _stop_acquisition: final call to indicate acquisition has stopped; subclasses should synchronize stop here
    """

    def __init__(self, continuous: bool):
        self.__started = False
        self.__finished = False
        self.__is_suspended = False
        self.__aborted = False
        self.__is_stopping = False
        self.__is_continuous = continuous
        self.__last_acquire_time = 0.0
        self.__minimum_period = 1 / 1000.0
        self.__frame_index = 0
        self.__view_id = str(uuid.uuid4()) if not continuous else None
        self._test_acquire_exception: typing.Optional[typing.Callable[[Exception], None]] = None
        self._test_start_hook: typing.Optional[typing.Callable[[], None]] = None
        self._test_acquire_hook: typing.Optional[typing.Callable[[], None]] = None
        self.start_event = Event.Event()
        self.stop_event = Event.Event()
        self.data_elements_changed_event = Event.Event()
        self.finished_callback_fn: typing.Optional[_FinishedCallbackType] = None
        self.activity: typing.Optional[Activity.Activity] = None

    def __mark_as_finished(self) -> None:
        self.__finished = True
        self.data_elements_changed_event.fire(list(), self.__view_id, False, self.__is_stopping, None)

    def __mark_as_error(self, e: Exception) -> None:
        self.__finished = True
        self.data_elements_changed_event.fire(list(), self.__view_id, False, self.__is_stopping, e)

    # called from the hardware source
    # note: abort, suspend and execute are always called from the same thread, ensuring that
    # one can't be executing when the other is called.
    def execute(self) -> None:
        # first start the task
        if not self.__started:
            try:
                self.__start()
            except Exception as e:
                # the task is finished if it doesn't start
                self.__mark_as_error(e)
                raise
            self.__started = True
            # logging.debug("%s started", self)
        if self.__is_suspended:
            try:
                self._resume_acquisition()
            finally:
                self.__is_suspended = False
        if not self.__finished:
            try:
                # if aborted, abort here
                if self.__aborted:
                    # logging.debug("%s aborted", self)
                    self._abort_acquisition()
                    # logging.debug("%s stopped", self)
                    self._mark_acquisition()
                    self._stop_acquisition()
                    self.__mark_as_finished()
                # otherwise execute the task
                else:
                    complete = self.__execute_acquire_data_elements()
                    # logging.debug("%s executed %s", self, complete)
                    if complete and (self.__is_stopping or not self.__is_continuous):
                        # logging.debug("%s finished", self)
                        self._stop_acquisition()
                        self.__mark_as_finished()
            except Exception as e:
                # the task is finished if it doesn't execute
                # logging.debug("exception")
                self._safe_stop_acquisition()
                self.__mark_as_error(e)
                raise

    # called from the hardware source
    # note: abort, suspend and execute are always called from the same thread, ensuring that
    # one can't be executing when the other is called.
    def suspend(self) -> None:
        if not self.__is_suspended:
            self.__is_suspended = True
            self._suspend_acquisition()

    @property
    def is_finished(self) -> bool:
        return self.__finished

    # called from the hardware source
    # note: abort, suspend and execute are always called from the same thread, ensuring that
    # one can't be executing when the other is called.
    def abort(self) -> None:
        self.__aborted = True
        self._request_abort_acquisition()

    # called from the hardware source
    def stop(self) -> None:
        self.__is_stopping = True
        self._mark_acquisition()

    def __start(self) -> None:
        if callable(self._test_start_hook):
            self._test_start_hook()

        if not self._start_acquisition():
            self.abort()
        self.__last_acquire_time = time.time() - self.__minimum_period

    def __execute_acquire_data_elements(self) -> bool:
        # with Utility.trace(): # (min_elapsed=0.0005, discard="anaconda"):
        # impose maximum frame rate so that acquire_data_elements can't starve main thread
        elapsed = time.time() - self.__last_acquire_time
        time.sleep(max(0.0, self.__minimum_period - elapsed))

        if callable(self._test_acquire_hook):
            self._test_acquire_hook()

        partial_data_elements = self._acquire_data_elements()
        assert partial_data_elements is not None  # data_elements should never be empty

        # update frame_index if not supplied
        for data_element in partial_data_elements:
            data_element.setdefault("properties", dict()).setdefault("frame_index", self.__frame_index)

        data_elements = copy.copy(partial_data_elements)

        # record the last acquisition time
        self.__last_acquire_time = time.time()

        # figure out whether all data elements are complete
        complete = True
        for data_element in data_elements:
            sub_area = data_element.get("sub_area")
            state = data_element.get("state", "complete")
            section_state = data_element.get("section_state")
            if not (sub_area is None or state == "complete" or section_state == "complete"):
                complete = False
                break

        # notify that data elements have changed. at this point data_elements may contain data stored in low level code.
        self.data_elements_changed_event.fire(data_elements, self.__view_id, complete, self.__is_stopping, None)

        if complete:
            self.__frame_index += 1

        return complete

    # override these routines. the default implementation is to
    # call back to the hardware source.

    # subclasses can implement to start acquisition. it is called once.
    # return True if successful, False if not.
    # called synchronously from execute thread.
    # must be thread safe
    def _start_acquisition(self) -> bool:
        self.start_event.fire()
        return True

    # subclasses can implement this method to abort acquisition.
    # aborted tasks will still get marked, stopped, and send out final
    # data_elements_changed_events and finished_events.
    # called synchronously from execute thread.
    # must be thread safe
    def _abort_acquisition(self) -> None:
        pass

    # subclasses can implement this method which is called when acquisition abort is requested.
    # this is useful if a flag/event needs to be set to break out of the acquisition loop.
    # this method may be called asynchronously from the other methods.
    # must be thread safe. it may be called from either UI thread or a thread.
    def _request_abort_acquisition(self) -> None:
        pass

    # subclasses can implement this method which is called when acquisition is suspended for higher priority acquisition.
    # if a view starts during a record, it will start in a suspended state and resume will be called without a prior
    # suspend.
    # called synchronously from execute thread.
    # must be thread safe
    def _suspend_acquisition(self) -> None:
        pass

    # subclasses can implement this method which is called when acquisition is resumed from higher priority acquisition.
    # if a view starts during a record, it will start in a suspended state and resume will be called without a prior
    # suspend.
    # called synchronously from execute thread.
    # must be thread safe
    def _resume_acquisition(self) -> None:
        pass

    # subclasses can implement this method which is called when acquisition is marked for stopping.
    # subclasses that feature a continuous mode will need implement this method so that continuous
    # mode is marked for stopping at the end of the current frame.
    # called synchronously from execute thread.
    # must be thread safe
    def _mark_acquisition(self) -> None:
        pass

    # subclasses can implement this method which is called to stop acquisition.
    # no more data is expected to be generated after this call.
    # called synchronously from execute thread.
    # must be thread safe
    def _stop_acquisition(self) -> None:
        self.stop_event.fire()

    def _safe_stop_acquisition(self) -> None:
        try:
            self._stop_acquisition()
        except Exception as e:
            if callable(self._test_acquire_exception):
                self._test_acquire_exception(e)
            else:
                import traceback
                logging.debug(f"STOP Error: {e}")
                traceback.print_exc()

    # subclasses are expected to implement this function efficiently since it will
    # be repeatedly called. in practice that means that subclasses MUST sleep (directly
    # or indirectly) unless the data is immediately available, which it shouldn't be on
    # a regular basis. it is an error for this function to return an empty list of data_elements.
    # this method can throw exceptions, it will result in the acquisition loop being aborted.
    # returns a tuple of a list of data elements.
    # called synchronously from execute thread.
    # must be thread safe
    def _acquire_data_elements(self) -> typing.Sequence[DataElementType]:
        raise NotImplementedError()


class DataChannel:
    """A channel of raw data from a hardware source.

    The channel buffer is an interface to the stream of data from a hardware source to a client of that stream.

    The client can listen to the following events from the channel:
        * data_channel_updated_event
        * data_channel_start_event
        * data_channel_stop_event

    All events will be fired on the acquisition thread.

    The client can access the following properties of the channel:
        * channel_id
        * name
        * state
        * src_channel_index
        * sub_area

    This class is used when the document model or data item is not available to be called directly. The document model
    will watch for registered hardware sources and query each hardware source for its predefined data channels and
    listen to them for start/updated/stop events. Setting data on this object will trigger a data_channel_updated_event
    which will set pending data on the data item and eventually call set_data_and_metadata on the data item from the
    main thread.
    """

    def __init__(self, hardware_source: HardwareSource, index: int, channel_id: typing.Optional[str] = None,
                 name: typing.Optional[str] = None, src_channel_index: typing.Optional[int] = None,
                 processor: typing.Optional[SumProcessor] = None) -> None:
        self.__hardware_source = hardware_source
        self.__index = index
        self.__channel_id = channel_id
        self.__name = name
        self.__src_channel_index = src_channel_index
        self.__processor = processor
        self.__start_count = 0
        self.__state: typing.Optional[str] = None
        self.__data_shape: typing.Optional[DataAndMetadata.ShapeType] = None
        self.__sub_area: typing.Optional[Geometry.IntRect] = None
        self.__dest_sub_area: typing.Optional[Geometry.IntRect] = None
        self.__data_and_metadata: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.is_dirty = False
        self.__is_error = False
        self.data_channel_updated_event = Event.Event()
        self.data_channel_start_event = Event.Event()
        self.data_channel_stop_event = Event.Event()
        self.data_channel_state_changed_event = Event.Event()

    @property
    def index(self) -> int:
        return self.__index

    @property
    def channel_id(self) -> typing.Optional[str]:
        return self.__channel_id

    @property
    def name(self) -> typing.Optional[str]:
        return self.__name

    @property
    def state(self) -> typing.Optional[str]:
        return self.__state

    @property
    def data_shape(self) -> typing.Optional[DataAndMetadata.ShapeType]:
        return self.__data_shape

    @property
    def sub_area(self) -> typing.Optional[Geometry.IntRect]:
        return self.__sub_area

    @property
    def dest_sub_area(self) -> typing.Optional[Geometry.IntRect]:
        return self.__dest_sub_area

    @property
    def src_channel_index(self) -> typing.Optional[int]:
        return self.__src_channel_index

    @property
    def processor(self) -> typing.Optional[SumProcessor]:
        return self.__processor

    @property
    def data_and_metadata(self) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
        return self.__data_and_metadata

    @property
    def is_error(self) -> bool:
        return self.__is_error

    @is_error.setter
    def is_error(self, value: bool) -> None:
        if value != self.__is_error:
            self.__is_error = value
            self.data_channel_state_changed_event.fire()

    @property
    def is_started(self) -> bool:
        return self.__start_count > 0

    def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, data_shape: typing.Optional[DataAndMetadata.ShapeType], dest_sub_area: typing.Optional[Geometry.IntRectTuple], sub_area: typing.Optional[Geometry.IntRectTuple], view_id: typing.Optional[str]) -> None:
        """Called from hardware source when new data arrives."""
        old_state = self.__state
        self.__state = state
        self.__data_shape = data_shape or data_and_metadata.data_shape

        sub_area_r = Geometry.IntRect.make(sub_area) if sub_area is not None else None
        dest_sub_area = dest_sub_area if dest_sub_area is not None else sub_area
        dest_sub_area_r = Geometry.IntRect.make(dest_sub_area) if dest_sub_area is not None else None

        self.__dest_sub_area = dest_sub_area_r
        self.__sub_area = sub_area_r

        hardware_source_id = self.__hardware_source.hardware_source_id
        channel_index = self.index
        channel_id = self.channel_id
        channel_name = self.name
        metadata = dict(copy.deepcopy(data_and_metadata.metadata))
        hardware_source_metadata: typing.Dict[str, typing.Any] = dict()
        hardware_source_metadata["hardware_source_id"] = hardware_source_id
        if channel_index is not None:
            hardware_source_metadata["channel_index"] = channel_index
        if channel_id is not None:
            hardware_source_metadata["reference_key"] = "_".join([hardware_source_id, channel_id])
            hardware_source_metadata["channel_id"] = channel_id
        else:
            hardware_source_metadata["reference_key"] = hardware_source_id
        if channel_name is not None:
            hardware_source_metadata["channel_name"] = channel_name
        if view_id:
            hardware_source_metadata["view_id"] = view_id
        metadata.setdefault("hardware_source", dict()).update(hardware_source_metadata)

        data = data_and_metadata.data
        assert data is not None
        data_shape = data_shape or data.shape
        master_data = self.__data_and_metadata.data if self.__data_and_metadata else None
        if master_data is None or (master_data.shape != data_shape and data.shape != data_shape):
            master_data = numpy.zeros(data_shape, data.dtype)
        data_matches = master_data is not None and data_shape == master_data.shape and data.dtype == master_data.dtype
        if data_matches and sub_area_r and dest_sub_area_r:
            src_rect = sub_area_r
            dst_rect = dest_sub_area_r
            if (dst_rect.top > 0 or dst_rect.left > 0 or dst_rect.bottom < master_data.shape[0] or dst_rect.right < master_data.shape[1])\
                    or (src_rect.top > 0 or src_rect.left > 0 or src_rect.bottom < data.shape[0] or src_rect.right < data.shape[1]):
                master_data[dst_rect.slice] = data[src_rect.slice]
            else:
                master_data = numpy.copy(data)  # type: ignore
        else:
            assert data.shape == data_shape
            master_data = data  # numpy.copy(data). assume data does not need a copy.
        assert master_data is not None

        data_descriptor = data_and_metadata.data_descriptor
        intensity_calibration = data_and_metadata.intensity_calibration if data_and_metadata else None
        dimensional_calibrations = data_and_metadata.dimensional_calibrations if data_and_metadata else None
        timestamp = data_and_metadata.timestamp
        new_extended_data = DataAndMetadata.new_data_and_metadata(master_data, intensity_calibration=intensity_calibration, dimensional_calibrations=dimensional_calibrations, metadata=metadata, timestamp=timestamp, data_descriptor=data_descriptor)

        self.__data_and_metadata = new_extended_data

        self.data_channel_updated_event.fire(new_extended_data)
        self.is_dirty = True

        if old_state != self.__state:
            self.data_channel_state_changed_event.fire()

    def start(self) -> None:
        """Called from hardware source when data starts streaming."""
        old_start_count = self.__start_count
        self.__start_count += 1
        if old_start_count == 0:
            self.is_error = False
            self.data_channel_start_event.fire()
            self.data_channel_state_changed_event.fire()

    def stop(self) -> None:
        """Called from hardware source when data stops streaming."""
        self.__start_count -= 1
        if self.__start_count == 0:
            self.data_channel_stop_event.fire()
            self.data_channel_state_changed_event.fire()


class Instrument(typing.Protocol):
    instrument_id: typing.Optional[str]
    def close(self) -> None: ...


class FrameParameters(typing.Protocol):
    def as_dict(self) -> typing.Dict[str, typing.Any]: ...


class HardwareSource(typing.Protocol):

    # methods

    def close(self) -> None: ...
    def start_playing(self, *args: typing.Any, **kwargs: typing.Any) -> None: ...
    def abort_playing(self, *, sync_timeout: typing.Optional[float] = None) -> None: ...
    def stop_playing(self, *, sync_timeout: typing.Optional[float] = None) -> None: ...
    def start_recording(self, sync_timeout: typing.Optional[float] = None, finished_callback_fn: typing.Optional[_FinishedCallbackType] = None, *, frame_parameters: typing.Optional[FrameParameters] = None, **kwargs: typing.Any) -> None: ...
    def abort_recording(self, sync_timeout: typing.Optional[float] = None) -> None: ...
    def stop_recording(self, sync_timeout: typing.Optional[float] = None) -> None: ...
    def get_next_xdatas_to_finish(self, timeout: typing.Optional[float] = None) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]: ...
    def get_next_xdatas_to_start(self, timeout: typing.Optional[float] = None) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]: ...
    def set_current_frame_parameters(self, frame_parameters: FrameParameters) -> None: ...

    # properties

    @property
    def features(self) -> typing.Dict[str, typing.Any]:
        return dict()

    @property
    def hardware_source_id(self) -> str:
        return str()

    @hardware_source_id.setter
    def hardware_source_id(self, value: str) -> None:
        pass

    @property
    def display_name(self) -> str:
        return str()

    @display_name.setter
    def display_name(self, value: str) -> None:
        pass

    @property
    def is_playing(self) -> bool:
        return False

    @property
    def is_recording(self) -> bool:
        return False

    @property
    def data_channel_count(self) -> int:
        return len(self.data_channels)

    @property
    def data_channels(self) -> typing.Sequence[DataChannel]:
        return list()

    # private. do not use outside of instrumentation-kit.

    data_channel_states_updated: Event.Event
    data_channel_state_changed_event: Event.Event
    xdatas_available_event: Event.Event
    abort_event: Event.Event
    acquisition_state_changed_event: Event.Event

    def add_data_channel(self, channel_id: typing.Optional[str] = None, name: typing.Optional[str] = None) -> None: ...
    def add_channel_processor(self, channel_index: int, processor: SumProcessor) -> None: ...
    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> FrameParameters: ...
    def set_channel_enabled(self, channel_index: int, enabled: bool) -> None: ...
    def data_channel_map_updated(self, data_channel_map: typing.Mapping[str, DataItem.DataItem]) -> None: ...
    def set_record_frame_parameters(self, frame_parameters: FrameParameters) -> None: ...
    def get_record_frame_parameters(self) -> FrameParameters: ...


class ConcreteHardwareSource(Observable.Observable, HardwareSource):
    """Represents a source of data and metadata frames.

    The hardware source generates data on a background thread.
    """

    def __init__(self, hardware_source_id: str, display_name: str) -> None:
        super().__init__()
        self.__hardware_source_id = hardware_source_id
        self.__display_name = display_name
        self.__data_channels: typing.List[DataChannel] = list()
        self.__data_channel_state_changed_listeners: typing.List[Event.EventListener] = list()
        self.__features: typing.Dict[str, typing.Any] = dict()
        self.data_channel_states_updated = Event.Event()
        self.xdatas_available_event = Event.Event()
        self.abort_event = Event.Event()
        self.acquisition_state_changed_event = Event.Event()
        self.data_channel_state_changed_event = Event.Event()
        self.__break_for_closing = False
        self.__acquire_thread_trigger = threading.Event()
        self.__tasks: typing.Dict[str, AcquisitionTask] = dict()
        self.__data_elements_changed_event_listeners: typing.Dict[str, Event.EventListener] = dict()
        self.__start_event_listeners: typing.Dict[str, Event.EventListener] = dict()
        self.__stop_event_listeners: typing.Dict[str, Event.EventListener] = dict()
        self.__acquire_thread: typing.Optional[threading.Thread] = threading.Thread(target=self.__acquire_thread_loop)
        self.__acquire_thread.daemon = True
        self.__acquire_thread.start()
        self._test_acquire_exception: typing.Optional[typing.Callable[[Exception], None]] = None
        self._test_start_hook: typing.Optional[typing.Callable[[], None]] = None
        self._test_acquire_hook: typing.Optional[typing.Callable[[], None]] = None

    def close(self) -> None:
        self.close_thread()
        self.__data_channel_state_changed_listeners = typing.cast(typing.Any, None)

    @property
    def features(self) -> typing.Dict[str, typing.Any]:
        return self.__features

    @property
    def hardware_source_id(self) -> str:
        return self.__hardware_source_id

    @hardware_source_id.setter
    def hardware_source_id(self, value: str) -> None:
        self.__hardware_source_id = value
        self.property_changed_event.fire("hardware_source_id")

    @property
    def display_name(self) -> str:
        return self.__display_name

    @display_name.setter
    def display_name(self, value: str) -> None:
        self.__display_name = value
        self.property_changed_event.fire("display_name")

    def close_thread(self) -> None:
        if self.__acquire_thread:
            # when overriding hardware source close, the acquisition loop may still be running
            # so nothing can be changed here that will make the acquisition loop fail.
            self.__break_for_closing = True
            self.__acquire_thread_trigger.set()
            # acquire_thread should always be non-null here, otherwise close was called twice.
            self.__acquire_thread.join()
            self.__acquire_thread = None

    def __acquire_thread_loop(self) -> None:
        # acquire_thread_trigger should be set whenever the task list change.
        while self.__acquire_thread_trigger.wait():
            self.__acquire_thread_trigger.clear()
            # record task gets highest priority
            break_for_closing = self.__break_for_closing
            suspend_task_id_list = list()
            task_id = None
            if self.__tasks.get('idle'):
                task_id = 'idle'
            if self.__tasks.get('view'):
                task_id = 'view'
                suspend_task_id_list.append('idle')
            if self.__tasks.get('record'):
                task_id = 'record'
                suspend_task_id_list.append('idle')
                suspend_task_id_list.append('view')
            if task_id:
                task = self.__tasks[task_id]
                if break_for_closing:
                    # abort the task, but execute one last time to make sure stop
                    # gets called.
                    task.abort()
                    self.abort_event.fire()
                try:
                    for suspend_task_id in suspend_task_id_list:
                        suspend_task = self.__tasks.get(suspend_task_id)
                        if suspend_task:
                            suspend_task.suspend()
                    task.execute()
                except Exception as e:
                    task.abort()
                    self.abort_event.fire()
                    if callable(self._test_acquire_exception):
                        self._test_acquire_exception(e)
                    else:
                        import traceback
                        logging.debug("{} Error: {}".format(task_id.capitalize(), e))
                        traceback.print_exc()
                if task.is_finished:
                    activity = self.__tasks[task_id].activity
                    if activity:
                        Activity.activity_finished(activity)
                    self.__tasks[task_id].activity = None
                    del self.__tasks[task_id]
                    self.__data_elements_changed_event_listeners[task_id].close()
                    del self.__data_elements_changed_event_listeners[task_id]
                    self.__start_event_listeners[task_id].close()
                    del self.__start_event_listeners[task_id]
                    self.__stop_event_listeners[task_id].close()
                    del self.__stop_event_listeners[task_id]
                    self.acquisition_state_changed_event.fire(False)
                self.__acquire_thread_trigger.set()
            if break_for_closing:
                break

    # subclasses can implement this method which is called when the data channels are updated.
    def data_channel_map_updated(self, data_channel_map: typing.Mapping[str, DataItem.DataItem]) -> None:
        pass

    # subclasses should implement this method to create a continuous-style acquisition task.
    # create the view task
    # will be called from the UI thread and should return quickly.
    def _create_acquisition_view_task(self) -> AcquisitionTask:
        raise NotImplementedError()

    # subclasses can implement this method to get notification that the view task has been changed.
    # subclasses may have a need to access the view task and this method can help keep track of the
    # current view task.
    # will be called from the UI thread and should return quickly.
    def _view_task_updated(self, view_task: typing.Optional[AcquisitionTask]) -> None:
        pass

    # subclasses should implement this method to create a non-continuous-style acquisition task.
    # create the view task
    # will be called from the UI thread and should return quickly.
    def _create_acquisition_record_task(self, *, frame_parameters: typing.Optional[FrameParameters] = None, **kwargs: typing.Any) -> AcquisitionTask:
        raise NotImplementedError()

    # subclasses can implement this method to get notification that the record task has been changed.
    # subclasses may have a need to access the record task and this method can help keep track of the
    # current record task.
    # will be called from the UI thread and should return quickly.
    def _record_task_updated(self, record_task: typing.Optional[AcquisitionTask]) -> None:
        pass

    def __data_elements_changed(self, task: AcquisitionTask, data_elements: typing.Sequence[DataElementType], view_id: typing.Optional[str], is_complete: bool, is_stopping: bool, e: typing.Optional[Exception]) -> None:
        """Called in response to a data_elements_changed event from the task.

        data_elements is a list of data_elements; may be an empty list

        data_elements optionally include 'channel_id', 'section_state', 'state', 'data_shape', 'sub_area', and 'dest_sub_area'.

        the 'channel_id' will be used to determine channel index if applicable. default will be None / channel 0.

        the 'section_state' may be 'partial' or 'complete'. default is 'partial'. it is used to indicate data should be
        returned from a grab but that the frame is still incomplete. used during partial SI.

        the 'state' may be 'partial', 'complete', or 'marked' (requested stop at end of frame). default is 'partial'. it
        is used to indicated that the entire frame is complete.

        the 'data_shape' will be used to determine the shape of the destination data. if omitted, the size of the data
        in the data element will be used.

        the 'dest_sub_area' will be used to determine destination sub-area if applicable. if data is returned in
        chunks or sections, dest sub area can be used to indicate the destination area.

        the 'sub_area' will be used to determine source sub-area if applicable. data can be returned in partial
        chunks from top to bottom with a constant width.

        beyond these three items, the data element will be converted to xdata using convert_data_element_to_data_and_metadata.
        thread safe
        """
        is_error = e is not None
        xdatas: typing.List[typing.Optional[DataAndMetadata.DataAndMetadata]] = list()
        data_channels: typing.List[DataChannel] = list()
        for data_element in data_elements:
            assert data_element is not None
            channel_id = data_element.get("channel_id")
            # find channel_index for channel_id
            channel_index = next((data_channel.index for data_channel in self.__data_channels if data_channel.channel_id == channel_id), 0)
            data_and_metadata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
            # data_and_metadata data may still point to low level code memory at this point.
            channel_state = data_element.get("state", "complete" if not is_error else "error")
            if channel_state != "complete" and is_stopping and not is_error:
                channel_state = "marked"
            data_shape = data_element.get("data_shape")
            dest_sub_area = data_element.get("dest_sub_area")
            sub_area = data_element.get("sub_area")
            data_channel = self.__data_channels[channel_index]
            # data_channel.update will make a copy of the data_and_metadata
            data_channel.update(data_and_metadata, channel_state, data_shape, dest_sub_area, sub_area, view_id)
            data_channels.append(data_channel)
            xdatas.append(data_channel.data_and_metadata)
        # update channel buffers with processors
        for data_channel in self.__data_channels:
            src_channel_index = data_channel.src_channel_index
            if src_channel_index is not None:
                src_data_channel = self.__data_channels[src_channel_index]
                if src_data_channel in data_channels:
                    src_data_and_metadata = src_data_channel.data_and_metadata
                    data_channel_processor = data_channel.processor
                    if not is_error:
                        if data_channel_processor and src_data_and_metadata and src_data_channel.is_dirty and src_data_channel.state == "complete":
                            processed_data_and_metadata = data_channel_processor.process(src_data_and_metadata)
                            data_channel.update(processed_data_and_metadata, "complete", None, None, None, view_id)
                    else:
                        assert data_channel.data_and_metadata
                        data_channel.update(data_channel.data_and_metadata, "error", None, None, None, view_id)
                    data_channels.append(data_channel)
                    xdatas.append(data_channel.data_and_metadata)
        # all channel buffers are clean now
        for data_channel in self.__data_channels:
            data_channel.is_error = is_error
            data_channel.is_dirty = False

        self.data_channel_states_updated.fire(data_channels)

        if is_complete:
            # xdatas are may still be pointing to memory in low level code here
            # send promises which will give access to the data, but not copy it unless it is used.
            data_promises = [DataAndMetadataPromise(xdata) for xdata in xdatas]
            self.xdatas_available_event.fire(data_promises)
            # hack to allow record to know when its data is finished
            if callable(task.finished_callback_fn):
                task.finished_callback_fn(data_promises)

        if is_error:
            from nion.swift.model import Notification
            Notification.notify(Notification.Notification("nion.acquisition.error", "\N{WARNING SIGN} Acquisition",
                                                          "Acquisition Failed", str(e)))

    def __start(self) -> None:
        for data_channel in self.__data_channels:
            data_channel.start()

    def __stop(self) -> None:
        for data_channel in self.__data_channels:
            data_channel.stop()
        self.data_channel_states_updated.fire(list())

    # return whether task is running
    def is_task_running(self, task_id: str) -> bool:
        return task_id in self.__tasks

    # call this to start the task running
    # not thread safe
    def start_task(self, task_id: str, task: AcquisitionTask) -> None:
        assert not task in self.__tasks.values()
        assert not task_id in self.__tasks
        assert task_id in ('idle', 'view', 'record')
        self.__data_elements_changed_event_listeners[task_id] = task.data_elements_changed_event.listen(functools.partial(self.__data_elements_changed, task))
        self.__start_event_listeners[task_id] = task.start_event.listen(self.__start)
        self.__stop_event_listeners[task_id] = task.stop_event.listen(self.__stop)
        task.activity = Activity.Activity(self.hardware_source_id + "_" + task_id, f"{self.display_name} ({task_id})")
        Activity.append_activity(task.activity)
        self.__tasks[task_id] = task
        # TODO: sync the thread start by waiting for an event on the task which gets set when the acquire thread starts executing the task
        self.__acquire_thread_trigger.set()
        self.acquisition_state_changed_event.fire(True)

    # call this to stop task immediately
    # not thread safe
    def abort_task(self, task_id: str) -> None:
        task = self.__tasks.get(task_id)
        assert task is not None
        task.abort()
        self.abort_event.fire()

    # call this to stop acquisition gracefully
    # not thread safe
    def stop_task(self, task_id: str) -> None:
        task = self.__tasks.get(task_id)
        assert task is not None
        task.stop()

    # return whether acquisition is running
    @property
    def is_playing(self) -> bool:
        return self.is_task_running('view')

    # call this to start acquisition
    # not thread safe
    def start_playing(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        if not self.is_playing:
            view_task = self._create_acquisition_view_task()
            view_task._test_start_hook = self._test_start_hook
            view_task._test_acquire_hook = self._test_acquire_hook
            self._view_task_updated(view_task)
            self.start_task('view', view_task)
        if "sync_timeout" in kwargs:
            start = time.time()
            while not self.is_playing:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < float(kwargs["sync_timeout"])

    # call this to stop acquisition immediately
    # not thread safe
    def abort_playing(self, *, sync_timeout: typing.Optional[float] = None) -> None:
        if self.is_playing:
            self.abort_task('view')
            self._view_task_updated(None)
        if sync_timeout is not None:
            start = time.time()
            while self.is_playing:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < float(sync_timeout)

    # call this to stop acquisition gracefully
    # not thread safe
    def stop_playing(self, *, sync_timeout: typing.Optional[float] = None) -> None:
        if self.is_playing:
            self.stop_task('view')
            self._view_task_updated(None)
        if sync_timeout is not None:
            start = time.time()
            while self.is_playing:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < float(sync_timeout)

    # return whether acquisition is running
    @property
    def is_recording(self) -> bool:
        return self.is_task_running('record')

    # call this to start acquisition
    # thread safe
    def start_recording(self, sync_timeout: typing.Optional[float] = None, finished_callback_fn: typing.Optional[_FinishedCallbackType] = None, *, frame_parameters: typing.Optional[FrameParameters] = None, **kwargs: typing.Any) -> None:
        if not self.is_recording:
            record_task = self._create_acquisition_record_task(frame_parameters=frame_parameters)
            old_finished_callback_fn = record_task.finished_callback_fn

            def finished(data_promises: typing.Sequence[DataAndMetadataPromise]) -> None:
                if callable(old_finished_callback_fn):
                    old_finished_callback_fn(data_promises)
                if callable(finished_callback_fn):
                    finished_callback_fn(data_promises)

            record_task.finished_callback_fn = finished
            self._record_task_updated(record_task)
            self.start_task('record', record_task)
        if sync_timeout is not None:
            start = time.time()
            while not self.is_recording:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < float(sync_timeout)

    # call this to stop acquisition immediately
    # not thread safe
    def abort_recording(self, sync_timeout: typing.Optional[float] = None) -> None:
        if self.is_recording:
            self.abort_task('record')
            self._record_task_updated(None)
        if sync_timeout is not None:
            start = time.time()
            while self.is_recording:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < float(sync_timeout)

    # call this to stop acquisition gracefully
    # not thread safe
    def stop_recording(self, sync_timeout: typing.Optional[float] = None) -> None:
        if self.is_recording:
            self.stop_task('record')
            self._record_task_updated(None)
        if sync_timeout is not None:
            start = time.time()
            while self.is_recording:
                time.sleep(0.01)  # 10 msec
                assert time.time() - start < float(sync_timeout)

    def get_next_xdatas_to_finish(self, timeout: typing.Optional[float] = None) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        new_data_event = threading.Event()
        new_xdatas: typing.List[typing.Optional[DataAndMetadata.DataAndMetadata]] = list()

        def receive_new_xdatas(data_promises: typing.Sequence[DataAndMetadataPromise]) -> None:
            new_xdatas[:] = [data_promise.xdata for data_promise in data_promises]
            new_data_event.set()

        def abort() -> None:
            new_data_event.set()

        with contextlib.closing(self.xdatas_available_event.listen(receive_new_xdatas)):
            with contextlib.closing(self.abort_event.listen(abort)):
                # wait for the current frame to finish
                if not new_data_event.wait(timeout):
                    raise Exception("Could not start data_source " + str(self.hardware_source_id))

                return new_xdatas

    def get_next_xdatas_to_start(self, timeout: typing.Optional[float] = None) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        new_data_event = threading.Event()
        new_xdatas: typing.List[typing.Optional[DataAndMetadata.DataAndMetadata]] = list()

        def receive_new_xdatas(data_promises: typing.Sequence[DataAndMetadataPromise]) -> None:
            new_xdatas[:] = [data_promise.xdata for data_promise in data_promises]
            new_data_event.set()

        def abort() -> None:
            new_data_event.set()

        with contextlib.closing(self.xdatas_available_event.listen(receive_new_xdatas)):
            with contextlib.closing(self.abort_event.listen(abort)):
                # wait for the current frame to finish
                if not new_data_event.wait(timeout):
                    raise Exception("Could not start data_source " + str(self.hardware_source_id))

                new_data_event.clear()

                if len(new_xdatas) > 0:
                    new_data_event.wait(timeout)

                return new_xdatas

    @property
    def data_channel_count(self) -> int:
        return len(self.__data_channels)

    @property
    def data_channels(self) -> typing.Sequence[DataChannel]:
        return self.__data_channels

    def __data_channel_state_changed(self, data_channel: DataChannel) -> None:
        self.data_channel_state_changed_event.fire(data_channel)

    def add_data_channel(self, channel_id: typing.Optional[str] = None, name: typing.Optional[str] = None) -> None:
        data_channel = DataChannel(self, len(self.__data_channels), channel_id, name)
        self.__data_channels.append(data_channel)
        self.__data_channel_state_changed_listeners.append(data_channel.data_channel_state_changed_event.listen(weak_partial(ConcreteHardwareSource.__data_channel_state_changed, self, data_channel)))

    def add_channel_processor(self, channel_index: int, processor: SumProcessor) -> None:
        data_channel = DataChannel(self, len(self.__data_channels), processor.processor_id, None, channel_index, processor)
        self.__data_channels.append(data_channel)
        self.__data_channel_state_changed_listeners.append(data_channel.data_channel_state_changed_event.listen(weak_partial(ConcreteHardwareSource.__data_channel_state_changed, self, data_channel)))

    def get_property(self, name: str) -> typing.Any:
        return getattr(self, name)

    def set_property(self, name: str, value: typing.Any) -> None:
        setattr(self, name, value)

    # deprecated. for Facade use only.
    def create_view_task(self, frame_parameters: typing.Optional[FrameParameters] = None,
                         channels_enabled: typing.Optional[typing.Sequence[bool]] = None,
                         buffer_size: int = 1) -> ViewTask:
        return ViewTask(self, frame_parameters, channels_enabled, buffer_size)

    def get_api(self, version: str) -> typing.Any:
        actual_version = "1.0.0"
        if Utility.compare_versions(version, actual_version) > 0:
            raise NotImplementedError("Hardware Source API requested version %s is greater than %s." % (version, actual_version))

        class HardwareSourceFacade:
            def __init__(self) -> None:
                pass

        return HardwareSourceFacade()

    # some dummy methods to pass type checking. the hardware source needs to be refactored.

    def set_current_frame_parameters(self, frame_parameters: FrameParameters) -> None:
        pass

    def get_frame_parameters_from_dict(self, d: typing.Mapping[str, typing.Any]) -> FrameParameters:
        raise NotImplementedError()

    def set_channel_enabled(self, channel_index: int, enabled: bool) -> None:
        pass


class DelegateAcquisitionTaskProtocol(typing.Protocol):
    def start_acquisition(self) -> None: ...
    def stop_acquisition(self) -> None: ...
    def acquire_data_and_metadata(self) -> typing.Optional[DataAndMetadata.DataAndMetadata]: ...


# used for Facade backwards compatibility
class DelegateAcquisitionTask(AcquisitionTask):

    def __init__(self, delegate: DelegateAcquisitionTaskProtocol, hardware_source_id: str, hardware_source_name: str) -> None:
        super().__init__(True)
        self.__delegate = delegate
        self.__hardware_source_id = hardware_source_id
        self.__hardware_source_name = hardware_source_name

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self.__delegate.start_acquisition()
        return True

    def _acquire_data_elements(self) -> typing.Sequence[DataElementType]:
        data_and_metadata = self.__delegate.acquire_data_and_metadata()
        if data_and_metadata:
            data_element = {
                "version": 1,
                "data": data_and_metadata.data,
                "properties": {
                    "hardware_source_name": self.__hardware_source_name,
                    "hardware_source_id": self.__hardware_source_id,
                }
            }
            return [data_element]
        return list()

    def _stop_acquisition(self) -> None:
        self.__delegate.stop_acquisition()
        super()._stop_acquisition()


# used for Facade backwards compatibility
class DelegateHardwareSource(ConcreteHardwareSource):

    def __init__(self, delegate: DelegateAcquisitionTaskProtocol, hardware_source_id: str, hardware_source_name: str) -> None:
        super().__init__(hardware_source_id, hardware_source_name)
        self.__delegate = delegate
        self.features["is_video"] = True
        self.add_data_channel()

    def _create_acquisition_view_task(self) -> DelegateAcquisitionTask:
        return DelegateAcquisitionTask(self.__delegate, self.hardware_source_id, self.display_name)


class SumProcessor(Observable.Observable):
    def __init__(self, bounds: Geometry.FloatRect, processor_id: typing.Optional[str] = None, label: typing.Optional[str] = None) -> None:
        super().__init__()
        self.__bounds = bounds
        self.__processor_id = processor_id or "summed"
        self.__label = label or _("Summed")
        self.__crop_graphic: typing.Optional[Graphics.RectangleGraphic] = None
        self.__crop_listener: typing.Optional[Event.EventListener] = None
        self.__remove_listener: typing.Optional[Event.EventListener] = None
        self.__data_item_reference_changed_event_listener: typing.Optional[Event.EventListener] = None

    @property
    def label(self) -> str:
        return self.__label

    @property
    def processor_id(self) -> str:
        return self.__processor_id

    @property
    def bounds(self) -> Geometry.FloatRect:
        return self.__bounds

    @bounds.setter
    def bounds(self, value: Geometry.FloatRectTuple) -> None:
        bounds = Geometry.FloatRect.make(value)
        if self.__bounds != bounds:
            self.__bounds = bounds
            self.notify_property_changed("bounds")
            if self.__crop_graphic:
                self.__crop_graphic.bounds = bounds

    def process(self, data_and_metadata: DataAndMetadata.DataAndMetadata) -> DataAndMetadata.DataAndMetadata:
        if data_and_metadata.datum_dimension_count > 1 and data_and_metadata.data_shape[0] > 1:
            cropped_xdata = Core.function_crop(data_and_metadata, self.__bounds.as_tuple())
            assert cropped_xdata
            summed = Core.function_sum(cropped_xdata, 0)
            assert summed
            summed._set_metadata(data_and_metadata.metadata)
            return summed
        elif len(data_and_metadata.data_shape) > 1:
            summed = Core.function_sum(data_and_metadata, 0)
            assert summed
            summed._set_metadata(data_and_metadata.metadata)
            return summed
        else:
            return copy.deepcopy(data_and_metadata)

    def connect_data_item_reference(self, data_item_reference: DocumentModel.DocumentModel.DataItemReference) -> None:
        """Connect to the data item reference, creating a crop graphic if necessary.

        If the data item reference does not yet have an associated data item, add a
        listener and wait for the data item to be set, then connect.
        """
        display_item = data_item_reference.display_item
        data_item = display_item.data_item if display_item else None
        if data_item and display_item:
            self.__connect_display(display_item)
        else:
            def data_item_reference_changed() -> None:
                if self.__data_item_reference_changed_event_listener:
                    self.__data_item_reference_changed_event_listener.close()
                    self.__data_item_reference_changed_event_listener = None
                self.connect_data_item_reference(data_item_reference)  # ugh. recursive mess.
            self.__data_item_reference_changed_event_listener = data_item_reference.data_item_reference_changed_event.listen(data_item_reference_changed)

    def __connect_display(self, display_item: DisplayItem.DisplayItem) -> None:
        assert threading.current_thread() == threading.main_thread()
        crop_graphic: typing.Optional[Graphics.RectangleGraphic] = None
        for graphic in display_item.graphics:
            if graphic.graphic_id == self.__processor_id and isinstance(graphic, Graphics.RectangleGraphic):
                crop_graphic = graphic
                break
        def close_all() -> None:
            self.__crop_graphic = None
            if self.__crop_listener:
                self.__crop_listener.close()
                self.__crop_listener = None
            if self.__remove_listener:
                self.__remove_listener.close()
                self.__remove_listener = None
        if not crop_graphic:
            close_all()
            crop_graphic = Graphics.RectangleGraphic()
            crop_graphic.bounds = self.bounds
            crop_graphic.is_bounds_constrained = True
            crop_graphic.graphic_id = self.__processor_id
            crop_graphic.label = _("Crop")
            display_item.add_graphic(crop_graphic)
        if not self.__crop_listener:
            def property_changed(k: str) -> None:
                if k == "bounds" and crop_graphic:
                    self.bounds = crop_graphic.bounds
            def graphic_removed(k: str, v: Graphics.RectangleGraphic, i: int) -> None:
                if v == crop_graphic:
                    close_all()
            self.__crop_listener = crop_graphic.property_changed_event.listen(property_changed)
            self.__remove_listener = display_item.item_removed_event.listen(graphic_removed)
            self.__crop_graphic = crop_graphic


class ViewTask:

    def __init__(self, hardware_source: HardwareSource, frame_parameters: typing.Optional[FrameParameters], channels_enabled: typing.Optional[typing.Sequence[bool]], buffer_size: int):
        self.__hardware_source = hardware_source
        self.__was_playing = self.__hardware_source.is_playing
        if frame_parameters:
            self.__hardware_source.set_current_frame_parameters(frame_parameters)
        if channels_enabled is not None:
            for channel_index, channel_enabled in enumerate(channels_enabled):
                self.__hardware_source.set_channel_enabled(channel_index, channel_enabled)
        if not self.__was_playing:
            self.__hardware_source.start_playing()
        self.__data_channel_buffer = DataChannelBuffer(self.__hardware_source.data_channels, buffer_size)
        self.__data_channel_buffer.start()
        self.on_will_start_frame = None  # prepare the hardware here
        self.on_did_finish_frame = None  # restore the hardware here, modify the data_and_metadata here

    def close(self) -> None:
        """Close the task. Must be called when the task is no longer needed."""
        self.__data_channel_buffer.stop()
        self.__data_channel_buffer.close()
        self.__data_channel_buffer = typing.cast(typing.Any, None)
        if not self.__was_playing:
            self.__hardware_source.stop_playing()

    def grab_immediate(self) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab list of data/metadata from the task.

        This method will return immediately if data is available.
        """
        return self.__data_channel_buffer.grab_latest()

    def grab_next_to_finish(self) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab list of data/metadata from the task.

        This method will wait until the current frame completes.
        """
        return self.__data_channel_buffer.grab_next()

    def grab_next_to_start(self) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab list of data/metadata from the task.

        This method will wait until the current frame completes and the next one finishes.
        """
        return self.__data_channel_buffer.grab_following()

    def grab_earliest(self) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab list of data/metadata from the task.

        This method will return the earliest item in the buffer or wait for the next one to finish.
        """
        return self.__data_channel_buffer.grab_earliest()


class RecordTask:
    """Run acquisition in a thread and record the result."""

    def __init__(self, hardware_source: HardwareSource, frame_parameters: FrameParameters) -> None:
        self.__hardware_source = hardware_source

        assert not self.__hardware_source.is_recording

        self.__data_and_metadata_list: typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]] = list()
        # synchronize start of thread; if this sync doesn't occur, the task can be closed before the acquisition
        # is started. in that case a deadlock occurs because the abort doesn't apply and the thread is waiting
        # for the acquisition.
        self.__recording_started = threading.Event()

        def record_thread() -> None:
            self.__hardware_source.set_record_frame_parameters(frame_parameters)
            self.__hardware_source.start_recording()
            self.__recording_started.set()
            self.__data_and_metadata_list = self.__hardware_source.get_next_xdatas_to_finish()
            self.__hardware_source.stop_recording(sync_timeout=3.0)

        self.__thread = threading.Thread(target=record_thread)
        self.__thread.start()
        self.__recording_started.wait()

    def close(self) -> None:
        if self.__thread.is_alive():
            self.__hardware_source.abort_recording()
            self.__thread.join()
        self.__data_and_metadata_list = typing.cast(typing.Any, None)
        self.__recording_started = typing.cast(typing.Any, None)

    @property
    def is_finished(self) -> bool:
        return not self.__thread.is_alive()

    def grab(self) -> typing.Sequence[typing.Optional[DataAndMetadata.DataAndMetadata]]:
        self.__thread.join()
        return self.__data_and_metadata_list

    def cancel(self) -> None:
        self.__hardware_source.abort_recording()


@contextlib.contextmanager
def get_data_generator_by_id(hardware_source_id: str, sync: bool = True) -> typing.Iterator[typing.Callable[[], typing.Optional[_NDArray]]]:
    """
        Return a generator for data.

        :param bool sync: whether to wait for current frame to finish then collect next frame

        NOTE: a new ndarray is created for each call.
    """
    hardware_source = HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
    def get_last_data() -> typing.Optional[_NDArray]:
        if hardware_source:
            xdatas = hardware_source.get_next_xdatas_to_finish()
            if xdatas:
                first_xdata = xdatas[0]
                first_data = first_xdata.data if first_xdata else None
                return first_data.copy() if first_data is not None else None
        return None
    yield get_last_data


class DataChannelBuffer:
    """A fixed size buffer for a list of hardware source data channels.

    The buffer takes care of waiting until all channels in the list have produced
    a full frame of data, then stores it if it matches criteria (for instance every
    n seconds). Clients can retrieve earliest or latest data.

    Possible uses: record every frame, record every nth frame, record frame periodically,
      frame averaging, spectrum imaging.
    """

    class State(enum.Enum):
        idle = 0
        started = 1
        paused = 2

    def __init__(self, data_channels: typing.Sequence[DataChannel], buffer_size: int = 16) -> None:
        self.__state_lock = threading.RLock()
        self.__state = DataChannelBuffer.State.idle
        self.__buffer_size = buffer_size
        self.__buffer_lock = threading.RLock()
        self.__buffer: typing.List[typing.List[DataAndMetadata.DataAndMetadata]] = list()
        self.__done_events: typing.List[threading.Event] = list()
        self.__active_channel_ids: typing.Set[typing.Optional[str]] = set()
        self.__latest: typing.Dict[typing.Optional[str], DataAndMetadata.DataAndMetadata] = dict()
        self.__data_channel_updated_listeners: typing.List[Event.EventListener] = list()
        self.__data_channel_start_listeners: typing.List[Event.EventListener] = list()
        self.__data_channel_stop_listeners: typing.List[Event.EventListener] = list()
        self.__data_channels = list(data_channels)
        for data_channel in self.__data_channels:
            data_channel_updated_listener = data_channel.data_channel_updated_event.listen(functools.partial(self.__data_channel_updated, data_channel))
            self.__data_channel_updated_listeners.append(data_channel_updated_listener)
            data_channel_start_listener = data_channel.data_channel_start_event.listen(functools.partial(self.__data_channel_start, data_channel))
            self.__data_channel_start_listeners.append(data_channel_start_listener)
            data_channel_stop_listener = data_channel.data_channel_stop_event.listen(functools.partial(self.__data_channel_stop, data_channel))
            self.__data_channel_stop_listeners.append(data_channel_stop_listener)
            if data_channel.is_started:
                self.__active_channel_ids.add(data_channel.channel_id)

    def close(self) -> None:
        for listener in self.__data_channel_updated_listeners:
            listener.close()
        for listener in self.__data_channel_start_listeners:
            listener.close()
        for listener in self.__data_channel_stop_listeners:
            listener.close()
        self.__data_channel_updated_listeners = typing.cast(typing.Any, None)
        self.__data_channel_start_listeners = typing.cast(typing.Any, None)
        self.__data_channel_stop_listeners = typing.cast(typing.Any, None)

    def __data_channel_updated(self, data_channel: DataChannel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
        if self.__state == DataChannelBuffer.State.started:
            if data_channel.state == "complete":
                with self.__buffer_lock:
                    self.__latest[data_channel.channel_id] = data_and_metadata
                    if set(self.__latest.keys()).issuperset(self.__active_channel_ids):
                        data_and_metadata_list = list()
                        for data_channel in self.__data_channels:
                            if data_channel.channel_id in self.__latest:
                                data_and_metadata_list.append(copy.deepcopy(self.__latest[data_channel.channel_id]))
                        self.__buffer.append(data_and_metadata_list)
                        self.__latest = dict()
                        if len(self.__buffer) > self.__buffer_size:
                            self.__buffer.pop(0)
                        for done_event in self.__done_events:
                            done_event.set()
                        self.__done_events = list()

    def __data_channel_start(self, data_channel: DataChannel) -> None:
        self.__active_channel_ids.add(data_channel.channel_id)

    def __data_channel_stop(self, data_channel: DataChannel) -> None:
        self.__active_channel_ids.remove(data_channel.channel_id)

    def grab_latest(self, timeout: typing.Optional[float] = None) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab the most recent data from the buffer, blocking until one is available. Clear earlier data."""
        timeout = timeout if timeout is not None else 10.0
        with self.__buffer_lock:
            if len(self.__buffer) == 0:
                done_event = threading.Event()
                self.__done_events.append(done_event)
                self.__buffer_lock.release()
                done = done_event.wait(timeout)
                self.__buffer_lock.acquire()
                if not done:
                    raise Exception("Could not grab latest.")
            result = self.__buffer[-1]
            self.__buffer = list()
            return result

    def grab_earliest(self, timeout: typing.Optional[float] = None) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab the earliest data from the buffer, blocking until one is available."""
        timeout = timeout if timeout is not None else 10.0
        with self.__buffer_lock:
            if len(self.__buffer) == 0:
                done_event = threading.Event()
                self.__done_events.append(done_event)
                self.__buffer_lock.release()
                done = done_event.wait(timeout)
                self.__buffer_lock.acquire()
                if not done:
                    raise Exception("Could not grab latest.")
            return self.__buffer.pop(0)

    def grab_next(self, timeout: typing.Optional[float] = None) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab the next data to finish from the buffer, blocking until one is available."""
        with self.__buffer_lock:
            self.__buffer = list()
        return self.grab_latest(timeout)

    def grab_following(self, timeout: typing.Optional[float] = None) -> typing.Sequence[DataAndMetadata.DataAndMetadata]:
        """Grab the next data to start from the buffer, blocking until one is available."""
        self.grab_next(timeout)
        return self.grab_next(timeout)

    def start(self) -> None:
        """Start recording.

        Thread safe and UI safe."""
        with self.__state_lock:
            self.__state = DataChannelBuffer.State.started

    def pause(self) -> None:
        """Pause recording.

        Thread safe and UI safe."""
        with self.__state_lock:
            if self.__state == DataChannelBuffer.State.started:
                self.__state = DataChannelBuffer.State.paused

    def resume(self) -> None:
        """Resume recording after pause.

        Thread safe and UI safe."""
        with self.__state_lock:
            if self.__state == DataChannelBuffer.State.paused:
                self.__state = DataChannelBuffer.State.started

    def stop(self) -> None:
        """Stop or abort recording.

        Thread safe and UI safe."""
        with self.__state_lock:
            self.__state = DataChannelBuffer.State.idle


class MetadataDisplayComponent:

    # populate the dictionary d with the keys 'frame_index' and 'info_items' if they can be extracted
    # from the metadata
    def populate(self, d: typing.Dict[str, typing.Any], metadata: DataAndMetadata.MetadataType) -> None:
        frame_index = Metadata.get_metadata_value(metadata, "stem.hardware_source.frame_number")
        if frame_index is not None:
            d["frame_index"] = frame_index

        valid_rows = Metadata.get_metadata_value(metadata, "stem.hardware_source.valid_rows")
        if valid_rows is not None:
            d["valid_rows"] = valid_rows

        info_items = list()
        voltage = Metadata.get_metadata_value(metadata, "stem.high_tension")
        if voltage is not None:
            units = "V"
            if voltage % 1000 == 0:
                voltage = voltage // 1000
                units = "kV"
            info_items.append(f"{voltage} {units}")

        hardware_source_name = Metadata.get_metadata_value(metadata, "stem.hardware_source.name")
        if hardware_source_name:
            info_items.append(str(hardware_source_name))

        if info_items:
            d["info_items"] = info_items


def matches_hardware_source(hardware_source_id: str, channel_id: typing.Optional[str], document_model: DocumentModel.DocumentModel, data_item: DataItem.DataItem) -> bool:
    if not document_model.get_data_item_computation(data_item):
        hardware_source_metadata = data_item.metadata.get("hardware_source", dict())
        data_item_hardware_source_id = hardware_source_metadata.get("hardware_source_id")
        data_item_channel_id = hardware_source_metadata.get("channel_id")
        return data_item.category == "temporary" and hardware_source_id == data_item_hardware_source_id and channel_id == data_item_channel_id
    return False


hardware_source_bridges: typing.Dict[uuid.UUID, HardwareSourceBridge] = dict()


def handle_component_registered(component: typing.Any, component_types: typing.Set[str]) -> None:
    if "document_model" in component_types:
        assert component.uuid not in hardware_source_bridges
        hardware_source_bridges[component.uuid] = HardwareSourceManager().register_document_model(component)


def handle_component_unregistered(component: typing.Any, component_types: typing.Set[str]) -> None:
    if "application" in component_types:
        HardwareSourceManager().close()
    elif "document_model" in component_types:
        hardware_source_bridges.pop(component.uuid).close()


_component_registered_event_listener = None
_component_unregistered_event_listener = None


def run() -> None:
    Registry.register_component(MetadataDisplayComponent(), {"metadata_display"})
    Registry.register_component(HardwareSourceManager(), {"hardware_source_manager"})
    global _component_registered_event_listener
    global _component_unregistered_event_listener
    _component_registered_event_listener = Registry.listen_component_registered_event(handle_component_registered)
    _component_unregistered_event_listener = Registry.listen_component_unregistered_event(handle_component_unregistered)

def stop() -> None:
    Registry.unregister_component(Registry.get_component("metadata_display"), {"metadata_display"})
    Registry.unregister_component(Registry.get_component("hardware_source_manager"), {"hardware_source_manager"})
    global _component_registered_event_listener
    global _component_unregistered_event_listener
    _component_registered_event_listener = None
    _component_unregistered_event_listener = None
