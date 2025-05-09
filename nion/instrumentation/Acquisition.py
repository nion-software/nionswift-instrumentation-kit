"""Acquisition pipeline.

Devices produce a set of data streams specific to the device.

Camera devices produce a data stream of 2D or 1D data.

Scan devices produce a data stream of 2D or 1D data.

Operators take data streams as inputs and produce different data streams.

Operators may also control hardware.

Actions control hardware.

The goal is to set up an acquisition graph that controls hardware and produces output data streams.

Multiple devices can be configured; but how they are synchronized must be described.

Data streams may be used in multiple places in a graph, but only active in one spot at a time.

Data streams should only be active between calls to start and finish.

- scan position.
    - instrument parameters.
    - scan pattern.
    - modifications (drift).
    - processing.
    - repeat.
- detectors involved.
    - synchronized.
    - sequence.
    - passive.
    - processing.
- synchronization scheme.
- data products.
- metadata products.

Defines processing of data arriving from detectors.

Current use cases:

- Camera (CCD + DED) view mode
    - Full frame
    - Vertical summed frame
- Scan view mode with multiple channels
    - Sub scan
    - Line scan
- Video camera
- Camera (CCD + DED) record
- Scan record
- Scan record N frames into sequence
- Camera record N consecutive frames
- Synchronized acquisition
- Multi acquire
- Multi acquire spectrum image
- Multiple shift EELS
- Marcel scan acquisition

Future use cases:

- X-ray
- Synchronized acquisition with multiple cameras
- Synchronized acquisition with alternative master
- Multiple passes per scan line
- Tilt or focus series
- Scan within scan (two scan units?)

Options:

- Full scan vs sub-scan, switching
- Live FoV, rotation, size changes
- Partial data
- Flyback cropping
- Requested vs actual parameters
- Software binning
- Drift correction
- Summing, 2D, 1D, along various axes
- Alignment, 2D, 1D, and 1D within 2D, along various axes (sequence/collection).
- Reduction (2D -> 1D; 1D -> 0D; 2D -> 0D)
- Masking options
- Progress bars
- Pause
- Metadata
- Acquisition description
- Data sets, connected items
- Movies, buffering, averaging, windowing
- Saturation indicator
- Detect various error conditions (no data, missing frames, low speed, etc.)
- Associated dark and gain images
- Show scan position during slower scans
- Clean camera mode - one or more cleared frames then the data
- Multi-threaded processing

Scan position:

- Epochs are overall repeats
- Sub areas are positions within scan reference frame
- Areas are instrument positions or sub areas
- Batch areas are batches of sub areas - grid, stack, overlapping, etc.
- Actions can occur at every pixel, every line, every batch, every area, every epoch

Devices:

- Devices produce output that must be compatible with acquisition description
- Devices should know about immediate "layer above" processing and have option to implement it themselves

"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import datetime
import enum
import functools
import gettext
import logging
import time
import typing
import uuid
import weakref

import numpy
import numpy.typing

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.instrumentation import AcquisitionPreferences
from nion.instrumentation import stem_controller as STEMController
from nion.swift.model import DocumentModel
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Registry

if typing.TYPE_CHECKING:
    from nion.instrumentation import DriftTracker

_ = gettext.gettext

ShapeType = typing.Sequence[int]
SliceType = typing.Tuple[slice, ...]
SliceListType = typing.Sequence[SliceType]
ChannelSegment = str

_NDArray = numpy.typing.NDArray[typing.Any]


class SessionManager:
    def __init__(self) -> None:
        self.__indexes = dict[uuid.UUID, int]()

    def begin_acquisition(self, document_model: DocumentModel.DocumentModel) -> None:
        document_model_uuid = document_model.uuid
        self.__indexes[document_model_uuid] = self.__get_index(document_model) + 1

    def update_session_metadata_dict(self, document_model: DocumentModel.DocumentModel, session_metadata_dict: typing.Dict[str, typing.Any]) -> None:
        session_metadata_dict["project_acquisition_index"] = self.__get_index(document_model)

    def get_project_acquisition_index(self, document_model: DocumentModel.DocumentModel) -> int:
        return self.__get_index(document_model)

    def __get_index(self, document_model: DocumentModel.DocumentModel) -> int:
        # return the current index; but find it if document_model hasn't been indexed yet.
        document_model_uuid = document_model.uuid
        if not document_model_uuid in self.__indexes:
            next_index = 0
            for data_item in document_model.data_items:
                data_item_index = data_item.session_data.get("project_acquisition_index")
                next_index = max(data_item_index if data_item_index else 0, next_index)
            self.__indexes[document_model_uuid] = next_index
        return self.__indexes[document_model_uuid]


session_manager = SessionManager()


class Channel:
    def __init__(self, *segments: ChannelSegment) -> None:
        self.__segments: typing.List[ChannelSegment] = list(segments)

    def __repr__(self) -> str:
        return ".".join(self.__segments)

    def __hash__(self) -> int:
        return sum(hash(s) for s in self.__segments)

    def __eq__(self, other: typing.Any) -> bool:
        return isinstance(other, self.__class__) and other.segments == self.segments

    @property
    def segments(self) -> typing.Sequence[str]:
        return list(self.__segments)

    def join_segment(self, segment: ChannelSegment) -> Channel:
        return Channel(*(self.__segments + [segment]))

    @property
    def parent(self) -> Channel:
        return Channel(*(self.__segments[:-1]))


class DataStreamStateEnum(enum.Enum):
    PARTIAL = 1
    COMPLETE = 2


def index_slice(n: int) -> slice:
    return slice(n, n + 1)


def offset_slice(s: slice, n: int) -> slice:
    """Add n to start/stop of slice s."""
    return slice(s.start + n, s.stop + n, s.step)


def get_slice_shape(slices: SliceType, shape: ShapeType) -> ShapeType:
    assert slices[0].start is not None
    assert slices[0].stop is not None
    assert len(slices) == len(shape)
    result_shape = list()
    for slice, l in zip(slices, shape):
        if slice.start is not None and slice.stop is not None:
            result_shape.append(slice.stop - slice.start)
        else:
            result_shape.append(l)
    return tuple(result_shape)


def expand_shape(shape: ShapeType) -> int:
    return int(numpy.prod(shape, dtype=numpy.int64))


def get_slice_rect(slices: SliceType, shape: ShapeType) -> Geometry.IntRect:
    assert slices[0].start is not None
    assert slices[0].stop is not None
    assert len(slices) == len(shape)
    assert len(slices) == 2
    top = slices[0].start
    bottom = slices[0].stop
    left = slices[1].start if slices[1].start is not None else 0
    right = slices[1].stop if slices[1].stop is not None else shape[1]
    return Geometry.IntRect.from_tlbr(top, left, bottom, right)


def better_ravel_index(index: ShapeType, shape: ShapeType) -> int:
    # ravel index only works for indexes less than the dimension size.
    # this version gives the index for indexes at the dimension size too.
    m = 1
    ii = 0
    for i, l in reversed(list(zip(index, shape))):
        ii += i * m
        m *= l
    return ii


def better_unravel_index(index: int, shape: ShapeType) -> ShapeType:
    # ravel index only works for indexes less than the dimension size.
    # this version gives the index for indexes at the dimension size too.
    return typing.cast(ShapeType, numpy.unravel_index(index, (shape[0] + 1,) + tuple(shape[1:])))


def ravel_slice_start(slices: SliceType, shape: ShapeType) -> int:
    # return the flattened index for the start of the slices index.
    i = tuple(s.start if s.start is not None else 0 for s in slices)
    return better_ravel_index(i, (shape[0] + 1,) + tuple(shape[1:]))


def ravel_slice_stop(slices: SliceType, shape: ShapeType) -> int:
    # return the flattened index for the stop of the slices index.
    # note: all but the first slice have to have one subtracted from the
    # start point. the latter indexes are indexing into the previous "row".
    i = tuple(s.stop if s.stop is not None else l for s, l in zip(slices, shape))
    return better_ravel_index(i, shape) - better_ravel_index((1,) * len(slices[:-1]) + (0,), shape)

def simple_unravel_flat_slice(range_slice: slice, shape: typing.Sequence[int]) -> typing.Tuple[typing.Tuple[slice, ...], ...]:
    shape_plus = (shape[0] + 1,) + tuple(shape[1:])
    index0 = numpy.unravel_index(range_slice.start, shape_plus)
    index1 = numpy.unravel_index(range_slice.stop, shape_plus)
    ss: typing.List[slice] = list()
    b = False
    for i in reversed(range(0, len(shape))):
        i0 = index0[i]
        i1 = index1[i]
        if i0 == i1 and not b:
            ss.append(slice(None))
        else:
            if b:
                ss.append(slice(i0, i0 + 1))
            else:
                b = True
                if i1 == 0:
                    ss.append(slice(i0, i1 + shape[i]))
                else:
                    ss.append(slice(i0, i1))
    return (tuple(reversed(ss)),)

def unravel_flat_slice(range_slice: slice, shape: typing.Sequence[int]) -> typing.Tuple[typing.Tuple[slice, ...], ...]:
    start, stop = range_slice.start, range_slice.stop
    slices: typing.List[typing.Tuple[slice, ...]] = list()
    # increase the start until the lower dimensions are filled
    ss: typing.List[slice]
    for i in reversed(range(0, len(shape))):
        # print(f"{start=} {stop=}")
        d = shape[i]
        dd = expand_shape(shape[i:])
        ddl = expand_shape(shape[i+1:])
        cc = numpy.unravel_index(start, (shape[0] + 1,) + tuple(shape[1:]))
        c = cc[i]
        if c % dd != 0:
            # print(f"{d=} {dd=} {ddl=}")
            x = min(d - c, (stop - start) // ddl)
            # print(f"{x=}")
            if x > 0:
                ss = list()
                for xi in range(0, len(shape)):
                    if xi < i:
                        ss.append(slice(cc[xi],cc[xi]+1))
                    elif xi == i:
                        ss.append(slice(cc[xi], cc[xi] + x))
                    else:
                        ss.append(slice(None))
                # print(tuple(ss))
                slices.append(tuple(ss))
                start += x * ddl
    # fill the lower dimensions until everything up to stop is filled
    for i in range(0, len(shape)):
        # print(f"{start=} {stop=}")
        ddl = expand_shape(shape[i + 1:])
        cc = numpy.unravel_index(start, (shape[0] + 1,) + tuple(shape[1:]))
        # print(f"{ddl=} {cc=}")
        x = (stop - start) // ddl
        if x > 0:
            ss = list()
            for xi in range(0, len(shape)):
                if xi < i:
                    ss.append(slice(cc[xi],cc[xi]+1))
                elif xi == i:
                    ss.append(slice(cc[xi], cc[xi] + x))
                else:
                    ss.append(slice(None))
            # print(tuple(ss))
            slices.append(tuple(ss))
            start += x * expand_shape(shape[i + 1:])
    return tuple(slices)


class DataStreamEventArgs:
    """Data stream event arguments.

    The purpose of the data stream event arguments is to provide a way to pass data from a lower-level data stream,
    such as a camera, to a higher-level data stream, such as a stream to collate the camera data into a larger data
    structure like a sequence of images. The data stream event arguments must also be able to describe the channel of
    the data, the description of the layout of the data (data metadata), the data itself, and the state of the stream.

    For flexibility, data can be provided as multiple instances of the described data or as just a slice of the data.

    For instance, if the natural unit of the lower-level stream is a frame and multiple frames can be sent at once,
    the data can be described as a frame and the count property can be set. The data array is expected to have an
    extra dimension representing the count. The source slice must be provided to specify the slice of the data to be
    used. The source slice allows lower-level streams to fill a large array incrementally and only send part of the
    array for a given data stream event. However, lower-level streams can also send a new array for each data stream
    event.

    On the other hand, if the natural unit of the lower-level stream is a single frame sent incrementally,
    the count property can be set to None, and the source slice can be set to a partial slice of the frame.

    Finally, the state property indicates whether the lower-level stream has completed sending the data for which it
    was prepared and started, which is useful for the higher-level stream to know when to start processing the data.

    The `data_stream` property should be passed in from the data stream caller.

    The `channel` property should be set to something unique to allow for collecting multiple
    data streams in one collector.

    The `data_metadata` property describes the data layout, calibrations, metadata,
    time stamps, etc. of each data chunk. This is not describing the partial data (if partial),
    but instead is describing the entire data chunk.

    A data stream can send data chunks partially or n at a time.

    If sent partially, the `count` property should be `None`. `source_slice` will specify the
    slice of data.

    If the `count` property is not `None`, the data should have an extra dimension whose size
    matches the `count` property. `source_slice` will specify the slice of data and must include
    a slice for the extra dimension. slices past the first can be `slice(None)` to use the full dimension.

    The `source_slice` describes the slice on the source data. The length of the `source_slice` tuple
    should match the length of the source data shape tuple.

    The `state` property indicates if the data chunk completes a frame or not.
    """

    def __init__(self, channel: Channel, data_metadata: DataAndMetadata.DataMetadata,
                 source_data: _NDArray, count: typing.Optional[int], source_slice: SliceType,
                 state: DataStreamStateEnum, update_in_place: bool = False) -> None:
        self.__print = False

        # check data shapes
        if count is None:
            assert data_metadata.data_descriptor.expected_dimension_count == len(source_data.shape)
        else:
            assert data_metadata.data_descriptor.expected_dimension_count < len(source_data.shape)
            assert source_slice[0].start is not None
            assert source_slice[0].stop is not None

        # check the slices
        assert len(source_slice) == len(source_data.shape)
        for slice, dim in zip(source_slice, source_data.shape):
            assert slice.start is None or slice.start >= 0, f"{source_slice}, {source_data.shape}"
            assert slice.stop is None or slice.stop <= dim, f"{source_slice}, {source_data.shape}"

        # the data stream channel. must be unique within a data stream collector.
        self.channel = channel

        # the data description of the data chunks that this data stream produces.
        self.data_metadata = data_metadata

        # the data and source data slice list within this data chunk.
        self.source_data = source_data

        # the count of the data
        self.count = count

        # the slice lists of this data chunk.
        self.source_slice = source_slice

        # the state of data after this event, partial or complete. pass None if not producing partial datums.
        self.state = state

        # frame reset trigger. used to tell frames that data is being resent for the same frame.
        self.reset_frame = False

        # whether to count frame for progress. used in accumulated frames so data doesn't get counted multiple times towards progress.
        self.count_frame = True

        # the type of update
        self.update_in_place = update_in_place

    @property
    def total_bytes(self) -> int:
        """Return the total bytes contained in this packet."""
        data_dtype = self.data_metadata.data_dtype
        assert data_dtype
        itemsize = numpy.dtype(data_dtype).itemsize
        if self.count:
            return expand_shape(self.data_metadata.data_shape) * self.count * itemsize
        else:
            total_bytes = itemsize
            for i, s in enumerate(self.source_slice):
                if s.start is not None and s.stop is not None:
                    total_bytes *= int(s.stop - s.start)
                else:
                    total_bytes *= self.data_metadata.data_shape[i]
            return total_bytes


@dataclasses.dataclass
class IndexDescription:
    """Describe an index within a shape."""
    index: ShapeType
    shape: ShapeType


IndexDescriptionList = typing.List[IndexDescription]


class DataStreamArgs:
    def __init__(self, shape: ShapeType, max_count: typing.Optional[int] = None) -> None:
        self.__shape = shape
        self.max_count = max_count

    def __str__(self) -> str:
        return f"{self!r} {self.slice} {self.shape} max:{self.max_count}"

    @property
    def shape(self) -> ShapeType:
        return self.__shape

    @property
    def slice(self) -> SliceType:
        return tuple(slice(0, length) for length in self.shape)

    @property
    def slice_shape(self) -> ShapeType:
        return get_slice_shape(self.slice, self.shape)

    @property
    def sequence_count(self) -> int:
        return expand_shape(self.slice_shape)

    @property
    def slice_rect(self) -> Geometry.IntRect:
        return get_slice_rect(self.slice, self.shape)


class DataStreamInfo:
    def __init__(self, data_metadata: DataAndMetadata.DataMetadata, duration: float) -> None:
        self.data_metadata = data_metadata
        self.duration = duration

    def __str__(self) -> str:
        return f"{self.data_metadata.data_descriptor} {self.data_metadata.data_shape} {self.data_metadata.data_dtype} {self.duration}"


# wraps a data available listener with an exception handler
def _handle_data_available(data_stream: DataStream, fn: typing.Callable[[DataStream, DataStreamEventArgs], None], data_stream_event: DataStreamEventArgs, exceptions: typing.List[Exception]) -> None:
    try:
        fn(data_stream, data_stream_event)
    except Exception as e:
        # exceptions are added here for the caller to handle.
        # this avoids throwing exceptions within the fire method
        # which has limited ways of handling them.
        exceptions.append(e)


class DeviceState:
    def __init__(self) -> None:
        self.__state_unprepare_fns: typing.Dict[str, typing.Callable[[], None]] = dict()

    def add_state(self, state_id: str, state_unprepare_fn: typing.Callable[[], None]) -> None:
        if state_id not in self.__state_unprepare_fns:
            self.__state_unprepare_fns[state_id] = state_unprepare_fn

    def restore(self) -> None:
        for state_unprepare_fn in reversed(self.__state_unprepare_fns.values()):
            state_unprepare_fn()


class DataStream:
    """Provide a stream of data chunks.

    A data chunk is data that can be collected into a sequence or a 1d or 2d collection.

    1d or 2d collections can themselves be data chunks and collected into a sequence.

    Currently, neither collections nor sequences can themselves be collected into sequences or collections. This is
    due to limitations in DataAndMetadata. Future versions may include the ability to collect multiple data streams into
    lists, tables, or structures.

    The stream generates a series of data available events which can be used to update data and state in listeners.

    The stream is controlled by calls to start_stream and advance_stream and the current state can be determined with
    is_finished.

    The stream includes the concept of a sequence of acquisition. The length of the sequence gets updated in calls
    to start_stream.

    Subclasses may override channels, is_finished, _send_next, _start_stream, and _advance_stream.

    The is_error property is set if this stream or one of its contained streams enters an error state.
    """

    def __init__(self, sequence_count: int = 1) -> None:
        super().__init__()
        # these two events are used to communicate data updates and errors to the listening data streams or clients.
        self.data_available_event = Event.Event()
        # data handlers
        self.__unprocessed_data_handler: typing.Optional[DataHandler] = None
        self.__serial_data_handler: typing.Optional[SerialDataHandler] = None
        # sequence counts are used for acquiring a sequence of frames controlled by the upstream
        self.__sequence_count = sequence_count
        self.__sequence_counts: typing.Dict[Channel, int] = dict()
        self.__sequence_indexes: typing.Dict[Channel, int] = dict()
        self.is_aborted = False
        self.is_error = False
        # optional advisory fields
        self.channel_names: typing.Mapping[Channel, str] = dict()
        self.title: typing.Optional[str] = None
        # used for progress tracking. total bytes is calculated in prepare_stream. and the attached data handler
        # is used to count the bytes sent to the data handler. _attached_data_handler is a weak reference to the
        # data handler and is set when the data handler is attached to the stream. in practice, only the top level
        # data stream has a valid progress value.
        self.__total_bytes = 0
        self._attached_data_handler: weakref.ref[DataHandler] | None = None

    def _print(self, indent: typing.Optional[str] = None) -> None:
        indent = indent or str()
        print(f".{indent} {self} [{self.channels} {self.data_shapes} {self.data_types}] {self.is_finished}")
        for data_stream in self.data_streams:
            data_stream._print(indent + "  ")

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> DataStream:
        raise NotImplementedError(f"{type(self)} deepcopy not implemented")

    def connect_unprocessed_data_handler(self, data_handler: DataHandler) -> None:
        if self.__unprocessed_data_handler != data_handler:
            assert not self.__unprocessed_data_handler, f"{type(self)} {self.__unprocessed_data_handler} {data_handler}"
            self.__unprocessed_data_handler = data_handler

    def build_data_handler(self, data_handler: DataHandler) -> bool:
        # build a new data handler for this data stream and connect its output to data_handler.
        # then ask each input data stream to build a data handler and connect it to the new data handler.
        # return False if this data stream node does not support building a data handler.
        # in that case, the data_handler should be connected to the unprocessed stream directly.
        return self._build_data_handler(data_handler)

    def _get_serial_data_handler(self) -> typing.Tuple[SerialDataHandler, bool]:
        if not self.__serial_data_handler:
            self.__serial_data_handler = SerialDataHandler()
            return self.__serial_data_handler, True
        return self.__serial_data_handler, False

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        print(f"{type(self)} cannot build data handler.")
        return False

    def attach_root_data_handler(self, framed_data_handler: FramedDataHandler) -> None:
        assert not self._attached_data_handler
        if not self.build_data_handler(framed_data_handler):
            raise RuntimeError(f"{type(self)} cannot build data handler.")
        self._attached_data_handler = weakref.ref(framed_data_handler)

    @property
    def progress(self) -> float:
        """Return the progress of the data stream.

        Progress is calculated as the number of bytes sent to the data handler divided by the total bytes calculated
        in prepare_stream. The current calculation assumes all top level data is the same shape, which will not
        always be the case going forward.
        """
        assert self._attached_data_handler
        data_handler = self._attached_data_handler()
        assert isinstance(data_handler, FramedDataHandler)
        return data_handler.sent_bytes / self.__total_bytes if self.__total_bytes else 0.0

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        """Return the channels for this data stream."""
        return tuple()

    @property
    def input_channels(self) -> typing.Tuple[Channel, ...]:
        """Return the input channels for this data stream."""
        return self.channels

    def get_info(self, channel: Channel) -> DataStreamInfo:
        """Return channel info.

        Should only be called with a channel return by channels property."""
        return self._get_info(channel)

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        return DataStreamInfo(DataAndMetadata.DataMetadata(data_shape=(), data_dtype=float), 0.0)

    def prepare_device_state(self) -> DeviceState:
        """Prepares the device state (a dict of entries to restore state)."""
        device_state = DeviceState()
        self._prepare_device_state(device_state)
        return device_state

    def _prepare_device_state(self, device_state: DeviceState) -> None:
        for data_stream in self.data_streams:
            data_stream._prepare_device_state(device_state)

    @property
    def is_finished(self) -> bool:
        """Return true if stream is finished.

        The stream is finished if all channels have sent the number of items in their sequence.
        """
        return all(self._is_channel_finished(channel) for channel in self.input_channels)

    def _is_channel_finished(self, channel: Channel) -> bool:
        # return True if the channel has sent the number of items in its sequence.
        # subclasses may in limited cases override this to work around counting issues concerning side bands of
        # data such as a sequence also producing summed data. overriding this should be a warning sign and the
        # probable long term solution should be to fix the counting issue and more completely separate the data
        # handlers from the streams.
        return self.__sequence_indexes.get(channel, 0) == self.__sequence_counts.get(channel, self.__sequence_count)

    @property
    def _total_bytes(self) -> int:
        return self.__total_bytes

    def abort_stream(self) -> None:
        """Abort the stream. Called to stop the stream. Also called during exceptions."""
        self._abort_stream()
        self.is_aborted = True

    def _abort_stream(self) -> None:
        pass

    def get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        """Return raw data stream events."""
        if not self.is_finished and not self.is_aborted:
            return self._get_raw_data_stream_events()
        return list()

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        return list()

    def process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        """Process the raw stream events for counting."""
        data_stream_events = list[DataStreamEventArgs]()
        if not self.is_finished and not self.is_aborted:
            for channel in self.input_channels:
                assert self.__sequence_indexes.get(channel, 0) <= self.__sequence_counts.get(channel, self.__sequence_count)

            next_data_stream_events = list(self._process_raw_stream_events(raw_data_stream_events))

            for data_stream_ref, data_stream_event in raw_data_stream_events:
                if data_stream_ref() == self:
                    next_data_stream_events.append(data_stream_event)

            for data_stream_event in next_data_stream_events:
                if data_stream_event.channel in self.input_channels:
                    processed_data_stream_events = self._process_data_stream_event(data_stream_event)
                    for processed_data_stream_event in processed_data_stream_events:
                        self.handle_data_available(processed_data_stream_event)
                        data_stream_events.append(processed_data_stream_event)

        return data_stream_events

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        return list()

    def send_raw_data_stream_event_to_data_handler(self, data_stream_event: DataStreamEventArgs) -> None:
        """Process raw data stream event by sending it to the data handler."""
        if self.__unprocessed_data_handler:
            self.__unprocessed_data_handler.handle_data_available(data_stream_event)

    def _process_data_stream_event(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        return [data_stream_event]

    def prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        """Prepare stream. Top level prepare_stream is called before start_stream.

        The prepare function allows streams to perform any preparations before any other
        stream has started.

        The `index_stack` is a list of index descriptions, updated by any enclosing collection data streams. Each
        index description is an index and a shape. Elements in the index description list are changed during
        acquisition, i.e. they are mutable. The index stack can be used to determine whether a particular stream will
        be called repeatedly during acquisition. For example, a sequence of spectrum images may show an index stack
        like `((2,), (4,)), ((0,0), (512, 512))` meaning that it is the second of four 512x512 scans. Due to the way
        sequences of synchronized camera acquisitions are implemented, the last index may never be updated during the
        512x512 acquisition.
        """
        self._prepare_stream(stream_args, index_stack, **kwargs)

        # calculate total bytes for progress handling.
        self.__total_bytes = 0
        for channel in self.channels:
            data_stream_info = self.get_info(channel)
            data_stream_data_metadata = data_stream_info.data_metadata
            self.__total_bytes += expand_shape(data_stream_data_metadata.data_shape) * numpy.dtype(data_stream_data_metadata.data_dtype).itemsize

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        """Prepare a sequence of acquisitions.

        Subclasses can override to pass along to contained data streams.
        """
        pass

    def start_stream(self, stream_args: DataStreamArgs) -> None:
        """Restart a sequence of acquisitions. Sets the sequence count for this stream. Always balance by finish_stream."""
        for channel in self.input_channels:
            assert self.__sequence_indexes.get(channel, 0) % self.__sequence_counts.get(channel, self.__sequence_count) == 0
            self.__sequence_counts[channel] = stream_args.sequence_count
            self.__sequence_indexes[channel] = 0
        self._start_stream(stream_args)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        """Restart a sequence of acquisitions.

        Subclasses can override to pass along to contained data streams.

        Always balanced by _finish_stream.
        """
        pass

    def advance_stream(self) -> None:
        """Advance the acquisition. Must be called repeatedly until finished."""
        self._advance_stream()

    def _advance_stream(self) -> None:
        """Advance the acquisition. Must be called repeatedly until finished.

        Subclasses can override to pass along to contained data streams or handle advancement.
        """
        pass

    def finish_stream(self) -> None:
        """Finish the started acquisition."""
        self._finish_stream()

    def _finish_stream(self) -> None:
        """Finish the started acquisition.

        Subclasses can override.
        """
        pass

    def handle_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        self._handle_data_available(data_stream_event)

    def _handle_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        """
        update_in_place is a hack to allow for operations such as summing in place to not trigger sequence index updates
        since they are repeating the same update over and over. future plans would be to include an update operation
        with the data stream event, perhaps 'clear', 'replace', 'update', 'final_update' with only 'replace' and
        'final_update' advancing the sequence indexes.
        """
        if data_stream_event.state == DataStreamStateEnum.COMPLETE and not data_stream_event.update_in_place:
            count = data_stream_event.count or 1
            channel = data_stream_event.channel
            assert self.__sequence_indexes.get(channel, 0) + count <= self.__sequence_counts.get(channel, self.__sequence_count)
            self.__sequence_indexes[channel] = self.__sequence_indexes.get(channel, 0) + count

    def wrap_in_sequence(self, length: int) -> DataStream:
        """Wrap this data stream in a sequence of length."""
        return SequenceDataStream(self, length)

    @property
    def data_shapes(self) -> typing.Sequence[DataAndMetadata.ShapeType]:
        return tuple(self.get_info(channel).data_metadata.data_shape for channel in self.channels)

    @property
    def data_types(self) -> typing.Sequence[typing.Optional[numpy.typing.DTypeLike]]:
        return tuple(self.get_info(channel).data_metadata.data_dtype for channel in self.channels)

    @property
    def data_descriptors(self) -> typing.Sequence[DataAndMetadata.DataDescriptor]:
        return tuple(self.get_info(channel).data_metadata.data_descriptor for channel in self.channels)

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return tuple()


class CollectedDataStream(DataStream):
    """Collect a data stream of chunks into a collection of those chunks.

    The data is formed into a collection by resending it reshaped into the
    collection with additional slice information. No data is copied during
    reshaping.
    """

    def __init__(self, data_stream: DataStream, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration]) -> None:
        super().__init__()
        self.__data_stream = data_stream
        assert len(shape) in (1, 2)
        self.__index_stack: IndexDescriptionList = list()
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        # sub-slice indexes track the destination of the next data within the current slice.
        self.__indexes: typing.Dict[Channel, int] = dict()
        # needs starts tracks whether the downstream data stream needs a start call.
        self.__data_stream_started = False
        self.__all_channels_need_start = False
        self.__collection_list = list[DataAndMetadata.MetadataType]()
        self.__last_collection_index: typing.Optional[ShapeType] = None

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> CollectedDataStream:
        return CollectedDataStream(copy.deepcopy(self.__data_stream), copy.deepcopy(self.__collection_shape), copy.deepcopy(self.__collection_calibrations))

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return (self.__data_stream,)

    @property
    def collection_shape(self) -> DataAndMetadata.ShapeType:
        return self.__collection_shape

    @property
    def collection_calibrations(self) -> typing.Sequence[Calibration.Calibration]:
        return self.__collection_calibrations

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return self.__data_stream.channels

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        data_stream_info = self.__data_stream.get_info(channel)
        count = expand_shape(self.__collection_shape)
        data_metadata = data_stream_info.data_metadata
        data_dtype = data_metadata.data_dtype
        assert data_dtype is not None
        data_metadata = DataAndMetadata.DataMetadata(
            data_shape=self.__collection_shape + data_metadata.data_shape,
            data_dtype=data_dtype,
            intensity_calibration=data_metadata.intensity_calibration,
            dimensional_calibrations=list(self.__collection_calibrations) + list(data_metadata.dimensional_calibrations),
            metadata=data_metadata.metadata,
            timestamp=data_metadata.timestamp,
            data_descriptor=self._get_new_data_descriptor(data_metadata),
            timezone=data_metadata.timezone,
            timezone_offset=data_metadata.timezone_offset
        )
        return DataStreamInfo(data_metadata, count * data_stream_info.duration)

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_stream.process_raw_stream_events(raw_data_stream_events)

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        return self.__data_stream.get_raw_data_stream_events()

    def _process_data_stream_event(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        # grab the current collection index to be used to update the collection list of state metadata.
        # __data_available may update the collection index, so we need to grab it before calling __data_available.
        collection_index = self.__index_stack[-1].index
        # call __data_available to process the data_stream_event.
        data_stream_events = self.__data_available(data_stream_event)
        # for each data stream event, add the collection list to the metadata if action_state is present.
        for data_stream_event in data_stream_events:
            metadata = dict(data_stream_event.data_metadata.metadata)
            action_state = typing.cast(typing.MutableMapping[str, typing.Any], metadata.pop("action_state", dict()))
            if action_state:
                if collection_index != self.__last_collection_index:
                    action_state["index"] = collection_index
                    self.__collection_list.append(action_state)
                    self.__last_collection_index = collection_index
                collection_list = copy.deepcopy(self.__collection_list)
                metadata["collection"] = collection_list
                data_stream_event.data_metadata._set_metadata(metadata)
        return data_stream_events

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        self.__index_stack = list(index_stack) + [IndexDescription(better_unravel_index(0, self.__collection_shape), self.__collection_shape)]
        super()._prepare_stream(stream_args, index_stack, **kwargs)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__collection_list = list()
        self.__last_collection_index = None
        self._start_next_sub_stream()

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _start_next_sub_stream(self) -> None:
        if self.__data_stream_started:
            self.__data_stream.finish_stream()
            self.__data_stream_started = False
        self.__indexes.clear()
        self.__all_channels_need_start = False
        self.__index_stack[-1].index = better_unravel_index(0, self.__collection_shape)
        self.__data_stream.prepare_stream(DataStreamArgs(self.__collection_shape), list(self.__index_stack))
        self.__data_stream.start_stream(DataStreamArgs(self.__collection_shape))
        self.__data_stream_started = True

    def _advance_stream(self) -> None:
        # handle calling finish and start for the contained data stream.
        if self.__all_channels_need_start:
            if not self.is_finished:
                self._start_next_sub_stream()
        self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        assert self.__data_stream_started
        self.__data_stream.finish_stream()
        self.__data_stream_started = False

    def _get_new_data_descriptor(self, data_metadata: DataAndMetadata.DataMetadata) -> DataAndMetadata.DataDescriptor:
        # subclasses can override this method to provide different collection shapes.

        # collections of sequences and collections of collections are not currently supported
        assert not data_metadata.is_sequence
        assert not data_metadata.is_collection

        # new data descriptor is a collection
        collection_rank = len(self.__collection_shape)
        datum_dimension_count = data_metadata.datum_dimension_count

        if datum_dimension_count > 0:
            return DataAndMetadata.DataDescriptor(False, collection_rank, datum_dimension_count)
        else:
            return DataAndMetadata.DataDescriptor(False, 0, collection_rank)

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        # when data arrives, put it into the sequence/collection and send it out again.
        # data will be arriving as either partial data or as frame data. partial data
        # is restricted to arrive in groups that are multiples of the product of all
        # dimensions except the first one or with a count of exactly one. frame data
        # is restricted to arrive in groups that are multiples of the collection size
        # and cannot overlap the end of a collection chunk.

        processed_data_stream_events = list[DataStreamEventArgs]()

        # useful variables
        data_metadata = data_stream_event.data_metadata
        channel = data_stream_event.channel
        collection_count = expand_shape(self.__collection_shape)

        # get the new data descriptor
        new_data_descriptor = self._get_new_data_descriptor(data_metadata)

        # add the collection count to the downstream data shape to produce the new collection shape.
        new_shape = self.__collection_shape + tuple(data_metadata.data_shape)

        # add designated calibrations for the new collection dimensions.
        new_dimensional_calibrations = self.__collection_calibrations + tuple(data_metadata.dimensional_calibrations)

        # create a new data metadata object
        dtype = data_metadata.data_dtype
        assert dtype is not None
        new_data_metadata = DataAndMetadata.DataMetadata(data_shape=new_shape,
                                                         data_dtype=dtype,
                                                         intensity_calibration=data_metadata.intensity_calibration,
                                                         dimensional_calibrations=new_dimensional_calibrations,
                                                         metadata=data_metadata.metadata,
                                                         timestamp=data_metadata.timestamp,
                                                         data_descriptor=new_data_descriptor,
                                                         timezone=data_metadata.timezone,
                                                         timezone_offset=data_metadata.timezone_offset)

        # index for channel should be mod collection_count. the index is allowed to be equal
        # to collection_count to signal that the channel is complete. this fact is used to
        # calculate progress. self.__indexes[channel] will get set directly to next_channel below.
        index = self.__indexes.get(channel, 0)
        collection_rank = len(self.__collection_shape)
        collection_row_length = expand_shape(self.__collection_shape[1:])
        if data_stream_event.count is not None:
            remaining_count = data_stream_event.count
            current_index = index
            # incoming data is frames
            # count must either be a multiple of last dimensions of collection shape or one
            assert remaining_count == data_stream_event.source_slice[0].stop - data_stream_event.source_slice[0].start
            assert current_index + remaining_count <= collection_count
            assert remaining_count > 0
            next_index = index + remaining_count
            if remaining_count == 1:
                # a single data chunk has been provided.
                # if the count is one, add new dimensions of length one for the collection shape and form
                # the new slice with indexes of 0 for each collection dimension and full slices for the remaining
                # dimensions.
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                new_source_data = old_source_data.reshape((1,) * collection_rank + old_source_data.shape[1:])
                new_source_slice = (index_slice(0), ) * collection_rank + (slice(None),) * (len(old_source_data.shape) - 1)
                new_state = DataStreamStateEnum.COMPLETE if current_index + remaining_count == collection_count else DataStreamStateEnum.PARTIAL
                data_stream_event = DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state)
                processed_data_stream_events.append(data_stream_event)
            else:
                # multiple data chunks have been provided.
                # if the count is greater than one, provide the "rows" of the collection. row is just first dimension.
                # start by calculating the start/stop rows. reshape the data into collection shape and add a slice
                # for the rows and full slices for the remaining dimensions.
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                source_index = 0
                if current_index % collection_row_length != 0:
                    # finish the current row
                    assert len(self.__collection_shape) == 2
                    slice_column = current_index % collection_row_length
                    slice_width = min(collection_row_length - slice_column, remaining_count)
                    new_source_slice = (slice(0, 1), slice(0, slice_width)) + (slice(None),) * (len(new_shape) - 2)
                    next_source_index = source_index + slice_width
                    new_source_data = old_source_data[source_index:next_source_index].reshape((1, slice_width) + old_source_data.shape[1:])
                    new_state = DataStreamStateEnum.COMPLETE if current_index + slice_width == collection_count else DataStreamStateEnum.PARTIAL
                    data_stream_event = DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state)
                    processed_data_stream_events.append(data_stream_event)
                    source_index = next_source_index
                    current_index += slice_width
                    remaining_count -= slice_width
                    assert remaining_count is not None  # for type checker bug
                else:
                    new_state = DataStreamStateEnum.COMPLETE  # satisfy type checker
                if remaining_count // collection_row_length > 0:
                    # send as many complete rows as possible
                    slice_offset = current_index // collection_row_length
                    slice_start = better_unravel_index(current_index, self.__collection_shape)[0] - slice_offset
                    slice_stop = better_unravel_index(current_index + remaining_count, self.__collection_shape)[0] - slice_offset
                    assert 0 <= slice_start <= self.__collection_shape[0]
                    assert 0 <= slice_stop <= self.__collection_shape[0]
                    row_count = remaining_count // collection_row_length
                    next_source_index = source_index + row_count * collection_row_length
                    new_source_data = old_source_data[source_index:next_source_index].reshape((row_count,) + self.__collection_shape[1:] + old_source_data.shape[1:])
                    new_source_slice = (slice(slice_start, slice_stop),) + (slice(None),) * (len(new_shape) - 1)
                    new_state = DataStreamStateEnum.COMPLETE if current_index + row_count * collection_row_length == collection_count else DataStreamStateEnum.PARTIAL
                    data_stream_event = DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state)
                    processed_data_stream_events.append(data_stream_event)
                    source_index = next_source_index
                    current_index += row_count * collection_row_length
                    remaining_count -= row_count * collection_row_length
                    assert remaining_count is not None  # for type checker bug
                if remaining_count > 0:
                    # any remaining count means a partial row
                    assert len(self.__collection_shape) == 2
                    assert remaining_count < collection_row_length
                    new_source_slice = (slice(0, 1), slice(0, remaining_count)) + (slice(None),) * (len(new_shape) - 2)
                    next_source_index = source_index + remaining_count
                    new_source_data = old_source_data[source_index:next_source_index].reshape((1, remaining_count) + old_source_data.shape[1:])
                    new_state = DataStreamStateEnum.PARTIAL  # always partial, otherwise would have been sent in previous section
                    data_stream_event = DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state)
                    processed_data_stream_events.append(data_stream_event)
                    # source_index = next_source_index  # no need for this
                    current_index += remaining_count
                    remaining_count -= remaining_count
                    assert remaining_count is not None  # for type checker bug
                assert remaining_count == 0  # everything has been accounted for
        else:
            # incoming data is partial
            # form the new slice with indexes of 0 for each collection dimension and the incoming slice for the
            # remaining dimensions. add dimensions for collection dimensions to the new source data.
            new_source_slice = (index_slice(0), ) * collection_rank + tuple(data_stream_event.source_slice)
            new_source_data = data_stream_event.source_data.reshape((1,) * collection_rank + data_stream_event.source_data.shape)
            # new state is complete if data chunk is complete and it will put us at the end of our collection.
            new_state = DataStreamStateEnum.PARTIAL
            next_index = index
            if data_stream_event.state == DataStreamStateEnum.COMPLETE:
                next_index += 1
                if next_index == collection_count:
                    new_state = DataStreamStateEnum.COMPLETE
            data_stream_event = DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state)
            processed_data_stream_events.append(data_stream_event)
        self.__indexes[channel] = next_index
        # whether all channels are in the 'needs_start' state.
        needs_starts = {channel: self.__indexes.get(channel, 0) == collection_count for channel in self.channels}
        self.__all_channels_need_start = all(needs_starts.get(channel, False) for channel in self.channels)
        self.__index_stack[-1].index = better_unravel_index(min(self.__indexes.get(channel, 0) for channel in self.channels), self.__collection_shape)
        return processed_data_stream_events

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        collection_data_handler = self._get_data_handler()
        # send the result of the collection data handler to data_handler
        collection_data_handler.connect_data_handler(self._get_target_data_handler(data_handler))
        # ask the child data stream build its data handler and send it to the collection data handler. the collected
        # data stream uses a trick (hopefully temporary) of asking the input data stream to supply a serial data
        # handler; then this collected stream feeds the serial data handler into the collection_data_handler and asks
        # the input data stream to connect to the serial data handler. this allows the same input data stream to be
        # connected to multiple collection data handlers, with the common serial data stream stored in the (common)
        # input data stream.
        count = expand_shape(self.__collection_shape)
        serial_data_handler, created = self.__data_stream._get_serial_data_handler()
        serial_data_handler.add_data_handler(collection_data_handler, count)
        if created:
            if not self.__data_stream.build_data_handler(serial_data_handler):
                self.__data_stream.connect_unprocessed_data_handler(serial_data_handler)
        return True

    def _get_target_data_handler(self, data_handler: DataHandler) -> DataHandler:
        # subclass (SequenceDataStream) can override this to insert a more complex data handler network between
        # the data handlers feeding the collection and the output data handler, useful for providing the sequence
        # summing feature. this is a temporary solution until the data handlers are more completely separated from
        # the data streams.
        return data_handler

    def _get_data_handler(self) -> DataHandler:
        # subclass (SequenceDataStream) can override this to provide a more specific data handler for its needs.
        return CollectionDataHandler(self.collection_shape, self.collection_calibrations)


class SequenceDataStream(CollectedDataStream):
    """Collect a data stream into a sequence of datums.

    This is a subclass of CollectedDataStream.
    """
    def __init__(self, data_stream: DataStream, count: int, calibration: typing.Optional[Calibration.Calibration] = None, *, include_sum: bool = False) -> None:
        calibration_ = calibration or Calibration.Calibration()
        super().__init__(data_stream, (count,), (calibration_,))
        self.__include_sum = include_sum

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> SequenceDataStream:
        return SequenceDataStream(copy.deepcopy(self.data_streams[-1]), copy.deepcopy(self.collection_shape[-1]), copy.deepcopy(self.collection_calibrations[-1]))

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        channels = list(super().channels)
        if self.__include_sum:
            for channel in list(channels):
                channels.append(Channel(*channel.segments, "sum"))
        return tuple(channels)

    def _is_channel_finished(self, channel: Channel) -> bool:
        if channel.segments[-1] == "sum":
            return True
        return super()._is_channel_finished(channel)

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        if channel.segments[-1] == "sum":
            data_stream_info = super()._get_info(Channel(*channel.segments[:-1]))
            data_metadata = data_stream_info.data_metadata
            return DataStreamInfo(DataAndMetadata.DataMetadata(
                data_shape=data_metadata.data_shape[1:],
                data_dtype=data_metadata.data_dtype,
                intensity_calibration=data_metadata.intensity_calibration,
                dimensional_calibrations=data_metadata.dimensional_calibrations[1:],
                metadata=data_metadata.metadata,
                timestamp=data_metadata.timestamp,
                data_descriptor=DataAndMetadata.DataDescriptor(False, data_metadata.collection_dimension_count, data_metadata.datum_dimension_count),
                timezone=data_metadata.timezone,
                timezone_offset=data_metadata.timezone_offset
            ), 0)
        else:
            return super()._get_info(channel)

    def _get_new_data_descriptor(self, data_metadata: DataAndMetadata.DataMetadata) -> DataAndMetadata.DataDescriptor:
        # scalar data is not supported. and the data must not be a sequence already.
        assert not data_metadata.is_sequence
        assert data_metadata.datum_dimension_count > 0

        # new data descriptor is a sequence
        collection_dimension_count = data_metadata.collection_dimension_count
        datum_dimension_count = data_metadata.datum_dimension_count
        return DataAndMetadata.DataDescriptor(True, collection_dimension_count, datum_dimension_count)

    def _get_target_data_handler(self, data_handler: DataHandler) -> DataHandler:
        if self.__include_sum:
            accumulated_framed_data_handler = FramedDataHandler(Framer(DataAndMetadataDataChannel()))
            accumulated_framed_data_handler.connect_data_handler(data_handler)
            accumulated_data_handler = AccumulatedDataHandler(True)
            accumulated_data_handler.connect_data_handler(accumulated_framed_data_handler)
            return SplittingDataHandler((data_handler, accumulated_data_handler))
        else:
            return data_handler

    def _get_data_handler(self) -> DataHandler:
        count = self.collection_shape[-1]
        return SequenceDataHandler(count, self.collection_calibrations[-1])


class CombinedDataStream(DataStream):
    """Combine multiple streams into a single stream producing multiple channels.

    Each stream can also produce multiple channels.
    """
    def __init__(self, data_streams: typing.Sequence[DataStream]) -> None:
        super().__init__()
        self.__data_streams = tuple(data_streams)

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> CombinedDataStream:
        return CombinedDataStream(copy.deepcopy(self.__data_streams))

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return self.__data_streams

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        channels: typing.List[Channel] = list()
        for data_stream in self.__data_streams:
            channels.extend(data_stream.channels)
        return tuple(channels)

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        for data_stream in self.__data_streams:
            if channel in data_stream.channels:
                return data_stream.get_info(channel)
        assert False, f"No info for channel {channel}"

    @property
    def is_finished(self) -> bool:
        return all(data_stream.is_finished for data_stream in self.__data_streams)

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        data_stream_events = list[DataStreamEventArgs]()
        for data_stream in self.__data_streams:
            data_stream_events.extend(data_stream.process_raw_stream_events(raw_data_stream_events))
        return data_stream_events

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        raw_data_stream_events = list[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]()
        for data_stream in self.__data_streams:
            raw_data_stream_events.extend(data_stream.get_raw_data_stream_events())
        return raw_data_stream_events

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        for data_stream in self.__data_streams:
            data_stream.prepare_stream(stream_args, index_stack)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        for data_stream in self.__data_streams:
            data_stream.start_stream(stream_args)

    def _abort_stream(self) -> None:
        for data_stream in self.__data_streams:
            data_stream.abort_stream()

    def _advance_stream(self) -> None:
        for data_stream in self.__data_streams:
            data_stream.advance_stream()

    def _finish_stream(self) -> None:
        for data_stream in self.__data_streams:
            data_stream.finish_stream()

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        for data_stream in self.__data_streams:
            if not data_stream.build_data_handler(data_handler):
                data_stream.connect_unprocessed_data_handler(data_handler)
        return True


class StackedDataStream(DataStream):
    """Acquire multiple streams and stack the results.

    For instance, if stream A produces 10 frames of 8x8 and stream B produces 20 of 8x8 frames, the resulting stacked
    stream of A + B will produce a new stream of the shape 30x8x8.

    Each stream can produce multiple channels but the channels must match shape/type between streams.
    """
    def __init__(self, data_streams: typing.Sequence[DataStream]) -> None:
        super().__init__()
        self.__data_streams = tuple(data_streams)
        self.__stream_args = DataStreamArgs(list())
        self.__current_index = 0  # data stream index
        assert len(set(data_stream.channels for data_stream in self.__data_streams)) == 1
        self.__channels = self.__data_streams[0].channels
        self.__sequence_count = 0
        self.__sequence_index = 0
        self.__height = 0

        # check that the data streams have the same shape (except for height), the same dtype, and the same data
        # descriptor.
        for channel in self.__channels:
            height = 0
            data_metadata = self.__data_streams[0].get_info(channel).data_metadata
            for data_stream in self.__data_streams:
                data_stream_data_metadata = data_stream.get_info(channel).data_metadata
                assert data_metadata.data_shape[1:] == data_stream_data_metadata.data_shape[1:]
                assert data_metadata.data_dtype == data_stream_data_metadata.data_dtype
                assert data_metadata.data_descriptor == data_stream_data_metadata.data_descriptor
                height += data_stream_data_metadata.data_shape[0]
            assert self.__height == 0 or self.__height == height
            self.__height = height

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> StackedDataStream:
        return StackedDataStream(copy.deepcopy(self.__data_streams))

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return self.__data_streams

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return self.__channels

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        # compute duration and height as a sum from contained data streams.
        data_metadata = copy.deepcopy(self.__data_streams[0].get_info(channel).data_metadata)
        duration = 0.0
        height = 0
        for data_stream in self.__data_streams:
            duration += data_stream.get_info(channel).duration
            data_stream_data_metadata = data_stream.get_info(channel).data_metadata
            height += data_stream_data_metadata.data_shape[0]
            assert data_metadata.data_shape[1:] == data_stream_data_metadata.data_shape[1:]
            assert data_metadata.data_dtype == data_stream_data_metadata.data_dtype
            assert data_metadata.data_descriptor == data_stream_data_metadata.data_descriptor
        data_metadata._set_data_shape_and_dtype(((height,) + data_metadata.data_shape[1:], numpy.dtype(data_metadata.data_dtype)))
        return DataStreamInfo(data_metadata, duration)

    @property
    def is_finished(self) -> bool:
        return self.__current_index == len(self.__data_streams)

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_streams[self.__current_index].process_raw_stream_events(raw_data_stream_events)

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        return self.__data_streams[self.__current_index].get_raw_data_stream_events()

    def _process_data_stream_event(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_available(data_stream_event)

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        self.__current_index = 0
        self.__stream_args = DataStreamArgs((1,))
        self.__index_stack = list(index_stack)
        self.__data_streams[self.__current_index].prepare_stream(self.__stream_args, list(self.__index_stack))
        self.__sequence_count = stream_args.sequence_count
        self.__sequence_index = 0

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert self.__current_index == 0
        self.__data_streams[self.__current_index].start_stream(self.__stream_args)

    def _abort_stream(self) -> None:
        self.__data_streams[self.__current_index].abort_stream()

    def _advance_stream(self) -> None:
        # handle calling finish and start for the contained data stream.
        if self.__current_index < len(self.__data_streams):
            if self.__data_streams[self.__current_index].is_finished:
                self.__data_streams[self.__current_index].finish_stream()
                self.__current_index += 1
                if self.__current_index == len(self.__data_streams) and self.__sequence_index + 1 < self.__sequence_count:
                    self.__current_index = 0
                    self.__sequence_index += 1
                if self.__current_index < len(self.__data_streams):
                    self.__data_streams[self.__current_index].prepare_stream(self.__stream_args, list(self.__index_stack))
                    self.__data_streams[self.__current_index].start_stream(self.__stream_args)

        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].advance_stream()

    def _finish_stream(self) -> None:
        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].finish_stream()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        if data_stream_event.state == DataStreamStateEnum.COMPLETE and self.__current_index + 1 == len(self.__data_streams):
            state = DataStreamStateEnum.COMPLETE
        else:
            state = DataStreamStateEnum.PARTIAL
        # compute height as a sum from contained data streams and configure a new data_metadata
        data_metadata = copy.deepcopy(data_stream_event.data_metadata)
        data_metadata._set_data_shape_and_dtype(((self.__height,) + data_metadata.data_shape[1:], numpy.dtype(data_metadata.data_dtype)))
        # create the data stream event with the overridden data_metadata and state.
        data_stream_event = DataStreamEventArgs(
            data_stream_event.channel,
            data_metadata,
            data_stream_event.source_data,
            data_stream_event.count,
            data_stream_event.source_slice,
            state
        )
        return [data_stream_event]

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        stacked_data_handler = StackedDataHandler(len(self.__data_streams), self.__height)
        # send the result of the framed data handler to data_handler
        stacked_data_handler.connect_data_handler(data_handler)
        # let the child data stream build its data handler or attach the handler to this data stream.
        for data_stream in self.__data_streams:
            if not data_stream.build_data_handler(stacked_data_handler):
                data_stream.connect_unprocessed_data_handler(stacked_data_handler)
        return True


class SequentialDataStream(DataStream):
    """Acquire multiple streams sequentially.

    Each stream can also produce multiple channels.
    """
    def __init__(self, data_streams: typing.Sequence[DataStream]) -> None:
        super().__init__()
        self.__data_streams = tuple(data_streams)
        self.__stream_args = DataStreamArgs(list())
        self.__current_index = 0
        self.__sequence_count = 0
        self.__sequence_index = 0

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> SequentialDataStream:
        return SequentialDataStream(copy.deepcopy(self.__data_streams))

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return self.__data_streams

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        channels: typing.List[Channel] = list()
        for index, data_stream in enumerate(self.__data_streams):
            channels.extend([Channel(str(index), *channel.segments) for channel in data_stream.channels])
        return tuple(channels)

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        if channel in self.channels:
            index = int(channel.segments[0])
            sub_channel = Channel(*channel.segments[1:])
            if sub_channel in self.__data_streams[index].channels:
                return self.__data_streams[index].get_info(sub_channel)
        assert False, f"No info for channel {channel}"

    @property
    def is_finished(self) -> bool:
        return self.__current_index == len(self.__data_streams)

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_streams[self.__current_index].process_raw_stream_events(raw_data_stream_events)

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        return self.__data_streams[self.__current_index].get_raw_data_stream_events()

    def _process_data_stream_event(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_available(data_stream_event)

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        self.__current_index = 0
        self.__stream_args = DataStreamArgs((1,))
        self.__index_stack = list(index_stack)
        self.__data_streams[self.__current_index].prepare_stream(self.__stream_args, list(self.__index_stack))
        self.__sequence_count = stream_args.sequence_count
        self.__sequence_index = 0

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert self.__current_index == 0
        self.__data_streams[self.__current_index].start_stream(self.__stream_args)

    def _abort_stream(self) -> None:
        self.__data_streams[self.__current_index].abort_stream()

    def _advance_stream(self) -> None:
        # handle calling finish and start for the contained data stream.
        if self.__data_streams[self.__current_index].is_finished:
            self.__data_streams[self.__current_index].finish_stream()
            self.__current_index += 1
            if self.__current_index == len(self.__data_streams) and self.__sequence_index + 1 < self.__sequence_count:
                self.__current_index = 0
                self.__sequence_index += 1
            if self.__current_index < len(self.__data_streams):
                self.__data_streams[self.__current_index].prepare_stream(self.__stream_args, list(self.__index_stack))
                self.__data_streams[self.__current_index].start_stream(self.__stream_args)

        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].advance_stream()

    def _finish_stream(self) -> None:
        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].finish_stream()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        state = DataStreamStateEnum.COMPLETE if data_stream_event.state == DataStreamStateEnum.COMPLETE and self.__current_index + 1 == len(self.__data_streams) else DataStreamStateEnum.PARTIAL
        data_stream_event = DataStreamEventArgs(
            Channel(str(self.__current_index), *data_stream_event.channel.segments),
            data_stream_event.data_metadata,
            data_stream_event.source_data,
            data_stream_event.count,
            data_stream_event.source_slice,
            state
        )
        return [data_stream_event]

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        sequential_data_handler = SequentialDataHandler([data_stream.channels for data_stream in self.__data_streams])
        # send the result of the framed data handler to data_handler
        sequential_data_handler.connect_data_handler(data_handler)
        # let the child data stream build its data handler or attach the handler to this data stream.
        for data_stream in self.__data_streams:
            if not data_stream.build_data_handler(sequential_data_handler):
                data_stream.connect_unprocessed_data_handler(sequential_data_handler)
        return True


class DataStreamFunctor:
    """Define a data stream functor: a class which creates a new data stream from another data stream.

    This is useful to pass the code to construct a data stream to a function.
    """
    def apply(self, data_stream: DataStream) -> DataStream:
        return data_stream


class ChannelData:
    def __init__(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
        self.channel = channel
        self.data_and_metadata = data_and_metadata


class DataStreamOperator:
    def __init__(self) -> None:
        self.__applied = False

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> DataStreamOperator:
        raise NotImplementedError(f"{type(self)} deepcopy not implemented")

    def reset(self) -> None:
        self.__applied = False

    def apply(self) -> None:
        self.__applied = True

    @property
    def is_applied(self) -> bool:
        return self.__applied

    def get_channels(self, input_channels: typing.Sequence[Channel]) -> typing.Sequence[Channel]:
        return input_channels

    def transform_data_stream_info(self, channel: Channel, data_stream_info: DataStreamInfo) -> DataStreamInfo:
        return data_stream_info

    def process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        return self._process(channel_data)

    def process_multiple(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        return self._process_multiple(channel_data)

    def _process_multiple(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        channel = channel_data.channel
        data_and_metadata = channel_data.data_and_metadata
        assert data_and_metadata.is_sequence
        channel_data_list_list = [self.process(ChannelData(channel, data_and_metadata[i])) for i in range(data_and_metadata.data_shape[0])]
        new_channel_data_list = list()
        for index, new_channel_data in enumerate(channel_data_list_list[0]):
            new_channel = new_channel_data.channel
            new_data_and_metadata = xd.sequence_join([cdl[index].data_and_metadata for cdl in channel_data_list_list])
            assert new_data_and_metadata
            new_data_and_metadata._set_metadata(data_and_metadata.metadata)
            new_data_and_metadata._set_timestamp(data_and_metadata.timestamp)
            new_data_and_metadata.timezone = data_and_metadata.timezone
            new_data_and_metadata.timezone_offset = data_and_metadata.timezone_offset
            new_channel_data_list.append(ChannelData(new_channel, new_data_and_metadata))
        return new_channel_data_list

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        raise NotImplementedError()


class NullDataStreamOperator(DataStreamOperator):
    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "null"

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        return [channel_data]


class CompositeDataStreamOperator(DataStreamOperator):
    def __init__(self, operator_map: typing.Dict[Channel, DataStreamOperator]) -> None:
        super().__init__()
        self.__operator_map = operator_map

    def __str__(self) -> str:
        return f"composite ({self.__operator_map})"

    def get_channels(self, input_channels: typing.Sequence[Channel]) -> typing.Sequence[Channel]:
        return list(self.__operator_map.keys())

    def transform_data_stream_info(self, channel: Channel, data_stream_info: DataStreamInfo) -> DataStreamInfo:
        return self.__operator_map[channel].transform_data_stream_info(channel, data_stream_info)

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        channel_data_list = list()
        for channel, operator in self.__operator_map.items():
            for new_channel_data in operator.process(channel_data):
                channel_data_list.append(ChannelData(channel, new_channel_data.data_and_metadata))
        return channel_data_list


class StackedDataStreamOperator(DataStreamOperator):
    # puts the last dimension into a stack.
    # this means if you have a list of operators producing scalar data, it will apply each operator to the input stream
    # and stack them so that the resulting stream is 1d data where each element is the result from the corresponding
    # operator.

    def __init__(self, operators: typing.Sequence[DataStreamOperator]) -> None:
        super().__init__()
        self.__operators = list(operators)

    def __str__(self) -> str:
        return f"stacked ({self.__operators})"

    @property
    def operators(self) -> typing.Sequence[DataStreamOperator]:
        return self.__operators

    def transform_data_stream_info(self, channel: Channel, data_stream_info: DataStreamInfo) -> DataStreamInfo:
        assert self.__operators
        duration = sum(operator.transform_data_stream_info(channel, data_stream_info).duration for operator in self.__operators)
        data_stream_info = self.__operators[0].transform_data_stream_info(channel, data_stream_info)
        operator_count = len(self.__operators)
        if operator_count > 1:
            data_metadata = data_stream_info.data_metadata
            assert not data_metadata.is_sequence
            assert not data_metadata.is_collection
            assert data_metadata.datum_dimension_count == 0
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 1)
            dtype = data_metadata.data_dtype
            assert dtype is not None
            data_metadata = DataAndMetadata.DataMetadata(
                data_shape=(operator_count, ) + data_metadata.data_shape,
                data_dtype=dtype,
                intensity_calibration=data_metadata.intensity_calibration,
                dimensional_calibrations=[Calibration.Calibration()] + list(data_metadata.dimensional_calibrations),
                metadata=data_metadata.metadata,
                timestamp=data_metadata.timestamp,
                data_descriptor=data_descriptor,
                timezone=data_metadata.timezone,
                timezone_offset=data_metadata.timezone_offset
            )
            data_stream_info = DataStreamInfo(data_metadata, duration)
        return data_stream_info

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        data_list: typing.List[DataAndMetadata.DataAndMetadata] = list()
        for operator in self.__operators:
            for new_channel_data in operator.process(channel_data):
                data_list.append(new_channel_data.data_and_metadata[..., numpy.newaxis])
        if data_list[0].data_shape == (1,):
            if len(data_list) == 1:
                # handle special case where we're only concatenating a single scalar
                # squeeze the numpy arrays to keep the dtype the same. xdata promotes to float64.
                data = data_list[0].data
                assert data is not None
                new_data = numpy.squeeze(data, axis=-1)
                data_metadata = data_list[0].data_metadata
                dtype = data_metadata.data_dtype
                assert dtype is not None
                data_metadata = DataAndMetadata.DataMetadata(data_shape=(), data_dtype=dtype, intensity_calibration=data_metadata.intensity_calibration,
                                                             dimensional_calibrations=None,
                                                             metadata=data_metadata.metadata, timestamp=data_metadata.timestamp,
                                                             data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 0),
                                                             timezone=data_metadata.timezone, timezone_offset=data_metadata.timezone_offset)
            else:
                # not sure if this special case is needed...? it is only different in that it produces
                # slightly different data_metadata.
                # concatenate the numpy arrays to keep the dtype the same. xdata promotes to float64.
                new_data = numpy.concatenate([data_and_metadata.data for data_and_metadata in data_list], axis=-1)
                data_metadata = data_list[0].data_metadata
                data_metadata = DataAndMetadata.DataMetadata(data_shape=new_data.shape, data_dtype=new_data.dtype,
                                                             intensity_calibration=data_metadata.intensity_calibration,
                                                             dimensional_calibrations=[Calibration.Calibration()] + list(data_metadata.dimensional_calibrations[:-1]),
                                                             metadata=data_metadata.metadata, timestamp=data_metadata.timestamp,
                                                             data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 1),
                                                             timezone=data_metadata.timezone, timezone_offset=data_metadata.timezone_offset)
        else:
            # concatenate the numpy arrays to keep the dtype the same. xdata promotes to float64.
            new_data = numpy.concatenate([data_and_metadata.data for data_and_metadata in data_list], axis=-1)
            data_metadata = data_list[0].data_metadata
        new_data_and_metadata = DataAndMetadata.new_data_and_metadata(new_data,
                                                                      data_metadata.intensity_calibration,
                                                                      data_metadata.dimensional_calibrations,
                                                                      data_metadata.metadata,
                                                                      data_metadata.timestamp,
                                                                      data_metadata.data_descriptor,
                                                                      data_metadata.timezone,
                                                                      data_metadata.timezone_offset)
        return [ChannelData(channel_data.channel, new_data_and_metadata)]


class DataChannel:
    """Acquisition data channel.

    An acquisition data channel receives partial data and must return full data when required.
    """
    def __init__(self) -> None:
        super().__init__()

    def prepare(self, channel_info_map: typing.Mapping[Channel, DataStreamInfo]) -> None:
        # prepare will be called on the main thread.
        pass

    def update_data(self, channel: Channel, source_data: _NDArray, source_slice: SliceType, dest_slice: slice, data_metadata: DataAndMetadata.DataMetadata) -> None:
        # update_data will be called on an acquisition thread.
        raise NotImplementedError()

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        # get_data may be called on an acquisition thread or the main thread.
        raise NotImplementedError()


class DataAndMetadataDataChannel(DataChannel):
    def __init__(self) -> None:
        super().__init__()
        self.__data: typing.Dict[Channel, DataAndMetadata.DataAndMetadata] = dict()

    def __make_data(self, channel: Channel, data_metadata: DataAndMetadata.DataMetadata) -> DataAndMetadata.DataAndMetadata:
        data_and_metadata = self.__data.get(channel, None)
        if not data_and_metadata:
            data: numpy.typing.NDArray[typing.Any] = numpy.zeros(data_metadata.data_shape, data_metadata.data_dtype)
            data_descriptor = data_metadata.data_descriptor
            data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                      data_metadata.intensity_calibration,
                                                                      data_metadata.dimensional_calibrations,
                                                                      data_metadata.metadata,
                                                                      data_metadata.timestamp,
                                                                      data_descriptor,
                                                                      data_metadata.timezone,
                                                                      data_metadata.timezone_offset)
            self.__data[channel] = data_and_metadata
        return data_and_metadata

    def clear_data(self) -> None:
        self.__data.clear()

    def update_data(self, channel: Channel, source_data: _NDArray, source_slice: SliceType, dest_slice: slice, data_metadata: DataAndMetadata.DataMetadata) -> None:
        data_and_metadata = self.__make_data(channel, data_metadata)
        # copy data
        data = data_and_metadata.data
        assert data is not None
        data.reshape(-1)[dest_slice] = source_data[source_slice].reshape(-1)
        # recopy metadata. this isn't perfect; but it's the chosen way for now. if changed, ensure tests pass.
        # the effect of this is that the last chunk of data defines the final metadata. this is useful if the
        # metadata contains in-progress information.
        data_and_metadata._set_data_descriptor(data_metadata.data_descriptor)
        data_and_metadata._set_intensity_calibration(data_metadata.intensity_calibration)
        data_and_metadata._set_dimensional_calibrations(data_metadata.dimensional_calibrations)
        data_and_metadata._set_metadata(data_metadata.metadata)
        data_and_metadata._set_timestamp(data_metadata.timestamp)
        data_and_metadata.timezone = data_metadata.timezone
        data_and_metadata.timezone_offset = data_metadata.timezone_offset

    def accumulate_data(self, channel: Channel, source_data: _NDArray, source_slice: SliceType, dest_slice: slice, data_metadata: DataAndMetadata.DataMetadata) -> None:
        data_and_metadata = self.__make_data(channel, data_metadata)
        # accumulate data
        data = data_and_metadata.data
        assert data is not None
        data.reshape(-1)[dest_slice] += source_data[source_slice].reshape(-1)
        # recopy metadata. this isn't perfect; but it's the chosen way for now. if changed, ensure tests pass.
        # the effect of this is that the last chunk of data defines the final metadata. this is useful if the
        # metadata contains in-progress information.
        data_and_metadata._set_metadata(data_metadata.metadata)

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        assert channel in self.__data, f"{channel} not in {list(self.__data.keys())}"
        return self.__data[channel]


class FrameCallbacks(typing.Protocol):
    def _send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count_frame: bool) -> typing.Sequence[DataStreamEventArgs]: ...
    def _send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int, count_frame: bool) -> typing.Sequence[DataStreamEventArgs]: ...


class Framer:
    def __init__(self, data_channel: DataChannel) -> None:
        super().__init__()
        # data and indexes use the _incoming_ data channels as keys.
        self.__data_channel = data_channel
        self.__indexes: typing.Dict[Channel, int] = dict()

    def prepare(self, channel_info_map: typing.Mapping[Channel, DataStreamInfo]) -> None:
        self.__data_channel.prepare(channel_info_map)

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__data_channel.get_data(channel)

    def data_available(self, data_stream_event: DataStreamEventArgs, callbacks: FrameCallbacks) -> typing.Sequence[DataStreamEventArgs]:
        # when data arrives, store it into a data item with the same description/shape.
        # data is assumed to be partial data. this restriction may be removed in a future
        # version. separate indexes are kept for each channel and represent the next destination
        # for the data.

        # return the processed events for the next level up.
        processed_data_stream_events = list[DataStreamEventArgs]()

        # useful variables
        channel = data_stream_event.channel
        source_slice = data_stream_event.source_slice
        count = data_stream_event.count

        if count is None or count == 1:
            # check to see if data has already been allocated. allocated it if not.
            data_metadata = copy.deepcopy(data_stream_event.data_metadata)
            # determine the start/stop indexes. then copy the source data into the destination using
            # flattening to allow for use of simple indexing. then increase the index.
            source_start = ravel_slice_start(source_slice, data_stream_event.source_data.shape)
            source_stop = ravel_slice_stop(source_slice, data_stream_event.source_data.shape)
            source_count = source_stop - source_start
            flat_shape = (expand_shape(data_metadata.data_shape),)
            if data_stream_event.reset_frame:
                index = 0
            else:
                index = self.__indexes.get(channel, 0)
            dest_slice = slice(index, index + source_count)
            assert index + source_count <= flat_shape[0], f"{index=} + {source_count=} <= {flat_shape[0]=}; {channel=}"
            if count is None:
                source_data = data_stream_event.source_data
                source_slice = source_slice
            else:
                source_data = data_stream_event.source_data[source_slice].reshape(data_stream_event.source_data.shape[1:])
                source_slice = data_stream_event.source_slice[1:]
            self.__data_channel.update_data(channel, source_data, source_slice, dest_slice, data_metadata)
            # proceed
            index = index + source_count
            self.__indexes[channel] = index
            # if the data chunk is complete, perform processing and send out the new data.
            if data_stream_event.state == DataStreamStateEnum.COMPLETE:
                assert index == flat_shape[0], f"{data_stream_event.channel}: {index=} == {flat_shape[0]=}"  # index should be at the end.
                processed_data_stream_events.extend(callbacks._send_data(data_stream_event.channel, self.__data_channel.get_data(data_stream_event.channel), data_stream_event.count_frame))
                self.__indexes[channel] = 0
        else:
            # no storage takes place in this case; receiving full frames and sending out full (processed) frames.
            # add 'sequence' to data descriptor; process it; strip 'sequence' and send it on.
            data_metadata = data_stream_event.data_metadata
            data_descriptor = data_metadata.data_descriptor
            assert not data_descriptor.is_sequence
            new_data_descriptor = DataAndMetadata.DataDescriptor(True,
                                                                 data_descriptor.collection_dimension_count,
                                                                 data_descriptor.datum_dimension_count)
            new_data = data_stream_event.source_data[data_stream_event.source_slice]
            new_dimensional_calibrations = (Calibration.Calibration(),) + tuple(data_metadata.dimensional_calibrations)
            data_and_metadata = DataAndMetadata.new_data_and_metadata(new_data,
                                                                      data_metadata.intensity_calibration,
                                                                      new_dimensional_calibrations,
                                                                      data_metadata.metadata,
                                                                      data_metadata.timestamp,
                                                                      new_data_descriptor,
                                                                      data_metadata.timezone,
                                                                      data_metadata.timezone_offset)
            processed_data_stream_events.extend(callbacks._send_data_multiple(data_stream_event.channel, data_and_metadata, count, data_stream_event.count_frame))
            self.__indexes[channel] = 0
        return processed_data_stream_events


class FramedDataStream(DataStream):
    """Change a data stream producing continuous data into one producing frame by frame data.

    This is useful when finalizing data or when processing needs the full frame before proceeding.

    Pass an operator to process the resulting frame on any channel before sending it out.

    Pass an operator map to process the resulting frame using multiple operators and send them out on different
    channels.

    Pass a data channel to accept the frame as it arrives.
    """

    def __init__(self, data_stream: DataStream, *, operator: typing.Optional[DataStreamOperator] = None, data_channel: typing.Optional[DataChannel] = None) -> None:
        super().__init__()
        self.__data_stream = data_stream
        self.__operator = operator or NullDataStreamOperator()
        self.__framer = Framer(data_channel or DataAndMetadataDataChannel())

    def __str__(self) -> str:
        s = super().__str__()
        if self.__operator and not isinstance(self.__operator, NullDataStreamOperator):
            s = s + f" ({self.__operator})"
        return s

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> FramedDataStream:
        return FramedDataStream(copy.deepcopy(self.__data_stream), operator=copy.deepcopy(self.__operator))

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return (self.__data_stream,)

    @property
    def operator(self) -> typing.Optional[DataStreamOperator]:
        return self.__operator

    @property
    def data_stream(self) -> DataStream:
        return self.__data_stream

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return tuple(self.__operator.get_channels(self.input_channels))

    @property
    def input_channels(self) -> typing.Tuple[Channel, ...]:
        return self.__data_stream.channels

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        return self.__operator.transform_data_stream_info(channel, self.__data_stream.get_info(channel))

    @property
    def is_finished(self) -> bool:
        return self.__data_stream.is_finished

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_stream.process_raw_stream_events(raw_data_stream_events)

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        return self.__data_stream.get_raw_data_stream_events()

    def _process_data_stream_event(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_available(data_stream_event)

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        self.__operator.reset()
        self.__data_stream.prepare_stream(stream_args, index_stack, operator=self.__operator)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__data_stream.start_stream(stream_args)

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _advance_stream(self) -> None:
        self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        self.__data_stream.finish_stream()

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__framer.get_data(channel)

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> typing.Sequence[DataStreamEventArgs]:
        return self.__framer.data_available(data_stream_event, typing.cast(FrameCallbacks, self))

    def _send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count_frame: bool) -> typing.Sequence[DataStreamEventArgs]:
        # callback for Framer
        processed_data_stream_events = list[DataStreamEventArgs]()
        if not self.__operator.is_applied:
            for new_channel_data in self.__operator.process(ChannelData(channel, data_and_metadata)):
                processed_data_stream_events.extend(self.__send_data(new_channel_data.channel, new_channel_data.data_and_metadata, update_in_place=True))
        else:
            processed_data_stream_events.extend(self.__send_data(channel, data_and_metadata))
        return processed_data_stream_events

    def _send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int, count_frame: bool) -> typing.Sequence[DataStreamEventArgs]:
        # callback for Framer
        processed_data_stream_events = list[DataStreamEventArgs]()
        if not self.__operator.is_applied:
            for new_channel_data in self.__operator.process_multiple(ChannelData(channel, data_and_metadata)):
                processed_data_stream_events.extend(self.__send_data_multiple(new_channel_data.channel, new_channel_data.data_and_metadata, count, update_in_place=True))
        else:
            # special case for camera compatibility. cameras should not return empty dimensions.
            if data_and_metadata.data_shape[-1] == 1:
                data_metadata = data_and_metadata.data_metadata
                data = data_and_metadata.data
                assert data is not None
                data_and_metadata = DataAndMetadata.new_data_and_metadata(data.squeeze(axis=-1),
                                                                          data_metadata.intensity_calibration,
                                                                          data_metadata.dimensional_calibrations[:-1],
                                                                          data_metadata.metadata,
                                                                          data_metadata.timestamp,
                                                                          DataAndMetadata.DataDescriptor(
                                                                              data_metadata.data_descriptor.is_sequence,
                                                                              data_metadata.data_descriptor.collection_dimension_count,
                                                                              data_metadata.data_descriptor.datum_dimension_count - 1),
                                                                          data_metadata.timezone,
                                                                          data_metadata.timezone_offset)
            processed_data_stream_events.extend(self.__send_data_multiple(channel, data_and_metadata, count))
        return processed_data_stream_events

    def __send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, update_in_place: bool = False) -> typing.Sequence[DataStreamEventArgs]:
        new_data_metadata, new_data = data_and_metadata.data_metadata, data_and_metadata.data
        new_count: typing.Optional[int] = None
        new_source_slice: typing.Tuple[slice, ...]
        # special case for scalar
        if new_data_metadata.data_descriptor.expected_dimension_count == 0:
            new_data = numpy.array([new_data])
            assert len(new_data.shape) == 1
            new_count = new_data.shape[0]
        assert new_data is not None
        # form the new slice
        new_source_slice = (slice(0, new_data.shape[0]),) + (slice(None),) * (len(new_data.shape) - 1)
        # send the new data chunk
        new_data_stream_event = DataStreamEventArgs(channel, new_data_metadata, new_data, new_count, new_source_slice, DataStreamStateEnum.COMPLETE, update_in_place)
        return [new_data_stream_event]

    def __send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int, update_in_place: bool = False) -> typing.Sequence[DataStreamEventArgs]:
        assert data_and_metadata.is_sequence
        new_data_descriptor = DataAndMetadata.DataDescriptor(False, data_and_metadata.collection_dimension_count, data_and_metadata.datum_dimension_count)
        data_dtype = data_and_metadata.data_dtype
        assert data_dtype is not None
        new_data_metadata = DataAndMetadata.DataMetadata(
            data_shape=data_and_metadata.data_shape[1:], data_dtype=data_dtype,
            intensity_calibration=data_and_metadata.intensity_calibration,
            dimensional_calibrations=data_and_metadata.dimensional_calibrations[1:],
            metadata=data_and_metadata.metadata,
            timestamp=data_and_metadata.timestamp,
            data_descriptor=new_data_descriptor,
            timezone=data_and_metadata.timezone,
            timezone_offset=data_and_metadata.timezone_offset)
        new_source_slice = (slice(0, count),) + (slice(None),) * len(data_and_metadata.data_shape[1:])
        data = data_and_metadata.data
        assert data is not None
        new_data_stream_event = DataStreamEventArgs(channel, new_data_metadata, data, count, new_source_slice, DataStreamStateEnum.COMPLETE, update_in_place)
        return [new_data_stream_event]

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        framed_data_handler = FramedDataHandler(Framer(DataAndMetadataDataChannel()), operator=self.__operator)
        # send the result of the framed data handler to data_handler
        framed_data_handler.connect_data_handler(data_handler)
        # let the child data stream build its data handler or attach the handler to this data stream.
        if not self.__data_stream.build_data_handler(framed_data_handler):
            self.__data_stream.connect_unprocessed_data_handler(framed_data_handler)
        return True


class MaskLike(typing.Protocol):
    def get_mask_array(self, data_shape: ShapeType) -> _NDArray: ...


AxisType = typing.Union[int, typing.Tuple[int, ...]]


class SumOperator(DataStreamOperator):
    def __init__(self, *, axis: typing.Optional[AxisType] = None) -> None:
        super().__init__()
        self.__axis = axis

    def __str__(self) -> str:
        return "sum"

    def __deepcopy__(self, memo: typing.Dict[typing.Any, typing.Any]) -> SumOperator:
        return SumOperator(axis=self.__axis)

    @property
    def axis(self) -> typing.Optional[AxisType]:
        return self.__axis

    def transform_data_stream_info(self, channel: Channel, data_stream_info: DataStreamInfo) -> DataStreamInfo:
        data_metadata = data_stream_info.data_metadata
        assert not data_metadata.data_descriptor.is_sequence
        assert not data_metadata.data_descriptor.is_collection
        old_shape = data_metadata.data_shape
        old_dimension_calibrations = list(data_metadata.dimensional_calibrations)
        axes = set()
        if isinstance(self.__axis, int):
            axes.add(self.__axis)
        elif isinstance(self.__axis, tuple):
            axes.update(set(self.__axis))
        else:
            axes.update(set(range(len(old_shape))))
        new_shape = list()
        new_dimensional_calibrations = list()
        for i in range(len(old_shape)):
            if i not in axes:
                new_shape.append(old_shape[i])
                new_dimensional_calibrations.append(old_dimension_calibrations[i])
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(new_shape))
        data_dtype = data_metadata.data_dtype
        assert data_dtype is not None
        data_metadata = DataAndMetadata.DataMetadata(
            data_shape=tuple(new_shape), data_dtype=data_dtype,
            intensity_calibration=data_metadata.intensity_calibration,
            dimensional_calibrations=new_dimensional_calibrations,
            metadata=data_metadata.metadata,
            timestamp=data_metadata.timestamp,
            data_descriptor=data_descriptor,
            timezone=data_metadata.timezone,
            timezone_offset=data_metadata.timezone_offset
        )
        return DataStreamInfo(data_metadata, data_stream_info.duration)

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        data_and_metadata = channel_data.data_and_metadata
        if self.__axis is not None:
            summed_xdata = xd.sum(data_and_metadata, self.__axis)
            assert summed_xdata
            summed_xdata._set_metadata(data_and_metadata.metadata)
            summed_xdata.timestamp = data_and_metadata.timestamp
            summed_xdata.timezone = data_and_metadata.timezone
            summed_xdata.timezone_offset = data_and_metadata.timezone_offset
            return [ChannelData(channel_data.channel, summed_xdata)]
        else:
            data = data_and_metadata.data
            assert data is not None
            data_dtype = data_and_metadata.data_dtype
            assert data_dtype is not None
            summed_data: numpy.typing.NDArray[typing.Any] = numpy.array(data.sum(), dtype=data_dtype)
            summed_xdata = DataAndMetadata.new_data_and_metadata(summed_data,
                                                                 intensity_calibration=data_and_metadata.intensity_calibration,
                                                                 data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 0),
                                                                 metadata=data_and_metadata.metadata,
                                                                 timestamp=data_and_metadata.timestamp,
                                                                 timezone=data_and_metadata.timezone,
                                                                 timezone_offset=data_and_metadata.timezone_offset)
            return [ChannelData(channel_data.channel, summed_xdata)]


class MaskedSumOperator(DataStreamOperator):
    def __init__(self, mask: MaskLike) -> None:
        super().__init__()
        self.__mask = mask

    def __str__(self) -> str:
        return "masked"

    @property
    def mask(self) -> MaskLike:
        return self.__mask

    def transform_data_stream_info(self, channel: Channel, data_stream_info: DataStreamInfo) -> DataStreamInfo:
        data_metadata = data_stream_info.data_metadata
        data_metadata = DataAndMetadata.DataMetadata(
            data_shape=(), data_dtype=data_metadata.data_dtype,
            intensity_calibration=data_metadata.intensity_calibration,
            dimensional_calibrations=None,
            metadata=data_metadata.metadata,
            timestamp=data_metadata.timestamp,
            data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 0),
            timezone=data_metadata.timezone,
            timezone_offset=data_metadata.timezone_offset
        )
        return DataStreamInfo(data_metadata, data_stream_info.duration)

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        data_and_metadata = channel_data.data_and_metadata
        mask_array = self.__mask.get_mask_array(data_and_metadata.data_shape)
        data = data_and_metadata.data
        summed_data = numpy.array((data * mask_array).sum(), dtype=data_and_metadata.data_dtype)  # type: ignore
        summed_xdata = DataAndMetadata.new_data_and_metadata(summed_data,
                                                             intensity_calibration=data_and_metadata.intensity_calibration,
                                                             data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 0),
                                                             metadata=data_and_metadata.metadata,
                                                             timestamp=data_and_metadata.timestamp,
                                                             timezone=data_and_metadata.timezone,
                                                             timezone_offset=data_and_metadata.timezone_offset)
        return [ChannelData(channel_data.channel, summed_xdata)]


class MoveAxisDataStreamOperator(DataStreamOperator):
    def __init__(self, channel: typing.Optional[Channel] = None) -> None:
        super().__init__()
        self.__channel = channel

    def __str__(self) -> str:
        return f"move-axis"

    def transform_data_stream_info(self, channel: Channel, data_stream_info: DataStreamInfo) -> DataStreamInfo:
        if self.__channel is None or channel == self.__channel:
            data_metadata = data_stream_info.data_metadata
            assert not data_metadata.data_descriptor.is_sequence
            assert data_metadata.data_descriptor.is_collection
            assert data_metadata.data_descriptor.datum_dimension_count == 1
            data_descriptor = DataAndMetadata.DataDescriptor(False, 1, data_metadata.data_descriptor.collection_dimension_count)
            new_shape = data_metadata.data_shape[-1:] + data_metadata.data_shape[:-1]
            new_dimensional_calibrations = tuple(data_metadata.dimensional_calibrations)[-1:] + tuple(data_metadata.dimensional_calibrations)[:-1]
            data_dtype = data_metadata.data_dtype
            assert data_dtype is not None
            data_metadata = DataAndMetadata.DataMetadata(
                data_shape=tuple(new_shape), data_dtype=data_dtype,
                intensity_calibration=data_metadata.intensity_calibration,
                dimensional_calibrations=new_dimensional_calibrations,
                metadata=data_metadata.metadata,
                timestamp=data_metadata.timestamp,
                data_descriptor=data_descriptor,
                timezone=data_metadata.timezone,
                timezone_offset=data_metadata.timezone_offset
            )
            return DataStreamInfo(data_metadata, data_stream_info.duration)
        else:
            return data_stream_info

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        if self.__channel is None or channel_data.channel == self.__channel:
            data_and_metadata = channel_data.data_and_metadata
            data_metadata = data_and_metadata.data_metadata
            data = data_and_metadata.data
            assert data is not None
            moved_data = numpy.moveaxis(data, -1, 0)
            data_descriptor = DataAndMetadata.DataDescriptor(False, 1, data_metadata.data_descriptor.collection_dimension_count)
            new_dimensional_calibrations = tuple(data_metadata.dimensional_calibrations)[-1:] + tuple(data_metadata.dimensional_calibrations)[:-1]
            moved_xdata = DataAndMetadata.new_data_and_metadata(moved_data,
                                                                intensity_calibration=data_and_metadata.intensity_calibration,
                                                                dimensional_calibrations=new_dimensional_calibrations,
                                                                data_descriptor=data_descriptor,
                                                                metadata=data_and_metadata.metadata,
                                                                timestamp=data_and_metadata.timestamp,
                                                                timezone=data_and_metadata.timezone,
                                                                timezone_offset=data_and_metadata.timezone_offset)
            return [ChannelData(channel_data.channel, moved_xdata)]
        else:
            return [channel_data]


class ContainerDataStream(DataStream):
    """An abstract class to contain another data stream and facilitate decoration in subclasses."""
    def __init__(self, data_stream: DataStream) -> None:
        super().__init__()
        self.__data_stream = data_stream

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return (self.__data_stream,)

    @property
    def data_stream(self) -> DataStream:
        return self.__data_stream

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return self.__data_stream.channels

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        return self.__data_stream.get_info(channel)

    @property
    def is_finished(self) -> bool:
        return self.__data_stream.is_finished

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        return self.__data_stream.process_raw_stream_events(raw_data_stream_events)

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]:
        return self.__data_stream.get_raw_data_stream_events()

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        self.__data_stream.prepare_stream(stream_args, index_stack, **kwargs)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__data_stream.start_stream(stream_args)

    def _advance_stream(self) -> None:
        self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        self.__data_stream.finish_stream()

    def _get_serial_data_handler(self) -> typing.Tuple[SerialDataHandler, bool]:
        return self.__data_stream._get_serial_data_handler()

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        return self.data_stream.build_data_handler(data_handler)

    def connect_unprocessed_data_handler(self, data_handler: DataHandler) -> None:
        self.data_stream.connect_unprocessed_data_handler(data_handler)


class ActionValueControllerLike(typing.Protocol):
    """A delegate to perform the action.

    Subclasses can override start, perform, and finish.
    """

    # start the value control procedure; save any values that need to be restored later.
    def start(self, **kwargs: typing.Any) -> None:
        return

    # perform the control with the given index. return action state metadata if desired.
    def perform(self, index: ShapeType, **kwargs: typing.Any) -> DataAndMetadata.MetadataType:
        return dict()

    # finish the procedure; restore any values that need to be restored.
    def finish(self, **kwargs: typing.Any) -> None:
        return


class ActionDataStreamFnDelegate(ActionValueControllerLike):
    def __init__(self, fn: typing.Callable[[ShapeType], DataAndMetadata.MetadataType]) -> None:
        self.__fn = fn

    def perform(self, index: ShapeType, **kwargs: typing.Any) -> DataAndMetadata.MetadataType:
        return self.__fn(index)


def make_action_data_stream_delegate(fn: typing.Callable[[typing.Sequence[int]], DataAndMetadata.MetadataType]) -> ActionValueControllerLike:
    return ActionDataStreamFnDelegate(fn)


class ActionDataStream(ContainerDataStream):
    """Action data stream. Runs action on each complete frame, passing coordinates."""

    def __init__(self, data_stream: DataStream, delegate: ActionValueControllerLike) -> None:
        super().__init__(data_stream)
        self.__delegate = delegate
        self.__shape = typing.cast(ShapeType, (0,))
        self.__index = 0
        self.__channel_count = len(self.channels)
        self.__complete_channel_count = 0
        self.__action_state: DataAndMetadata.MetadataType = dict()

    def _prepare_stream(self, stream_args: DataStreamArgs, index_stack: IndexDescriptionList, **kwargs: typing.Any) -> None:
        # stream_args are reused, so make a copy before modifying.
        stream_args_copy = copy.copy(stream_args)
        stream_args_copy.max_count = 1
        super()._prepare_stream(stream_args_copy, index_stack, **kwargs)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__shape = stream_args.shape
        self.__index = 0
        self.__complete_channel_count = self.__channel_count
        self.__action_state = dict()
        self.__delegate.start()
        self.__check_action()
        # stream_args are reused, so make a copy before modifying.
        stream_args_copy = copy.copy(stream_args)
        stream_args_copy.max_count = 1
        super()._start_stream(stream_args_copy)

    def __check_action(self) -> None:
        if not self.is_finished and not self.is_aborted:
            if self.__complete_channel_count == self.__channel_count:
                # only proceed if all channels have completed the frame and index is in range.
                if self.__index < numpy.prod(self.__shape, dtype=numpy.uint64):
                    c = better_unravel_index(self.__index, self.__shape)
                    self.__action_state = self.__delegate.perform(c)
                    self.__index += 1
                    self.__complete_channel_count = 0

    def _advance_stream(self) -> None:
        self.__check_action()
        super()._advance_stream()

    def _finish_stream(self) -> None:
        self.__delegate.finish()
        super()._finish_stream()

    def _process_raw_stream_events(self, raw_data_stream_events: typing.Sequence[typing.Tuple[weakref.ReferenceType[DataStream], DataStreamEventArgs]]) -> typing.Sequence[DataStreamEventArgs]:
        data_stream_events = super()._process_raw_stream_events(raw_data_stream_events)
        for data_stream_event in data_stream_events:
            metadata = dict(data_stream_event.data_metadata.metadata)
            metadata["action_state"] = copy.deepcopy(self.__action_state)
            data_stream_event.data_metadata._set_metadata(metadata)
        return data_stream_events

    def _handle_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        assert data_stream_event.count is None or data_stream_event.count == 1
        super()._handle_data_available(data_stream_event)
        if data_stream_event.state == DataStreamStateEnum.COMPLETE:
            assert (data_stream_event.count or 1) == 1
            self.__complete_channel_count += 1


class AccumulatedDataStream(ContainerDataStream):
    """Change a data stream producing a sequence into an accumulated non-sequence.
    """

    def __init__(self, data_stream: DataStream, include_raw: bool, include_sum: bool) -> None:
        super().__init__(data_stream)
        self.__include_raw = include_raw
        self.__include_sum = include_sum

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        raw_channels = list(super().channels)
        channels = list[Channel]()
        if self.__include_sum:
            for channel in list(raw_channels):
                channels.append(Channel(*channel.segments, "sum"))
        if self.__include_raw:
            for channel in list(raw_channels):
                channels.append(channel)
        return tuple(channels)

    def _is_channel_finished(self, channel: Channel) -> bool:
        return super()._is_channel_finished(channel)

    def _get_info(self, channel: Channel) -> DataStreamInfo:
        data_stream_info = super()._get_info(channel)
        if channel.segments[-1] == "sum":
            data_stream_info = super()._get_info(Channel(*channel.segments[:-1]))
            data_metadata = data_stream_info.data_metadata
            return DataStreamInfo(DataAndMetadata.DataMetadata(
                data_shape=data_metadata.data_shape[1:],
                data_dtype=data_metadata.data_dtype,
                intensity_calibration=data_metadata.intensity_calibration,
                dimensional_calibrations=data_metadata.dimensional_calibrations[1:],
                metadata=data_metadata.metadata,
                timestamp=data_metadata.timestamp,
                data_descriptor=DataAndMetadata.DataDescriptor(False, data_metadata.collection_dimension_count, data_metadata.datum_dimension_count),
                timezone=data_metadata.timezone,
                timezone_offset=data_metadata.timezone_offset
            ), 0)
        else:
            return data_stream_info

    def _build_data_handler(self, data_handler: DataHandler) -> bool:
        data_handlers = list[DataHandler]()

        if self.__include_raw:
            null_data_handler = NullDataHandler()
            null_data_handler.connect_data_handler(data_handler)
            data_handlers.append(null_data_handler)

        if self.__include_sum:
            accumulated_framed_data_handler = FramedDataHandler(Framer(DataAndMetadataDataChannel()), force_count=True)
            accumulated_framed_data_handler.connect_data_handler(data_handler)
            accumulated_data_handler = AccumulatedDataHandler(True)
            accumulated_data_handler.connect_data_handler(accumulated_framed_data_handler)
            data_handlers.append(accumulated_data_handler)

        splitting_data_handler = SplittingDataHandler(data_handlers)

        serial_data_handler, created = self.data_stream._get_serial_data_handler()
        serial_data_handler.add_data_handler(splitting_data_handler, 1)
        if created:
            if not self.data_stream.build_data_handler(serial_data_handler):
                self.data_stream.connect_unprocessed_data_handler(serial_data_handler)

        # self.data_stream.connect_unprocessed_data_handler(splitting_data_handler)

        return True


class DataHandler:
    def __init__(self) -> None:
        self.__data_handler: typing.Optional[DataHandler] = None

    def connect_data_handler(self, data_handler: DataHandler) -> None:
        assert not self.__data_handler
        self.__data_handler = data_handler

    def handle_data_available(self, packet: DataStreamEventArgs) -> None:
        raise NotImplementedError()

    def send_packet(self, packet: DataStreamEventArgs) -> None:
        if self.__data_handler:
            self.__data_handler.handle_data_available(packet)


class NullDataHandler(DataHandler):
    def __init__(self) -> None:
        super().__init__()

    def handle_data_available(self, packet: DataStreamEventArgs) -> None:
        self.send_packet(packet)


class SplittingDataHandler(DataHandler):
    def __init__(self, data_handlers: typing.Sequence[DataHandler]) -> None:
        super().__init__()
        self.__data_handlers = list(data_handlers)

    def handle_data_available(self, packet: DataStreamEventArgs) -> None:
        for data_handler in self.__data_handlers:
            data_handler.handle_data_available(packet)


class SerialDataHandler(DataHandler):
    """A data handler that sends data to a list of handlers in sequence.

    The data is not reshaped in any way.
    """
    def __init__(self) -> None:
        super().__init__()
        self.__data_handlers = list[DataHandler]()
        self.__counts = list[int]()
        self.__indexes = dict[Channel, int]()

    def add_data_handler(self, data_handler: DataHandler, count: int) -> None:
        self.__data_handlers.append(data_handler)
        self.__counts.append(count)

    def handle_data_available(self, packet: DataStreamEventArgs) -> None:
        # send the data to the appropriate data handler.
        # the data handler will be determined by the index.
        # update the indexes for handling the next packet.
        index = self.__indexes.get(packet.channel, 0)
        current_index = index
        for i, count in enumerate(self.__counts):
            if current_index < count:
                self.__data_handlers[i].handle_data_available(packet)
                self.__indexes[packet.channel] = (index + (packet.count or 1)) % sum(self.__counts)
                break
            current_index -= count


class CollectionDataHandler(DataHandler):
    """A data handler that collects data and sends it out as a collection.

    The top/height/total_height can be used to filter the data that is collected. This is required when stacked data is
    being collected and multiple collections get sent into the stacked data handler. The top/height/total_height
    ensures that only one of the feeder data handlers is active at a time.
    """
    def __init__(self, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration]) -> None:
        super().__init__()
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        self.__indexes = dict[Channel, int]()
        self.__collection_list = list[DataAndMetadata.MetadataType]()
        self.__last_collection_index: typing.Optional[ShapeType] = None

    def get_info(self, channel: Channel, data_stream_info: DataStreamInfo) -> DataStreamInfo:
        count = expand_shape(self.__collection_shape)
        data_metadata = data_stream_info.data_metadata
        data_dtype = data_metadata.data_dtype
        assert data_dtype is not None
        data_metadata = DataAndMetadata.DataMetadata(
            data_shape=self.__collection_shape + data_metadata.data_shape, data_dtype=data_dtype,
            intensity_calibration=data_metadata.intensity_calibration,
            dimensional_calibrations=list(self.__collection_calibrations) + list(data_metadata.dimensional_calibrations),
            metadata=data_metadata.metadata,
            timestamp=data_metadata.timestamp,
            data_descriptor=self._get_new_data_descriptor(data_metadata),
            timezone=data_metadata.timezone,
            timezone_offset=data_metadata.timezone_offset
        )
        return DataStreamInfo(data_metadata, count * data_stream_info.duration)

    def handle_data_available(self, packet: DataStreamEventArgs) -> None:
        collection_count = expand_shape(self.__collection_shape)
        assert self.__indexes.get(packet.channel, 0) < collection_count
        # this will update the index too.
        self.__process_packet(packet)

    def _get_new_data_descriptor(self, data_metadata: DataAndMetadata.DataMetadata) -> DataAndMetadata.DataDescriptor:
        # subclasses can override this method to provide different collection shapes.

        # collections of sequences and collections of collections are not currently supported
        assert not data_metadata.is_sequence
        assert not data_metadata.is_collection

        # new data descriptor is a collection
        collection_rank = len(self.__collection_shape)
        datum_dimension_count = data_metadata.datum_dimension_count

        if datum_dimension_count > 0:
            return DataAndMetadata.DataDescriptor(False, collection_rank, datum_dimension_count)
        else:
            return DataAndMetadata.DataDescriptor(False, 0, collection_rank)

    def __process_packet(self, data_stream_event: DataStreamEventArgs) -> None:
        # when data arrives, put it into the sequence/collection and send it out again.
        # data will be arriving as either partial data or as frame data. partial data
        # is restricted to arrive in groups that are multiples of the product of all
        # dimensions except the first one or with a count of exactly one. frame data
        # is restricted to arrive in groups that are multiples of the collection size
        # and cannot overlap the end of a collection chunk.

        results = list[DataStreamEventArgs]()

        # useful variables
        data_metadata = data_stream_event.data_metadata
        channel = data_stream_event.channel
        collection_count = expand_shape(self.__collection_shape)
        index = self.__indexes.get(channel, 0)

        # ensure that it is not complete already.
        assert index < collection_count

        # for each data stream event, add the collection list to the metadata if action_state is present.
        metadata = dict(data_metadata.metadata)
        action_state = typing.cast(typing.MutableMapping[str, typing.Any], metadata.pop("action_state", dict()))
        if action_state:
            collection_index = better_unravel_index(index, self.__collection_shape)
            if collection_index != self.__last_collection_index:
                action_state["index"] = collection_index
                self.__collection_list.append(action_state)
                self.__last_collection_index = collection_index
            collection_list = copy.deepcopy(self.__collection_list)
            metadata["collection"] = collection_list
            data_stream_event.data_metadata._set_metadata(metadata)

        # get the new data descriptor
        new_data_descriptor = self._get_new_data_descriptor(data_metadata)

        # add the collection count to the downstream data shape to produce the new collection shape.
        new_shape = self.__collection_shape + tuple(data_metadata.data_shape)

        # add designated calibrations for the new collection dimensions.
        new_dimensional_calibrations = self.__collection_calibrations + tuple(data_metadata.dimensional_calibrations)

        # create a new data metadata object
        dtype = data_metadata.data_dtype
        assert dtype is not None
        new_data_metadata = DataAndMetadata.DataMetadata(data_shape=new_shape, data_dtype=dtype,
                                                         intensity_calibration=data_metadata.intensity_calibration,
                                                         dimensional_calibrations=new_dimensional_calibrations,
                                                         metadata=data_metadata.metadata,
                                                         timestamp=data_metadata.timestamp,
                                                         data_descriptor=new_data_descriptor,
                                                         timezone=data_metadata.timezone,
                                                         timezone_offset=data_metadata.timezone_offset)

        collection_rank = len(self.__collection_shape)
        collection_row_length = expand_shape(self.__collection_shape[1:])
        if data_stream_event.count is not None:
            remaining_count = data_stream_event.count
            current_index = index
            # incoming data is frames
            # count must either be a multiple of last dimensions of collection shape or one
            assert remaining_count == data_stream_event.source_slice[0].stop - data_stream_event.source_slice[0].start
            assert current_index + remaining_count <= collection_count
            assert remaining_count > 0
            next_index = index + remaining_count
            if remaining_count == 1:
                # a single data chunk has been provided.
                # if the count is one, add new dimensions of length one for the collection shape and form
                # the new slice with indexes of 0 for each collection dimension and full slices for the remaining
                # dimensions.
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                new_source_data = old_source_data.reshape((1,) * collection_rank + old_source_data.shape[1:])
                new_source_slice = (index_slice(0), ) * collection_rank + (slice(None),) * (len(old_source_data.shape) - 1)
                new_state = DataStreamStateEnum.COMPLETE if current_index + remaining_count == collection_count else DataStreamStateEnum.PARTIAL
                results.append(DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
            else:
                # multiple data chunks have been provided.
                # if the count is greater than one, provide the "rows" of the collection. row is just first dimension.
                # start by calculating the start/stop rows. reshape the data into collection shape and add a slice
                # for the rows and full slices for the remaining dimensions.
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                source_index = 0
                if current_index % collection_row_length != 0:
                    # finish the current row
                    assert len(self.__collection_shape) == 2
                    slice_column = current_index % collection_row_length
                    slice_width = min(collection_row_length - slice_column, remaining_count)
                    new_source_slice = (slice(0, 1), slice(0, slice_width)) + (slice(None),) * (len(new_shape) - 2)
                    next_source_index = source_index + slice_width
                    new_source_data = old_source_data[source_index:next_source_index].reshape((1, slice_width) + old_source_data.shape[1:])
                    new_state = DataStreamStateEnum.COMPLETE if current_index + slice_width == collection_count else DataStreamStateEnum.PARTIAL
                    results.append(DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                    source_index = next_source_index
                    current_index += slice_width
                    remaining_count -= slice_width
                    assert remaining_count is not None  # for type checker bug
                else:
                    new_state = DataStreamStateEnum.COMPLETE  # satisfy type checker
                if remaining_count // collection_row_length > 0:
                    # send as many complete rows as possible
                    slice_offset = current_index // collection_row_length
                    slice_start = better_unravel_index(current_index, self.__collection_shape)[0] - slice_offset
                    slice_stop = better_unravel_index(current_index + remaining_count, self.__collection_shape)[0] - slice_offset
                    assert 0 <= slice_start <= self.__collection_shape[0]
                    assert 0 <= slice_stop <= self.__collection_shape[0]
                    row_count = remaining_count // collection_row_length
                    next_source_index = source_index + row_count * collection_row_length
                    new_source_data = old_source_data[source_index:next_source_index].reshape((row_count,) + self.__collection_shape[1:] + old_source_data.shape[1:])
                    new_source_slice = (slice(slice_start, slice_stop),) + (slice(None),) * (len(new_shape) - 1)
                    new_state = DataStreamStateEnum.COMPLETE if current_index + row_count * collection_row_length == collection_count else DataStreamStateEnum.PARTIAL
                    results.append(DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                    source_index = next_source_index
                    current_index += row_count * collection_row_length
                    remaining_count -= row_count * collection_row_length
                    assert remaining_count is not None  # for type checker bug
                if remaining_count > 0:
                    # any remaining count means a partial row
                    assert len(self.__collection_shape) == 2
                    assert remaining_count < collection_row_length
                    new_source_slice = (slice(0, 1), slice(0, remaining_count)) + (slice(None),) * (len(new_shape) - 2)
                    next_source_index = source_index + remaining_count
                    new_source_data = old_source_data[source_index:next_source_index].reshape((1, remaining_count) + old_source_data.shape[1:])
                    new_state = DataStreamStateEnum.PARTIAL  # always partial, otherwise would have been sent in previous section
                    results.append(DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                    # source_index = next_source_index  # no need for this
                    current_index += remaining_count
                    remaining_count -= remaining_count
                    assert remaining_count is not None  # for type checker bug
                assert remaining_count == 0  # everything has been accounted for
        else:
            # incoming data is partial
            # form the new slice with indexes of 0 for each collection dimension and the incoming slice for the
            # remaining dimensions. add dimensions for collection dimensions to the new source data.
            new_source_slice = (index_slice(0), ) * collection_rank + tuple(data_stream_event.source_slice)
            new_source_data = data_stream_event.source_data.reshape((1,) * collection_rank + data_stream_event.source_data.shape)
            # new state is complete if data chunk is complete and it will put us at the end of our collection.
            new_state = DataStreamStateEnum.PARTIAL
            next_index = index
            if data_stream_event.state == DataStreamStateEnum.COMPLETE:
                next_index += 1
                if next_index == collection_count:
                    new_state = DataStreamStateEnum.COMPLETE
            results.append(DataStreamEventArgs(channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
        if results and results[-1].state == DataStreamStateEnum.COMPLETE:
            assert next_index == collection_count
            self.__indexes[channel] = 0
        else:
            assert next_index < collection_count
            self.__indexes[channel] = next_index
        for data_stream_event in results:
            self.send_packet(data_stream_event)


class SequenceDataHandler(CollectionDataHandler):
    def __init__(self, count: int, calibration: Calibration.Calibration) -> None:
        super().__init__((count,), (calibration,))

    def _get_new_data_descriptor(self, data_metadata: DataAndMetadata.DataMetadata) -> DataAndMetadata.DataDescriptor:
        # scalar data is not supported. and the data must not be a sequence already.
        assert not data_metadata.is_sequence
        assert data_metadata.datum_dimension_count > 0

        # new data descriptor is a sequence
        collection_dimension_count = data_metadata.collection_dimension_count
        datum_dimension_count = data_metadata.datum_dimension_count
        return DataAndMetadata.DataDescriptor(True, collection_dimension_count, datum_dimension_count)


class FramedDataHandler(DataHandler):
    def __init__(self, framer: Framer, *, operator: typing.Optional[DataStreamOperator] = None, force_count: bool = False) -> None:
        super().__init__()
        self.__framer = framer
        self.__operator = operator or NullDataStreamOperator()
        self.sent_bytes = 0
        # force count is used to force a single frame to be counted. used for accumulated data handlers.
        # force count is on/off but kept track of whether it has been applied per channel.
        self.__force_count = force_count
        self.__force_counts = dict[Channel, bool]()

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__framer.get_data(channel)

    def handle_data_available(self, packet: DataStreamEventArgs) -> None:
        if packet.count_frame:
            # count the bytes in the packet, for progress tracking.
            self.sent_bytes += packet.total_bytes
        # the framer will call back to the callbacks _send_data and _send_data_multiple
        # the framer will pass the appropriate count_frame setting to the callbacks, essentially whatever is specified
        # in the packet.
        # the count_frame is used when a frame is overwritten, so it only gets counted once.
        self.__framer.data_available(packet, typing.cast(FrameCallbacks, self))

    def _send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count_frame: bool) -> typing.Sequence[DataStreamEventArgs]:
        # callback for Framer
        if not self.__operator.is_applied:
            for new_channel_data in self.__operator.process(ChannelData(channel, data_and_metadata)):
                self.__send_data(new_channel_data.channel, new_channel_data.data_and_metadata, count_frame)
        else:
            self.__send_data(channel, data_and_metadata, count_frame)
        return list()

    def _send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int, count_frame: bool) -> typing.Sequence[DataStreamEventArgs]:
        # callback for Framer
        if not self.__operator.is_applied:
            for new_channel_data in self.__operator.process_multiple(ChannelData(channel, data_and_metadata)):
                self.__send_data_multiple(new_channel_data.channel, new_channel_data.data_and_metadata, count, count_frame)
        else:
            # special case for camera compatibility. cameras should not return empty dimensions.
            if data_and_metadata.data_shape[-1] == 1:
                data_metadata = data_and_metadata.data_metadata
                data = data_and_metadata.data
                assert data is not None
                data_and_metadata = DataAndMetadata.new_data_and_metadata(data.squeeze(axis=-1),
                                                                          data_metadata.intensity_calibration,
                                                                          data_metadata.dimensional_calibrations[:-1],
                                                                          data_metadata.metadata,
                                                                          data_metadata.timestamp,
                                                                          DataAndMetadata.DataDescriptor(
                                                                              data_metadata.data_descriptor.is_sequence,
                                                                              data_metadata.data_descriptor.collection_dimension_count,
                                                                              data_metadata.data_descriptor.datum_dimension_count - 1),
                                                                          data_metadata.timezone,
                                                                          data_metadata.timezone_offset)
            self.__send_data_multiple(channel, data_and_metadata, count, count_frame)
        return list()

    def __send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count_frame: bool) -> None:
        new_data_metadata, new_data = data_and_metadata.data_metadata, data_and_metadata.data
        new_count: typing.Optional[int] = None
        new_source_slice: typing.Tuple[slice, ...]
        # special case for scalar
        if new_data_metadata.data_descriptor.expected_dimension_count == 0:
            new_data = numpy.array([new_data])
            assert len(new_data.shape) == 1
            new_count = new_data.shape[0]
        assert new_data is not None
        # form the new slice
        new_source_slice = (slice(0, new_data.shape[0]),) + (slice(None),) * (len(new_data.shape) - 1)
        # send the new data chunk
        new_data_stream_event = DataStreamEventArgs(channel, new_data_metadata, new_data, new_count, new_source_slice, DataStreamStateEnum.COMPLETE)
        new_data_stream_event.count_frame = count_frame
        if self.__force_counts.get(channel, self.__force_count):
            self.__force_counts[channel] = False
            new_data_stream_event.count_frame = True
        # TODO: what about update in place?
        self.send_packet(new_data_stream_event)

    def __send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int, count_frame: bool) -> None:
        assert data_and_metadata.is_sequence
        new_data_descriptor = DataAndMetadata.DataDescriptor(False, data_and_metadata.collection_dimension_count, data_and_metadata.datum_dimension_count)
        data_dtype = data_and_metadata.data_dtype
        assert data_dtype is not None
        new_data_metadata = DataAndMetadata.DataMetadata(
            data_shape=data_and_metadata.data_shape[1:], data_dtype=data_dtype,
            intensity_calibration=data_and_metadata.intensity_calibration,
            dimensional_calibrations=data_and_metadata.dimensional_calibrations[1:],
            metadata=data_and_metadata.metadata,
            timestamp=data_and_metadata.timestamp,
            data_descriptor=new_data_descriptor,
            timezone=data_and_metadata.timezone,
            timezone_offset=data_and_metadata.timezone_offset)
        new_source_slice = (slice(0, count),) + (slice(None),) * len(data_and_metadata.data_shape[1:])
        data = data_and_metadata.data
        assert data is not None
        new_data_stream_event = DataStreamEventArgs(channel, new_data_metadata, data, count, new_source_slice, DataStreamStateEnum.COMPLETE)
        new_data_stream_event.count_frame = count_frame
        # TODO: what about update in place?
        self.send_packet(new_data_stream_event)


class StackedDataHandler(DataHandler):
    """A data handler that stacks count sections into new data with a new index of size height.

    Send 4x10x10 then 2x10x10 to get a 6x10x10 output.
    """
    def __init__(self, count: int, height: int) -> None:
        super().__init__()
        self.__count = count
        self.__height = height
        self.__indexes: typing.Dict[Channel, int] = dict()

    def handle_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        if data_stream_event.state == DataStreamStateEnum.COMPLETE and self.__indexes.get(data_stream_event.channel, 0) + 1 == self.__count:
            state = DataStreamStateEnum.COMPLETE
        else:
            state = DataStreamStateEnum.PARTIAL

        if data_stream_event.state == DataStreamStateEnum.COMPLETE:
            self.__indexes[data_stream_event.channel] = (self.__indexes.get(data_stream_event.channel, 0) + 1) % self.__count

        # print(f"{data_stream_event.state=} {self.__index=}/{self.__count=}")

        data_metadata = copy.deepcopy(data_stream_event.data_metadata)
        data_metadata._set_data_shape_and_dtype(((self.__height,) + data_metadata.data_shape[1:], numpy.dtype(data_metadata.data_dtype)))

        # print(f"{data_stream_event.source_slice=} {data_stream_event.source_data.shape=}")
        # print(f"{data_stream_event.source_data}")

        # create the data stream event with the overridden data_metadata and state.
        data_stream_event = DataStreamEventArgs(
            data_stream_event.channel,
            data_metadata,
            data_stream_event.source_data,
            data_stream_event.count,
            data_stream_event.source_slice,
            state
        )

        self.send_packet(data_stream_event)


class SequentialDataHandler(DataHandler):
    def __init__(self, channels: typing.Sequence[typing.Sequence[Channel]]) -> None:
        super().__init__()
        self.__count = len(channels)
        self.__index = 0
        self.__channels = channels
        self.__channels_complete = [0 for _ in range(self.__count)]

    def handle_data_available(self, packet: DataStreamEventArgs) -> None:
        index = self.__index

        state = DataStreamStateEnum.COMPLETE if packet.state == DataStreamStateEnum.COMPLETE and index + 1 == self.__count else DataStreamStateEnum.PARTIAL
        new_packet = DataStreamEventArgs(
            Channel(str(index), *packet.channel.segments),
            packet.data_metadata,
            packet.source_data,
            packet.count,
            packet.source_slice,
            state
        )

        self.send_packet(new_packet)

        if packet.state == DataStreamStateEnum.COMPLETE:
            self.__channels_complete[index] += 1
            if self.__channels_complete[index] == len(self.__channels[index]):
                self.__index = (index + 1) % self.__count


class AccumulatedDataHandler(DataHandler):
    def __init__(self, do_rename: bool = False) -> None:
        super().__init__()
        self.__data_channel = DataAndMetadataDataChannel()
        self.__dest_indexes = dict[Channel, int]()
        self.__do_rename = do_rename
        # handle the count_frame state. the first pass through the first frame should be counted. subsequent frames
        # should not be counted. since the first frame is only indicated by the slice start being 0, we need to keep
        # count_frame on until another "first slice" is received. use these two dictionaries to track that state per
        # channel.
        self.__first_pass = dict[Channel, bool]()
        self.__had_first_slice = dict[Channel, bool]()

    def handle_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        count = data_stream_event.count
        assert count is None
        old_data_metadata = data_stream_event.data_metadata
        data_descriptor = copy.deepcopy(old_data_metadata.data_descriptor)
        assert data_descriptor.is_sequence
        data_descriptor.is_sequence = False
        sequence_slice = data_stream_event.source_slice[0]
        for sequence_index in range(sequence_slice.start, sequence_slice.stop):
            sequence_slices = (slice(sequence_index, sequence_index + 1),) + data_stream_event.source_slice[1:]
            data_dtype = old_data_metadata.data_dtype
            assert data_dtype is not None
            data_metadata = DataAndMetadata.DataMetadata(
                data_shape=tuple(old_data_metadata.data_shape[1:]), data_dtype=data_dtype,
                intensity_calibration=old_data_metadata.intensity_calibration,
                dimensional_calibrations=old_data_metadata.dimensional_calibrations[1:],
                metadata=old_data_metadata.metadata,
                timestamp=old_data_metadata.timestamp,
                data_descriptor=data_descriptor,
                timezone=old_data_metadata.timezone,
                timezone_offset=old_data_metadata.timezone_offset
            )
            channel = data_stream_event.channel
            dest_count = expand_shape(data_metadata.data_shape)
            sequence_slice_offset = sequence_index * dest_count
            source_start = ravel_slice_start(sequence_slices, data_stream_event.source_data.shape) - sequence_slice_offset
            source_stop = ravel_slice_stop(sequence_slices, data_stream_event.source_data.shape) - sequence_slice_offset
            dest_slice_offest = self.__dest_indexes.get(channel, 0)
            dest_slice = slice(dest_slice_offest + source_start, dest_slice_offest + source_stop)
            self.__dest_indexes[channel] = dest_slice.stop % dest_count
            new_source_slices = simple_unravel_flat_slice(dest_slice, data_metadata.data_shape)
            assert len(new_source_slices) == 1
            new_source_slice = new_source_slices[0]
            self.__data_channel.accumulate_data(channel, data_stream_event.source_data[sequence_index],
                                                sequence_slices[1:], dest_slice, data_metadata)
            data_channel_data = self.__data_channel.get_data(channel).data
            assert data_channel_data is not None
            new_channel = channel if not self.__do_rename else Channel(*channel.segments, "sum")
            new_data_stream_event = DataStreamEventArgs(new_channel, data_metadata,
                                                        data_channel_data, None, new_source_slice,
                                                        data_stream_event.state)
            new_data_stream_event.reset_frame = dest_slice.start == 0
            new_data_stream_event.update_in_place = True
            # ensure count_frame is set to true for the first frame. see notes above.
            if dest_slice.start == 0 and self.__had_first_slice.get(channel, False):
                self.__first_pass[channel] = False
            new_data_stream_event.count_frame = self.__first_pass.get(channel, True)
            self.__had_first_slice[channel] = True
            self.send_packet(new_data_stream_event)


class MakerDataStream(ContainerDataStream):
    """A utility data stream to allow access to acquired data during tests."""

    def __init__(self, data_stream: DataStream) -> None:
        super().__init__(data_stream)
        self.__framer = Framer(DataAndMetadataDataChannel())
        self.__framed_data_handler = FramedDataHandler(self.__framer)
        data_stream.attach_root_data_handler(self.__framed_data_handler)

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__framer.get_data(channel)

    @property
    def progress(self) -> float:
        # override progress to use the internal frame data handler instead of the attached one (which won't be attached during tests).
        return self.__framed_data_handler.sent_bytes / self._total_bytes if self._total_bytes else 1.0


def acquire(data_stream: DataStream, *, error_handler: typing.Optional[typing.Callable[[Exception], None]] = None) -> None:
    """Perform an acquire. This is the main acquisition loop. It runs on a thread.

    Performs consistency checks on progress and data.

    Progress must be made once per 60s or else an exception is thrown.
    """
    TIMEOUT = 60.0
    data_stream.prepare_stream(DataStreamArgs((1,)), [])
    try:
        data_stream.start_stream(DataStreamArgs((1,)))
        try:
            last_progress = 0.0
            last_progress_time = time.time()

            # useful for debugging
            # data_stream._print()
            # print(data_stream.channels)

            # assert isinstance(data_stream, FramedDataStream)

            while not data_stream.is_finished and not data_stream.is_aborted:
                # progress checking is for tests and self-consistency
                pre_progress = data_stream.progress
                raw_data_stream_events = data_stream.get_raw_data_stream_events()
                for data_stream_ref, raw_data_stream_event in raw_data_stream_events:
                    if data_stream_ := data_stream_ref():
                        if not data_stream_.is_finished and not data_stream_.is_aborted:
                            data_stream_.send_raw_data_stream_event_to_data_handler(raw_data_stream_event)
                            last_progress_time = time.time()
                data_stream.process_raw_stream_events(raw_data_stream_events)
                post_progress = data_stream.progress
                data_stream.advance_stream()
                next_progress = data_stream.progress
                assert pre_progress <= post_progress <= next_progress, f"{pre_progress=} <= {post_progress=} <= {next_progress=}"
                assert next_progress >= last_progress
                if next_progress > last_progress:
                    last_progress = next_progress
                    last_progress_time = time.time()
                assert time.time() - last_progress_time < TIMEOUT
                time.sleep(0.005)  # play nice with other threads
            if data_stream.is_finished:
                # ensure that the data stream is finished. when things go wrong here, it is usually because a stream
                # is reporting the wrong total number of bytes, the wrong number of bytes was received,
                # or the counting logic is wrong.
                # assert data_stream.progress == 1.0, f"{data_stream.progress=}"
                if data_stream.progress != 1.0:
                    logging.warning(f"Data stream progress should be 1.0 [{data_stream.progress}]")
        except Exception as e:
            data_stream.is_error = True
            data_stream.abort_stream()
            raise
        finally:
            data_stream.finish_stream()
    except Exception as e:
        from nion.swift.model import Notification
        Notification.notify(Notification.Notification("nion.acquisition.error", "\N{WARNING SIGN} Acquisition", "Acquisition Failed", str(e)))
        if error_handler:
            error_handler(e)
        else:
            import traceback
            traceback.print_exc()


class Acquisition:
    def __init__(self, data_stream: DataStream, framer: Framer) -> None:
        self.__data_stream = data_stream
        self.__framer = framer
        self.__task: typing.Optional[asyncio.Task[None]] = None
        self.__is_aborted = False
        self.__is_error = False
        self.__is_finished = False
        self.__device_state: typing.Optional[DeviceState] = None

    def save_device_state(self) -> None:
        assert self.__device_state is None
        self.__device_state = self.__data_stream.prepare_device_state()
        time.sleep(0.5)

    def restore_device_state(self) -> None:
        assert self.__device_state is not None
        self.__device_state.restore()

    def prepare_acquire(self) -> None:
        # this is called on the main thread. give data channel a chance to prepare.
        self.__framer.prepare({channel: self.__data_stream.get_info(channel) for channel in self.__data_stream.channels})

    def acquire(self, *, error_handler: typing.Optional[typing.Callable[[Exception], None]] = None) -> None:
        try:
            acquire(self.__data_stream, error_handler=error_handler)
            self.__is_aborted = self.__data_stream.is_aborted
            self.__is_error = self.__data_stream.is_error
            self.__is_finished = True
        finally:
            self.__data_stream = typing.cast(typing.Any, None)

    async def grab_async(self, *, on_completion: typing.Callable[[], None], error_handler: typing.Optional[typing.Callable[[Exception], None]] = None) -> None:
        try:
            self.prepare_acquire()

            def call_acquire() -> None:
                self.acquire(error_handler=error_handler)

            await asyncio.get_running_loop().run_in_executor(None, call_acquire)
        finally:
            on_completion()
            self.__task = None

    def acquire_async(self, *, event_loop: asyncio.AbstractEventLoop, on_completion: typing.Callable[[], None], error_handler: typing.Optional[typing.Callable[[Exception], None]] = None) -> None:
        self.__task = event_loop.create_task(self.grab_async(on_completion=on_completion, error_handler=error_handler))

    def abort_acquire(self) -> None:
        if self.__data_stream:
            self.__data_stream.abort_stream()

    def wait_acquire(self, timeout: float = 60.0, *, on_periodic: typing.Callable[[], None]) -> None:
        start = time.time()
        while self.__task and not self.__task.done() and time.time() - start < timeout:
            on_periodic()
            time.sleep(0.005)  # don't take all the CPU
        on_periodic()  # one more periodic for clean up

    @property
    def progress(self) -> float:
        if self.__data_stream:
            return self.__data_stream.progress
        return 0.0

    @property
    def is_aborted(self) -> bool:
        return self.__is_aborted

    @property
    def is_error(self) -> bool:
        return self.__is_error

    @property
    def is_finished(self) -> bool:
        return self.__is_finished


class _MultiSectionLike(typing.Protocol):
    offset: float
    exposure: float
    count: int
    include_sum: bool


class AcquisitionDeviceLike(typing.Protocol):
    def build_acquisition_device_data_stream(self, device_map: typing.MutableMapping[str, STEMController.DeviceController]) -> DataStream: ...


class AcquisitionMethodLike(typing.Protocol):
    def wrap_acquisition_device_data_stream(self, device_data_stream: DataStream, device_map: typing.Mapping[str, STEMController.DeviceController]) -> DataStream: ...


class BasicAcquisitionMethod(AcquisitionMethodLike):
    def __init__(self, title_base: typing.Optional[str] = None) -> None:
        self.__title_base = title_base or str()

    def wrap_acquisition_device_data_stream(self, device_data_stream: DataStream, device_map: typing.Mapping[str, STEMController.DeviceController]) -> DataStream:
        device_data_stream.title = self.__title_base
        return device_data_stream


class SequenceAcquisitionMethod(AcquisitionMethodLike):
    def __init__(self, count: int) -> None:
        self.__count = count

    def wrap_acquisition_device_data_stream(self, device_data_stream: DataStream, device_map: typing.Mapping[str, STEMController.DeviceController]) -> DataStream:
        count = self.__count
        # given an acquisition data stream, wrap this acquisition method around the acquisition data stream.
        if count > 1:
            # special case for framed-data-stream with sum operator; in the frame-by-frame case, the camera device
            # has the option of doing the processing itself and the operator will not be applied to the result. in
            # this case, the framed-data-stream-with-sum-operator is wrapped so that the processing can be performed
            # on the entire sequence. there is probably a better way to abstract this in the future.
            if isinstance(device_data_stream, FramedDataStream) and isinstance(device_data_stream.operator, SumOperator):
                sequence_data_stream = device_data_stream.data_stream.wrap_in_sequence(count)
                sequence_data_stream.channel_names = device_data_stream.channel_names
                sequence_data_stream.title = _("Sequence")
                return sequence_data_stream
            else:
                sequence_data_stream = device_data_stream.wrap_in_sequence(count)
                sequence_data_stream.channel_names = device_data_stream.channel_names
                sequence_data_stream.title = _("Sequence")
                return sequence_data_stream
        else:
            device_data_stream.title = _("Sequence")
            return device_data_stream


class ControlCustomizationValueController(ActionValueControllerLike):
    def __init__(self,
                 device_controller: STEMController.DeviceController,
                 control_customization: AcquisitionPreferences.ControlCustomization,
                 values: numpy.typing.NDArray[typing.Any],
                 axis: typing.Optional[STEMController.AxisType]) -> None:
        self.__device_controller = device_controller
        self.__control_customization = control_customization
        self.__values = values
        self.__axis = axis
        self.__original_values: typing.Sequence[float] = list()

    def start(self, **kwargs: typing.Any) -> None:
        self.__original_values = self.__device_controller.get_values(self.__control_customization, self.__axis)

    # define an action function to apply control values during acquisition
    def perform(self, index: ShapeType, **kwargs: typing.Any) -> DataAndMetadata.MetadataType:
        # calculate the current value (in each dimension) and send the result to the
        # device controller. the device controller may be a camera, scan, or stem device
        # controller.
        self.__device_controller.update_values(self.__control_customization,
                                               self.__original_values,
                                               typing.cast(typing.Sequence[float], self.__values[index]),
                                               self.__axis)
        action_state = dict[str, typing.Any]()
        action_state["control_value"] = tuple(self.__values[index])
        control_description = self.__control_customization.control_description
        if control_description:
            action_state["control_id"] = control_description.control_id,
            action_state["device_id"] = control_description.device_id
            action_state["device_control_id"] = control_description.device_control_id,
            if axis := control_description.axis:
                action_state["axis"] = axis
        return action_state

    def finish(self, **kwargs: typing.Any) -> None:
        self.__device_controller.set_values(self.__control_customization, self.__original_values, self.__axis)


class ValueControllersActionValueController(ActionValueControllerLike):
    def __init__(self, value_controllers: typing.Sequence[ActionValueControllerLike]) -> None:
        self.__value_controllers = list(value_controllers)

    def start(self, **kwargs: typing.Any) -> None:
        for value_controller in self.__value_controllers:
            value_controller.start()

    # define an action function to apply control values during acquisition
    def perform(self, index: ShapeType, **kwargs: typing.Any) -> DataAndMetadata.MetadataType:
        action_states = list[DataAndMetadata.MetadataType]()
        for value_controller in self.__value_controllers:
            action_states.append(value_controller.perform(index))
        if action_states:
            return {"controls": action_states}
        return dict()

    def finish(self, **kwargs: typing.Any) -> None:
        for value_controller in self.__value_controllers:
            value_controller.finish()


class SeriesAcquisitionMethod(AcquisitionMethodLike):
    def __init__(self, control_customization: AcquisitionPreferences.ControlCustomization, control_values: numpy.typing.NDArray[typing.Any]) -> None:
        assert control_values.ndim == 2
        assert control_values.shape[-1] == 1
        self.__control_customization = control_customization
        self.__control_values = control_values

    def wrap_acquisition_device_data_stream(self, device_data_stream: DataStream, device_map: typing.Mapping[str, STEMController.DeviceController]) -> DataStream:
        control_customization = self.__control_customization
        # given an acquisition data stream, wrap this acquisition method around the acquisition data stream.
        # get the associated control handler that was created in create_handler and used within the stack
        # of control handlers declarative components.
        control_description = control_customization.control_description
        assert control_description
        device_controller = device_map.get(control_description.device_id)
        if device_controller:
            value_controller = ControlCustomizationValueController(device_controller, control_customization, self.__control_values, None)
            action_delegate = ValueControllersActionValueController([value_controller])
            device_data_stream = SequenceDataStream(ActionDataStream(device_data_stream, action_delegate), self.__control_values.shape[0])
            device_data_stream.channel_names = device_data_stream.channel_names
            device_data_stream.title = _("Series")
        return device_data_stream


class TableAcquisitionMethod(AcquisitionMethodLike):
    def __init__(self, control_customization: AcquisitionPreferences.ControlCustomization, axis_id: typing.Optional[str], control_values: numpy.typing.NDArray[typing.Any]) -> None:
        assert control_values.ndim == 3
        assert control_values.shape[-1] == 2
        self.__control_customization = control_customization
        self.__axis_id = axis_id
        self.__control_values = control_values

    def wrap_acquisition_device_data_stream(self, device_data_stream: DataStream, device_map: typing.Mapping[str, STEMController.DeviceController]) -> DataStream:
        control_customization = self.__control_customization
        axis_id = self.__axis_id
        # given a acquisition data stream, wrap this acquisition method around the acquisition data stream.
        control_description = control_customization.control_description
        assert control_description
        device_controller = device_map.get(control_description.device_id)
        if device_controller:
            axis = typing.cast(STEMController.STEMDeviceController, device_map["stem"]).stem_controller.resolve_axis(axis_id)
            value_controller = ControlCustomizationValueController(device_controller, control_customization, self.__control_values, axis)
            action_delegate = ValueControllersActionValueController([value_controller])
            device_data_stream = CollectedDataStream(
                ActionDataStream(device_data_stream, action_delegate),
                self.__control_values.shape[0:2],
                (Calibration.Calibration(), Calibration.Calibration()))
            device_data_stream.channel_names = device_data_stream.channel_names
            device_data_stream.title = _("Tableau")
        return device_data_stream


class MultipleAcquisitionMethod(AcquisitionMethodLike):
    def __init__(self, sections: typing.Sequence[_MultiSectionLike]) -> None:
        self.__sections = sections

    def wrap_acquisition_device_data_stream(self, device_data_stream: DataStream, device_map: typing.Mapping[str, STEMController.DeviceController]) -> DataStream:
        sections = self.__sections
        # given an acquisition data stream, wrap this acquisition method around the acquisition data stream.
        assert AcquisitionPreferences.acquisition_preferences
        # define a list of data streams that will be acquired sequentially.
        data_streams: typing.List[DataStream] = list()
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
        for multi_acquire_entry in sections:
            value_controllers: typing.List[ActionValueControllerLike] = list()
            if stem_value_controller:
                values = numpy.array([[multi_acquire_entry.offset * control_description_energy_offset.multiplier]] * multi_acquire_entry.count)
                value_controllers.append(
                    ControlCustomizationValueController(stem_value_controller, control_customization_energy_offset, values, None))
            if camera_value_controller:
                values = numpy.array([[multi_acquire_entry.exposure * control_description_exposure.multiplier]] * multi_acquire_entry.count)
                value_controllers.append(
                    ControlCustomizationValueController(camera_value_controller, control_customization_exposure, values, None))
            # copy the original device stream when constructing the sequence data stream. this is a workaround for
            # the problem of sequencing complex data handler trees. without this, the low level streams send data to
            # multiple collectors simultaneously and this causes problems with counting.
            sequence_data_stream = SequenceDataStream(
                ActionDataStream(copy.deepcopy(device_data_stream), ValueControllersActionValueController(value_controllers)),
                max(1, multi_acquire_entry.count), include_sum=multi_acquire_entry.include_sum)
            data_streams.append(sequence_data_stream)

        # create a sequential data stream from the section data streams.
        sequential_data_stream = SequentialDataStream(data_streams)
        # the sequential data stream will emit channels of the form n.sub-channel. add a name for each of those
        # channels. do this by getting the name of the sub channel and constructing a new name for n.sub_channel
        # for each index.
        channel_names = dict(device_data_stream.channel_names)
        for channel in sequential_data_stream.channels:
            channel_ = Channel(*channel.segments[1:])
            channel_name = channel_names.get(channel_, "Sum")
            channel_names[channel] = " ".join((f"{int(channel.segments[0]) + 1} / {str(len(data_streams))}", channel_name))
        sequential_data_stream.channel_names = channel_names
        sequential_data_stream.title = _("Multiple")
        return sequential_data_stream


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


class DataChannelProviderLike(typing.Protocol):
    def get_data_channel(self, title_base: str, channel_names: typing.Mapping[Channel, str], **kwargs: typing.Any) -> DataChannel: ...


def _acquire_data_stream(data_stream: DataStream,
                         data_channel: DataChannel,
                         progress_value_model: Model.PropertyModel[int],
                         is_acquiring_model: Model.PropertyModel[bool],
                         scan_drift_logger: typing.Optional[DriftTracker.DriftLogger],
                         event_loop: asyncio.AbstractEventLoop,
                         completion_fn: typing.Callable[[], None],
                         *, error_handler: typing.Optional[typing.Callable[[Exception], None]] = None,
                         title_base: typing.Optional[str] = None,
                         ) -> Acquisition:
    """Perform acquisition of the data stream."""
    title = _("Acquisition")
    if title_base:
        title += f" ({title_base})"

    logger = logging.getLogger("acquisition")

    logger.info(f"{title} started: {datetime.datetime.now()}")

    framer = Framer(data_channel)

    framed_data_handler = FramedDataHandler(framer)

    data_stream.attach_root_data_handler(framed_data_handler)

    # create the acquisition state/controller object based on the data item data channel data stream.
    acquisition = Acquisition(data_stream, framer)
    acquisition.save_device_state()

    # define a method that gets called when the async acquisition method finished. this closes the various
    # objects and updates the UI as 'complete'.
    def finish_grab_async(acquisition: Acquisition,
                          scan_drift_logger: typing.Optional[DriftTracker.DriftLogger],
                          progress_task: typing.Optional[asyncio.Task[None]],
                          progress_value_model: Model.PropertyModel[int],
                          is_acquiring_model: Model.PropertyModel[bool]) -> None:
        acquisition.restore_device_state()
        logger.info(f"{title} finished: {datetime.datetime.now()}" + (" canceled" if acquisition.is_aborted else "") + (" with error" if acquisition.is_error else ""))
        if scan_drift_logger:
            scan_drift_logger.close()
        is_acquiring_model.value = False
        if progress_task:
            progress_task.cancel()
        progress_value_model.value = 100
        completion_fn()

    # manage the 'is_acquiring' state.
    is_acquiring_model.value = True

    # define a task to update progress every 250ms.
    async def update_progress(acquisition: Acquisition, progress_value_model: Model.PropertyModel[int]) -> None:
        while True:
            try:
                progress = acquisition.progress
                progress_value_model.value = int(100 * progress)
                await asyncio.sleep(0.25)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

    progress_task = asyncio.get_event_loop_policy().get_event_loop().create_task(update_progress(acquisition, progress_value_model))

    # start async acquire.
    acquisition.acquire_async(event_loop=event_loop,
                              on_completion=functools.partial(finish_grab_async,
                                                              acquisition,
                                                              scan_drift_logger, progress_task,
                                                              progress_value_model,
                                                              is_acquiring_model),
                              error_handler=error_handler)

    return acquisition


def start_acquire(data_stream: DataStream,
                  title_base: str,
                  channel_names: typing.Mapping[Channel, str],
                  data_channel_provider: DataChannelProviderLike,
                  drift_logger: typing.Optional[DriftTracker.DriftLogger],
                  progress_value_model: Model.PropertyModel[int],
                  is_acquiring_model: Model.PropertyModel[bool],
                  event_loop: asyncio.AbstractEventLoop,
                  completion_fn: typing.Callable[[], None],
                  *, error_handler: typing.Optional[typing.Callable[[Exception], None]] = None,
                  ) -> Acquisition:
    return _acquire_data_stream(data_stream,
                                data_channel_provider.get_data_channel(title_base, channel_names),
                                progress_value_model,
                                is_acquiring_model,
                                drift_logger,
                                event_loop,
                                completion_fn,
                                error_handler=error_handler,
                                title_base=title_base)



def acquire_immediate(data_stream: DataStream) -> typing.Mapping[Channel, DataAndMetadata.DataAndMetadata]:
    framer = Framer(DataAndMetadataDataChannel())
    data_stream.attach_root_data_handler(FramedDataHandler(framer))
    acquire(data_stream)
    return {channel: framer.get_data(channel) for channel in data_stream.channels}


class LinearSpace:
    # an object representing a linear space of values.
    # may be moved to niondata eventually.

    def __init__(self, start: float, stop: float, num: int) -> None:
        self.start = start
        self.stop = stop
        self.num = num

    def __array__(self, dtype: typing.Optional[numpy.typing.DTypeLike] = None, copy: typing.Optional[bool] = None) -> numpy.typing.NDArray[typing.Any]:
        return numpy.linspace(self.start, self.stop, self.num, dtype=dtype)


class MeshGrid:
    # an object representing a mesh of y/x values.
    # may be moved to niondata eventually.

    def __init__(self, y_space: numpy.typing.ArrayLike, x_space: numpy.typing.ArrayLike) -> None:
        self.y_space = y_space
        self.x_space = x_space

    def __array__(self, dtype: typing.Optional[numpy.typing.DTypeLike] = None, copy: typing.Optional[bool] = None) -> numpy.typing.NDArray[typing.Any]:
        return numpy.stack(numpy.meshgrid(numpy.array(self.y_space), numpy.array(self.x_space), indexing='ij'), axis=-1)


class AcquisitionProcedureFactoryInterface(typing.Protocol):

    class Device(typing.Protocol): pass

    class DeviceChannelSpecifier(typing.Protocol): pass

    class ProcessingChannelLike(typing.Protocol): pass

    class DeviceParametersLike(typing.Protocol): pass

    class ProcedureStepLike(typing.Protocol): pass

    class DeviceAcquisitionStep(ProcedureStepLike): pass

    class MultiDeviceAcquisitionStep(ProcedureStepLike): pass

    class ScanParameters(DeviceParametersLike): pass

    class CameraParameters(DeviceParametersLike): pass

    class DeviceAcquisitionParameters(typing.Protocol): pass

    class DriftCorrectionParameters(typing.Protocol): pass

    class ControlController(typing.Protocol): pass

    class DeviceControlController(ControlController): pass

    class CollectionStep(ProcedureStepLike): pass

    class SequentialStep(ProcedureStepLike): pass

    class AcquisitionProcedure(typing.Protocol): pass

    class AcquisitionController(typing.Protocol):
        def acquire_immediate(self) -> typing.Mapping[Channel, DataAndMetadata.DataAndMetadata]: ...

    def create_device(self, device_type_id: str, *, device_id: typing.Optional[str] = None) -> AcquisitionProcedureFactoryInterface.Device: ...

    def create_stem_device(self, *, device_id: typing.Optional[str] = None) -> AcquisitionProcedureFactoryInterface.Device: ...

    def create_scan_device(self, *, device_id: typing.Optional[str] = None) -> AcquisitionProcedureFactoryInterface.Device: ...

    def create_ronchigram_device(self, *, device_id: typing.Optional[str] = None) -> AcquisitionProcedureFactoryInterface.Device: ...

    def create_eels_device(self, *, device_id: typing.Optional[str] = None) -> AcquisitionProcedureFactoryInterface.Device: ...

    def create_device_channel_specifier(self, *,
                                        channel_index: typing.Optional[int] = None,
                                        channel_type_id: typing.Optional[str] = None,
                                        channel_id: typing.Optional[str] = None) -> AcquisitionProcedureFactoryInterface.DeviceChannelSpecifier: ...

    def create_scan_parameters(self, *,
                               pixel_time_us: typing.Optional[float] = None,
                               pixel_size: typing.Optional[Geometry.IntSize] = None,
                               fov_nm: typing.Optional[float] = None,
                               rotation_rad: typing.Optional[float] = None,
                               center_nm: typing.Optional[Geometry.FloatPoint] = None,
                               subscan_pixel_size: typing.Optional[Geometry.IntSize] = None,
                               subscan_fractional_size: typing.Optional[Geometry.FloatSize] = None,
                               subscan_fractional_center: typing.Optional[Geometry.FloatPoint] = None,
                               subscan_rotation: typing.Optional[float] = None,
                               ac_line_sync: typing.Optional[bool] = None,
                               # flyback_time_us: typing.Optional[float] = None,
                               # ac_frame_sync: typing.Optional[bool] = None,
                               **kwargs: typing.Any,
                               ) -> AcquisitionProcedureFactoryInterface.ScanParameters: ...

    def create_camera_parameters(self, *,
                                 exposure_ms: typing.Optional[float] = None,
                                 binning: typing.Optional[int] = None,
                                 **kwargs: typing.Any,
                                 ) -> AcquisitionProcedureFactoryInterface.CameraParameters: ...

    def create_drift_parameters(self, *,
                                drift_correction_enabled: bool = False,
                                drift_interval_lines: int = 0,
                                drift_scan_lines: int = 0,
                                drift_channel: typing.Optional[AcquisitionProcedureFactoryInterface.DeviceChannelSpecifier] = None,
                                drift_region: typing.Optional[Geometry.FloatRect] = None,
                                drift_rotation: float = 0.0,
                                **kwargs: typing.Any,
                                ) -> AcquisitionProcedureFactoryInterface.DriftCorrectionParameters: ...

    def create_device_acquisition_parameters(self, *,
                                             device: AcquisitionProcedureFactoryInterface.Device,
                                             device_parameters: typing.Optional[AcquisitionProcedureFactoryInterface.DeviceParametersLike] = None,
                                             device_channels: typing.Optional[typing.Sequence[AcquisitionProcedureFactoryInterface.DeviceChannelSpecifier]] = None,
                                             processing_channels: typing.Optional[typing.Sequence[AcquisitionProcedureFactoryInterface.ProcessingChannelLike]] = None
                                             ) -> AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters: ...

    def create_device_acquisition_step(self, *,
                                       device_acquisition_parameters: AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters,
                                       ) -> AcquisitionProcedureFactoryInterface.DeviceAcquisitionStep: ...

    def create_multi_device_acquisition_step(self, *,
                                             primary_device_acquisition_parameters: AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters,
                                             secondary_device_acquisition_parameters: typing.Sequence[AcquisitionProcedureFactoryInterface.DeviceAcquisitionParameters],
                                             drift_parameters: typing.Optional[AcquisitionProcedureFactoryInterface.DriftCorrectionParameters] = None) -> AcquisitionProcedureFactoryInterface.MultiDeviceAcquisitionStep: ...

    def create_device_controller(self, *,
                                 device: AcquisitionProcedureFactoryInterface.Device,
                                 control_id: str,
                                 device_control_id: typing.Optional[str] = None,
                                 values: typing.Optional[numpy.typing.ArrayLike] = None,
                                 delay: typing.Optional[float] = None,
                                 axis_id: typing.Optional[str] = None) -> AcquisitionProcedureFactoryInterface.DeviceControlController: ...

    def create_collection_step(self, *,
                               sub_step: AcquisitionProcedureFactoryInterface.ProcedureStepLike,
                               control_controller: AcquisitionProcedureFactoryInterface.ControlController
                               ) -> AcquisitionProcedureFactoryInterface.CollectionStep: ...

    def create_sequential_step(self, *,
                               sub_steps: typing.Sequence[AcquisitionProcedureFactoryInterface.ProcedureStepLike]
                               ) -> AcquisitionProcedureFactoryInterface.SequentialStep: ...

    def create_acquisition_procedure(self, *,
                                     devices: typing.Sequence[AcquisitionProcedureFactoryInterface.Device],
                                     steps: typing.Sequence[AcquisitionProcedureFactoryInterface.ProcedureStepLike]
                                     ) -> AcquisitionProcedureFactoryInterface.AcquisitionProcedure: ...

    def create_processing_channel(self, *,
                                  processing_id: str,
                                  processing_parameters: typing.Optional[typing.Mapping[str, typing.Any]] = None
                                  ) -> AcquisitionProcedureFactoryInterface.ProcessingChannelLike: ...

    def create_acquisition_controller(self, *, acquisition_procedure: AcquisitionProcedureFactoryInterface.AcquisitionProcedure) -> AcquisitionProcedureFactoryInterface.AcquisitionController: ...


def acquisition_procedure_factory() -> AcquisitionProcedureFactoryInterface:
    return typing.cast(AcquisitionProcedureFactoryInterface, Registry.get_component("acquisition_procedure_factory_interface"))


"""
Architectural Decision Records.

ADR 2021-04-22. FramedDataStream can have operators that go from one channel to multiple channels, but we will not
provide support for multiple channels to one channel since that would require passing full generations of data to
the operator and since data may come in multiple frames at a time, it would require new buffering.

ADR 2021-03-26: CollectedDataStream will be used to break a large acquisition into smaller chunks. Doing this allows
scan devices to have a uniform view of their collection space. Rejected idea is to have a stacking operator and
explicitly configure each scan section; rejected because it would be difficult to have parallel scan and camera
acquisitions and then collect them into a uniform structure.

ADR 2021-02-09: Device and other data streams should be able to produce data in two major ways: partial frame and
multiple frames at once. All incoming streams need to handle both partial and multiple frames. Rejected using only
partial or only multiple frames since there are fewer opportunities for optimization if only one is used.

ADR 2021-02-09: Introduce an acquisition pipeline so that the machinery to collect and collate acquisition data from
devices such as cameras, scans, etc. is abstracted, tested, and optimized independently of the actual acquisition.
The pipeline mainly consists of a standard architecture for sending data from the device to collection and framing
objects which organize and process the data.
"""
