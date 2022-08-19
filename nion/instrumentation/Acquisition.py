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
import enum
import time
import typing
import warnings

import numpy
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import ReferenceCounting
from nion.utils.ReferenceCounting import weak_partial

ShapeType = typing.Sequence[int]
SliceType = typing.Tuple[slice, ...]
SliceListType = typing.Sequence[SliceType]
ChannelSegment = str

_NDArray = numpy.typing.NDArray[typing.Any]


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
    return int(numpy.product(shape, dtype=numpy.int64))


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

    def __init__(self, data_stream: DataStream, channel: Channel, data_metadata: DataAndMetadata.DataMetadata,
                 source_data: _NDArray, count: typing.Optional[int], source_slice: SliceType,
                 state: DataStreamStateEnum) -> None:
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

        # the data stream sending this event
        self.data_stream = data_stream

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


class DataStream(ReferenceCounting.ReferenceCounted):
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
        self.handle_error_event = Event.Event()
        # sequence counts are used for acquiring a sequence of frames controlled by the upstream
        self.__sequence_count = sequence_count
        self.__sequence_counts: typing.Dict[Channel, int] = dict()
        self.__sequence_indexes: typing.Dict[Channel, int] = dict()
        self.is_aborted = False
        self.is_error = False

    def add_ref(self) -> DataStream:
        super().add_ref()
        return self

    def _print(self, indent: typing.Optional[str] = None) -> None:
        indent = indent or str()
        print(f".{indent} {self} [{self.channels} {self.data_shapes} {self.data_types}]")
        for data_stream in self.data_streams:
            data_stream._print(indent + "  ")

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
        return DataStreamInfo(DataAndMetadata.DataMetadata(((), float)), 0.0)

    @property
    def is_finished(self) -> bool:
        """Return true if stream is finished.

        The stream is finished if all channels have sent the number of items in their sequence.
        """
        return all(self.__sequence_indexes.get(channel, 0) == self.__sequence_counts.get(channel, self.__sequence_count) for channel in self.input_channels)

    def _acquire_finished(self) -> None:
        pass

    @property
    def progress(self) -> float:
        return self._progress if not self.is_finished else 1.0

    @property
    def _progress(self) -> float:
        return 0.0

    def abort_stream(self) -> None:
        """Abort the stream. Called to stop the stream. Also called during exceptions."""
        self._abort_stream()
        self.is_aborted = True

    def _abort_stream(self) -> None:
        pass

    def handle_error(self) -> None:
        """Handle an error occurring on this data stream."""
        self.is_error = True
        self.handle_error_event.fire()

    def send_next(self) -> None:
        """Send next data."""
        if not self.is_finished and not self.is_aborted:
            for channel in self.input_channels:
                assert self.__sequence_indexes.get(channel, 0) <= self.__sequence_counts.get(channel, self.__sequence_count)
            self._send_next()

    def _send_next(self) -> None:
        """Used for testing. Send next data. Subclasses can override as required."""
        pass

    def prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        """Prepare stream. Top level prepare_stream is called before start_stream.

        The prepare function allows streams to perform any preparations before any other
        stream has started.
        """
        self._prepare_stream(stream_args, **kwargs)

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
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

    def fire_data_available(self, data_stream_event: DataStreamEventArgs, update_in_place: bool = False) -> None:
        """Fire the data available event.

        update_in_place is a hack to allow for operations such as summing in place to not trigger sequence index updates
        since they are repeating the same update over and over. future plans would be to include an update operation
        with the data stream event, perhaps 'clear', 'replace', 'update', 'final_update' with only 'replace' and
        'final_update' advancing the sequence indexes.
        """
        self._fire_data_available(data_stream_event)
        if data_stream_event.state == DataStreamStateEnum.COMPLETE and not update_in_place:
            count = data_stream_event.count or 1
            channel = data_stream_event.channel
            assert self.__sequence_indexes.get(channel, 0) + count <= self.__sequence_counts.get(channel, self.__sequence_count)
            self.__sequence_indexes[channel] = self.__sequence_indexes.get(channel, 0) + count

    def _fire_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        """Fire the data available event.

        Subclasses can override.
        """
        exceptions: typing.List[Exception] = list()

        self.data_available_event.fire(data_stream_event, exceptions)

        # ensure that exceptions occurring in data_available get raised.
        # this mechanism allows handlers to raise exceptions without regard
        # to how they will be handled through the event.fire call.
        # exceptions raised here are typically caught at the top level handler.
        for e in exceptions:
            raise e

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

    Optionally pass in a list of sub-slices to break collection into sections.
    """

    def __init__(self, data_stream: DataStream, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration]) -> None:
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__data_available_event_listener = typing.cast(Event.EventListener, None)
        self.__handle_error_event_listener = typing.cast(Event.EventListener, None)
        assert len(shape) in (1, 2)
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        # sub-slice indexes track the destination of the next data within the current slice.
        self.__indexes: typing.Dict[Channel, int] = dict()
        # needs starts tracks whether the downstream data stream needs a start call.
        self.__data_stream_started = False
        self.__all_channels_need_start = False

    def about_to_delete(self) -> None:
        if self.__data_available_event_listener:
            self.__data_available_event_listener.close()
            self.__data_available_event_listener = typing.cast(typing.Any, None)
        if self.__handle_error_event_listener:
            self.__handle_error_event_listener.close()
            self.__handle_error_event_listener = typing.cast(typing.Any, None)
        if self.__data_stream_started:
            warnings.warn("Stream deleted but not finished.", category=RuntimeWarning)
        self.__data_stream.remove_ref()
        self.__data_stream = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return (self.__data_stream,)

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return self.__data_stream.channels

    def get_info(self, channel: Channel) -> DataStreamInfo:
        data_stream_info = self.__data_stream.get_info(channel)
        count = expand_shape(self.__collection_shape)
        data_metadata = data_stream_info.data_metadata
        data_dtype = data_metadata.data_dtype
        assert data_dtype is not None
        data_metadata = DataAndMetadata.DataMetadata(
            (self.__collection_shape + data_metadata.data_shape, data_dtype),
            data_metadata.intensity_calibration,
            list(self.__collection_calibrations) + list(data_metadata.dimensional_calibrations),
            data_metadata.metadata,
            data_metadata.timestamp,
            self._get_new_data_descriptor(data_metadata),
            data_metadata.timezone,
            data_metadata.timezone_offset
        )
        return DataStreamInfo(data_metadata, count * data_stream_info.duration)

    @property
    def _progress(self) -> float:
        # p will be the progress for the current frame
        # count is the number of frames in this collection
        # index is the number of frames completed in this collection, calculated as the minimum progress among incomplete channels
        # adding p to index will give the number of frames completed plus the fraction of the current one completed
        # all channels progress simultaneously in a collection; so use the last one for calculation
        count = expand_shape(self.__collection_shape)
        # only add current progress if we're not about to advance to the next sub slice
        p = 0.0 if self.__all_channels_need_start else self.__data_stream.progress
        incomplete_indexes = list(self.__indexes.get(c, 0) for c in self.channels if self.__indexes.get(c, 0) != count)
        if not incomplete_indexes:
            return 0.0
        index = min(incomplete_indexes)
        return (index + p) / count

    def _send_next(self) -> None:
        assert self.__data_stream_started
        self.__data_stream.send_next()

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert not self.__data_available_event_listener
        assert not self.__handle_error_event_listener
        self.__data_available_event_listener = self.__data_stream.data_available_event.listen(weak_partial(_handle_data_available, self, CollectedDataStream.__data_available))
        self.__handle_error_event_listener = self.__data_stream.handle_error_event.listen(weak_partial(DataStream.handle_error, self))
        self._start_next_sub_stream()

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _start_next_sub_stream(self) -> None:
        if self.__data_stream_started:
            self.__data_stream.finish_stream()
            self.__data_stream_started = False
        self.__indexes.clear()
        self.__all_channels_need_start = False
        self.__data_stream.prepare_stream(DataStreamArgs(self.__collection_shape))
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
        self.__data_available_event_listener.close()
        self.__data_available_event_listener = typing.cast(typing.Any, None)
        self.__handle_error_event_listener.close()
        self.__handle_error_event_listener = typing.cast(typing.Any, None)

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

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        # when data arrives, put it into the sequence/collection and send it out again.
        # data will be arriving as either partial data or as frame data. partial data
        # is restricted to arrive in groups that are multiples of the product of all
        # dimensions except the first one or with a count of exactly one. frame data
        # is restricted to arrive in groups that are multiples of the collection size
        # and cannot overlap the end of a collection chunk.

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
        new_data_metadata = DataAndMetadata.DataMetadata((new_shape, dtype),
                                                         data_metadata.intensity_calibration,
                                                         new_dimensional_calibrations,
                                                         data_metadata.metadata,
                                                         data_metadata.timestamp,
                                                         new_data_descriptor,
                                                         data_metadata.timezone,
                                                         data_metadata.timezone_offset)

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
                self.fire_data_available(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
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
                    self.fire_data_available(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
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
                    self.fire_data_available(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
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
                    self.fire_data_available(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
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
            self.fire_data_available(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
        self.__indexes[channel] = next_index
        # whether all channels are in the 'needs_start' state.
        needs_starts = {channel: self.__indexes.get(channel, 0) == collection_count for channel in self.channels}
        self.__all_channels_need_start = all(needs_starts.get(channel, False) for channel in self.input_channels)


class SequenceDataStream(CollectedDataStream):
    """Collect a data stream into a sequence of datums.

    This is a subclass of CollectedDataStream.
    """
    def __init__(self, data_stream: DataStream, count: int, calibration: typing.Optional[Calibration.Calibration] = None) -> None:
        super().__init__(data_stream, (count,), (calibration or Calibration.Calibration(),))

    def _get_new_data_descriptor(self, data_metadata: DataAndMetadata.DataMetadata) -> DataAndMetadata.DataDescriptor:
        # scalar data is not supported. and the data must not be a sequence already.
        assert not data_metadata.is_sequence
        assert data_metadata.datum_dimension_count > 0

        # new data descriptor is a sequence
        collection_dimension_count = data_metadata.collection_dimension_count
        datum_dimension_count = data_metadata.datum_dimension_count
        return DataAndMetadata.DataDescriptor(True, collection_dimension_count, datum_dimension_count)


class CombinedDataStream(DataStream):
    """Combine multiple streams into a single stream producing multiple channels.

    Each stream can also produce multiple channels.
    """
    def __init__(self, data_streams: typing.Sequence[DataStream]) -> None:
        super().__init__()
        self.__data_streams = [data_stream.add_ref() for data_stream in data_streams]
        self.__data_available_event_listeners: typing.List[Event.EventListener] = list()
        self.__handle_error_event_listeners: typing.List[Event.EventListener] = list()

    def about_to_delete(self) -> None:
        for listener in self.__data_available_event_listeners:
            listener.close()
        self.__data_available_event_listeners = typing.cast(typing.Any, None)
        for listener in self.__handle_error_event_listeners:
            listener.close()
        self.__handle_error_event_listeners = typing.cast(typing.Any, None)
        for data_stream in self.__data_streams:
            data_stream.remove_ref()
        self.__data_streams = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return tuple(self.__data_streams)

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        channels: typing.List[Channel] = list()
        for data_stream in self.__data_streams:
            channels.extend(data_stream.channels)
        return tuple(channels)

    def get_info(self, channel: Channel) -> DataStreamInfo:
        for data_stream in self.__data_streams:
            if channel in data_stream.channels:
                return data_stream.get_info(channel)
        assert False, f"No info for channel {channel}"

    @property
    def is_finished(self) -> bool:
        return all(data_stream.is_finished for data_stream in self.__data_streams)

    @property
    def _progress(self) -> float:
        # return the average of combined streams progress
        return sum(data_stream.progress for data_stream in self.__data_streams) / len(self.__data_streams)

    def _send_next(self) -> None:
        for data_stream in self.__data_streams:
            data_stream.send_next()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        for data_stream in self.__data_streams:
            data_stream.prepare_stream(stream_args)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert not self.__data_available_event_listeners
        assert not self.__handle_error_event_listeners
        self.__data_available_event_listeners = [data_stream.data_available_event.listen(weak_partial(_handle_data_available, self, CombinedDataStream.__data_available)) for data_stream in self.__data_streams]
        self.__handle_error_event_listeners = [data_stream.handle_error_event.listen(weak_partial(DataStream.handle_error, self)) for data_stream in self.__data_streams]
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
        for listener in self.__data_available_event_listeners:
            listener.close()
        self.__data_available_event_listeners = list()
        for listener in self.__handle_error_event_listeners:
            listener.close()
        self.__handle_error_event_listeners = list()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        self.fire_data_available(data_stream_event)


class StackedDataStream(DataStream):
    """Acquire multiple streams and stack the results.

    Each stream can produce multiple channels but the channels must match between streams.
    """
    def __init__(self, data_streams: typing.Sequence[DataStream]) -> None:
        super().__init__()
        self.__data_streams: typing.List[DataStream] = [data_stream.add_ref() for data_stream in data_streams]
        self.__data_available_event_listener = typing.cast(Event.EventListener, None)
        self.__handle_error_event_listener = typing.cast(Event.EventListener, None)
        self.__stream_args = DataStreamArgs(list())
        self.__current_index = 0
        assert len(set(data_stream.channels for data_stream in self.__data_streams)) == 1
        self.__channels = self.__data_streams[0].channels
        self.__sequence_count = 0
        self.__sequence_index = 0
        # confirm that the contained data streams are sensible (same shape except for height, same dtype, same descriptor)
        for channel in self.__channels:
            data_metadata = self.__data_streams[0].get_info(channel).data_metadata
            for data_stream in self.__data_streams:
                data_stream_data_metadata = data_stream.get_info(channel).data_metadata
                assert data_metadata.data_shape[1:] == data_stream_data_metadata.data_shape[1:]
                assert data_metadata.data_dtype == data_stream_data_metadata.data_dtype
                assert data_metadata.data_descriptor == data_stream_data_metadata.data_descriptor

    def about_to_delete(self) -> None:
        if self.__data_available_event_listener:
            self.__data_available_event_listener.close()
            self.__data_available_event_listener = typing.cast(typing.Any, None)
        if self.__handle_error_event_listener:
            self.__handle_error_event_listener.close()
            self.__handle_error_event_listener = typing.cast(typing.Any, None)
        for data_stream in self.__data_streams:
            data_stream.remove_ref()
        self.__data_streams = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return tuple(self.__data_streams)

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return self.__channels

    def get_info(self, channel: Channel) -> DataStreamInfo:
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
        data_metadata.data_shape_and_dtype = ((height,) + data_metadata.data_shape[1:], numpy.dtype(data_metadata.data_dtype))
        return DataStreamInfo(data_metadata, duration)

    @property
    def is_finished(self) -> bool:
        return self.__current_index == len(self.__data_streams)

    @property
    def _progress(self) -> float:
        # return the average of combined streams progress
        if self.__sequence_count:
            return (self.__sequence_index + (self.__current_index + self.__data_streams[self.__current_index].progress) / len(self.__data_streams)) / self.__sequence_count
        return 0.0

    def _send_next(self) -> None:
        self.__data_streams[self.__current_index].send_next()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        self.__current_index = 0
        self.__stream_args = DataStreamArgs((1,))
        self.__data_streams[self.__current_index].prepare_stream(self.__stream_args)
        self.__sequence_count = stream_args.sequence_count
        self.__sequence_index = 0

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert self.__current_index == 0
        assert not self.__data_available_event_listener
        assert not self.__handle_error_event_listener
        self.__data_available_event_listener = self.__data_streams[self.__current_index].data_available_event.listen(weak_partial(_handle_data_available, self, StackedDataStream.__data_available))
        self.__handle_error_event_listener = self.__data_streams[self.__current_index].handle_error_event.listen(weak_partial(DataStream.handle_error, self))
        self.__data_streams[self.__current_index].start_stream(self.__stream_args)

    def _abort_stream(self) -> None:
        self.__data_streams[self.__current_index].abort_stream()

    def _advance_stream(self) -> None:
        # handle calling finish and start for the contained data stream.
        if self.__current_index < len(self.__data_streams):
            if self.__data_streams[self.__current_index].is_finished:
                self.__data_streams[self.__current_index].finish_stream()
                self.__data_available_event_listener.close()
                self.__data_available_event_listener = typing.cast(typing.Any, None)
                self.__handle_error_event_listener.close()
                self.__handle_error_event_listener = typing.cast(typing.Any, None)
                self.__current_index += 1
                if self.__current_index == len(self.__data_streams) and self.__sequence_index + 1 < self.__sequence_count:
                    self.__current_index = 0
                    self.__sequence_index += 1
                if self.__current_index < len(self.__data_streams):
                    self.__data_streams[self.__current_index].prepare_stream(self.__stream_args)
                    self.__data_available_event_listener = self.__data_streams[self.__current_index].data_available_event.listen(weak_partial(_handle_data_available, self, StackedDataStream.__data_available))
                    self.__handle_error_event_listener = self.__data_streams[self.__current_index].handle_error_event.listen(weak_partial(DataStream.handle_error, self))
                    self.__data_streams[self.__current_index].start_stream(self.__stream_args)

        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].advance_stream()

    def _finish_stream(self) -> None:
        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].finish_stream()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        state = DataStreamStateEnum.COMPLETE if data_stream_event.state == DataStreamStateEnum.COMPLETE and self.__current_index + 1 == len(self.__data_streams) else DataStreamStateEnum.PARTIAL
        # compute height as a sum from contained data streams and configure a new data_metadata
        data_metadata = copy.deepcopy(data_stream_event.data_metadata)
        height = 0
        for data_stream in self.__data_streams:
            data_stream_data_metadata = data_stream.get_info(data_stream_event.channel).data_metadata
            height += data_stream_data_metadata.data_shape[0]
        data_metadata.data_shape_and_dtype = ((height,) + data_metadata.data_shape[1:], numpy.dtype(data_metadata.data_dtype))
        # create the data stream event with the overridden data_metadata and state.
        data_stream_event = DataStreamEventArgs(
            data_stream_event.data_stream,
            data_stream_event.channel,
            data_metadata,
            data_stream_event.source_data,
            data_stream_event.count,
            data_stream_event.source_slice,
            state
        )
        self.fire_data_available(data_stream_event)


class SequentialDataStream(DataStream):
    """Acquire multiple streams sequentially.

    Each stream can also produce multiple channels.
    """
    def __init__(self, data_streams: typing.Sequence[DataStream]) -> None:
        super().__init__()
        self.__data_streams: typing.List[DataStream] = [data_stream.add_ref() for data_stream in data_streams]
        self.__data_available_event_listener = typing.cast(Event.EventListener, None)
        self.__handle_error_event_listener = typing.cast(Event.EventListener, None)
        self.__stream_args = DataStreamArgs(list())
        self.__current_index = 0
        self.__sequence_count = 0
        self.__sequence_index = 0

    def about_to_delete(self) -> None:
        if self.__data_available_event_listener:
            self.__data_available_event_listener.close()
            self.__data_available_event_listener = typing.cast(typing.Any, None)
        if self.__handle_error_event_listener:
            self.__handle_error_event_listener.close()
            self.__handle_error_event_listener = typing.cast(typing.Any, None)
        for data_stream in self.__data_streams:
            data_stream.remove_ref()
        self.__data_streams = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return tuple(self.__data_streams)

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        channels: typing.List[Channel] = list()
        for index, data_stream in enumerate(self.__data_streams):
            channels.extend([Channel(str(index), *channel.segments) for channel in data_stream.channels])
        return tuple(channels)

    def get_info(self, channel: Channel) -> DataStreamInfo:
        if channel in self.channels:
            index = int(channel.segments[0])
            sub_channel = Channel(*channel.segments[1:])
            if sub_channel in self.__data_streams[index].channels:
                return self.__data_streams[index].get_info(sub_channel)
        assert False, f"No info for channel {channel}"

    @property
    def is_finished(self) -> bool:
        return self.__current_index == len(self.__data_streams)

    @property
    def _progress(self) -> float:
        # return the average of combined streams progress
        return (self.__sequence_index + (self.__current_index + self.__data_streams[self.__current_index].progress) / len(self.__data_streams)) / self.__sequence_count

    def _send_next(self) -> None:
        self.__data_streams[self.__current_index].send_next()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        self.__current_index = 0
        self.__stream_args = DataStreamArgs((1,))
        # self.__stream_args = copy.deepcopy(stream_args)
        self.__data_streams[self.__current_index].prepare_stream(self.__stream_args)
        self.__sequence_count = stream_args.sequence_count
        self.__sequence_index = 0

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert self.__current_index == 0
        assert not self.__data_available_event_listener
        assert not self.__handle_error_event_listener
        self.__data_available_event_listener = self.__data_streams[self.__current_index].data_available_event.listen(weak_partial(_handle_data_available, self, SequentialDataStream.__data_available))
        self.__handle_error_event_listener = self.__data_streams[self.__current_index].handle_error_event.listen(weak_partial(DataStream.handle_error, self))
        self.__data_streams[self.__current_index].start_stream(self.__stream_args)

    def _abort_stream(self) -> None:
        self.__data_streams[self.__current_index].abort_stream()

    def _advance_stream(self) -> None:
        # handle calling finish and start for the contained data stream.
        if self.__data_streams[self.__current_index].is_finished:
            self.__data_streams[self.__current_index].finish_stream()
            self.__data_available_event_listener.close()
            self.__data_available_event_listener = typing.cast(typing.Any, None)
            self.__handle_error_event_listener.close()
            self.__handle_error_event_listener = typing.cast(typing.Any, None)
            self.__current_index += 1
            if self.__current_index == len(self.__data_streams) and self.__sequence_index + 1 < self.__sequence_count:
                self.__current_index = 0
                self.__sequence_index += 1
            if self.__current_index < len(self.__data_streams):
                self.__data_streams[self.__current_index].prepare_stream(self.__stream_args)
                self.__data_available_event_listener = self.__data_streams[self.__current_index].data_available_event.listen(weak_partial(_handle_data_available, self, SequentialDataStream.__data_available))
                self.__handle_error_event_listener = self.__data_streams[self.__current_index].handle_error_event.listen(weak_partial(DataStream.handle_error, self))
                self.__data_streams[self.__current_index].start_stream(self.__stream_args)

        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].advance_stream()

    def _finish_stream(self) -> None:
        if self.__current_index < len(self.__data_streams):
            self.__data_streams[self.__current_index].finish_stream()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        state = DataStreamStateEnum.COMPLETE if data_stream_event.state == DataStreamStateEnum.COMPLETE and self.__current_index + 1 == len(self.__data_streams) else DataStreamStateEnum.PARTIAL
        data_stream_event = DataStreamEventArgs(
            data_stream_event.data_stream,
            Channel(str(self.__current_index), *data_stream_event.channel.segments),
            data_stream_event.data_metadata,
            data_stream_event.source_data,
            data_stream_event.count,
            data_stream_event.source_slice,
            state
        )
        self.fire_data_available(data_stream_event)


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
                ((operator_count, ) + data_metadata.data_shape, dtype),
                data_metadata.intensity_calibration,
                [Calibration.Calibration()] + list(data_metadata.dimensional_calibrations),
                data_metadata.metadata,
                data_metadata.timestamp,
                data_descriptor,
                data_metadata.timezone,
                data_metadata.timezone_offset
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
                data_metadata = DataAndMetadata.DataMetadata(((), dtype), data_metadata.intensity_calibration, [],
                                                             data_metadata.metadata, data_metadata.timestamp,
                                                             DataAndMetadata.DataDescriptor(False, 0, 0),
                                                             data_metadata.timezone, data_metadata.timezone_offset)
            else:
                # not sure if this special case is needed...? it is only different in that it produces
                # slightly different data_metadata.
                # concatenate the numpy arrays to keep the dtype the same. xdata promotes to float64.
                new_data = numpy.concatenate([data_and_metadata.data for data_and_metadata in data_list], axis=-1)  # type: ignore
                data_metadata = data_list[0].data_metadata
                data_metadata = DataAndMetadata.DataMetadata(((new_data.shape), new_data.dtype),
                                                             data_metadata.intensity_calibration,
                                                             [Calibration.Calibration()] + list(data_metadata.dimensional_calibrations[:-1]),
                                                             data_metadata.metadata, data_metadata.timestamp,
                                                             DataAndMetadata.DataDescriptor(False, 0, 1),
                                                             data_metadata.timezone, data_metadata.timezone_offset)
        else:
            # concatenate the numpy arrays to keep the dtype the same. xdata promotes to float64.
            new_data = numpy.concatenate([data_and_metadata.data for data_and_metadata in data_list], axis=-1)  # type: ignore
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


class DataChannel(ReferenceCounting.ReferenceCounted):
    """Acquisition data channel.

    An acquisition data channel receives partial data and must return full data when required.
    """
    def __init__(self) -> None:
        super().__init__()

    def add_ref(self) -> DataChannel:
        super().add_ref()
        return self

    def prepare(self, data_stream: DataStream) -> None:
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

    def add_ref(self) -> DataAndMetadataDataChannel:
        super().add_ref()
        return self

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
        return self.__data[channel]


class FrameCallbacks:
    def _send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None: ...
    def _send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int) -> None: ...


class Framer(ReferenceCounting.ReferenceCounted):
    def __init__(self, data_channel: DataChannel) -> None:
        super().__init__()
        # data and indexes use the _incoming_ data channels as keys.
        self.__data_channel = data_channel.add_ref()
        self.__indexes: typing.Dict[Channel, int] = dict()

    def about_to_delete(self) -> None:
        self.__data_channel.remove_ref()
        self.__data_channel = typing.cast(typing.Any, None)
        super().about_to_delete()

    def add_ref(self) -> Framer:
        super().add_ref()
        return self

    def prepare(self, data_stream: DataStream) -> None:
        self.__indexes = dict()
        self.__data_channel.prepare(data_stream)

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__data_channel.get_data(channel)

    def data_available(self, data_stream_event: DataStreamEventArgs, callbacks: FrameCallbacks) -> None:
        # when data arrives, store it into a data item with the same description/shape.
        # data is assumed to be partial data. this restriction may be removed in a future
        # version. separate indexes are kept for each channel and represent the next destination
        # for the data.

        # useful variables
        channel = data_stream_event.channel
        source_slice = data_stream_event.source_slice
        count = data_stream_event.count

        if count is None:
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
            assert index + source_count <= flat_shape[0]
            self.__data_channel.update_data(channel, data_stream_event.source_data, source_slice, dest_slice, data_metadata)
            # proceed
            index = index + source_count
            self.__indexes[channel] = index
            # if the data chunk is complete, perform processing and send out the new data.
            if data_stream_event.state == DataStreamStateEnum.COMPLETE:
                assert index == flat_shape[0]  # index should be at the end.
                callbacks._send_data(data_stream_event.channel, self.__data_channel.get_data(data_stream_event.channel))
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
            callbacks._send_data_multiple(data_stream_event.channel, data_and_metadata, count)
            self.__indexes[channel] = 0


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
        self.__data_stream = data_stream.add_ref()
        self.__operator = operator or NullDataStreamOperator()
        self.__data_available_event_listener = typing.cast(Event.EventListener, None)
        self.__handle_error_event_listener = typing.cast(Event.EventListener, None)
        self.__framer = Framer(data_channel or DataAndMetadataDataChannel()).add_ref()

    def about_to_delete(self) -> None:
        if self.__data_available_event_listener:
            self.__data_available_event_listener.close()
            self.__data_available_event_listener = typing.cast(typing.Any, None)
        if self.__handle_error_event_listener:
            self.__handle_error_event_listener.close()
            self.__handle_error_event_listener = typing.cast(typing.Any, None)
        self.__data_stream.remove_ref()
        self.__data_stream = typing.cast(typing.Any, None)
        self.__framer.remove_ref()
        self.__framer = typing.cast(typing.Any, None)
        super().about_to_delete()

    def __str__(self) -> str:
        s = super().__str__()
        if self.__operator and not isinstance(self.__operator, NullDataStreamOperator):
            s = s + f" ({self.__operator})"
        return s

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return (self.__data_stream,)

    def add_ref(self) -> FramedDataStream:
        super().add_ref()
        return self

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

    def get_info(self, channel: Channel) -> DataStreamInfo:
        return self.__operator.transform_data_stream_info(channel, self.__data_stream.get_info(channel))

    @property
    def is_finished(self) -> bool:
        return self.__data_stream.is_finished

    def _acquire_finished(self) -> None:
        assert all(self.get_info(c).data_metadata.data_shape == self.get_data(c).data_shape for c in self.channels)

    @property
    def _progress(self) -> float:
        return self.__data_stream.progress

    def _send_next(self) -> None:
        self.__data_stream.send_next()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        self.__operator.reset()
        self.__data_stream.prepare_stream(stream_args, operator=self.__operator)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert not self.__data_available_event_listener
        assert not self.__handle_error_event_listener
        self.__data_available_event_listener = self.__data_stream.data_available_event.listen(weak_partial(_handle_data_available, self, FramedDataStream.__data_available))
        self.__handle_error_event_listener = self.__data_stream.handle_error_event.listen(weak_partial(DataStream.handle_error, self))
        self.__data_stream.start_stream(stream_args)

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _advance_stream(self) -> None:
        self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        self.__data_stream.finish_stream()
        self.__data_available_event_listener.close()
        self.__data_available_event_listener = typing.cast(typing.Any, None)
        self.__handle_error_event_listener.close()
        self.__handle_error_event_listener = typing.cast(typing.Any, None)

    def prepare_data_channel(self) -> None:
        self.__framer.prepare(self)

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__framer.get_data(channel)

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        self.__framer.data_available(data_stream_event, typing.cast(FrameCallbacks, self))

    def _send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
        # callback for Framer
        if not self.__operator.is_applied:
            for new_channel_data in self.__operator.process(ChannelData(channel, data_and_metadata)):
                self.__send_data(new_channel_data.channel, new_channel_data.data_and_metadata, update_in_place=True)
        else:
            self.__send_data(channel, data_and_metadata)

    def _send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int) -> None:
        # callback for Framer
        if not self.__operator.is_applied:
            for new_channel_data in self.__operator.process_multiple(
                    ChannelData(channel, data_and_metadata)):
                self.__send_data_multiple(new_channel_data.channel, new_channel_data.data_and_metadata, count, update_in_place=True)
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
            self.__send_data_multiple(channel, data_and_metadata, count)

    def __send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, update_in_place: bool = False) -> None:
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
        new_data_stream_event = DataStreamEventArgs(self, channel, new_data_metadata, new_data, new_count,
                                                    new_source_slice, DataStreamStateEnum.COMPLETE)
        self.fire_data_available(new_data_stream_event, update_in_place)

    def __send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int, update_in_place: bool = False) -> None:
        assert data_and_metadata.is_sequence
        new_data_descriptor = DataAndMetadata.DataDescriptor(False, data_and_metadata.collection_dimension_count,
                                                             data_and_metadata.datum_dimension_count)
        data_dtype = data_and_metadata.data_dtype
        assert data_dtype is not None
        new_data_metadata = DataAndMetadata.DataMetadata(
            (data_and_metadata.data_shape[1:], data_dtype),
            data_and_metadata.intensity_calibration,
            data_and_metadata.dimensional_calibrations[1:],
            data_and_metadata.metadata,
            data_and_metadata.timestamp,
            new_data_descriptor,
            data_and_metadata.timezone,
            data_and_metadata.timezone_offset)
        new_source_slice = (slice(0, count),) + (slice(None),) * len(data_and_metadata.data_shape[1:])
        data = data_and_metadata.data
        assert data is not None
        new_data_stream_event = DataStreamEventArgs(self, channel, new_data_metadata, data, count, new_source_slice, DataStreamStateEnum.COMPLETE)
        self.fire_data_available(new_data_stream_event, update_in_place)


class MaskLike(typing.Protocol):
    def get_mask_array(self, data_shape: ShapeType) -> _NDArray: ...


AxisType = typing.Union[int, typing.Tuple[int, ...]]


class SumOperator(DataStreamOperator):
    def __init__(self, *, axis: typing.Optional[AxisType] = None) -> None:
        super().__init__()
        self.__axis = axis

    def __str__(self) -> str:
        return "sum"

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
            (tuple(new_shape), data_dtype),
            data_metadata.intensity_calibration,
            new_dimensional_calibrations,
            data_metadata.metadata,
            data_metadata.timestamp,
            data_descriptor,
            data_metadata.timezone,
            data_metadata.timezone_offset
        )
        return DataStreamInfo(data_metadata, data_stream_info.duration)

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        data_and_metadata = channel_data.data_and_metadata
        if self.__axis is not None:
            summed_xdata = xd.sum(data_and_metadata, self.__axis)
            assert summed_xdata
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
            ((), numpy.float32),
            data_metadata.intensity_calibration,
            [],
            data_metadata.metadata,
            data_metadata.timestamp,
            DataAndMetadata.DataDescriptor(False, 0, 0),
            data_metadata.timezone,
            data_metadata.timezone_offset
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
                (tuple(new_shape), data_dtype),
                data_metadata.intensity_calibration,
                new_dimensional_calibrations,
                data_metadata.metadata,
                data_metadata.timestamp,
                data_descriptor,
                data_metadata.timezone,
                data_metadata.timezone_offset
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
        self.__data_stream = data_stream.add_ref()
        self.__data_available_event_listener = typing.cast(Event.EventListener, None)
        self.__handle_error_event_listener = typing.cast(Event.EventListener, None)

    def about_to_delete(self) -> None:
        if self.__data_available_event_listener:
            self.__data_available_event_listener.close()
            self.__data_available_event_listener = typing.cast(typing.Any, None)
        if self.__handle_error_event_listener:
            self.__handle_error_event_listener.close()
            self.__handle_error_event_listener = typing.cast(typing.Any, None)
        self.__data_stream.remove_ref()
        self.__data_stream = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return (self.__data_stream,)

    @property
    def data_stream(self) -> DataStream:
        return self.__data_stream

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return self.__data_stream.channels

    def get_info(self, channel: Channel) -> DataStreamInfo:
        return self.__data_stream.get_info(channel)

    @property
    def is_finished(self) -> bool:
        return self.__data_stream.is_finished

    def _acquire_finished(self) -> None:
        self.__data_stream._acquire_finished()

    @property
    def _progress(self) -> float:
        return self.__data_stream.progress

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _send_next(self) -> None:
        self.__data_stream.send_next()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        self.__data_stream.prepare_stream(stream_args, **kwargs)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert not self.__data_available_event_listener
        assert not self.__handle_error_event_listener
        self.__data_available_event_listener = self.__data_stream.data_available_event.listen(weak_partial(_handle_data_available, self, ContainerDataStream.__data_available))
        self.__handle_error_event_listener = self.__data_stream.handle_error_event.listen(weak_partial(DataStream.handle_error, self))
        self.__data_stream.start_stream(stream_args)

    def _advance_stream(self) -> None:
        self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        self.__data_stream.finish_stream()
        self.__data_available_event_listener.close()
        self.__data_available_event_listener = typing.cast(typing.Any, None)
        self.__handle_error_event_listener.close()
        self.__handle_error_event_listener = typing.cast(typing.Any, None)

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        self.fire_data_available(data_stream_event)


class ActionDataStreamDelegate:
    """A delegate to perform the action.

    Subclasses can override start, perform, and finish.
    """
    def start(self) -> None:
        pass

    def perform(self, c: ShapeType) -> None:
        pass

    def finish(self) -> None:
        pass


class ActionDataStreamFnDelegate(ActionDataStreamDelegate):
    def __init__(self, fn: typing.Callable[[ShapeType], None]) -> None:
        self.__fn = fn

    def perform(self, c: ShapeType) -> None:
        return self.__fn(c)


def make_action_data_stream_delegate(fn: typing.Callable[[typing.Sequence[int]], None]) -> ActionDataStreamDelegate:
    return ActionDataStreamFnDelegate(fn)


class ActionDataStream(ContainerDataStream):
    """Action data stream. Runs action on each complete frame, passing coordinates."""

    def __init__(self, data_stream: DataStream, delegate: ActionDataStreamDelegate) -> None:
        super().__init__(data_stream)
        self.__delegate = delegate
        self.__shape = typing.cast(ShapeType, (0,))
        self.__index = 0
        self.__channel_count = len(self.channels)
        self.__complete_channel_count = 0

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        stream_args.max_count = 1
        super()._prepare_stream(stream_args, **kwargs)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__shape = stream_args.shape
        self.__index = 0
        self.__complete_channel_count = self.__channel_count
        self.__delegate.start()
        self.__check_action()
        stream_args.max_count = 1
        super()._start_stream(stream_args)

    def __check_action(self) -> None:
        if not self.is_finished and not self.is_aborted:
            if self.__complete_channel_count == self.__channel_count:
                c = better_unravel_index(self.__index, self.__shape)
                self.__delegate.perform(c)
                self.__index += 1
                self.__complete_channel_count = 0

    def _advance_stream(self) -> None:
        self.__check_action()
        super()._advance_stream()

    def _finish_stream(self) -> None:
        self.__delegate.finish()
        super()._finish_stream()

    def _fire_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        assert data_stream_event.count is None or data_stream_event.count == 1
        super()._fire_data_available(data_stream_event)
        if data_stream_event.state == DataStreamStateEnum.COMPLETE:
            assert (data_stream_event.count or 1) == 1
            self.__complete_channel_count += 1


class MonitorDataStream(DataStream):
    """Non-controlling data stream. Monitors data coming out of data stream."""

    def __init__(self, data_stream: DataStream, channel_segment: ChannelSegment) -> None:
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__channel_segment = channel_segment
        self.__data_available_event_listener = typing.cast(Event.EventListener, None)
        self.__handle_error_event_listener = typing.cast(Event.EventListener, None)

    def about_to_delete(self) -> None:
        if self.__data_available_event_listener:
            self.__data_available_event_listener.close()
            self.__data_available_event_listener = typing.cast(typing.Any, None)
        if self.__handle_error_event_listener:
            self.__handle_error_event_listener.close()
            self.__handle_error_event_listener = typing.cast(typing.Any, None)
        self.__data_stream.remove_ref()
        self.__data_stream = typing.cast(typing.Any, None)
        super().about_to_delete()

    @property
    def data_streams(self) -> typing.Sequence[DataStream]:
        return (self.__data_stream,)

    @property
    def data_stream(self) -> DataStream:
        return self.__data_stream

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return tuple(c.join_segment(self.__channel_segment) for c in self.__data_stream.channels)

    def get_info(self, channel: Channel) -> DataStreamInfo:
        return self.__data_stream.get_info(channel.parent)

    @property
    def is_finished(self) -> bool:
        return self.__data_stream.is_finished

    def _acquire_finished(self) -> None:
        pass

    @property
    def _progress(self) -> float:
        return self.__data_stream.progress

    def _abort_stream(self) -> None:
        pass

    def _send_next(self) -> None:
        pass

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        pass

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        assert not self.__data_available_event_listener
        assert not self.__handle_error_event_listener
        self.__data_available_event_listener = self.__data_stream.data_available_event.listen(weak_partial(_handle_data_available, self, MonitorDataStream.__data_available))
        self.__handle_error_event_listener = self.__data_stream.handle_error_event.listen(weak_partial(DataStream.handle_error, self))

    def _advance_stream(self) -> None:
        pass

    def _finish_stream(self) -> None:
        self.__data_available_event_listener.close()
        self.__data_available_event_listener = typing.cast(typing.Any, None)
        self.__handle_error_event_listener.close()
        self.__handle_error_event_listener = typing.cast(typing.Any, None)

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        data_stream_event.channel = data_stream_event.channel.join_segment(self.__channel_segment)
        self.fire_data_available(data_stream_event)
        data_stream_event.channel = data_stream_event.channel.parent


class AccumulatedDataStream(ContainerDataStream):
    """Change a data stream producing a sequence into an accumulated non-sequence.
    """

    def __init__(self, data_stream: DataStream) -> None:
        super().__init__(data_stream)
        self.__data_channel = DataAndMetadataDataChannel().add_ref()
        self.__dest_indexes: typing.Dict[Channel, int] = dict()

    def about_to_delete(self) -> None:
        self.__data_channel.remove_ref()
        self.__data_channel = typing.cast(typing.Any, None)
        super().about_to_delete()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs: typing.Any) -> None:
        self.__data_channel.clear_data()
        super()._prepare_stream(stream_args, **kwargs)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__dest_indexes = dict()
        super()._start_stream(stream_args)

    def get_info(self, channel: Channel) -> DataStreamInfo:
        data_stream_info = super().get_info(channel)
        old_data_metadata = data_stream_info.data_metadata
        data_descriptor = copy.deepcopy(old_data_metadata.data_descriptor)
        assert data_descriptor.is_sequence
        data_descriptor.is_sequence = False
        data_dtype = old_data_metadata.data_dtype
        assert data_dtype is not None
        data_metadata = DataAndMetadata.DataMetadata(
            (tuple(old_data_metadata.data_shape[1:]), data_dtype),
            old_data_metadata.intensity_calibration,
            old_data_metadata.dimensional_calibrations[1:],
            old_data_metadata.metadata,
            old_data_metadata.timestamp,
            data_descriptor,
            old_data_metadata.timezone,
            old_data_metadata.timezone_offset
        )
        return DataStreamInfo(data_metadata, data_stream_info.duration)

    def _fire_data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        count = data_stream_event.count
        assert count is None
        old_data_metadata = data_stream_event.data_metadata
        data_descriptor = copy.deepcopy(old_data_metadata.data_descriptor)
        assert data_descriptor.is_sequence
        sequence_slice = data_stream_event.source_slice[0]
        assert sequence_slice.stop - sequence_slice.start == 1
        data_descriptor.is_sequence = False
        data_dtype = old_data_metadata.data_dtype
        assert data_dtype is not None
        data_metadata = DataAndMetadata.DataMetadata(
            (tuple(old_data_metadata.data_shape[1:]), data_dtype),
            old_data_metadata.intensity_calibration,
            old_data_metadata.dimensional_calibrations[1:],
            old_data_metadata.metadata,
            old_data_metadata.timestamp,
            data_descriptor,
            old_data_metadata.timezone,
            old_data_metadata.timezone_offset
        )
        channel = data_stream_event.channel
        dest_count = expand_shape(data_metadata.data_shape)
        sequence_slice_offset = sequence_slice.start * dest_count
        source_start = ravel_slice_start(data_stream_event.source_slice, data_stream_event.source_data.shape) - sequence_slice_offset
        source_stop = ravel_slice_stop(data_stream_event.source_slice, data_stream_event.source_data.shape) - sequence_slice_offset
        dest_slice_offest = self.__dest_indexes.get(channel, 0)
        dest_slice = slice(dest_slice_offest + source_start, dest_slice_offest + source_stop)
        self.__dest_indexes[channel] = dest_slice.stop % dest_count
        new_source_slices = unravel_flat_slice(dest_slice, data_metadata.data_shape)
        assert len(new_source_slices) == 1
        new_source_slice = new_source_slices[0]
        self.__data_channel.accumulate_data(channel, data_stream_event.source_data[sequence_slice.start],
                                            data_stream_event.source_slice[1:], dest_slice, data_metadata)
        data_channel_data = self.__data_channel.get_data(channel).data
        assert data_channel_data is not None
        new_data_stream_event = DataStreamEventArgs(self, channel, data_metadata,
                                                    data_channel_data, None, new_source_slice,
                                                    data_stream_event.state)
        new_data_stream_event.reset_frame = dest_slice.start == 0
        super()._fire_data_available(new_data_stream_event)


def acquire(data_stream: DataStream, *, error_handler: typing.Optional[typing.Callable[[Exception], None]] = None) -> None:
    """Perform an acquire.

    Performs consistency checks on progress and data.

    Progress must be made once per 60s or else an exception is thrown.
    """
    TIMEOUT = 60.0
    data_stream.prepare_stream(DataStreamArgs((1,)))
    data_stream.start_stream(DataStreamArgs((1,)))
    try:
        start = time.time()
        last_progress = 0.0
        last_progress_time = time.time()
        while not data_stream.is_finished and not data_stream.is_aborted:
            # progress checking is for tests and self consistency
            pre_progress = data_stream.progress
            data_stream.send_next()
            post_progress = data_stream.progress
            data_stream.advance_stream()
            next_progress = data_stream.progress
            assert pre_progress <= post_progress <= next_progress, f"{pre_progress=} <= {post_progress=} <= {next_progress=}"
            assert next_progress >= last_progress
            if next_progress > last_progress:
                last_progress = next_progress
                last_progress_time = time.time()
            assert time.time() - last_progress_time < TIMEOUT
            time.sleep(0.05)  # play nice with other threads
        if data_stream.is_finished:
            assert data_stream.progress == 1.0
            data_stream._acquire_finished()
    except Exception as e:
        data_stream.is_error = True
        data_stream.abort_stream()
        from nion.swift.model import Notification
        Notification.notify(Notification.Notification("nion.acquisition.error", "\N{WARNING SIGN} Acquisition", "Acquisition Failed", str(e)))
        if error_handler:
            error_handler(e)
        else:
            import traceback
            traceback.print_exc()
    finally:
        data_stream.finish_stream()


class Acquisition:
    def __init__(self, data_stream: FramedDataStream) -> None:
        self.__data_stream = data_stream.add_ref()
        self.__task: typing.Optional[asyncio.Task[None]] = None
        self.__is_aborted = False
        self.__is_error = False

    def close(self) -> None:
        if self.__data_stream:
            self.__data_stream.remove_ref()
            self.__data_stream = typing.cast(typing.Any, None)

    def prepare_acquire(self) -> None:
        # this is called on the main thread. give data channel a chance to prepare.
        self.__data_stream.prepare_data_channel()

    def acquire(self) -> None:
        try:
            with self.__data_stream.ref():
                acquire(self.__data_stream)
                self.__is_aborted = self.__data_stream.is_aborted
                self.__is_error = self.__data_stream.is_error
        finally:
            self.__data_stream.remove_ref()
            self.__data_stream = typing.cast(typing.Any, None)

    def acquire_async(self, *, event_loop: asyncio.AbstractEventLoop, on_completion: typing.Callable[[], None]) -> None:
        async def grab_async() -> None:
            try:
                self.prepare_acquire()
                await asyncio.get_running_loop().run_in_executor(None, self.acquire)
            finally:
                on_completion()
                self.__task = None

        self.__task = event_loop.create_task(grab_async())

    def abort_acquire(self) -> None:
        if self.__data_stream:
            self.__data_stream.abort_stream()

    def wait_acquire(self, timeout: float = 60.0, *, on_periodic: typing.Callable[[], None]) -> None:
        start = time.time()
        while self.__task and not self.__task.done() and time.time() - start < timeout:
            on_periodic()
            time.sleep(0.05)  # don't take all of the CPU
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
