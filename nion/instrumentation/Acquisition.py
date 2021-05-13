"""Acquisition pipeline.

Devices produce a set of data streams specific to the device.

Camera devices produce a data stream of 2D or 1D data.

Scan devices produce a data stream of 2D or 1D data.

Operators take data streams as inputs and produce different data streams.

Operators may also control hardware.

Actions control hardware.

The goal is to set up an acquisition graph that controls hardware and produces output data streams.

Multiple devices can be configured; but how they are synchronized must be described.

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

import enum
import typing
import warnings

import numpy
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import Shape
from nion.data import xdata_1_0 as xd
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import ReferenceCounting

ShapeType = typing.Sequence[int]
SliceType = typing.Sequence[slice]
SliceListType = typing.Sequence[SliceType]
Channel = typing.Union[str, int]

ChannelPassThrough: Channel = -1


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
    return numpy.unravel_index(index, (shape[0] + 1,) + tuple(shape[1:]))


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
                 source_data: numpy.ndarray, count: typing.Optional[int], source_slice: SliceType,
                 state: DataStreamStateEnum):
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

    def print(self, receiver: DataStream) -> None:
        if self.__print:
            print(f"{receiver} received {self.data_stream} / {self.channel}")
            print(f"  {self.data_metadata.data_shape} [{self.data_metadata.data_dtype}] {self.data_metadata.data_descriptor}")
            print(f"  {self.count}: {self.source_data.shape} {self.source_slice}")
            print(f"  {self.state}")
            print("")

    def __str__(self) -> str:
        s = str()
        s += f"-- received {self.data_stream} / {self.channel}\n"
        s += f"  {self.data_metadata.data_shape} [{self.data_metadata.data_dtype}] {self.data_metadata.data_descriptor}\n"
        s += f"  {self.count}: {self.source_data.shape} {self.source_slice}\n"
        s += f"  {self.state}\n"
        return s


class DataStreamArgs:
    def __init__(self, slice: SliceType, shape: ShapeType):
        for s, l in zip(slice, shape):
            assert s.start < s.stop
            assert 0 <= s.start <= l
            assert 0 <= s.stop <= l
        self.slice = slice
        self.shape = shape

    def __str__(self) -> str:
        return f"{self!r} {self.slice} {self.shape}"

    @property
    def slice_shape(self) -> ShapeType:
        return get_slice_shape(self.slice, self.shape)

    @property
    def sequence_count(self) -> int:
        return numpy.product(self.slice_shape, dtype=numpy.int64)

    @property
    def slice_rect(self) -> Geometry.IntRect:
        return get_slice_rect(self.slice, self.shape)


class DataStream(ReferenceCounting.ReferenceCounted):
    """Provide a stream of data chunks.

    A data chunk is data that can be collected into a sequence or a 1d or 2d collection.

    1d or 2d collections can themselves be data chunks and collected into a sequence.

    Currently, sequences themselves cannot be collected into sequences. Nor can collections be collected.

    Future versions may include the ability to collect multiple data streams into lists, tables, or structures.

    The stream generates a series of data available events which can be used to update data and state in listeners.

    The stream is controlled by calls to start_stream and advance_stream and the current state can be determined with
    is_finished.

    The stream includes the concept of a sequence of acquisition. The length of the sequence gets updated in calls
    to start_stream.

    Subclasses may override channels, is_finished, _send_next (for testing), _sequence_next, _start_stream, and
    _advance_stream.
    """

    def __init__(self, sequence_count: int = 1):
        super().__init__()
        self.data_available_event = Event.Event()
        # sequence counts are used for acquiring a sequence of frames controlled by the upstream
        self.__sequence_count = sequence_count
        self.__sequence_counts: typing.Dict[Channel, int] = dict()
        self.__sequence_indexes: typing.Dict[Channel, int] = dict()
        self.is_aborted = False

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        """Return the channels for this data stream."""
        return tuple()

    @property
    def input_channels(self) -> typing.Tuple[Channel, ...]:
        """Return the input channels for this data stream."""
        return self.channels

    @property
    def is_finished(self) -> bool:
        """Return true if stream is finished.

        The stream is finished if all channels have sent the number of items in their sequence.
        """
        return all(self.__sequence_indexes.get(channel, 0) == self.__sequence_counts.get(channel, self.__sequence_count) for channel in self.input_channels)

    @property
    def progress(self) -> typing.Tuple[int, int]:
        p = self._progress
        assert p[0] <= p[1]
        return p

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        return 0, 0

    def abort_stream(self) -> None:
        self._abort_stream()
        self.is_aborted = True

    def _abort_stream(self) -> None:
        pass

    def send_next(self) -> None:
        """Used for testing. Send next data."""
        if not self.is_finished and not self.is_aborted:
            for channel in self.input_channels:
                assert self.__sequence_indexes.get(channel, 0) <= self.__sequence_counts.get(channel, self.__sequence_count)
            self._send_next()

    def _send_next(self) -> None:
        """Used for testing. Send next data. Subclasses can override as required."""
        pass

    def _sequence_next(self, channel: Channel, n: int = 1) -> None:
        """Move the sequence for channel forward by n. Subclasses can override."""
        assert self.__sequence_indexes.get(channel, 0) + n <= self.__sequence_counts.get(channel, self.__sequence_count)
        self.__sequence_indexes[channel] = self.__sequence_indexes.get(channel, 0) + n

    def prepare_stream(self, stream_args: DataStreamArgs, **kwargs) -> None:
        """Prepare stream."""
        self._prepare_stream(stream_args, **kwargs)

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs) -> None:
        """Prepare a sequence of acquisitions.

        Subclasses can override to pass along to contained data streams.
        """
        pass

    def start_stream(self, stream_args: DataStreamArgs) -> None:
        """Restart a sequence of acquisitions."""
        for channel in self.input_channels:
            assert self.__sequence_indexes.get(channel, 0) % self.__sequence_counts.get(channel, self.__sequence_count) == 0
            self.__sequence_counts[channel] = stream_args.sequence_count
            self.__sequence_indexes[channel] = 0
        self._start_stream(stream_args)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        """Restart a sequence of acquisitions.

        Subclasses can override to pass along to contained data streams.
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


class CollectedDataStream(DataStream):
    """Collect a data stream of chunks into a collection of those chunks.

    The data is formed into a collection by resending it reshaped into the
    collection with additional slice information. No data is copied during
    reshaping.

    Optionally pass in a list of sub-slices to break collection into sections.
    """

    def __init__(self, data_stream: DataStream, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration], sub_slices: typing.Optional[SliceListType] = None):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__listener = data_stream.data_available_event.listen(self.__data_available)
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        self.__collection_sub_slices = [tuple(sub_slice) for sub_slice in sub_slices] if sub_slices else [tuple(slice(0, length) for length in self.__collection_shape)]
        self.__collection_sub_slice_index = 0
        # indexes track the destination of the next data within the collection.
        self.__indexes: typing.Dict[Channel, int] = dict()
        # sub-slice indexes track the destination of the next data within the current slice.
        self.__sub_slice_indexes: typing.Dict[Channel, int] = dict()
        # needs starts tracks whether the downstream data stream needs a start call.
        self.__needs_starts: typing.Dict[Channel, bool] = dict()
        self.__data_stream_started = False

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__listener = None
        if self.__data_stream_started:
            warnings.warn("Stream deleted but not finished.", category=RuntimeWarning)
        self.__data_stream.remove_ref()
        self.__data_stream = None
        super().about_to_delete()

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return self.__data_stream.channels

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        p = self.__data_stream.progress
        count = int(p[1] * numpy.product(self.__collection_shape, dtype=numpy.int64))
        index = sum(p[1] * self.__indexes.get(k, 0) + p[0] for k in self.channels) // len(self.channels)
        return index, count

    def _send_next(self) -> None:
        assert self.__data_stream_started
        return self.__data_stream.send_next()

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__collection_sub_slice_index = 0
        self._start_next_sub_stream()

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _start_next_sub_stream(self) -> None:
        if self.__data_stream_started:
            self.__data_stream.finish_stream()
            self.__data_stream_started = False
        self.__collection_sub_slice = self.__collection_sub_slices[self.__collection_sub_slice_index]
        self.__sub_slice_indexes.clear()
        self.__needs_starts.clear()
        self.__data_stream.prepare_stream(DataStreamArgs(self.__collection_sub_slice, self.__collection_shape))
        self.__data_stream.start_stream(DataStreamArgs(self.__collection_sub_slice, self.__collection_shape))
        self.__data_stream_started = True
        self.__collection_sub_slice_index += 1

    def _advance_stream(self) -> None:
        # this will be repeatedly called during tests, where it serves to restart the stream.
        if all(self.__needs_starts.get(channel, False) for channel in self.input_channels):
            if self.__data_stream.is_finished and self.__collection_sub_slice_index < len(self.__collection_sub_slices):
                self._start_next_sub_stream()
            elif not self.is_finished:
                self.__collection_sub_slice_index = 0
                self._start_next_sub_stream()
            self.__data_stream.advance_stream()
        else:
            self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        assert self.__data_stream_started
        self.__data_stream.finish_stream()
        self.__data_stream_started = False

    def _get_new_data_descriptor(self, data_metadata):
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

        data_stream_event.print(self)

        # useful variables
        data_metadata = data_stream_event.data_metadata
        count = data_stream_event.count
        channel = data_stream_event.channel
        collection_count = int(numpy.product(self.__collection_shape, dtype=numpy.int64))

        # get the new data descriptor
        new_data_descriptor = self._get_new_data_descriptor(data_metadata)

        # add the collection count to the downstream data shape to produce the new collection shape.
        new_shape = self.__collection_shape + tuple(data_metadata.data_shape)

        # add designated calibrations for the new collection dimensions.
        new_dimensional_calibrations = self.__collection_calibrations + tuple(data_metadata.dimensional_calibrations)

        # create a new data metadata object
        new_data_metadata = DataAndMetadata.DataMetadata((new_shape, data_metadata.data_dtype),
                                                         data_metadata.intensity_calibration,
                                                         new_dimensional_calibrations,
                                                         data_metadata.metadata,
                                                         data_metadata.timestamp,
                                                         new_data_descriptor,
                                                         data_metadata.timezone,
                                                         data_metadata.timezone_offset)

        # send out the new data stream event.
        index = self.__indexes.get(channel, 0)
        sub_slice_index = self.__sub_slice_indexes.get(channel, 0)
        collection_rank = len(self.__collection_shape)
        collection_sub_slice_shape = tuple(get_slice_shape(self.__collection_sub_slice, self.__collection_shape))
        collection_sub_slice_length = int(numpy.product(collection_sub_slice_shape, dtype=numpy.int64))
        collection_sub_slice_row_length = int(numpy.product(collection_sub_slice_shape[1:], dtype=numpy.int64))
        if count is not None:
            # incoming data is frames
            # count must either be a multiple of last dimensions of collection shape or one
            assert count == data_stream_event.source_slice[0].stop - data_stream_event.source_slice[0].start
            assert index + count <= collection_count
            assert count > 0
            next_index = index + count
            next_sub_slice_index = sub_slice_index + count
            if count == 1:
                # a single data chunk has been provided.
                # if the count is one, add new dimensions of length one for the collection shape and form
                # the new slice with indexes of 0 for each collection dimension and full slices for the remaining
                # dimensions.
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                new_source_data = old_source_data.reshape((1,) * collection_rank + old_source_data.shape[1:])
                new_source_slice = (index_slice(0), ) * collection_rank + (slice(None),) * (len(old_source_data.shape) - 1)
                new_state = DataStreamStateEnum.COMPLETE if index + count == collection_count else DataStreamStateEnum.PARTIAL
                self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                self.__indexes[channel] = next_index % collection_count
                self.__sub_slice_indexes[channel] = next_sub_slice_index
                self.__needs_starts[channel] = next_sub_slice_index == collection_sub_slice_length
            else:
                # multiple data chunks have been provided.
                # if the count is greater than one, provide the "rows" of the collection. row is just first dimension.
                # start by calculating the start/stop rows. reshape the data into collection shape and add a slice
                # for the rows and full slices for the remaining dimensions.
                collection_slice_offset = better_unravel_index(ravel_slice_start(self.__collection_sub_slice, self.__collection_shape), self.__collection_shape)[0]
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                source_index = 0
                if index % collection_sub_slice_row_length != 0:
                    # finish the current row
                    assert len(self.__collection_shape) == 2
                    slice_column = index % collection_sub_slice_row_length
                    slice_width = min(collection_sub_slice_row_length - slice_column, count)
                    new_source_slice = (slice(0, 1), slice(0, slice_width)) + (slice(None),) * (len(new_shape) - 2)
                    next_source_index = source_index + slice_width
                    new_source_data = old_source_data[source_index:next_source_index].reshape((1, slice_width) + old_source_data.shape[1:])
                    new_state = DataStreamStateEnum.COMPLETE if index + slice_width == collection_count else DataStreamStateEnum.PARTIAL
                    self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                    source_index = next_source_index
                    index += slice_width
                    count -= slice_width
                    assert count is not None  # for type checker bug
                else:
                    new_state = DataStreamStateEnum.COMPLETE  # satisfy type checker
                if count // collection_sub_slice_row_length > 0:
                    # send as many complete rows as possible
                    slice_offset = index // collection_sub_slice_row_length - collection_slice_offset
                    slice_start = better_unravel_index(index, self.__collection_shape)[0] - slice_offset - collection_slice_offset
                    slice_stop = better_unravel_index(index + count, self.__collection_shape)[0] - slice_offset - collection_slice_offset
                    assert 0 <= slice_start <= self.__collection_shape[0]
                    assert 0 <= slice_stop <= self.__collection_shape[0]
                    source_slice_shape = get_slice_shape(data_stream_event.source_slice, old_source_data.shape)
                    source_slice_length = source_slice_shape[0]
                    row_count = source_slice_length // collection_sub_slice_row_length
                    next_source_index = source_index + row_count * collection_sub_slice_row_length
                    new_source_data = old_source_data[source_index:next_source_index].reshape((row_count,) + collection_sub_slice_shape[1:] + old_source_data.shape[1:])
                    new_source_slice = (slice(slice_start, slice_stop),) + (slice(None),) * (len(new_shape) - 1)
                    new_state = DataStreamStateEnum.COMPLETE if index + row_count * collection_sub_slice_row_length == collection_count else DataStreamStateEnum.PARTIAL
                    self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                    source_index = next_source_index
                    index += row_count * collection_sub_slice_row_length
                    count -= row_count * collection_sub_slice_row_length
                    assert count is not None  # for type checker bug
                if count > 0:
                    # any remaining count means a partial row
                    assert len(self.__collection_shape) == 2
                    assert count < collection_sub_slice_row_length
                    new_source_slice = (slice(0, 1), slice(0, count)) + (slice(None),) * (len(new_shape) - 2)
                    next_source_index = source_index + count
                    new_source_data = old_source_data[source_index:next_source_index].reshape((1, count) + old_source_data.shape[1:])
                    new_state = DataStreamStateEnum.PARTIAL  # always partial, otherwise would have been sent in previous section
                    self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                    # source_index = next_source_index  # no need for this
                    index += count
                    count -= count
                    assert count is not None  # for type checker bug
                assert count == 0  # everything has been accounted for
                self.__indexes[channel] = next_index % collection_count
                self.__sub_slice_indexes[channel] = next_sub_slice_index
                self.__needs_starts[channel] = next_sub_slice_index == collection_sub_slice_length
            if new_state == DataStreamStateEnum.COMPLETE:
                self._sequence_next(channel)
        else:
            # incoming data is partial
            # form the new slice with indexes of 0 for each collection dimension and the incoming slice for the
            # remaining dimensions. add dimensions for collection dimensions to the new source data.
            new_source_slice = (index_slice(0), ) * collection_rank + tuple(data_stream_event.source_slice)
            new_source_data = data_stream_event.source_data.reshape((1,) * collection_rank + data_stream_event.source_data.shape)
            # new state is complete if data chunk is complete and it will put us at the end of our collection.
            new_state = DataStreamStateEnum.PARTIAL
            next_index = index
            next_sub_slice_index = sub_slice_index
            if data_stream_event.state == DataStreamStateEnum.COMPLETE:
                next_index += 1
                next_sub_slice_index += 1
                if next_index == collection_count:
                    new_state = DataStreamStateEnum.COMPLETE
            self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
            self.__indexes[channel] = next_index % collection_count
            self.__sub_slice_indexes[channel] = next_sub_slice_index
            self.__needs_starts[channel] = next_sub_slice_index == collection_sub_slice_length
            if new_state == DataStreamStateEnum.COMPLETE:
                self._sequence_next(channel)


class SequenceDataStream(CollectedDataStream):
    """Collect a data stream into a sequence of datums.

    This is a subclass of CollectedDataStream.
    """
    def __init__(self, data_stream: DataStream, count: int, calibration: typing.Optional[Calibration.Calibration] = None):
        super().__init__(data_stream, (count,), (calibration,))

    def _get_new_data_descriptor(self, data_metadata):
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
    def __init__(self, data_streams: typing.Sequence[DataStream]):
        super().__init__()
        self.__data_streams = [data_stream.add_ref() for data_stream in data_streams]
        self.__listeners = [data_stream.data_available_event.listen(self.__data_available) for data_stream in data_streams]

    def about_to_delete(self) -> None:
        for listener in self.__listeners:
            listener.close()
        self.__listeners = typing.cast(typing.List, None)
        for data_stream in self.__data_streams:
            data_stream.remove_ref()
        self.__data_streams = typing.cast(typing.List, None)
        super().about_to_delete()

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        channels: typing.List[Channel] = list()
        for data_stream in self.__data_streams:
            channels.extend(data_stream.channels)
        return tuple(channels)

    @property
    def is_finished(self) -> bool:
        return all(data_stream.is_finished for data_stream in self.__data_streams)

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        ps = (data_stream.progress for data_stream in self.__data_streams)
        count = sum(p[1] for p in ps)
        index = sum(p[0] for p in ps)
        return index, count

    def _send_next(self) -> None:
        for data_stream in self.__data_streams:
            data_stream.send_next()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs) -> None:
        for data_stream in self.__data_streams:
            data_stream.prepare_stream(stream_args)

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

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        data_stream_event.print(self)
        self.data_available_event.fire(data_stream_event)


class ChannelData:
    def __init__(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata):
        self.channel = channel
        self.data_and_metadata = data_and_metadata


class DataStreamOperator:
    def __init__(self):
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

    def process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        return self._process(channel_data)

    def process_multiple(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        return self._process_multiple(channel_data)

    def _process_multiple(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        channel = channel_data.channel
        data_and_metadata = channel_data.data_and_metadata
        assert data_and_metadata.is_sequence
        channel_data_list_list = [self.process(ChannelData(channel, data_and_metadata[i])) for i in range(data_and_metadata.data_shape[0])]
        # ([1, A1], [2, B1], [3, C1]), ([1, A2], [2, B2], [3, C2]), ([1, A3], [2, B3], [3, C3])
        # ([1, A1/A2/A3], [2, B1/B2/B3], [3, C1/C2/C3])
        new_channel_data_list = list()
        for index, new_channel_data in enumerate(channel_data_list_list[0]):
            new_channel = new_channel_data.channel
            new_data_and_metadata = xd.sequence_join([cdl[index].data_and_metadata for cdl in channel_data_list_list])
            new_data_and_metadata._set_metadata(data_and_metadata.metadata)
            new_data_and_metadata._set_timestamp(data_and_metadata.timestamp)
            new_data_and_metadata.timezone = data_and_metadata.timezone
            new_data_and_metadata.timezone_offset = data_and_metadata.timezone_offset
            new_channel_data_list.append(ChannelData(new_channel, new_data_and_metadata))
        return new_channel_data_list

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        raise NotImplementedError()


class NullDataStreamOperator(DataStreamOperator):
    def __init__(self):
        super().__init__()

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        return [channel_data]


class CompositeDataStreamOperator(DataStreamOperator):
    def __init__(self, operator_map: typing.Dict[Channel, DataStreamOperator]):
        super().__init__()
        self.__operator_map = operator_map

    def get_channels(self, input_channels: typing.Sequence[Channel]) -> typing.Sequence[Channel]:
        return list(self.__operator_map.keys())

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        channel_data_list = list()
        for channel, operator in self.__operator_map.items():
            for new_channel_data in operator.process(channel_data):
                channel_data_list.append(ChannelData(channel, new_channel_data.data_and_metadata))
        return channel_data_list


class StackedDataStreamOperator(DataStreamOperator):
    def __init__(self, operators: typing.Sequence[DataStreamOperator]):
        super().__init__()
        self.__operators = list(operators)

    @property
    def operators(self) -> typing.Sequence[DataStreamOperator]:
        return self.__operators

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        data_list = list()
        for operator in self.__operators:
            for new_channel_data in operator.process(channel_data):
                data_list.append(new_channel_data.data_and_metadata[..., numpy.newaxis])
        new_data = numpy.concatenate(data_list, axis=-1)
        new_data_and_metadata = DataAndMetadata.new_data_and_metadata(new_data,
                                                                      data_list[0].intensity_calibration,
                                                                      data_list[0].dimensional_calibrations,
                                                                      data_list[0].metadata,
                                                                      data_list[0].timestamp,
                                                                      data_list[0].data_descriptor,
                                                                      data_list[0].timezone,
                                                                      data_list[0].timezone_offset)
        return [ChannelData(channel_data.channel, new_data_and_metadata)]


class FramedDataStream(DataStream):
    """Change a data stream producing continuous data into one producing frame by frame data.

    This is useful when finalizing data or when processing needs the full frame before proceeding.

    Pass an operator to process the resulting frame on any channel before sending it out.

    Pass an operator map to process the resulting frame using multiple operators and send them out on different
    channels.
    """

    def __init__(self, data_stream: DataStream, operator: typing.Optional[DataStreamOperator] = None):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__operator = operator or NullDataStreamOperator()
        self.__listener = data_stream.data_available_event.listen(self.__data_available)
        # data and indexes use the _incoming_ data channels as keys.
        self.__data: typing.Dict[Channel, DataAndMetadata.DataAndMetadata] = dict()
        self.__indexes: typing.Dict[Channel, int] = dict()

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__listener = None
        self.__data_stream.remove_ref()
        self.__data_stream = None

    @property
    def channels(self) -> typing.Tuple[Channel, ...]:
        return tuple(self.__operator.get_channels(self.input_channels))

    @property
    def input_channels(self) -> typing.Tuple[Channel, ...]:
        return self.__data_stream.channels

    @property
    def is_finished(self) -> bool:
        return self.__data_stream.is_finished

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        return self.__data_stream.progress

    def _send_next(self) -> None:
        self.__data_stream.send_next()

    def _prepare_stream(self, stream_args: DataStreamArgs, **kwargs) -> None:
        self.__operator.reset()
        self.__data_stream.prepare_stream(stream_args, operator=self.__operator)

    def _start_stream(self, stream_args: DataStreamArgs) -> None:
        self.__data_stream.start_stream(stream_args)

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _advance_stream(self) -> None:
        self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        self.__data_stream.finish_stream()

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__data[channel]

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        # when data arrives, store it into a data item with the same description/shape.
        # data is assumed to be partial data. this restriction may be removed in a future
        # version. separate indexes are kept for each channel and represent the next destination
        # for the data.

        data_stream_event.print(self)

        # useful variables
        channel = data_stream_event.channel
        source_slice = data_stream_event.source_slice
        count = data_stream_event.count

        if count is None:
            # check to see if data has already been allocated. allocated it if not.
            data_and_metadata = self.__data.get(data_stream_event.channel, None)
            if not data_and_metadata:
                data_metadata = data_stream_event.data_metadata
                data = numpy.zeros(data_metadata.data_shape, data_metadata.data_dtype)
                data_descriptor = data_metadata.data_descriptor
                data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                          data_metadata.intensity_calibration,
                                                                          data_metadata.dimensional_calibrations,
                                                                          data_metadata.metadata,
                                                                          data_metadata.timestamp,
                                                                          data_descriptor,
                                                                          data_metadata.timezone,
                                                                          data_metadata.timezone_offset)
                self.__data[data_stream_event.channel] = data_and_metadata
            assert data_and_metadata
            # determine the start/stop indexes. then copy the source data into the destination using
            # flattening to allow for use of simple indexing. then increase the index.
            source_start = ravel_slice_start(source_slice, data_stream_event.source_data.shape)
            source_stop = ravel_slice_stop(source_slice, data_stream_event.source_data.shape)
            source_count = source_stop - source_start
            flat_shape = (numpy.product(data_and_metadata.data.shape, dtype=numpy.int64),)
            index = self.__indexes.get(channel, 0)
            dest_slice = slice(index, index + source_count)
            assert index + source_count <= flat_shape[0]
            # copy data
            data_and_metadata.data.reshape(-1)[dest_slice] = data_stream_event.source_data[source_slice].reshape(-1)
            # recopy metadata. this isn't perfect; but it's the chosen way for now. if changed, ensure tests pass.
            # the effect of this is that the last chunk of data defines the final metadata. this is useful if the
            # metadata contains in-progress information.
            data_and_metadata._set_metadata(data_stream_event.data_metadata.metadata)
            # proceed
            index = index + source_count
            self.__indexes[channel] = index
            # if the data chunk is complete, perform processing and send out the new data.
            if data_stream_event.state == DataStreamStateEnum.COMPLETE:
                assert index == flat_shape[0]  # index should be at the end.
                # processing
                if not self.__operator.is_applied:
                    for new_channel_data in self.__operator.process(ChannelData(data_stream_event.channel, self.__data[data_stream_event.channel])):
                        self.__send_data(new_channel_data.channel, new_channel_data.data_and_metadata)
                else:
                    self.__send_data(data_stream_event.channel, self.__data[data_stream_event.channel])
                self.__indexes[channel] = 0
                self._sequence_next(channel)
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
            if not self.__operator.is_applied:
                for new_channel_data in self.__operator.process_multiple(ChannelData(data_stream_event.channel, data_and_metadata)):
                    self.__send_data_multiple(new_channel_data.channel, new_channel_data.data_and_metadata, count)
            else:
                self.__send_data_multiple(data_stream_event.channel, data_and_metadata, count)
            self.__indexes[channel] = 0
            self._sequence_next(channel, count)

    def __send_data(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
        new_data_metadata, new_data = data_and_metadata.data_metadata, data_and_metadata.data
        new_count: typing.Optional[int] = None
        new_source_slice: typing.Tuple[slice, ...]
        # special case for scalar
        if new_data_metadata.data_descriptor.expected_dimension_count == 0:
            new_data = numpy.array([new_data])
            assert len(new_data.shape) == 1
            new_count = new_data.shape[0]
        # form the new slice
        new_source_slice = (slice(0, new_data.shape[0]),) + (slice(None),) * (len(new_data.shape) - 1)
        # send the new data chunk
        new_data_stream_event = DataStreamEventArgs(self, channel, new_data_metadata, new_data, new_count,
                                                    new_source_slice, DataStreamStateEnum.COMPLETE)
        self.data_available_event.fire(new_data_stream_event)

    def __send_data_multiple(self, channel: Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int) -> None:
        assert data_and_metadata.is_sequence
        new_data_descriptor = DataAndMetadata.DataDescriptor(False, data_and_metadata.collection_dimension_count,
                                                             data_and_metadata.datum_dimension_count)
        new_data_metadata = DataAndMetadata.DataMetadata(
            (data_and_metadata.data_shape[1:], data_and_metadata.data_dtype),
            data_and_metadata.intensity_calibration,
            data_and_metadata.dimensional_calibrations[1:],
            data_and_metadata.metadata,
            data_and_metadata.timestamp,
            new_data_descriptor,
            data_and_metadata.timezone,
            data_and_metadata.timezone_offset)
        new_source_slice = (slice(0, count),) + (slice(None),) * len(data_and_metadata.data_shape[1:])
        new_data_stream_event = DataStreamEventArgs(self, channel, new_data_metadata, data_and_metadata.data, count,
                                                    new_source_slice, DataStreamStateEnum.COMPLETE)
        self.data_available_event.fire(new_data_stream_event)


AxisType = typing.Union[int, typing.Tuple[int]]


class SumOperator(DataStreamOperator):
    def __init__(self, *, axis: typing.Optional[AxisType] = None):
        super().__init__()
        self.__axis = axis

    @property
    def axis(self) -> typing.Optional[AxisType]:
        return self.__axis

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        data_and_metadata = channel_data.data_and_metadata
        if self.__axis is not None:
            summed_xdata = data_and_metadata
            summed_xdata = xd.sum(summed_xdata, self.__axis)
            return [ChannelData(channel_data.channel, summed_xdata)]
        else:
            summed_data = numpy.array(data_and_metadata.data.sum())
            summed_xdata = DataAndMetadata.new_data_and_metadata(summed_data,
                                                                 intensity_calibration=data_and_metadata.intensity_calibration,
                                                                 data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 0),
                                                                 metadata=data_and_metadata.metadata,
                                                                 timestamp=data_and_metadata.timestamp,
                                                                 timezone=data_and_metadata.timezone,
                                                                 timezone_offset=data_and_metadata.timezone_offset)
            return [ChannelData(channel_data.channel, summed_xdata)]


class MaskedSumOperator(DataStreamOperator):
    def __init__(self, mask: Shape.Mask2DShape):
        super().__init__()
        self.__mask = mask

    @property
    def mask(self) -> Shape.Mask2DShape:
        return self.__mask

    def _process(self, channel_data: ChannelData) -> typing.Sequence[ChannelData]:
        data_and_metadata = channel_data.data_and_metadata
        reference_frame = data_and_metadata.reference_frame_2d
        summed_xdata = DataAndMetadata.new_data_and_metadata(numpy.array((data_and_metadata.data * self.__mask.get_mask_data_2d(reference_frame)).sum()),
                                                             intensity_calibration=data_and_metadata.intensity_calibration,
                                                             data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 0),
                                                             metadata=data_and_metadata.metadata,
                                                             timestamp=data_and_metadata.timestamp,
                                                             timezone=data_and_metadata.timezone,
                                                             timezone_offset=data_and_metadata.timezone_offset)
        return [ChannelData(channel_data.channel, summed_xdata)]


class DataStreamToDataAndMetadata(FramedDataStream):
    def __init__(self, data_stream: DataStream):
        super().__init__(data_stream)

    def active_context(self) -> typing.ContextManager:
        class ContextManager:
            def __init__(self, item: DataStreamToDataAndMetadata):
                self.__item = item

            def __enter__(self):
                self.__item.prepare_stream(DataStreamArgs((slice(0, 1),), (1,)))
                self.__item.start_stream(DataStreamArgs((slice(0, 1),), (1,)))
                return self

            def __exit__(self, type, value, traceback):
                self.__item.finish_stream()

        return ContextManager(self)

    def acquire(self) -> None:
        with self.active_context():
            while not self.is_finished and not self.is_aborted:
                self.send_next()
                self.advance_stream()


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
