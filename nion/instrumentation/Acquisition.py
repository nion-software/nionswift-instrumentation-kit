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
import numpy
import typing

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.utils import Event
from nion.utils import ReferenceCounting

ShapeType = typing.Sequence[int]
SliceType = typing.Sequence[slice]
Channel = typing.Union[str, int]


class DataStreamStateEnum(enum.Enum):
    PARTIAL = 1
    COMPLETE = 2


def index_slice(n: int) -> slice:
    return slice(n, n + 1)


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
    time stamps, etc. of each data chunk.

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

    def print(self) -> None:
        if self.__print:
            print(f"received {self.data_stream} / {self.channel}")
            print(f"{self.data_metadata.data_shape} [{self.data_metadata.data_dtype}] {self.data_metadata.data_descriptor}")
            print(f"{self.count}: {self.source_data.shape=} {self.source_slice}")
            print(f"{self.state}")
            print("")


class DataStream(ReferenceCounting.ReferenceCounted):
    """Provide a stream of data chunks.

    A data chunk is data that can be collected into a sequence or a 1d or 2d collection.

    1d or 2d collections can themselves be data chunks and collected into a sequence.

    Currently, sequences themselves cannot be collected into sequences. Nor can collections be collected.

    Future versions may include the ability to collect multiple data streams into lists, tables, or structures.

    The stream communicates by a series of data available events which update data and state.

    The stream can be controlled with a controller.

    The stream may trigger other events in addition to data available events.
    """
    def __init__(self):
        super().__init__()
        self.data_available_event = Event.Event()

    @property
    def is_finished(self) -> bool:
        """Return true if stream is finished."""
        return True

    def send_next(self) -> None:
        """Used for testing. Send next data."""
        pass


class CollectedDataStream(DataStream):
    """Collect a data stream of chunks into a collection of those chunks.

    The data is formed into a collection by resending it reshaped into the
    collection with additional slice information. No data is copied during
    reshaping.
    """

    def __init__(self, data_stream: DataStream, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration]):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        self.__indexes: typing.Dict[Channel, int] = dict()
        self.__listener = data_stream.data_available_event.listen(self.__data_available)

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__listener = None
        self.__data_stream.remove_ref()
        self.__data_stream = None
        super().about_to_delete()

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

        data_stream_event.print()

        # useful variables
        data_metadata = data_stream_event.data_metadata
        count = data_stream_event.count
        channel = data_stream_event.channel
        collection_count = numpy.product(self.__collection_shape, dtype=numpy.int64)

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
                                                         data_descriptor=new_data_descriptor)

        # send out the new data stream event.
        index = self.__indexes.get(channel, 0)
        if count is not None:
            # incoming data is frames
            # count must either be a multiple of last dimensions of collection shape or one
            assert count == 1 or count % int(numpy.product(self.__collection_shape[1:], dtype=numpy.int64)) == 0
            assert count == data_stream_event.source_slice[0].stop - data_stream_event.source_slice[0].start
            assert index + count <= collection_count
            assert count > 0
            if count == 1:
                # if the count is one, add new dimensions of length one for the collection shape and form
                # the new slice with indexes of 0 for each collection dimension and full slices for the remaining
                # dimensions.
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                new_source_data = old_source_data.reshape((1,) * len(self.__collection_shape) + old_source_data.shape[1:])
                new_source_slice = (index_slice(0), ) * len(self.__collection_shape) + (slice(None),) * (len(old_source_data.shape) - 1)
                new_state = DataStreamStateEnum.COMPLETE if index + count == collection_count else DataStreamStateEnum.PARTIAL
                self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                self.__indexes[channel] = (index + count) % collection_count
            else:
                # if the count is greater than one, provide the "rows" of the collection. row is just first dimension.
                # start by calculating the start/stop rows. reshape the data into collection shape and add a slice
                # for the rows and full slices for the remaining dimensions.
                slice_start = better_unravel_index(index, self.__collection_shape)[0]
                slice_stop = better_unravel_index(index + count, self.__collection_shape)[0]
                assert slice_stop <= self.__collection_shape[0]
                old_source_data = data_stream_event.source_data[data_stream_event.source_slice]
                new_source_data = data_stream_event.source_data.reshape((count,) + self.__collection_shape[1:] + old_source_data.shape[1:])
                new_source_slice = (slice(slice_start, slice_stop),) + (slice(None),) * (len(new_shape) - 1)
                new_state = DataStreamStateEnum.COMPLETE if index + count == collection_count else DataStreamStateEnum.PARTIAL
                self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
                self.__indexes[channel] = (index + count) % collection_count
        else:
            # incoming data is partial
            # form the new slice with indexes of 0 for each collection dimension and the incoming slice for the
            # remaining dimensions. add dimensions for collection dimensions to the new source data.
            new_source_slice = (index_slice(0), ) * len(self.__collection_shape) + tuple(data_stream_event.source_slice)
            new_source_data = data_stream_event.source_data.reshape((1,) * len(self.__collection_shape) + data_stream_event.source_data.shape)
            # new state is complete if data chunk is complete and it will put us at the end of our collection.
            if index + 1 == collection_count and data_stream_event.state == DataStreamStateEnum.COMPLETE:
                new_state = DataStreamStateEnum.COMPLETE
            else:
                new_state = DataStreamStateEnum.PARTIAL
            self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, new_source_data, None, new_source_slice, new_state))
            if data_stream_event.state == DataStreamStateEnum.COMPLETE:
                self.__indexes[channel] = (index + 1) % collection_count


class SequenceDataStream(CollectedDataStream):
    """Collect a data stream into a sequence of datums.

    This is a subclass of CollectedDataStream.
    """
    def __init__(self, data_stream: DataStream, count: int, calibration: typing.Optional[Calibration.Calibration] = None):
        super().__init__(data_stream, (count,), (calibration,))

    def _get_new_data_descriptor(self, data_metadata):
        # scalar data is not supported
        assert data_metadata.datum_dimension_count > 0

        # new data descriptor is a collection
        collection_dimension_count = data_metadata.collection_dimension_count
        datum_dimension_count = data_metadata.datum_dimension_count
        return DataAndMetadata.DataDescriptor(True, collection_dimension_count, datum_dimension_count)


class CombinedDataStream(DataStream):
    """Combine multiple streams into a single stream produceing multiple channels.

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

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        data_stream_event.print()
        self.data_available_event.fire(data_stream_event)


class FramedDataStream(DataStream):
    """Change a data stream producing continuous data into one producing frame by frame data.

    This is useful when finalizing data or when processing needs the full frame before proceeding.
    """
    def __init__(self, data_stream: DataStream):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__listener = data_stream.data_available_event.listen(self.__data_available)
        self.__data: typing.Dict[Channel, DataAndMetadata.DataAndMetadata] = dict()
        self.__indexes: typing.Dict[Channel, int] = dict()

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__listener = None
        self.__data_stream.remove_ref()
        self.__data_stream = None

    @property
    def is_finished(self) -> bool:
        return self.__data_stream.is_finished

    def send_next(self) -> None:
        self.__data_stream.send_next()

    def get_data(self, channel: Channel) -> DataAndMetadata.DataAndMetadata:
        return self.__data[channel]

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        # when data arrives, store it into a data item with the same description/shape.
        # data is assumed to be partial data. this restriction may be removed in a future
        # version. separate indexes are kept for each channel and represent the next destination
        # for the data.

        data_stream_event.print()

        # useful variables
        channel = data_stream_event.channel
        source_slice = data_stream_event.source_slice
        count = data_stream_event.count

        # check to see if data has already been allocated. allocated it if not.
        data_and_metadata = self.__data.get(data_stream_event.channel, None)
        if not data_and_metadata:
            data_metadata = data_stream_event.data_metadata
            data = numpy.zeros(data_metadata.data_shape, data_metadata.data_dtype)
            data_descriptor = data_metadata.data_descriptor
            data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                      data_metadata.intensity_calibration,
                                                                      data_metadata.dimensional_calibrations,
                                                                      data_descriptor=data_descriptor)
            self.__data[data_stream_event.channel] = data_and_metadata
        assert data_and_metadata

        # assume partial data.
        assert count is None

        # determine the start/stop indexes. then copy the source data into the destination using
        # flattening to allow for use of simple indexing. then increase the index.
        index = self.__indexes.get(channel, 0)
        source_start = ravel_slice_start(source_slice, data_stream_event.source_data.shape)
        source_stop = ravel_slice_stop(source_slice, data_stream_event.source_data.shape)
        source_count = source_stop - source_start
        flat_shape = (numpy.product(data_and_metadata.data.shape, dtype=numpy.int64),)
        dest_slice = slice(index, index + source_count)
        assert index + source_count <= flat_shape[0]
        data_and_metadata.data.reshape(-1)[dest_slice] = data_stream_event.source_data[source_slice].reshape(-1)
        index = index + source_count
        self.__indexes[channel] = index

        # if the data chunk is complete, perform processing and send out the new data.
        if data_stream_event.state == DataStreamStateEnum.COMPLETE:
            assert index == flat_shape[0]  # index should be at the end.
            # processing
            new_data_metadata, new_data = self._process(self.__data[data_stream_event.channel])
            new_count: typing.Optional[int] = None
            new_source_slice: typing.Tuple[slice, ...]
            # special case for scalar
            if new_data_metadata.data_descriptor.expected_dimension_count == 0:
                assert len(new_data.shape) == 1
                new_count = new_data.shape[0]
            # form the new slice
            new_source_slice = (slice(0, new_data.shape[0]),) + (slice(None),) * (len(new_data.shape) - 1)
            # send the new data chunk
            new_data_stream_event = DataStreamEventArgs(self, data_stream_event.channel, new_data_metadata, new_data,
                                                        new_count, new_source_slice, DataStreamStateEnum.COMPLETE)
            self.data_available_event.fire(new_data_stream_event)
            self.__indexes[channel] = 0

    def _process(self, data_and_metadata: DataAndMetadata.DataAndMetadata) -> typing.Tuple[DataAndMetadata.DataMetadata, numpy.ndarray]:
        return data_and_metadata.data_metadata, data_and_metadata.data


class SummedDataStream(FramedDataStream):
    def __init__(self, data_stream: DataStream, axis: typing.Optional[int] = None):
        super().__init__(data_stream)
        self.__axis = axis

    def _process(self, data_and_metadata: DataAndMetadata.DataAndMetadata) -> typing.Tuple[DataAndMetadata.DataMetadata, numpy.ndarray]:
        if self.__axis is not None:
            summed = xd.sum(data_and_metadata, self.__axis)
            return summed.data_metadata, summed.data
        else:
            data_metadata = DataAndMetadata.DataMetadata(((), float), data_descriptor=DataAndMetadata.DataDescriptor(False, 0, 0))
            data_summed = numpy.array([data_and_metadata.data.sum()])
            return data_metadata, data_summed


class DataStreamToDataAndMetadata(FramedDataStream):
    def __init__(self, data_stream: DataStream):
        super().__init__(data_stream)
