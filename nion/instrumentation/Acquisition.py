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

Defines

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


class DataStreamEventArgs:
    """Data stream event arguments.

    The `data_stream` property should be passed in from the data stream caller.

    The `channel` property should be set to something unique to allow for collecting multiple
    data streams in one collector.

    The `data_metadata` property describes the data layout, calibrations, metadata,
    time stamps, etc. of each data chunk.

    A data stream can send data chunks partially or n at a time.

    If sent partially, the `count` property should be `None` and the `source_slice` and `dest_slice`
    properties should be filled.

    If the `count` property is not `None`, the data should have an extra dimension whose size
    matches the `count` property.

    The `source_slice` describes the slice on the source data. It can be `None` to use the entire
    data chunk.

    The `dest_slice` describes the slice on the destination data. It can be `None` to use the entire
    data chunk.

    The `state` property indicates if the data chunk is complete or not.
    """

    def __init__(self, data_stream: DataStream, channel: Channel, data_metadata: DataAndMetadata.DataMetadata,
                 source_data: numpy.ndarray, source_slice: SliceType, dest_slice: SliceType,
                 state: typing.Optional[DataStreamStateEnum]):
        self.__print = False

        # the data stream sending this event
        self.data_stream = data_stream

        # the data stream channel. must be unique within a data stream collector.
        self.channel = channel

        # the data description of the data chunks that this data stream produces.
        self.data_metadata = data_metadata

        # the data and source data slice list within this data chunk. the first slice represents the index
        # and must start with 0.
        self.source_data = source_data
        self.source_slice = source_slice

        # the destination slice list of this data chunk. the first slice represents the index and can be
        # repeated for partial data or be a range for multiple data chunks.
        self.dest_slice = dest_slice

        # the state of data after this event, partial or complete. pass None if not producing partial datums.
        self.state = state or DataStreamStateEnum.COMPLETE

    def print(self) -> None:
        if self.__print:
            print(f"received {self.data_stream} / {self.channel}")
            print(f"{self.data_metadata.data_shape} [{self.data_metadata.data_dtype}] {self.data_metadata.data_descriptor}")
            print(f"{self.source_data.shape=} {self.source_slice} -> {self.dest_slice}")
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
        return True

    def send_next(self) -> None:
        pass


class CollectedDataStream(DataStream):
    def __init__(self, data_stream: DataStream, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration]):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        self.__listener = data_stream.data_available_event.listen(self.__data_available)

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__listener = None
        self.__data_stream.remove_ref()
        self.__data_stream = None
        super().about_to_delete()

    def _get_new_data_descriptor(self, data_metadata):
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
        # when data arrives, put it into the sequence/collection and send it out again
        data_stream_event.print()

        data_metadata = data_stream_event.data_metadata

        # get the new data descriptor
        new_data_descriptor = self._get_new_data_descriptor(data_metadata)

        # add the collection count to the shape
        shape = self.__collection_shape + tuple(data_metadata.data_shape)

        # add an empty calibration for the new collection dimensions
        new_dimensional_calibrations = self.__collection_calibrations + tuple(data_metadata.dimensional_calibrations)

        # create a new data metadata object
        new_data_metadata = DataAndMetadata.DataMetadata((shape, data_metadata.data_dtype),
                                                         data_metadata.intensity_calibration,
                                                         new_dimensional_calibrations,
                                                         data_descriptor=new_data_descriptor)

        # send out the new data stream event. this sends out a single datum at a time, so index slice will be None
        # and datum slice will be same as source stream.
        channel = data_stream_event.channel
        source_data = data_stream_event.source_data
        source_slice = data_stream_event.source_slice
        dest_slice = data_stream_event.dest_slice
        count = numpy.product(self.__collection_shape, dtype=numpy.int64)
        state = DataStreamStateEnum.COMPLETE if dest_slice[0].stop == count else DataStreamStateEnum.PARTIAL
        self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, source_data, source_slice, dest_slice, state))


class SequenceDataStream(CollectedDataStream):
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
    def __init__(self, data_stream: DataStream):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__listener = data_stream.data_available_event.listen(self.__data_available)
        self.__data: typing.Dict[Channel, DataAndMetadata.DataAndMetadata] = dict()
        self.__index = 0

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
        data_stream_event.print()
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
        source_slice = data_stream_event.source_slice
        dest_slice = data_stream_event.dest_slice
        data_chunk_rank = len(data_stream_event.source_data.shape) - 1
        assert data_chunk_rank >= 0
        if data_chunk_rank == 0:
            # this is a case where scalar data is being provided.
            flat_shape = (numpy.product(data_and_metadata.data.shape, dtype=numpy.int64),)
            data_and_metadata.data.reshape(flat_shape)[dest_slice] = data_stream_event.source_data[source_slice]
        else:
            # this is a case where data is being stored.
            # TODO: optimize cases where dest data is contiguous.
            ravel_data_shape = data_and_metadata.data_shape[:-data_chunk_rank]
            for i in range(source_slice[0].start, source_slice[0].stop):
                # the index is modulo the ravel_data_shape so that the source data stream
                # can produce increasing frame numbers without worrying about the enclosing collectors
                # and processing.
                index = (dest_slice[0].start + i) % numpy.product(ravel_data_shape, dtype=numpy.int64)
                dest_index = numpy.unravel_index(index, ravel_data_shape)
                data_and_metadata.data[dest_index][dest_slice[1:]] = data_stream_event.source_data[i][source_slice[1:]]
        if data_stream_event.state == DataStreamStateEnum.COMPLETE:
            # data metadata describes the data being sent from this stream: shape, data type, and descriptor
            new_data_metadata, new_data = self._process(self.__data[data_stream_event.channel])
            # data slice describes what slice of the data chunk being sent is valid. here, the entire frame is valid.
            slices = tuple(slice(None) for s in range(len(new_data_metadata.dimensional_shape)))
            source_data_slice = (slice(0, 1),) + slices
            dest_data_slice = (slice(self.__index, self.__index + 1),) + slices
            # send the data with a count of one. this requires padding the data chunk with an index axis
            # unless the data is scalar in which case it is already padded.
            if len(new_data_metadata.dimensional_shape) > 0:
                new_data = new_data[numpy.newaxis, :]
            else:
                assert len(new_data.shape) == 1
            new_data_stream_event = DataStreamEventArgs(self, data_stream_event.channel,
                                                        new_data_metadata,
                                                        new_data, source_data_slice,
                                                        dest_data_slice, DataStreamStateEnum.COMPLETE)
            self.data_available_event.fire(new_data_stream_event)
            self.__index += 1

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
