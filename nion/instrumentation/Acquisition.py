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
from nion.utils import Event
from nion.utils import ReferenceCounting

ShapeType = typing.Sequence[int]
SliceType = typing.Sequence[slice]
Channel = typing.Union[str, int]
PositionType = typing.Tuple[int, ...]


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


class Collector(DataStream):
    def __init__(self, data_stream: DataStream, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration]):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        self.__listener = data_stream.data_available_event.listen(self.__data_available)

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__data_stream.remove_ref()
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


class Sequencer(Collector):
    def __init__(self, data_stream: DataStream, count: int, calibration: typing.Optional[Calibration.Calibration] = None):
        super().__init__(data_stream, (count,), (calibration,))

    def _get_new_data_descriptor(self, data_metadata):
        # scalar data is not supported
        assert data_metadata.datum_dimension_count > 0

        # new data descriptor is a collection
        collection_dimension_count = data_metadata.collection_dimension_count
        datum_dimension_count = data_metadata.datum_dimension_count
        return DataAndMetadata.DataDescriptor(True, collection_dimension_count, datum_dimension_count)


class DataStreamToDataAndMetadata:
    def __init__(self, data_stream: DataStream):
        self.__data_stream = data_stream.add_ref()
        self.__listener = data_stream.data_available_event.listen(self.__data_available)
        self.data: typing.Dict[Channel, DataAndMetadata.DataAndMetadata] = dict()

    def close(self) -> None:
        self.__listener.close()
        self.__data_stream.remove_ref()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        data_and_metadata = self.data.get(data_stream_event.channel, None)
        if not data_and_metadata:
            data_metadata = data_stream_event.data_metadata
            data = numpy.zeros(data_metadata.data_shape, data_metadata.data_dtype)
            data_descriptor = data_metadata.data_descriptor
            data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                      data_metadata.intensity_calibration,
                                                                      data_metadata.dimensional_calibrations,
                                                                      data_descriptor=data_descriptor)
            self.data[data_stream_event.channel] = data_and_metadata
        assert data_and_metadata
        source_slice = data_stream_event.source_slice
        dest_slice = data_stream_event.dest_slice
        # print(f"{data_and_metadata.data_shape=}")
        # print(f"{data_stream_event.count=}")
        # print(f"{data_stream_event.data_metadata.data_shape=}")
        # print(f"{data_stream_event.source_data.shape=}")
        # print(f"{data_stream_event.source_slice=}")
        # print(f"{data_stream_event.dest_slice=}")
        data_chunk_rank = len(data_stream_event.source_data.shape) - 1
        ravel_data_shape = data_and_metadata.data_shape[:-data_chunk_rank]
        if not ravel_data_shape:
            flat_shape = (numpy.product(data_and_metadata.data.shape, dtype=numpy.int64),)
            data_and_metadata.data.reshape(flat_shape)[dest_slice] = data_stream_event.source_data[source_slice]
        else:
            # TODO: optimize cases where dest data is contiguous.
            for i in range(source_slice[0].start, source_slice[0].stop):
                dest_index = numpy.unravel_index(dest_slice[0].start + i, ravel_data_shape)
                data_and_metadata.data[dest_index][dest_slice[1:]] = data_stream_event.source_data[i][source_slice[1:]]
