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
    def __init__(self, data_stream: DataStream, channel: Channel, data_metadata: DataAndMetadata.DataMetadata,
                 slice_index: SliceType, data: numpy.ndarray, data_slice: typing.Optional[SliceType] = None,
                 state: typing.Optional[DataStreamStateEnum] = None):
        self.data_stream = data_stream
        self.channel = channel
        self.data_metadata = data_metadata
        self.slice_index = slice_index
        self.data = data
        self.data_slice = data_slice or Ellipsis
        self.state = state or DataStreamStateEnum.COMPLETE

    def copy_data(self, dest_data: numpy.ndarray) -> None:
        dest_data[self.slice_index] = self.data[self.data_slice]


class DataStream(ReferenceCounting.ReferenceCounted):
    """Provide a single data stream.

    A data stream is a series of data available events which update
    the state and data of a stream.
    """
    def __init__(self):
        super().__init__()
        self.data_available_event = Event.Event()


class Sequencer(DataStream):
    def __init__(self, data_stream: DataStream, count: int,
                 calibration: typing.Optional[Calibration.Calibration] = None):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__count = count
        self.__calibration = calibration or Calibration.Calibration()
        self.__listener = data_stream.data_available_event.listen(self.__data_available)
        self.__indexes: typing.Dict[Channel, int] = dict()

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__data_stream.remove_ref()
        super().about_to_delete()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        # when data arrives, put it into the sequence and send it out again
        # as a new data stream event where the data descriptor is now a sequence
        # and the slice represents the destination the data in the sequence.

        data_metadata = data_stream_event.data_metadata

        # sequences of sequences are not currently supported
        assert not data_metadata.is_sequence

        # new data descriptor is a sequence
        new_data_descriptor = DataAndMetadata.DataDescriptor(True, data_metadata.collection_dimension_count,
                                                             data_metadata.datum_dimension_count)

        # add the sequence count to the shape
        shape = (self.__count,) + tuple(data_metadata.data_shape)

        # get the current index for the channel
        index = self.__indexes.get(data_stream_event.channel, 0)

        # add the index to the slice index as the sequence dimension
        slice_index = (slice(index, index + 1),) + tuple(data_stream_event.slice_index)

        # add an empty calibration for the new sequence dimension
        new_dimensional_calibrations = (self.__calibration,) + tuple(data_metadata.dimensional_calibrations)

        # create a new data metadata object
        new_data_metadata = DataAndMetadata.DataMetadata((shape, data_metadata.data_dtype),
                                                         data_metadata.intensity_calibration,
                                                         new_dimensional_calibrations,
                                                         data_descriptor=new_data_descriptor)

        # increment the index is frame is complete
        new_index = index + 1 if data_stream_event.state == DataStreamStateEnum.COMPLETE else index

        # send out the new data stream event
        state = DataStreamStateEnum.PARTIAL if new_index < self.__count else DataStreamStateEnum.COMPLETE
        channel = data_stream_event.channel
        self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, slice_index, data_stream_event.data, data_stream_event.data_slice, state))

        # update the index for this channel
        self.__indexes[data_stream_event.channel] = new_index


class Collector(DataStream):
    def __init__(self, data_stream: DataStream, shape: DataAndMetadata.ShapeType, calibrations: typing.Sequence[Calibration.Calibration]):
        super().__init__()
        self.__data_stream = data_stream.add_ref()
        self.__collection_shape = tuple(shape)
        self.__collection_calibrations = tuple(calibrations)
        self.__listener = data_stream.data_available_event.listen(self.__data_available)
        self.__indexes: typing.Dict[Channel, PositionType] = dict()

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__data_stream.remove_ref()
        super().about_to_delete()

    def __data_available(self, data_stream_event: DataStreamEventArgs) -> None:
        # when data arrives, put it into the collection and send it out again
        # as a new data stream event where the data descriptor is now a collection
        # and the slice represents the destination the data in the collection.

        data_metadata = data_stream_event.data_metadata

        # collections of sequences and collections of collections are not currently supported
        assert not data_metadata.is_sequence
        assert not data_metadata.is_collection

        # new data descriptor is a collection
        collection_rank = len(self.__collection_shape)
        new_data_descriptor = DataAndMetadata.DataDescriptor(False, collection_rank, data_metadata.datum_dimension_count)

        # add the collection count to the shape
        shape = self.__collection_shape + tuple(data_metadata.data_shape)

        # get the current index for the channel
        index = self.__indexes.get(data_stream_event.channel, [0] * collection_rank)

        # add the index to the slice index as the collection dimension
        slice_index = tuple(slice(i, i + 1) for i in index) + tuple(data_stream_event.slice_index)

        # add an empty calibration for the new collection dimensions
        new_dimensional_calibrations = self.__collection_calibrations + tuple(data_metadata.dimensional_calibrations)

        # create a new data metadata object
        new_data_metadata = DataAndMetadata.DataMetadata((shape, data_metadata.data_dtype),
                                                         data_metadata.intensity_calibration,
                                                         new_dimensional_calibrations,
                                                         data_descriptor=new_data_descriptor)

        # increment the index is frame is complete
        unravel_shape = (self.__collection_shape[0] + 1, ) + self.__collection_shape[1:]
        flattened_index = numpy.ravel_multi_index(index, unravel_shape)
        new_flattened_index = flattened_index + 1 if data_stream_event.state == DataStreamStateEnum.COMPLETE else flattened_index
        new_index = numpy.unravel_index(new_flattened_index, unravel_shape)

        # send out the new data stream event
        state = DataStreamStateEnum.COMPLETE if new_flattened_index == numpy.product(self.__collection_shape) else DataStreamStateEnum.PARTIAL
        channel = data_stream_event.channel
        self.data_available_event.fire(DataStreamEventArgs(self, channel, new_data_metadata, slice_index, data_stream_event.data, data_stream_event.data_slice, state))

        # update the index for this channel
        self.__indexes[data_stream_event.channel] = new_index


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
            data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                      data_metadata.intensity_calibration,
                                                                      data_metadata.dimensional_calibrations,
                                                                      data_descriptor=data_metadata.data_descriptor)
            self.data[data_stream_event.channel] = data_and_metadata
        assert data_and_metadata
        data_stream_event.copy_data(data_and_metadata.data)
