import contextlib
import numpy
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition


class CameraFrameDataStream(Acquisition.DataStream):
    """Provide a single data stream.
    """
    def __init__(self, count: int, shape: Acquisition.ShapeType, channel: Acquisition.Channel):
        super().__init__()
        self.count = count
        self.shape = tuple(shape)
        self.channel = channel
        self.index = 0
        self.data = numpy.random.randn(self.count, *self.shape)

    @property
    def is_finished(self) -> bool:
        return self.index == self.count

    def send_next(self) -> None:
        assert self.index < self.count
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.shape))
        data_metadata = DataAndMetadata.DataMetadata((self.shape, float), data_descriptor=data_descriptor)
        # slice index describes what slice of the data being sent is valid
        slice_index = tuple(slice(0, s) for s in self.shape)
        # data slice describes the source slice of the data being sent. in this case it is not passed and uses
        # the default of the entire data.
        # state describes the state of the stream - can be PARTIAL, meaning only part of the current frame is valid,
        # or can be COMPLETE meaning the current frame is complete with the inclusion of the current slice.
        # always uses the default (COMPLETE) for this frame-by-frame stream.
        data_stream_event = Acquisition.DataStreamEventArgs(self, self.channel, data_metadata, slice_index, self.data[self.index])
        self.data_available_event.fire(data_stream_event)
        self.index += 1


class ScanFrameDataStream(Acquisition.DataStream):
    """Provide a single data stream.
    """
    def __init__(self, count: int, shape: Acquisition.ShapeType, channel: Acquisition.Channel, partial_height: int):
        super().__init__()
        assert len(shape) == 2
        self.count = count
        self.shape = tuple(shape)
        self.channel = channel
        self.partial_height = partial_height
        self.index = 0
        self.partial = 0
        self.data = numpy.random.randn(self.count, *self.shape)

    @property
    def is_finished(self) -> bool:
        return self.index == self.count

    def send_next(self) -> None:
        assert self.index < self.count
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.shape))
        data_metadata = DataAndMetadata.DataMetadata((self.shape, float), data_descriptor=data_descriptor)
        # slice index describes the destination slice of the data being sent
        new_partial = min(self.partial + self.partial_height, self.shape[0])
        slice_index = (slice(self.partial, new_partial), slice(0, self.shape[1]))
        # data slice describes the source slice of the data being sent. in this case they are the same. they could
        # be different if only a part of the data was being provided in each event.
        data_index = slice_index
        # state describes the state of the stream - can be PARTIAL, meaning only part of the current frame is valid,
        # or can be COMPLETE meaning the current frame is complete with the inclusion of the current slice.
        state = Acquisition.DataStreamStateEnum.PARTIAL if new_partial < self.shape[0] else Acquisition.DataStreamStateEnum.COMPLETE
        data_stream_event = Acquisition.DataStreamEventArgs(self, self.channel, data_metadata, slice_index, self.data[self.index], data_index, state)
        self.data_available_event.fire(data_stream_event)
        if state == Acquisition.DataStreamStateEnum.PARTIAL:
            self.partial = new_partial
        else:
            self.partial = 0
            self.index += 1


class TestAcquisitionClass(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_camera_sequence_acquisition(self):
        sequence_len = 4
        data_stream = CameraFrameDataStream(sequence_len, (2, 2), 0)
        sequencer = Acquisition.Sequencer(data_stream, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            self.assertTrue(numpy.array_equal(data_stream.data, maker.data[0].data))

    def test_camera_collection_acquisition(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (4, 3)
        data_stream = CameraFrameDataStream(numpy.product(collection_shape), (2, 2), 0)
        collector = Acquisition.Collector(data_stream, collection_shape,
                                          [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            expected_shape = collection_shape + maker.data[0].data.shape[-len(collection_shape):]
            self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.data[0].data))

    def test_scan_sequence_acquisition(self):
        sequence_len = 4
        data_stream = ScanFrameDataStream(sequence_len, (4, 4), 0, 2)
        sequencer = Acquisition.Sequencer(data_stream, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            self.assertTrue(numpy.array_equal(data_stream.data, maker.data[0].data))

    def test_scan_collection_acquisition(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan.
        # the scan data is hard coded to produce a scan.
        collection_shape = (5, 3)
        data_stream = ScanFrameDataStream(numpy.product(collection_shape), (4, 4), 0, 2)
        collector = Acquisition.Collector(data_stream, collection_shape,
                                          [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            expected_shape = collection_shape + maker.data[0].data.shape[-len(collection_shape):]
            self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.data[0].data))
