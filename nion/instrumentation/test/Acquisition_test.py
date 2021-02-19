import contextlib
import numpy
import typing
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
        # data slice describes what slice of the data chunk being sent is valid. here, the entire frame is valid.
        source_data_slice = (slice(0, 1),) + tuple(slice(None) for s in self.shape)
        dest_data_slice = (slice(self.index, self.index + 1),) + tuple(slice(None) for s in self.shape)
        # send the data with a count of one. this requires padding the data chunk with an index axis.
        data_stream_event = Acquisition.DataStreamEventArgs(self, self.channel, data_metadata,
                                                            self.data[self.index:self.index + 1], source_data_slice,
                                                            dest_data_slice, Acquisition.DataStreamStateEnum.COMPLETE)
        self.data_available_event.fire(data_stream_event)
        self.index += 1


class ScanDataStream(Acquisition.DataStream):
    """Provide a data stream for one scan with the given channel.
    """
    def __init__(self, count: int, length: int, channels: typing.Sequence[Acquisition.Channel], partial: int):
        super().__init__()
        self.count = count
        self.length = length
        self.channels = channels
        self.partial = partial
        self.index = 0
        self.partial_index = 0
        self.data = {channel: numpy.random.randn(count, length) for channel in channels}

    @property
    def is_finished(self) -> bool:
        return self.index == self.count

    def send_next(self) -> None:
        assert self.index < self.count
        assert self.partial_index < self.length
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 0)
        data_metadata = DataAndMetadata.DataMetadata(((), float), data_descriptor=data_descriptor)
        # update the index to be used in the data slice
        new_index = min(self.partial_index + self.partial, self.length)
        source_data_slice = (slice(self.partial_index, new_index),)
        dest_data_slice = (slice(self.partial_index + self.index * self.length, new_index + self.index * self.length),)
        # send the data with no count. this is required when using partial.
        state = Acquisition.DataStreamStateEnum.PARTIAL if new_index < self.length else Acquisition.DataStreamStateEnum.COMPLETE
        for channel in self.channels:
            data_stream_event = Acquisition.DataStreamEventArgs(self, channel, data_metadata, self.data[channel][self.index],
                                                                source_data_slice, dest_data_slice, state)
            self.data_available_event.fire(data_stream_event)
        if state == Acquisition.DataStreamStateEnum.COMPLETE:
            self.partial_index = 0
            self.index += 1
        else:
            self.partial_index = new_index


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
        # update the index to be used in the data slice
        new_partial = min(self.partial + self.partial_height, self.shape[0])
        source_data_slice = (slice(0, 1),) + (slice(self.partial, new_partial), slice(None))
        dest_data_slice = (slice(self.index, self.index + 1),) + (slice(self.partial, new_partial), slice(None))
        # send the data with no count. this is required when using partial.
        state = Acquisition.DataStreamStateEnum.PARTIAL if new_partial < self.shape[0] else Acquisition.DataStreamStateEnum.COMPLETE
        data_stream_event = Acquisition.DataStreamEventArgs(self, self.channel, data_metadata,
                                                            self.data[self.index:self.index + 1], source_data_slice,
                                                            dest_data_slice, state)
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
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            self.assertTrue(numpy.array_equal(data_stream.data, maker.data[0].data))

    def test_camera_collection_acquisition(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (4, 3)
        data_stream = CameraFrameDataStream(numpy.product(collection_shape), (2, 2), 0)
        collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
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
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
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
        collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                    [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            expected_shape = collection_shape + maker.data[0].data.shape[-len(collection_shape):]
            self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.data[0].data))

    def test_scan_as_collection(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        scan_shape = (8, 8)
        data_stream = ScanDataStream(1, numpy.product(scan_shape), [0], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            expected_shape = scan_shape
            self.assertTrue(numpy.array_equal(data_stream.data[0].reshape(expected_shape), maker.data[0].data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.data[0].data_descriptor)

    def test_scan_as_collection_as_sequence(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        # that will be collected to a sequence.
        sequence_len = 4
        scan_shape = (8, 8)
        data_stream = ScanDataStream(sequence_len, numpy.product(scan_shape), [0], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            expected_shape = (sequence_len,) + scan_shape
            self.assertTrue(numpy.array_equal(data_stream.data[0].reshape(expected_shape), maker.data[0].data))
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.data[0].data_descriptor)

    def test_scan_as_collection_two_channels(self):
        # scan will produce two data streams of pixels.
        # the collection must make it into two images.
        scan_shape = (8, 8)
        data_stream = ScanDataStream(1, numpy.product(scan_shape), [0, 1], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with contextlib.closing(maker):
            while not data_stream.is_finished:
                data_stream.send_next()
            expected_shape = scan_shape
            self.assertTrue(numpy.array_equal(data_stream.data[0].reshape(expected_shape), maker.data[0].data))
            self.assertTrue(numpy.array_equal(data_stream.data[1].reshape(expected_shape), maker.data[1].data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.data[0].data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.data[1].data_descriptor)

    def test_scan_as_collection_two_channels_and_camera(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        scan_data_stream = ScanDataStream(sequence_len, numpy.product(scan_shape), [0, 1], scan_shape[1])
        camera_data_stream = CameraFrameDataStream(sequence_len * numpy.product(scan_shape), (2, 2), 2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with contextlib.closing(maker):
            while not scan_data_stream.is_finished or not camera_data_stream.is_finished:
                if not scan_data_stream.is_finished:
                    scan_data_stream.send_next()
                if not camera_data_stream.is_finished:
                    camera_data_stream.send_next()
            expected_scan_shape = (sequence_len,) + scan_shape
            expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
            self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.data[0].data))
            self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.data[1].data))
            self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.data[2].data))
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.data[0].data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.data[1].data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.data[2].data_descriptor)
