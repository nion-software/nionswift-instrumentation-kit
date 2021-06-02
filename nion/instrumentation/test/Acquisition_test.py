import numpy
import typing
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
# from nion.data import Shape
from nion.instrumentation import Acquisition


class ScanDataStream(Acquisition.DataStream):
    """Provide a data stream for one scan with the given channel.

    frame_count is the number of frames to generate.

    scan_shape is the shape of each frame.

    channels are the list of channels to generate.

    partial_length is the size of each chunk of data (number of samples) to send at once.
    """
    def __init__(self, frame_count: int, scan_shape: Acquisition.ShapeType, channels: typing.Sequence[Acquisition.Channel], partial_length: int):
        super().__init__(frame_count * numpy.product(scan_shape, dtype=numpy.int64))
        # frame counts are used for allocating and returning test data
        self.__frame_count = frame_count
        self.__frame_index = 0
        # scan length is total samples in scan shape
        self.__scan_shape = scan_shape
        self.__scan_length = int(numpy.product(scan_shape, dtype=numpy.int64))
        # channels
        self.__channels = tuple(channels)
        # partial length is the size of each chunk sent. partial index is the next sample to be sent.
        self.__partial_length = partial_length
        self.__partial_index = 0
        self.data = {channel: numpy.random.randn(self.__frame_count, self.__scan_length) for channel in channels}
        self.prepare_count = 0

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return self.__channels

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        return self.__partial_index, self.__scan_length

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs) -> None:
        self.prepare_count += 1
        self.__frame_index = 0

    def _send_next(self) -> None:
        assert self.__frame_index < self.__frame_count
        assert self.__partial_index < self.__scan_length
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 0)
        data_metadata = DataAndMetadata.DataMetadata(((), float), data_descriptor=data_descriptor)
        # update the index to be used in the data slice
        start_index = self.__partial_index
        stop_index = min(start_index + self.__partial_length, self.__scan_length)
        new_count = stop_index - start_index
        # source data slice is relative to data start/stop
        source_data_slice = (slice(start_index, stop_index),)
        state = Acquisition.DataStreamStateEnum.PARTIAL if stop_index < self.__scan_length else Acquisition.DataStreamStateEnum.COMPLETE
        for channel in self.channels:
            data_stream_event = Acquisition.DataStreamEventArgs(self, channel, data_metadata,
                                                                self.data[channel][self.__frame_index],
                                                                new_count, source_data_slice, state)
            self.data_available_event.fire(data_stream_event)
            self._sequence_next(channel, new_count)
        # update indexes
        if state == Acquisition.DataStreamStateEnum.COMPLETE:
            self.__partial_index = 0
            self.__frame_index = self.__frame_index + 1
        else:
            self.__partial_index = stop_index


class SingleFrameDataStream(Acquisition.DataStream):
    """Provide a single data stream frame by frame.

    frame_count is the number of frames to generate.

    frame_shape is the shape of each frame.

    channel is the channel on which to send the data.

    partial_height is the size of each chunk of data (number of samples) to send at once.
    """
    def __init__(self, frame_count: int, frame_shape: Acquisition.ShapeType, channel: Acquisition.Channel, partial_height: typing.Optional[int] = None):
        super().__init__(frame_count)
        assert len(frame_shape) == 2
        # frame counts are used for allocating and returning test data
        self.__frame_count = frame_count
        self.__frame_index = 0
        # frame shape and channel
        self.__frame_shape = tuple(frame_shape)
        self.__channel = channel
        # partial height is the size of each chunk sent. partial index is the next sample to be sent.
        self.__partial_height = partial_height or frame_shape[0]
        self.__partial_index = 0
        self.data = numpy.random.randn(self.__frame_count, *self.__frame_shape)

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return (self.__channel,)

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        return self.__partial_index, self.__frame_shape[0]

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs) -> None:
        self.__frame_index = 0

    def _send_next(self) -> None:
        assert self.__frame_index < self.__frame_count
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape))
        data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape, float), data_descriptor=data_descriptor)
        # update the index to be used in the data slice
        new_partial = min(self.__partial_index + self.__partial_height, self.__frame_shape[0])
        source_data_slice = (slice(self.__partial_index, new_partial), slice(None))
        # send the data with no count. this is required when using partial.
        state = Acquisition.DataStreamStateEnum.PARTIAL if new_partial < self.__frame_shape[0] else Acquisition.DataStreamStateEnum.COMPLETE
        data_stream_event = Acquisition.DataStreamEventArgs(self, self.__channel, data_metadata,
                                                            self.data[self.__frame_index], None,
                                                            source_data_slice, state)
        self.data_available_event.fire(data_stream_event)
        if state == Acquisition.DataStreamStateEnum.PARTIAL:
            self.__partial_index = new_partial
        else:
            self.__partial_index = 0
            self.__frame_index += 1
            self._sequence_next(self.__channel)


class MultiFrameDataStream(Acquisition.DataStream):
    """Provide a single data stream frame by frame, n at a time.

    frame_count is the number of frames to generate.

    frame_shape is the shape of each frame.

    channel is the channel on which to send the data.

    n is the number of frames to send at once.
    """
    def __init__(self, frame_count: int, frame_shape: Acquisition.ShapeType, channel: Acquisition.Channel, count: typing.Optional[int] = None, do_processing: bool = False):
        super().__init__(frame_count)
        assert len(frame_shape) == 2
        # frame counts are used for allocating and returning test data
        self.__frame_count = frame_count
        self.__frame_index = 0
        # frame shape and channel
        self.__frame_shape = tuple(frame_shape)
        self.__channel = channel
        # count is the number of chunks sent at once
        self.__count = count or 1
        self.__do_processing = do_processing
        self.data = numpy.random.randn(self.__frame_count, *self.__frame_shape)

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return (self.__channel,)

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        return 0, self.__frame_shape[0]

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs) -> None:
        if self.__do_processing:
            operator = typing.cast(Acquisition.DataStreamOperator, kwargs.get("operator", Acquisition.NullDataStreamOperator()))
            operator.apply()

    def _send_next(self) -> None:
        assert self.__frame_index < self.__frame_count
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        if self.__do_processing:
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape) - 1)
            data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape[1:], float), data_descriptor=data_descriptor)
        else:
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape))
            data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape, float), data_descriptor=data_descriptor)
        # update the index to be used in the data slice
        count = min(self.__count, self.__frame_count - self.__frame_index)
        source_data_slice: typing.Tuple[slice, ...] = (slice(0, count), slice(None), slice(None))
        if self.__do_processing:
            source_data_slice = source_data_slice[:-1]
        # send the data with no count. this is required when using partial.
        state = Acquisition.DataStreamStateEnum.COMPLETE
        source_data = self.data[self.__frame_index:self.__frame_index + count]
        if self.__do_processing:
            source_data = source_data.sum(axis=1)
        data_stream_event = Acquisition.DataStreamEventArgs(self, self.__channel, data_metadata, source_data, count,
                                                            source_data_slice, state)
        self.data_available_event.fire(data_stream_event)
        self.__frame_index += count
        self._sequence_next(self.__channel, count)


class ChangeParameterDataStream(Acquisition.DataStream):
    """
    This is an example for a DataStream that you would plug in between a SequenceDataStream and the underlying
    DataStream so that you can change a parameter for each sequence slice.
    First it does not work like this (_prepare_stream only gets called once and not for each sequence slice).
    Second it would be great if we had a better base class to do things like this: Most of the code below just passes
    through function calls between the encapsulating DataStreams. If you forget one of these calls, the whole system
    breaks down in a hard-to-debug way. Especially you need to ensure to connect the "data_available_event", which is
    not obvious. I'd like a "PassThroughDataStream" that you can subclass and only override the methods you actually
    want to change, otherwise it would be a no-op DataStream.
    """

    def __init__(self, data_stream: Acquisition.DataStream):
        super().__init__()
        self.__data_stream = data_stream
        self.parameter = 0
        self.__listener = data_stream.data_available_event.listen(self.data_available_event.fire)

    def about_to_delete(self) -> None:
        self.__listener.close()
        self.__listener = None

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return self.__data_stream.channels

    @property
    def _progress(self) -> typing.Tuple[int, int]:
        return self.__data_stream.progress

    def _prepare_stream(self, stream_args: Acquisition.DataStream) -> None:
        self.parameter += 1
        self.__data_stream.prepare_stream(stream_args)

    def _abort_stream(self) -> None:
        self.__data_stream.abort_stream()

    def _send_next(self) -> None:
        self.__data_stream.send_next()

    def _start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        self.__data_stream.start_stream(stream_args)

    def _advance_stream(self) -> None:
        self.__data_stream.advance_stream()

    def _finish_stream(self) -> None:
        self.__data_stream.finish_stream()


class TestAcquisitionClass(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_camera_sequence_acquisition(self):
        sequence_len = 4
        data_stream = SingleFrameDataStream(sequence_len, (2, 2), 0)
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with maker.ref():
            maker.acquire()
            self.assertTrue(numpy.array_equal(data_stream.data, maker.get_data(0).data))

    def test_camera_collection_acquisition(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (4, 3)
        data_stream = SingleFrameDataStream(numpy.product(collection_shape), (2, 2), 0)
        collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                    [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_shape = collection_shape + maker.get_data(0).data.shape[-len(collection_shape):]
            self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(0).data))

    def test_camera_collection_acquisition_with_grouping(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (12, 12)
        for count in (4, 9, 16, 36):
            with self.subTest(count=count):
                data_stream = MultiFrameDataStream(numpy.product(collection_shape), (2, 2), 0, count)
                collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                            [Calibration.Calibration(), Calibration.Calibration()])
                maker = Acquisition.DataStreamToDataAndMetadata(collector)
                with maker.ref():
                    maker.acquire()
                    expected_shape = collection_shape + maker.get_data(0).data.shape[-len(collection_shape):]
                    self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(0).data))

    def test_scan_sequence_acquisition(self):
        sequence_len = 4
        data_stream = SingleFrameDataStream(sequence_len, (4, 4), 0, 2)
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with maker.ref():
            maker.acquire()
            self.assertTrue(numpy.array_equal(data_stream.data, maker.get_data(0).data))

    def test_scan_collection_acquisition(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan.
        # the scan data is hard coded to produce a scan.
        collection_shape = (5, 3)
        data_stream = SingleFrameDataStream(numpy.product(collection_shape), (4, 4), 0, 2)
        collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                    [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_shape = collection_shape + maker.get_data(0).data.shape[-len(collection_shape):]
            self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(0).data))

    def test_scan_as_collection(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        scan_shape = (8, 8)
        data_stream = ScanDataStream(1, scan_shape, [0], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_shape = scan_shape
            self.assertTrue(numpy.array_equal(data_stream.data[0].reshape(expected_shape), maker.get_data(0).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(0).data_descriptor)

    def test_scan_as_collection_with_arbitrary_length(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        scan_shape = (12, 12)
        for partial_length in (4, 9, 16, 36):
            with self.subTest(partial_length=partial_length):
                data_stream = ScanDataStream(1, scan_shape, [0], 9)
                collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
                maker = Acquisition.DataStreamToDataAndMetadata(collector)
                with maker.ref():
                    maker.acquire()
                    expected_shape = scan_shape
                    self.assertTrue(numpy.array_equal(data_stream.data[0].reshape(expected_shape), maker.get_data(0).data))
                    self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(0).data_descriptor)

    def test_scan_as_collection_as_sequence(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        # that will be collected to a sequence.
        sequence_len = 4
        scan_shape = (8, 8)
        data_stream = ScanDataStream(sequence_len, scan_shape, [0], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with maker.ref():
            maker.acquire()
            expected_shape = (sequence_len,) + scan_shape
            self.assertTrue(numpy.array_equal(data_stream.data[0].reshape(expected_shape), maker.get_data(0).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(0).data_descriptor)

    def test_scan_as_collection_two_channels(self):
        # scan will produce two data streams of pixels.
        # the collection must make it into two images.
        scan_shape = (8, 8)
        data_stream = ScanDataStream(1, scan_shape, [0, 1], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_shape = scan_shape
            self.assertTrue(numpy.array_equal(data_stream.data[0].reshape(expected_shape), maker.get_data(0).data))
            self.assertTrue(numpy.array_equal(data_stream.data[1].reshape(expected_shape), maker.get_data(1).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(0).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(1).data_descriptor)

    def test_scan_as_collection_two_channels_and_camera_summed_vertically(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        scan_data_stream = ScanDataStream(1, scan_shape, [0, 1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (2, 2), 2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator(axis=0))
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, summed_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_scan_shape = scan_shape
            expected_camera_shape = scan_shape + (2,)
            self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.get_data(0).data))
            self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.get_data(1).data))
            self.assertTrue(numpy.array_equal(camera_data_stream.data.sum(-2).reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(0).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(1).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 2, 1), maker.get_data(2).data_descriptor)

    def test_scan_as_collection_two_channels_and_multi_camera_summed_vertically(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        for with_processing in (False, True):
            with self.subTest(with_processing=with_processing):
                scan_shape = (8, 8)
                scan_data_stream = ScanDataStream(1, scan_shape, [0, 1], scan_shape[1])
                camera_data_stream = MultiFrameDataStream(numpy.product(scan_shape), (2, 2), 2, scan_shape[1], with_processing)
                summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator(axis=0))
                combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, summed_data_stream])
                collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
                maker = Acquisition.DataStreamToDataAndMetadata(collector)
                with maker.ref():
                    maker.acquire()
                    expected_scan_shape = scan_shape
                    expected_camera_shape = scan_shape + (2,)
                    self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.get_data(0).data))
                    self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.get_data(1).data))
                    self.assertTrue(numpy.array_equal(camera_data_stream.data.sum(-2).reshape(expected_camera_shape), maker.get_data(2).data))
                    self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(0).data_descriptor)
                    self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(1).data_descriptor)
                    self.assertEqual(DataAndMetadata.DataDescriptor(False, 2, 1), maker.get_data(2).data_descriptor)

    def test_scan_as_collection_camera_summed_to_single_scalar(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (2, 2), 2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator())
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_camera_shape = scan_shape
            self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(2).data_descriptor)

    def test_scan_as_collection_camera_summed_to_single_scalar_in_mask(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        mask1 = Shape.Rectangle.from_tlbr_fractional(0.0, 0.0, 0.5, 0.5)
        mask2 = Shape.Rectangle.from_tlbr_fractional(0.5, 0.5, 1.0, 1.0)
        mask = Shape.OrComposite2DShape(mask1, mask2)
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (8, 8), 2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.MaskedSumOperator(mask))
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_camera_shape = scan_shape
            mask_data = numpy.zeros((8, 8))
            mask_data[0:4, 0:4] = 1
            mask_data[4:8, 4:8] = 1
            self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(2).data_descriptor)

    def test_scan_as_collection_camera_summed_to_two_scalars(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (2, 2), 2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, Acquisition.CompositeDataStreamOperator({11: Acquisition.SumOperator(), 22: Acquisition.SumOperator()}))
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_camera_shape = scan_shape
            self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(11).data))
            self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(22).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(11).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(22).data_descriptor)

    def test_scan_as_collection_camera_summed_to_two_scalars_in_masks(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        mask1 = Shape.Rectangle.from_tlbr_fractional(0.0, 0.0, 0.5, 0.5)
        mask2 = Shape.Rectangle.from_tlbr_fractional(0.5, 0.5, 1.0, 1.0)
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (8, 8), 2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, Acquisition.CompositeDataStreamOperator({11: Acquisition.MaskedSumOperator(mask1), 22: Acquisition.MaskedSumOperator(mask2)}))
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_camera_shape = scan_shape
            mask_data1 = numpy.zeros((8, 8))
            mask_data1[0:4, 0:4] = 1
            mask_data2 = numpy.zeros((8, 8))
            mask_data2[4:8, 4:8] = 1
            self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data1).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(11).data))
            self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data2).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(22).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(11).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(22).data_descriptor)

    def test_scan_as_collection_two_channels_and_camera_summed_to_scalar(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        scan_data_stream = ScanDataStream(1, scan_shape, [0, 1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (2, 2), 2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator())
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, summed_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(collector)
        with maker.ref():
            maker.acquire()
            expected_scan_shape = scan_shape
            expected_camera_shape = scan_shape
            self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.get_data(0).data))
            self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.get_data(1).data))
            self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(0).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(1).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(2).data_descriptor)

    def test_sequence_of_scan_as_collection_two_channels_and_camera(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        scan_data_stream = ScanDataStream(sequence_len, scan_shape, [0, 1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(sequence_len * numpy.product(scan_shape), (2, 2), 2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with maker.ref():
            maker.acquire()
            expected_scan_shape = (sequence_len,) + scan_shape
            expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
            self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.get_data(0).data))
            self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.get_data(1).data))
            self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(0).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(1).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.get_data(2).data_descriptor)
            self.assertEqual(sequence_len, scan_data_stream.prepare_count)
            p = maker.progress
            self.assertEqual(p[0], p[1])

    def test_sequence_grouped_into_sections_of_scan_as_collection_two_channels_and_camera(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        scan_data_stream = ScanDataStream(sequence_len, scan_shape, [0, 1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(sequence_len * numpy.product(scan_shape), (2, 2), 2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        scan_slices = [[slice(0, 4), slice(0, 8)], [slice(4, 8), slice(0, 8)]]
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()], scan_slices)
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with maker.ref():
            maker.acquire()
            expected_scan_shape = (sequence_len,) + scan_shape
            expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
            self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.get_data(0).data))
            self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.get_data(1).data))
            self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(0).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(1).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.get_data(2).data_descriptor)
            self.assertEqual(sequence_len * 2, scan_data_stream.prepare_count)

    def test_sequence_of_individually_started_scans_as_collection_two_channels_and_camera(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        scan_data_stream = ScanDataStream(1, scan_shape, [0, 1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (2, 2), 2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with maker.ref():
            maker.acquire()
            expected_scan_shape = (sequence_len,) + scan_shape
            expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
            # self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.get_data(0).data))
            # self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.get_data(1).data))
            # self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertSequenceEqual(expected_scan_shape, maker.get_data(0).data.shape)
            self.assertSequenceEqual(expected_scan_shape, maker.get_data(1).data.shape)
            self.assertSequenceEqual(expected_camera_shape, maker.get_data(2).data.shape)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(0).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(1).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.get_data(2).data_descriptor)
            self.assertEqual(sequence_len, scan_data_stream.prepare_count)
            p = maker.progress
            self.assertEqual(p[0], p[1])

    def test_sequence_of_individually_started_scans_as_collection_two_channels_and_camera_with_parameter_change(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        scan_data_stream = ScanDataStream(1, scan_shape, [0, 1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(numpy.product(scan_shape), (2, 2), 2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        changer = ChangeParameterDataStream(collector)
        sequencer = Acquisition.SequenceDataStream(changer, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with maker.ref():
            maker.acquire()
            expected_scan_shape = (sequence_len,) + scan_shape
            expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
            # self.assertTrue(numpy.array_equal(scan_data_stream.data[0].reshape(expected_scan_shape), maker.get_data(0).data))
            # self.assertTrue(numpy.array_equal(scan_data_stream.data[1].reshape(expected_scan_shape), maker.get_data(1).data))
            # self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.get_data(2).data))
            self.assertSequenceEqual(expected_scan_shape, maker.get_data(0).data.shape)
            self.assertSequenceEqual(expected_scan_shape, maker.get_data(1).data.shape)
            self.assertSequenceEqual(expected_camera_shape, maker.get_data(2).data.shape)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(0).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(1).data_descriptor)
            self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.get_data(2).data_descriptor)
            self.assertEqual(sequence_len, scan_data_stream.prepare_count)
            self.assertEqual(sequence_len, changer.parameter)
            p = maker.progress
            self.assertEqual(p[0], p[1])
