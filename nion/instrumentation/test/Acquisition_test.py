import numpy
import typing
import unittest
import weakref

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition
from nion.instrumentation.test import AcquisitionTestContext
from nion.utils import Geometry


class ScanDataStream(Acquisition.DataStream):
    """Provide a data stream for one scan with the given channel.

    frame_count is the number of frames to generate.

    scan_shape is the shape of each frame.

    channels are the list of channels to generate.

    partial_length is the size of each chunk of data (number of samples) to send at once.
    """
    def __init__(self, frame_count: int, scan_shape: Acquisition.ShapeType, channels: typing.Sequence[Acquisition.Channel], partial_length: int, *, error_after: typing.Optional[int] = None):
        super().__init__(frame_count * numpy.prod(scan_shape, dtype=numpy.int64))
        # frame counts are used for allocating and returning test data
        self.__frame_count = frame_count
        self.__frame_index = 0
        # scan length is total samples in scan shape
        self.__scan_shape = scan_shape
        self.__scan_length = int(numpy.prod(scan_shape, dtype=numpy.int64))
        # channels
        self.__channels = tuple(channels)
        # partial length is the size of each chunk sent. partial index is the next sample to be sent.
        self.__partial_length = partial_length
        self.__partial_index = 0
        self.data = {channel: numpy.random.randn(self.__frame_count, self.__scan_length) for channel in channels}
        self.prepare_count = 0
        self.__error_after = error_after

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return self.__channels

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, index_stack: Acquisition.IndexDescriptionList, **kwargs) -> None:
        self.prepare_count += 1

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[Acquisition.DataStream], Acquisition.DataStreamEventArgs]]:
        raw_data_stream_events = list[typing.Tuple[weakref.ReferenceType[Acquisition.DataStream], Acquisition.DataStreamEventArgs]]()
        assert self.__frame_index < self.__frame_count
        assert self.__partial_index < self.__scan_length
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 0)
        # update the index to be used in the data slice
        start_index = self.__partial_index
        stop_index = min(start_index + self.__partial_length, self.__scan_length)
        new_count = stop_index - start_index
        # source data slice is relative to data start/stop
        source_data_slice = (slice(start_index, stop_index),)
        state = Acquisition.DataStreamStateEnum.PARTIAL if stop_index < self.__scan_length else Acquisition.DataStreamStateEnum.COMPLETE
        for channel in self.channels:
            if self.__error_after is not None:
                if self.__error_after == 0:
                    new_count = 0  # this will trigger an exception in send data
                self.__error_after -= 1
            data_metadata = DataAndMetadata.DataMetadata(((), self.data[channel].dtype), data_descriptor=data_descriptor)
            data_stream_event = Acquisition.DataStreamEventArgs(channel, data_metadata,
                                                                self.data[channel][self.__frame_index],
                                                                new_count, source_data_slice, Acquisition.DataStreamStateEnum.COMPLETE)
            raw_data_stream_events.append((weakref.ref(self), data_stream_event))
        # update indexes
        if state == Acquisition.DataStreamStateEnum.COMPLETE:
            self.__partial_index = 0
            self.__frame_index = self.__frame_index + 1
        else:
            self.__partial_index = stop_index
        return raw_data_stream_events

    def _build_data_handler(self, data_handler: Acquisition.DataHandler) -> bool:
        return False


class SingleFrameDataStream(Acquisition.DataStream):
    """Provide a single data stream frame by frame.

    frame_count is the number of frames to generate.

    frame_shape is the shape of each frame.

    channel is the channel on which to send the data.

    partial_height is the size of each chunk of data (number of samples) to send at once.
    """
    def __init__(self, frame_count: int, frame_shape: Acquisition.ShapeType, channel: Acquisition.Channel, partial_height: typing.Optional[int] = None, error_after: typing.Optional[int] = None):
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
        self.__error_after = error_after

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return (self.__channel,)

    def _get_info(self, channel: Acquisition.Channel) -> Acquisition.DataStreamInfo:
        return Acquisition.DataStreamInfo(DataAndMetadata.DataMetadata((self.__frame_shape, self.data.dtype)), 0.1)

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[Acquisition.DataStream], Acquisition.DataStreamEventArgs]]:
        raw_data_stream_events = list[typing.Tuple[weakref.ReferenceType[Acquisition.DataStream], Acquisition.DataStreamEventArgs]]()
        assert self.__frame_index < self.__frame_count
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape))
        data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape, typing.cast(numpy.dtype, float)), data_descriptor=data_descriptor)
        # update the index to be used in the data slice
        new_partial = min(self.__partial_index + self.__partial_height, self.__frame_shape[0])
        source_data_slice = (slice(self.__partial_index, new_partial), slice(None))
        # send the data with no count. this is required when using partial.
        state = Acquisition.DataStreamStateEnum.PARTIAL if new_partial < self.__frame_shape[0] else Acquisition.DataStreamStateEnum.COMPLETE
        data_stream_event = Acquisition.DataStreamEventArgs(self.__channel, data_metadata,
                                                            self.data[self.__frame_index], None,
                                                            source_data_slice, state)
        raw_data_stream_events.append((weakref.ref(self), data_stream_event))
        if state == Acquisition.DataStreamStateEnum.PARTIAL:
            self.__partial_index = new_partial
        else:
            self.__partial_index = 0
            self.__frame_index += 1
        if self.__error_after is not None:
            if self.__error_after == 0:
                raise Exception()
            self.__error_after -= 1
        return raw_data_stream_events

    def _build_data_handler(self, data_handler: Acquisition.DataHandler) -> bool:
        return False



class MultiFrameDataStream(Acquisition.DataStream):
    """Provide a single data stream frame by frame, n at a time.

    frame_count is the number of frames to generate.

    frame_shape is the shape of each frame.

    channel is the channel on which to send the data.

    n is the number of frames to send at once.
    """
    def __init__(self, frame_count: int, frame_shape: Acquisition.ShapeType, channel: Acquisition.Channel, count: typing.Optional[int] = None, do_processing: bool = False, counts: typing.Optional[typing.Sequence[int]] = None):
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
        self.__counts = counts
        self.__counts_index = 0
        self.__do_processing = do_processing
        self.data = numpy.random.randn(self.__frame_count, *self.__frame_shape)

    @property
    def channels(self) -> typing.Tuple[Acquisition.Channel, ...]:
        return (self.__channel,)

    def _get_info(self, channel: Acquisition.Channel) -> Acquisition.DataStreamInfo:
        # this is a hack - it ignores 'do_processing' since that will be handled by the operator
        return Acquisition.DataStreamInfo(DataAndMetadata.DataMetadata((self.__frame_shape, self.data.dtype)), 0.1)

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, index_stack: Acquisition.IndexDescriptionList, **kwargs) -> None:
        if self.__do_processing:
            operator = typing.cast(Acquisition.DataStreamOperator, kwargs.get("operator", Acquisition.NullDataStreamOperator()))
            operator.apply()

    def _get_raw_data_stream_events(self) -> typing.Sequence[typing.Tuple[weakref.ReferenceType[Acquisition.DataStream], Acquisition.DataStreamEventArgs]]:
        raw_data_stream_events = list[typing.Tuple[weakref.ReferenceType[Acquisition.DataStream], Acquisition.DataStreamEventArgs]]()
        assert self.__frame_index < self.__frame_count
        # data metadata describes the data being sent from this stream: shape, data type, and descriptor
        if self.__do_processing:
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape) - 1)
            data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape[1:], self.data.dtype), data_descriptor=data_descriptor)
        else:
            data_descriptor = DataAndMetadata.DataDescriptor(False, 0, len(self.__frame_shape))
            data_metadata = DataAndMetadata.DataMetadata((self.__frame_shape, self.data.dtype), data_descriptor=data_descriptor)
        # update the index to be used in the data slice
        if self.__counts:
            count = self.__counts[self.__counts_index]
            self.__counts_index += 1
        else:
            count = min(self.__count, self.__frame_count - self.__frame_index)
        source_data_slice: typing.Tuple[slice, ...] = (slice(0, count), slice(None), slice(None))
        if self.__do_processing:
            source_data_slice = source_data_slice[:-1]
        # send the data with no count. this is required when using partial.
        state = Acquisition.DataStreamStateEnum.COMPLETE
        source_data = self.data[self.__frame_index:self.__frame_index + count]
        if self.__do_processing:
            source_data = source_data.sum(axis=1)
        data_stream_event = Acquisition.DataStreamEventArgs(self.__channel, data_metadata, source_data, count, source_data_slice, state)
        raw_data_stream_events.append((weakref.ref(self), data_stream_event))
        self.__frame_index += count
        return raw_data_stream_events

    def _build_data_handler(self, data_handler: Acquisition.DataHandler) -> bool:
        return False


class RectangleMask(Acquisition.MaskLike):
    def __init__(self, r: Geometry.FloatRect):
        self.__r = r

    def get_mask_array(self, data_shape: Acquisition.ShapeType) -> numpy.ndarray:
        ir = Geometry.IntRect.from_tlbr(round(self.__r.top * data_shape[0]),
                                        round(self.__r.left * data_shape[1]),
                                        round(self.__r.bottom * data_shape[0]),
                                        round(self.__r.right * data_shape[1]))
        mask = numpy.zeros(data_shape)
        mask[ir.slice] = 1.0
        return mask


class OrMask(Acquisition.MaskLike):
    def __init__(self, mask1: Acquisition.MaskLike, mask2: Acquisition.MaskLike):
        self.__mask1 = mask1
        self.__mask2 = mask2

    def get_mask_array(self, data_shape: Acquisition.ShapeType) -> numpy.ndarray:
        mask_data1 = self.__mask1.get_mask_array(data_shape)
        mask_data2 = self.__mask2.get_mask_array(data_shape)
        return numpy.where(numpy.logical_or(mask_data1, mask_data2), 1.0, 0.0)


class TestAcquisitionClass(unittest.TestCase):

    def setUp(self):
        AcquisitionTestContext.begin_leaks()

    def tearDown(self):
        AcquisitionTestContext.end_leaks(self)

    def test_unravel_flat_slice(self):
        test_cases = [
            ((4, 3, 5), (2, 42)),
            ((4, 3, 5), (7, 42)),
            ((4, 3, 5), (5, 42)),
            ((4, 3, 5), (2, 4)),
            ((4, 3, 5), (2, 7)),
            ((4, 3, 5), (2, 17)),
            ((1, 3, 5), (2, 13)),
            ((4, 1, 5), (2, 17)),
            ((4, 3, 5), (0, 60)),
            ((2, 4, 3, 5), (45, 60)),
        ]
        for shape, range_slice in test_cases:
            start, stop = range_slice
            count = stop - start
            d = numpy.zeros(shape, dtype=int)
            for s in Acquisition.unravel_flat_slice(slice(start, stop), shape):
                for si in s:
                    assert (si.start is None and si.stop is None) or (si.start < si.stop)
                d[s] = 1
            self.assertEqual(count, numpy.sum(d))
            self.assertEqual(count, numpy.sum(d.reshape(-1)[start:stop]))

    def test_channel_equality_and_hash(self):
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        self.assertNotEqual(channel0, channel1)
        self.assertNotEqual(hash(channel0), hash(channel1))
        self.assertEqual(channel0, channel0)
        self.assertEqual(hash(channel0), hash(channel0))
        self.assertEqual(channel1, channel1)
        self.assertEqual(hash(channel1), hash(channel1))
        self.assertEqual(channel0, Acquisition.Channel("0"))
        self.assertEqual(hash(channel0), hash(Acquisition.Channel("0")))
        self.assertEqual(channel1, Acquisition.Channel("1"))
        self.assertEqual(hash(channel1), hash(Acquisition.Channel("1")))

    def test_camera_sequence_acquisition(self):
        sequence_len = 4
        channel = Acquisition.Channel("0")
        data_stream = SingleFrameDataStream(sequence_len, (2, 2), channel)
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
        maker = Acquisition.MakerDataStream(sequencer)
        Acquisition.acquire(maker)
        self.assertTrue(numpy.array_equal(data_stream.data, maker.get_data(channel).data))

    def test_camera_collection_acquisition(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (4, 3)
        channel = Acquisition.Channel("0")
        data_stream = SingleFrameDataStream(numpy.prod(collection_shape), (2, 2), channel)
        collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                    [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_shape = collection_shape + maker.get_data(channel).data.shape[len(collection_shape):]
        self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(channel).data))

    def test_camera_collection_acquisition_with_individual_slices(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (4, )
        channel = Acquisition.Channel("0")
        data_stream = SingleFrameDataStream(numpy.prod(collection_shape), (2, 2), channel, partial_height=1)
        collectors = [Acquisition.CollectedDataStream(data_stream, (1,), [Calibration.Calibration()]) for i in numpy.ndindex(*collection_shape)]
        collector = Acquisition.StackedDataStream(collectors)
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_shape = collection_shape + maker.get_data(channel).data.shape[len(collection_shape):]
        self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(channel).data))

    def test_camera_collection_acquisition_with_grouping(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (12, 12)
        for count in (4, 9, 16, 36):
            with self.subTest(count=count):
                channel = Acquisition.Channel("0")
                data_stream = MultiFrameDataStream(numpy.prod(collection_shape), (2, 2), channel, count)
                collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                            [Calibration.Calibration(), Calibration.Calibration()])
                maker = Acquisition.MakerDataStream(collector)
                Acquisition.acquire(maker)
                expected_shape = collection_shape + maker.get_data(channel).data.shape[len(collection_shape):]
                self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(channel).data))
                maker = None

    def test_camera_collection_acquisition_with_inconsistent_grouping(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan
        collection_shape = (12, 12)
        # the 32 will be 10 from previous row; 12 in complete row; 10 in next row
        # this triggered an error
        counts = (2, 24, 32, 86)
        channel = Acquisition.Channel("0")
        data_stream = MultiFrameDataStream(numpy.prod(collection_shape), (2, 2), channel, counts=counts)
        collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                    [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_shape = collection_shape + maker.get_data(channel).data.shape[len(collection_shape):]
        self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(channel).data))

    def test_scan_sequence_acquisition(self):
        sequence_len = 4
        channel = Acquisition.Channel("0")
        data_stream = SingleFrameDataStream(sequence_len, (4, 4), channel, 2)
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
        maker = Acquisition.MakerDataStream(sequencer)
        Acquisition.acquire(maker)
        self.assertTrue(numpy.array_equal(data_stream.data, maker.get_data(channel).data))

    def test_scan_collection_acquisition(self):
        # in this case the collector is acting only to arrange the data, not to provide any scan.
        # the scan data is hard coded to produce a scan.
        collection_shape = (5, 3)
        channel = Acquisition.Channel("0")
        data_stream = SingleFrameDataStream(numpy.prod(collection_shape), (4, 4), channel, 2)
        collector = Acquisition.CollectedDataStream(data_stream, collection_shape,
                                                    [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_shape = collection_shape + maker.get_data(channel).data.shape[len(collection_shape):]
        self.assertTrue(numpy.array_equal(data_stream.data.reshape(expected_shape), maker.get_data(channel).data))

    def test_scan_as_collection(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        scan_shape = (8, 8)
        channel = Acquisition.Channel("0")
        data_stream = ScanDataStream(1, scan_shape, [channel], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_shape = scan_shape
        self.assertTrue(numpy.array_equal(data_stream.data[channel].reshape(expected_shape), maker.get_data(channel).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel).data_descriptor)

    def test_scan_as_collection_with_arbitrary_length(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        scan_shape = (12, 12)
        channel = Acquisition.Channel("0")
        for partial_length in (4, 9, 16, 36):
            with self.subTest(partial_length=partial_length):
                data_stream = ScanDataStream(1, scan_shape, [channel], 9)
                collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
                maker = Acquisition.MakerDataStream(collector)
                Acquisition.acquire(maker)
                expected_shape = scan_shape
                self.assertTrue(numpy.array_equal(data_stream.data[channel].reshape(expected_shape), maker.get_data(channel).data))
                self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel).data_descriptor)
                maker = None

    def test_scan_as_collection_as_sequence(self):
        # scan will produce a data stream of pixels.
        # the collection must make it into an image.
        # that will be collected to a sequence.
        sequence_len = 4
        scan_shape = (8, 8)
        channel = Acquisition.Channel("0")
        data_stream = ScanDataStream(sequence_len, scan_shape, [channel], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.MakerDataStream(sequencer)
        Acquisition.acquire(maker)
        expected_shape = (sequence_len,) + scan_shape
        self.assertTrue(numpy.array_equal(data_stream.data[channel].reshape(expected_shape), maker.get_data(channel).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(channel).data_descriptor)

    def test_scan_as_collection_two_channels(self):
        # scan will produce two data streams of pixels.
        # the collection must make it into two images.
        scan_shape = (8, 8)
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        data_stream = ScanDataStream(1, scan_shape, [channel0, channel1], scan_shape[1])
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_shape = scan_shape
        self.assertTrue(numpy.array_equal(data_stream.data[channel0].reshape(expected_shape), maker.get_data(channel0).data))
        self.assertTrue(numpy.array_equal(data_stream.data[channel1].reshape(expected_shape), maker.get_data(channel1).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel0).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel1).data_descriptor)

    def test_scan_as_collection_two_channels_and_camera_summed_vertically(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        channel2 = Acquisition.Channel("2")
        scan_data_stream = ScanDataStream(1, scan_shape, [channel0, channel1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(numpy.prod(scan_shape), (2, 2), channel2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator(axis=0))
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, summed_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_scan_shape = scan_shape
        expected_camera_shape = scan_shape + (2,)
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel0].reshape(expected_scan_shape), maker.get_data(channel0).data))
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel1].reshape(expected_scan_shape), maker.get_data(channel1).data))
        self.assertTrue(numpy.array_equal(camera_data_stream.data.sum(-2).reshape(expected_camera_shape), maker.get_data(channel2).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel0).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel1).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 2, 1), maker.get_data(channel2).data_descriptor)

    def test_scan_as_collection_two_channels_and_multi_camera_summed_vertically(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        for with_processing in (False, True):
            with self.subTest(with_processing=with_processing):
                scan_shape = (8, 8)
                channel0 = Acquisition.Channel("0")
                channel1 = Acquisition.Channel("1")
                channel2 = Acquisition.Channel("2")
                scan_data_stream = ScanDataStream(1, scan_shape, [channel0, channel1], scan_shape[1])
                camera_data_stream = MultiFrameDataStream(numpy.prod(scan_shape), (2, 2), channel2, scan_shape[1], with_processing)
                summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator(axis=0))
                combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, summed_data_stream])
                collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
                maker = Acquisition.MakerDataStream(collector)
                Acquisition.acquire(maker)
                expected_scan_shape = scan_shape
                expected_camera_shape = scan_shape + (2,)
                self.assertTrue(numpy.array_equal(scan_data_stream.data[channel0].reshape(expected_scan_shape), maker.get_data(channel0).data))
                self.assertTrue(numpy.array_equal(scan_data_stream.data[channel1].reshape(expected_scan_shape), maker.get_data(channel1).data))
                self.assertTrue(numpy.array_equal(camera_data_stream.data.sum(-2).reshape(expected_camera_shape), maker.get_data(channel2).data))
                self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel0).data_descriptor)
                self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel1).data_descriptor)
                self.assertEqual(DataAndMetadata.DataDescriptor(False, 2, 1), maker.get_data(channel2).data_descriptor)

    def test_collection_camera_summed_to_single_scalar(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        channel = Acquisition.Channel("2")
        camera_data_stream = SingleFrameDataStream(numpy.prod(scan_shape), (2, 2), channel)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator())
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_camera_shape = scan_shape
        self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel).data_descriptor)

    def test_collection_camera_summed_to_single_scalar_in_mask(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        mask1 = RectangleMask(Geometry.FloatRect.from_tlbr(0.0, 0.0, 0.5, 0.5))
        mask2 = RectangleMask(Geometry.FloatRect.from_tlbr(0.5, 0.5, 1.0, 1.0))
        mask = OrMask(mask1, mask2)
        channel = Acquisition.Channel("2")
        camera_data_stream = SingleFrameDataStream(numpy.prod(scan_shape), (8, 8), channel)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.MaskedSumOperator(mask))
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_camera_shape = scan_shape
        mask_data = numpy.zeros((8, 8))
        mask_data[0:4, 0:4] = 1
        mask_data[4:8, 4:8] = 1
        self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel).data_descriptor)

    def test_collection_camera_summed_to_two_scalars(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (2, 2)
        channel = Acquisition.Channel("2")
        channel11 = Acquisition.Channel("11")
        channel22 = Acquisition.Channel("22")
        camera_data_stream = SingleFrameDataStream(numpy.prod(scan_shape), (2, 2), channel)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.CompositeDataStreamOperator({channel11: Acquisition.SumOperator(), channel22: Acquisition.SumOperator()}))
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_camera_shape = scan_shape
        self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel11).data))
        self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel22).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel11).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel22).data_descriptor)

    def test_collection_camera_summed_to_two_scalars_in_masks(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        mask1 = RectangleMask(Geometry.FloatRect.from_tlbr(0.0, 0.0, 0.5, 0.5))
        mask2 = RectangleMask(Geometry.FloatRect.from_tlbr(0.5, 0.5, 1.0, 1.0))
        channel = Acquisition.Channel("2")
        channel11 = Acquisition.Channel("11")
        channel22 = Acquisition.Channel("22")
        camera_data_stream = SingleFrameDataStream(numpy.prod(scan_shape), (8, 8), channel)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.CompositeDataStreamOperator({channel11: Acquisition.MaskedSumOperator(mask1), channel22: Acquisition.MaskedSumOperator(mask2)}))
        collector = Acquisition.CollectedDataStream(summed_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_camera_shape = scan_shape
        mask_data1 = numpy.zeros((8, 8))
        mask_data1[0:4, 0:4] = 1
        mask_data2 = numpy.zeros((8, 8))
        mask_data2[4:8, 4:8] = 1
        self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data1).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel11).data))
        self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data2).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel22).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel11).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel22).data_descriptor)

    def test_collection_camera_summed_to_two_scalars_in_masks_into_stack(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        mask1 = RectangleMask(Geometry.FloatRect.from_tlbr(0.0, 0.0, 0.5, 0.5))
        mask2 = RectangleMask(Geometry.FloatRect.from_tlbr(0.5, 0.5, 1.0, 1.0))
        channel = Acquisition.Channel("2")
        camera_data_stream = SingleFrameDataStream(numpy.prod(scan_shape), (8, 8), channel)
        stacked_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.StackedDataStreamOperator([Acquisition.MaskedSumOperator(mask1), Acquisition.MaskedSumOperator(mask2)]))
        collector = Acquisition.CollectedDataStream(stacked_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        move_axis_data_stream = Acquisition.FramedDataStream(collector, operator=Acquisition.MoveAxisDataStreamOperator())
        maker = Acquisition.MakerDataStream(move_axis_data_stream)
        Acquisition.acquire(maker)
        expected_camera_shape = scan_shape
        mask_data1 = numpy.zeros((8, 8))
        mask_data1[0:4, 0:4] = 1
        mask_data2 = numpy.zeros((8, 8))
        mask_data2[4:8, 4:8] = 1
        self.assertEqual((2, 8, 8), maker.get_data(channel).data_shape)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 1, 2), maker.get_data(channel).data_descriptor)
        self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data1).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel).data[0]))
        self.assertTrue(numpy.array_equal((camera_data_stream.data * mask_data2).sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel).data[1]))

    def test_scan_as_collection_two_channels_and_camera_summed_to_scalar(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        scan_shape = (8, 8)
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        channel2 = Acquisition.Channel("2")
        scan_data_stream = ScanDataStream(1, scan_shape, [channel0, channel1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(numpy.prod(scan_shape), (2, 2), channel2)
        summed_data_stream = Acquisition.FramedDataStream(camera_data_stream, operator=Acquisition.SumOperator())
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, summed_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        Acquisition.acquire(maker)
        expected_scan_shape = scan_shape
        expected_camera_shape = scan_shape
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel0].reshape(expected_scan_shape), maker.get_data(channel0).data))
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel1].reshape(expected_scan_shape), maker.get_data(channel1).data))
        self.assertTrue(numpy.array_equal(camera_data_stream.data.sum((-2, -1)).reshape(expected_camera_shape), maker.get_data(channel2).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel0).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel1).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel2).data_descriptor)

    def test_sequence_of_scan_as_collection_two_channels_and_camera(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        channel2 = Acquisition.Channel("2")
        scan_data_stream = ScanDataStream(sequence_len, scan_shape, [channel0, channel1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(sequence_len * numpy.prod(scan_shape), (2, 2), channel2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.MakerDataStream(sequencer)
        Acquisition.acquire(maker)
        expected_scan_shape = (sequence_len,) + scan_shape
        expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel0].reshape(expected_scan_shape), maker.get_data(channel0).data))
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel1].reshape(expected_scan_shape), maker.get_data(channel1).data))
        self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.get_data(channel2).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(channel0).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(channel1).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.get_data(channel2).data_descriptor)
        self.assertEqual(sequence_len, scan_data_stream.prepare_count)

    def test_sequence_grouped_into_sections_of_scan_as_collection_two_channels_and_camera(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        channel2 = Acquisition.Channel("2")
        scan_data_stream = ScanDataStream(sequence_len, scan_shape, [channel0, channel1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(sequence_len * numpy.prod(scan_shape), (2, 2), channel2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.StackedDataStream([
            Acquisition.CollectedDataStream(combined_data_stream, (4, 8), [Calibration.Calibration(), Calibration.Calibration()]),
            Acquisition.CollectedDataStream(combined_data_stream, (4, 8), [Calibration.Calibration(), Calibration.Calibration()]),
            ])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        maker = Acquisition.MakerDataStream(sequencer)
        Acquisition.acquire(maker)
        expected_scan_shape = (sequence_len,) + scan_shape
        expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel0].reshape(expected_scan_shape), maker.get_data(channel0).data))
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel1].reshape(expected_scan_shape), maker.get_data(channel1).data))
        self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.get_data(channel2).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(channel0).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(channel1).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.get_data(channel2).data_descriptor)
        self.assertEqual(sequence_len * 2, scan_data_stream.prepare_count)

    def test_sequence_split_into_slices_of_scan_as_collection_two_channels_and_camera(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        channel2 = Acquisition.Channel("2")
        scan_data_stream = ScanDataStream(sequence_len, scan_shape, [channel0, channel1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(sequence_len * numpy.prod(scan_shape), (2, 2), channel2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.CollectedDataStream(combined_data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        sequencer = Acquisition.StackedDataStream([Acquisition.SequenceDataStream(collector, 1) for i in range(sequence_len)])
        maker = Acquisition.MakerDataStream(sequencer)
        Acquisition.acquire(maker)
        expected_scan_shape = (sequence_len,) + scan_shape
        expected_camera_shape = (sequence_len,) + scan_shape + (2, 2)
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel0].reshape(expected_scan_shape), maker.get_data(channel0).data))
        self.assertTrue(numpy.array_equal(scan_data_stream.data[channel1].reshape(expected_scan_shape), maker.get_data(channel1).data))
        self.assertTrue(numpy.array_equal(camera_data_stream.data.reshape(expected_camera_shape), maker.get_data(channel2).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(channel0).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 0, 2), maker.get_data(channel1).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(True, 2, 2), maker.get_data(channel2).data_descriptor)
        self.assertEqual(sequence_len, scan_data_stream.prepare_count)

    def test_sequence_grouped_into_sections_of_scan_as_collection_two_channels_and_camera_and_accumulated(self):
        # scan will produce two data streams of pixels.
        # camera will produce one stream of frames.
        # the sequence must make it into two images and a sequence of images.
        sequence_len = 4
        scan_shape = (8, 8)
        channel0 = Acquisition.Channel("0")
        channel1 = Acquisition.Channel("1")
        channel2 = Acquisition.Channel("2")
        scan_data_stream = ScanDataStream(sequence_len, scan_shape, [channel0, channel1], scan_shape[1])
        camera_data_stream = SingleFrameDataStream(sequence_len * numpy.prod(scan_shape), (2, 2), channel2)
        combined_data_stream = Acquisition.CombinedDataStream([scan_data_stream, camera_data_stream])
        collector = Acquisition.StackedDataStream([
            Acquisition.CollectedDataStream(combined_data_stream, (4, 8), [Calibration.Calibration(), Calibration.Calibration()]),
            Acquisition.CollectedDataStream(combined_data_stream, (4, 8), [Calibration.Calibration(), Calibration.Calibration()]),
            ])
        sequencer = Acquisition.SequenceDataStream(collector, sequence_len)
        accumulator = Acquisition.AccumulatedDataStream(sequencer, False, True)
        maker = Acquisition.MakerDataStream(accumulator)
        Acquisition.acquire(maker)
        expected_scan_shape = scan_shape
        sequence_camera_shape = (sequence_len,) + scan_shape + (2, 2)
        channel0sum = channel0.join_segment("sum")
        channel1sum = channel1.join_segment("sum")
        channel2sum = channel2.join_segment("sum")
        self.assertTrue(numpy.array_equal(numpy.sum(scan_data_stream.data[channel0], axis=0).reshape(expected_scan_shape), maker.get_data(channel0sum).data))
        self.assertTrue(numpy.array_equal(numpy.sum(scan_data_stream.data[channel1], axis=0).reshape(expected_scan_shape), maker.get_data(channel1sum).data))
        numpy.testing.assert_array_almost_equal(numpy.sum(camera_data_stream.data.reshape(sequence_camera_shape), axis=0), maker.get_data(channel2sum).data)
        self.assertTrue(numpy.array_equal(numpy.sum(camera_data_stream.data.reshape(sequence_camera_shape), axis=0), maker.get_data(channel2sum).data))
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel0sum).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 0, 2), maker.get_data(channel1sum).data_descriptor)
        self.assertEqual(DataAndMetadata.DataDescriptor(False, 2, 2), maker.get_data(channel2sum).data_descriptor)
        self.assertEqual(sequence_len * 2, scan_data_stream.prepare_count)

    def test_sequence_of_stacked_collections(self):
        channel = Acquisition.Channel("Cam")
        camera_data_stream = SingleFrameDataStream(32, (2, 2), channel)
        section1 = Acquisition.CollectedDataStream(camera_data_stream, (2, 4), [Calibration.Calibration(), Calibration.Calibration()])
        section2 = Acquisition.CollectedDataStream(camera_data_stream, (2, 4), [Calibration.Calibration(), Calibration.Calibration()])
        sections = Acquisition.StackedDataStream((section1, section2))
        maker = Acquisition.MakerDataStream(Acquisition.SequenceDataStream(sections, 2))
        Acquisition.acquire(maker)
        self.assertFalse(maker.is_error)
        maker_data = maker.get_data(channel)
        self.assertTrue(numpy.array_equal(numpy.reshape(camera_data_stream.data, maker_data.data_shape), maker_data.data))

    def test_scan_as_collection_sequential(self):
        channel = Acquisition.Channel("Cam")
        sequence_len1 = 4
        data_stream1 = SingleFrameDataStream(sequence_len1, (2, 2), channel)
        sequencer1 = Acquisition.SequenceDataStream(data_stream1, sequence_len1)
        sequence_len2 = 8
        data_stream2 = SingleFrameDataStream(sequence_len2, (2, 2), channel)
        sequencer2 = Acquisition.SequenceDataStream(data_stream2, sequence_len2)
        maker = Acquisition.MakerDataStream(Acquisition.SequentialDataStream((sequencer1, sequencer2)))
        Acquisition.acquire(maker)
        self.assertTrue(numpy.array_equal(data_stream1.data, maker.get_data(Acquisition.Channel("0", *channel.segments)).data))
        self.assertTrue(numpy.array_equal(data_stream2.data, maker.get_data(Acquisition.Channel("1", *channel.segments)).data))

    def test_scan_as_collection_sequential_with_reuse(self):
        channel = Acquisition.Channel("Cam")
        sequence_len = 4
        data_stream = SingleFrameDataStream(sequence_len * 2, (2, 2), channel)
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
        maker = Acquisition.MakerDataStream(Acquisition.SequentialDataStream((sequencer, sequencer)))
        Acquisition.acquire(maker)
        self.assertEqual(2, len(maker.channels))
        self.assertTrue(numpy.array_equal(data_stream.data[0:4], maker.get_data(Acquisition.Channel("0", *channel.segments)).data))
        self.assertTrue(numpy.array_equal(data_stream.data[4:8], maker.get_data(Acquisition.Channel("1", *channel.segments)).data))

    def test_action_stream_start_and_finish(self):
        sequence_len = 4
        channel = Acquisition.Channel("0")
        data_stream = SingleFrameDataStream(sequence_len, (2, 2), channel)

        class Action(Acquisition.ActionValueControllerLike):
            def __init__(self) -> None:
                self._s = 0
                self._p = 0
                self._f = 0

            def start(self, **kwargs: typing.Any) -> None:
                self._s += 1

            def perform(self, index: Acquisition.ShapeType, **kwargs: typing.Any) -> None:
                assert self._s == 1
                assert self._f == 0
                self._p += 1

            def finish(self, **kwargs: typing.Any) -> None:
                self._f += 1

        action = Action()
        sequencer = Acquisition.SequenceDataStream(Acquisition.ActionDataStream(data_stream, action), sequence_len)
        maker = Acquisition.MakerDataStream(sequencer)
        Acquisition.acquire(maker)
        self.assertTrue(numpy.array_equal(data_stream.data, maker.get_data(channel).data))
        self.assertEqual(1, action._s)
        self.assertEqual(4, action._p)
        self.assertEqual(1, action._f)

    def test_error_during_send_next(self):
        sequence_len = 4
        channel = Acquisition.Channel("0")
        data_stream = SingleFrameDataStream(sequence_len, (2, 2), channel, error_after=0)
        sequencer = Acquisition.SequenceDataStream(data_stream, sequence_len)
        maker = Acquisition.MakerDataStream(sequencer)
        had_error = False

        def handle_error(e: Exception) -> None:
            nonlocal had_error
            had_error = True

        Acquisition.acquire(maker, error_handler=handle_error)
        self.assertTrue(had_error)
        self.assertTrue(maker.is_error)

    def test_error_during_data_available(self):
        scan_shape = (8, 8)
        channel = Acquisition.Channel("0")
        data_stream = ScanDataStream(1, scan_shape, [channel], scan_shape[1], error_after=1)
        collector = Acquisition.CollectedDataStream(data_stream, scan_shape, [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.MakerDataStream(collector)
        had_error = False

        def handle_error(e: Exception) -> None:
            nonlocal had_error
            had_error = True

        Acquisition.acquire(maker, error_handler=handle_error)
        self.assertTrue(had_error)
        self.assertTrue(maker.is_error)
        self.assertAlmostEqual(1/8, collector.progress)

    def test_serial_handler(self) -> None:

        class CountingDataHandler(Acquisition.DataHandler):
            def __init__(self) -> None:
                super().__init__()
                self.count = 0

            def handle_data_available(self, packet: Acquisition.DataStreamEventArgs) -> None:
                self.count += packet.count or 1

        counter1 = CountingDataHandler()
        counter2 = CountingDataHandler()

        serial_handler = Acquisition.SerialDataHandler()
        serial_handler.add_data_handler(counter1, 4)
        serial_handler.add_data_handler(counter2, 4)

        channel = Acquisition.Channel("0")
        data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 1)
        data_metadata = DataAndMetadata.DataMetadata(((4,), float), data_descriptor=data_descriptor)
        source_data = numpy.zeros((4, 1), dtype=float)

        data_stream_event = Acquisition.DataStreamEventArgs(channel, data_metadata, source_data, 2, (slice(0, 2), slice(None)), Acquisition.DataStreamStateEnum.COMPLETE)

        # send the data 2 rows at a time
        for _ in range(4):
            serial_handler.handle_data_available(data_stream_event)

        # check the counts to ensure the data is sent to both handlers
        self.assertEqual(4, counter1.count)
        self.assertEqual(4, counter2.count)
