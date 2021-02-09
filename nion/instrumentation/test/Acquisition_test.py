import contextlib
import numpy
import unittest

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import Acquisition

class TestAcquisitionClass(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sequence_acquisition(self):
        sequence_len = 4
        width = 2
        height = 2
        data_stream = Acquisition.DataStream()
        sequencer = Acquisition.Sequencer(data_stream, sequence_len)
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with contextlib.closing(maker):
            data = numpy.random.randn(sequence_len, height, width)
            for i in range(sequence_len):
                data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
                data_metadata = DataAndMetadata.DataMetadata(((height, width), float),
                                                             data_descriptor=data_descriptor)
                data_stream_event = Acquisition.DataStreamEvent(data_stream, 0, data_metadata,
                                                                (slice(0, height), slice(0, width)),
                                                                data[i])
                data_stream.data_available_event.fire(data_stream_event)
            self.assertTrue(numpy.array_equal(data, maker.data[0].data))

    def test_collection_acquisition(self):
        collection_shape = (4, 3)
        width = 2
        height = 2
        data_stream = Acquisition.DataStream()
        sequencer = Acquisition.Collector(data_stream, collection_shape,
                                          [Calibration.Calibration(), Calibration.Calibration()])
        maker = Acquisition.DataStreamToDataAndMetadata(sequencer)
        with contextlib.closing(maker):
            data = numpy.random.randn(*collection_shape, height, width)
            for y in range(collection_shape[0]):
                for x in range(collection_shape[1]):
                    data_descriptor = DataAndMetadata.DataDescriptor(False, 0, 2)
                    data_metadata = DataAndMetadata.DataMetadata(((height, width), float),
                                                                 data_descriptor=data_descriptor)
                    data_stream_event = Acquisition.DataStreamEvent(data_stream, 0, data_metadata,
                                                                    (slice(0, height), slice(0, width)), data[y, x])
                    data_stream.data_available_event.fire(data_stream_event)
            self.assertTrue(numpy.array_equal(data, maker.data[0].data))
