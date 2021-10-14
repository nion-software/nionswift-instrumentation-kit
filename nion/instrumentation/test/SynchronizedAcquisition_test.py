import copy
import math
import numpy
import threading
import time
import typing
import unittest
import uuid

from nion.data import DataAndMetadata
from nion.instrumentation import HardwareSource
from nion.swift import Application
from nion.swift import Facade
from nion.swift.model import ApplicationData
from nion.swift.model import Metadata
from nion.ui import TestUI
from nion.utils import Geometry
from nion.instrumentation import Acquisition
from nion.instrumentation import DataChannel
from nion.instrumentation import DriftTracker
from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation.test import AcquisitionTestContext
from nionswift_plugin.nion_instrumentation_ui import ScanAcquisition

"""
# running in Swift
import sys, unittest
from superscan import SimulatorScanControl_test
suite = unittest.TestLoader().loadTestsFromTestCase(SimulatorScanControl_test.TestSimulatorScanControlClass)
result = unittest.TextTestResult(sys.stdout, True, True)
suite.run(result)
"""

class TestSynchronizedAcquisitionClass(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)

    def __test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.test_context(is_eels=is_eels)

    def _acquire_one(self, document_controller, hardware_source):
        hardware_source.start_playing(sync_timeout=3.0)
        hardware_source.stop_playing(sync_timeout=3.0)
        document_controller.periodic()

    def test_grab_synchronized_basic_eels(self):
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None
            scans, spectrum_images = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters, camera=camera_hardware_source, camera_frame_parameters=camera_frame_parameters, camera_data_channel=camera_data_channel)
            # check the acquisition state
            self.assertFalse(camera_hardware_source.camera._is_acquire_synchronized_running)
            # check the data
            self.assertEqual(scans[0].data_shape, spectrum_images[0].data_shape[:-1])
            self.assertEqual(numpy.float32, spectrum_images[0].data_dtype)
            self.assertEqual(DataAndMetadata.DataDescriptor(False, 2, 1), spectrum_images[0].data_descriptor)
            # check the calibrations
            self.assertEqual(tuple(scans[0].dimensional_calibrations), tuple(spectrum_images[0].dimensional_calibrations[:-1]))
            self.assertEqual("nm", spectrum_images[0].dimensional_calibrations[0].units)
            self.assertEqual("nm", spectrum_images[0].dimensional_calibrations[1].units)
            self.assertEqual("eV", spectrum_images[0].dimensional_calibrations[-1].units)
            self.assertEqual("counts", spectrum_images[0].intensity_calibration.units)
            # check the timestamp
            self.assertEqual(scans[0].timezone, spectrum_images[0].timezone)
            # self.assertEqual(scans[0].timezone_offset, spectrum_images[0].timezone_offset)
            # self.assertEqual(scans[0].timestamp, spectrum_images[0].timestamp)
            # check the metadata
            for metadata_source in spectrum_images:
                # import pprint; print(pprint.pformat(metadata_source.metadata))
                self.assertEqual(camera_hardware_source.hardware_source_id, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.id"))
                self.assertEqual(camera_hardware_source.display_name, Metadata.get_metadata_value(metadata_source, "stem.hardware_source.name"))
                self.assertGreater(Metadata.get_metadata_value(metadata_source, "stem.camera.exposure"), 0.0)
                self.assertIsNotNone(Metadata.get_metadata_value(metadata_source, "stem.signal_type"))
                self.assertIsNone(Metadata.get_metadata_value(metadata_source, "stem.scan.channel_index"))
                self.assertIsNone(Metadata.get_metadata_value(metadata_source, "stem.scan.channel_id"))
                self.assertIsNone(Metadata.get_metadata_value(metadata_source, "stem.scan.channel_name"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.center_x_nm"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.center_x_nm"))
                # self.assertIsNone(Metadata.get_metadata_value(metadata_source, "stem.scan.frame_time"))
                self.assertEqual(100.0, Metadata.get_metadata_value(metadata_source, "stem.scan.fov_nm"))
                self.assertIsNone(Metadata.get_metadata_value(metadata_source, "stem.scan.pixel_time_us"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata_source, "stem.scan.rotation"))
                self.assertIsNotNone(uuid.UUID(Metadata.get_metadata_value(metadata_source, "stem.scan.scan_id")))
                self.assertIsNone(Metadata.get_metadata_value(metadata_source, "stem.hardware_source.valid_rows"))
                self.assertEqual(scan_frame_parameters.size[1], Metadata.get_metadata_value(metadata_source, "stem.scan.valid_rows"))
                self.assertIsNone(Metadata.get_metadata_value(metadata_source, "stem.scan.line_time_us"))
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("EHT"), Metadata.get_metadata_value(metadata_source, "stem.high_tension"))
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("C10"), Metadata.get_metadata_value(metadata_source, "stem.defocus"))
                self.assertEqual((4, 4), metadata_source.metadata["scan"]["scan_context_size"])
                self.assertEqual((4, 4), metadata_source.metadata["scan"]["scan_size"])

    def test_grab_synchronized_abort(self):
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None

            class TestAbortBehavior(scan_base.SynchronizedScanBehaviorInterface):
                def __init__(self) -> None:
                    self.__i = 0

                def prepare_section(self, **kwargs) -> None:
                    self.__i += 1
                    if self.__i == 2:
                        scan_hardware_source.grab_synchronized_abort()

            abort_behavior = TestAbortBehavior()
            scans_and_spectrum_images = scan_hardware_source.grab_synchronized(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters,
                camera_data_channel=camera_data_channel,
                section_height=2,
                scan_behavior=abort_behavior)
            self.assertIsNone(scans_and_spectrum_images)
            # check the acquisition state
            self.assertFalse(camera_hardware_source.camera._is_acquire_synchronized_running)

    def test_grab_synchronized_sequence_basic_eels(self):
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None
            scans, spectrum_images = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters,
                                                                            camera=camera_hardware_source,
                                                                            camera_frame_parameters=camera_frame_parameters,
                                                                            camera_data_channel=camera_data_channel,
                                                                            scan_count=3)
            # check the acquisition state
            self.assertFalse(camera_hardware_source.camera._is_acquire_synchronized_running)
            self.assertEqual(1, len(scans))
            self.assertEqual(1, len(spectrum_images))
            # self.assertEqual((4, 4, 512), spectrum_images[0].data_shape)  # assumes accumulate for now
            self.assertEqual((3, 4, 4, 512), spectrum_images[0].data_shape)

    def test_grab_synchronized_basic_eels_followed_by_record(self):
        # perform a synchronized acquisition followed by a record. tests that the record frame parameters are restored
        # after a synchronized acquisition.
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters2 = scan_hardware_source.get_frame_parameters(2)
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None
            scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters, camera=camera_hardware_source, camera_frame_parameters=camera_frame_parameters, camera_data_channel=camera_data_channel)
            frame_time = scan_frame_parameters2.pixel_time_us * scan_frame_parameters2.size[0] * scan_frame_parameters2.size[1] / 1000000.0
            scan_hardware_source.start_recording()
            time.sleep(frame_time * 0.6)
            self.assertEqual(scan_hardware_source.get_next_xdatas_to_finish(10.0)[0].data.shape, (1024, 1024))
            # check the acquisition state
            self.assertFalse(camera_hardware_source.camera._is_acquire_synchronized_running)

    def test_grab_synchronized_camera_data_channel_basic_use(self):
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            data_item_data_channel = DataChannel.DataItemDataChannel(test_context.document_model, "data", {
                Acquisition.Channel(scan_hardware_source.hardware_source_id, "0"): "HAADF",
                Acquisition.Channel(camera_hardware_source.hardware_source_id): "test"})
            scan_hardware_source.grab_synchronized(data_channel=data_item_data_channel,
                                                   scan_frame_parameters=scan_frame_parameters,
                                                   camera=camera_hardware_source,
                                                   camera_frame_parameters=camera_frame_parameters)
            # check the acquisition state
            self.assertFalse(camera_hardware_source.camera._is_acquire_synchronized_running)

    def test_grab_synchronized_camera_data_channel_basic_sum_masked(self):
        with self.__test_context() as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_masked"
            data_item_data_channel = DataChannel.DataItemDataChannel(test_context.document_model, "data", {
                Acquisition.Channel(scan_hardware_source.hardware_source_id, "0"): "HAADF",
                Acquisition.Channel(camera_hardware_source.hardware_source_id): "test"})
            scan_hardware_source.grab_synchronized(data_channel=data_item_data_channel,
                                                   scan_frame_parameters=scan_frame_parameters,
                                                   camera=camera_hardware_source,
                                                   camera_frame_parameters=camera_frame_parameters)

    def test_grab_synchronized_sum_masked_produces_data_of_correct_shape(self):
        with self.__test_context() as test_context:
            document_model = test_context.document_model
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 5)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_masked"
            masks = [camera_base.Mask(), camera_base.Mask(), camera_base.Mask()]
            camera_frame_parameters.active_masks = masks
            data_item_data_channel = DataChannel.DataItemDataChannel(document_model, "Spectrum Image", {
                Acquisition.Channel(scan_hardware_source.hardware_source_id, "0"): "HAADF",
                Acquisition.Channel(camera_hardware_source.hardware_source_id): "test"})
            scan_hardware_source.grab_synchronized(data_channel=data_item_data_channel,
                                                   scan_frame_parameters=scan_frame_parameters,
                                                   camera=camera_hardware_source,
                                                   camera_frame_parameters=camera_frame_parameters)
            document_controller.periodic()
            si_data_item = None
            for data_item in document_model.data_items:
                if "Spectrum Image" in data_item.title and "test" in data_item.title:
                    si_data_item = data_item
                    break
            self.assertIsNotNone(si_data_item)
            self.assertEqual((3, 4, 5), si_data_item.data_shape)

    def test_grab_rotated_synchronized_eels(self):
        # tests whether rotation was applied, as judged by the resulting metadata
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            scan_frame_parameters.rotation_rad = math.radians(30)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None
            scans, spectrum_images = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters, camera=camera_hardware_source, camera_frame_parameters=camera_frame_parameters, camera_data_channel=camera_data_channel)
            for metadata_source in spectrum_images:
                self.assertAlmostEqual(math.radians(30), Metadata.get_metadata_value(metadata_source, "stem.scan.rotation"))

    def test_grab_sync_info_has_proper_calibrations(self):
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(8, 8)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters)
            self.assertEqual(2, len(grab_sync_info.scan_calibrations))
            self.assertEqual("nm", grab_sync_info.scan_calibrations[0].units)
            self.assertEqual("nm", grab_sync_info.scan_calibrations[1].units)
            self.assertEqual(1, len(grab_sync_info.data_calibrations))
            self.assertEqual("eV", grab_sync_info.data_calibrations[0].units)
            self.assertEqual("counts", grab_sync_info.data_intensity_calibration.units)

    def test_grab_sync_info_has_proper_camera_metadata(self):
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(8, 8)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters)
            metadata = {
                "scan": copy.deepcopy(grab_sync_info.scan_metadata),
                "hardware_source": copy.deepcopy(grab_sync_info.camera_metadata),
                "instrument": copy.deepcopy(grab_sync_info.instrument_metadata)
            }
            self.assertEqual(camera_hardware_source.hardware_source_id, Metadata.get_metadata_value(metadata, "stem.hardware_source.id"))
            self.assertEqual(camera_hardware_source.display_name, Metadata.get_metadata_value(metadata, "stem.hardware_source.name"))
            self.assertEqual(scan_hardware_source.stem_controller.GetVal("EHT"), Metadata.get_metadata_value(metadata, "stem.high_tension"))
            self.assertEqual(scan_hardware_source.stem_controller.GetVal("C10"), Metadata.get_metadata_value(metadata, "stem.defocus"))
            self.assertGreater(Metadata.get_metadata_value(metadata, "stem.camera.exposure"), 0.0)
            self.assertIsNotNone(Metadata.get_metadata_value(metadata, "stem.signal_type"))
            # 'is_dark_subtracted': False,
            # 'is_flipped_horizontally': False,
            # 'is_gain_corrected': False,
            # 'sensor_dimensions_hw': (2048, 2048),
            # 'sensor_readout_area_tlbr': (964, 0, 1084, 2048),

    def test_grab_sync_info_has_proper_scan_metadata(self):
        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters)
            metadata = {
                "scan": copy.deepcopy(grab_sync_info.scan_metadata),
                "hardware_source": copy.deepcopy(grab_sync_info.camera_metadata),
                "instrument": copy.deepcopy(grab_sync_info.instrument_metadata)
            }
            self.assertEqual(camera_hardware_source.hardware_source_id, Metadata.get_metadata_value(metadata, "stem.hardware_source.id"))
            self.assertEqual(camera_hardware_source.display_name, Metadata.get_metadata_value(metadata, "stem.hardware_source.name"))
            self.assertEqual(scan_hardware_source.stem_controller.GetVal("EHT"), Metadata.get_metadata_value(metadata, "stem.high_tension"))
            self.assertEqual(scan_hardware_source.stem_controller.GetVal("C10"), Metadata.get_metadata_value(metadata, "stem.defocus"))
            self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.channel_index"))
            self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.channel_id"))
            self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.channel_name"))
            self.assertEqual(0.0, Metadata.get_metadata_value(metadata, "stem.scan.center_x_nm"))
            self.assertEqual(0.0, Metadata.get_metadata_value(metadata, "stem.scan.center_x_nm"))
            # self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.frame_time"))
            self.assertEqual(100.0, Metadata.get_metadata_value(metadata, "stem.scan.fov_nm"))
            self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.pixel_time_us"))
            self.assertEqual(0.0, Metadata.get_metadata_value(metadata, "stem.scan.rotation"))
            self.assertIsNotNone(uuid.UUID(Metadata.get_metadata_value(metadata, "stem.scan.scan_id")))
            self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.line_time_us"))
            self.assertEqual((4, 4), metadata["scan"]["scan_context_size"])
            self.assertEqual((4, 4), metadata["scan"]["scan_size"])

    def test_grab_sync_info_has_proper_session_metadata(self):
        with self.__test_context() as test_context:
            ApplicationData.get_session_metadata_model().microscopist = "Ned Flanders"

            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_specifier = ScanAcquisition.ScanSpecifier()
            scan_specifier.size = 4, 4
            scan_acquisition_controller = ScanAcquisition.ScanAcquisitionController(Facade.get_api("~1.0", "~1.0"),
                                                                                    Facade.DocumentWindow(document_controller),
                                                                                    Facade.HardwareSource(scan_hardware_source),
                                                                                    Facade.HardwareSource(camera_hardware_source),
                                                                                    scan_specifier)
            scan_acquisition_controller.start(ScanAcquisition.ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisition.ScanProcessing(True, False))
            scan_acquisition_controller._wait()
            document_controller.periodic()

            for data_item in document_model.data_items:
                self.assertEqual("Ned Flanders", Metadata.get_metadata_value(data_item, "stem.session.microscopist"))

            # self.assertEqual("Ned Flanders", data_item.session_metadata["microscopist"])

    def test_partial_acquisition_has_proper_metadata(self):

        class TestDataChannel(DataChannel.DataItemDataChannel):
            def __init__(self, document_model, channel_names: typing.Mapping[Acquisition.Channel, str]):
                super().__init__(document_model, "test", channel_names)
                self.__document_model = document_model
                self.updates: typing.List[DataAndMetadata.DataAndMetadata] = list()

            def update_data(self, channel: Acquisition.Channel, source_data: numpy.ndarray,
                            source_slice: Acquisition.SliceType, dest_slice: slice,
                            data_metadata: DataAndMetadata.DataMetadata) -> None:
                super().update_data(channel, source_data, source_slice, dest_slice, data_metadata)
                if channel == Acquisition.Channel(camera_hardware_source.hardware_source_id):
                    self.__document_model.perform_data_item_updates()
                    channel_xdata = self.get_data_item(channel).xdata
                    assert channel_xdata
                    self.updates.append(copy.deepcopy(channel_xdata))

        with self.__test_context(is_eels=True) as test_context:
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(6, 6)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            data_channel = TestDataChannel(test_context.document_model,
                                           {Acquisition.Channel(scan_hardware_source.hardware_source_id, "0"): "HAADF",
                                            Acquisition.Channel(camera_hardware_source.hardware_source_id): "test"})

            section_height = 5
            scan_hardware_source.grab_synchronized(data_channel=data_channel,
                                                   scan_frame_parameters=scan_frame_parameters,
                                                   camera=camera_hardware_source,
                                                   camera_frame_parameters=camera_frame_parameters,
                                                   section_height=section_height)
            self.assertTrue(data_channel.updates)
            for partial_xdata in data_channel.updates:
                self.assertEqual("counts", partial_xdata.intensity_calibration.units)
                self.assertEqual("nm", partial_xdata.dimensional_calibrations[0].units)
                self.assertEqual("nm", partial_xdata.dimensional_calibrations[1].units)
                self.assertEqual("eV", partial_xdata.dimensional_calibrations[2].units)
                metadata = partial_xdata.metadata
                # import pprint; print(pprint.pformat(metadata))
                self.assertEqual(camera_hardware_source.hardware_source_id, Metadata.get_metadata_value(metadata, "stem.hardware_source.id"))
                self.assertEqual(camera_hardware_source.display_name, Metadata.get_metadata_value(metadata, "stem.hardware_source.name"))
                self.assertNotIn("autostem", metadata["hardware_source"])
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("EHT"), Metadata.get_metadata_value(metadata, "stem.high_tension"))
                self.assertEqual(scan_hardware_source.stem_controller.GetVal("C10"), Metadata.get_metadata_value(metadata, "stem.defocus"))
                self.assertGreater(Metadata.get_metadata_value(metadata, "stem.camera.exposure"), 0.0)
                self.assertIsNotNone(Metadata.get_metadata_value(metadata, "stem.signal_type"))
                self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.channel_index"))
                self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.channel_id"))
                self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.channel_name"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata, "stem.scan.center_x_nm"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata, "stem.scan.center_x_nm"))
                # self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.frame_time"))
                self.assertEqual(100.0, Metadata.get_metadata_value(metadata, "stem.scan.fov_nm"))
                self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.pixel_time_us"))
                self.assertEqual(0.0, Metadata.get_metadata_value(metadata, "stem.scan.rotation"))
                self.assertIsNotNone(uuid.UUID(Metadata.get_metadata_value(metadata, "stem.scan.scan_id")))
                self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.hardware_source.valid_rows"))
                self.assertTrue(1 <= Metadata.get_metadata_value(metadata, "stem.scan.valid_rows") <= 6)
                self.assertIsNone(Metadata.get_metadata_value(metadata, "stem.scan.line_time_us"))
                self.assertEqual((6, 6), metadata["scan"]["scan_context_size"])
                self.assertEqual((6, 6), metadata["scan"]["scan_size"])
                # camera_metadata = partial_xdata.metadata["hardware_source"]
                # self.assertIn("autostem", camera_metadata)
                # self.assertIn("hardware_source_id", camera_metadata)
                # self.assertIn("hardware_source_name", camera_metadata)
                # self.assertIn("exposure", camera_metadata)
                # self.assertIn("binning", camera_metadata)
                # self.assertIn("signal_type", camera_metadata)
                # scan_metadata = partial_xdata.metadata["scan_detector"]
                # self.assertIn("autostem", scan_metadata)
                # self.assertIn("hardware_source_id", scan_metadata)
                # self.assertIn("hardware_source_name", scan_metadata)
                # self.assertIn("center_x_nm", scan_metadata)
                # self.assertIn("center_y_nm", scan_metadata)
                # self.assertNotIn("channel_index", scan_metadata)
                # self.assertNotIn("channel_name", scan_metadata)
                # self.assertIn("fov_nm", scan_metadata)
                # self.assertIn("rotation", scan_metadata)
                # self.assertIn("scan_id", scan_metadata)

    def test_grab_synchronized_basic_eels_with_drift_correction(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.drift_channel_id = scan_hardware_source.data_channels[0].channel_id
            scan_hardware_source.drift_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            document_controller.periodic()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(8, 8)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None
            drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters)
            scans, spectrum_images = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters,
                                                                            camera=camera_hardware_source,
                                                                            camera_frame_parameters=camera_frame_parameters,
                                                                            camera_data_channel=camera_data_channel,
                                                                            section_height=2,
                                                                            scan_behavior=drift_correction_behavior)

    def test_grab_synchronized_basic_eels_with_drift_correction_leaves_graphic_during_scan(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.drift_channel_id = scan_hardware_source.data_channels[0].channel_id
            scan_hardware_source.drift_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            document_controller.periodic()
            display_item = document_model.display_items[-1]
            drift_graphic = display_item.graphics[-1]
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(16, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None
            drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters)
            def do_grab():
                scans, spectrum_images = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters,
                                                                                camera=camera_hardware_source,
                                                                                camera_frame_parameters=camera_frame_parameters,
                                                                                camera_data_channel=camera_data_channel,
                                                                                section_height=2,
                                                                                scan_behavior=drift_correction_behavior)
            # run the grab synchronized in a thread so that periodic can be called so that
            # the graphics get updated in ui thread.
            t = threading.Thread(target=do_grab)
            t.start()
            while t.is_alive():
                document_controller.periodic()
                import time
                time.sleep(0.1)
            t.join()
            # ensure graphic is still the original one and hasn't flickered with a replacement
            self.assertEqual(drift_graphic, display_item.graphics[-1])

    def test_grab_synchronized_basic_eels_with_drift_correction_leaves_graphic_during_subscan(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.drift_channel_id = scan_hardware_source.data_channels[0].channel_id
            scan_hardware_source.drift_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            document_controller.periodic()
            display_item = document_model.display_items[-1]
            drift_graphic = display_item.graphics[-1]
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.scan_id = uuid.uuid4()
            scan_frame_parameters.size = Geometry.IntSize(32, 8)
            scan_frame_parameters.subscan_pixel_size = Geometry.IntSize(16, 4)
            scan_frame_parameters.subscan_fractional_size = Geometry.FloatSize(0.5, 0.5)
            scan_frame_parameters.subscan_fractional_center = Geometry.FloatPoint(0.5, 0.5)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters.processing = "sum_project"
            camera_data_channel = None
            drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters)
            def do_grab():
                scans, spectrum_images = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters,
                                                                                camera=camera_hardware_source,
                                                                                camera_frame_parameters=camera_frame_parameters,
                                                                                camera_data_channel=camera_data_channel,
                                                                                section_height=2,
                                                                                scan_behavior=drift_correction_behavior)
            # run the grab synchronized in a thread so that periodic can be called so that
            # the graphics get updated in ui thread.
            t = threading.Thread(target=do_grab)
            t.start()
            while t.is_alive():
                document_controller.periodic()
                import time
                time.sleep(0.1)
            t.join()
            # ensure graphic is still the original one and hasn't flickered with a replacement
            self.assertEqual(drift_graphic, display_item.graphics[-1])

    def test_drift_corrector(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            drift_tracker = scan_hardware_source.drift_tracker
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.drift_channel_id = scan_hardware_source.data_channels[0].channel_id
            scan_hardware_source.drift_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            document_controller.periodic()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.width)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.height)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertLess(dist_nm, 0.1)
            stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id("usim_stem_controller")
            stem_controller.SetValDeltaAndConfirm("CSH.x", 2e-9, 1.0, 1000)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertTrue(1.9 < dist_nm < 2.1)
            self.assertTrue(1.9 < abs(last_delta_nm.width) < 2.1)
            self.assertTrue(abs(last_delta_nm.height) < 0.1)
            stem_controller.SetValDeltaAndConfirm("CSH.x", -2e-9, 1.0, 1000)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertTrue(1.9 < dist_nm < 2.1)
            self.assertTrue(1.9 < abs(last_delta_nm.width) < 2.1)
            self.assertTrue(abs(last_delta_nm.height) < 0.1)

    def test_drift_corrector_with_drift_sub_area_rotation(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            drift_tracker = scan_hardware_source.drift_tracker
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.drift_channel_id = scan_hardware_source.data_channels[0].channel_id
            scan_hardware_source.drift_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            rotation = math.radians(30)
            scan_hardware_source.drift_rotation = rotation
            document_controller.periodic()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            drift_correction_behavior = DriftTracker.DriftCorrectionBehavior(scan_hardware_source, scan_frame_parameters)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.width)
            self.assertEqual(0.0, drift_tracker.last_delta_nm.height)
            # offset will be rotated into the context reference frame
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertLess(dist_nm, 0.1)
            stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id("usim_stem_controller")
            stem_controller.SetValDeltaAndConfirm("CSH.x", 2e-9, 1.0, 1000)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertTrue(1.9 < dist_nm < 2.1)
            self.assertTrue(1.9 < abs(last_delta_nm.width) < 2.1)
            self.assertTrue(abs(last_delta_nm.height) < 0.1)
            stem_controller.SetValDeltaAndConfirm("CSH.x", -2e-9, 1.0, 1000)
            drift_correction_behavior.prepare_section(utc_time=drift_tracker._last_entry_utc_time)
            last_delta_nm = drift_tracker.last_delta_nm
            dist_nm = math.sqrt(pow(last_delta_nm.width, 2) + pow(last_delta_nm.height, 2))
            self.assertTrue(1.9 < dist_nm < 2.1)
            self.assertTrue(1.9 < abs(last_delta_nm.width) < 2.1)
            self.assertTrue(abs(last_delta_nm.height) < 0.1)

    def test_scan_acquisition_controller(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            scan_specifier = ScanAcquisition.ScanSpecifier()
            scan_specifier.scan_context = copy.deepcopy(scan_hardware_source.scan_context)
            scan_specifier.size = 4, 4
            scan_acquisition_controller = ScanAcquisition.ScanAcquisitionController(Facade.get_api("~1.0", "~1.0"),
                                                                                    Facade.DocumentWindow(document_controller),
                                                                                    Facade.HardwareSource(scan_hardware_source),
                                                                                    Facade.HardwareSource(camera_hardware_source),
                                                                                    scan_specifier)
            scan_acquisition_controller.start(ScanAcquisition.ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisition.ScanProcessing(True, False))
            scan_acquisition_controller._wait()
            document_controller.periodic()
            si_data_item = None
            for data_item in document_model.data_items:
                if "Spectrum Image" in data_item.title and "EELS" in data_item.title:
                    si_data_item = data_item
                    break
            self.assertIsNotNone(si_data_item)
            self.assertEqual((4, 4, 512), si_data_item.data_shape)
            metadata = si_data_item.metadata
            self.assertEqual((256, 256), metadata["scan"]["scan_context_size"])  # the synchronized scan is the context
            self.assertEqual((4, 4), metadata["scan"]["scan_size"])

    def slow_test_scan_acquisition_controller_eels(self):
        # tests case where camera data arrives in partial chunks.
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_specifier = ScanAcquisition.ScanSpecifier()
            scan_specifier.scan_context = copy.deepcopy(scan_hardware_source.scan_context)
            scan_specifier.size = 32, 32  # must be large enough to split acquisition into parts to highlight one error
            scan_acquisition_controller = ScanAcquisition.ScanAcquisitionController(Facade.get_api("~1.0", "~1.0"),
                                                                                    Facade.DocumentWindow(document_controller),
                                                                                    Facade.HardwareSource(scan_hardware_source),
                                                                                    Facade.HardwareSource(camera_hardware_source),
                                                                                    scan_specifier)
            scan_acquisition_controller.start(ScanAcquisition.ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisition.ScanProcessing(True, False))
            scan_acquisition_controller._wait()
            document_controller.periodic()
            si_data_item = None
            for data_item in document_model.data_items:
                if "Spectrum Image" in data_item.title and "EELS" in data_item.title:
                    si_data_item = data_item
                    break
            self.assertIsNotNone(si_data_item)
            self.assertEqual((32, 32, 512), si_data_item.data_shape)
            metadata = si_data_item.metadata
            self.assertEqual((32, 32), metadata["scan"]["scan_context_size"])  # the synchronized scan is the context
            self.assertEqual((32, 32), metadata["scan"]["scan_size"])

    def test_scan_acquisition_controller_with_rect(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            scan_specifier = ScanAcquisition.ScanSpecifier()
            scan_specifier.scan_context = copy.deepcopy(scan_hardware_source.scan_context)
            scan_specifier.size = 4, 4
            scan_acquisition_controller = ScanAcquisition.ScanAcquisitionController(Facade.get_api("~1.0", "~1.0"),
                                                                                    Facade.DocumentWindow(document_controller),
                                                                                    Facade.HardwareSource(scan_hardware_source),
                                                                                    Facade.HardwareSource(camera_hardware_source),
                                                                                    scan_specifier)
            scan_acquisition_controller.start(ScanAcquisition.ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisition.ScanProcessing(True, False))
            scan_acquisition_controller._wait()

            document_controller.periodic()
            si_data_item = None
            for data_item in document_model.data_items:
                if "Spectrum Image" in data_item.title and "EELS" in data_item.title:
                    si_data_item = data_item
                    break
            self.assertIsNotNone(si_data_item)
            self.assertEqual((4, 4, 512), si_data_item.data_shape)
            metadata = si_data_item.metadata
            self.assertEqual((256, 256), metadata["scan"]["scan_context_size"])  # the synchronized scan is the context
            self.assertEqual((4, 4), metadata["scan"]["scan_size"])

    def test_scan_acquisition_controller_with_rect_4d(self):
        with self.__test_context(is_eels=True) as test_context:
            document_controller = test_context.document_controller
            document_model = test_context.document_model
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            scan_hardware_source.subscan_region = Geometry.FloatRect.from_tlhw(0.25, 0.25, 0.5, 0.5)
            scan_specifier = ScanAcquisition.ScanSpecifier()
            scan_specifier.scan_context = copy.deepcopy(scan_hardware_source.scan_context)
            scan_specifier.size = 4, 4
            scan_acquisition_controller = ScanAcquisition.ScanAcquisitionController(Facade.get_api("~1.0", "~1.0"),
                                                                                    Facade.DocumentWindow(document_controller),
                                                                                    Facade.HardwareSource(scan_hardware_source),
                                                                                    Facade.HardwareSource(camera_hardware_source),
                                                                                    scan_specifier)
            scan_acquisition_controller.start(ScanAcquisition.ScanAcquisitionProcessing.NONE, ScanAcquisition.ScanProcessing(True, False))
            scan_acquisition_controller._wait()

            document_controller.periodic()
            si_data_item = None
            for data_item in document_model.data_items:
                if "Spectrum Image" in data_item.title and "EELS" in data_item.title:
                    si_data_item = data_item
                    break
            self.assertIsNotNone(si_data_item)
            self.assertEqual((4, 4, 128, 512), si_data_item.data_shape)
            metadata = si_data_item.metadata
            self.assertEqual((256, 256), metadata["scan"]["scan_context_size"])  # the synchronized scan is the context
            self.assertEqual((4, 4), metadata["scan"]["scan_size"])

    def test_scan_acquisition_controller_sum_masked(self):
        masks_list = [[], [camera_base.Mask()], [camera_base.Mask(), camera_base.Mask()]]
        for masks in masks_list:
            with self.subTest(number_masks=len(masks)):
                with self.__test_context() as test_context:
                    document_controller = test_context.document_controller
                    document_model = test_context.document_model
                    scan_hardware_source = test_context.scan_hardware_source
                    camera_hardware_source = test_context.camera_hardware_source
                    camera_frame_parameters = camera_hardware_source.get_frame_parameters(0)
                    camera_frame_parameters.active_masks = masks
                    camera_hardware_source.set_frame_parameters(0, camera_frame_parameters)
                    self._acquire_one(document_controller, scan_hardware_source)
                    scan_hardware_source.subscan_enabled = True
                    scan_specifier = ScanAcquisition.ScanSpecifier()
                    scan_specifier.scan_context = copy.deepcopy(scan_hardware_source.scan_context)
                    scan_specifier.size = 4, 4
                    scan_acquisition_controller = ScanAcquisition.ScanAcquisitionController(Facade.get_api("~1.0", "~1.0"),
                                                                                            Facade.DocumentWindow(document_controller),
                                                                                            Facade.HardwareSource(scan_hardware_source),
                                                                                            Facade.HardwareSource(camera_hardware_source),
                                                                                            scan_specifier)
                    scan_acquisition_controller.start(ScanAcquisition.ScanAcquisitionProcessing.SUM_MASKED, ScanAcquisition.ScanProcessing(True, False))
                    scan_acquisition_controller._wait()
                    document_controller.periodic()
                    si_data_item = None
                    for data_item in document_model.data_items:
                        if "Spectrum Image" in data_item.title and "Ronchigram" in data_item.title:
                            si_data_item = data_item
                            break
                    self.assertIsNotNone(si_data_item)
                    if len(masks) > 1:
                        self.assertEqual(si_data_item.collection_dimension_count, 1)
                        self.assertEqual((len(masks), 4, 4), si_data_item.data_shape)
                    else:
                        self.assertEqual(si_data_item.collection_dimension_count, 0)
                        self.assertEqual((4, 4), si_data_item.data_shape)
                    metadata = si_data_item.metadata
                    self.assertEqual((256, 256), metadata["scan"]["scan_context_size"])  # the synchronized scan is the context
                    self.assertEqual((4, 4), metadata["scan"]["scan_size"])

    def test_update_from_device_puts_current_profile_into_valid_state(self):
        with self.__test_context() as test_context:
            document_controller = test_context.document_controller
            scan_hardware_source = test_context.scan_hardware_source
            camera_hardware_source = test_context.camera_hardware_source
            self._acquire_one(document_controller, scan_hardware_source)
            scan_hardware_source.subscan_enabled = True
            self._acquire_one(document_controller, scan_hardware_source)
            scan_specifier = ScanAcquisition.ScanSpecifier()
            scan_specifier.scan_context = copy.deepcopy(scan_hardware_source.scan_context)
            scan_specifier.size = 4, 4
            scan_acquisition_controller = ScanAcquisition.ScanAcquisitionController(Facade.get_api("~1.0", "~1.0"),
                                                                                    Facade.DocumentWindow(document_controller),
                                                                                    Facade.HardwareSource(scan_hardware_source),
                                                                                    Facade.HardwareSource(camera_hardware_source),
                                                                                    scan_specifier)
            scan_acquisition_controller.start(ScanAcquisition.ScanAcquisitionProcessing.SUM_PROJECT, ScanAcquisition.ScanProcessing(True, False))
            scan_acquisition_controller._wait()
            updated_frame_parameters = scan_hardware_source.get_current_frame_parameters().as_dict()
            for k in list(updated_frame_parameters.keys()):
                if k not in ("size", "center_nm", "pixel_time_us", "fov_nm", "rotation_rad", "flyback_time_us"):
                    updated_frame_parameters.pop(k)
            scan_hardware_source._update_frame_parameters_test(0, scan_base.ScanFrameParameters(updated_frame_parameters))
            current_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            # import pprint; print(pprint.pformat(dict(current_frame_parameters)))
            self.assertIsNotNone(current_frame_parameters.channel_modifier)
            self.assertIsNotNone(current_frame_parameters.subscan_fractional_center)
            self.assertIsNotNone(current_frame_parameters.subscan_fractional_size)
            self.assertIsNotNone(current_frame_parameters.subscan_pixel_size)

    # TODO: check for counts per electron


if __name__ == '__main__':
    unittest.main()
