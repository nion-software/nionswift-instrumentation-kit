import collections
import copy
import numpy
import unittest
import uuid

from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import DocumentController
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.swift.model import HardwareSource
from nion.ui import TestUI
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry
from nion.instrumentation import camera_base
from nion.instrumentation import stem_controller
from nion.instrumentation import scan_base
from nionswift_plugin.nion_instrumentation_ui import ScanAcquisition
from nionswift_plugin.usim import CameraDevice
from nionswift_plugin.usim import InstrumentDevice
from nionswift_plugin.usim import ScanDevice

"""
# running in Swift
import sys, unittest
from superscan import SimulatorScanControl_test
suite = unittest.TestLoader().loadTestsFromTestCase(SimulatorScanControl_test.TestSimulatorScanControlClass)
result = unittest.TextTestResult(sys.stdout, True, True)
suite.run(result)
"""

class TestScanControlClass(unittest.TestCase):

    def setUp(self):
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)
        self.camera_exposure = 0.025
        HardwareSource.HardwareSourceManager().hardware_sources = []
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()

    def tearDown(self):
        HardwareSource.HardwareSourceManager()._close_hardware_sources()
        HardwareSource.HardwareSourceManager()._close_instruments()

    def _setup_hardware(self, is_eels: bool):
        document_model = DocumentModel.DocumentModel()
        document_controller = DocumentController.DocumentController(self.app.ui, document_model, workspace_id="library")
        instrument = self._setup_instrument()
        scan_hardware_source = self._setup_scan_hardware_source(instrument)
        camera_hardware_source = self._setup_camera_hardware_source(instrument, is_eels)
        HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)
        HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)
        return document_controller, document_model, instrument, scan_hardware_source, camera_hardware_source

    def _close_hardware(self, document_controller, instrument, scan_hardware_source, camera_hardware_source):
        camera_hardware_source.close()
        scan_hardware_source.close()
        HardwareSource.HardwareSourceManager().unregister_hardware_source(camera_hardware_source)
        HardwareSource.HardwareSourceManager().unregister_hardware_source(scan_hardware_source)
        self._close_camera_hardware_source()
        self._close_scan_hardware_source()
        self._close_instrument(instrument)
        document_controller.close()

    def _setup_instrument(self):
        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        Registry.register_component(instrument, {"stem_controller"})
        return instrument

    def _close_instrument(self, instrument) -> None:
        HardwareSource.HardwareSourceManager().unregister_instrument("usim_stem_controller")

    def _setup_scan_hardware_source(self, instrument) -> HardwareSource.HardwareSource:
        stem_controller = HardwareSource.HardwareSourceManager().get_instrument_by_id("usim_stem_controller")
        scan_hardware_source = scan_base.ScanHardwareSource(stem_controller, ScanDevice.Device(instrument), "usim_scan_device", "uSim Scan")
        return scan_hardware_source

    def _close_scan_hardware_source(self) -> None:
        pass

    def _setup_camera_hardware_source(self, instrument, is_eels: bool) -> HardwareSource.HardwareSource:
        camera_id = "usim_ronchigram_camera" if not is_eels else "usim_eels_camera"
        camera_type = "ronchigram" if not is_eels else "eels"
        camera_name = "uSim Camera"
        camera_settings = CameraDevice.CameraSettings(camera_id)
        camera_device = CameraDevice.Camera(camera_id, camera_type, camera_name, instrument)
        camera_hardware_source = camera_base.CameraHardwareSource("usim_stem_controller", camera_device, camera_settings, None, None)
        if is_eels:
            camera_hardware_source.features["is_eels_camera"] = True
            camera_hardware_source.add_channel_processor(0, HardwareSource.SumProcessor(((0.25, 0.0), (0.5, 1.0))))
        camera_hardware_source.set_frame_parameters(0, camera_base.CameraFrameParameters({"exposure_ms": self.camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(1, camera_base.CameraFrameParameters({"exposure_ms": self.camera_exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(2, camera_base.CameraFrameParameters({"exposure_ms": self.camera_exposure * 1000 * 2, "binning": 1}))
        camera_hardware_source.set_selected_profile_index(0)
        return camera_hardware_source

    def _close_camera_hardware_source(self) -> None:
        pass

    def _make_acquisition_context(self, is_eels: bool = True):

        class ScanContext:

            def __init__(self, scan_test):
                self.__scan_test = scan_test

            def __enter__(self):
                document_controller, document_model, instrument, scan_hardware_source, camera_hardware_source =\
                    self.__scan_test._setup_hardware(is_eels)

                self.document_controller = document_controller
                self.document_model = document_model
                self.instrument = instrument
                self.scan_hardware_source = scan_hardware_source
                self.camera_hardware_source = camera_hardware_source
                self.scan_context_controller = stem_controller.ScanContextController(self.document_model, self.document_controller.event_loop)

                return self

            def __exit__(self, *exc_details):
                self.scan_context_controller.close()
                self.scan_context_controller = None
                self.__scan_test._close_hardware(self.document_controller, self.instrument, self.scan_hardware_source, self.camera_hardware_source)

            @property
            def objects(self):
                return self.document_controller, self.document_model, self.scan_hardware_source, self.camera_hardware_source

        return ScanContext(self)

    def test_grab_synchronized_basic_eels(self):
        with self._make_acquisition_context() as context:
            document_controller, document_model, scan_hardware_source, camera_hardware_source = context.objects
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters["scan_id"] = str(uuid.uuid4())
            scan_frame_parameters["size"] = (4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters["processing"] = "sum_project"
            camera_data_channel = None
            scans, spectrum_images = scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters, camera=camera_hardware_source, camera_frame_parameters=camera_frame_parameters, camera_data_channel=camera_data_channel)
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
            self.assertIn("scan_detector", spectrum_images[0].metadata)
            self.assertIn("hardware_source", spectrum_images[0].metadata)
            self.assertIn("center_x_nm", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("center_y_nm", spectrum_images[0].metadata["scan_detector"])
            self.assertNotIn("channel_index", spectrum_images[0].metadata["scan_detector"])
            self.assertNotIn("channel_name", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("fov_nm", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("hardware_source_id", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("hardware_source_name", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("rotation", spectrum_images[0].metadata["scan_detector"])
            self.assertEqual(scan_frame_parameters["scan_id"], scans[0].metadata["hardware_source"]["scan_id"])
            self.assertEqual(scan_frame_parameters["scan_id"], spectrum_images[0].metadata["scan_detector"]["scan_id"])

    def test_grab_synchronized_camera_data_channel_basic_use(self):
        with self._make_acquisition_context() as context:
            document_controller, document_model, scan_hardware_source, camera_hardware_source = context.objects
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters["scan_id"] = str(uuid.uuid4())
            scan_frame_parameters["size"] = (4, 4)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters["processing"] = "sum_project"
            grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters)
            camera_data_channel = ScanAcquisition.CameraDataChannel(document_model, "test", grab_sync_info)
            camera_data_channel.start()
            try:
                scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters, camera=camera_hardware_source, camera_frame_parameters=camera_frame_parameters, camera_data_channel=camera_data_channel)
            finally:
                camera_data_channel.stop()

    def test_grab_sync_info_has_proper_calibrations(self):
        with self._make_acquisition_context() as context:
            document_controller, document_model, scan_hardware_source, camera_hardware_source = context.objects
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters["scan_id"] = str(uuid.uuid4())
            scan_frame_parameters["size"] = (8, 8)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters["processing"] = "sum_project"
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
        with self._make_acquisition_context() as context:
            document_controller, document_model, scan_hardware_source, camera_hardware_source = context.objects
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters["scan_id"] = str(uuid.uuid4())
            scan_frame_parameters["size"] = (8, 8)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters["processing"] = "sum_project"
            grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters)
            camera_metadata = grab_sync_info.camera_metadata
            self.assertIn("autostem", camera_metadata)
            self.assertIn("hardware_source_id", camera_metadata)
            self.assertIn("hardware_source_name", camera_metadata)
            self.assertIn("exposure", camera_metadata)
            self.assertIn("binning", camera_metadata)
            self.assertIn("signal_type", camera_metadata)
            # 'is_dark_subtracted': False,
            # 'is_flipped_horizontally': False,
            # 'is_gain_corrected': False,
            # 'sensor_dimensions_hw': (2048, 2048),
            # 'sensor_readout_area_tlbr': (964, 0, 1084, 2048),

    def test_grab_sync_info_has_proper_scan_metadata(self):
        with self._make_acquisition_context() as context:
            document_controller, document_model, scan_hardware_source, camera_hardware_source = context.objects
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters["scan_id"] = str(uuid.uuid4())
            scan_frame_parameters["size"] = (8, 8)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters["processing"] = "sum_project"
            grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters)
            scan_metadata = grab_sync_info.scan_metadata
            self.assertIn("autostem", scan_metadata)
            self.assertIn("hardware_source_id", scan_metadata)
            self.assertIn("hardware_source_name", scan_metadata)
            self.assertIn("center_x_nm", scan_metadata)
            self.assertIn("center_y_nm", scan_metadata)
            self.assertNotIn("channel_index", scan_metadata)
            self.assertNotIn("channel_name", scan_metadata)
            self.assertIn("fov_nm", scan_metadata)
            self.assertIn("rotation", scan_metadata)
            self.assertEqual(scan_frame_parameters["scan_id"], scan_metadata["scan_id"])

    def test_partial_acquisition_has_proper_metadata(self):

        PartialUpdate = collections.namedtuple("PartialUpdate", ["xdata", "state", "scan_shape", "dest_sub_area", "sub_area", "view_id"])

        class CameraDataChannel(ScanAcquisition.CameraDataChannel):
            def __init__(self, document_model, channel_name: str, grab_sync_info: scan_base.ScanHardwareSource.GrabSynchronizedInfo, update_period: float = 1.0):
                super().__init__(document_model, channel_name, grab_sync_info)
                self._update_period = update_period
                self.updates = list()

            def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, scan_shape: Geometry.IntSize, dest_sub_area: Geometry.IntRect, sub_area: Geometry.IntRect, view_id) -> None:
                super().update(data_and_metadata, state, scan_shape, dest_sub_area, sub_area, view_id)
                self.updates.append(PartialUpdate(
                    copy.deepcopy(self.data_item.xdata),
                    state,
                    scan_shape,
                    dest_sub_area,
                    sub_area,
                    view_id
                ))
                # print(f"update {len(self.updates)}")

        with self._make_acquisition_context() as context:
            document_controller, document_model, scan_hardware_source, camera_hardware_source = context.objects
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters["scan_id"] = str(uuid.uuid4())
            scan_frame_parameters["size"] = (6, 6)
            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            camera_frame_parameters["processing"] = "sum_project"
            grab_sync_info = scan_hardware_source.grab_synchronized_get_info(
                scan_frame_parameters=scan_frame_parameters,
                camera=camera_hardware_source,
                camera_frame_parameters=camera_frame_parameters)
            camera_data_channel = CameraDataChannel(document_model, "test", grab_sync_info, update_period=0.0)
            section_height = 5
            scan_hardware_source.grab_synchronized(scan_frame_parameters=scan_frame_parameters,
                                                   camera=camera_hardware_source,
                                                   camera_frame_parameters=camera_frame_parameters,
                                                   camera_data_channel=camera_data_channel,
                                                   section_height=section_height)
            for partial_update in camera_data_channel.updates:
                self.assertEqual("counts", partial_update.xdata.intensity_calibration.units)
                self.assertEqual("nm", partial_update.xdata.dimensional_calibrations[0].units)
                self.assertEqual("nm", partial_update.xdata.dimensional_calibrations[1].units)
                self.assertEqual("eV", partial_update.xdata.dimensional_calibrations[2].units)
                camera_metadata = partial_update.xdata.metadata["hardware_source"]
                self.assertIn("autostem", camera_metadata)
                self.assertIn("hardware_source_id", camera_metadata)
                self.assertIn("hardware_source_name", camera_metadata)
                self.assertIn("exposure", camera_metadata)
                self.assertIn("binning", camera_metadata)
                self.assertIn("signal_type", camera_metadata)
                scan_metadata = partial_update.xdata.metadata["scan_detector"]
                self.assertIn("autostem", scan_metadata)
                self.assertIn("hardware_source_id", scan_metadata)
                self.assertIn("hardware_source_name", scan_metadata)
                self.assertIn("center_x_nm", scan_metadata)
                self.assertIn("center_y_nm", scan_metadata)
                self.assertNotIn("channel_index", scan_metadata)
                self.assertNotIn("channel_name", scan_metadata)
                self.assertIn("fov_nm", scan_metadata)
                self.assertIn("rotation", scan_metadata)
                self.assertIn("scan_id", scan_metadata)

    # TODO: check for counts per electron


if __name__ == '__main__':
    unittest.main()
