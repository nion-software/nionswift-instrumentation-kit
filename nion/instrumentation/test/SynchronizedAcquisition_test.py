import numpy
import unittest

from nion.data import DataAndMetadata
from nion.swift import Application
from nion.swift import DocumentController
from nion.swift.model import DocumentModel
from nion.swift.model import HardwareSource
from nion.ui import TestUI
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry
from nion.instrumentation import camera_base
from nion.instrumentation import stem_controller
from nion.instrumentation import scan_base
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
                self.probe_view_controller = stem_controller.ProbeViewController(self.document_model, self.document_controller.event_loop)

                return self

            def __exit__(self, *exc_details):
                self.probe_view_controller.close()
                self.probe_view_controller = None
                self.__scan_test._close_hardware(self.document_controller, self.instrument, self.scan_hardware_source, self.camera_hardware_source)

            @property
            def objects(self):
                return self.document_controller, self.document_model, self.scan_hardware_source, self.camera_hardware_source

        return ScanContext(self)

    def test_grab_synchronized_basic_eels(self):
        with self._make_acquisition_context() as context:
            document_controller, document_model, scan_hardware_source, camera_hardware_source = context.objects
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters["size"] = (16, 16)
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
            self.assertIn("exposure", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("fov_nm", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("hardware_source_id", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("hardware_source_name", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("line_time_us", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("pixel_time_us", spectrum_images[0].metadata["scan_detector"])
            self.assertIn("rotation", spectrum_images[0].metadata["scan_detector"])
            self.assertEqual(scans[0].metadata["hardware_source"]["scan_id"], spectrum_images[0].metadata["scan_detector"]["scan_id"])


if __name__ == '__main__':
    unittest.main()
