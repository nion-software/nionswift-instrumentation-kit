import contextlib
import copy
import numpy as np
import threading
import unittest
import time

from nion.instrumentation import camera_base, scan_base, stem_controller
from nion.swift.model import HardwareSource, DocumentModel
from nion.swift import Facade, Application
from nion.swift.test import TestContext
from nion.utils import Event
from nion.utils import Registry, Geometry
from nionswift_plugin.usim import InstrumentDevice, CameraDevice, ScanDevice
from nion.ui import TestUI

from nion.instrumentation import MultiAcquire
from nionswift_plugin.nion_instrumentation_ui import MultiAcquirePanel


class TestMultiAcquire(unittest.TestCase):

    def setUp(self):
        HardwareSource.HardwareSourceManager().hardware_sources = []
        HardwareSource.HardwareSourceManager().hardware_source_added_event = Event.Event()
        HardwareSource.HardwareSourceManager().hardware_source_removed_event = Event.Event()

    def tearDown(self):
        HardwareSource.HardwareSourceManager()._close_hardware_sources()
        HardwareSource.HardwareSourceManager()._close_instruments()

    def _get_stem_controller_and_camera(self, initialize: bool=True, is_eels: bool=False):
        # this is simulator specific. replace this code but be sure to set up self.exposure and blanked and positioned
        # initial settings.
        self.exposure = 0.04

        instrument = InstrumentDevice.Instrument("usim_stem_controller")
        Registry.register_component(instrument, {"stem_controller"})

        camera_id = "usim_ronchigram_camera" if not is_eels else "usim_eels_camera"
        camera_type = "ronchigram" if not is_eels else "eels"
        camera_name = "uSim Camera"
        camera_settings = CameraDevice.CameraSettings(camera_id)
        camera_device = CameraDevice.Camera(camera_id, camera_type, camera_name, instrument)
        camera_hardware_source = camera_base.CameraHardwareSource("usim_stem_controller", camera_device, camera_settings, None, None)
        if is_eels:
            camera_hardware_source.features["is_eels_camera"] = True
            camera_hardware_source.add_channel_processor(0, HardwareSource.SumProcessor(((0.25, 0.0), (0.5, 1.0))))
            # EELS camera only produces data if a valid scan context is presend in intrument controller, so set one up here
            scan_context = stem_controller.ScanContext()
            scan_context.update(Geometry.IntSize(128, 128), Geometry.FloatPoint(), 10.0, 0.0)
            # This is the only method that allows access to the scan context
            instrument._set_scan_context_probe_position(scan_context, Geometry.FloatPoint(0.5, 0.5))
        camera_hardware_source.set_frame_parameters(0, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(1, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(2, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000 * 2, "binning": 1}))
        camera_hardware_source.set_selected_profile_index(0)

        HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)

        return instrument, camera_hardware_source

    def _get_scan_controller(self, stem_controller: InstrumentDevice):
        scan_hardware_source = scan_base.ScanHardwareSource(stem_controller, ScanDevice.Device(stem_controller), "usim_scan_device", "uSim Scan")
        HardwareSource.HardwareSourceManager().register_hardware_source(scan_hardware_source)
        return scan_hardware_source

    def _set_up_multi_acquire(self, settings: dict, parameters: list, multi_acquire_instance=None,
                              document_controller=None, document_model=None):
        multi_acquire = multi_acquire_instance or MultiAcquire.MultiAcquireController(document_model, document_controller)
        multi_acquire._MultiAcquireController__savepath = None
        multi_acquire.settings.update(settings)
        multi_acquire.spectrum_parameters[:] = parameters
        return multi_acquire

    def test_acquire_multi_eels_spectrum_works_and_finishes_in_time(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank',
                    'x_shift_delay': 0.05, 'focus': '', 'focus_delay': 0, 'auto_dark_subtract': True,
                    'bin_spectra': True, 'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': '',
                    'use_multi_eels_calibration': False}
        parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 160, 'exposure_ms': 8, 'frames': 1},
                      {'index': 2, 'offset_x': 320, 'exposure_ms': 16, 'frames': 1}]
        total_acquisition_time = 0.0
        for parms in parameters:
            # the simulator cant go super fast, so make sure we give it enough time
            total_acquisition_time += parms['frames']*max(parms['exposure_ms'], 100)*1e-3
            # add some extra overhead time
            total_acquisition_time += 0.15
            total_acquisition_time += settings['x_shift_delay']*2
        total_acquisition_time += settings['x_shift_delay']*2
        total_acquisition_time += settings['blanker_delay']*2 if settings['auto_dark_subtract'] else 0
        multi_acquire = self._set_up_multi_acquire(settings, parameters)
        multi_acquire.stem_controller, multi_acquire.camera = self._get_stem_controller_and_camera(is_eels=True)
        # enable binning for speed
        frame_parameters = multi_acquire.camera.get_current_frame_parameters()
        frame_parameters['binning'] = 8
        multi_acquire.camera.set_current_frame_parameters(frame_parameters)
        progress = 0
        def update_progress(minimum, maximum, value):
            nonlocal progress
            progress = minimum + value/maximum
        progress_event_listener = multi_acquire.progress_updated_event.listen(update_progress)
        t0 = time.time()
        data_dict = multi_acquire.acquire_multi_eels_spectrum()
        elapsed = time.time() - t0
        progress_event_listener.close()
        self.assertLess(elapsed, total_acquisition_time, msg=f'Exceeded allowed acquisition time ({total_acquisition_time} s).')
        self.assertEqual(len(data_dict['data_element_list']), len(parameters))
        self.assertAlmostEqual(progress, 1.0)

    def test_data_intensity_scale_is_correct_for_summed_frames(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'bin_spectra': True,
                    'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': '',
                    'use_multi_eels_calibration': True}
        parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 0, 'exposure_ms': 8, 'frames': 5},
                      {'index': 2, 'offset_x': 0, 'exposure_ms': 16, 'frames': 1}]
        multi_acquire = self._set_up_multi_acquire(settings, parameters)
        multi_acquire.stem_controller, multi_acquire.camera = self._get_stem_controller_and_camera(is_eels=True)
        data_dict = multi_acquire.acquire_multi_eels_spectrum()

        calibrated_intensities = []
        for data_element in data_dict['data_element_list']:
            calibrated_intensities.append(np.mean(data_element['data'] * data_element['intensity_calibration']['scale']))

        for val in calibrated_intensities:
            self.assertAlmostEqual(val, calibrated_intensities[0], delta=100)

    def test_data_intensity_scale_is_correct_for_non_summed_frames(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'bin_spectra': True,
                    'blanker_delay': 0.05, 'sum_frames': False, 'camera_hardware_source_id': '',
                    'use_multi_eels_calibration': True}
        parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 0, 'exposure_ms': 8, 'frames': 5},
                      {'index': 2, 'offset_x': 0, 'exposure_ms': 16, 'frames': 1}]
        multi_acquire = self._set_up_multi_acquire(settings, parameters)
        multi_acquire.stem_controller, multi_acquire.camera = self._get_stem_controller_and_camera(is_eels=True)
        data_dict = multi_acquire.acquire_multi_eels_spectrum()

        calibrated_intensities = []
        for data_element in data_dict['data_element_list']:
            calibrated_intensities.append(np.mean(data_element['data'] * data_element['intensity_calibration']['scale']))

        for val in calibrated_intensities:
            self.assertAlmostEqual(val, calibrated_intensities[0], delta=100)

    @unittest.skip('Not updated to new version yet')
    def test_acquire_multi_eels_spectrum_produces_data_with_correct_number_of_dimensional_calibrations(self):
        for sum_frames in [True, False]:
            with self.subTest(sum_frames=sum_frames):
                settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                            'focus': '', 'focus_delay': 0, 'auto_dark_subtract': True, 'bin_spectra': True,
                            'blanker_delay': 0.05, 'sum_frames': sum_frames, 'camera_hardware_source_id': ''}
                parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 10},
                              {'index': 1, 'offset_x': 160, 'exposure_ms': 8, 'frames': 1},
                              {'index': 2, 'offset_x': 320, 'exposure_ms': 16, 'frames': 1}]
                total_acquisition_time = 0.0
                for parms in parameters:
                    # the simulator cant go super fast, so make sure we give it enough time
                    total_acquisition_time += parms['frames']*max(parms['exposure_ms'], 100)*1e-3
                    # add some extra overhead time
                    total_acquisition_time += 0.15
                    total_acquisition_time += settings['x_shift_delay']*2
                total_acquisition_time += settings['x_shift_delay']*2
                total_acquisition_time += settings['blanker_delay']*2 if settings['auto_dark_subtract'] else 0
                multi_acquire = self._set_up_multi_acquire(settings, parameters)
                multi_acquire.stem_controller, multi_acquire.camera = self._get_stem_controller_and_camera()
                # enable binning for speed
                frame_parameters = multi_acquire.camera.get_current_frame_parameters()
                frame_parameters['binning'] = 8
                multi_acquire.camera.set_current_frame_parameters(frame_parameters)
                event = threading.Event()
                data_dict = None
                def run():
                    nonlocal data_dict
                    data_dict = multi_acquire.acquire_multi_eels_spectrum()
                    event.set()

                thread = threading.Thread(target=run, daemon=True)
                thread.start()
                self.assertTrue(event.wait(total_acquisition_time), msg=f'Exceeded allowed acquisition time ({total_acquisition_time} s).')
                self.assertEqual(len(data_dict['data_element_list']), len(parameters))
                for i, data_element in enumerate(data_dict['data_element_list']):
                    with self.subTest(parameters=parameters[i]):
                        self.assertEqual(len(data_element['spatial_calibrations']), len(data_element['data'].shape))

    def test_acquire_multi_eels_spectrum_image_produces_data_of_correct_shape(self):
        app = Application.Application(TestUI.UserInterface(), set_global=False)
        for sum_frames in [True, False]:
            for bin_spectra in [True, False]:
                with self.subTest(sum_frames=sum_frames, bin_spectra=bin_spectra):
                    settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                                'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'bin_spectra': bin_spectra,
                                'blanker_delay': 0.05, 'sum_frames': sum_frames, 'camera_hardware_source_id': ''}
                    parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 2},
                                  {'index': 1, 'offset_x': 160, 'exposure_ms': 8, 'frames': 1}]
                    with TestContext.create_memory_context() as test_context:
                        document_model = test_context.create_document_model(auto_close=False)
                        document_controller = app.create_document_controller(document_model, "library")
                        with contextlib.ExitStack() as cm:
                            cm.callback(document_controller.close)
                            total_acquisition_time = 0.0
                            for params in parameters:
                                # the simulator cant go super fast, so make sure we give it enough time
                                total_acquisition_time += params['frames']*max(params['exposure_ms'], 100)*1e-3
                                # add some extra overhead time
                                total_acquisition_time += 0.15
                                total_acquisition_time += settings['x_shift_delay']*2
                            total_acquisition_time += settings['x_shift_delay']*2
                            # api = Facade.get_api('~1.0', '~1.0')
                            # si_receiver = MultiAcquirePanel.MultiAcquirePanelDelegate(api)
                            # si_receiver._close_listeners_for_test()
                            # cm.callback(si_receiver.close)
                            multi_acquire = self._set_up_multi_acquire(settings, parameters,
                                                                       document_controller=Facade.DocumentWindow(document_controller),
                                                                       document_model=document_model)
                            multi_acquire.stem_controller, multi_acquire.camera = self._get_stem_controller_and_camera(is_eels=True)
                            multi_acquire.scan_controller = self._get_scan_controller(multi_acquire.stem_controller)
                            # enable binning for speed
                            frame_parameters = multi_acquire.camera.get_current_frame_parameters()
                            frame_parameters['binning'] = 8
                            multi_acquire.camera.set_current_frame_parameters(frame_parameters)
                            scan_frame_parameters = multi_acquire.scan_controller.get_current_frame_parameters()
                            scan_frame_parameters['size'] = (10, 10)
                            scan_frame_parameters['fov_size_nm'] = 16
                            multi_acquire.scan_controller.set_current_frame_parameters(scan_frame_parameters)
                            progress = 0
                            def update_progress(minimum, maximum, value):
                                nonlocal progress
                                progress = minimum + value/maximum
                                document_controller.periodic()
                            progress_event_listener = multi_acquire.progress_updated_event.listen(update_progress)
                            cm.callback(progress_event_listener.close)
                            # def acquisition_state_changed(info_dict):
                            #     if info_dict.get('message') in {'end processing', 'exception'}:
                            #         si_receiver._data_processed_event.set()
                            # acquisition_state_changed_event_listener = multi_acquire.acquisition_state_changed_event.listen(acquisition_state_changed)
                            # cm.callback(acquisition_state_changed_event_listener.close)
                            # si_receiver._start_display_queue_thread()
                            # starttime = time.time()
                            multi_acquire.camera.start_playing()
                            multi_acquire.acquire_multi_eels_spectrum_image()
                            document_controller.periodic()
                            # self.assertTrue(si_receiver._data_processed_event.wait(10))
                            self.assertGreaterEqual(progress, 1)
                            #self.assertLess(time.time() - starttime, total_acquisition_time)
                            multi_acquire_data_items = list()
                            for data_item in document_model.data_items:
                                if 'MultiAcquire' in data_item.title:
                                    multi_acquire_data_items.append(data_item)
                            self.assertEqual(len(multi_acquire_data_items), len(parameters))
                            for data_item in multi_acquire_data_items:
                                with self.subTest():
                                    camera_dims = multi_acquire.camera.get_expected_dimensions(frame_parameters['binning'])
                                    total_shape = tuple(scan_frame_parameters['size'])
                                    haadf_shape = tuple(scan_frame_parameters['size'])
                                    index = data_item.xdata.metadata['MultiAcquire.parameters']['index']
                                    if parameters[index]['frames'] > 1 and not settings['sum_frames']:
                                        total_shape = (parameters[index]['frames'],) + total_shape
                                        haadf_shape = (parameters[index]['frames'],) + haadf_shape
                                    if settings['bin_spectra']:
                                        total_shape += camera_dims[1:]
                                    else:
                                        total_shape += camera_dims

# TODO: Setup receiver for HAADF data item and check their shape, too

                                    self.assertSequenceEqual(data_item.data.shape, total_shape)

if __name__ == '__main__':
    unittest.main()
