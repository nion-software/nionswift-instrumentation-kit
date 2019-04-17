import unittest
import threading
import copy
import numpy as np

from nion.instrumentation import camera_base
from nion.swift.model import HardwareSource
from nion.utils import Event
from nion.utils import Registry
from nionswift_plugin.usim import CameraDevice
from nionswift_plugin.usim import InstrumentDevice

from nion.instrumentation import MultiAcquire


class TestMultiAcquire(unittest.TestCase):

    def setUp(self):
        #self.app = Application.Application(TestUI.UserInterface(), set_global=False)
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
        camera_hardware_source.set_frame_parameters(0, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(1, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000, "binning": 2}))
        camera_hardware_source.set_frame_parameters(2, camera_base.CameraFrameParameters({"exposure_ms": self.exposure * 1000 * 2, "binning": 1}))
        camera_hardware_source.set_selected_profile_index(0)

        HardwareSource.HardwareSourceManager().register_hardware_source(camera_hardware_source)

        return instrument, camera_hardware_source

    def _set_up_multi_acquire(self, settings: dict, parameters: list):
        multi_acquire = MultiAcquire.MultiAcquireController()
        multi_acquire._multi_acquire_controller__savepath = None
        multi_acquire.settings.update(settings)
        multi_acquire.spectrum_parameters[:] = parameters
        return multi_acquire

    def test_acquire_multi_eels_spectrum_works_and_finishes_in_time(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'y_shifter': '', 'x_units_per_ev': 1,
                    'y_units_per_px': 0.00081, 'blanker': 'C_Blank', 'x_shift_delay': 0.05, 'y_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'saturation_value': 12000, 'y_align': True,
                    'stitch_spectra': False, 'auto_dark_subtract': True, 'bin_spectra': True,
                    'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': ''}
        parameters = [{'index': 0, 'offset_x': 0, 'offset_y': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 160, 'offset_y': 0, 'exposure_ms': 8, 'frames': 1},
                      {'index': 2, 'offset_x': 320, 'offset_y': 0, 'exposure_ms': 16, 'frames': 1}]
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
        def run_and_check_result():
            data_dict = multi_acquire.acquire_multi_eels_spectrum()
            self.assertEqual(len(data_dict['data_element_list']), len(parameters))

        thread = threading.Thread(target=run_and_check_result, daemon=True)
        def react_to_event(message):
            if message['message'] == 'end' and message['description'] == 'single spectrum':
                event.set()
        acquisition_event_listener = multi_acquire.acquisition_state_changed_event.listen(react_to_event)
        progress = 0
        def update_progress(minimum, maximum, value):
            nonlocal progress
            progress = minimum + value/maximum
        progress_event_listener = multi_acquire.progress_updated_event.listen(update_progress)
        thread.start()
        self.assertTrue(event.wait(total_acquisition_time), msg=f'Exceeded allowed acquisition time ({total_acquisition_time} s).')
        self.assertAlmostEqual(progress, 1.0)
        del acquisition_event_listener, progress_event_listener

    def test_data_intensity_scale_is_correct_for_summed_frames(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'y_shifter': '', 'x_units_per_ev': 1,
                    'y_units_per_px': 0.00081, 'blanker': 'C_Blank', 'x_shift_delay': 0.05, 'y_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'saturation_value': 12000, 'y_align': True,
                    'stitch_spectra': False, 'auto_dark_subtract': False, 'bin_spectra': True,
                    'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': ''}
        parameters = [{'index': 0, 'offset_x': 0, 'offset_y': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 160, 'offset_y': 0, 'exposure_ms': 8, 'frames': 5},
                      {'index': 2, 'offset_x': 320, 'offset_y': 0, 'exposure_ms': 16, 'frames': 1}]
        multi_acquire = self._set_up_multi_acquire(settings, parameters)
        multi_acquire._MultiAcquireController__active_settings = copy.deepcopy(multi_acquire.settings)
        multi_acquire._MultiAcquireController__active_spectrum_parameters = copy.deepcopy(multi_acquire.spectrum_parameters)
        for parms in parameters:
            data = np.ones((parms['frames'], 10))*parms['exposure_ms']
            data_element = {'data': data,
                            'spatial_calibrations': [{'offset': 0, 'scale': 1, 'units': ''},
                                                     {'offset': parms['offset_x'], 'scale': 0.123, 'units': 'ev'}],
                            'intensity_calibration': {'offset': 0, 'scale': 0.9, 'units': 'counts'},
                            'properties': {'exposure': parms['exposure_ms'], 'counts_per_electron': 37}}
            data_dict = {'data_element': data_element,
                         'parameters': {'line_number': 0, 'flyback_pixels': 0, 'frames': parms['frames']}}
            multi_acquire._MultiAcquireController__queue.put(data_dict)
        multi_acquire._MultiAcquireController__acquisition_finished_event.set()
        processed_data = []
        def react_to_event(data_dict):
            processed_data.append(np.mean(data_dict['data_element']['data']*
                                          data_dict['data_element']['intensity_calibration']['scale']))
        event_listener = multi_acquire.new_data_ready_event.listen(react_to_event)
        multi_acquire.process_and_send_data()
        for val in processed_data:
            self.assertAlmostEqual(val, processed_data[0])
        del event_listener

    def test_data_intensity_scale_is_correct_for_non_summed_frames(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'y_shifter': '', 'x_units_per_ev': 1,
                    'y_units_per_px': 0.00081, 'blanker': 'C_Blank', 'x_shift_delay': 0.05, 'y_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'saturation_value': 12000, 'y_align': True,
                    'stitch_spectra': False, 'auto_dark_subtract': False, 'bin_spectra': True,
                    'blanker_delay': 0.05, 'sum_frames': False, 'camera_hardware_source_id': ''}
        parameters = [{'index': 0, 'offset_x': 0, 'offset_y': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 160, 'offset_y': 0, 'exposure_ms': 8, 'frames': 5},
                      {'index': 2, 'offset_x': 320, 'offset_y': 0, 'exposure_ms': 16, 'frames': 1}]
        multi_acquire = self._set_up_multi_acquire(settings, parameters)
        multi_acquire._MultiAcquireController__active_settings = copy.deepcopy(multi_acquire.settings)
        multi_acquire._MultiAcquireController__active_spectrum_parameters = copy.deepcopy(multi_acquire.spectrum_parameters)
        for parms in parameters:
            data = np.ones((parms['frames'], 10))*parms['exposure_ms']
            data_element = {'data': data,
                            'spatial_calibrations': [{'offset': 0, 'scale': 1, 'units': ''},
                                                     {'offset': parms['offset_x'], 'scale': 0.123, 'units': 'ev'}],
                            'intensity_calibration': {'offset': 0, 'scale': 0.9, 'units': 'counts'},
                            'properties': {'exposure': parms['exposure_ms'], 'counts_per_electron': 37}}
            data_dict = {'data_element': data_element,
                         'parameters': {'line_number': 0, 'flyback_pixels': 0, 'frames': parms['frames']}}
            multi_acquire._MultiAcquireController__queue.put(data_dict)
        multi_acquire._MultiAcquireController__acquisition_finished_event.set()
        processed_data = []
        def react_to_event(data_dict):
            processed_data.append(np.mean(data_dict['data_element']['data']*
                                          data_dict['data_element']['intensity_calibration']['scale']))
        event_listener = multi_acquire.new_data_ready_event.listen(react_to_event)
        multi_acquire.process_and_send_data()
        for val in processed_data:
            self.assertAlmostEqual(val, processed_data[0])
        del event_listener

if __name__ == '__main__':
    unittest.main()
