import numpy as np
import unittest
import time
import uuid

from nion.instrumentation import camera_base
from nion.swift import Application
from nion.swift.test import TestContext
from nion.instrumentation.test import AcquisitionTestContext
from nion.ui import TestUI
from nion.utils import Geometry

from nion.instrumentation import MultiAcquire


class TestMultiAcquire(unittest.TestCase):

    def setUp(self):
        TestContext.begin_leaks()
        self.app = Application.Application(TestUI.UserInterface(), set_global=False)

    def tearDown(self):
        TestContext.end_leaks(self)

    def __test_context(self, *, is_eels: bool = False) -> AcquisitionTestContext.AcquisitionTestContext:
        return AcquisitionTestContext.test_context(is_eels=is_eels)

    def _set_up_multi_acquire(self, settings: dict, parameters: list, stem_controller):
        multi_acquire = MultiAcquire.MultiAcquireController(stem_controller)
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
        with self.__test_context(is_eels=True) as test_context:
            total_acquisition_time = 0.0
            for parms in parameters:
                # the simulator cant go super fast, so make sure we give it enough time
                total_acquisition_time += parms['frames']*max(parms['exposure_ms'], 100)/1000
                # add some extra overhead time
                total_acquisition_time += 0.15
                total_acquisition_time += settings['x_shift_delay']*2
            total_acquisition_time += settings['x_shift_delay']*2
            total_acquisition_time += settings['blanker_delay']*2 if settings['auto_dark_subtract'] else 0
            stem_controller = test_context.instrument
            camera_hardware_source = test_context.camera_hardware_source
            multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
            multi_acquire.camera = camera_hardware_source
            # enable binning for speed
            frame_parameters = multi_acquire.camera.get_current_frame_parameters()
            frame_parameters.binning = 8
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
            self.assertAlmostEqual(progress, 1.0, places=1)

    def test_acquire_multi_eels_spectrum_applies_shift_for_each_frame(self):
        settings = {'x_shifter': 'C10', 'blanker': 'C_Blank',
                    'x_shift_delay': 0.05, 'focus': '', 'focus_delay': 0, 'auto_dark_subtract': True,
                    'bin_spectra': True, 'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': '',
                    'use_multi_eels_calibration': False, 'shift_each_sequence_slice': True}
        parameters = [{'index': 0, 'offset_x': 1e-7, 'exposure_ms': 5, 'frames': 5},
                      {'index': 1, 'offset_x': -1e-7, 'exposure_ms': 8, 'frames': 3},
                      {'index': 2, 'offset_x': 1e-6, 'exposure_ms': 16, 'frames': 1}]
        with self.__test_context(is_eels=True) as test_context:
            total_acquisition_time = 0.0
            for parms in parameters:
                # the simulator cant go super fast, so make sure we give it enough time
                total_acquisition_time += parms['frames']*max(parms['exposure_ms'], 100)/1000
                # add some extra overhead time
                total_acquisition_time += 0.15
                total_acquisition_time += settings['x_shift_delay']*2
            total_acquisition_time += settings['x_shift_delay']*2
            total_acquisition_time += settings['blanker_delay']*2 if settings['auto_dark_subtract'] else 0
            stem_controller = test_context.instrument
            camera_hardware_source = test_context.camera_hardware_source
            multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
            multi_acquire.camera = camera_hardware_source
            # enable binning for speed
            frame_parameters = multi_acquire.camera.get_current_frame_parameters()
            frame_parameters.binning = 8
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

            self.assertLess(elapsed, total_acquisition_time + 1.0, msg=f'Exceeded allowed acquisition time ({total_acquisition_time} s).')
            self.assertEqual(len(data_dict['data_element_list']), len(parameters))
            self.assertAlmostEqual(progress, 1.0, places=1)
            self.assertAlmostEqual(data_dict['data_element_list'][0]['metadata']['instrument']['defocus'], 9e-7)
            self.assertAlmostEqual(data_dict['data_element_list'][1]['metadata']['instrument']['defocus'], 7e-7)
            self.assertAlmostEqual(data_dict['data_element_list'][2]['metadata']['instrument']['defocus'], 7e-7)
            self.assertAlmostEqual(stem_controller.GetVal('C10'), 5e-7)

    def test_data_intensity_scale_is_correct_for_summed_frames(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'bin_spectra': True,
                    'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': '',
                    'use_multi_eels_calibration': True}
        parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 0, 'exposure_ms': 8, 'frames': 5},
                      {'index': 2, 'offset_x': 0, 'exposure_ms': 16, 'frames': 1}]
        with self.__test_context(is_eels=True) as test_context:
            stem_controller = test_context.instrument
            camera_hardware_source = test_context.camera_hardware_source
            multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
            multi_acquire.camera = camera_hardware_source
            data_dict = multi_acquire.acquire_multi_eels_spectrum()

            calibrated_intensities = []
            for data_element in data_dict['data_element_list']:
                calibrated_intensities.append(np.mean(data_element['data'] * data_element['intensity_calibration']['scale']))

            for val in calibrated_intensities:
                self.assertAlmostEqual(val, calibrated_intensities[0], delta=200)

    def test_data_intensity_scale_is_correct_for_non_summed_frames(self):
        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'bin_spectra': True,
                    'blanker_delay': 0.05, 'sum_frames': False, 'camera_hardware_source_id': '',
                    'use_multi_eels_calibration': True}
        parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 10},
                      {'index': 1, 'offset_x': 0, 'exposure_ms': 8, 'frames': 5},
                      {'index': 2, 'offset_x': 0, 'exposure_ms': 16, 'frames': 1}]
        with self.__test_context(is_eels=True) as test_context:
            stem_controller = test_context.instrument
            camera_hardware_source = test_context.camera_hardware_source
            multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
            multi_acquire.camera = camera_hardware_source
            data_dict = multi_acquire.acquire_multi_eels_spectrum()

            calibrated_intensities = []
            for data_element in data_dict['data_element_list']:
                calibrated_intensities.append(np.mean(data_element['data'] * data_element['intensity_calibration']['scale']))

            for val in calibrated_intensities:
                self.assertAlmostEqual(val, calibrated_intensities[0], delta=200)

    def test_acquire_multi_eels_spectrum_image(self):
        scan_size = Geometry.IntSize(6, 6)
        masks_list = [[], [camera_base.Mask()], [camera_base.Mask(), camera_base.Mask()]]

        for sum_frames in [True, False]:
            for processing in ['sum_project', None, 'sum_masked']:
                for masks in (masks_list if processing == 'sum_masked' else [[]]):
                    with self.subTest(sum_frames=sum_frames, processing=processing, n_masks=len(masks)):
                        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                                    'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'processing': processing,
                                    'blanker_delay': 0.05, 'sum_frames': sum_frames, 'camera_hardware_source_id': ''}
                        parameters = [{'index': 0, 'offset_x': 0, 'exposure_ms': 5, 'frames': 2},
                                      {'index': 1, 'offset_x': 160, 'exposure_ms': 8, 'frames': 1}]
                        with self.__test_context(is_eels=True) as test_context:
                            document_model = test_context.document_model
                            document_controller = test_context.document_controller
                            total_acquisition_time = 0.0
                            for params in parameters:
                                # the simulator cant go super fast, so make sure we give it enough time
                                total_acquisition_time += params['frames']*max(params['exposure_ms'], 100)/1000*scan_size[0]*scan_size[1]
                                # add some extra overhead time
                                total_acquisition_time += 0.15
                                total_acquisition_time += settings['x_shift_delay']*2
                            total_acquisition_time += settings['x_shift_delay']*2

                            stem_controller = test_context.instrument
                            camera_hardware_source = test_context.camera_hardware_source
                            scan_hardware_source = test_context.scan_hardware_source
                            scan_hardware_source.set_enabled_channels([0, 1])
                            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
                            scan_frame_parameters.size = scan_size
                            scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)

                            multi_acquire_controller = self._set_up_multi_acquire(settings, parameters, stem_controller)
                            multi_acquire_controller.scan_controller = scan_hardware_source

                            def get_acquisition_handler_fn(multi_acquire_parameters, current_parameters_index, multi_acquire_settings):
                                camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
                                scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
                                camera_frame_parameters.exposure_ms = multi_acquire_parameters[current_parameters_index]['exposure_ms']
                                camera_frame_parameters.processing = multi_acquire_settings['processing']
                                camera_frame_parameters.active_masks = masks
                                scan_frame_parameters.scan_id = scan_frame_parameters.scan_id or uuid.uuid4()
                                grab_synchronized_info = scan_hardware_source.grab_synchronized_get_info(scan_frame_parameters=scan_frame_parameters,
                                                                                                    camera=camera_hardware_source,
                                                                                                    camera_frame_parameters=camera_frame_parameters)

                                camera_data_channel = MultiAcquire.CameraDataChannel(document_model, camera_hardware_source.display_name, grab_synchronized_info,
                                                                                     multi_acquire_parameters, multi_acquire_settings, current_parameters_index,
                                                                                     stack_metadata_keys=[['hardware_source', 'binning']])
                                enabled_channels = scan_hardware_source.get_enabled_channels()
                                enabled_channel_names = [scan_hardware_source.data_channels[i].name for i in enabled_channels]
                                scan_data_channel = MultiAcquire.ScanDataChannel(document_model, enabled_channel_names, grab_synchronized_info,
                                                                                 multi_acquire_parameters, multi_acquire_settings, current_parameters_index)
                                camera_data_channel.start()
                                scan_data_channel.start()
                                handler =  MultiAcquire.SISequenceAcquisitionHandler(camera_hardware_source, camera_data_channel, camera_frame_parameters,
                                                                                     scan_hardware_source, scan_data_channel, scan_frame_parameters)

                                listener = handler.camera_data_channel.progress_updated_event.listen(multi_acquire_controller.set_progress_counter)

                                def finish_fn():
                                    listener.close()
                                    handler.camera_data_channel.stop()
                                    handler.scan_data_channel.stop()

                                handler.finish_fn = finish_fn

                                return handler

                            progress = 0
                            def update_progress(minimum, maximum, value):
                                nonlocal progress
                                progress = minimum + value/maximum
                                document_controller.periodic()

                            progress_event_listener = multi_acquire_controller.progress_updated_event.listen(update_progress)
                            test_context.push(progress_event_listener)
                            starttime = time.time()
                            multi_acquire_controller.start_multi_acquire_spectrum_image(get_acquisition_handler_fn)
                            endtime = time.time()
                            document_controller.periodic()

                            self.assertAlmostEqual(progress, 1, places=1)

                            multi_acquire_data_items = list()
                            haadf_data_items = list()
                            maadf_data_items = list()
                            for data_item in document_model.data_items:
                                if 'MultiAcquire' in data_item.title:
                                    if 'HAADF' in data_item.title:
                                        haadf_data_items.append(data_item)
                                    elif 'MAADF' in data_item.title:
                                        maadf_data_items.append(data_item)
                                    else:
                                        multi_acquire_data_items.append(data_item)

                            self.assertEqual(len(multi_acquire_data_items), len(parameters))
                            self.assertEqual(len(haadf_data_items), len(parameters))
                            self.assertEqual(len(maadf_data_items), len(parameters))

                            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
                            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()

                            for data_item, haadf_data_item in zip(multi_acquire_data_items, haadf_data_items):
                                camera_dims = camera_hardware_source.get_expected_dimensions(camera_frame_parameters.binning)
                                total_shape = tuple(scan_frame_parameters.size)
                                haadf_shape = tuple(scan_frame_parameters.size)
                                index = data_item.xdata.metadata['MultiAcquire.parameters']['index']
                                if parameters[index]['frames'] > 1 and not settings['sum_frames']:
                                    total_shape = (parameters[index]['frames'],) + total_shape
                                    haadf_shape = (parameters[index]['frames'],) + haadf_shape
                                if settings['processing'] == 'sum_project':
                                    total_shape += camera_dims[1:]
                                elif settings['processing'] == 'sum_masked':
                                    if len(masks) > 1:
                                        if parameters[index]['frames'] > 1 and not settings['sum_frames']:
                                            total_shape = (total_shape[0], len(masks)) + total_shape[1:]
                                        else:
                                            total_shape = (len(masks),) + total_shape
                                else:
                                    total_shape += camera_dims

                                self.assertSequenceEqual(data_item.data.shape, total_shape)
                                self.assertSequenceEqual(haadf_data_item.data.shape, haadf_shape)
                                self.assertEqual(len(data_item.metadata['hardware_source']['binning']), parameters[index]['frames'])

                            self.assertLess(starttime - endtime, total_acquisition_time)

    def test_acquire_multi_eels_spectrum_image_applies_shift_for_each_frame(self):
        scan_size = Geometry.IntSize(6, 6)

        settings = {'x_shifter': 'C10', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                    'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'processing': 'sum_project',
                    'blanker_delay': 0.05, 'sum_frames': False, 'camera_hardware_source_id': '',
                    'shift_each_sequence_slice': True}
        parameters = [{'index': 0, 'offset_x': 1e-7, 'exposure_ms': 5, 'frames': 2},
                      {'index': 1, 'offset_x': -1e-7, 'exposure_ms': 8, 'frames': 3}]

        # This is used in the end to test that the shifts were applied correctly. So if you update the parameters above
        # you also need to update these expected results. The keys in the dict are the respective indices of the
        # parameters dict.
        result_expected_defocus = {0: [5e-7, 6e-7], 1: [6e-7, 5e-7, 4e-7]}

        with self.__test_context(is_eels=True) as test_context:
            document_model = test_context.document_model
            document_controller = test_context.document_controller
            stem_controller = test_context.instrument
            camera_hardware_source = test_context.camera_hardware_source
            scan_hardware_source = test_context.scan_hardware_source
            scan_hardware_source.set_enabled_channels([0, 1])
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
            scan_frame_parameters.size = scan_size
            scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)

            multi_acquire_controller = self._set_up_multi_acquire(settings, parameters, stem_controller)
            multi_acquire_controller.scan_controller = scan_hardware_source

            def get_acquisition_handler_fn(multi_acquire_parameters, current_parameters_index, multi_acquire_settings):
                camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
                scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()
                camera_frame_parameters.exposure_ms = multi_acquire_parameters[current_parameters_index]['exposure_ms']
                camera_frame_parameters.processing = multi_acquire_settings['processing']
                scan_frame_parameters.scan_id = scan_frame_parameters.scan_id or uuid.uuid4()
                grab_synchronized_info = scan_hardware_source.grab_synchronized_get_info(scan_frame_parameters=scan_frame_parameters,
                                                                                    camera=camera_hardware_source,
                                                                                    camera_frame_parameters=camera_frame_parameters)

                camera_data_channel = MultiAcquire.CameraDataChannel(document_model, camera_hardware_source.display_name, grab_synchronized_info,
                                                                     multi_acquire_parameters, multi_acquire_settings, current_parameters_index,
                                                                     stack_metadata_keys=[['hardware_source', 'defocus']])
                enabled_channels = scan_hardware_source.get_enabled_channels()
                enabled_channel_names = [scan_hardware_source.data_channels[i].name for i in enabled_channels]
                scan_data_channel = MultiAcquire.ScanDataChannel(document_model, enabled_channel_names, grab_synchronized_info,
                                                                 multi_acquire_parameters, multi_acquire_settings, current_parameters_index)
                camera_data_channel.start()
                scan_data_channel.start()
                sequence_behavior = MultiAcquire.SequenceBehavior(multi_acquire_controller, current_parameters_index)
                si_sequence_behavior = MultiAcquire.SISequenceBehavior(None, None, sequence_behavior, 1)
                handler =  MultiAcquire.SISequenceAcquisitionHandler(camera_hardware_source, camera_data_channel, camera_frame_parameters,
                                                                     scan_hardware_source, scan_data_channel, scan_frame_parameters,
                                                                     si_sequence_behavior)

                listener = handler.camera_data_channel.progress_updated_event.listen(multi_acquire_controller.set_progress_counter)

                def finish_fn():
                    listener.close()
                    handler.camera_data_channel.stop()
                    handler.scan_data_channel.stop()

                handler.finish_fn = finish_fn

                return handler

            progress = 0
            def update_progress(minimum, maximum, value):
                nonlocal progress
                progress = minimum + value/maximum
                document_controller.periodic()

            progress_event_listener = multi_acquire_controller.progress_updated_event.listen(update_progress)
            test_context.push(progress_event_listener)
            multi_acquire_controller.start_multi_acquire_spectrum_image(get_acquisition_handler_fn)
            document_controller.periodic()

            self.assertAlmostEqual(progress, 1, places=1)

            multi_acquire_data_items = list()
            haadf_data_items = list()
            maadf_data_items = list()
            for data_item in document_model.data_items:
                if 'MultiAcquire' in data_item.title:
                    if 'HAADF' in data_item.title:
                        haadf_data_items.append(data_item)
                    elif 'MAADF' in data_item.title:
                        maadf_data_items.append(data_item)
                    else:
                        multi_acquire_data_items.append(data_item)

            self.assertEqual(len(multi_acquire_data_items), len(parameters))
            self.assertEqual(len(haadf_data_items), len(parameters))
            self.assertEqual(len(maadf_data_items), len(parameters))

            camera_frame_parameters = camera_hardware_source.get_current_frame_parameters()
            scan_frame_parameters = scan_hardware_source.get_current_frame_parameters()

            for data_item, haadf_data_item in zip(multi_acquire_data_items, haadf_data_items):
                camera_dims = camera_hardware_source.get_expected_dimensions(camera_frame_parameters.binning)
                total_shape = tuple(scan_frame_parameters.size)
                haadf_shape = tuple(scan_frame_parameters.size)
                index = data_item.xdata.metadata['MultiAcquire.parameters']['index']
                if parameters[index]['frames'] > 1 and not settings['sum_frames']:
                    total_shape = (parameters[index]['frames'],) + total_shape
                    haadf_shape = (parameters[index]['frames'],) + haadf_shape

                total_shape += camera_dims[1:]

                self.assertSequenceEqual(data_item.data.shape, total_shape)
                self.assertSequenceEqual(haadf_data_item.data.shape, haadf_shape)
                self.assertEqual(len(data_item.metadata['hardware_source']['defocus']), parameters[index]['frames'])
                self.assertSequenceEqual(data_item.metadata['hardware_source']['defocus'], result_expected_defocus[index])


if __name__ == '__main__':
    unittest.main()
