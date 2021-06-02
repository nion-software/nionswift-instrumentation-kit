import contextlib
import copy
import numpy as np
import threading
import unittest
import time
import uuid
import typing

from nion.instrumentation import camera_base, scan_base, stem_controller, Acquisition
from nion.swift.model import HardwareSource, DocumentModel, DataItem
from nion.swift import Facade, Application
from nion.swift.test import TestContext
from nion.utils import Event
from nion.utils import Registry, Geometry
from nionswift_plugin.usim import InstrumentDevice, CameraDevice, ScanDevice
from nion.ui import TestUI

from nion.instrumentation import MultiAcquire


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
        total_acquisition_time = 0.0
        for parms in parameters:
            # the simulator cant go super fast, so make sure we give it enough time
            total_acquisition_time += parms['frames']*max(parms['exposure_ms'], 100)*1e-3
            # add some extra overhead time
            total_acquisition_time += 0.15
            total_acquisition_time += settings['x_shift_delay']*2
        total_acquisition_time += settings['x_shift_delay']*2
        total_acquisition_time += settings['blanker_delay']*2 if settings['auto_dark_subtract'] else 0
        stem_controller, camera = self._get_stem_controller_and_camera(is_eels=True)
        multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
        multi_acquire.camera = camera
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
        self.assertAlmostEqual(progress, 1.0, places=1)

    def test_acquire_multi_eels_spectrum_applies_shift_for_each_frame(self):
        settings = {'x_shifter': 'C10', 'blanker': 'C_Blank',
                    'x_shift_delay': 0.05, 'focus': '', 'focus_delay': 0, 'auto_dark_subtract': True,
                    'bin_spectra': True, 'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': '',
                    'use_multi_eels_calibration': False, 'shift_each_sequence_slice': True}
        parameters = [{'index': 0, 'offset_x': 1e-7, 'exposure_ms': 5, 'frames': 5},
                      {'index': 1, 'offset_x': -1e-7, 'exposure_ms': 8, 'frames': 3},
                      {'index': 2, 'offset_x': 1e-6, 'exposure_ms': 16, 'frames': 1}]
        total_acquisition_time = 0.0
        for parms in parameters:
            # the simulator cant go super fast, so make sure we give it enough time
            total_acquisition_time += parms['frames']*max(parms['exposure_ms'], 100)*1e-3
            # add some extra overhead time
            total_acquisition_time += 0.15
            total_acquisition_time += settings['x_shift_delay']*2
        total_acquisition_time += settings['x_shift_delay']*2
        total_acquisition_time += settings['blanker_delay']*2 if settings['auto_dark_subtract'] else 0
        stem_controller, camera = self._get_stem_controller_and_camera(is_eels=True)
        multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
        multi_acquire.camera = camera
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
        stem_controller, camera = self._get_stem_controller_and_camera(is_eels=True)
        multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
        multi_acquire.camera = camera
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
        stem_controller, camera = self._get_stem_controller_and_camera(is_eels=True)
        multi_acquire = self._set_up_multi_acquire(settings, parameters, stem_controller)
        multi_acquire.camera = camera
        data_dict = multi_acquire.acquire_multi_eels_spectrum()

        calibrated_intensities = []
        for data_element in data_dict['data_element_list']:
            calibrated_intensities.append(np.mean(data_element['data'] * data_element['intensity_calibration']['scale']))

        for val in calibrated_intensities:
            self.assertAlmostEqual(val, calibrated_intensities[0], delta=200)

    def test_acquire_multi_eels_spectrum_image(self):
        scan_size = (10, 10)
        masks_list = [[], [camera_base.Mask()], [camera_base.Mask(), camera_base.Mask()]]

        app = Application.Application(TestUI.UserInterface(), set_global=False)
        for sum_frames in [True, False]:
            for processing in ['sum_project', None, 'sum_masked']:
                for masks in (masks_list if processing == 'sum_masked' else [[]]):
                    with self.subTest(sum_frames=sum_frames, processing=processing, n_masks=len(masks)):
                        settings = {'x_shifter': 'EELS_MagneticShift_Offset', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                                    'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'processing': processing,
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
                                    total_acquisition_time += params['frames']*max(params['exposure_ms'], 100)*1e-3*scan_size[0]*scan_size[1]
                                    # add some extra overhead time
                                    total_acquisition_time += 0.15
                                    total_acquisition_time += settings['x_shift_delay']*2
                                total_acquisition_time += settings['x_shift_delay']*2

                                stem_controller, camera = self._get_stem_controller_and_camera(is_eels=True)
                                scan_controller = self._get_scan_controller(stem_controller)
                                scan_controller.set_enabled_channels([0, 1])
                                scan_frame_parameters = scan_controller.get_current_frame_parameters()
                                scan_frame_parameters['size'] = scan_size
                                scan_controller.set_current_frame_parameters(scan_frame_parameters)

                                multi_acquire_controller = self._set_up_multi_acquire(settings, parameters, stem_controller)
                                multi_acquire_controller.scan_controller = scan_controller

                                def get_acquisition_handler_fn(multi_acquire_parameters, current_parameters_index, multi_acquire_settings):
                                    camera_frame_parameters = camera.get_current_frame_parameters()
                                    scan_frame_parameters = scan_controller.get_current_frame_parameters()
                                    camera_frame_parameters['exposure_ms'] = multi_acquire_parameters[current_parameters_index]['exposure_ms']
                                    camera_frame_parameters['processing'] = multi_acquire_settings['processing']
                                    camera_frame_parameters['active_masks'] = masks
                                    scan_frame_parameters.setdefault('scan_id', str(uuid.uuid4()))
                                    grab_synchronized_info = scan_controller.grab_synchronized_get_info(scan_frame_parameters=scan_frame_parameters,
                                                                                                        camera=camera,
                                                                                                        camera_frame_parameters=camera_frame_parameters)

                                    camera_data_channel = MultiAcquire.CameraDataChannel(document_model, camera.display_name, grab_synchronized_info,
                                                                                         multi_acquire_parameters, multi_acquire_settings, current_parameters_index,
                                                                                         stack_metadata_keys=[['hardware_source', 'binning']])
                                    enabled_channels = scan_controller.get_enabled_channels()
                                    enabled_channel_names = [scan_controller.data_channels[i].name for i in enabled_channels]
                                    scan_data_channel = MultiAcquire.ScanDataChannel(document_model, enabled_channel_names, grab_synchronized_info,
                                                                                     multi_acquire_parameters, multi_acquire_settings, current_parameters_index)
                                    camera_data_channel.start()
                                    scan_data_channel.start()
                                    handler =  MultiAcquire.SISequenceAcquisitionHandler(camera, camera_data_channel, camera_frame_parameters,
                                                                                         scan_controller, scan_data_channel, scan_frame_parameters)

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
                                cm.callback(progress_event_listener.close)
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

                                camera_frame_parameters = camera.get_current_frame_parameters()
                                scan_frame_parameters = scan_controller.get_current_frame_parameters()

                                for data_item, haadf_data_item in zip(multi_acquire_data_items, haadf_data_items):
                                    with self.subTest():
                                        camera_dims = camera.get_expected_dimensions(camera_frame_parameters['binning'])
                                        total_shape = tuple(scan_frame_parameters['size'])
                                        haadf_shape = tuple(scan_frame_parameters['size'])
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
        scan_size = (10, 10)

        app = Application.Application(TestUI.UserInterface(), set_global=False)

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

        with TestContext.create_memory_context() as test_context:
            document_model = test_context.create_document_model(auto_close=False)
            document_controller = app.create_document_controller(document_model, "library")
            with contextlib.ExitStack() as cm:
                cm.callback(document_controller.close)

                stem_controller, camera = self._get_stem_controller_and_camera(is_eels=True)
                scan_controller = self._get_scan_controller(stem_controller)
                scan_controller.set_enabled_channels([0, 1])
                scan_frame_parameters = scan_controller.get_current_frame_parameters()
                scan_frame_parameters['size'] = scan_size
                scan_controller.set_current_frame_parameters(scan_frame_parameters)
                camera_frame_parameters = camera.get_current_frame_parameters()
                camera_frame_parameters['processing'] = settings['processing']

                multi_acquire_controller = self._set_up_multi_acquire(settings, parameters, stem_controller)
                multi_acquire_controller.scan_controller = scan_controller

                def get_acquisition_handler_fn(multi_acquire_parameters, current_parameters_index, multi_acquire_settings):
                    camera_frame_parameters = camera.get_current_frame_parameters()
                    scan_frame_parameters = scan_controller.get_current_frame_parameters()
                    camera_frame_parameters['exposure_ms'] = multi_acquire_parameters[current_parameters_index]['exposure_ms']
                    camera_frame_parameters['processing'] = multi_acquire_settings['processing']
                    scan_frame_parameters.setdefault('scan_id', str(uuid.uuid4()))
                    grab_synchronized_info = scan_controller.grab_synchronized_get_info(scan_frame_parameters=scan_frame_parameters,
                                                                                        camera=camera,
                                                                                        camera_frame_parameters=camera_frame_parameters)

                    camera_data_channel = MultiAcquire.CameraDataChannel(document_model, camera.display_name, grab_synchronized_info,
                                                                         multi_acquire_parameters, multi_acquire_settings, current_parameters_index,
                                                                         stack_metadata_keys=[['hardware_source', 'defocus']])
                    enabled_channels = scan_controller.get_enabled_channels()
                    enabled_channel_names = [scan_controller.data_channels[i].name for i in enabled_channels]
                    scan_data_channel = MultiAcquire.ScanDataChannel(document_model, enabled_channel_names, grab_synchronized_info,
                                                                     multi_acquire_parameters, multi_acquire_settings, current_parameters_index)
                    camera_data_channel.start()
                    scan_data_channel.start()
                    sequence_behavior = MultiAcquire.SequenceBehavior(multi_acquire_controller, current_parameters_index)
                    si_sequence_behavior = MultiAcquire.SISequenceBehavior(None, None, sequence_behavior, 1)
                    handler =  MultiAcquire.SISequenceAcquisitionHandler(camera, camera_data_channel, camera_frame_parameters,
                                                                         scan_controller, scan_data_channel, scan_frame_parameters,
                                                                         si_sequence_behavior)

                    listener = handler.camera_data_channel.progress_updated_event.listen(multi_acquire_controller.set_progress_counter)

                    def finish_fn():
                        listener.close()
                        handler.camera_data_channel.stop()
                        handler.scan_data_channel.stop()

                    handler.finish_fn = finish_fn

                    return handler

                def create_display_data_stream(data_stream: Acquisition.DataStream):
                    display_data_stream = MultiAcquire.DisplayDataStream(data_stream, document_model, document_controller,
                                                                         stack_metadata_keys=[['hardware_source', 'defocus']])
                    def update_progress(*args, **kwargs):
                        if args[0].channel == 999:
                            p = display_data_stream.progress
                            print(f'{p=}')
                            multi_acquire_controller.set_progress_counter(p[0], maximum=p[1])
                    listener = display_data_stream.data_available_event.listen(update_progress)
                    display_data_stream.finishes.append(lambda: listener.close())
                    camera_data_item = DataItem.DataItem(large_format=True)
                    document_controller.queue_task(lambda: document_model.append_data_item(camera_data_item))
                    scan_data_items: typing.List[DataItem.DataItem] = list()
                    def queue_append_di(di):
                        document_controller.queue_task(lambda: document_model.append_data_item(di))
                    for channel in data_stream.channels:
                        if channel != 999:
                            di = DataItem.DataItem(large_format=True)
                            scan_data_items.append(di)
                            queue_append_di(di)
                    display_data_stream.scan_data_items = scan_data_items
                    display_data_stream.camera_data_item = camera_data_item
                    return display_data_stream

                progress = 0
                def update_progress(minimum, maximum, value):
                    nonlocal progress
                    print(f'{minimum=}, {maximum=}, {value=}')
                    progress = minimum + value/maximum
                    document_controller.periodic()

                progress_event_listener = multi_acquire_controller.progress_updated_event.listen(update_progress)
                cm.callback(progress_event_listener.close)
                multi_acquire_controller.start_multi_acquire_spectrum_image(scan_controller, scan_frame_parameters,
                                                                            camera, camera_frame_parameters,
                                                                            create_display_data_stream)
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

                camera_frame_parameters = camera.get_current_frame_parameters()
                scan_frame_parameters = scan_controller.get_current_frame_parameters()

                for data_item, haadf_data_item in zip(multi_acquire_data_items, haadf_data_items):
                    with self.subTest():
                        camera_dims = camera.get_expected_dimensions(camera_frame_parameters['binning'])
                        total_shape = tuple(scan_frame_parameters['size'])
                        haadf_shape = tuple(scan_frame_parameters['size'])
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
