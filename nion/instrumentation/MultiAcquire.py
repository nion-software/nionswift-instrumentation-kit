# standard libraries
import copy
import json
import logging
import numpy
import os
import queue
import threading
import time

# local libraries
from nion.utils import Event, Geometry
from nion.data import DataAndMetadata, Calibration
from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller


class MultiEELSSettings(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settings_changed_event = Event.Event()

    def __setitem__(self, key, value):
        old_value = self.__getitem__(key)
        super().__setitem__(key, value)
        if value != old_value:
            self.settings_changed_event.fire()

    def __copy__(self):
        return MultiEELSSettings(super().copy())

    def __deepcopy__(self, memo):
        return MultiEELSSettings(copy.deepcopy(super().copy()))

    def copy(self):
        return self.__copy__()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self.settings_changed_event.fire()


class MultiEELSParameters(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters_changed_event = Event.Event()

    def __setitem__(self, index, value):
        old_value = self.__getitem__(index)
        super().__setitem__(index, value)
        if old_value != value:
            self.parameters_changed_event.fire()

    def append(self, value):
        super().append(value)
        self.parameters_changed_event.fire()

    def pop(self, index=-1):
        super().pop(index)
        self.parameters_changed_event.fire()

    def __copy__(self):
        return MultiEELSParameters(super().copy())

    def __deepcopy__(self, memo):
        return MultiEELSParameters(copy.deepcopy(super().copy()))

    def copy(self):
        return self.__copy__()


class CameraDataChannel(scan_base.SynchronizedDataChannelInterface):
    def __init__(self):
        self.new_data_ready_event = Event.Event()
        self.get_parameters_fn = None
        self.get_settings_fn = None

    def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, data_shape: Geometry.IntSize,
               dest_sub_area: Geometry.IntRect, sub_area: Geometry.IntRect, view_id):
        if callable(self.get_parameters_fn):
            parameters = self.get_parameters_fn()
        else:
            parameters = dict()
        if callable(self.get_settings_fn):
            settings = self.get_settings_fn()
        else:
            settings = dict()
        data_dict = dict()
        start_ev = data_and_metadata.dimensional_calibrations[-1].offset
        end_ev = start_ev + data_and_metadata.dimensional_calibrations[-1].scale * data_and_metadata.data_shape[-1]
        parameters['start_ev'] = start_ev
        parameters['end_ev'] = end_ev
        if parameters.get('frames', 1) > 1 and not settings.get('sum_frames'):
            data = data_and_metadata.data[numpy.newaxis, ...]
            dimensional_calibrations = [Calibration.Calibration()] + list(data_and_metadata.dimensional_calibrations)
            data_descriptor = DataAndMetadata.DataDescriptor(True,
                                                             data_and_metadata.data_descriptor.collection_dimension_count,
                                                             data_and_metadata.data_descriptor.datum_dimension_count)
            data_and_metadata = DataAndMetadata.new_data_and_metadata(data,
                                                                      intensity_calibration=data_and_metadata.intensity_calibration,
                                                                      dimensional_calibrations=dimensional_calibrations,
                                                                      metadata=data_and_metadata.metadata,
                                                                      timestamp=data_and_metadata.timestamp,
                                                                      data_descriptor=data_descriptor,
                                                                      timezone=data_and_metadata.timezone,
                                                                      timezone_offset=data_and_metadata.timezone_offset)

        # start_ev = data_element.get('spatial_calibrations', [{}])[-1].get('offset', 0)
        # end_ev = start_ev + (data_element.get('spatial_calibrations', [{}])[-1].get('scale', 0) *
        #                      data_element.get('data').shape[-1])
        data_dict['parameters'] = parameters
        data_dict['settings'] = settings
        data_dict['xdata'] = data_and_metadata
        data_dict['state'] = state
        data_dict['dest_sub_area'] = dest_sub_area
        data_dict['sub_area'] = sub_area
        data_dict['view_id'] = view_id
        self.new_data_ready_event.fire(data_dict)


class MultiAcquireController:
    def __init__(self, **kwargs):
        self.spectrum_parameters = MultiEELSParameters(
                                   [{'index': 0, 'offset_x': 0, 'exposure_ms': 1, 'frames': 1},
                                    {'index': 1, 'offset_x': 160, 'exposure_ms': 8, 'frames': 1},
                                    {'index': 2, 'offset_x': 320, 'exposure_ms': 16, 'frames': 1}])
        self.settings = MultiEELSSettings(
                        {'x_shifter': 'LossMagnetic', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                         'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'bin_spectra': True,
                         'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': ''})
        self.stem_controller: stem_controller.STEMController = None
        self.camera: camera_base.CameraHardwareSource = None
        self.scan_controller: scan_base.ScanHardwareSource = None
        self.zeros = {'x': 0, 'focus': 0}
        self.scan_calibrations = [{'offset': 0, 'scale': 1, 'units': ''}, {'offset': 0, 'scale': 1, 'units': ''}]
        self.__progress_counter = 0
        self.__flyback_pixels = 0
        self.acquisition_state_changed_event = Event.Event()
        self.new_data_ready_event = Event.Event()
        self.progress_updated_event = Event.Event()
        self.__stop_processing_event = threading.Event()
        self.__queue = queue.Queue()
        self.__acquisition_finished_event = threading.Event()
        self.__process_and_send_data_thread = None
        self.__active_settings = self.settings
        self.__active_spectrum_parameters = self.spectrum_parameters
        self.abort_event = threading.Event()
        self.__savepath = os.path.join(os.path.expanduser('~'), 'MultiAcquire')
        self.load_settings()
        self.load_parameters()
        self.__settings_changed_event_listener = self.settings.settings_changed_event.listen(self.save_settings)
        self.__spectrum_parameters_changed_event_listener = self.spectrum_parameters.parameters_changed_event.listen(self.save_parameters)

    def save_settings(self):
        if self.__savepath:
            os.makedirs(self.__savepath, exist_ok=True)
            with open(os.path.join(self.__savepath, 'settings.json'), 'w+') as f:
                json.dump(self.settings, f)

    def load_settings(self):
        if os.path.isfile(os.path.join(self.__savepath, 'settings.json')):
            with open(os.path.join(self.__savepath, 'settings.json')) as f:
                self.settings.update(json.load(f))

    def save_parameters(self):
        if self.__savepath:
            os.makedirs(self.__savepath, exist_ok=True)
            with open(os.path.join(self.__savepath, 'spectrum_parameters.json'), 'w+') as f:
                json.dump(self.spectrum_parameters, f)

    def load_parameters(self):
        if os.path.isfile(os.path.join(self.__savepath, 'spectrum_parameters.json')):
            with open(os.path.join(self.__savepath, 'spectrum_parameters.json')) as f:
                self.spectrum_parameters[:] = json.load(f)

    def add_spectrum(self, parameters=None):
        if parameters is None:
            parameters = self.spectrum_parameters[-1].copy()
        parameters['index'] = len(self.spectrum_parameters)
        self.spectrum_parameters.append(parameters)

    def remove_spectrum(self):
        assert len(self.spectrum_parameters) > 1, 'Number of spectra cannot become smaller than 1.'
        self.spectrum_parameters.pop()

    def get_offset_x(self, index):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined!'.format(index)
        return self.spectrum_parameters[index]['offset_x']

    def set_offset_x(self, index, offset_x):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined. Add a new spectrum before changing its parameters!'.format(index)
        parameters = self.spectrum_parameters[index].copy()
        if offset_x != parameters.get('offset_x'):
            parameters['offset_x'] = offset_x
            self.spectrum_parameters[index] = parameters

    def get_exposure_ms(self, index):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined!'.format(index)
        return self.spectrum_parameters[index]['exposure_ms']

    def set_exposure_ms(self, index, exposure_ms):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined. Add a new spectrum before changing its parameters!'.format(index)
        parameters = self.spectrum_parameters[index].copy()
        if exposure_ms != parameters.get('exposure_ms'):
            parameters['exposure_ms'] = exposure_ms
            self.spectrum_parameters[index] = parameters

    def get_frames(self, index):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined!'.format(index)
        return self.spectrum_parameters[index]['frames']

    def set_frames(self, index, frames):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined. Add a new spectrum before changing its parameters!'.format(index)
        parameters = self.spectrum_parameters[index].copy()
        if frames != parameters.get('frames'):
            parameters['frames'] = frames
            self.spectrum_parameters[index] = parameters

    def shift_x(self, eV):
        if callable(self.__active_settings['x_shifter']):
            self.__active_settings['x_shifter'](self.zeros['x'] + eV)
        elif self.__active_settings['x_shifter']:
            self.stem_controller.SetValAndConfirm(self.__active_settings['x_shifter'],
                                                  self.zeros['x'] + eV, 1.0, 1000)
        else: # do not wait if nothing was done
            return
        time.sleep(self.__active_settings['x_shift_delay'])

    def adjust_focus(self, x_shift_ev):
        pass

    def blank_beam(self):
        self.__set_beam_blanker(True)

    def unblank_beam(self):
        self.__set_beam_blanker(False)

    def __set_beam_blanker(self, blanker_on):
        if callable(self.__active_settings['blanker']):
            self.__active_settings['blanker'](blanker_on)
        elif self.__active_settings['blanker']:
            self.stem_controller.SetValAndConfirm(self.__active_settings['blanker'], 1 if blanker_on else 0, 1.0, 1000)
        else: # do not wait if nothing was done
            return
        time.sleep(self.__active_settings['blanker_delay'])

    def __calculate_total_acquisition_time(self, spectrum_parameters, settings, scan_parameters=None):
        total_time = 0
        if scan_parameters is not None:
            scan_size = scan_parameters['size']
        else:
            scan_size = (1, 1)
        for parameters in spectrum_parameters:
            total_time += (scan_size[1] * parameters['frames'] + self.__flyback_pixels) * parameters['exposure_ms']
        total_time *= scan_size[0]
        if settings['auto_dark_subtract']:
            total_time *= 2
        return total_time

    def __get_progress_maximum(self):
        return self.__calculate_total_acquisition_time(self.__active_spectrum_parameters, self.__active_settings,
                                                       getattr(self, 'scan_parameters', None))

    def get_total_acquisition_time(self):
        scan_parameters = self.scan_controller.get_current_frame_parameters() if self.scan_controller else None
        acquisition_time = self.__calculate_total_acquisition_time(self.spectrum_parameters, self.settings, None) * 0.001
        si_acquisition_time = self.__calculate_total_acquisition_time(self.spectrum_parameters, self.settings, scan_parameters) * 0.001
        if self.settings['auto_dark_subtract']:
            # Auto dark subtract has no effect for acquire SI, so correct the total acquisition time
            si_acquisition_time *= 0.5
        return acquisition_time, si_acquisition_time

    def increment_progress_counter(self, time):
        self.__progress_counter += time
        self.progress_updated_event.fire(0, self.__get_progress_maximum(), self.__progress_counter)

    def set_progress_counter(self, value, minimum=0, maximum=None):
        self.__progress_counter = value
        if maximum is None:
            maximum = self.__get_progress_maximum()
        self.progress_updated_event.fire(minimum, maximum, value)

    def reset_progress_counter(self):
        self.set_progress_counter(0, 0, 100)

    def __acquire_multi_acquire_data(self, number_pixels, line_number=0, flyback_pixels=0):
        for parameters in self.__active_spectrum_parameters:
            logging.debug('start preparations')
            starttime = time.time()
            if self.abort_event.is_set():
                break
            self.shift_x(parameters['offset_x'])
            self.adjust_focus(parameters['offset_x'])
            frame_parameters = self.camera.get_current_frame_parameters()
            frame_parameters['exposure_ms'] =  parameters['exposure_ms']
            frame_parameters['processing'] = 'sum_project' if self.__active_settings['bin_spectra'] else None
            self.camera.set_current_frame_parameters(frame_parameters)
            self.camera.acquire_sequence_prepare(parameters['frames']*number_pixels+flyback_pixels)
            logging.debug('finished preparations in {:g} s'.format(time.time() - starttime))
            starttime = 0
            logging.debug('start sequence')
            starttime = time.time()
            data_element = self.camera.acquire_sequence(parameters['frames']*number_pixels+flyback_pixels)
            if data_element:
                data_element = data_element[0]
            logging.debug('end sequence in {:g} s'.format(time.time() - starttime))
            if self.abort_event.is_set():
                break
            start_ev = data_element.get('spatial_calibrations', [{}])[-1].get('offset', 0)
            end_ev = start_ev + (data_element.get('spatial_calibrations', [{}])[-1].get('scale', 0) *
                                 data_element.get('data').shape[-1])
            parms = {'index': parameters['index'],
                     'exposure_ms': parameters['exposure_ms'],
                     'frames': parameters['frames'],
                     'start_ev': start_ev ,
                     'end_ev': end_ev,
                     'line_number': line_number,
                     'flyback_pixels': flyback_pixels}
            data_dict = {'data_element': data_element, 'parameters': parms, 'settings': dict(self.__active_settings)}
            self.__queue.put(data_dict)
            self.__flyback_pixels = flyback_pixels
            self.increment_progress_counter((parameters['frames']*number_pixels+flyback_pixels)*parameters['exposure_ms'])
            del data_element
            del data_dict
            logging.debug('finished acquisition')

    def __clean_up(self):
        # clear the queue to prevent deadlocks
        try:
            while True:
                self.__queue.get(block=False)
        except queue.Empty:
            pass
        else:
            self.__queue.task_done()
        # call task done again to make sure we can finish
        try:
            while True:
                self.__queue.task_done()
        except ValueError:
            pass

    def cancel(self):
        self.abort_event.set()
        try:
            self.camera.acquire_sequence_cancel()
            self.scan_controller.grab_synchronized_abort()
        except Exception as e:
            print(e)
        # give it some time to finish processing
        counter = 0
        while not self.__queue.empty():
            time.sleep(0.1)
            if counter > 10:
                break
            counter += 1
        # make sure we are in a good state to start again
        self.__clean_up()

    def acquire_multi_eels_spectrum(self):
        start_frame_parameters = None
        try:
            if hasattr(self, 'scan_parameters'):
                delattr(self, 'scan_parameters')
            self.reset_progress_counter()
            self.__active_settings = copy.deepcopy(self.settings)
            self.__active_spectrum_parameters = copy.deepcopy(self.spectrum_parameters)
            self.abort_event.clear()
            self.__acquisition_finished_event.clear()
            self.__process_and_send_data_thread = threading.Thread(target=self.process_and_send_data)
            self.__process_and_send_data_thread.start()
            self.acquisition_state_changed_event.fire({'message': 'start', 'description': 'single spectrum'})
            if hasattr(self, 'number_lines'):
                delattr(self, 'number_lines')
            data_dict_list = []
            def add_data_to_list(data_dict):
                data_dict_list.append(data_dict)
            new_data_listener = self.new_data_ready_event.listen(add_data_to_list)
            if not callable(self.__active_settings['x_shifter']) and self.__active_settings['x_shifter']:
                self.zeros['x'] = self.stem_controller.GetVal(self.__active_settings['x_shifter'])
            start_frame_parameters = self.camera.get_current_frame_parameters()
            # also use flyback pixels here to make sure we get fresh images from the camera (they get removed
            # automatically by "process_and_send_data")
            self.__acquire_multi_acquire_data(1, flyback_pixels=2)
            if self.__active_settings['auto_dark_subtract']:
                self.blank_beam()
                self.__acquire_multi_acquire_data(1, flyback_pixels=2)
                self.unblank_beam()
                self.__queue.join()
                for i in range(len(self.__active_spectrum_parameters)):
                    dark_data_dict = data_dict_list.pop(len(self.__active_spectrum_parameters))
                    dark_data = dark_data_dict['data_element']['data']
                    # if sum_frames is off we take the mean of the dark frames here. The frames axis will be
                    # the first axis in this case
                    if not self.__active_settings['sum_frames']:
                        dark_data = numpy.mean(dark_data, axis=0)
                    data_dict_list[i]['data_element']['data'] -= dark_data
        except:
            self.acquisition_state_changed_event.fire({'message': 'exception'})
            self.__clean_up()
            import traceback
            traceback.print_exc()
        finally:
            logging.debug('finished acquisition and dark subtraction')
            self.__acquisition_finished_event.set()
            self.acquisition_state_changed_event.fire({'message': 'end', 'description': 'single spectrum'})
            if start_frame_parameters:
                self.camera.set_current_frame_parameters(start_frame_parameters)
            self.shift_x(0)
            self.adjust_focus(0)
        self.__queue.join()
        new_data_listener.close()
        del new_data_listener

        data_element_list = []
        parameter_list = []
        settings_list = []
        for i in range(len(data_dict_list)):
            data_element = data_dict_list[i]['data_element']
            # remove the collection calibration (we acquired only "one pixel")
            data_element['spatial_calibrations'].pop(0 if self.__active_settings['sum_frames'] else 1)
            data_element['data'] = numpy.squeeze(data_element['data'])
            # this makes sure we do not create a length 1 sequence
            if not self.__active_settings['sum_frames'] and data_dict_list[i]['parameters']['frames'] < 2:
                data_element['is_sequence'] = False
                # We also need to delete the sequence axis calibration
                data_element['spatial_calibrations'].pop(0)
            data_element['collection_dimension_count'] = 0
            data_element_list.append(data_element)
            parameter_list.append(data_dict_list[i]['parameters'])
            settings_list.append(data_dict_list[i]['settings'])

        multi_eels_data = {'data_element_list' : data_element_list, 'parameter_list': parameter_list,
                           'settings_list': settings_list}
        return multi_eels_data

    def process_and_send_data(self):
        while True:
            try:
                data_dict = self.__queue.get(timeout=1)
            except queue.Empty:
                if self.__acquisition_finished_event.is_set():
                    self.acquisition_state_changed_event.fire({'message': 'end processing'})
                    break
            else:
                logging.debug('got data from queue')
                line_number = data_dict['parameters']['line_number']

                if (self.abort_event.is_set() or hasattr(self, 'number_lines') and
                    line_number == self.number_lines-1):
                    data_dict['parameters']['is_last_line'] = True

                if hasattr(self, 'number_lines'):
                    data_dict['parameters']['number_lines'] = self.number_lines

                data_element = data_dict['data_element']
                data = data_element['data']
                old_spatial_calibrations = data_element.get('spatial_calibrations', list())
                if self.__active_settings['bin_spectra'] and len(data.shape) > 2:
                    if len(old_spatial_calibrations) == len(data.shape):
                        old_spatial_calibrations.pop(1)
                    data = numpy.sum(data, axis=1)
                # remove flyback pixels
                flyback_pixels = data_dict['parameters']['flyback_pixels']
                data = data[flyback_pixels:, ...]
                # bring data to universal shape: ('pixels', 'frames', 'data', 'data')
                number_frames = data_dict['parameters']['frames']
                data = numpy.reshape(data, (-1, number_frames) + (data.shape[1:]))
                # sum along frames axis
                if self.__active_settings['sum_frames']:
                    data = numpy.sum(data, axis=1)
                # make frames axis the sequence axis
                else:
                    data = numpy.swapaxes(data, 0, 1)
                # put it back
                data_element['data'] = data
                # create correct data descriptors
                data_element['is_sequence'] = False if self.__active_settings['sum_frames'] else True
                data_element['collection_dimension_count'] = 1
                data_element['datum_dimension_count'] = 1 if self.__active_settings['bin_spectra'] else 2
                # update calibrations
                spatial_calibrations = [self.scan_calibrations[1].copy()]
                # check if raw data had correct number of calibrations, if not default to correct number of empty
                # calibrations to prevent errors
                if len(old_spatial_calibrations) == (len(data.shape) if self.__active_settings['sum_frames'] else
                                                     len(data.shape)-1):
                    spatial_calibrations.extend(old_spatial_calibrations[1:])
                    if not self.__active_settings['sum_frames']:
                        spatial_calibrations.insert(0, {'offset': 0, 'scale': 1, 'units': ''})
                else:
                    spatial_calibrations.extend([{'offset': 0, 'scale': 1, 'units': ''}
                                                 for i in range(len(data.shape)-1)])
                data_element['spatial_calibrations'] = spatial_calibrations
                counts_per_electron = data_element.get('properties', {}).get('counts_per_electron', 1)
                exposure_ms = data_element.get('properties', {}).get('exposure', 1)
                _number_frames = 1 if not self.__active_settings['sum_frames'] else number_frames
                intensity_scale = (data_element.get('intensity_calibration', {}).get('scale', 1) /
                                   counts_per_electron /
                                   data_element.get('spatial_calibrations', [{}])[-1].get('scale', 1) /
                                   exposure_ms / _number_frames)
                data_element['intensity_calibration'] = {'offset': 0, 'scale': intensity_scale, 'units': 'e/eV/s'}
                self.new_data_ready_event.fire(data_dict)
                logging.debug('processed line {:.0f}'.format(line_number))
                del data
                del data_element
                del data_dict
            try:
                self.__queue.task_done()
            except ValueError:
                pass

    def acquire_multi_eels_spectrum_image(self):
        self.__active_settings = copy.deepcopy(self.settings)
        self.__active_spectrum_parameters = copy.deepcopy(self.spectrum_parameters)
        self.abort_event.clear()
        self.reset_progress_counter()
        self.__acquisition_finished_event.clear()
        self.__process_and_send_data_thread = threading.Thread(target=self.process_and_send_data, daemon=True)
        self.__process_and_send_data_thread.start()
        if not callable(self.__active_settings['x_shifter']) and self.__active_settings['x_shifter']:
            self.zeros['x'] = self.stem_controller.GetVal(self.__active_settings['x_shifter'])

        def send_new_data_and_update_progress(data_dict):
            current_time = 0
            current_frame = data_dict['parameters']['current_frame']
            current_index = data_dict['parameters']['index']
            complete_shape = data_dict['parameters']['complete_shape']
            for parameters in self.__active_spectrum_parameters:
                if parameters['index'] >= current_index:
                    break
                current_time += complete_shape[0] * (complete_shape[1] + self.__flyback_pixels) * parameters['exposure_ms'] * parameters['frames']
            dest_sub_area = data_dict['dest_sub_area']
            current_time += complete_shape[0] * (complete_shape[1] + self.__flyback_pixels) * parameters['exposure_ms'] * current_frame
            current_time += dest_sub_area.bottom_right[0] * (complete_shape[1] + self.__flyback_pixels) * parameters['exposure_ms']
            self.set_progress_counter(current_time)
            self.new_data_ready_event.fire(data_dict)
        try:
            self.acquisition_state_changed_event.fire({'message': 'start', 'description': 'spectrum image'})
            for parameters in self.__active_spectrum_parameters:
                if self.abort_event.is_set():
                    break
                self.shift_x(parameters['offset_x'])
                self.adjust_focus(parameters['offset_x'])
                frame_parameters = self.camera.get_current_frame_parameters()
                frame_parameters['exposure_ms'] = parameters['exposure_ms']
                frame_parameters['processing'] = 'sum_project' if self.__active_settings['bin_spectra'] else None
                for n in range(parameters['frames']):
                    if self.abort_event.is_set():
                        break
                    parameters['current_frame'] = n
                    camera_data_channel = CameraDataChannel()
                    camera_data_channel.get_parameters_fn = lambda: parameters.copy()
                    camera_data_channel.get_settings_fn = lambda: self.__active_settings.copy()
                    new_data_listener = camera_data_channel.new_data_ready_event.listen(send_new_data_and_update_progress)
                    # grab_synchronized_info = self.scan_controller.grab_synchronized_get_info(scan_frame_parameters=self.scan_controller.get_current_frame_parameters(),
                    #                                                                    camera=self.camera,
                    #                                                                    camera_frame_parameters=frame_parameters)
                    self.scan_parameters = self.scan_controller.get_current_frame_parameters()
                    self.__flyback_pixels = 2
                    parameters['complete_shape'] = tuple(self.scan_parameters.size)
                    result = self.scan_controller.grab_synchronized(camera=self.camera, camera_frame_parameters=frame_parameters,
                                                              camera_data_channel=camera_data_channel,
                                                              scan_frame_parameters=self.scan_parameters)
                    if result is not None:
                        scan_xdata_list, _ = result
                        scan_data_dict = dict()
                        scan_data_dict['is_scan_data'] = True
                        scan_data_dict['xdata_list'] = scan_xdata_list
                        scan_data_dict['parameters'] = parameters.copy()
                        scan_data_dict['settings'] = self.__active_settings
                        self.new_data_ready_event.fire(scan_data_dict)
                    new_data_listener.close()
                    new_data_listener = None
        except Exception as e:
            self.acquisition_state_changed_event.fire({'message': 'exception', 'content': str(e)})
            import traceback
            traceback.print_stack()
            print(e)
            self.cancel()
            raise
        finally:
            self.__acquisition_finished_event.set()
            self.acquisition_state_changed_event.fire({'message': 'end', 'description': 'spectrum image'})
            self.acquisition_state_changed_event.fire({'message': 'end processing'})
            # TODO: configure line repeat
            self.shift_x(0)
            self.adjust_focus(0)
            if hasattr(self, 'scan_parameters'):
                delattr(self, 'scan_parameters')
