# standard libraries
import contextlib
import copy
import json
import logging
import numpy
import os
import queue
import threading
import time

# local libraries
from nion.utils import Event
from nion.instrumentation.scan_base import RecordTask


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


class MultiAcquireController:
    def __init__(self, **kwargs):
        self.spectrum_parameters = MultiEELSParameters(
                                   [{'index': 0, 'offset_x': 0, 'offset_y': 0, 'exposure_ms': 1, 'frames': 1},
                                    {'index': 1, 'offset_x': 160, 'offset_y': 0, 'exposure_ms': 8, 'frames': 1},
                                    {'index': 2, 'offset_x': 320, 'offset_y': 0, 'exposure_ms': 16, 'frames': 1}])
        self.settings = MultiEELSSettings(
                        {'x_shifter': 'LossMagnetic', 'y_shifter': '', 'x_units_per_ev': 1,
                         'y_units_per_px': 0.00081, 'blanker': 'C_Blank', 'x_shift_delay': 0.05, 'y_shift_delay': 0.05,
                         'focus': '', 'focus_delay': 0, 'saturation_value': 12000, 'y_align': True,
                         'stitch_spectra': False, 'auto_dark_subtract': False, 'bin_spectra': True,
                         'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': ''})
        self.stem_controller = None
        self.camera = None
        self.superscan = None
        self.zeros = {'x': 0, 'y': 0, 'focus': 0}
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

    def get_offset_y(self, index):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined!'.format(index)
        return self.spectrum_parameters[index]['offset_y']

    def set_offset_y(self, index, offset_y):
        assert index < len(self.spectrum_parameters), 'Index {:.0f} > then number of spectra defined. Add a new spectrum before changing its parameters!'.format(index)
        parameters = self.spectrum_parameters[index].copy()
        if offset_y != parameters.get('offset_y'):
            parameters['offset_y'] = offset_y
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
            self.__active_settings['x_shifter'](self.zeros['x'] + eV*self.__active_settings['x_units_per_ev'])
        elif self.__active_settings['x_shifter']:
            self.stem_controller.SetValAndConfirm(self.__active_settings['x_shifter'],
                                                  self.zeros['x'] + eV*self.__active_settings['x_units_per_ev'], 1.0, 1000)
        else: # do not wait if nothing was done
            return
        time.sleep(self.__active_settings['x_shift_delay'])

    def shift_y(self, px):
        if callable(self.__active_settings['y_shifter']):
            self.__active_settings['y_shifter'](self.zeros['y'] + px*self.__active_settings['y_units_per_px'])
        elif self.__active_settings['y_shifter']:
            self.stem_controller.SetValAndConfirm(self.__active_settings['y_shifter'],
                                                  self.zeros['y'] + px*self.__active_settings['y_units_per_px'], 1.0, 1000)
        else: # do not wait if nothing was done
            return
        time.sleep(self.__active_settings['y_shift_delay'])

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

    def increment_progress_counter(self, time):
        self.__progress_counter += time
        maximum = 0
        if hasattr(self, 'scan_parameters'):
            scan_size = self.scan_parameters['size']
        else:
            scan_size = (1, 1)
        for parameters in self.__active_spectrum_parameters:
            maximum += (scan_size[1] * parameters['frames'] + self.__flyback_pixels) * parameters['exposure_ms']
        maximum *= scan_size[0]
        if self.__active_settings['auto_dark_subtract']:
            maximum *= 2
        self.progress_updated_event.fire(0, maximum, self.__progress_counter)

    def set_progress_counter(self, minimum, maximum, value):
        self.__progress_counter = value
        self.progress_updated_event.fire(minimum, maximum, value)

    def reset_progress_counter(self):
        self.set_progress_counter(0, 100, 0)

    def get_stitch_ranges(self, spectra):
        crop_ranges = []
        y_range_0 = numpy.array((self.__active_spectrum_parameters[0]['offset_y'], self.__active_spectrum_parameters[1]['offset_y']-1))
        x_range_target = numpy.array((0, 0))
        for i in range(1, len(spectra)):
            y_range = y_range_0 + self.__active_spectrum_parameters[i-1]['offset_y']
            calibration = spectra[i-1].dimensional_calibrations[-1].write_dict()
            calibration_next = spectra[i].dimensional_calibrations[-1].write_dict()
            end = numpy.rint((calibration_next['offset']-calibration['offset'])/calibration['scale']).astype(numpy.int)
            x_range_source = numpy.array((0, end))
            x_range_target = numpy.array((x_range_target[1], x_range_target[1] + x_range_source[1]-x_range_source[0]))
            crop_ranges.append((y_range, x_range_source, x_range_target))
        y_range = y_range_0 + self.__active_spectrum_parameters[-1]['offset_y']
        x_range_source = numpy.array((0, None))
        x_range_target = numpy.array((x_range_target[1], x_range_target[1] + spectra[-1].data.shape[1]))
        crop_ranges.append((y_range, x_range_source, x_range_target))

        return crop_ranges

    def stitch_spectra(self, spectra, crop_ranges):
        result = numpy.zeros(crop_ranges[-1][2][1])
        last_mean = None
        last_overlap = None
        for i in range(len(crop_ranges)):
            data = (spectra[i].data)/self.__active_spectrum_parameters[i]['exposure_ms']
            # check if hdr needs to be done in this interval, which is true when the overlap with the next one is 100%
            if i > 0 and (crop_ranges[i-1][1][1] - crop_ranges[i-1][1][0]) == 0:
                data[data>self.__active_settings['saturation_value']] = (spectra[i-1].data[[data>self.__active_settings['saturation_value']]])/self.__active_spectrum_parameters[i]['exposure_ms']

            data = numpy.sum(data[crop_ranges[i][0][0]:crop_ranges[i][0][1]], axis=0)
            if last_mean is not None:
                data -= numpy.mean(data[crop_ranges[i][1][0]:last_overlap]) - last_mean
                print(last_mean, last_overlap)
            if self.__active_settings['y_align'] and i < len(crop_ranges) - 1:
                last_overlap = data.shape[-1] - crop_ranges[i][1][1]
                last_mean = numpy.mean(data[crop_ranges[i][1][1]:])
            result[crop_ranges[i][2][0]:crop_ranges[i][2][1]] = data[crop_ranges[i][1][0]:crop_ranges[i][1][1]]

        return result

    def __acquire_multi_acquire_data(self, number_pixels, line_number=0, flyback_pixels=0):
        for parameters in self.__active_spectrum_parameters:
            print('start preparations')
            starttime = time.time()
            if self.abort_event.is_set():
                break
            self.shift_x(parameters['offset_x'])
            self.shift_y(parameters['offset_y'])
            self.adjust_focus(parameters['offset_x'])
            frame_parameters = self.camera.get_current_frame_parameters()
            frame_parameters['exposure_ms'] =  parameters['exposure_ms']
            frame_parameters['processing'] = 'sum_project' if self.__active_settings['bin_spectra'] else None
            self.camera.set_current_frame_parameters(frame_parameters)
            self.camera.acquire_sequence_prepare(parameters['frames']*number_pixels+flyback_pixels)
            print('finished preparations in {:g} s'.format(time.time() - starttime))
            starttime = 0
            print('start sequence')
            starttime = time.time()
            data_element = self.camera.acquire_sequence(parameters['frames']*number_pixels+flyback_pixels)
            if data_element:
                data_element = data_element[0]
            print('end sequence in {:g} s'.format(time.time() - starttime))
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
            print('finished acquisition')

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
            if not callable(self.__active_settings['y_shifter']) and self.__active_settings['y_shifter']:
                self.zeros['y'] = self.stem_controller.GetVal(self.__active_settings['y_shifter'])
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
            print('finished acquisition and dark subtraction')
            self.__acquisition_finished_event.set()
            self.acquisition_state_changed_event.fire({'message': 'end', 'description': 'single spectrum'})
            if start_frame_parameters:
                self.camera.set_current_frame_parameters(start_frame_parameters)
            self.shift_y(0)
            self.shift_x(0)
            self.adjust_focus(0)
        self.__queue.join()
        del new_data_listener
        if self.__active_settings['stitch_spectra']:
            raise NotImplementedError
        else:
            data_element_list = []
            parameter_list = []
            settings_list = []
            for i in range(len(data_dict_list)):
                data_element = data_dict_list[i]['data_element']
                data_element['data'] = numpy.squeeze(data_element['data'])
                # this makes sure we do not create a length 1 sequence
                if not self.__active_settings['sum_frames'] and data_dict_list[i]['parameters']['frames'] < 2:
                    data_element['is_sequence'] = False
                data_element['spatial_calibrations'].pop(0 if self.__active_settings['sum_frames'] else 1)
                data_element['collection_dimension_count'] = 0
                data_element_list.append(data_element)
                parameter_list.append(data_dict_list[i]['parameters'])
                settings_list.append(data_dict_list[i]['settings'])

            multi_eels_data = {'data_element_list' : data_element_list, 'parameter_list': parameter_list,
                               'settings_list': settings_list, 'stitched_data': False}
            return multi_eels_data

    def acquire_multi_eels_line(self, x_pixels, line_number, flyback_pixels=2, first_line=False, last_line=False):
        self.__acquire_multi_acquire_data(x_pixels, line_number, flyback_pixels)

    def process_and_send_data(self):
        while True:
            try:
                data_dict = self.__queue.get(timeout=1)
            except queue.Empty:
                if self.__acquisition_finished_event.is_set():
                    self.acquisition_state_changed_event.fire({'message': 'end processing'})
                    break
            else:
                print('got data from queue')
                if self.__active_settings['stitch_spectra']:
                    raise NotImplementedError
                    data_dict_list = [data_dict]
                    while len(data_dict_list) < len(self.__active_spectrum_parameters):
                        try:
                            data_dict = self.__queue.get(timeout=1)
                        except queue.Empty:
                            continue
                    del data_dict_list
                else:
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
                    print('processed line {:.0f}'.format(line_number))
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
        self.__process_and_send_data_thread = threading.Thread(target=self.process_and_send_data)
        self.__process_and_send_data_thread.start()
        if not callable(self.__active_settings['x_shifter']) and self.__active_settings['x_shifter']:
            self.zeros['x'] = self.stem_controller.GetVal(self.__active_settings['x_shifter'])
        if not callable(self.__active_settings['y_shifter']) and self.__active_settings['y_shifter']:
            self.zeros['y'] = self.stem_controller.GetVal(self.__active_settings['y_shifter'])
        try:
            logging.debug("start")
            self.acquisition_state_changed_event.fire({'message': 'start', 'description': 'spectrum image'})
            self.superscan.abort_playing()
            self.camera.abort_playing()
            self.scan_parameters = self.superscan.get_record_frame_parameters()
            scan_max_size = numpy.inf
            self.scan_parameters["size"] = (min(scan_max_size, self.scan_parameters["size"][0]),
                                            min(scan_max_size, self.scan_parameters["size"][1]))
            self.scan_parameters["pixel_time_us"] = int(100) #int(1000 * eels_camera_parameters["exposure_ms"] * 0.75)
            self.scan_parameters["external_clock_wait_time_ms"] = int(20000) #int(eels_camera_parameters["exposure_ms"]) + 100
            self.scan_parameters["external_clock_mode"] = 1
            self.scan_parameters["ac_frame_sync"] = False
            self.scan_parameters["ac_line_sync"] = False
            self.scan_calibrations = [{'offset': -self.scan_parameters['fov_size_nm'][0]/2,
                                       'scale': self.scan_parameters['fov_size_nm'][0]/self.scan_parameters['size'][0],
                                       'units': 'nm'},
                                      {'offset': -self.scan_parameters['fov_size_nm'][1]/2,
                                       'scale': self.scan_parameters['fov_size_nm'][1]/self.scan_parameters['size'][1],
                                       'units': 'nm'}]
            flyback_pixels = self.superscan.flyback_pixels
            self.number_lines = self.scan_parameters["size"][0]
            self.queue = queue.Queue()
            self.process_and_send_data_thread = threading.Thread(target=self.process_and_send_data)
            self.process_and_send_data_thread.start()
            # TODO: configure line repeat
            with contextlib.closing(RecordTask(self.superscan, self.scan_parameters)) as scan_task:
                for line in range(self.number_lines):
                    if self.abort_event.is_set():
                        break
                    print(line)
                    starttime = time.time()
                    self.acquire_multi_eels_line(self.scan_parameters["size"][1], line, flyback_pixels=flyback_pixels, first_line=line==0)
                    print('acquired line in {:g} s'.format(time.time() - starttime))
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
            # TODO: configure line repeat
            self.shift_y(0)
            self.shift_x(0)
            self.adjust_focus(0)
            if hasattr(self, 'scan_parameters'):
                delattr(self, 'scan_parameters')
            logging.debug("end")