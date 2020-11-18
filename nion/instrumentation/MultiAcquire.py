# standard libraries
import copy
import json
import logging
import numpy
import os
import queue
import threading
import time
import gettext
import math
import functools
import typing
import uuid
import collections

# local libraries
from nion.utils import Event, Geometry
from nion.data import DataAndMetadata, Calibration
from nion.data import xdata_1_0 as xd
from nion.swift.model import DataItem
from nion.swift.model import DocumentModel
from nion.swift import Facade
from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.utils import Event

_ = gettext.gettext

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


class ScanDataChannel:
    title_base = _("MultiAcquire")

    def __init__(self, document_model: DocumentModel.DocumentModel, channel_names: typing.Sequence[str],
                 grab_sync_info: scan_base.ScanHardwareSource.GrabSynchronizedInfo,
                 multi_acquire_parameters: list, multi_acquire_settings: dict, current_parameters_index: int):
        self.__document_model = document_model
        self.__grab_sync_info = grab_sync_info
        self.__data_item_transactions: list = []
        self.__data_and_metadata = None
        self.__multi_acquire_parameters = multi_acquire_parameters
        self.__multi_acquire_settings = multi_acquire_settings
        self.__current_parameters_index = current_parameters_index
        self.current_frames_index = 0
        self.__data_items = [self.__create_data_item(channel_name) for channel_name in channel_names]

    def __create_data_item(self, channel_name: str) -> DataItem.DataItem:
        scan_calibrations = self.__grab_sync_info.scan_calibrations
        data_item = DataItem.DataItem(large_format=True)
        data_item.title = f'{ScanDataChannel.title_base} ({channel_name}) #{self.__multi_acquire_parameters[self.__current_parameters_index]["index"]+1}'
        self.__document_model.append_data_item(data_item)
        frames = self.__multi_acquire_parameters[self.__current_parameters_index]['frames']
        sum_frames = self.__multi_acquire_settings['sum_frames']
        if hasattr(data_item, "reserve_data"):
            scan_size = tuple(self.__grab_sync_info.scan_size)
            data_shape = scan_size
            if frames > 1 and not sum_frames:
                data_shape = (frames,) + scan_size
            data_descriptor = DataAndMetadata.DataDescriptor(frames > 1 and not sum_frames, 0, len(scan_size))
            data_item.reserve_data(data_shape=data_shape, data_dtype=numpy.float32, data_descriptor=data_descriptor)
        dimensional_calibrations = scan_calibrations
        if frames > 1 and not sum_frames:
            dimensional_calibrations = (Calibration.Calibration(),) + tuple(dimensional_calibrations)
        data_item.dimensional_calibrations = dimensional_calibrations
        data_item_metadata = data_item.metadata
        data_item_metadata["instrument"] = copy.deepcopy(self.__grab_sync_info.instrument_metadata)
        data_item_metadata["hardware_source"] = copy.deepcopy(self.__grab_sync_info.camera_metadata)
        data_item_metadata["scan"] = copy.deepcopy(self.__grab_sync_info.scan_metadata)
        data_item_metadata["MultiAcquire.settings"] = copy.deepcopy(self.__multi_acquire_settings)
        data_item_metadata["MultiAcquire.parameters"] = copy.deepcopy(self.__multi_acquire_parameters[self.__current_parameters_index])
        data_item.metadata = data_item_metadata
        return data_item

    def start(self) -> None:
        for data_item in self.__data_items:
            data_item.increment_data_ref_count()
            self.__data_item_transactions.append(self.__document_model.item_transaction(data_item))
            self.__document_model.begin_data_item_live(data_item)

    def update(self, data_and_metadata_list: typing.Sequence[DataAndMetadata.DataAndMetadata], state: str, view_id) -> None:
        frames = self.__multi_acquire_parameters[self.__current_parameters_index]['frames']
        sum_frames = self.__multi_acquire_settings['sum_frames']
        for i, data_and_metadata in enumerate(data_and_metadata_list):
            data_item = self.__data_items[i]
            scan_shape = data_and_metadata.data_shape
            data_shape_and_dtype = (tuple(scan_shape), data_and_metadata.data_dtype)
            data_descriptor = DataAndMetadata.DataDescriptor(frames > 1 and not sum_frames, 0, len(tuple(scan_shape)))
            dimensional_calibrations = data_and_metadata.dimensional_calibrations
            if frames > 1 and not sum_frames:
                dimensional_calibrations = (Calibration.Calibration(),) + tuple(dimensional_calibrations)
                data_shape = data_shape_and_dtype[0]
                data_shape = (frames,) + data_shape
                data_shape_and_dtype = (data_shape, data_shape_and_dtype[1])
            intensity_calibration = data_and_metadata.intensity_calibration
            metadata = data_and_metadata.metadata
            metadata["MultiAcquire.settings"] = copy.deepcopy(self.__multi_acquire_settings)
            metadata["MultiAcquire.parameters"] = copy.deepcopy(self.__multi_acquire_parameters[self.__current_parameters_index])
            data_metadata = DataAndMetadata.DataMetadata(data_shape_and_dtype,
                                                         intensity_calibration,
                                                         dimensional_calibrations,
                                                         metadata=data_and_metadata.metadata,
                                                         data_descriptor=data_descriptor)
            src_slice = (Ellipsis,)
            dst_slice = (Ellipsis,)
            if frames > 1:
                if sum_frames:
                    existing_data = data_item.data
                    if existing_data is not None:
                        summed_data = existing_data[dst_slice] + data_and_metadata.data[src_slice]
                        data_and_metadata._set_data(summed_data)
                else:
                    dst_slice = (self.current_frames_index,) + dst_slice # type: ignore
            self.__document_model.update_data_item_partial(data_item, data_metadata, data_and_metadata, src_slice, dst_slice)


    def stop(self) -> None:
        data_item_transactions = self.__data_item_transactions
        self.__data_item_transactions = []
        for transaction in data_item_transactions:
            transaction.close()
        for data_item in self.__data_items:
            self.__document_model.end_data_item_live(data_item)
            data_item.decrement_data_ref_count()


class CameraDataChannel(scan_base.SynchronizedDataChannelInterface):
    title_base = _("MultiAcquire")

    def __init__(self, document_model: DocumentModel.DocumentModel, channel_name: str,
                 grab_sync_info: scan_base.ScanHardwareSource.GrabSynchronizedInfo,
                 multi_acquire_parameters: list, multi_acquire_settings: dict,
                 current_parameters_index: int):
        self.__document_model = document_model
        self.__grab_sync_info = grab_sync_info
        self.__data_item_transaction = None
        self.__data_and_metadata = None
        self.__multi_acquire_parameters = multi_acquire_parameters
        self.__multi_acquire_settings = multi_acquire_settings
        self.__current_parameters_index = current_parameters_index
        self.current_frames_index = 0
        self.__data_item = self.__create_data_item(channel_name)
        self.progress_updated_event = Event.Event()

    def __create_data_item(self, channel_name: str) -> DataItem.DataItem:
        scan_calibrations = self.__grab_sync_info.scan_calibrations
        data_calibrations = self.__grab_sync_info.data_calibrations
        data_intensity_calibration = self.__grab_sync_info.data_intensity_calibration
        data_item = DataItem.DataItem(large_format=True)
        parameters = self.__multi_acquire_parameters[self.__current_parameters_index]
        data_item.title = f"{CameraDataChannel.title_base} ({channel_name}) #{self.__current_parameters_index+1}"
        self.__document_model.append_data_item(data_item)
        frames = parameters['frames']
        sum_frames = self.__multi_acquire_settings['sum_frames']
        if hasattr(data_item, "reserve_data"):
            scan_size = tuple(self.__grab_sync_info.scan_size)
            camera_readout_size = tuple(self.__grab_sync_info.camera_readout_size_squeezed)
            data_shape = scan_size + camera_readout_size
            if frames > 1 and not sum_frames:
                data_shape = (frames,) + data_shape
            data_descriptor = DataAndMetadata.DataDescriptor(frames > 1 and not sum_frames, 2, len(camera_readout_size))
            data_item.reserve_data(data_shape=data_shape, data_dtype=numpy.float32, data_descriptor=data_descriptor)
        dimensional_calibrations = scan_calibrations + data_calibrations
        if frames > 1 and not sum_frames:
            dimensional_calibrations = (Calibration.Calibration(),) + tuple(dimensional_calibrations)
        data_item.dimensional_calibrations = dimensional_calibrations
        data_item.intensity_calibration = data_intensity_calibration
        data_item_metadata = data_item.metadata
        data_item_metadata["instrument"] = copy.deepcopy(self.__grab_sync_info.instrument_metadata)
        data_item_metadata["hardware_source"] = copy.deepcopy(self.__grab_sync_info.camera_metadata)
        data_item_metadata["scan"] = copy.deepcopy(self.__grab_sync_info.scan_metadata)
        data_item_metadata["MultiAcquire.settings"] = copy.deepcopy(self.__multi_acquire_settings)
        data_item_metadata["MultiAcquire.parameters"] = copy.deepcopy(self.__multi_acquire_parameters[self.__current_parameters_index])
        data_item.metadata = data_item_metadata
        return data_item

    @property
    def data_item(self) -> DataItem.DataItem:
        return self.__data_item

    def update_progress(self, last_complete_line):
        current_time = 0
        current_frame = self.current_frames_index
        scan_size = tuple(self.__grab_sync_info.scan_size)
        for parameters in self.__multi_acquire_parameters:
            if parameters['index'] >= self.__current_parameters_index:
                break
            current_time += scan_size[0] * scan_size[1] * parameters['exposure_ms'] * parameters['frames']
        current_time += scan_size[0] * scan_size[1] * parameters['exposure_ms'] * current_frame
        current_time += last_complete_line * scan_size[1] * parameters['exposure_ms']
        self.progress_updated_event.fire(current_time)

    def start(self) -> None:
        self.__data_item.increment_data_ref_count()
        self.__data_item_transaction = self.__document_model.item_transaction(self.__data_item)
        self.__document_model.begin_data_item_live(self.__data_item)

    def update(self, data_and_metadata: DataAndMetadata.DataAndMetadata, state: str, scan_shape: Geometry.IntSize, dest_sub_area: Geometry.IntRect, sub_area: Geometry.IntRect, view_id) -> None:
        # there are a few techniques for getting data into a data item. this method prefers directly calling the
        # document model method update_data_item_partial, which is thread safe. if that method is not available, it
        # falls back to the data item method set_data_and_metadata, which must be called from the main thread.
        # the hardware source also supplies a data channel which is thread safe and ends up calling set_data_and_metadata
        # but we skip that so that the updates fit into this class instead.
        frames = self.__multi_acquire_parameters[self.__current_parameters_index]['frames']
        sum_frames = self.__multi_acquire_settings['sum_frames']
        collection_rank = len(tuple(scan_shape))
        data_shape_and_dtype = (tuple(scan_shape) + data_and_metadata.data_shape[collection_rank:], data_and_metadata.data_dtype)
        data_descriptor = DataAndMetadata.DataDescriptor(frames > 1 and not sum_frames, collection_rank, len(data_and_metadata.data_shape) - collection_rank)
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        if frames > 1 and not sum_frames:
            dimensional_calibrations = (Calibration.Calibration(),) + tuple(dimensional_calibrations)
            data_shape = data_shape_and_dtype[0]
            data_shape = (frames,) + data_shape
            data_shape_and_dtype = (data_shape, data_shape_and_dtype[1])

        intensity_calibration = data_and_metadata.intensity_calibration
        if self.__multi_acquire_settings['use_multi_eels_calibration']:
            metadata = data_and_metadata.metadata.get('hardware_source', {})
            counts_per_electron = metadata.get('counts_per_electron', 1)
            exposure_s = metadata.get('exposure', self.__multi_acquire_parameters[self.__current_parameters_index]['exposure_ms']*0.001)
            _number_frames = 1 if not sum_frames else frames
            intensity_scale = (data_and_metadata.intensity_calibration.scale / counts_per_electron /
                               data_and_metadata.dimensional_calibrations[-1].scale / exposure_s / _number_frames)
            intensity_calibration = Calibration.Calibration(scale=intensity_scale)


        metadata = data_and_metadata.metadata
        metadata["MultiAcquire.settings"] = copy.deepcopy(self.__multi_acquire_settings)
        metadata["MultiAcquire.parameters"] = copy.deepcopy(self.__multi_acquire_parameters[self.__current_parameters_index])
        data_metadata = DataAndMetadata.DataMetadata(data_shape_and_dtype,
                                                     intensity_calibration,
                                                     dimensional_calibrations,
                                                     metadata=data_and_metadata.metadata,
                                                     data_descriptor=data_descriptor)
        src_slice = sub_area.slice + (Ellipsis,)
        dst_slice = dest_sub_area.slice + (Ellipsis,)
        if frames > 1:
            if sum_frames:
                existing_data = self.__data_item.data
                if existing_data is not None:
                    existing_data[dst_slice] += data_and_metadata.data[src_slice]
            else:
                dst_slice = (self.current_frames_index,) + dst_slice # type: ignore

        self.__document_model.update_data_item_partial(self.__data_item, data_metadata, data_and_metadata, src_slice, dst_slice)
        self.update_progress(dest_sub_area.bottom)

    def stop(self) -> None:
        if self.__data_item_transaction:
            self.__data_item_transaction.close()
            self.__data_item_transaction = None
            self.__document_model.end_data_item_live(self.__data_item)
            self.__data_item.decrement_data_ref_count()


class SequenceBehavior:
    def __init__(self):
        ...

    def prepare_frame(self):
        ...


SISequenceBehavior = collections.namedtuple('SISequenceBehavior', ['scan_behavior', 'scan_section_height',
                                                                   'sequence_behavior', 'sequence_section_length'])

class SISequenceAcquisitionHandler:
    def __init__(self,
                 camera: camera_base.CameraHardwareSource,
                 camera_data_channel: scan_base.SynchronizedDataChannelInterface,
                 camera_frame_parameters: dict,
                 scan_controller: scan_base.ScanHardwareSource,
                 scan_data_channel: scan_base.SynchronizedDataChannelInterface,
                 scan_frame_parameters: dict,
                 si_sequence_behavior: typing.Optional[SISequenceBehavior] = None):

        self.__camera = camera
        self.camera_data_channel = camera_data_channel
        self.camera_frame_parameters = camera_frame_parameters
        self.__scan_controller = scan_controller
        self.scan_data_channel = scan_data_channel
        self.scan_frame_parameters = scan_frame_parameters
        self.__si_sequence_behavior = si_sequence_behavior or SISequenceBehavior(None, None, None, None)
        self.is_canceled = False
        self.finish_fn: typing.Optional[typing.Callable[[], None]] = None

    def run(self, number_frames: int):
        for frame in range(number_frames):
            if self.is_canceled:
                break
            self.camera_data_channel.current_frames_index = frame
            self.scan_data_channel.current_frames_index = frame
            if self.__si_sequence_behavior.sequence_behavior and self.__si_sequence_behavior.sequence_section_length and frame % self.__si_sequence_behavior.sequence_section_length == 0:
                self.__si_sequence_behavior.sequence_behavior.prepare_frame()
            combined_data = self.__scan_controller.grab_synchronized(camera=self.__camera,
                                                                     camera_frame_parameters=self.camera_frame_parameters,
                                                                     camera_data_channel=self.camera_data_channel,
                                                                     scan_frame_parameters=self.scan_frame_parameters,
                                                                     section_height=self.__si_sequence_behavior.scan_section_height,
                                                                     scan_behavior=self.__si_sequence_behavior.scan_behavior)
            if combined_data is not None:
                scan_xdata_list, _ = combined_data
                data_channel_state = 'complete' if frame >= number_frames - 1 else 'partial'
                data_channel_view_id = None
                self.scan_data_channel.update(scan_xdata_list, data_channel_state, data_channel_view_id)
        if callable(self.finish_fn):
            self.finish_fn()


class MultiAcquireController:
    def __init__(self, stem_controller: stem_controller.STEMController, savepath=None):
        self.spectrum_parameters = MultiEELSParameters(
                                   [{'index': 0, 'offset_x': 0, 'exposure_ms': 1, 'frames': 1},
                                    {'index': 1, 'offset_x': 160, 'exposure_ms': 8, 'frames': 1},
                                    {'index': 2, 'offset_x': 320, 'exposure_ms': 16, 'frames': 1}])
        self.settings = MultiEELSSettings(
                        {'x_shifter': 'LossMagnetic', 'blanker': 'C_Blank', 'x_shift_delay': 0.05,
                         'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'bin_spectra': True,
                         'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': '',
                         'use_multi_eels_calibration': False})
        self.stem_controller = stem_controller
        self.camera: camera_base.CameraHardwareSource = None
        self.scan_controller: scan_base.ScanHardwareSource = None
        self.zeros = {'x': 0, 'focus': 0}
        self.scan_calibrations = [{'offset': 0, 'scale': 1, 'units': ''}, {'offset': 0, 'scale': 1, 'units': ''}]
        self.__progress_counter = 0
        self.acquisition_state_changed_event = Event.Event()
        self.new_scan_data_ready_event = Event.Event()
        self.progress_updated_event = Event.Event()
        self.__active_settings = self.settings
        self.__active_spectrum_parameters = self.spectrum_parameters
        self.abort_event = threading.Event()
        self.__savepath = savepath # or os.path.join(os.path.expanduser('~'), 'MultiAcquire')
        self.load_settings()
        self.load_parameters()
        self.__settings_changed_event_listener = self.settings.settings_changed_event.listen(self.save_settings)
        self.__spectrum_parameters_changed_event_listener = self.spectrum_parameters.parameters_changed_event.listen(self.save_parameters)

    @property
    def active_settings(self):
        return self.__active_settings

    @property
    def active_spectrum_parameters(self):
        return self.__active_spectrum_parameters

    def save_settings(self):
        if self.__savepath:
            os.makedirs(self.__savepath, exist_ok=True)
            with open(os.path.join(self.__savepath, 'settings.json'), 'w+') as f:
                json.dump(self.settings, f)

    def load_settings(self):
        if self.__savepath and os.path.isfile(os.path.join(self.__savepath, 'settings.json')):
            with open(os.path.join(self.__savepath, 'settings.json')) as f:
                self.settings.update(json.load(f))

    def save_parameters(self):
        if self.__savepath:
            os.makedirs(self.__savepath, exist_ok=True)
            with open(os.path.join(self.__savepath, 'spectrum_parameters.json'), 'w+') as f:
                json.dump(self.spectrum_parameters, f)

    def load_parameters(self):
        if self.__savepath and os.path.isfile(os.path.join(self.__savepath, 'spectrum_parameters.json')):
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
            total_time += scan_size[1] * parameters['frames'] * parameters['exposure_ms']
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

    def cancel(self):
        self.abort_event.set()
        try:
            self.scan_controller.grab_synchronized_abort()
            self.camera.acquire_sequence_cancel()
        except Exception as e:
            print(e)

    def acquire_multi_eels_spectrum(self):
        start_frame_parameters = None
        try:
            if hasattr(self, 'scan_parameters'):
                delattr(self, 'scan_parameters')
            self.reset_progress_counter()
            self.__active_settings = copy.deepcopy(self.settings)
            self.__active_spectrum_parameters = copy.deepcopy(self.spectrum_parameters)
            self.abort_event.clear()
            self.acquisition_state_changed_event.fire({'message': 'start', 'description': 'single spectrum'})
            data_dict_list = []
            if not callable(self.__active_settings['x_shifter']) and self.__active_settings['x_shifter']:
                self.zeros['x'] = self.stem_controller.GetVal(self.__active_settings['x_shifter'])
            start_frame_parameters = self.camera.get_current_frame_parameters()

            for parameters in self.__active_spectrum_parameters:
                if self.abort_event.is_set():
                    break
                self.shift_x(parameters['offset_x'])
                frame_parameters = self.camera.get_current_frame_parameters()
                frame_parameters['exposure_ms'] =  parameters['exposure_ms']
                frame_parameters['processing'] = 'sum_project' if self.__active_settings['bin_spectra'] else None
                self.camera.set_current_frame_parameters(frame_parameters)
                self.camera.acquire_sequence_prepare(parameters['frames'])
                data_element = self.camera.acquire_sequence(parameters['frames'])
                if data_element:
                    data_element = data_element[0]
                else:
                    break
                start_ev = data_element.get('spatial_calibrations', [{}])[-1].get('offset', 0)
                end_ev = start_ev + (data_element.get('spatial_calibrations', [{}])[-1].get('scale', 0) *
                                     data_element.get('data').shape[-1])

                parameters['start_ev'] = start_ev
                parameters['end_ev'] = end_ev
                data_element['collection_dimension_count'] = 0
                data_element['datum_dimension_count'] = 1 if self.__active_settings['bin_spectra'] else 2
                # sum along frames axis
                if self.__active_settings['sum_frames'] or parameters['frames'] < 2:
                    data_element['data'] = numpy.sum(data_element['data'], axis=0)
                    data_element['is_sequence'] = False
                    spatial_calibrations = data_element['spatial_calibrations']
                    if spatial_calibrations:
                        data_element['spatial_calibrations'] = spatial_calibrations[1:]
                else:
                    data_element['is_sequence'] = True

                if self.__active_settings['use_multi_eels_calibration']:
                    counts_per_electron = data_element.get('properties', {}).get('counts_per_electron', 1)
                    exposure_s = data_element.get('properties', {}).get('exposure', parameters['exposure_ms']*0.001)
                    _number_frames = 1 if not self.__active_settings['sum_frames'] else parameters['frames']
                    intensity_scale = (data_element.get('intensity_calibration', {}).get('scale', 1) /
                                       counts_per_electron /
                                       data_element.get('spatial_calibrations', [{}])[-1].get('scale', 1) /
                                       exposure_s / _number_frames)
                    data_element['intensity_calibration'] = {'offset': 0, 'scale': intensity_scale, 'units': 'e/eV/s'}

                self.increment_progress_counter(parameters['frames']*parameters['exposure_ms'])

                if self.__active_settings['auto_dark_subtract']:
                    self.blank_beam()
                    self.camera.acquire_sequence_prepare(parameters['frames'])
                    dark_data_element = self.camera.acquire_sequence(parameters['frames'])
                    self.unblank_beam()
                    if dark_data_element:
                        dark_data = dark_data_element[0]['data']
                    else:
                        break
                    if self.__active_settings['sum_frames']:
                        dark_data = numpy.sum(dark_data, axis=0)
                    else:
                        dark_data = numpy.mean(dark_data, axis=0)

                    data_element['data'] -= dark_data

                    self.increment_progress_counter(parameters['frames']*parameters['exposure_ms'])

                data_dict_list.append({'data_element': data_element, 'parameters': parameters, 'settings': dict(self.__active_settings)})

        except:
            self.acquisition_state_changed_event.fire({'message': 'exception'})
            import traceback
            traceback.print_exc()
        finally:
            self.acquisition_state_changed_event.fire({'message': 'end', 'description': 'single spectrum'})
            if start_frame_parameters:
                self.camera.set_current_frame_parameters(start_frame_parameters)
            self.shift_x(0)

        data_element_list = []
        parameter_list = []
        settings_list = []
        for i in range(len(data_dict_list)):
            data_element_list.append(data_dict_list[i]['data_element'])
            parameter_list.append(data_dict_list[i]['parameters'])
            settings_list.append(data_dict_list[i]['settings'])

        multi_eels_data = {'data_element_list': data_element_list, 'parameter_list': parameter_list,
                           'settings_list': settings_list}
        return multi_eels_data

    def start_multi_acquire_spectrum_image(self, get_acquisition_handler_fn: typing.Callable[[list, int, dict], SISequenceAcquisitionHandler]):
        self.__active_settings = copy.deepcopy(self.settings)
        self.__active_spectrum_parameters = copy.deepcopy(self.spectrum_parameters)
        self.reset_progress_counter()
        self.abort_event.clear()
        if not callable(self.__active_settings['x_shifter']) and self.__active_settings['x_shifter']:
            self.zeros['x'] = self.stem_controller.GetVal(self.__active_settings['x_shifter'])
        self.acquisition_state_changed_event.fire({'message': 'start', 'description': 'spectrum image'})
        try:
            for parameters in self.__active_spectrum_parameters:
                self.shift_x(parameters['offset_x'])
                acquisition_handler = get_acquisition_handler_fn(list(self.__active_spectrum_parameters), parameters['index'], dict(self.__active_settings))
                # Set scan frame parameters as attribute so that acquistion time for progress bar will be calculated correctly
                self.scan_parameters = acquisition_handler.scan_frame_parameters
                acquisition_handler.run(parameters['frames'])
        except Exception as e:
            self.acquisition_state_changed_event.fire({'message': 'exception', 'content': str(e)})
            import traceback
            traceback.print_stack()
            self.cancel()
            raise
        finally:
            self.acquisition_state_changed_event.fire({'message': 'end', 'description': 'spectrum image'})
            self.shift_x(0)
            if hasattr(self, 'scan_parameters'):
                delattr(self, 'scan_parameters')
