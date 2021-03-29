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
from nion.swift.model import ImportExportManager
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
            data_item.reserve_data(data_shape=data_shape, data_dtype=numpy.dtype(numpy.float32), data_descriptor=data_descriptor)
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
                 current_parameters_index: int,
                 stack_metadata_keys: typing.Optional[typing.Sequence[typing.Sequence[str]]] = None):
        self.__document_model = document_model
        self.__grab_sync_info = grab_sync_info
        self.__data_item_transaction = None
        self.__data_and_metadata = None
        self.__multi_acquire_parameters = multi_acquire_parameters
        self.__multi_acquire_settings = multi_acquire_settings
        self.__current_parameters_index = current_parameters_index
        self.__stack_metadata_keys = stack_metadata_keys
        self.current_frames_index = 0
        self.__data_item = self.__create_data_item(channel_name, grab_sync_info)
        self.progress_updated_event = Event.Event()

    def __calculate_axes_order_and_data_shape(self, axes_descriptor: scan_base.ScanHardwareSource.AxesDescriptor, scan_shape: Geometry.IntSize, camera_readout_size: typing.Tuple[int, ...]) -> typing.Tuple[typing.List[int], typing.Tuple[int, ...]]:
        # axes_descriptor provides the information needed to re-order the axes of the result data approprietly.
        axes_order = []
        if axes_descriptor.sequence_axes:
            axes_order.extend(axes_descriptor.sequence_axes)
        if axes_descriptor.collection_axes:
            axes_order.extend(axes_descriptor.collection_axes)
        if axes_descriptor.data_axes:
            axes_order.extend(axes_descriptor.data_axes)

        data_shape = tuple(scan_shape) + camera_readout_size

        assert len(axes_order) == len(data_shape)

        data_shape = numpy.array(data_shape)[axes_order].tolist()
        collection_dimension_count = len(axes_descriptor.collection_axes) if axes_descriptor.collection_axes is not None else 0
        is_sequence = axes_descriptor.sequence_axes is not None
        if collection_dimension_count == 1:
            collection_axis = 1 if is_sequence else 0
            if data_shape[collection_axis] == 1:
                data_shape.pop(collection_axis)

        return axes_order, tuple(data_shape)

    def __create_data_item(self, channel_name: str, grab_sync_info: scan_base.ScanHardwareSource.GrabSynchronizedInfo) -> DataItem.DataItem:
        scan_calibrations = grab_sync_info.scan_calibrations
        data_calibrations = grab_sync_info.data_calibrations
        axes_descriptor = grab_sync_info.axes_descriptor
        data_intensity_calibration = grab_sync_info.data_intensity_calibration
        _, data_shape = self.__calculate_axes_order_and_data_shape(axes_descriptor, grab_sync_info.scan_size, grab_sync_info.camera_readout_size_squeezed)
        data_item = DataItem.DataItem(large_format=True)
        parameters = self.__multi_acquire_parameters[self.__current_parameters_index]
        data_item.title = f"{CameraDataChannel.title_base} ({channel_name}) #{self.__current_parameters_index+1}"
        self.__document_model.append_data_item(data_item)
        frames = parameters['frames']
        sum_frames = self.__multi_acquire_settings['sum_frames']
        is_sequence = False
        if frames > 1 and not sum_frames:
            data_shape = (frames,) + tuple(data_shape)
            is_sequence = True
        collection_dimension_count = len(axes_descriptor.collection_axes) if axes_descriptor.collection_axes is not None else 0
        datum_dimension_count = len(axes_descriptor.data_axes) if axes_descriptor.data_axes is not None else 0
        # This is for the case of one virtual detector, where we squeeze the length-1 axis
        if collection_dimension_count == 1:
            expected_ndim = int(is_sequence) + collection_dimension_count + datum_dimension_count
            if expected_ndim > len(data_shape):
                collection_dimension_count = 0
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)
        data_item.reserve_data(data_shape=data_shape, data_dtype=numpy.dtype(numpy.float32), data_descriptor=data_descriptor)
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
        # This method is always called with a collection of 1d or 2d data. Re-order axes as required and remove length-1-axes.

        axes_descriptor = self.__grab_sync_info.axes_descriptor

        # Calibrations, data descriptor and shape estimates are updated accordingly.
        dimensional_calibrations = data_and_metadata.dimensional_calibrations
        data = data_and_metadata.data
        axes_order, data_shape = self.__calculate_axes_order_and_data_shape(axes_descriptor, scan_shape, data.shape[len(tuple(scan_shape)):])
        assert len(axes_order) == data.ndim
        data = numpy.moveaxis(data, axes_order, list(range(data.ndim)))
        dimensional_calibrations = numpy.array(dimensional_calibrations)[axes_order].tolist()
        is_sequence = axes_descriptor.sequence_axes is not None
        collection_dimension_count = len(axes_descriptor.collection_axes) if axes_descriptor.collection_axes is not None else 0
        datum_dimension_count = len(axes_descriptor.data_axes) if axes_descriptor.data_axes is not None else 0
        metadata = data_and_metadata.metadata

        src_slice = tuple()
        dst_slice = tuple()
        for index in axes_order:
            if index >= len(sub_area.slice):
                src_slice += (slice(None),)
            else:
                src_slice += (sub_area.slice[index],)
            if index >= len(dest_sub_area.slice):
                dst_slice += (slice(None),)
            else:
                dst_slice += (dest_sub_area.slice[index],)

        # This is to make virtual detector data look sensible: If only one virtual detector is defined the data should
        # not have a length-1 collection axis
        if collection_dimension_count == 1:
            collection_axis = 1 if is_sequence else 0
            if data.shape[collection_axis] == 1:
                data = numpy.squeeze(data, axis=collection_axis)
                dimensional_calibrations = dimensional_calibrations[1:]
                src_slice = list(src_slice)
                src_slice.pop(collection_axis)
                src_slice = tuple(src_slice)
                dst_slice = list(dst_slice)
                dst_slice.pop(collection_axis)
                dst_slice = tuple(dst_slice)
                collection_dimension_count = 0
        data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension_count, datum_dimension_count)

        data_and_metadata = DataAndMetadata.new_data_and_metadata(data, data_and_metadata.intensity_calibration,
                                                                  dimensional_calibrations,
                                                                  data_and_metadata.metadata, None,
                                                                  data_descriptor, None,
                                                                  None)

        frames = self.__multi_acquire_parameters[self.__current_parameters_index]['frames']
        sum_frames = self.__multi_acquire_settings['sum_frames']
        data_shape_and_dtype = (data_shape, data_and_metadata.data_dtype)
        data_descriptor = DataAndMetadata.DataDescriptor(frames > 1 and not sum_frames, collection_dimension_count, datum_dimension_count)
        if frames > 1 and not sum_frames:
            if self.__multi_acquire_settings['shift_each_sequence_slice']:
                sequence_calibration = Calibration.Calibration(scale=self.__multi_acquire_parameters[self.__current_parameters_index]['offset_x'])
            else:
                sequence_calibration = Calibration.Calibration()
            dimensional_calibrations = (sequence_calibration,) + tuple(dimensional_calibrations)
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

        metadata["MultiAcquire.settings"] = copy.deepcopy(self.__multi_acquire_settings)
        metadata["MultiAcquire.parameters"] = copy.deepcopy(self.__multi_acquire_parameters[self.__current_parameters_index])
        # This is needed for metadata that changes with each spectrum image in the stack and needs to be preserved.
        # One usecase is the storage information that comes with virtual detector data that has the full dataset saved
        # in the background. Currently the camera defines which metadata keys to stack and we copy that information
        # when setting up the CameraDataChannel
        if self.__stack_metadata_keys is not None:
            for key_path in self.__stack_metadata_keys:
                existing_data = None
                if isinstance(key_path, str):
                    key_path = [key_path]
                sub_dict = self.__data_item.metadata
                for key in key_path:
                    sub_dict = sub_dict.get(key)
                    if sub_dict is None:
                        break
                else:
                    existing_data = copy.deepcopy(sub_dict)

                parent = None
                sub_dict = metadata
                for key in key_path:
                    parent = sub_dict
                    sub_dict = sub_dict.get(key)
                    if sub_dict is None:
                        break
                else:
                    if self.current_frames_index == 0 and parent is not None:
                        parent[key_path[-1]] = [sub_dict]
                    elif existing_data is not None and parent is not None:
                        if isinstance(existing_data, list):
                            if len(existing_data) <= self.current_frames_index:
                                existing_data.append(copy.deepcopy(sub_dict))
                            else:
                                existing_data[self.current_frames_index] = copy.deepcopy(sub_dict)
                            parent[key_path[-1]] = existing_data

        data_metadata = DataAndMetadata.DataMetadata(data_shape_and_dtype,
                                                     intensity_calibration,
                                                     dimensional_calibrations,
                                                     metadata=metadata,
                                                     data_descriptor=data_descriptor)

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
    def __init__(self, multi_acquire_controller: 'MultiAcquireController', current_parameters_index: int):
        self.__multi_acquire_controller = multi_acquire_controller
        self.__current_parameters_index = current_parameters_index
        self.__last_shift = 0

    def prepare_frame(self):
        if self.__multi_acquire_controller.active_settings['shift_each_sequence_slice']:
            self.__multi_acquire_controller.shift_x(self.__last_shift)
            self.__last_shift += self.__multi_acquire_controller.active_spectrum_parameters[self.__current_parameters_index]['offset_x']


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
        self.abort_event = threading.Event()
        self.finish_fn: typing.Optional[typing.Callable[[], None]] = None

    def run(self, number_frames: int):
        for frame in range(number_frames):
            if self.abort_event.is_set():
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
                         'focus': '', 'focus_delay': 0, 'auto_dark_subtract': False, 'processing': 'sum_project',
                         'blanker_delay': 0.05, 'sum_frames': True, 'camera_hardware_source_id': '',
                         'use_multi_eels_calibration': False, 'shift_each_sequence_slice': False})
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
                settings_dict = json.load(f)
                # Upgrade the settings dict to the new version. We replaced "bin_spectra" with "processing" to allow
                # "sum_masked" as an additional option
                bin_spectra = settings_dict.pop('bin_spectra', None)
                if bin_spectra is not None:
                    settings_dict['processing'] = 'sum_project' if bin_spectra else None
                self.settings.update(settings_dict)

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

    def __calculate_total_acquisition_time(self, spectrum_parameters, settings, scan_parameters=None, include_shift_delay=False):
        total_time = 0
        if scan_parameters is not None:
            scan_size = scan_parameters['size']
        else:
            scan_size = (1, 1)
        for parameters in spectrum_parameters:
            total_time += scan_size[1] * scan_size[0] * parameters['frames'] * parameters['exposure_ms']
            if include_shift_delay:
                if settings['shift_each_sequence_slice']:
                    total_time += settings['x_shift_delay'] * parameters['frames'] * 1000
                else:
                    total_time += settings['x_shift_delay'] * 1000
        if settings['auto_dark_subtract']:
            total_time *= 2
        return total_time

    def __get_progress_maximum(self):
        return self.__calculate_total_acquisition_time(self.__active_spectrum_parameters, self.__active_settings,
                                                       getattr(self, 'scan_parameters', None))

    def get_total_acquisition_time(self):
        scan_parameters = self.scan_controller.get_current_frame_parameters() if self.scan_controller else None
        acquisition_time = self.__calculate_total_acquisition_time(self.spectrum_parameters, self.settings, None, True) * 0.001
        si_acquisition_time = self.__calculate_total_acquisition_time(self.spectrum_parameters, self.settings, scan_parameters, True) * 0.001
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
                # When we shift each spectrum, we change our zero posiiton after each frame. But at the end of the acquisiton
                # we still want to go back to the initial value of the control, so we "back up" the zero point here
                self.zeros['x_start'] = self.zeros['x']
            start_frame_parameters = self.camera.get_current_frame_parameters()

            for parameters in self.__active_spectrum_parameters:
                if self.abort_event.is_set():
                    break

                frame_parameters = self.camera.get_current_frame_parameters()
                frame_parameters['exposure_ms'] =  parameters['exposure_ms']
                frame_parameters['processing'] = self.__active_settings['processing']
                self.camera.set_current_frame_parameters(frame_parameters)
                if self.__active_settings['shift_each_sequence_slice']:
                    data_element = None
                    for i in range(parameters['frames']):
                        if self.abort_event.is_set():
                            break
                        if i > 0:
                            self.shift_x(parameters['offset_x'])
                            # If we shift each slice we need to save the current state after each frame
                            self.zeros['x'] = self.stem_controller.GetVal(self.__active_settings['x_shifter'])
                        xdata_list = self.camera.grab_next_to_start()
                        data_element = ImportExportManager.create_data_element_from_extended_data(xdata_list[0])
                        if i == 0:
                            if self.__active_settings['processing'] == 'sum_project':
                                shape = (parameters['frames'], data_element['data'].shape[-1])
                            else:
                                shape = (parameters['frames'],) + data_element['data'].shape
                            data = numpy.empty(shape, dtype=data_element['data'].dtype)
                        if self.__active_settings['processing'] == 'sum_project' and data_element['data'].ndim == 2:
                            data[i] = numpy.sum(data_element['data'], axis=0)
                            if 'spatial_calibrations' in data_element:
                                data_element['spatial_calibrations'] = [data_element['spatial_calibrations'][-1],]
                        else:
                            data[i] = data_element['data']
                        self.increment_progress_counter(parameters['exposure_ms'])
                    if data_element:
                        data_element['data'] = data
                        if 'spatial_calibrations' in data_element:
                            data_element['spatial_calibrations'] = [dict(),] + data_element['spatial_calibrations']
                        data_element = [data_element]

                else:
                    self.shift_x(parameters['offset_x'])
                    self.camera.acquire_sequence_prepare(parameters['frames'])
                    data_element = self.camera.acquire_sequence(parameters['frames'])
                    self.increment_progress_counter(parameters['frames'] * parameters['exposure_ms'])

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
                data_element['datum_dimension_count'] = 1 if self.__active_settings['processing'] == 'sum_project' else 2
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
            # When each frame was shifted we want to use the backed up initial value when shifting back to zero so
            # that we can actually go fully back to the start.
            if 'x_start' in self.zeros:
                self.zeros['x'] = self.zeros['x_start']
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
            # When we shift each spectrum, we change our zero posiiton after each frame. But at the end of the acquisiton
            # we still want to go back to the initial value of the control, so we "back up" the zero point here
            self.zeros['x_start'] = self.zeros['x']
        self.acquisition_state_changed_event.fire({'message': 'start', 'description': 'spectrum image'})
        try:
            for parameters in self.__active_spectrum_parameters:
                if self.abort_event.is_set():
                    break
                if not self.__active_settings['shift_each_sequence_slice']:
                    self.shift_x(parameters['offset_x'])
                acquisition_handler = get_acquisition_handler_fn(list(self.__active_spectrum_parameters), parameters['index'], dict(self.__active_settings))
                acquisition_handler.abort_event = self.abort_event
                # Set scan frame parameters as attribute so that acquistion time for progress bar will be calculated correctly
                self.scan_parameters = acquisition_handler.scan_frame_parameters
                acquisition_handler.run(parameters['frames'])
                # If we shift each slice we need to save the current state after each sequence
                if self.__active_settings['shift_each_sequence_slice']:
                    self.zeros['x'] = self.stem_controller.GetVal(self.__active_settings['x_shifter'])
        except Exception as e:
            self.acquisition_state_changed_event.fire({'message': 'exception', 'content': str(e)})
            import traceback
            traceback.print_stack()
            self.cancel()
            raise
        finally:
            self.acquisition_state_changed_event.fire({'message': 'end', 'description': 'spectrum image'})
            # When each frame was shifted we want to use the backed up initial value when shifting back to zero so
            # that we can actually go fully back to the start.
            if 'x_start' in self.zeros:
                self.zeros['x'] = self.zeros['x_start']
            self.shift_x(0)
            if hasattr(self, 'scan_parameters'):
                delattr(self, 'scan_parameters')
