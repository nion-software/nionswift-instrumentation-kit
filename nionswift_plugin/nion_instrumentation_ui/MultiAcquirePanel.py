# standard libraries
import itertools
import numpy
import queue
import threading
import logging

# local libraries
import typing
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation import MultiAcquire
from nion.swift.model import ImportExportManager
from nion.ui import Dialog
from nion.utils import Registry
from nion.data import DataAndMetadata, Calibration


# copied from https://github.com/mrmrs/colors
color_names = {
    'aqua':    '#7fdbff',
    'blue':    '#0074d9',
    'lime':    '#01ff70',
    'navy':    '#001f3f',
    'teal':    '#39cccc',
    'olive':   '#3d9970',
    'green':   '#2ecc40',
    'red':     '#ff4136',
    'maroon':  '#85144b',
    'orange':  '#ff851b',
    'purple':  '#b10dc9',
    'yellow':  '#ffdc00',
    'fuchsia': '#f012be',
    'gray':    '#aaaaaa',
    'white':   '#ffffff',
    'black':   '#111111',
    'silver':  '#dddddd'
}

color_order = ['blue',
               'red',
               'green',
               'navy',
               'aqua',
               'yellow',
               'gray',
               'black',
               'orange',
               'maroon',
               'purple',
               'teal',
               'olive',
               'lime',
               'fuchsia',
               'silver']

_color_cycle = itertools.cycle(color_order)

def get_next_color():
    color = next(_color_cycle)
    return color_names[color]

def reset_color_cycle():
    global _color_cycle
    _color_cycle = itertools.cycle(color_order)


def show_display_item(document_window, display_item):
    for display_panel in document_window._document_window.workspace_controller.display_panels:
        if display_panel.display_item == display_item:
            display_panel.request_focus()
            return
    result_display_panel = document_window._document_window.next_result_display_panel()
    if result_display_panel:
        result_display_panel.set_display_panel_display_item(display_item)
        result_display_panel.request_focus()


class AppendableDataItemFixedSize:
    def __init__(self, data_item, final_data_shape, api, dtype=numpy.float32):
        self.__data_cache = numpy.zeros(final_data_shape, dtype=dtype)
        self.__api = api
        self.__data_item_ref = None
        self.__data_item = data_item

    @property
    def _data_item(self):
        return self.__data_item

    def enter_write_suspend_state(self):
        if not self.__data_item_ref:
            self.__data_item_ref = self.__api.library.data_ref_for_data_item(self.__data_item)
            self.__data_item_ref.__enter__()

    def exit_write_suspend_state(self):
        if self.__data_item_ref:
            self.__data_item_ref.__exit__(None, None, None)
        self.__data_item_ref = None

    def add_data(self, slice_tuple, data):
        self.__data_cache[slice_tuple] = data

    def get_data(self, slice_tuple):
        return self.__data_cache[slice_tuple]

    def get_partial_data_item(self, slice_tuple):
        xdata = self.__api.create_data_and_metadata(self.__data_cache[slice_tuple],
                                                    intensity_calibration=self.__data_item.xdata.intensity_calibration,
                                                    dimensional_calibrations=self.__data_item.xdata.dimensional_calibrations,
                                                    metadata=self.__data_item.xdata.metadata,
                                                    data_descriptor=self.__data_item.xdata.data_descriptor)
        self.__data_item.set_data_and_metadata(xdata)
        return self.__data_item

    def get_full_data_item(self):
        return self.get_partial_data_item((...,))


class MultiAcquirePanelDelegate:

    def __init__(self, api):
        self.__api = api
        self.panel_id = 'MultiAcquire-Panel'
        self.panel_name = 'Multi-Acquire'
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
        self.api = api
        self.line_edit_widgets = None
        self.multi_acquire_controller = None
        self.__acquisition_state_changed_event_listener = None
        self.__multi_eels_parameters_changed_event_listener = None
        self.__progress_updated_event_listener = None
        self.__settings_changed_event_listener = None
        self.__component_registered_event_listener = None
        self.__component_unregistered_event_listener = None
        self.__scan_frame_parameters_changed_event_listener = None
        self.__new_data_ready_event_listener = None
        self._stem_controller = None
        self.eels_camera = None
        self._scan_controller: typing.Optional[scan_base.ScanHardwareSource] = None
        self.settings_window_open = False
        self.parameters_window_open = False
        self.parameter_column = None
        self.result_data_items = None
        self.__acquisition_running = False
        self.__display_queue = None
        self.__display_thread = None
        self.__acquisition_thread = None
        self.__data_processed_event = None
        self.time_estimate_label = None

    @property
    def scan_controller(self) -> typing.Optional[scan_base.ScanHardwareSource]:
        if not self._scan_controller and self.stem_controller:
            self._scan_controller = typing.cast(scan_base.ScanHardwareSource, self.stem_controller.scan_controller)
        return self._scan_controller

    @property
    def stem_controller(self) -> typing.Optional[stem_controller.STEMController]:
        if self._stem_controller is None:
            self._stem_controller = Registry.get_component('stem_controller')
        return self._stem_controller

    # For testing
    @property
    def _data_processed_event(self):
        return self.__data_processed_event

    def _close_listeners_for_test(self) -> None:
        if self.__progress_updated_event_listener:
            self.__progress_updated_event_listener.close()
            self.__progress_updated_event_listener = None
        if self.__multi_eels_parameters_changed_event_listener:
            self.__multi_eels_parameters_changed_event_listener.close()
            self.__multi_eels_parameters_changed_event_listener = None
        if self.__acquisition_state_changed_event_listener:
            self.__acquisition_state_changed_event_listener.close()
            self.__acquisition_state_changed_event_listener = None

    def close(self):
        # close anything created in `create_panel_widget`.
        # called when the panel closes, not when the delegate closes.
        if self.__acquisition_state_changed_event_listener:
            self.__acquisition_state_changed_event_listener.close()
            self.__acquisition_state_changed_event_listener = None
        if self.__multi_eels_parameters_changed_event_listener:
            self.__multi_eels_parameters_changed_event_listener.close()
            self.__multi_eels_parameters_changed_event_listener = None
        if self.__progress_updated_event_listener:
            self.__progress_updated_event_listener.close()
            self.__progress_updated_event_listener = None
        if self.__settings_changed_event_listener:
            self.__settings_changed_event_listener.close()
            self.__settings_changed_event_listener = None
        if self.__component_registered_event_listener:
            self.__component_registered_event_listener.close()
            self.__component_registered_event_listener = None
        if self.__component_unregistered_event_listener:
            self.__component_unregistered_event_listener.close()
            self.__component_unregistered_event_listener = None
        if self.__new_data_ready_event_listener:
            self.__new_data_ready_event_listener.close()
            self.__new_data_ready_event_listener = None
        if self.__scan_frame_parameters_changed_event_listener:
            self.__scan_frame_parameters_changed_event_listener.close()
            self.__scan_frame_parameters_changed_event_listener = None
        if self.__data_processed_event:
            self.__data_processed_event.set()
        if self.__display_thread:
            self.__display_thread.join()
            self.__display_thread = None
        self.__display_queue = None
        self.line_edit_widgets = None
        self.multi_acquire_controller = None
        self._stem_controller = None
        self.eels_camera = None
        self._scan_controller = None
        self.settings_window_open = False
        self.parameters_window_open = False
        self.parameter_column = None
        self.result_data_items = None
        self.__acquisition_running = False
        self.__acquisition_thread = None
        self.__data_processed_event = None
        self.time_estimate_label = None

    def spectrum_parameters_changed(self):
        parameter_list = self.multi_acquire_controller.spectrum_parameters.copy()
        if len(parameter_list) != len(self.parameter_column._widget.children):
            self.parameter_column._widget.remove_all()
            for spectrum_parameters in parameter_list:
                self.parameter_column.add(self.create_parameter_line(spectrum_parameters))
        else:
            for spectrum_parameters in parameter_list:
                self.update_parameter_line(spectrum_parameters)
        self.update_time_estimate()

    def create_result_data_item(self, data_dict):
        display_layers = []
        reset_color_cycle()
        display_item = None
        sorted_indices = numpy.argsort([parms['start_ev'] for parms in data_dict['parameter_list']])
        for i in sorted_indices:
            index = data_dict['parameter_list'][i]['index']
            xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_dict['data_element_list'][i])
            start_ev = data_dict['parameter_list'][i]['start_ev']
            end_ev = data_dict['parameter_list'][i]['end_ev']
            number_frames = data_dict['parameter_list'][i]['frames']
            exposure_ms = data_dict['parameter_list'][i]['exposure_ms']
            summed = ' (summed)' if not data_dict['data_element_list'][i].get('is_sequence', False) and number_frames > 1 else ''
            data_item = None
            if i == sorted_indices[0] and xdata.datum_dimension_count == 1:
                data_item = self.document_controller.library.create_data_item_from_data_and_metadata(
                                                                    xdata,
                                                                    title='MultiAcquire (stacked)')
                display_item = self.__api.library._document_model.get_display_item_for_data_item(data_item._data_item)
                #new_data_item = data_item
            #else:
            new_data_item = self.document_controller.library.create_data_item_from_data_and_metadata(
                                xdata,
                                title='MultiAcquire #{:d}, {:g}-{:g} eV, {:g}x{:g} ms{}'.format(index+1,
                                                                                                start_ev,
                                                                                                end_ev,
                                                                                                number_frames,
                                                                                                exposure_ms,
                                                                                                summed))
            metadata = new_data_item.metadata
            metadata['MultiAcquire.parameters'] = data_dict['parameter_list'][i]
            metadata['MultiAcquire.settings'] = data_dict['settings_list'][i]
            new_data_item.set_metadata(metadata)
            if display_item:
                display_item.append_display_data_channel_for_data_item(data_item._data_item if data_item else new_data_item._data_item)
                start_ev = data_dict['parameter_list'][i]['start_ev']
                end_ev = data_dict['parameter_list'][i]['end_ev']
                display_layers.append({'label': '#{:d}: {:g}-{:g} eV, {:g}x{:g} ms{}'.format(index+1,
                                                                                            start_ev,
                                                                                            end_ev,
                                                                                            number_frames,
                                                                                            exposure_ms,
                                                                                            summed),
                                       'data_index': len(display_layers),
                                       'stroke_color': get_next_color(),
                                       'fill_color':None})
        if display_item:
            display_item.display_layers = display_layers
            display_item.set_display_property('legend_position', 'top-right')
            display_item.title = 'MultiAcquire (stacked)'
            show_display_item(self.document_controller, display_item)

    def acquisition_state_changed(self, info_dict):
        if info_dict.get('message') == 'start':
            self.__acquisition_running = True
            if info_dict.get('description') == 'spectrum image':
                def update_buttons():
                    self.start_si_button.text = 'Abort Multi-Acquire spectrum image'
                    self.start_button._widget.enabled = False
            else:
                def update_buttons():
                    self.start_button.text = 'Abort Multi-Acquire'
                    self.start_si_button._widget.enabled = False
            self.__api.queue_task(update_buttons)
        elif info_dict.get('message') == 'end':
            self.__acquisition_running = False
            def update_buttons():
                self.start_si_button.text = 'Start Multi-Acquire spectrum image'
                self.start_button.text = 'Start Multi-Acquire'
                self.start_button._widget.enabled = True
                self.start_si_button._widget.enabled = True
            self.__api.queue_task(update_buttons)
        elif info_dict.get('message') == 'exception':
            self.__acquisition_running = False
            self.__close_data_item_refs()
            def update_buttons():
                self.start_si_button.text = 'Start Multi-Acquire spectrum image'
                self.start_button.text = 'Start Multi-Acquire'
                self.start_button._widget.enabled = True
                self.start_si_button._widget.enabled = True
            self.__api.queue_task(update_buttons)
            self.__data_processed_event.set()
        elif info_dict.get('message') == 'end processing':
            self.__data_processed_event.set()

    def __close_data_item_refs(self):
        logging.debug('Closing data item refs')
        for item in self.result_data_items.values():
            item.exit_write_suspend_state()
#        for item in self.result_data_items:
#            try:
#                while True:
#                    item.xdata.decrement_data_ref_count()
#            except AssertionError:
#                pass
        if self.__new_data_ready_event_listener:
            self.__new_data_ready_event_listener.close()
        self.__new_data_ready_event_listener = None
        self.result_data_items = None

    def add_to_display_queue(self, data_dict):
        self.__display_queue.put(data_dict)

    def process_scan_data(self, scan_data_dict):
        index = scan_data_dict['parameters']['index']
        number_frames = scan_data_dict['parameters']['frames']
        exposure_ms = scan_data_dict['parameters']['exposure_ms']
        current_frame = scan_data_dict['parameters']['current_frame']
        scan_xdata_list = scan_data_dict['xdata_list']
        for scan_xdata in scan_xdata_list:
            channel_name = scan_xdata.metadata['hardware_source']['channel_name']
            channel_id = scan_xdata.metadata['hardware_source']['channel_id']
            data_item_key = f'{index}{channel_id}'
            if not self.result_data_items.get(data_item_key):
                metadata = scan_xdata.metadata
                metadata['MultiAcquire.parameters'] = dict(scan_data_dict['parameters'])
                metadata['MultiAcquire.settings'] = dict(scan_data_dict['settings'])
                scan_xdata._set_metadata(metadata)
                title = 'MultiAcquire ({}) #{:d}, {:g}x{:g} ms'.format(channel_name, index+1, number_frames, exposure_ms)
                data_item_ready_event = threading.Event()
                scan_shape = scan_xdata.data.shape
                new_data_item = None
                def create_data_item():
                    nonlocal new_data_item, scan_xdata
                    if number_frames > 1 and not scan_data_dict['settings']['sum_frames']:
                        data = scan_xdata.data[numpy.newaxis, ...]
                        dimensional_calibrations = [Calibration.Calibration()] + list(scan_xdata.dimensional_calibrations)
                        data_descriptor = DataAndMetadata.DataDescriptor(True,
                                                                         scan_xdata.data_descriptor.collection_dimension_count,
                                                                         scan_xdata.data_descriptor.datum_dimension_count)
                        scan_xdata = DataAndMetadata.new_data_and_metadata(data,
                                                                           intensity_calibration=scan_xdata.intensity_calibration,
                                                                           dimensional_calibrations=dimensional_calibrations,
                                                                           metadata=scan_xdata.metadata,
                                                                           timestamp=scan_xdata.timestamp,
                                                                           data_descriptor=data_descriptor,
                                                                           timezone=scan_xdata.timezone,
                                                                           timezone_offset=scan_xdata.timezone_offset)
                    new_data_item = self.__api.library.create_data_item_from_data_and_metadata(scan_xdata,
                                                                                               title=title)
                    data_item_ready_event.set()
                self.__api.queue_task(create_data_item)
                data_item_ready_event.wait()
                if number_frames > 1:
                    if scan_data_dict['settings']['sum_frames']:
                        max_shape = scan_shape
                    else:
                        max_shape = (number_frames,) + scan_shape
                    new_appendable_data_item = AppendableDataItemFixedSize(new_data_item, max_shape, self.__api)
                    new_appendable_data_item.enter_write_suspend_state()
                    self.result_data_items[data_item_key] = new_appendable_data_item
                    del new_appendable_data_item

            if scan_data_dict['settings']['sum_frames'] and number_frames > 1:
                data = self.result_data_items[data_item_key].get_data((...,))
                data += scan_xdata.data
                def get_and_display_data_item():
                    data_item = self.result_data_items[data_item_key].get_full_data_item()
                    try:
                        self.__api.application.document_controllers[0].display_data_item(data_item)
                    except AttributeError:
                        pass
                self.__api.queue_task(get_and_display_data_item)
            elif number_frames > 1:
                self.result_data_items[data_item_key].add_data((current_frame, ...), scan_xdata.data)
                def get_and_display_data_item():
                    data_item = self.result_data_items[data_item_key].get_partial_data_item((slice(0, current_frame+1), ...))
                    try:
                        self.__api.application.document_controllers[0].display_data_item(data_item)
                    except AttributeError:
                        pass
                self.__api.queue_task(get_and_display_data_item)

    def process_display_queue(self):
        while True:
            try:
                data_dict = self.__display_queue.get(timeout=1)
            except queue.Empty:
                if self.__data_processed_event.is_set() and not self.__acquisition_running:
                    self.__close_data_item_refs()
                    break
            else:
                if data_dict.get('is_scan_data'):
                    self.process_scan_data(data_dict)
                    continue
                index = data_dict['parameters']['index']
                start_ev = data_dict['parameters']['start_ev']
                end_ev = data_dict['parameters']['end_ev']
                number_frames = data_dict['parameters']['frames']
                exposure_ms = data_dict['parameters']['exposure_ms']
                dest_sub_area = data_dict['dest_sub_area']
                current_frame = data_dict['parameters']['current_frame']
                # state = data_dict['state']
                xdata = data_dict['xdata']
                logging.debug('got data from display queue')
                if not self.result_data_items.get(index):
                    logging.debug('creating new data item')
                    metadata = xdata.metadata
                    metadata['MultiAcquire.parameters'] = dict(data_dict['parameters'])
                    metadata['MultiAcquire.settings'] = dict(data_dict['settings'])
                    xdata._set_metadata(metadata)
                    title = 'MultiAcquire #{:d}, {:g}-{:g} eV, {:g}x{:g} ms'.format(index+1, start_ev, end_ev,
                                                                                    number_frames, exposure_ms)
                    data_item_ready_event = threading.Event()
                    new_data_item = None
                    def create_data_item():
                        nonlocal new_data_item
                        # we have to create the initial data item with some data that has more than 2 dimensions
                        # otherwise Swift does not use HDF5 and we will have a problem if it grows too big later
                        new_data_item = self.__api.library.create_data_item_from_data_and_metadata(xdata,
                                                                                                   title=title)
                        try:
                            self.__api.application.document_controllers[0].display_data_item(new_data_item)
                        except AttributeError:
                            pass
                        data_item_ready_event.set()
                    self.__api.queue_task(create_data_item)
                    data_item_ready_event.wait()
                    max_shape = data_dict['parameters']['complete_shape']
                    if number_frames > 1 and not data_dict['settings']['sum_frames']:
                        max_shape = (number_frames,) + max_shape
                    max_shape += xdata.datum_dimension_shape
                    new_appendable_data_item = AppendableDataItemFixedSize(new_data_item, max_shape, self.__api)
                    new_appendable_data_item.enter_write_suspend_state()
                    self.result_data_items[index] = new_appendable_data_item
                    del new_appendable_data_item

                data_item = self.result_data_items[index]
                if data_dict['settings']['sum_frames']:
                    data = data_item.get_data(dest_sub_area.slice)
                    data += xdata.data
                    def get_and_display_data_item():
                        data_item.get_partial_data_item((slice(0, dest_sub_area.bottom_right[0]),
                                                         slice(0, dest_sub_area.bottom_right[1])))

                    self.__api.queue_task(get_and_display_data_item)
                elif number_frames > 1:
                    data_item.add_data((current_frame,) + dest_sub_area.slice, xdata.data)
                    def get_and_display_data_item():
                        data_item.get_partial_data_item((slice(0, current_frame+1),
                                                         slice(0, dest_sub_area.bottom_right[0]),
                                                         slice(0, dest_sub_area.bottom_right[1])))
                        #self.__api.application.document_controllers[0].display_data_item(data_item)
                    self.__api.queue_task(get_and_display_data_item)
                else:
                    data_item.add_data(dest_sub_area.slice, xdata.data)
                    def get_and_display_data_item():
                        data_item.get_partial_data_item((slice(0, dest_sub_area.bottom_right[0]),
                                                         slice(0, dest_sub_area.bottom_right[1])))
                        #self.__api.application.document_controllers[0].display_data_item(data_item)
                    self.__api.queue_task(get_and_display_data_item)
                del data_dict
                self.__display_queue.task_done()

    def update_progress_bar(self, minimum, maximum, value):
        if self.progress_bar:
            def update():
                self.progress_bar.minimum = minimum
                self.progress_bar.maximum = maximum
                self.progress_bar.value = value
            self.__api.queue_task(update)

    def create_panel_widget(self, ui, document_controller):
        # note: anything created here should be disposed in close.
        # this method may be called more than once.

        self.line_edit_widgets = {}
        self.multi_acquire_controller = MultiAcquire.MultiAcquireController()
        self.__acquisition_state_changed_event_listener = self.multi_acquire_controller.acquisition_state_changed_event.listen(self.acquisition_state_changed)
        self.__multi_eels_parameters_changed_event_listener = self.multi_acquire_controller.spectrum_parameters.parameters_changed_event.listen(self.spectrum_parameters_changed)
        self.__progress_updated_event_listener = self.multi_acquire_controller.progress_updated_event.listen(self.update_progress_bar)
        self.__settings_changed_event_listener = None
        self.__component_registered_event_listener = None
        self.__component_unregistered_event_listener = None
        self.__scan_frame_parameters_changed_event_listener = None
        self.__new_data_ready_event_listener = None
        self._stem_controller = None
        self.eels_camera = None
        self._scan_controller = None
        self.settings_window_open = False
        self.parameters_window_open = False
        self.parameter_column = None
        self.result_data_items = None
        self.__acquisition_running = False
        self.__display_queue = None
        self.__display_thread = None
        self.__acquisition_thread = None
        self.__data_processed_event = None

        self.ui = ui
        self.document_controller = document_controller

        def start_clicked():
            if self.__acquisition_running:
                self.multi_acquire_controller.cancel()
            else:
                self.multi_acquire_controller.stem_controller = self.stem_controller
                self.multi_acquire_controller.camera = self.camera_choice_combo_box.current_item
                def run_multi_eels():
                    data_dict = self.multi_acquire_controller.acquire_multi_eels_spectrum()
                    def create_and_display_data_item():
                        self.create_result_data_item(data_dict)
                    document_controller.queue_task(create_and_display_data_item)  # must occur on UI thread
                self.__acquisition_thread = threading.Thread(target=run_multi_eels, daemon=True)
                self.__acquisition_thread.start()

        def start_si_clicked():
            if self.__acquisition_running:
                self.multi_acquire_controller.cancel()
            else:
                self.multi_acquire_controller.stem_controller = self.stem_controller
                self.multi_acquire_controller.camera = self.camera_choice_combo_box.current_item
                self.multi_acquire_controller.scan_controller = self.scan_controller
                self.__new_data_ready_event_listener = self.multi_acquire_controller.new_data_ready_event.listen(self.add_to_display_queue)
                self._start_display_queue_thread()
                self.__acquisition_thread = threading.Thread(
                        target=self.multi_acquire_controller.acquire_multi_eels_spectrum_image, daemon=True)
                self.__acquisition_thread.start()

        def settings_button_clicked():
            if not self.settings_window_open:
                self.settings_window_open = True
                self.show_config_box()

        def camera_changed(current_item):
            if current_item:
                self.multi_acquire_controller.settings['camera_hardware_source_id'] = current_item.hardware_source_id

        camera_choice_row = ui.create_row_widget()
        settings_button = ui.create_push_button_widget('Settings...')
        settings_button.on_clicked = settings_button_clicked
        self.camera_choice_combo_box = ui.create_combo_box_widget(item_text_getter=lambda camera: getattr(camera, 'display_name'))
        self.camera_choice_combo_box.on_current_item_changed = camera_changed
        camera_choice_row.add_spacing(5)
        camera_choice_row.add(ui.create_label_widget('Camera: '))
        camera_choice_row.add(self.camera_choice_combo_box)
        camera_choice_row.add_stretch()
        camera_choice_row.add_spacing(5)
        camera_choice_row.add(settings_button)
        camera_choice_row.add_spacing(5)
        self.update_camera_list()
        self.update_current_camera()
        self.__settings_changed_event_listener = self.multi_acquire_controller.settings.settings_changed_event.listen(self.update_current_camera)
        def component_changed(component, component_types):
            if 'camera_hardware_source' in component_types:
                self.update_camera_list()
        self.__component_registered_event_listener = Registry.listen_component_registered_event(component_changed)
        self.__component_unregistered_event_listener = Registry.listen_component_unregistered_event(component_changed)

        change_parameters_row = ui.create_row_widget()
        change_parameters_row.add_spacing(5)
        change_parameters_row.add(ui.create_label_widget('MultiAcquire parameters:'))
        change_parameters_row.add_stretch()
        change_parameters_row.add_spacing(20)

        parameter_description_row = ui.create_row_widget()
        parameter_description_row.add_spacing(5)
        parameter_description_row.add(ui.create_label_widget('#'))
        parameter_description_row.add_spacing(20)
        parameter_description_row.add(ui.create_label_widget('X Offset (eV)'))
        parameter_description_row.add_spacing(25)
        parameter_description_row.add_stretch()
        parameter_description_row.add(ui.create_label_widget('Exposure (ms)'))
        parameter_description_row.add_spacing(25)
        parameter_description_row.add_stretch()
        parameter_description_row.add(ui.create_label_widget('Frames'))
        parameter_description_row.add_spacing(5)
        parameter_description_row.add_stretch()

        add_remove_parameters_row = ui.create_row_widget()
        add_parameters_button = ui.create_push_button_widget('+')
        add_parameters_button._widget.set_property('width', 40)
        add_parameters_button.on_clicked = self.multi_acquire_controller.add_spectrum
        remove_parameters_button = ui.create_push_button_widget('-')
        remove_parameters_button._widget.set_property('width', 40)
        remove_parameters_button.on_clicked = self.multi_acquire_controller.remove_spectrum

        add_remove_parameters_row.add_spacing(5)
        add_remove_parameters_row.add(add_parameters_button)
        add_remove_parameters_row.add_spacing(5)
        add_remove_parameters_row.add(remove_parameters_button)
        add_remove_parameters_row.add_spacing(20)
        add_remove_parameters_row.add_stretch()

        progress_row = ui.create_row_widget()
        progress_row.add_spacing(180)
        self.progress_bar = ui.create_progress_bar_widget()
        progress_row.add(self.progress_bar)
        progress_row.add_spacing(5)

        time_estimate_row = ui.create_row_widget()
        time_estimate_row.add_spacing(6)
        self.time_estimate_label = ui.create_label_widget()
        time_estimate_row.add(self.time_estimate_label)
        time_estimate_row.add_spacing(5)
        time_estimate_row.add_stretch()
        self.si_time_estimate_label = ui.create_label_widget()
        time_estimate_row.add(self.si_time_estimate_label)
        time_estimate_row.add_spacing(6)
        self.multi_acquire_controller.stem_controller = self.stem_controller
        self.multi_acquire_controller.scan_controller = self.scan_controller
        def frame_parameters_changed(profile_index, frame_parameters):
            self.update_time_estimate()
        if self.scan_controller:
            self.__scan_frame_parameters_changed_event_listener = self.scan_controller.frame_parameters_changed_event.listen(frame_parameters_changed)
        self.update_camera_list()

        self.start_button = ui.create_push_button_widget('Start Multi-Acquire')
        self.start_button.on_clicked = start_clicked
        self.start_si_button = ui.create_push_button_widget('Start Multi-Acquire spectrum image')
        self.start_si_button.on_clicked = start_si_clicked
        start_row = ui.create_row_widget()
        start_row.add_spacing(5)
        start_row.add(self.start_button)
        start_row.add_spacing(5)
        start_row.add_stretch()
        start_row.add(self.start_si_button)
        start_row.add_spacing(5)

        column = ui.create_column_widget()
        column.add_spacing(5)
        column.add(camera_choice_row)
        column.add_spacing(5)
        column.add(change_parameters_row)
        column.add_spacing(10)
        column.add(parameter_description_row)
        column.add_spacing(10)
        self.parameter_column = ui.create_column_widget()
        for spectrum_parameters in self.multi_acquire_controller.spectrum_parameters:
            line = self.create_parameter_line(spectrum_parameters)
            self.parameter_column.add(line)
        column.add(self.parameter_column)
        column.add_spacing(5)
        column.add(add_remove_parameters_row)
        column.add_spacing(5)
        column.add(progress_row)
        column.add_spacing(10)
        column.add(time_estimate_row)
        column.add_spacing(5)
        column.add(start_row)
        column.add_spacing(5)
        column.add_stretch()
        return column

    def _start_display_queue_thread(self) -> None:
        # private. used internally and for tests.
        # initialize any instance variables used for display queue thread.
        self.__data_processed_event = threading.Event()
        self.__data_processed_event.clear()
        self.result_data_items = dict()
        self.__display_queue = queue.Queue()
        self.__display_thread = threading.Thread(target=self.process_display_queue)
        self.__display_thread.start()

    def update_camera_list(self):
        cameras = list(Registry.get_components_by_type('camera_hardware_source'))
        self.camera_choice_combo_box.items = cameras
        self.update_current_camera()

    def update_current_camera(self):
        current_camera_name = self.multi_acquire_controller.settings['camera_hardware_source_id']
        for camera in self.camera_choice_combo_box.items:
            if camera.hardware_source_id == current_camera_name:
                break
        else:
            if self.camera_choice_combo_box.current_item:
                self.multi_acquire_controller.settings['camera_hardware_source_id'] = self.camera_choice_combo_box.current_item.hardware_source_id
            return
        self.camera_choice_combo_box.current_item = camera
        self.update_time_estimate()

    def update_parameter_line(self, spectrum_parameters):
        widgets = self.line_edit_widgets[spectrum_parameters['index']]
        widgets['offset_x'].text = '{:g}'.format(spectrum_parameters['offset_x'])
        widgets['exposure_ms'].text = '{:g}'.format(spectrum_parameters['exposure_ms'])
        widgets['frames'].text = '{:.0f}'.format(spectrum_parameters['frames'])

    def __format_time_string(self, acquisition_time):
        if acquisition_time > 3600:
            time_str = '{0:.1f} hours'.format(acquisition_time / 3600)
        elif acquisition_time > 90:
            time_str = '{0:.1f} minutes'.format(acquisition_time / 60)
        else:
            time_str = '{:.1f} seconds'.format(acquisition_time)
        return time_str

    def update_time_estimate(self):
        if self.time_estimate_label:
            acquisition_time, si_acquisition_time = self.multi_acquire_controller.get_total_acquisition_time()
            time_str = self.__format_time_string(acquisition_time)
            si_time_str = self.__format_time_string(si_acquisition_time)
            #def update():
            self.time_estimate_label.text = time_str
            self.si_time_estimate_label.text = si_time_str
            #self.__api.queue_task(update)

    def create_parameter_line(self, spectrum_parameters):
        row = self.ui.create_row_widget()
        column = self.ui.create_column_widget()
        widgets = {}

        index = self.ui.create_label_widget('{:g}'.format(spectrum_parameters['index']+1))
        offset_x = self.ui.create_line_edit_widget('{:g}'.format(spectrum_parameters['offset_x']))
        exposure_ms = self.ui.create_line_edit_widget('{:g}'.format(spectrum_parameters['exposure_ms']))
        frames = self.ui.create_line_edit_widget('{:.0f}'.format(spectrum_parameters['frames']))

        offset_x.on_editing_finished = lambda text: self.multi_acquire_controller.set_offset_x(spectrum_parameters['index'], float(text))
        exposure_ms.on_editing_finished = lambda text: self.multi_acquire_controller.set_exposure_ms(spectrum_parameters['index'], float(text))
        exposure_ms.on_editing_finished = lambda text: self.multi_acquire_controller.set_exposure_ms(spectrum_parameters['index'], float(text))
        frames.on_editing_finished = lambda text: self.multi_acquire_controller.set_frames(spectrum_parameters['index'], int(text))

        widgets['index'] = index
        widgets['offset_x'] = offset_x
        widgets['exposure_ms'] = exposure_ms
        widgets['frames'] = frames

        row.add_spacing(5)
        row.add(index)
        row.add_spacing(20)
        row.add(offset_x)
        row.add_stretch()
        row.add_spacing(25)
        row.add(exposure_ms)
        row.add_stretch()
        row.add_spacing(25)
        row.add(frames)
        row.add_stretch()
        row.add_spacing(5)

        self.line_edit_widgets[spectrum_parameters['index']] = widgets

        column.add(row)
        column.add_spacing(10)

        return column

    def show_config_box(self):
        dc = self.document_controller._document_controller

        class ConfigDialog(Dialog.ActionDialog):
            def __init__(self, ui, multi_eels_panel):
                super().__init__(ui, parent_window=dc)
                def report_window_close():
                    multi_eels_panel.settings_window_open = False
                self.multi_eels_panel = multi_eels_panel
                self.on_accept = report_window_close
                self.on_reject = report_window_close
                self.checkboxes = {}
                self.line_edits = {}
                self.__multi_eels_settings_changed_event_listener = multi_eels_panel.multi_acquire_controller.settings.settings_changed_event.listen(self.settings_changed)

                def x_shifter_finished(text):
                    newvalue = str(text)
                    multi_eels_panel.multi_acquire_controller.settings['x_shifter'] = newvalue

                def x_shift_delay_finished(text):
                    try:
                        newvalue = float(text)
                    except ValueError:
                        pass
                    else:
                        multi_eels_panel.multi_acquire_controller.settings['x_shift_delay'] = newvalue
                    finally:
                        x_shift_delay_field.text = '{:g}'.format(multi_eels_panel.multi_acquire_controller.settings['x_shift_delay'])

                def blanker_finished(text):
                    newvalue = str(text)
                    multi_eels_panel.multi_acquire_controller.settings['blanker'] = newvalue

                def blanker_delay_finished(text):
                    try:
                        newvalue = float(text)
                    except ValueError:
                        pass
                    else:
                        multi_eels_panel.multi_acquire_controller.settings['blanker_delay'] = newvalue
                    finally:
                        blanker_delay_field.text = '{:g}'.format(multi_eels_panel.multi_acquire_controller.settings['blanker_delay'])

                def auto_dark_subtract_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['auto_dark_subtract'] = check_state == 'checked'

                def bin_1D_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['bin_spectra'] = check_state == 'checked'

                def sum_frames_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['sum_frames'] = check_state == 'checked'

                column = self.ui.create_column_widget()
                row1 = self.ui.create_row_widget()
                row2 = self.ui.create_row_widget()
                row3 = self.ui.create_row_widget()
                row4 = self.ui.create_row_widget()

                x_shifter_label = self.ui.create_label_widget('X-shift control name: ')
                x_shifter_field = self.ui.create_line_edit_widget(properties={'min-width': 120})
                x_shift_delay_label = self.ui.create_label_widget('X shifter delay (s): ')
                x_shift_delay_field = self.ui.create_line_edit_widget(properties={'min-width': 40})
                auto_dark_subtract_checkbox = self.ui.create_check_box_widget('Auto dark subtraction ')
                bin_1D_checkbox = self.ui.create_check_box_widget('Bin data in y direction ')
                sum_frames_checkbox = self.ui.create_check_box_widget('Sum frames')
                blanker_label = self.ui.create_label_widget('Blanker control name: ')
                blanker_field = self.ui.create_line_edit_widget(properties={'min-width': 120})
                blanker_delay_label = self.ui.create_label_widget('Blanker delay (s): ')
                blanker_delay_field = self.ui.create_line_edit_widget(properties={'min-width': 40})

                row1.add_spacing(5)
                row1.add(x_shifter_label)
                row1.add(x_shifter_field)
                row1.add_spacing(10)
                row1.add(x_shift_delay_label)
                row1.add(x_shift_delay_field)
                row1.add_spacing(5)
                row1.add_stretch()

                row3.add_spacing(5)
                row3.add(blanker_label)
                row3.add(blanker_field)
                row3.add_spacing(10)
                row3.add(blanker_delay_label)
                row3.add(blanker_delay_field)
                row3.add_stretch()
                row3.add_spacing(5)

                row4.add_spacing(5)
                row4.add(bin_1D_checkbox)
                row4.add_spacing(20)
                row4.add(auto_dark_subtract_checkbox)
                row4.add_spacing(20)
                row4.add(sum_frames_checkbox)
                row4.add_spacing(5)
                row4.add_stretch()

                column.add(row1)
                column.add_spacing(5)
                column.add(row2)
                column.add_spacing(5)
                column.add(row3)
                column.add_spacing(5)
                column.add(row4)
                column.add_stretch()

                self.content.add_spacing(5)
                self.content.add(column)
                self.content.add_spacing(5)

                auto_dark_subtract_checkbox.on_check_state_changed = auto_dark_subtract_checkbox_changed
                bin_1D_checkbox.on_check_state_changed = bin_1D_checkbox_changed
                sum_frames_checkbox.on_check_state_changed = sum_frames_checkbox_changed
                x_shifter_field.on_editing_finished = x_shifter_finished
                x_shift_delay_field.on_editing_finished = x_shift_delay_finished
                blanker_field.on_editing_finished = blanker_finished
                blanker_delay_field.on_editing_finished = blanker_delay_finished

                self.line_edits.update({'x_shifter': x_shifter_field,
                                        'x_shift_delay': x_shift_delay_field,
                                        'blanker': blanker_field,
                                        'blanker_delay': blanker_delay_field})

                self.checkboxes.update({'auto_dark_subtract': auto_dark_subtract_checkbox,
                                        'bin_spectra': bin_1D_checkbox,
                                        'sum_frames': sum_frames_checkbox})

                self.settings_changed()

            def about_to_close(self, geometry: str, state: str) -> None:
                if self.on_reject:
                    self.on_reject()
                super().about_to_close(geometry, state)
                self.checkboxes = {}
                self.line_edits = {}
                self.__multi_eels_settings_changed_event_listener.close()
                self.__multi_eels_settings_changed_event_listener = None

            def settings_changed(self):
                for key, value in self.multi_eels_panel.multi_acquire_controller.settings.items():
                    if key in self.checkboxes:
                        self.checkboxes[key].checked = value
                    elif key in self.line_edits:
                        if type(value) in (int, float):
                            self.line_edits[key].text = '{:g}'.format(value)
                        else:
                            self.line_edits[key].text = '{}'.format(str(value))

        ConfigDialog(dc.ui, self).show()

class MultiAcquireExtension:
    extension_id = 'nion.extension.multiacquire'

    def __init__(self, api_broker):
        api = api_broker.get_api(version='1', ui_version='1')
        self.__panel_ref = api.create_panel(MultiAcquirePanelDelegate(api))

    def close(self):
        self.__panel_ref.close()
        self.__panel_ref = None
