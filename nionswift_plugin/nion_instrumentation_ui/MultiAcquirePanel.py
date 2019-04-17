# standard libraries
import itertools
import numpy
import queue
import threading

# local libraries
from nion.instrumentation import MultiAcquire
from nion.swift.model import ImportExportManager
from nion.ui import Dialog
from nion.utils import Registry


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
        self.__data_cache = numpy.empty(final_data_shape, dtype=dtype)
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
        self.line_edit_widgets = {}
        self.push_button_widgets = {}
        self.multi_acquire_controller = MultiAcquire.MultiAcquireController()
        self.__acquisition_state_changed_event_listener = self.multi_acquire_controller.acquisition_state_changed_event.listen(self.acquisition_state_changed)
        self.__multi_eels_parameters_changed_event_listener = self.multi_acquire_controller.spectrum_parameters.parameters_changed_event.listen(self.spectrum_parameters_changed)
        self.__progress_updated_event_listener = self.multi_acquire_controller.progress_updated_event.listen(self.update_progress_bar)
        self.__settings_changed_event_listener = None
        self.__component_registered_event_listener = None
        self.__component_unregistered_event_listener = None
        self.__new_data_ready_event_listener = None
        self.stem_controller = None
        self.eels_camera = None
        self.superscan = None
        self.settings_window_open = False
        self.parameters_window_open = False
        self.parameter_column = None
        self.result_data_items = {}
        self.__result_data_items_refs = []
        self.__acquisition_running = False
        self.__display_queue = queue.Queue()
        self.__display_thread = None
        self.__acquisition_thread = None
        self.__data_processed_event = threading.Event()

    def spectrum_parameters_changed(self):
        parameter_list = self.multi_acquire_controller.spectrum_parameters.copy()
        if len(parameter_list) != len(self.parameter_column._widget.children):
            self.parameter_column._widget.remove_all()
            for spectrum_parameters in parameter_list:
                self.parameter_column.add(self.create_parameter_line(spectrum_parameters))
        else:
            for spectrum_parameters in parameter_list:
                self.update_parameter_line(spectrum_parameters)

    def create_result_data_item(self, data_dict):
        if data_dict.get('stitched_data'):
            data_item = self.document_controller.library.create_data_item_from_data_and_metadata(data_dict['data'][0],
                                                                                          title='Multi-Acquire (stitched)')
            metadata = data_item.metadata
            metadata['MultiAcquire'] = data_dict['parameters']
            data_item.set_metadata(metadata)
            self.document_controller.display_data_item(data_item)
        else:
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
        for item in self.result_data_items.values():
            item.exit_write_suspend_state()
#        for item in self.result_data_items:
#            try:
#                while True:
#                    item.xdata.decrement_data_ref_count()
#            except AssertionError:
#                pass
        self.__new_data_ready_event_listener = None
        self.result_data_items = {}

    def add_to_display_queue(self, data_dict):
        self.__display_queue.put(data_dict)

    def process_display_queue(self):
        while True:
            try:
                data_dict = self.__display_queue.get(timeout=1)
            except queue.Empty:
                if self.__data_processed_event.is_set() and not self.__acquisition_running:
                    self.__close_data_item_refs()
                    break
            else:
                index = data_dict['parameters']['index']
                line_number = data_dict['parameters']['line_number']
                data_element = data_dict['data_element']
                line_data = data_element['data']
                start_ev = data_dict['parameters']['start_ev']
                end_ev = data_dict['parameters']['end_ev']
                number_frames = data_dict['parameters']['frames']
                exposure_ms = data_dict['parameters']['exposure_ms']
                print('got data from display queue')
                if not self.result_data_items.get(index):
                    print('creating new data item')
                    spatial_calibrations = data_element['spatial_calibrations']
                    data_element['data'] = line_data[numpy.newaxis, ...]
                    data_element['collection_dimension_count'] = 2
                    data_element['spatial_calibrations'] = self.multi_acquire_controller.scan_calibrations[0:1] + spatial_calibrations
                    metadata = data_element.get('metadata', {})
                    metadata['MultiAcquire.parameters'] = data_dict['parameters']
                    metadata['MultiAcquire.settings'] = data_dict['settings']
                    data_element['metadata'] = data_dict['parameters']
                    new_xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_element)
                    title = ('MultiAcquire (stitched)' if data_dict.get('stitched_data') else
                             'MultiAcquire #{:d}, {:g}-{:g} eV, {:g}x{:g} ms'.format(index+1, start_ev, end_ev,
                                                                                     number_frames, exposure_ms))
                    data_item_ready_event = threading.Event()
                    new_data_item = None
                    def create_data_item():
                        nonlocal new_data_item
                        # we have to create the initial data item with some data that has more than 2 dimensions
                        # otherwise Swift does not use HDF5 and we will have a problem if it grows too big later
                        new_data_item = self.__api.library.create_data_item_from_data_and_metadata(new_xdata,
                                                                                                   title=title)
                        data_item_ready_event.set()
                    self.__api.queue_task(create_data_item)
                    data_item_ready_event.wait()
                    number_lines = data_dict['parameters'].get('number_lines', self.multi_acquire_controller.number_lines)
                    max_shape = (number_lines,) + line_data.shape
                    new_appendable_data_item = AppendableDataItemFixedSize(new_data_item, max_shape, self.__api)
                    new_appendable_data_item.enter_write_suspend_state()
                    self.result_data_items[index] = new_appendable_data_item
                    # add the line we have already in our data item to the appendable data item to have them in the
                    # same state
                    self.result_data_items[index].add_data((line_number,...), line_data)
                    self.__api.queue_task(
                            lambda: self.result_data_items[index].get_partial_data_item((slice(0, line_number+1),
                                                                                         ...)))
                    del new_xdata
                    del new_appendable_data_item
                else:
                    self.result_data_items[index].add_data((line_number,...), line_data)
                    self.__api.queue_task(
                            lambda: self.result_data_items[index].get_partial_data_item((slice(0, line_number+1),
                                                                                         ...)))
                del line_data
                del data_element
                del data_dict
                self.__display_queue.task_done()
                print('displayed line {:.0f}'.format(line_number))

    def update_progress_bar(self, minimum, maximum, value):
        if self.progress_bar:
            def update():
                self.progress_bar.minimum = minimum
                self.progress_bar.maximum = maximum
                self.progress_bar.value = value
            self.__api.queue_task(update)

    def create_panel_widget(self, ui, document_controller):
        self.ui = ui
        self.document_controller = document_controller

        def start_clicked():
            if self.__acquisition_running:
                self.multi_acquire_controller.cancel()
            else:
                self.stem_controller = Registry.get_component('stem_controller')
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
                self.stem_controller = Registry.get_component('stem_controller')
                self.superscan = self.stem_controller.scan_controller
                self.multi_acquire_controller.stem_controller = self.stem_controller
                self.multi_acquire_controller.camera = self.camera_choice_combo_box.current_item
                self.multi_acquire_controller.superscan = self.superscan
                self.result_data_items = {}
                self.__new_data_ready_event_listener = self.multi_acquire_controller.new_data_ready_event.listen(self.add_to_display_queue)
                self.__data_processed_event.clear()
                self.__display_thread = threading.Thread(target=self.process_display_queue)
                self.__display_thread.start()
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
        parameter_description_row.add_spacing(5)
        parameter_description_row.add_stretch()
        parameter_description_row.add(ui.create_label_widget('X Offset (eV)'))
        parameter_description_row.add_spacing(5)
        parameter_description_row.add_stretch()
        parameter_description_row.add(ui.create_label_widget('Y Offset (px)'))
        parameter_description_row.add_spacing(5)
        parameter_description_row.add_stretch()
        parameter_description_row.add(ui.create_label_widget('Exposure (ms)'))
        parameter_description_row.add_spacing(5)
        parameter_description_row.add_stretch()
        parameter_description_row.add(ui.create_label_widget('Frames'))
        parameter_description_row.add_spacing(5)
        parameter_description_row.add_stretch()

        add_remove_parameters_row = ui.create_row_widget()
        add_parameters_button = ui.create_push_button_widget('+')
        add_parameters_button.on_clicked = self.multi_acquire_controller.add_spectrum
        remove_parameters_button = ui.create_push_button_widget('-')
        remove_parameters_button.on_clicked = self.multi_acquire_controller.remove_spectrum

        add_remove_parameters_row.add_spacing(5)
        add_remove_parameters_row.add(add_parameters_button)
        add_remove_parameters_row.add_spacing(5)
        add_remove_parameters_row.add(remove_parameters_button)
        add_remove_parameters_row.add_spacing(20)
        add_remove_parameters_row.add_stretch()

        progress_row = ui.create_row_widget()
        progress_row.add_spacing(5)
        progress_row.add_stretch()
        self.progress_bar = ui.create_progress_bar_widget()
        progress_row.add(self.progress_bar)
        progress_row.add_spacing(5)

        self.start_button = ui.create_push_button_widget('Start Multi-Acquire')
        self.start_button.on_clicked = start_clicked
        self.start_si_button = ui.create_push_button_widget('Start Multi-Acquire spectrum image')
        self.start_si_button.on_clicked = start_si_clicked
        start_row = ui.create_row_widget()
        start_row.add_spacing(5)
        start_row.add(self.start_button)
        start_row.add_spacing(15)
        #start_row.add(self.start_si_button)
        #start_row.add(ui.create_label_widget('COMING SOON: Multi-Acquire Spectrum Imaging'))
        start_row.add_spacing(5)
        start_row.add_stretch()

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
        column.add(start_row)
        column.add_spacing(5)
        column.add_stretch()
        return column

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

    def update_parameter_line(self, spectrum_parameters):
        widgets = self.line_edit_widgets[spectrum_parameters['index']]
        widgets['offset_x'].text = '{:g}'.format(spectrum_parameters['offset_x'])
        widgets['offset_y'].text = '{:g}'.format(spectrum_parameters['offset_y'])
        widgets['exposure_ms'].text = '{:g}'.format(spectrum_parameters['exposure_ms'])
        widgets['frames'].text = '{:.0f}'.format(spectrum_parameters['frames'])

    def create_parameter_line(self, spectrum_parameters):
        row = self.ui.create_row_widget()
        column = self.ui.create_column_widget()
        widgets = {}

        index = self.ui.create_label_widget('{:g}'.format(spectrum_parameters['index']+1))
        offset_x = self.ui.create_line_edit_widget('{:g}'.format(spectrum_parameters['offset_x']))
        offset_y = self.ui.create_line_edit_widget('{:g}'.format(spectrum_parameters['offset_y']))
        exposure_ms = self.ui.create_line_edit_widget('{:g}'.format(spectrum_parameters['exposure_ms']))
        frames = self.ui.create_line_edit_widget('{:.0f}'.format(spectrum_parameters['frames']))

        offset_x.on_editing_finished = lambda text: self.multi_acquire_controller.set_offset_x(spectrum_parameters['index'], float(text))
        offset_y.on_editing_finished = lambda text: self.multi_acquire_controller.set_offset_y(spectrum_parameters['index'], float(text))
        exposure_ms.on_editing_finished = lambda text: self.multi_acquire_controller.set_exposure_ms(spectrum_parameters['index'], float(text))
        exposure_ms.on_editing_finished = lambda text: self.multi_acquire_controller.set_exposure_ms(spectrum_parameters['index'], float(text))
        frames.on_editing_finished = lambda text: self.multi_acquire_controller.set_frames(spectrum_parameters['index'], int(text))

        widgets['index'] = index
        widgets['offset_x'] = offset_x
        widgets['offset_y'] = offset_y
        widgets['exposure_ms'] = exposure_ms
        widgets['frames'] = frames

        row.add_spacing(5)
        row.add(index)
        row.add_spacing(10)
        row.add(offset_x)
        row.add_spacing(10)
        row.add(offset_y)
        row.add_spacing(10)
        row.add(exposure_ms)
        row.add_spacing(10)
        row.add(frames)
        row.add_spacing(5)

        self.line_edit_widgets[spectrum_parameters['index']] = widgets

        column.add(row)
        column.add_spacing(10)

        return column

    def show_config_box(self):
        dc = self.document_controller._document_controller

        class ConfigDialog(Dialog.ActionDialog):
            def __init__(self, ui, multi_eels_panel):
                super(ConfigDialog, self).__init__(ui)
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

                def x_shift_strength_finished(text):
                    try:
                        newvalue = float(text)
                    except ValueError:
                        pass
                    else:
                        multi_eels_panel.multi_acquire_controller.settings['x_units_per_ev'] = newvalue
                    finally:
                        x_shift_strength_field.text = '{:g}'.format(multi_eels_panel.multi_acquire_controller.settings['x_units_per_ev'])

                def y_shifter_finished(text):
                    newvalue = str(text)
                    multi_eels_panel.multi_acquire_controller.settings['y_shifter'] = newvalue

                def y_shift_strength_finished(text):
                    try:
                        newvalue = float(text)
                    except ValueError:
                        pass
                    else:
                        multi_eels_panel.multi_acquire_controller.settings['y_units_per_px'] = newvalue
                    finally:
                        y_shift_strength_field.text = '{:g}'.format(multi_eels_panel.multi_acquire_controller.settings['y_units_per_px'])

                def x_shift_delay_finished(text):
                    try:
                        newvalue = float(text)
                    except ValueError:
                        pass
                    else:
                        multi_eels_panel.multi_acquire_controller.settings['x_shift_delay'] = newvalue
                    finally:
                        x_shift_delay_field.text = '{:g}'.format(multi_eels_panel.multi_acquire_controller.settings['x_shift_delay'])

                def y_shift_delay_finished(text):
                    try:
                        newvalue = float(text)
                    except ValueError:
                        pass
                    else:
                        multi_eels_panel.multi_acquire_controller.settings['y_shift_delay'] = newvalue
                    finally:
                        y_shift_delay_field.text = '{:g}'.format(multi_eels_panel.multi_acquire_controller.settings['y_shift_delay'])

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

                def align_y_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['y_align'] = check_state == 'checked'

                def auto_dark_subtract_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['auto_dark_subtract'] = check_state == 'checked'

                def bin_1D_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['bin_spectra'] = check_state == 'checked'

                def sum_frames_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['sum_frames'] = check_state == 'checked'

                def saturation_value_finished(text):
                    try:
                        newvalue = float(text)
                    except ValueError:
                        pass
                    else:
                        multi_eels_panel.multi_acquire_controller.settings['saturation_value'] = newvalue
                    finally:
                        saturation_value_field.text = '{:g}'.format(multi_eels_panel.multi_acquire_controller.settings['saturation_value'])

                column = self.ui.create_column_widget()
                row1 = self.ui.create_row_widget()
                row2 = self.ui.create_row_widget()
                row3 = self.ui.create_row_widget()
                row4 = self.ui.create_row_widget()

                x_shifter_label = self.ui.create_label_widget('X-shift control name: ')
                x_shifter_field = self.ui.create_line_edit_widget()
                x_shift_strength_label = self.ui.create_label_widget('X shifter strength (units/ev): ')
                x_shift_strength_field = self.ui.create_line_edit_widget()
                x_shift_delay_label = self.ui.create_label_widget('X shifter delay (s): ')
                x_shift_delay_field = self.ui.create_line_edit_widget()
                y_shifter_label = self.ui.create_label_widget('Y-shift control name: ')
                y_shifter_field = self.ui.create_line_edit_widget()
                y_shift_strength_label = self.ui.create_label_widget('Y shifter strength (units/px): ')
                y_shift_strength_field = self.ui.create_line_edit_widget()
                y_shift_delay_label = self.ui.create_label_widget('Y shifter delay (s): ')
                y_shift_delay_field = self.ui.create_line_edit_widget()
                align_y_checkbox = self.ui.create_check_box_widget('Y-align spectra ')
                auto_dark_subtract_checkbox = self.ui.create_check_box_widget('Auto dark subtraction ')
                bin_1D_checkbox = self.ui.create_check_box_widget('Bin data in y direction ')
                sum_frames_checkbox = self.ui.create_check_box_widget('Sum frames')
                saturation_value_label = self.ui.create_label_widget('Camera saturation value: ')
                saturation_value_field = self.ui.create_line_edit_widget()
                blanker_label = self.ui.create_label_widget('Blanker control name: ')
                blanker_field = self.ui.create_line_edit_widget()
                blanker_delay_label = self.ui.create_label_widget('Blanker delay (s): ')
                blanker_delay_field = self.ui.create_line_edit_widget()

                row1.add_spacing(5)
                row1.add(x_shifter_label)
                row1.add(x_shifter_field)
                row1.add_spacing(5)
                row1.add(x_shift_strength_label)
                row1.add(x_shift_strength_field)
                row1.add_spacing(5)
                row1.add(x_shift_delay_label)
                row1.add(x_shift_delay_field)
                row1.add_spacing(5)
                row1.add_stretch()

                row2.add_spacing(5)
                row2.add(y_shifter_label)
                row2.add(y_shifter_field)
                row2.add_spacing(5)
                row2.add(y_shift_strength_label)
                row2.add(y_shift_strength_field)
                row2.add_spacing(5)
                row2.add(y_shift_delay_label)
                row2.add(y_shift_delay_field)
                row2.add_spacing(5)
                row2.add_stretch()

                row3.add_spacing(5)
                row3.add(blanker_label)
                row3.add(blanker_field)
                row3.add_spacing(5)
                row3.add_stretch()
                row3.add(blanker_delay_label)
                row3.add(blanker_delay_field)
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
                y_shifter_field.on_editing_finished = y_shifter_finished
                x_shift_strength_field.on_editing_finished = x_shift_strength_finished
                y_shift_strength_field.on_editing_finished = y_shift_strength_finished
                x_shift_delay_field.on_editing_finished = x_shift_delay_finished
                y_shift_delay_field.on_editing_finished = y_shift_delay_finished
                blanker_field.on_editing_finished = blanker_finished
                blanker_delay_field.on_editing_finished = blanker_delay_finished

                self.line_edits.update({'x_shifter': x_shifter_field,
                                        'x_units_per_ev': x_shift_strength_field,
                                        'x_shift_delay': x_shift_delay_field,
                                        'y_shifter': y_shifter_field,
                                        'y_units_per_px': y_shift_strength_field,
                                        'y_shift_delay': y_shift_delay_field,
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