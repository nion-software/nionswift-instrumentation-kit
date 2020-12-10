# standard libraries
import itertools
import numpy
import threading
import os
import gettext
import uuid
import pkgutil

# local libraries
import typing
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation import camera_base
from nion.instrumentation import MultiAcquire
from nion.swift.model import ImportExportManager
from nion.ui import Dialog
from nion.utils import Registry
from nion.ui import CanvasItem

_ = gettext.gettext


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


class MultiAcquirePanelDelegate:

    def __init__(self, api):
        self.__api = api
        self.panel_id = 'MultiAcquire-Panel'
        self.panel_name = 'MultiAcquire'
        self.panel_positions = ['left', 'right']
        self.panel_position = 'right'
        self.api = api
        self.line_edit_widgets = None
        self.multi_acquire_controller = None
        self.__acquisition_state_changed_event_listener = None
        self.__multi_eels_parameters_changed_event_listener = None
        self.__progress_updated_event_listener = None
        self.__settings_changed_event_listener = None
        self.__settings_changed_event_listener_2 = None
        self.__component_registered_event_listener = None
        self.__component_unregistered_event_listener = None
        self.__scan_frame_parameters_changed_event_listener = None
        self.__new_data_ready_event_listener = None
        self._stem_controller: typing.Optional[stem_controller.STEMController] = None
        self._camera: typing.Optional[camera_base.CameraHardwareSource] = None
        self._scan_controller: typing.Optional[scan_base.ScanHardwareSource] = None
        self.settings_window_open = False
        self.parameter_column = None
        self.__acquisition_running = False
        self.__display_queue = None
        self.__display_thread = None
        self.__acquisition_thread = None
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

    @property
    def camera(self) -> typing.Optional[camera_base.CameraHardwareSource]:
        if hasattr(self, 'camera_choice_combo_box'):
            return self.camera_choice_combo_box.current_item
        if self._camera is None and self.stem_controller:
            self._camera = self.stem_controller.eels_camera
        return self._camera

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
        if self.__settings_changed_event_listener_2:
            self.__settings_changed_event_listener_2.close()
            self.__settings_changed_event_listener_2 = None
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
        reset_color_cycle()
        display_item = None
        sorted_indices = numpy.argsort([parms['start_ev'] for parms in data_dict['parameter_list']])
        display_layer_index = 0
        for i in sorted_indices:
            index = data_dict['parameter_list'][i]['index']
            xdata = ImportExportManager.convert_data_element_to_data_and_metadata(data_dict['data_element_list'][i])
            start_ev = data_dict['parameter_list'][i]['start_ev']
            end_ev = data_dict['parameter_list'][i]['end_ev']
            number_frames = data_dict['parameter_list'][i]['frames']
            exposure_ms = data_dict['parameter_list'][i]['exposure_ms']
            summed = ' (summed)' if not data_dict['data_element_list'][i].get('is_sequence', False) and number_frames > 1 else ''
            data_item = None
            if i == sorted_indices[0] and xdata.datum_dimension_count == 1 and data_dict['settings_list'][i]['use_multi_eels_calibration']:
                data_item = self.document_controller.library.create_data_item_from_data_and_metadata(
                                                                    xdata,
                                                                    title='MultiAcquire (stacked)')
                display_item = self.__api.library._document_model.get_display_item_for_data_item(data_item._data_item)
                display_layer_index += 1  # display item has one display layer already
            #else:
            units_str = ' eV' if data_dict['settings_list'][i]['use_multi_eels_calibration'] else ''
            new_data_item = self.document_controller.library.create_data_item_from_data_and_metadata(
                                xdata,
                                title='MultiAcquire #{:d}, {:g}-{:g}{}, {:g}x{:g} ms{}'.format(index+1,
                                                                                               start_ev,
                                                                                               end_ev,
                                                                                               units_str,
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
                display_layer_label = '#{:d}: {:g}-{:g}{}, {:g}x{:g} ms{}'.format(index + 1, start_ev, end_ev,
                                                                                  units_str,
                                                                                  number_frames, exposure_ms, summed)
                display_item._set_display_layer_properties(display_layer_index, label=display_layer_label,
                                                           stroke_color=get_next_color(), fill_color=None)
                display_layer_index += 1
        if display_item:
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
            def update_buttons():
                self.start_si_button.text = 'Start Multi-Acquire spectrum image'
                self.start_button.text = 'Start Multi-Acquire'
                self.start_button._widget.enabled = True
                self.start_si_button._widget.enabled = True
            self.__api.queue_task(update_buttons)

        if self.__new_data_ready_event_listener:
            self.__new_data_ready_event_listener.close()
        self.__new_data_ready_event_listener = None
        self.result_data_items = None

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
        self._stem_controller = None
        self._camera = None
        self._scan_controller = None
        self.multi_acquire_controller = MultiAcquire.MultiAcquireController(self.stem_controller, savepath=os.path.join(os.path.expanduser('~'), 'MultiAcquire'))
        self.__acquisition_state_changed_event_listener = self.multi_acquire_controller.acquisition_state_changed_event.listen(self.acquisition_state_changed)
        self.__multi_eels_parameters_changed_event_listener = self.multi_acquire_controller.spectrum_parameters.parameters_changed_event.listen(self.spectrum_parameters_changed)
        self.__progress_updated_event_listener = self.multi_acquire_controller.progress_updated_event.listen(self.update_progress_bar)
        self.__settings_changed_event_listener = None
        self.__component_registered_event_listener = None
        self.__component_unregistered_event_listener = None
        self.__scan_frame_parameters_changed_event_listener = None
        self.__new_scan_data_ready_event_listener = None

        self.settings_window_open = False
        self.parameter_column = None
        self.result_data_items = None
        self.__acquisition_running = False
        self.__display_queue = None
        self.__display_thread = None
        self.__acquisition_thread = None

        self.ui = ui
        self.document_controller = document_controller

        def start_clicked():
            if self.__acquisition_running:
                self.multi_acquire_controller.cancel()
            else:
                self.multi_acquire_controller.stem_controller = self.stem_controller
                self.multi_acquire_controller.camera = self.camera
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
                # Camera must be accessed from the UI thread, so do it here and re-use later
                camera = self.camera
                self.multi_acquire_controller.stem_controller = self.stem_controller
                self.multi_acquire_controller.camera = camera
                self.multi_acquire_controller.scan_controller = self.scan_controller

                def create_acquisition_handler(multi_acquire_parameters: list, current_parameters_index: int, multi_acquire_settings: dict):
                    document_model = self.document_controller._document_controller.document_model
                    camera_frame_parameters = camera.get_current_frame_parameters()
                    scan_frame_parameters = self.scan_controller.get_current_frame_parameters()
                    camera_frame_parameters['exposure_ms'] = multi_acquire_parameters[current_parameters_index]['exposure_ms']
                    camera_frame_parameters['processing'] = 'sum_project' if multi_acquire_settings['bin_spectra'] else None
                    scan_frame_parameters.setdefault('scan_id', str(uuid.uuid4()))
                    grab_synchronized_info = self.scan_controller.grab_synchronized_get_info(scan_frame_parameters=scan_frame_parameters,
                                                                                             camera=camera,
                                                                                             camera_frame_parameters=camera_frame_parameters)
                    camera_data_channel = None
                    scan_data_channel = None
                    channels_ready_event = threading.Event()

                    def create_channels():
                        nonlocal camera_data_channel, scan_data_channel
                        camera_data_channel = MultiAcquire.CameraDataChannel(document_model, camera.display_name, grab_synchronized_info,
                                                                                     multi_acquire_parameters, multi_acquire_settings, current_parameters_index)
                        enabled_channels = self.scan_controller.get_enabled_channels()
                        enabled_channel_names = [self.scan_controller.data_channels[i].name for i in enabled_channels]
                        scan_data_channel = MultiAcquire.ScanDataChannel(document_model, enabled_channel_names, grab_synchronized_info,
                                                                         multi_acquire_parameters, multi_acquire_settings, current_parameters_index)
                        camera_data_channel.start()
                        scan_data_channel.start()
                        channels_ready_event.set()

                    self.document_controller.queue_task(create_channels)
                    assert channels_ready_event.wait(10)

                    si_sequence_behavior = MultiAcquire.SISequenceBehavior(None, None, None, None)
                    handler =  MultiAcquire.SISequenceAcquisitionHandler(camera, camera_data_channel, camera_frame_parameters,
                                                                         self.scan_controller, scan_data_channel, scan_frame_parameters,
                                                                         si_sequence_behavior)

                    listener = handler.camera_data_channel.progress_updated_event.listen(self.multi_acquire_controller.set_progress_counter)

                    def finish_fn():
                        listener.close()
                        def close_channels():
                            handler.camera_data_channel.stop()
                            handler.scan_data_channel.stop()
                        self.document_controller.queue_task(close_channels)

                    handler.finish_fn = finish_fn

                    return handler

                self.__acquisition_thread = threading.Thread(target=self.multi_acquire_controller.start_multi_acquire_spectrum_image, args=(create_acquisition_handler,))
                self.__acquisition_thread.start()


        def settings_button_clicked():
            if not self.settings_window_open:
                self.settings_window_open = True
                self.show_config_box()

        def camera_changed(current_item):
            if current_item:
                self.multi_acquire_controller.settings['camera_hardware_source_id'] = current_item.hardware_source_id

        def binning_changed(current_item):
            if current_item == 'Spectra':
                self.multi_acquire_controller.settings['bin_spectra'] = True
                self.multi_acquire_controller.settings['use_multi_eels_calibration'] = False
            elif current_item == 'Images':
                self.multi_acquire_controller.settings['bin_spectra'] = False
                self.multi_acquire_controller.settings['use_multi_eels_calibration'] = False
            elif current_item == 'MultiEELS Spectra':
                self.multi_acquire_controller.settings['bin_spectra'] = True
                self.multi_acquire_controller.settings['use_multi_eels_calibration'] = True

        camera_choice_row = ui.create_row_widget()
        self.binning_choice_combo_box = ui.create_combo_box_widget()
        self.binning_choice_combo_box.on_current_item_changed = binning_changed
        self.binning_choice_combo_box.items = ['Spectra', 'Images', 'MultiEELS Spectra']
        settings_button = CanvasItem.BitmapButtonCanvasItem(CanvasItem.load_rgba_data_from_bytes(pkgutil.get_data(__name__, "resources/sliders_icon_24.png"), "png"))
        settings_widget = ui._ui.create_canvas_widget(properties={"height": 24, "width": 24})
        settings_widget.canvas_item.add_canvas_item(settings_button)
        settings_button.on_button_clicked = settings_button_clicked
        self.camera_choice_combo_box = ui.create_combo_box_widget(item_text_getter=lambda camera: getattr(camera, 'display_name'))
        self.camera_choice_combo_box.on_current_item_changed = camera_changed
        camera_choice_row.add_spacing(5)
        camera_choice_row.add(self.camera_choice_combo_box)
        camera_choice_row.add_spacing(10)
        camera_choice_row.add(self.binning_choice_combo_box)
        camera_choice_row.add_stretch()
        camera_choice_row.add_spacing(10)
        camera_choice_row._widget.add(settings_widget)
        camera_choice_row.add_spacing(10)
        self.update_camera_list()
        self.update_current_camera()
        self.__settings_changed_event_listener = self.multi_acquire_controller.settings.settings_changed_event.listen(self.update_current_camera)
        self.__settings_changed_event_listener_2 = self.multi_acquire_controller.settings.settings_changed_event.listen(self.update_binning_combo_box)
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
        parameter_description_row.add(ui.create_label_widget('Offset'))
        parameter_description_row.add_spacing(36)
        parameter_description_row.add_stretch()
        parameter_description_row.add(ui.create_label_widget('Exposure (ms)'))
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
        add_remove_parameters_row.add_spacing(90)
        self.progress_bar = ui.create_progress_bar_widget()
        add_remove_parameters_row.add(self.progress_bar)
        add_remove_parameters_row.add_spacing(5)

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

        self.start_button = ui.create_push_button_widget('Start MultiAcquire')
        self.start_button.on_clicked = start_clicked
        self.start_si_button = ui.create_push_button_widget('Start MultiAcquire spectrum image')
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
        column.add_spacing(10)
        column.add(change_parameters_row)
        column.add_spacing(5)
        column.add(parameter_description_row)
        column.add_spacing(10)
        self.parameter_column = ui.create_column_widget()
        for spectrum_parameters in self.multi_acquire_controller.spectrum_parameters:
            line = self.create_parameter_line(spectrum_parameters)
            self.parameter_column.add(line)
        column.add(self.parameter_column)
        column.add_spacing(5)
        column.add(add_remove_parameters_row)
        column.add_spacing(15)
        column.add(time_estimate_row)
        column.add_spacing(5)
        column.add(start_row)
        column.add_spacing(10)
        column.add_stretch()
        # Make sure the binning combo box shows the actual settings
        self.update_binning_combo_box()
        return column

    def update_binning_combo_box(self):
        current_item = self.binning_choice_combo_box.current_item
        new_item = current_item
        if self.multi_acquire_controller.settings['bin_spectra'] and self.multi_acquire_controller.settings['use_multi_eels_calibration']:
            new_item = 'MultiEELS Spectra'
        elif self.multi_acquire_controller.settings['bin_spectra']:
            new_item = 'Spectra'
        else:
            new_item = 'Images'
        if new_item != current_item:
            self.binning_choice_combo_box.current_item = new_item

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

                def sum_frames_checkbox_changed(check_state):
                    multi_eels_panel.multi_acquire_controller.settings['sum_frames'] = check_state == 'checked'

                column = self.ui.create_column_widget()
                row1 = self.ui.create_row_widget()
                row2 = self.ui.create_row_widget()
                row3 = self.ui.create_row_widget()

                x_shifter_label = self.ui.create_label_widget('Offset control name: ')
                x_shifter_field = self.ui.create_line_edit_widget(properties={'min-width': 160})
                x_shift_delay_label = self.ui.create_label_widget('Offset delay (s): ')
                x_shift_delay_field = self.ui.create_line_edit_widget(properties={'min-width': 40})
                auto_dark_subtract_checkbox = self.ui.create_check_box_widget('Auto dark subtraction ')
                sum_frames_checkbox = self.ui.create_check_box_widget('Sum frames')
                blanker_label = self.ui.create_label_widget('Blanker control name: ')
                blanker_field = self.ui.create_line_edit_widget(properties={'min-width': 160})
                blanker_delay_label = self.ui.create_label_widget('Blanker delay (s): ')
                blanker_delay_field = self.ui.create_line_edit_widget(properties={'min-width': 40})

                row1.add_spacing(10)
                row1.add(x_shifter_label)
                row1.add(x_shifter_field)
                row1.add_spacing(10)
                row1.add(x_shift_delay_label)
                row1.add(x_shift_delay_field)
                row1.add_spacing(10)
                row1.add_stretch()

                row2.add_spacing(10)
                row2.add(blanker_label)
                row2.add(blanker_field)
                row2.add_spacing(10)
                row2.add(blanker_delay_label)
                row2.add(blanker_delay_field)
                row2.add_stretch()
                row2.add_spacing(10)

                row3.add_spacing(10)
                row3.add(auto_dark_subtract_checkbox)
                row3.add_spacing(10)
                row3.add(sum_frames_checkbox)
                row3.add_spacing(10)
                row3.add_stretch()

                column.add(row1)
                column.add_spacing(10)
                column.add(row2)
                column.add_spacing(10)
                column.add(row3)
                column.add_stretch()

                self.content.add_spacing(10)
                self.content.add(column)
                self.content.add_spacing(5)

                auto_dark_subtract_checkbox.on_check_state_changed = auto_dark_subtract_checkbox_changed
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
