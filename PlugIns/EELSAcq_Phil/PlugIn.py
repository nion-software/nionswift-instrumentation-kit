# standard libraries
import gettext
import logging
import threading
from time import sleep
import functools

# third party libraries
import numpy as np
import autostem

# local libraries
from nion.swift import Decorators
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import HardwareSource
from nion.swift.model import DataItem
from nion.swift.model import Operation
from nion.swift.model import ImportExportManager

import ImageAlignment.register

_ = gettext.gettext

name = "PhilEELSAcquire"
disp_name = _("Phil-Style EELS Acquire")

energy_adjust_control = "EELS_Prism_Temp"
blank_control = "C_Blank"

class AcquireController(object):
    __metaclass__ = Decorators.Singleton

    """
        Provides access to the AutoSTEM objects. AutoSTEM refers to the overall microscope system of cameras,
        instruments, and algorithms used to control them.
    """

    def __init__(self):
        super(AcquireController, self).__init__()
        self.__acquire_thread = None

    def get_high_tension_v(self):
        self.connect()
        eht = autostem.values["EHT"]
        return float(eht) if eht is not None else None

    def get_ccd_pixel_angle_mrad(self):
        self.connect()
        tv_pixel_angle = autostem.values["TVPixelAngle"]
        return float(tv_pixel_angle * 1000.0) if tv_pixel_angle else None

    def start_threaded_acquire_and_sum(self, number_frames, energy_offset_per_frame, document_controller):
        if self.__acquire_thread and self.__acquire_thread.is_alive():
            logging.debug("Already acquiring")
            return

        def set_offset_energy(offset):
            current_energy = autostem.TryGetVal(energy_adjust_control)
            # built-in 150ms delay to avoid double peaks and ghosting
            autostem.SetValWait(energy_adjust_control, float(current_energy[1])+offset, 150)

        def acquire_series(number_frames, offset_per_spectrum, task_object=None):
            logging.info("Starting image acquisition.")

            with HardwareSource.get_data_element_generator_by_id("eels_tv_camera") as data_generator:
                # grab one frame to get image size
                data_element = data_generator()
                frame = data_element["data"]
                image_stack = np.empty((number_frames, frame.shape[0], frame.shape[1]))
                dark_stack = np.empty((number_frames, frame.shape[0], frame.shape[1]))
                reference_energy = autostem.TryGetVal(energy_adjust_control)
                dark = False
                for stack in [image_stack, dark_stack]:
                    if dark:
                        autostem.SetValWait(blank_control, 1.0, 200)
                        # sleep 2 seconds to allow afterglow to die out
                        sleep(2)
                    for frame in xrange(number_frames):
                        with HardwareSource.get_data_element_generator_by_id("eels_tv_camera") as data_generator:
                            set_offset_energy(offset_per_spectrum)
                            stack[frame] = data_generator()["data"]
                            if task_object is not None:
                                task_object.update_progress(_("Grabbing frame {}.").format(frame+1), (frame + 1, number_frames), None)
                    if dark:
                        autostem.SetVal(blank_control, 0)
                    autostem.SetVal(energy_adjust_control, reference_energy[1])
                    dark = not dark
                data_element["data"] = image_stack-dark_stack
                # TODO: replace frame index with acquisition time (this is effectively chronospectroscopy before the sum)
                data_element["spatial_calibrations"] = ({"origin": 0.0,
                                                         "scale": 1,
                                                         "units": "frame"},) + \
                                                       data_element["spatial_calibrations"]
            return data_element

        def align_stack(stack, task_object=None):
            number_frames = stack.shape[0]
            if task_object is not None:
                task_object.update_progress(_("Starting image alignment."), (0, number_frames))
            # Pre-allocate an array for the shifts we'll measure
            shifts = np.zeros((number_frames, 2))
            # initial reference slice is first slice
            ref = stack[0][:]
            ref_shift = np.array([0, 0])
            for index, _slice in enumerate(stack):
                if task_object is not None:
                    task_object.update_progress(_("Cross correlating frame {}.").format(index), (index + 1, number_frames), None)
                # TODO: make interpolation factor variable (it is hard-coded to 100 here.)
                shifts[index] = ref_shift+np.array(ImageAlignment.register.get_shift(ref, _slice, 100))
                ref = _slice[:]
                ref_shift = shifts[index]
            # sum image needs to be big enough for shifted images
            sum_image = np.zeros(ref.shape)
            # add the images to the registered stack
            for index, _slice in enumerate(stack):
                if task_object is not None:
                    task_object.update_progress(_("Summing frame {}.").format(index), (index + 1, number_frames), None)
                sum_image += ImageAlignment.register.shift_image(_slice, shifts[index, 0], shifts[index, 1])
            return sum_image

        def show_in_panel(data_item, document_controller, image_panel_id):
            workspace = document_controller.workspace
            document_controller.document_model.append_data_item(data_item)
            workspace.get_image_panel_by_id(image_panel_id).set_displayed_data_item(data_item)

        def add_line_profile(data_item, document_controller, image_panel_id, midpoint=0.5, integration_width=100):
            document_model = document_controller.document_model
            workspace = document_controller.workspace
            # next, line profile through center of crop
            integrated_data_item = DataItem.DataItem()
            integrated_data_item.title = _("EELS Integrated")
            integration_operation = Operation.OperationItem("line-profile-operation")
            integration_operation.set_property("start", (midpoint, 0.0))
            integration_operation.set_property("end", (midpoint, 1.0))
            integration_operation.set_property("integration_width", integration_width)
            integrated_data_item.add_operation(integration_operation)
            integrated_data_item.add_data_source(data_item)
            document_model.append_data_item(integrated_data_item)
            workspace.get_image_panel_by_id(image_panel_id).set_displayed_data_item(integrated_data_item)

        def acquire_stack_and_sum(number_frames, energy_offset_per_frame, document_controller):
            # grab the document model and workspace for convenience
            with document_controller.create_task_context_manager(_("Phil-Style EELS Acquire"), "table") as task:
                data_element = acquire_series(number_frames, energy_offset_per_frame, task)
                data_element["title"] = "Spectrum stack"
                data_item = ImportExportManager.create_data_item_from_data_element(data_element)
                # add the stack to Swift
                document_controller.queue_main_thread_task(functools.partial(show_in_panel, data_item, document_controller, "stack"))

                # align and sum the stack
                summed_image = align_stack(data_element["data"], task)
                # add the summed image to Swift
                data_element["data"] = summed_image
                data_element["title"] = "Aligned and summed spectra"
                # strip off the first dimension that we sum over
                data_element["spatial_calibrations"] = data_element["spatial_calibrations"][1:]
                data_element["spatial_calibrations"][0]["origin"]=(summed_image.shape[1]/2.0-np.sum(summed_image, axis=0).argmax())*data_element["spatial_calibrations"][0]["scale"]
                data_item = ImportExportManager.create_data_item_from_data_element(data_element)
                document_controller.queue_main_thread_task(functools.partial(show_in_panel, data_item, document_controller, "aligned and summed stack"))

                dispersive_sum = np.sum(summed_image, axis=1)
                differential = np.diff(dispersive_sum)
                top = np.argmax(differential)
                bottom = np.argmin(differential)
                _midpoint = np.mean([bottom, top])/dispersive_sum.shape[0]
                _integration_width = np.abs(bottom-top)*data_element["spatial_calibrations"][1]["scale"]
                document_controller.queue_main_thread_task(functools.partial(add_line_profile, data_item, document_controller, "spectrum", _midpoint, _integration_width))

        # create and start the thread.
        self.__acquire_thread = threading.Thread(target=acquire_stack_and_sum, args=(number_frames,
                                                                                     energy_offset_per_frame,
                                                                                     document_controller))
        self.__acquire_thread.start()


class PhilEELSAcquireControlView(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super(PhilEELSAcquireControlView, self).__init__(document_controller, panel_id, name)

        ui = document_controller.ui

        # TODO: how to get text to align right?
        self.number_frames = self.ui.create_line_edit_widget(properties={"width": 30})
        # TODO: how to get text to align right?
        self.energy_offset = self.ui.create_line_edit_widget(properties={"width": 50})

        self.acquire_button = ui.create_push_button_widget(_("Start"), properties={"width": 40, "height": 23})

        dialog_row = ui.create_row_widget()
        dialog_row.add(ui.create_label_widget(_("Number of frames:"), properties={"width": 96}))
        dialog_row.add(self.number_frames)
        dialog_row.add_stretch()
        dialog_row.add(ui.create_label_widget(_("Energy offset/frame:"), properties={"width": 128}))
        dialog_row.add(self.energy_offset)
        dialog_row.add(self.acquire_button, alignment="right")

        self.acquire_button.on_clicked = lambda: self.acquire(int(self.number_frames.text),
                                                              float(self.energy_offset.text))

        properties["margin"] = 6
        properties["spacing"] = 2
        column = ui.create_column_widget(properties)

        column.add(dialog_row)
        column.add_stretch()

        self.widget = column

        self.__workspace_controller = None

    def acquire(self, number_frames, energy_offset):
        # change to the EELS workspace layout
        self.document_controller.workspace.change_layout("Phil-Style EELS", layout_fn=self.__configure_workspace)
        AcquireController().start_threaded_acquire_and_sum(number_frames, energy_offset, self.document_controller)

    def __get_workspace_controller(self):
        if not self.__workspace_controller:
            self.__workspace_controller = self.document_controller.create_workspace_controller()
        return self.__workspace_controller

    def __configure_workspace(self, workspace, layout_id):
        column = self.ui.create_splitter_widget("vertical")
        image_panel1 = workspace.create_image_panel("spectrum")
        row = self.ui.create_splitter_widget("horizontal")
        image_panel2 = workspace.create_image_panel("stack")
        image_panel3 = workspace.create_image_panel("aligned and summed stack")
        row.add(image_panel2.widget)
        row.add(image_panel3.widget)
        column.add(image_panel1.widget)
        column.add(row)
        return column, image_panel1, layout_id



panel_name = name+"-control-panel"
workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(PhilEELSAcquireControlView, panel_name, disp_name, ["left", "right"], "left")
