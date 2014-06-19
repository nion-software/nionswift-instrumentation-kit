# standard libraries
import gettext
import logging
import threading
from time import sleep

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

import ImageAlignment.register

_ = gettext.gettext

name = "PhilEELSAcquire"
disp_name = _("Phil-Style EELS Acquire")

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
            key = "EELS_Prism_Temp"
            current_energy = 0
            current_energy = autostem.TryGetVal(key)
            print current_energy
            autostem.SetVal(key, float(current_energy[1])+offset)
            pass

        def acquire_series(number_frames, offset_per_spectrum, task_object=None):
            logging.info("Starting image acquisition.")

            with HardwareSource.get_data_element_generator_by_id("eels_tv_camera") as data_generator:
                # grab one frame to get image size
                data_element = data_generator()
                frame = data_element["data"]
                image_stack = np.empty((number_frames, frame.shape[0], frame.shape[1]))
                dark_stack = np.empty((number_frames, frame.shape[0], frame.shape[1]))
                reference_energy = autostem.TryGetVal("EELS_Prism_Temp")
                dark = False
                for stack in [image_stack, dark_stack]:
                    if dark:
                        autostem.SetVal("C_Blank", 1.0)
                    for frame in xrange(number_frames):
                        with HardwareSource.get_data_element_generator_by_id("eels_tv_camera") as data_generator:
                            set_offset_energy(offset_per_spectrum)
                            stack[frame] = data_generator()["data"]
                            if task_object is not None:
                                task_object.update_progress(_("Grabbing frame {}.").format(frame+1), (frame + 1, number_frames), None)
                    if dark:
                        autostem.SetVal("C_Blank", 0)
                    autostem.SetVal("EELS_Prism_Temp", reference_energy[1])
                    dark = not dark
                    sleep(2)
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

        def acquire_stack_and_sum(number_frames, energy_offset_per_frame, document_controller):
            # grab the document model and workspace for convenience
            document_model = self.document_controller.document_model
            workspace = self.document_controller.workspace
            with document_controller.create_task_context_manager(_("Phil-Style EELS Acquire"), "table") as task:
                data_element = acquire_series(number_frames, energy_offset_per_frame, task)
                # add the stack to Swift
                data_item = document_controller.add_data_element(data_element)
                document_model.append_data_item(data_item)
                workspace.get_image_panel_by_id("stack").set_displayed_data_item(data_item)

                # align and sum the stack
                summed_image = align_stack(data_element["data"], task)
                # add the summed image to Swift
                data_element["data"] = summed_image
                # strip off the first dimension that we sum over
                data_element["spatial_calibrations"] = data_element["spatial_calibrations"][1:]
                data_item = document_controller.add_data_element(data_element)
                document_model.append_data_item(data_item)
                workspace.get_image_panel_by_id("aligned and summed stack").set_displayed_data_item(data_item)

                # next, line profile through center of crop
                integrated_data_item = DataItem.DataItem()
                integrated_data_item.title = _("EELS Integrated")
                integration_operation = Operation.OperationItem("line-profile-operation")
                integration_operation.set_property("start", (0.5, 0.0))
                integration_operation.set_property("end", (0.5, 1.0))
                integration_operation.set_property("integration_width", 40)
                integrated_data_item.add_operation(integration_operation)
                integrated_data_item.append_data_item(data_item)
                document_model.append_data_item(integrated_data_item)
                workspace.get_image_panel_by_id("spectrum").set_displayed_data_item(integrated_data_item)

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
        # TODO: is there a less manual spacer that appropriately stretches to fill a row's space?
        dialog_row.add_spacing(20)
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
