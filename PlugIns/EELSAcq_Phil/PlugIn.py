# standard libraries
import gettext
import logging
import threading

# third party libraries
import numpy as np
import autostem

# local libraries
from nion.swift import Decorators
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.Decorators import relative_file
from nion.swift.model import HardwareSource
from nion.swift.model import ImportExportManager

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
            """
            TODO: Phil should tell us how he's changing the energy offset for each spectrum
            """
            pass

        def acquire_series(number_frames, offset_per_spectrum, task_object=None):
            logging.info("Starting image acquisition.")
            with HardwareSource.get_data_element_generator_by_id("eels_tv_camera") as data_generator:
                # grab one frame to get image size
                data_element = data_generator()
                frame = data_element["data"]
                stack = np.empty((number_frames, frame.shape[0], frame.shape[1]))
                for frame in xrange(number_frames):
                    set_offset_energy(offset_per_spectrum)
                    stack[frame] = data_generator()["data"]
                    if task_object is not None:
                        task_object.update_progress(_("Grabbing frame {}.").format(frame+1), (frame + 1, number_frames), None)
                data_element["data"] = stack
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
            # we're going to use OpenCV to do the phase correlation
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
            with document_controller.create_task_context_manager(_("Phil-Style EELS Acquire"), "table") as task:
                data_element = acquire_series(number_frames, energy_offset_per_frame, task)
                # add the stack to Swift
                data_item = document_controller.add_data_element(data_element)

                # align and sum the stack
                summed_image = align_stack(data_element["data"], task)
                # add the summed image to Swift
                data_element["data"] = summed_image
                # strip off the first dimension that we sum over
                data_element["spatial_calibrations"] = data_element["spatial_calibrations"][1:]
                data_item = document_controller.add_data_element(data_element)

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

    def acquire(self, number_frames, energy_offset):
        AcquireController().start_threaded_acquire_and_sum(number_frames, energy_offset, self.document_controller)


panel_name = name+"-control-panel"
workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(PhilEELSAcquireControlView, panel_name, disp_name, ["left", "right"], "left")
