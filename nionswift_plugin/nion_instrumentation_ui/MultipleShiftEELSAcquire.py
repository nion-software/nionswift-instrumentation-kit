# standard libraries
import functools
import gettext
import logging
import threading
import time

import numpy

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import DataItem
from nion.swift.model import Graphics
from nion.swift.model import ImportExportManager
from nion.swift.model import Utility
from nion.utils import Registry

from . import HardwareSourceChoice

_ = gettext.gettext

name = "MultipleShiftEELSAcquire"
disp_name = _("Multiple Shift EELS Acquire")

eels_hardware_source_id = "eels_camera"

# Daresbury
#energy_adjust_control = "EELS_Prism_Temp"
#Rutgers
energy_adjust_control = "EELS_MagneticShift_Offset"
blank_control = "C_Blank"


class AcquireController(metaclass=Utility.Singleton):

    """
        Provides access to the AutoSTEM objects. AutoSTEM refers to the overall microscope system of cameras,
        instruments, and algorithms used to control them.
    """

    def __init__(self):
        super(AcquireController, self).__init__()
        self.__acquire_thread = None

    def start_threaded_acquire_and_sum(self, stem_controller, camera, number_frames, energy_offset_per_frame, sleep_time, document_controller, final_layout_fn):
        if self.__acquire_thread and self.__acquire_thread.is_alive():
            logging.debug("Already acquiring")
            return

        def set_offset_energy(offset, sleep_time=1):
            current_energy = stem_controller.GetVal(energy_adjust_control)
            # this function waits until the value is confirmed to be the desired value (or until timeout)
            stem_controller.SetValAndConfirm(energy_adjust_control, float(current_energy) + offset, 1, int(sleep_time * 1000))
            # sleep 1 sec to avoid double peaks and ghosting
            time.sleep(sleep_time)

        def acquire_series(number_frames, offset_per_spectrum, task_object=None) -> DataItem.DataItem:
            logging.info("Starting image acquisition.")

            # grab one frame to get image size
            first_xdata = camera.get_next_xdatas_to_start()[0]
            first_data = first_xdata.data

            image_stack_data = numpy.empty((number_frames, first_data.shape[0], first_data.shape[1]), dtype=numpy.float)

            reference_energy = stem_controller.GetVal(energy_adjust_control)
            for frame_index in range(number_frames):
                set_offset_energy(offset_per_spectrum, 1)
                # use next frame to start to make sure we're getting a frame with the new energy offset
                image_stack_data[frame_index] = camera.get_next_xdatas_to_start()[0].data
                if task_object is not None:
                    task_object.update_progress(_("Grabbing EELS data frame {}.").format(frame_index + 1),
                                                (frame_index + 1, number_frames), None)

            stem_controller.SetValWait(blank_control, 1.0, 200)
            # sleep 4 seconds to allow afterglow to die out
            time.sleep(sleep_time)
            dark_sum = None
            for frame_index in range(number_frames):
                if dark_sum is None:
                    # use next frame to start to make sure we're getting a blanked frame
                    dark_sum = camera.get_next_xdatas_to_start()[0].data
                else:
                    # but now use next frame to finish since we can know it's already blanked
                    dark_sum += camera.get_next_xdatas_to_finish()[0].data
                if task_object is not None:
                    task_object.update_progress(_("Grabbing dark data frame {}.").format(frame_index + 1),
                                                (frame_index + 1, number_frames), None)
            stem_controller.SetVal(blank_control, 0)
            stem_controller.SetVal(energy_adjust_control, reference_energy)
            image_stack_data -= dark_sum / number_frames

            dimension_calibration0 = first_xdata.dimensional_calibrations[0]
            dimension_calibration1 = first_xdata.dimensional_calibrations[1]
            # TODO: replace frame calibration with acquisition time (this is effectively chronospectroscopy before the sum)
            sequence_calibration = Calibration.Calibration(units="frame")
            image_stack_data_item = DataItem.DataItem(numpy.zeros((8, 8)))  # dummy data
            image_stack_data_item.set_xdata(DataAndMetadata.new_data_and_metadata(image_stack_data,
                                                                                  dimensional_calibrations=[
                                                                                      sequence_calibration,
                                                                                      dimension_calibration0,
                                                                                      dimension_calibration1],
                                                                                  data_descriptor=DataAndMetadata.DataDescriptor(
                                                                                      True, 0, 2)))

            return image_stack_data_item

        def align_stack(stack, task_object=None):
            number_frames = stack.shape[0]
            if task_object is not None:
                task_object.update_progress(_("Starting image alignment."), (0, number_frames))
            # Pre-allocate an array for the shifts we'll measure
            shifts = numpy.zeros((number_frames, 2))
            # initial reference slice is first slice
            ref = stack[0][:]
            ref_shift = numpy.array([0, 0])
            for index, _slice in enumerate(stack):
                if task_object is not None:
                    task_object.update_progress(_("Cross correlating frame {}.").format(index), (index + 1, number_frames), None)
                # TODO: make interpolation factor variable (it is hard-coded to 100 here.)
                ref_xdata = DataAndMetadata.new_data_and_metadata(ref)
                _slice_xdata = DataAndMetadata.new_data_and_metadata(_slice)
                shifts[index] = ref_shift + numpy.array(xd.register_translation(ref_xdata, _slice_xdata, 100))
                ref = _slice[:]
                ref_shift = shifts[index]
            # sum image needs to be big enough for shifted images
            sum_image = numpy.zeros(ref.shape)
            # add the images to the registered stack
            for index, _slice in enumerate(stack):
                if task_object is not None:
                    task_object.update_progress(_("Summing frame {}.").format(index), (index + 1, number_frames), None)
                _slice_xdata = DataAndMetadata.new_data_and_metadata(_slice)
                sum_image += xd.shift(_slice_xdata, shifts[index]).data
            return sum_image, shifts

        def show_in_panel(data_item, document_controller, display_panel_id):
            document_controller.document_model.append_data_item(data_item)
            display_item = document_controller.document_model.get_display_item_for_data_item(data_item)
            document_controller.workspace_controller.display_display_item_in_display_panel(display_item, display_panel_id)

        def add_line_profile(data_item, document_controller, display_panel_id, midpoint=0.5, integration_width=.25):
            logging.debug("midpoint: {:.4f}".format(midpoint))
            logging.debug("width: {:.4f}".format(integration_width))

            # next, line profile through center of crop
            # please don't copy this bad example code!
            crop_region = Graphics.RectangleGraphic()
            crop_region.center = (midpoint, 0.5)
            crop_region.size = (integration_width, 1)
            crop_region.is_bounds_constrained = True
            display_item = document_controller.document_model.get_display_item_for_data_item(data_item)
            display_item.add_graphic(crop_region)
            eels_data_item = document_controller.document_model.get_projection_new(display_item, crop_region)
            if eels_data_item:
                eels_data_item.title = _("EELS Summed")
                eels_display_item = document_controller.document_model.get_display_item_for_data_item(eels_data_item)
                document_controller.show_display_item(eels_display_item)
            else:
                eels_display_item = None

            document_controller.workspace_controller.display_display_item_in_display_panel(eels_display_item, display_panel_id)

        def acquire_stack_and_sum(number_frames, energy_offset_per_frame, document_controller, final_layout_fn):
            # grab the document model and workspace for convenience
            with document_controller.create_task_context_manager(_("Multiple Shift EELS Acquire"), "table") as task:
                # acquire the stack. it will be added to the document by queueing to the main thread at the end of this method.
                stack_data_item = acquire_series(number_frames, energy_offset_per_frame, task)
                stack_data_item.title = _("Spectrum Stack")

                # align and sum the stack
                data_element = dict()
                summed_image, shifts = align_stack(stack_data_item.data, task)
                # add the summed image to Swift
                data_element["data"] = summed_image
                data_element["title"] = "Aligned and summed spectra"
                # strip off the first dimension that we sum over
                for dimensional_calibration in stack_data_item.dimensional_calibrations[1:]:
                    data_element.setdefault("spatial_calibrations", list()).append({
                        "origin": dimensional_calibration.offset,  # TODO: fix me
                        "scale": dimensional_calibration.scale,
                        "units": dimensional_calibration.units
                    })
                # set the energy dispersive calibration so that the ZLP is at zero eV
                zlp_position_pixels = numpy.sum(summed_image, axis=0).argmax()
                zlp_position_calibrated_units = -zlp_position_pixels * data_element["spatial_calibrations"][1]["scale"]
                data_element["spatial_calibrations"][1]["offset"] = zlp_position_calibrated_units
                sum_data_item = ImportExportManager.create_data_item_from_data_element(data_element)


                dispersive_sum = numpy.sum(summed_image, axis=1)
                differential = numpy.diff(dispersive_sum)
                top = numpy.argmax(differential)
                bottom = numpy.argmin(differential)
                _midpoint = numpy.mean([bottom, top])/dispersive_sum.shape[0]
                _integration_width = float(numpy.abs(bottom-top)) / dispersive_sum.shape[0] #* data_element["spatial_calibrations"][0]["scale"]


                document_controller.queue_task(final_layout_fn)
                document_controller.queue_task(functools.partial(show_in_panel, stack_data_item, document_controller, "multiple_shift_eels_stack"))
                document_controller.queue_task(functools.partial(show_in_panel, sum_data_item, document_controller, "multiple_shift_eels_aligned_summed_stack"))
                document_controller.queue_task(functools.partial(add_line_profile, sum_data_item, document_controller, "multiple_shift_eels_spectrum", _midpoint, _integration_width))

        # create and start the thread.
        self.__acquire_thread = threading.Thread(target=acquire_stack_and_sum, args=(number_frames,
                                                                                     energy_offset_per_frame,
                                                                                     document_controller,
                                                                                     final_layout_fn))
        self.__acquire_thread.start()


class MultipleShiftEELSAcquireControlView(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super().__init__(document_controller, panel_id, name)

        ui = document_controller.ui

        self.__eels_camera_choice = HardwareSourceChoice.HardwareSourceChoice(ui, "eels_camera_hardware_source_id", lambda hardware_source: hardware_source.features.get("is_eels_camera"))

        # TODO: how to get text to align right?
        self.number_frames = self.ui.create_line_edit_widget(properties={"width": 30})
        self.number_frames.text = "30"
        # TODO: how to get text to align right?
        self.energy_offset = self.ui.create_line_edit_widget(properties={"width": 50})
        self.energy_offset.text = "0"
        # TODO: how to get text to align right?
        self.sleep_time = self.ui.create_line_edit_widget(properties={"width": 50})
        self.sleep_time.text = "15"

        self.acquire_button = ui.create_push_button_widget(_("Start"))

        camera_row = ui.create_row_widget()
        camera_row.add_spacing(12)
        camera_row.add(self.__eels_camera_choice.create_combo_box(ui))
        camera_row.add_stretch()

        dialog_row = ui.create_row_widget()
        dialog_row.add(ui.create_label_widget(_("Number of frames:")))
        dialog_row.add(self.number_frames)
        dialog_row.add_stretch()
        dialog_row2 = ui.create_row_widget()
        dialog_row2.add(ui.create_label_widget(_("Energy offset/frame:")))
        dialog_row2.add(self.energy_offset)
        dialog_row2.add_stretch()
        dialog_row3 = ui.create_row_widget()
        dialog_row3.add(ui.create_label_widget(_("Sleep seconds after blank:")))
        dialog_row3.add(self.sleep_time)
        dialog_row3.add_stretch()
        dialog_row4 = ui.create_row_widget()
        dialog_row4.add_stretch()
        dialog_row4.add(self.acquire_button)

        self.acquire_button.on_clicked = lambda: self.acquire(int(self.number_frames.text),
                                                              float(self.energy_offset.text),
                                                              int(self.sleep_time.text))

        properties["margin"] = 6
        properties["spacing"] = 2
        column = ui.create_column_widget(properties=properties)

        column.add(camera_row)
        column.add(dialog_row)
        column.add(dialog_row2)
        column.add(dialog_row3)
        column.add(dialog_row4)
        column.add_stretch()

        self.widget = column

        self.__workspace_controller = None

    def close(self):
        self.__eels_camera_choice.close()
        self.__eels_camera_choice = None
        super().close()

    def acquire(self, number_frames, energy_offset, sleep_time):
        eels_camera = self.__eels_camera_choice.hardware_source
        if eels_camera:
            # setup the workspace layout
            self.__configure_start_workspace(self.document_controller.workspace_controller, eels_camera.hardware_source_id)
            # start the EELS acquisition
            eels_camera.start_playing()
            stem_controller = Registry.get_component("stem_controller")
            AcquireController().start_threaded_acquire_and_sum(stem_controller, eels_camera, number_frames,
                                                               energy_offset, sleep_time, self.document_controller,
                                                               functools.partial(self.set_final_layout))

    def set_final_layout(self):
        # change to the EELS workspace layout
        self.__configure_final_workspace(self.document_controller.workspace_controller)

    def __create_canvas_widget_from_image_panel(self, image_panel):
        canvas_widget = self.ui.create_canvas_widget()
        canvas_widget.canvas_item.add_canvas_item(image_panel.canvas_item)
        image_row = self.ui.create_row_widget()
        image_row.add(canvas_widget)
        return image_row

    def __configure_final_workspace(self, workspace_controller):
        spectrum_display = {"type": "image", "selected": True, "display_panel_id": "multiple_shift_eels_spectrum"}
        stack_display = {"type": "image", "display_panel_id": "multiple_shift_eels_stack"}
        aligned_summer_stack_display = {"type": "image", "display_panel_id": "multiple_shift_eels_aligned_summed_stack"}
        layout_right_side = {"type": "splitter", "orientation": "horizontal", "splits": [0.5, 0.5],
            "children": [stack_display, aligned_summer_stack_display]}
        layout = {"type": "splitter", "orientation": "vertical", "splits": [0.5, 0.5],
            "children": [spectrum_display, layout_right_side]}
        workspace_controller.ensure_workspace(_("Multiple Shift EELS Results"), layout, "multiple_shift_eels_results")

    def __configure_start_workspace(self, workspace_controller, hardware_source_id):
        spectrum_display = {'type': 'image', 'hardware_source_id': hardware_source_id, 'controller_type': 'camera-live', 'show_processed_data': True}
        eels_raw_display = {'type': 'image', 'hardware_source_id': hardware_source_id, 'controller_type': 'camera-live', 'show_processed_data': False}
        layout = {"type": "splitter", "orientation": "vertical", "splits": [0.5, 0.5],
            "children": [spectrum_display, eels_raw_display]}
        workspace_controller.ensure_workspace(_("Multiple Shift EELS"), layout, "multiple_shift_eels_acquisition")


def run():
    panel_name = name+"-control-panel"
    workspace_manager = Workspace.WorkspaceManager()
    workspace_manager.register_panel(MultipleShiftEELSAcquireControlView, panel_name, disp_name, ["left", "right"], "left")
