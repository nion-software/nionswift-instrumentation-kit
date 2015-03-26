# standard libraries
import gettext
import logging
import threading
from time import sleep
import functools

# third party libraries
import numpy

try:
    import autostem
except ImportError:
    class AutoSTEM(object):
        def __init__(self):
            self.values = {"EHT": 100, "TVPixelAngle": 2/1000.0}
        def TryGetVal(self, property_id):
            return [1.0, 1.0]
        def SetValAndConfirm(self, property_id, v1, v2, timeout):
            pass
        def SetValWait(self, property_id, v1, timeout):
            pass
        def SetVal(self, property_id, value):
            pass
    autostem = AutoSTEM()

# local libraries
from nion.swift import Decorators
from nion.swift import Panel
from nion.swift import Workspace
from nion.swift.model import HardwareSource
from nion.swift.model import DataItem
from nion.swift.model import Operation
from nion.swift.model import ImportExportManager
from nion.swift.model import Region
from nion.ui import CanvasItem

import ImageAlignment.register

_ = gettext.gettext

name = "PhilEELSAcquire"
disp_name = _("Phil-Style EELS Acquire")

eels_hardware_source_id = "eels_camera"

# Daresbury
#energy_adjust_control = "EELS_Prism_Temp"
#Rutgers
energy_adjust_control = "EELS_MagneticShift_Offset"
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
        # self.connect()
        eht = autostem.values["EHT"]
        return float(eht) if eht is not None else None

    def get_ccd_pixel_angle_mrad(self):
        # self.connect()
        tv_pixel_angle = autostem.values["TVPixelAngle"]
        return float(tv_pixel_angle * 1000.0) if tv_pixel_angle else None

    def start_threaded_acquire_and_sum(self, number_frames, energy_offset_per_frame, document_controller,
                                       final_layout_fn):
        if self.__acquire_thread and self.__acquire_thread.is_alive():
            logging.debug("Already acquiring")
            return

        def set_offset_energy(offset, sleep_time=1):
            current_energy = autostem.TryGetVal(energy_adjust_control)
            # this function waits until the value is confirmed to be the desired value (or until timeout)
            autostem.SetValAndConfirm(energy_adjust_control, float(current_energy[1])+offset, 0, sleep_time*1000)
            # sleep 1 sec to avoid double peaks and ghosting
            sleep(sleep_time)

        def acquire_series(number_frames, offset_per_spectrum, task_object=None):
            logging.info("Starting image acquisition.")

            with HardwareSource.get_data_element_generator_by_id(eels_hardware_source_id) as data_generator:
                # grab one frame to get image size
                data_element = data_generator()
                frame = data_element["data"]
                image_stack = numpy.empty((number_frames, frame.shape[0], frame.shape[1]), dtype=numpy.float)
                dark_sum = numpy.zeros_like(frame)

                reference_energy = autostem.TryGetVal(energy_adjust_control)
                for frame in xrange(number_frames):
                    set_offset_energy(offset_per_spectrum, 1)
                    image_stack[frame] = data_generator()["data"]
                    if task_object is not None:
                            task_object.update_progress(_("Grabbing EELS data frame {}.").format(frame+1),
                                                        (frame + 1, number_frames), None)

                autostem.SetValWait(blank_control, 1.0, 200)
                # sleep 4 seconds to allow afterglow to die out
                sleep(4)
                for frame in xrange(number_frames):
                    dark_sum += data_generator()["data"]
                    if task_object is not None:
                            task_object.update_progress(_("Grabbing dark data frame {}.").format(frame+1),
                                                        (frame + 1, number_frames), None)
                autostem.SetVal(blank_control, 0)
                autostem.SetVal(energy_adjust_control, reference_energy[1])
                data_element["data"] = image_stack-dark_sum
                # TODO: replace frame index with acquisition time (this is effectively chronospectroscopy before the sum)
                data_element["spatial_calibrations"] = [{"origin": 0.0,
                                                         "scale": 1,
                                                         "units": "frame"}, ] + \
                                                       data_element["spatial_calibrations"]
            return data_element

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
                shifts[index] = ref_shift+numpy.array(ImageAlignment.register.get_shift(ref, _slice, 100))
                ref = _slice[:]
                ref_shift = shifts[index]
            # sum image needs to be big enough for shifted images
            sum_image = numpy.zeros(ref.shape)
            # add the images to the registered stack
            for index, _slice in enumerate(stack):
                if task_object is not None:
                    task_object.update_progress(_("Summing frame {}.").format(index), (index + 1, number_frames), None)
                sum_image += ImageAlignment.register.shift_image(_slice, shifts[index, 0], shifts[index, 1])
            return sum_image

        def show_in_panel(data_item, document_controller, image_panel_id):
            document_controller.document_model.append_data_item(data_item)
            document_controller.workspace_controller.get_display_panel(image_panel_id).set_displayed_data_item(data_item)

        def add_line_profile(data_item, document_controller, image_panel_id, midpoint=0.5, integration_width=.25):
            logging.debug("midpoint: {:.4f}".format(midpoint))
            logging.debug("width: {:.4f}".format(integration_width))

            # next, line profile through center of crop
            # please don't copy this bad example code!
            operation = Operation.OperationItem("projection-operation")
            buffered_data_source_specifier = DataItem.DisplaySpecifier.from_data_item(data_item)
            crop_region = Region.RectRegion()
            crop_region.center = (midpoint, 0.5)
            crop_region.size = (integration_width, 1)
            buffered_data_source_specifier.buffered_data_source.add_region(crop_region)
            display_specifier = document_controller.add_processing_operation(buffered_data_source_specifier, operation, crop_region=crop_region)
            self.__eels_data_item = display_specifier.data_item
            self.__eels_data_item.title = _("EELS Integrated")

            document_controller.workspace_controller.get_display_panel(image_panel_id).set_displayed_data_item(self.__eels_data_item)

        def acquire_stack_and_sum(number_frames, energy_offset_per_frame, document_controller, final_layout_fn):
            # grab the document model and workspace for convenience
            with document_controller.create_task_context_manager(_("Phil-Style EELS Acquire"), "table") as task:
                data_element = acquire_series(number_frames, energy_offset_per_frame, task)
                data_element["title"] = "Spectrum stack"
                stack_data_item = ImportExportManager.create_data_item_from_data_element(data_element)
                # add the stack to Swift

                # align and sum the stack
                summed_image = align_stack(data_element["data"], task)
                # add the summed image to Swift
                data_element["data"] = summed_image
                data_element["title"] = "Aligned and summed spectra"
                # strip off the first dimension that we sum over
                data_element["spatial_calibrations"] = data_element["spatial_calibrations"][1:]
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
                document_controller.queue_task(functools.partial(show_in_panel, stack_data_item, document_controller, "eels_phil_stack"))
                document_controller.queue_task(functools.partial(show_in_panel, sum_data_item, document_controller, "eels_phil_aligned_summed_stack"))
                document_controller.queue_task(functools.partial(add_line_profile, sum_data_item, document_controller, "eels_phil_spectrum", _midpoint, _integration_width))

        # create and start the thread.
        self.__acquire_thread = threading.Thread(target=acquire_stack_and_sum, args=(number_frames,
                                                                                     energy_offset_per_frame,
                                                                                     document_controller,
                                                                                     final_layout_fn))
        self.__acquire_thread.start()


class PhilEELSAcquireControlView(Panel.Panel):

    def __init__(self, document_controller, panel_id, properties):
        super(PhilEELSAcquireControlView, self).__init__(document_controller, panel_id, name)

        # data items used to show live progress
        self.__eels_raw_data_item = None
        self.__eels_data_item = None

        ui = document_controller.ui

        # TODO: how to get text to align right?
        self.number_frames = self.ui.create_line_edit_widget(properties={"width": 30})
        self.number_frames.text = "4"
        # TODO: how to get text to align right?
        self.energy_offset = self.ui.create_line_edit_widget(properties={"width": 50})
        self.energy_offset.text = "40"

        self.acquire_button = ui.create_push_button_widget(_("Start"))

        dialog_row = ui.create_row_widget()
        dialog_row.add(ui.create_label_widget(_("Number of frames:")))
        dialog_row.add(self.number_frames)
        dialog_row.add_stretch()
        dialog_row2 = ui.create_row_widget()
        dialog_row2.add(ui.create_label_widget(_("Energy offset/frame:")))
        dialog_row2.add(self.energy_offset)
        dialog_row2.add_stretch()
        dialog_row3 = ui.create_row_widget()
        dialog_row3.add_stretch()
        dialog_row3.add(self.acquire_button)

        self.acquire_button.on_clicked = lambda: self.acquire(int(self.number_frames.text),
                                                              float(self.energy_offset.text))

        properties["margin"] = 6
        properties["spacing"] = 2
        column = ui.create_column_widget(properties)

        column.add(dialog_row)
        column.add(dialog_row2)
        column.add(dialog_row3)
        column.add_stretch()

        self.widget = column

        self.__workspace_controller = None

    def acquire(self, number_frames, energy_offset):
        self.show_initial_plots()
        AcquireController().start_threaded_acquire_and_sum(number_frames, energy_offset, self.document_controller,
                                                           functools.partial(self.set_final_layout))
        # wait for the acq to finish

    def set_final_layout(self):
        # change to the EELS workspace layout
        self.__configure_final_workspace(self.document_controller.workspace_controller)

    def show_initial_plots(self):
        document_controller = self.document_controller
        document_model = document_controller.document_model

        self.__configure_start_workspace(document_controller.workspace_controller)

        # get the workspace controller, which is the object that will put acquisition items into the workspace
        workspace_controller = self.document_controller.workspace_controller
        eels_hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(eels_hardware_source_id)

        # create an acquisition image and put it in the lower left panel. the idea here (for now) is to create
        # an image that will be recognized by the workspace controller as 'the acquisition image'.
        # NOTE: this code is a hack until a better solution is available.
        if not self.__eels_raw_data_item:

            # create the new data item, add it to the document, and save a reference to it in this class
            eels_raw_data_item = DataItem.DataItem(numpy.zeros((16, 16), numpy.float))
            eels_raw_data_item.title = _("EELS Raw")
            document_model.append_data_item(eels_raw_data_item)
            self.__eels_raw_data_item = eels_raw_data_item

            # this next section sets up the eels_raw_data_item to be the one that gets used as the acquisition
            # NOTE: this code is a hack until a better solution is available.
            view_id = eels_hardware_source.hardware_source_id
            workspace_controller.setup_channel(eels_hardware_source.hardware_source_id, None, view_id, eels_raw_data_item)
            eels_raw_data_item.session_id = document_model.session_id

        workspace_controller.get_display_panel("eels_phil_raw").set_displayed_data_item(self.__eels_raw_data_item)

        # next, line profile through center of crop
        if not self.__eels_data_item:

            # create the new data item, add it to the document, and save a reference to it in this class
            # set up the crop and projection operation. the crop also gets a region on the source.
            operation = Operation.OperationItem("projection-operation")
            buffered_data_source_specifier = DataItem.DisplaySpecifier.from_data_item(self.__eels_raw_data_item)
            crop_region = Region.RectRegion()
            crop_region.center = (0.5, 0.5)
            crop_region.size = (0.5, 1.0)
            buffered_data_source_specifier.buffered_data_source.add_region(crop_region)
            display_specifier = document_controller.add_processing_operation(buffered_data_source_specifier, operation, crop_region=crop_region)
            self.__eels_data_item = display_specifier.data_item
            self.__eels_data_item.title = _("EELS")

        # display the eels data item
        workspace_controller.get_display_panel("eels_phil_spectrum").set_displayed_data_item(self.__eels_data_item)

        eels_hardware_source.start_playing()

    def __create_canvas_widget_from_image_panel(self, image_panel):
        image_panel.root_canvas_item = CanvasItem.RootCanvasItem(self.ui)
        image_panel.root_canvas_item.add_canvas_item(image_panel.canvas_item)
        image_row = self.ui.create_row_widget()
        image_row.add(image_panel.root_canvas_item.canvas_widget)
        return image_row

    def __configure_final_workspace(self, workspace_controller):
        spectrum_display = {"type": "image", "selected": True, "display_panel_id": "eels_phil_spectrum"}
        stack_display = {"type": "image", "display_panel_id": "eels_phil_stack"}
        aligned_summer_stack_display = {"type": "image", "display_panel_id": "eels_phil_aligned_summed_stack"}
        layout_right_side = {"type": "splitter", "orientation": "horizontal", "splits": [0.5, 0.5],
            "children": [stack_display, aligned_summer_stack_display]}
        layout = {"type": "splitter", "orientation": "vertical", "splits": [0.5, 0.5],
            "children": [spectrum_display, layout_right_side]}
        workspace_controller.ensure_workspace(_("Phil-Style EELS Results"), layout, "eels_phil_results")

    def __configure_start_workspace(self, workspace_controller):
        spectrum_display = {"type": "image", "selected": True, "display_panel_id": "eels_phil_spectrum"}
        eels_raw_display = {"type": "image", "display_panel_id": "eels_phil_raw"}
        layout = {"type": "splitter", "orientation": "vertical", "splits": [0.5, 0.5],
            "children": [spectrum_display, eels_raw_display]}
        workspace_controller.ensure_workspace(_("Phil-Style EELS"), layout, "eels_phil_acquisition")


panel_name = name+"-control-panel"
workspace_manager = Workspace.WorkspaceManager()
workspace_manager.register_panel(PhilEELSAcquireControlView, panel_name, disp_name, ["left", "right"], "left")
