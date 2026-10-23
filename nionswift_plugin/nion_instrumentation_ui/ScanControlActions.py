from __future__ import annotations

# standard libraries
import copy
import gettext
import typing

# third party libraries
# None

# local libraries
from nion.instrumentation import HardwareSource
from nion.instrumentation import scan_base
from nion.swift.model import Notification
from nion.ui import Window
from nion.utils import Geometry


_ = gettext.gettext


def set_current_frame_parameters(scan_hardware_source: scan_base.ScanHardwareSource, frame_parameters: scan_base.ScanFrameParameters) -> None:
    """Set the scan device's current frame parameters, unless it is recording or in sequence mode."""
    # a recording or a sequence acquisition scans with its own frame parameters, so an edit made then is ignored,
    # which leaves both the acquisition and the active profile unchanged.
    if not scan_hardware_source.is_recording and not scan_hardware_source.is_in_sequence_mode:
        scan_hardware_source.set_current_frame_parameters(frame_parameters)


# the functions below each return a copy of the frame parameters with one step of a scan control panel's step buttons applied.
# width_height_linked is whether the panel links the width and height, so that a step to one is also made to the other.

def with_increased_pixel_time(frame_parameters: scan_base.ScanFrameParameters) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    frame_parameters.pixel_time_us *= 2.0
    return frame_parameters


def with_decreased_pixel_time(frame_parameters: scan_base.ScanFrameParameters) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    frame_parameters.pixel_time_us = max(0.01, frame_parameters.pixel_time_us / 2.0)
    return frame_parameters


def with_increased_fov(frame_parameters: scan_base.ScanFrameParameters) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    frame_parameters.fov_nm *= 2.0
    return frame_parameters


def with_decreased_fov(frame_parameters: scan_base.ScanFrameParameters) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    frame_parameters.fov_nm = max(1.0, frame_parameters.fov_nm / 2.0)
    return frame_parameters


def with_increased_width(frame_parameters: scan_base.ScanFrameParameters, width_height_linked: bool) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    pixel_size = frame_parameters.pixel_size
    if width_height_linked:
        frame_parameters.pixel_size = Geometry.IntSize(pixel_size.height * 2, pixel_size.width * 2)
    else:
        frame_parameters.pixel_size = Geometry.IntSize(pixel_size.height, pixel_size.width * 2)
    return frame_parameters


def with_decreased_width(frame_parameters: scan_base.ScanFrameParameters, width_height_linked: bool) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    pixel_size = frame_parameters.pixel_size
    if width_height_linked:
        frame_parameters.pixel_size = Geometry.IntSize(max(1, pixel_size.height // 2), max(1, pixel_size.width // 2))
    else:
        frame_parameters.pixel_size = Geometry.IntSize(pixel_size.height, max(1, pixel_size.width // 2))
    return frame_parameters


def with_increased_height(frame_parameters: scan_base.ScanFrameParameters, width_height_linked: bool) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    pixel_size = frame_parameters.pixel_size
    if width_height_linked:
        frame_parameters.pixel_size = Geometry.IntSize(pixel_size.height * 2, pixel_size.width * 2)
    else:
        frame_parameters.pixel_size = Geometry.IntSize(pixel_size.height * 2, pixel_size.width)
    return frame_parameters


def with_decreased_height(frame_parameters: scan_base.ScanFrameParameters, width_height_linked: bool) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    pixel_size = frame_parameters.pixel_size
    if width_height_linked:
        frame_parameters.pixel_size = Geometry.IntSize(max(1, pixel_size.height // 2), max(1, pixel_size.width // 2))
    else:
        frame_parameters.pixel_size = Geometry.IntSize(max(1, pixel_size.height // 2), pixel_size.width)
    return frame_parameters


def with_increased_subscan_width(frame_parameters: scan_base.ScanFrameParameters) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    subscan_pixel_size = frame_parameters.subscan_pixel_size or frame_parameters.scan_size
    subscan_width = frame_parameters.subscan_pixel_width_override or subscan_pixel_size.width
    frame_parameters.subscan_pixel_width_override = subscan_width * 2
    return frame_parameters


def with_decreased_subscan_width(frame_parameters: scan_base.ScanFrameParameters) -> scan_base.ScanFrameParameters:
    frame_parameters = copy.copy(frame_parameters)
    subscan_pixel_size = frame_parameters.subscan_pixel_size or frame_parameters.scan_size
    subscan_width = frame_parameters.subscan_pixel_width_override or subscan_pixel_size.width
    frame_parameters.subscan_pixel_width_override = subscan_width // 2
    return frame_parameters


def increase_pmt(scan_hardware_source: scan_base.ScanHardwareSource, channel_index: int) -> None:
    """Increase the PMT of the scan channel, showing a notification if the device fails to."""
    # a failure to change the PMT is shown to the person who asked for it, and the window carries on.
    try:
        scan_hardware_source.increase_pmt(channel_index)
    except Exception as e:
        _notify_pmt_change_failed(e)


def decrease_pmt(scan_hardware_source: scan_base.ScanHardwareSource, channel_index: int) -> None:
    """Decrease the PMT of the scan channel, showing a notification if the device fails to."""
    # a failure to change the PMT is shown to the person who asked for it, and the window carries on.
    try:
        scan_hardware_source.decrease_pmt(channel_index)
    except Exception as e:
        _notify_pmt_change_failed(e)


def _notify_pmt_change_failed(exception: Exception) -> None:
    notification = Notification.Notification("notification", "\N{MICROSCOPE} STEM Controller", "Unable to change PMT", f"Exception: {exception}.")
    Notification.notify(notification)


def _get_scan_hardware_source(context: Window.ActionContext) -> scan_base.ScanHardwareSource | None:
    # the scan device given by the action's hardware_source_id parameter, or None if there is no such scan device.
    hardware_source_id = context.parameters.get("hardware_source_id")
    if hardware_source_id:
        hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
        if isinstance(hardware_source, scan_base.ScanHardwareSource):
            return hardware_source
    return None


class ScanAction(Window.Action):
    """An action on the scan device given by the hardware_source_id parameter.

    The action does nothing if there is no such scan device. Its summary includes the scan device's name.
    """

    def get_action_summary(self, context: Window.ActionContext) -> str:
        scan_hardware_source = _get_scan_hardware_source(context)
        hardware_source_name = scan_hardware_source.display_name if scan_hardware_source else "N/A"
        return f"{self.action_id}: {self.action_name} ({hardware_source_name})"


class FrameParametersStepAction(ScanAction):
    """An action which applies one step of a scan control panel's step buttons to the scan device's current frame parameters."""

    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        if scan_hardware_source := _get_scan_hardware_source(context):
            frame_parameters = self._get_stepped_frame_parameters(scan_hardware_source.get_current_frame_parameters(), context)
            set_current_frame_parameters(scan_hardware_source, frame_parameters)
        return Window.ActionResult(Window.ActionStatus.FINISHED)

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        raise NotImplementedError()


class SizeStepAction(FrameParametersStepAction):
    """A step of the scan width or height, which also takes the width_height_linked parameter.

    With width_height_linked true, the step is also made to the other dimension. Without the parameter, the step changes only the
    dimension the action names. The scan control panel passes its own link state.
    """

    def _get_width_height_linked(self, context: Window.ActionContext) -> bool:
        return bool(context.parameters.get("width_height_linked", False))


class DecreasePixelTimeAction(FrameParametersStepAction):
    action_id = "scan.decrease_pixel_time"
    action_name = _("Decrease Pixel Time")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_decreased_pixel_time(frame_parameters)


class IncreasePixelTimeAction(FrameParametersStepAction):
    action_id = "scan.increase_pixel_time"
    action_name = _("Increase Pixel Time")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_increased_pixel_time(frame_parameters)


class DecreaseFieldOfViewAction(FrameParametersStepAction):
    action_id = "scan.decrease_fov"
    action_name = _("Decrease Field of View")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_decreased_fov(frame_parameters)


class IncreaseFieldOfViewAction(FrameParametersStepAction):
    action_id = "scan.increase_fov"
    action_name = _("Increase Field of View")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_increased_fov(frame_parameters)


class DecreaseWidthAction(SizeStepAction):
    action_id = "scan.decrease_width"
    action_name = _("Decrease Scan Width")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_decreased_width(frame_parameters, self._get_width_height_linked(context))


class IncreaseWidthAction(SizeStepAction):
    action_id = "scan.increase_width"
    action_name = _("Increase Scan Width")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_increased_width(frame_parameters, self._get_width_height_linked(context))


class DecreaseHeightAction(SizeStepAction):
    action_id = "scan.decrease_height"
    action_name = _("Decrease Scan Height")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_decreased_height(frame_parameters, self._get_width_height_linked(context))


class IncreaseHeightAction(SizeStepAction):
    action_id = "scan.increase_height"
    action_name = _("Increase Scan Height")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_increased_height(frame_parameters, self._get_width_height_linked(context))


class DecreaseSubscanWidthAction(FrameParametersStepAction):
    action_id = "scan.decrease_subscan_width"
    action_name = _("Decrease Subscan Width")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_decreased_subscan_width(frame_parameters)


class IncreaseSubscanWidthAction(FrameParametersStepAction):
    action_id = "scan.increase_subscan_width"
    action_name = _("Increase Subscan Width")

    def _get_stepped_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters, context: Window.ActionContext) -> scan_base.ScanFrameParameters:
        return with_increased_subscan_width(frame_parameters)


class PMTAction(ScanAction):
    """An action on the PMT of the scan channel given by the channel_index parameter.

    The action does nothing without a channel index. Its summary also names the channel index.
    """

    def get_action_summary(self, context: Window.ActionContext) -> str:
        scan_hardware_source = _get_scan_hardware_source(context)
        hardware_source_name = scan_hardware_source.display_name if scan_hardware_source else "N/A"
        return f"{self.action_id}: {self.action_name} ({hardware_source_name}, channel {context.parameters.get('channel_index')})"


class DecreasePMTAction(PMTAction):
    action_id = "scan.decrease_pmt"
    action_name = _("Decrease PMT")

    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        channel_index = context.parameters.get("channel_index")
        if (scan_hardware_source := _get_scan_hardware_source(context)) and isinstance(channel_index, int):
            decrease_pmt(scan_hardware_source, channel_index)
        return Window.ActionResult(Window.ActionStatus.FINISHED)


class IncreasePMTAction(PMTAction):
    action_id = "scan.increase_pmt"
    action_name = _("Increase PMT")

    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        channel_index = context.parameters.get("channel_index")
        if (scan_hardware_source := _get_scan_hardware_source(context)) and isinstance(channel_index, int):
            increase_pmt(scan_hardware_source, channel_index)
        return Window.ActionResult(Window.ActionStatus.FINISHED)


Window.register_action(DecreasePixelTimeAction())
Window.register_action(IncreasePixelTimeAction())
Window.register_action(DecreaseFieldOfViewAction())
Window.register_action(IncreaseFieldOfViewAction())
Window.register_action(DecreaseWidthAction())
Window.register_action(IncreaseWidthAction())
Window.register_action(DecreaseHeightAction())
Window.register_action(IncreaseHeightAction())
Window.register_action(DecreaseSubscanWidthAction())
Window.register_action(IncreaseSubscanWidthAction())
Window.register_action(DecreasePMTAction())
Window.register_action(IncreasePMTAction())
