from __future__ import annotations

# standard libraries
import gettext
import typing

# third party libraries
# None

# local libraries
from nion.instrumentation import HardwareSource
from nion.ui import Window

if typing.TYPE_CHECKING:
    pass


_ = gettext.gettext


class StartPlayingAction(Window.Action):
    action_id = "acquisition.start_playing"
    action_name = _("Start Playing")

    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        hardware_source_id = context.parameters.get("hardware_source_id")
        if hardware_source_id:
            hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
            frame_parameters_d = context.parameters.get("frame_parameters")
            if hardware_source:
                frame_parameters = hardware_source.get_frame_parameters_from_dict(frame_parameters_d) if frame_parameters_d else None
                # frame parameters object must be passed by keyword, not positional.
                hardware_source.start_playing(frame_parameters=frame_parameters)
        return Window.ActionResult(Window.ActionStatus.FINISHED)

    def get_action_summary(self, context: Window.ActionContext) -> str:
        hardware_source_id = context.parameters.get("hardware_source_id")
        hardware_source_name = "N/A"
        if hardware_source_id:
            hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
            if hardware_source:
                hardware_source_name = hardware_source.display_name
        frame_parameters_d = context.parameters.get("frame_parameters")
        frame_parameters_str = f" {frame_parameters_d}" if frame_parameters_d else ""
        return f"{self.action_id}: {self.action_name} ({hardware_source_name}){frame_parameters_str}"


class StopPlayingAction(Window.Action):
    action_id = "acquisition.stop_playing"
    action_name = _("Stop Playing")

    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        hardware_source_id = context.parameters.get("hardware_source_id")
        if hardware_source_id:
            hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
            if hardware_source:
                hardware_source.stop_playing()
        return Window.ActionResult(Window.ActionStatus.FINISHED)

    def get_action_summary(self, context: Window.ActionContext) -> str:
        hardware_source_id = context.parameters.get("hardware_source_id")
        hardware_source_name = "N/A"
        if hardware_source_id:
            hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
            if hardware_source:
                hardware_source_name = hardware_source.display_name
        return f"{self.action_id}: {self.action_name} ({hardware_source_name})"


class AbortPlayingAction(Window.Action):
    action_id = "acquisition.abort_playing"
    action_name = _("Abort Playing")

    def execute(self, context: Window.ActionContext) -> Window.ActionResult:
        hardware_source_id = context.parameters.get("hardware_source_id")
        if hardware_source_id:
            hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
            if hardware_source:
                hardware_source.abort_playing()
        return Window.ActionResult(Window.ActionStatus.FINISHED)

    def get_action_summary(self, context: Window.ActionContext) -> str:
        hardware_source_id = context.parameters.get("hardware_source_id")
        hardware_source_name = "N/A"
        if hardware_source_id:
            hardware_source = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(hardware_source_id)
            if hardware_source:
                hardware_source_name = hardware_source.display_name
        return f"{self.action_id}: {self.action_name} ({hardware_source_name})"


Window.register_action(StartPlayingAction())
Window.register_action(StopPlayingAction())
Window.register_action(AbortPlayingAction())
