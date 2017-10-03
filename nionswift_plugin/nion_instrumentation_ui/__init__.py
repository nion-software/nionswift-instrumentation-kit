from . import CameraControlPanel
from . import ScanControlPanel
from . import ScanAcquisition
from . import MultipleShiftEELSAcquire

from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller


class STEMControllerExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.stem_controller"

    def __init__(self, api_broker):
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        self.__probe_view_controller = stem_controller.ProbeViewController(api.application._application.event_loop)

    def close(self):
        self.__probe_view_controller.close()
        self.__probe_view_controller = None


def run():
    camera_base.run()
    scan_base.run()
    CameraControlPanel.run()
    ScanControlPanel.run()
    MultipleShiftEELSAcquire.run()
