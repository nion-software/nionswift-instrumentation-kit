from . import CameraControlPanel
from . import ScanControlPanel
from . import ScanAcquisition

from nion.instrumentation import camera_base

def run():
    camera_base.run()
    CameraControlPanel.run()
    ScanControlPanel.run()
