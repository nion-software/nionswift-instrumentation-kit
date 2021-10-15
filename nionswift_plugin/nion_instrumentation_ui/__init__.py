from __future__ import annotations

import logging
import pathlib
import typing

from . import AcquisitionPanel
from . import CameraControlPanel
from . import ScanControlPanel
from . import ScanAcquisition
from . import AcquisitionRecorder
from . import MultiAcquirePanel
from . import MultipleShiftEELSAcquire
from . import VideoControlPanel

from nion.instrumentation import camera_base
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.instrumentation import video_base

if typing.TYPE_CHECKING:
    from nion.swift.model import DocumentModel


configuration_location: typing.Optional[pathlib.Path] = None


class STEMControllerExtension:

    # required for Swift to recognize this as an extension class.
    extension_id = "nion.stem_controller"

    def __init__(self, api_broker: typing.Any) -> None:
        # grab the api object.
        api = api_broker.get_api(version="1", ui_version="1")
        self.__scan_context_controller = None

        def document_model_available(document_model: DocumentModel.DocumentModel) -> None:
            self.__scan_context_controller = stem_controller.ScanContextController(document_model, api.application._application.event_loop)

        self.__document_model_available_event_listener = api.application._application.document_model_available_event.listen(document_model_available)
        config_file = api.application.configuration_location / pathlib.Path("video_device_config.json")
        logging.info("Video device configuration: " + str(config_file))
        video_base.video_configuration.load(config_file)
        global configuration_location
        configuration_location = pathlib.Path(api.application.configuration_location)

    def close(self) -> None:
        if self.__scan_context_controller:
            self.__scan_context_controller.close()
            self.__scan_context_controller = None
        self.__document_model_available_event_listener.close()
        self.__document_model_available_event_listener = None


def run() -> None:
    global configuration_location
    if configuration_location:
        camera_base.run(configuration_location)
        scan_base.run()
        video_base.run()
        CameraControlPanel.run()
        ScanControlPanel.run()
        MultipleShiftEELSAcquire.run()
        VideoControlPanel.run()
