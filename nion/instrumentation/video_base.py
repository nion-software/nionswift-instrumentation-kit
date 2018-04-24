# standard libraries
import abc

# typing
# None

# third party libraries
import numpy

# local libraries
from nion.swift.model import HardwareSource
from nion.utils import Registry


class AbstractVideoCamera(abc.ABC):

    @abc.abstractmethod
    def close(self) -> None:
        """Close the camera."""
        ...

    @abc.abstractmethod
    def start_acquisition(self) -> None:
        """Start live acquisition. Required before using acquire_image."""
        ...

    @abc.abstractmethod
    def acquire_data(self) -> numpy.ndarray:
        """Acquire the most recent image and return a data element dict.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict. Inside the 'properties' dict you must include 'frame_number' as an int.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).
        """
        ...

    @abc.abstractmethod
    def stop_acquisition(self) -> None:
        """Stop live acquisition."""
        ...


class AcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, hardware_source_id: str, camera: AbstractVideoCamera, display_name: str):
        super().__init__(True)
        self.__hardware_source_id = hardware_source_id
        self.__camera = camera
        self.__display_name = display_name

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self.__camera.start_acquisition()
        return True

    def _acquire_data_elements(self):
        data = self.__camera.acquire_data()
        data_element = {
            "version": 1,
            "data": data,
            "properties": {
                "hardware_source_name": self.__display_name,
                "hardware_source_id": self.__hardware_source_id,
            }
        }
        return [data_element]

    def _stop_acquisition(self) -> None:
        self.__camera.stop_acquisition()
        super()._stop_acquisition()


class VideoHardwareSource(HardwareSource.HardwareSource):

    def __init__(self, camera: AbstractVideoCamera):
        super().__init__(camera.camera_id, camera.camera_name)
        self.features["is_video"] = True
        self.add_data_channel()
        self.__camera = camera
        self.__acquisition_task = None

    def close(self):
        super().close()
        # keep the camera device around until super close is called, since super may do something that requires it.
        camera_close_method = getattr(self.__camera, "close", None)
        if callable(camera_close_method):
            camera_close_method()
        self.__camera = None

    @property
    def video_camera(self) -> AbstractVideoCamera:
        return self.__camera

    def _create_acquisition_view_task(self) -> AcquisitionTask:
        return AcquisitionTask(self.hardware_source_id, self.__camera, self.display_name)


_component_registered_listener = None
_component_unregistered_listener = None

def run():
    def component_registered(component, component_types):
        if "video_device" in component_types:
            hardware_source = VideoHardwareSource(component)
            HardwareSource.HardwareSourceManager().register_hardware_source(hardware_source)

    def component_unregistered(component, component_types):
        if "video_device" in component_types:
            for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
                if getattr(hardware_source, "video_camera"):
                    HardwareSource.HardwareSourceManager().unregister_hardware_source(hardware_source)

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    for component in Registry.get_components_by_type("video_device"):
        component_registered(component, {"video_device"})
