from __future__ import annotations

# standard libraries
import abc
import json
import os
import pathlib
import typing

# typing
# None

# third party libraries
import numpy
import numpy.typing

# local libraries
from nion.instrumentation import HardwareSource
from nion.swift.model import ImportExportManager
from nion.utils import ListModel
from nion.utils import Registry
from nion.utils import StructuredModel

_NDArray = numpy.typing.NDArray[typing.Any]


class AbstractVideoCamera(typing.Protocol):

    camera_id: typing.Optional[str]
    camera_name: typing.Optional[str]

    def close(self) -> None:
        """Close the camera."""
        ...

    def start_acquisition(self) -> None:
        """Start live acquisition. Required before using acquire_image."""
        ...

    def acquire_data(self) -> _NDArray:
        """Acquire the most recent image and return a data element dict.

        The data element dict should have a 'data' element with the ndarray of the data and a 'properties' element
        with a dict. Inside the 'properties' dict you must include 'frame_number' as an int.

        The 'data' may point to memory allocated in low level code, but it must remain valid and unmodified until
        released (Python reference count goes to zero).
        """
        ...

    def stop_acquisition(self) -> None:
        """Stop live acquisition."""
        ...

    def update_settings(self, settings: typing.Mapping[str, typing.Any]) -> None:
        """Update the settings."""
        ...


class AcquisitionTask(HardwareSource.AcquisitionTask):

    def __init__(self, hardware_source_id: str, camera: AbstractVideoCamera, display_name: str) -> None:
        super().__init__(True)
        self.__hardware_source_id = hardware_source_id
        self.__camera = camera
        self.__display_name = display_name

    def _start_acquisition(self) -> bool:
        if not super()._start_acquisition():
            return False
        self.__camera.start_acquisition()
        return True

    def _acquire_data_elements(self) -> typing.Sequence[ImportExportManager.DataElementType]:
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


class VideoHardwareSource(HardwareSource.ConcreteHardwareSource):

    def __init__(self, camera: AbstractVideoCamera) -> None:
        super().__init__(getattr(camera, "camera_id"), getattr(camera, "camera_name"))
        self.features["is_video"] = True
        self.add_data_channel()
        self.__camera = camera
        self.__acquisition_task = None

    def close(self) -> None:
        super().close()
        # keep the camera device around until super close is called, since super may do something that requires it.
        camera_close_method = getattr(self.__camera, "close", None)
        if callable(camera_close_method):
            camera_close_method()
        self.__camera = typing.cast(typing.Any, None)

    @property
    def video_device(self) -> AbstractVideoCamera:
        return self.__camera

    def _create_acquisition_view_task(self) -> AcquisitionTask:
        return AcquisitionTask(self.hardware_source_id, self.__camera, self.display_name)


class VideoDeviceFactoryLike(typing.Protocol):
    factory_id: str
    display_name: str
    def make_video_device(self, settings: typing.Mapping[str, typing.Any]) -> typing.Optional[AbstractVideoCamera]: ...
    def describe_settings(self) -> typing.Sequence[typing.Mapping[str, typing.Any]]: ...
    # these break encapsulation - UI should be separate, but the editor handler is UI
    def create_editor_handler(self, settings: typing.Optional[StructuredModel.ModelLike]) -> typing.Any: ...
    def get_editor_description(self) -> typing.Any: ...


class VideoDeviceInstance:
    def __init__(self, video_device_factory: typing.Optional[VideoDeviceFactoryLike], video_device: typing.Optional[AbstractVideoCamera], settings: typing.Mapping[str, typing.Any]) -> None:
        self.video_device_factory = video_device_factory
        self.video_device = video_device
        self.settings = settings


class VideoConfiguration:

    def __init__(self) -> None:
        self.__config_file: typing.Optional[pathlib.Path] = None

        # the active video sources (hardware sources). this list is updated when a video camera device is registered or
        # unregistered with the hardware source manager.
        self.video_sources = ListModel.ListModel[VideoHardwareSource]()

        # the list of instances of video cameras. this is similar to the video sources but is the devices plus settings
        # for the device. some devices might not have instances if the factory to create the instance hasn't been
        # registered yet.
        self.__instances: typing.List[VideoDeviceInstance] = list()

        # the list of video device factories. this is populated by responding to the registry messages.
        self.__video_device_factories: typing.List[VideoDeviceFactoryLike] = list()

        def component_registered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
            if "video_device_factory" in component_types:
                if not component in self.__video_device_factories:
                    self.__video_device_factories.append(component)
                self.__make_video_devices()

        def component_unregistered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
            if "video_device_factory" in component_types:
                if component in self.__video_device_factories:
                    self.__video_device_factories.remove(component)

        self.__component_registered_listener = Registry.listen_component_registered_event(component_registered)
        self.__component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

        for component in Registry.get_components_by_type("video_device_factory"):
            component_registered(component, {"video_device_factory"})

    def close(self) -> None:
        self.__component_registered_listener.close()
        self.__component_registered_listener = typing.cast(typing.Any, None)
        self.__component_unregistered_listener.close()
        self.__component_unregistered_listener = typing.cast(typing.Any, None)

    def _remove_video_device(self, video_device: AbstractVideoCamera) -> None:
        for instance in self.__instances:
            if instance.video_device == video_device:
                self.__instances.remove(instance)
                break

    def __make_video_devices(self) -> None:
        for video_device_factory in self.__video_device_factories:
            for instance in self.__instances:
                if not instance.video_device:
                    instance.video_device_factory = video_device_factory
                    video_device = video_device_factory.make_video_device(instance.settings)
                    if video_device:
                        instance.video_device = video_device
                        Registry.register_component(instance.video_device, {"video_device"})

    def load(self, config_file: pathlib.Path) -> None:
        # read the configured video cameras from the config file and populate the instances list.
        self.__config_file = config_file
        try:
            if config_file.is_file():
                with open(config_file) as f:
                    settings_list = json.load(f)
                if isinstance(settings_list, list):
                    for settings in settings_list:
                        self.__instances.append(VideoDeviceInstance(None, None, settings))
            self.__make_video_devices()
        except Exception as e:
            pass

    def __save(self) -> None:
        # atomically overwrite
        if self.__config_file:
            temp_filepath = self.__config_file.with_suffix(".temp")
            with open(temp_filepath, "w") as fp:
                json.dump([instance.settings for instance in self.__instances], fp, skipkeys=True, indent=4)
            os.replace(temp_filepath, self.__config_file)

    def get_settings_model(self, hardware_source: VideoHardwareSource) -> typing.Optional[StructuredModel.ModelLike]:
        for instance in self.__instances:
            if instance.video_device and instance.video_device == hardware_source.video_device:
                fields = [
                    StructuredModel.define_field("driver", StructuredModel.STRING),
                    StructuredModel.define_field("device_id", StructuredModel.STRING),
                    StructuredModel.define_field("name", StructuredModel.STRING),
                ]
                values = {
                    "driver": instance.settings.get("driver", None),
                    "device_id": instance.video_device.camera_id,
                    "name": instance.video_device.camera_name,
                }
                video_device_factory = instance.video_device_factory
                if video_device_factory:
                    for setting_description in video_device_factory.describe_settings():
                        setting_name = setting_description["name"]
                        fields.append(StructuredModel.define_field(setting_name, setting_description["type"]))
                        setting_value = instance.settings.get(setting_name, None)
                        if setting_value is not None:
                            values[setting_name] = setting_value

                schema = StructuredModel.define_record("settings", fields)
                model = StructuredModel.build_model(schema, value=values)
                return model
        return None

    def set_settings_model(self, hardware_source: VideoHardwareSource, settings_model: StructuredModel.ModelLike) -> None:
        video_device = hardware_source.video_device
        for instance in self.__instances:
            if instance.video_device == video_device:
                instance.settings = typing.cast(typing.Mapping[str, typing.Any], settings_model.to_dict_value())
                video_device.update_settings(instance.settings)
                video_device.camera_id = getattr(settings_model, "device_id")
                video_device.camera_name = getattr(settings_model, "name")
                hardware_source.hardware_source_id = getattr(settings_model, "device_id")
                hardware_source.display_name = getattr(settings_model, "name")
                self.__save()
                break

    def __generate_device_id(self) -> str:
        n = 1
        while True:
            device_id = "video_device_" + str(n)
            if not HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(device_id):
                break
            n += 1
        return device_id

    def create_hardware_source(self, video_device_factory: VideoDeviceFactoryLike) -> None:
        settings = {"driver": video_device_factory.factory_id, "device_id": self.__generate_device_id()}
        self.__instances.append(VideoDeviceInstance(video_device_factory, None, settings))
        self.__make_video_devices()
        self.__save()

    def remove_hardware_source(self, hardware_source: VideoHardwareSource) -> None:
        hardware_source.abort_playing()
        component = hardware_source.video_device
        Registry.unregister_component(component)
        self.__save()


video_configuration = VideoConfiguration()

_component_registered_listener = None
_component_unregistered_listener = None

def run() -> None:
    def component_registered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
        if "video_device" in component_types:
            hardware_source = VideoHardwareSource(component)
            Registry.register_component(hardware_source, {"hardware_source", "video_hardware_source"})
            HardwareSource.HardwareSourceManager().register_hardware_source(hardware_source)
            video_configuration.video_sources.append_item(hardware_source)

    def component_unregistered(component: Registry._ComponentType, component_types: typing.Set[str]) -> None:
        if "video_device" in component_types:
            for hardware_source in HardwareSource.HardwareSourceManager().hardware_sources:
                if isinstance(hardware_source, VideoHardwareSource) and getattr(hardware_source, "video_device", None) and hardware_source.video_device == component:
                    video_configuration.video_sources.remove_item(video_configuration.video_sources.items.index(hardware_source))
                    video_configuration._remove_video_device(component)
                    Registry.unregister_component(hardware_source)
                    HardwareSource.HardwareSourceManager().unregister_hardware_source(hardware_source)

    global _component_registered_listener
    global _component_unregistered_listener

    _component_registered_listener = Registry.listen_component_registered_event(component_registered)
    _component_unregistered_listener = Registry.listen_component_unregistered_event(component_unregistered)

    for component in Registry.get_components_by_type("video_device"):
        component_registered(component, {"video_device"})
