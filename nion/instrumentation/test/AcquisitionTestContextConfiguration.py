from __future__ import annotations

import logging
import math
import pathlib
import shutil
import threading
import typing

import numpy
import numpy.typing
import scipy

from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.device_kit import CameraDevice
from nion.device_kit import InstrumentDevice
from nion.device_kit import ScanDevice
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.utils import Geometry
from nion.utils import Registry


class ScanBoxSimulator(ScanDevice.ScanSimulatorLike):
    def __init__(self, scan_data_generator: ScanDevice.ScanDataGeneratorLike) -> None:
        self.__scan_data_generator = scan_data_generator
        self.__blanker_signal_condition = threading.Condition()
        self.__advance_pixel_lock = threading.RLock()
        self.__current_pixel_flat = 0
        self.__scan_shape_pixels = Geometry.IntSize()
        self.__pixel_size_nm = Geometry.FloatSize()
        self.flyback_pixels = 2
        self.__n_flyback_pixels = 0
        self.__current_line = 0
        self.external_clock = False

    def reset_frame(self) -> None:
        with self.__advance_pixel_lock:
            self.__current_pixel_flat = 0
            self.__current_line = 0
            self.__n_flyback_pixels = 0

    @property
    def scan_shape_pixels(self) -> Geometry.IntSize:
        return self.__scan_shape_pixels

    @scan_shape_pixels.setter
    def scan_shape_pixels(self, shape: typing.Union[Geometry.IntSize, Geometry.SizeIntTuple]) -> None:
        self.__scan_shape_pixels = Geometry.IntSize.make(shape)

    @property
    def pixel_size_nm(self) -> Geometry.FloatSize:
        return self.__pixel_size_nm

    @pixel_size_nm.setter
    def pixel_size_nm(self, size: typing.Union[Geometry.FloatSize, Geometry.SizeFloatTuple]) -> None:
        self.__pixel_size_nm = Geometry.FloatSize.make(size)

    @property
    def probe_position_pixels(self) -> Geometry.IntPoint:
        if self.__scan_shape_pixels.width != 0:
            current_pixel_flat = self.__current_pixel_flat
            return Geometry.IntPoint(y=current_pixel_flat // self.__scan_shape_pixels.width, x=current_pixel_flat % self.__scan_shape_pixels.width)
        return Geometry.IntPoint()

    @property
    def current_pixel_flat(self) -> int:
        return self.__current_pixel_flat

    @property
    def blanker_signal_condition(self) -> threading.Condition:
        return self.__blanker_signal_condition

    def _advance_pixel(self, n: int) -> None:
        with self.__advance_pixel_lock:
            next_line = (self.__current_pixel_flat + n) // self.__scan_shape_pixels.width
            if next_line > self.__current_line:
                self.__n_flyback_pixels = 0
                self.__current_line = next_line
                with self.__blanker_signal_condition:
                    self.__blanker_signal_condition.notify_all()
            if self.__n_flyback_pixels < self.flyback_pixels:
                new_flyback_pixels = min(self.flyback_pixels - self.__n_flyback_pixels, n)
                n -= new_flyback_pixels
                self.__n_flyback_pixels += new_flyback_pixels
            self.__current_pixel_flat += n

    def advance_pixel(self) -> None:
        if self.external_clock:
            self._advance_pixel(1)

    def generate_scan_data(self, instrument: InstrumentDevice.Instrument, scan_frame_parameters: ScanDevice.ScanFrameParameters) -> numpy.typing.NDArray[numpy.float32]:
        return self.__scan_data_generator.generate_scan_data(instrument, scan_frame_parameters)


class ScanModule(scan_base.ScanModule):
    def __init__(self, instrument: InstrumentDevice.Instrument, device_id: str, scan_data_generator: ScanDevice.ScanDataGeneratorLike) -> None:
        self.stem_controller_id = instrument.instrument_id
        self.device = ScanDevice.Device(device_id, "Scan", instrument, ScanBoxSimulator(scan_data_generator))
        setattr(self.device, "priority", 20)
        scan_modes = (
            scan_base.ScanSettingsMode("Fast", "fast", ScanDevice.ScanFrameParameters(pixel_size=(256, 256), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.1)),
            scan_base.ScanSettingsMode("Slow", "slow", ScanDevice.ScanFrameParameters(pixel_size=(512, 512), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.4)),
            scan_base.ScanSettingsMode("Record", "record", ScanDevice.ScanFrameParameters(pixel_size=(1024, 1024), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 1.0))
        )
        self.settings = scan_base.ScanSettings(device_id, scan_modes, lambda d: ScanDevice.ScanFrameParameters(d), 0, 2)


class CameraSimulator:
    def __init__(self, sensor_dimensions: typing.Optional[Geometry.IntSize]) -> None:
        self.__data_value = 0
        self.__sensor_dimensions = sensor_dimensions

    def close(self) -> None:
        pass

    def get_dimensional_calibrations(self, readout_area: typing.Optional[Geometry.IntRect], binning_shape: typing.Optional[Geometry.IntSize]) -> typing.Sequence[Calibration.Calibration]:
        dimensional_calibrations = [
            Calibration.Calibration(),
            Calibration.Calibration()
        ]
        return dimensional_calibrations

    def get_frame_data(self, readout_area: Geometry.IntRect, binning_shape: Geometry.IntSize, exposure_s: float, scan_context: stem_controller.ScanContext, probe_position: typing.Optional[Geometry.FloatPoint]) -> DataAndMetadata.DataAndMetadata:
        self.__data_value += 1
        shape = self.__sensor_dimensions if self.__sensor_dimensions else readout_area.size
        data = numpy.random.rand(shape.height // binning_shape.height, shape.width // binning_shape.width) * exposure_s
        return DataAndMetadata.new_data_and_metadata(data)


class ValueManager(InstrumentDevice.ValueManagerLike):
    def __init__(self) -> None:
        self.__values = {
            "EHT": 100000.0,
            "C10": 5e-7,
            "ronchigram_y_scale": 1.0,
            "ronchigram_y_offset": 0.0,
            "ronchigram_x_scale": 1.0,
            "ronchigram_x_offset": 0.0,
            "CSH.y": 0.0,
            "CSH.x": 0.0,
            "eels_x_scale": 1.0,
            "eels_x_offset": 0.0,
            "eels_y_scale": 1.0,
            "eels_y_offset": 0.0,
            "EELS_MagneticShift_Offset": -20.0
        }

    def get_value(self, name: str) -> typing.Optional[float]:
        return self.__values.get(name)

    def set_value(self, name: str, value: float) -> bool:
        self.__values[name] = value
        return True

    def inform_value(self, name: str, value: float) -> bool:
        return self.set_value(name, value)

    def get_value_2d(self, name: str, default_value: typing.Optional[Geometry.FloatPoint] = None, *, axis: typing.Optional[stem_controller.AxisType] = None) -> Geometry.FloatPoint:
        return Geometry.FloatPoint()

    def set_value_2d(self, name: str, value: Geometry.FloatPoint, *, axis: typing.Optional[stem_controller.AxisType] = None) -> bool:
        return True

    def inform_control_2d(self, name: str, value: Geometry.FloatPoint, *, axis: stem_controller.AxisType) -> bool:
        return True

    def get_reference_setting_index(self, settings_control: str) -> int:
        return 0


class AxisManager(InstrumentDevice.AxisManagerLike):

    @property
    def supported_axis_descriptions(self) -> typing.Sequence[stem_controller.AxisDescription]:
        return list()

    def axis_transform_point(self, point: Geometry.FloatPoint, from_axis: stem_controller.AxisDescription, to_axis: stem_controller.AxisDescription) -> Geometry.FloatPoint:
        return point


class ScanDataGenerator(ScanDevice.ScanDataGeneratorLike):
    def __init__(self) -> None:
        random_state = numpy.random.get_state()
        numpy.random.seed(100)
        pattern = scipy.ndimage.zoom(numpy.abs(numpy.random.randn(40, 40)), 25) + scipy.ndimage.zoom(numpy.abs(numpy.random.randn(100, 100)), 10)
        numpy.random.set_state(random_state)
        self.__pattern = typing.cast(numpy.typing.NDArray[numpy.float32], pattern)

    def generate_scan_data(self, instrument: InstrumentDevice.Instrument, scan_frame_parameters: ScanDevice.ScanFrameParameters) -> numpy.typing.NDArray[numpy.float32]:
        pattern = self.__pattern
        shift_nm = Geometry.FloatPoint(instrument.GetVal("CSH.y") * 1e9, instrument.GetVal("CSH.x") * 1e9)  # for drift tests
        size = scan_frame_parameters.size
        fov_size_nm = scan_frame_parameters.fov_size_nm
        rotation = scan_frame_parameters.rotation_rad
        center_nm = scan_frame_parameters.center_nm
        y_start = (50 + center_nm.y + shift_nm.y - fov_size_nm.height / 2) / 100 * pattern.shape[0]
        y_length = fov_size_nm.height / 100 * pattern.shape[0]
        x_start = (50 + center_nm.x + shift_nm.x - fov_size_nm.width / 2) / 100 * pattern.shape[1]
        x_length = fov_size_nm.width / 100 * pattern.shape[1]
        iy, ix = numpy.meshgrid(numpy.arange(size.width), numpy.arange(size.height))
        y = iy * y_length / size.height - y_length / 2
        x = ix * x_length / size.width - x_length / 2
        angle_sin = math.sin(-rotation)
        angle_cos = math.cos(-rotation)
        coordinates = [y_start + y_length / 2 + (x * angle_cos - y * angle_sin), x_start + x_length / 2 + (y * angle_cos + x * angle_sin)]
        return typing.cast(numpy.typing.NDArray[numpy.float32], scipy.ndimage.map_coordinates(pattern, coordinates, order=1) + numpy.random.randn(*size) * 0.1)


class AcquisitionTestContextConfiguration:
    def __init__(self) -> None:
        configuration_location = pathlib.Path.cwd() / "test_data"
        if configuration_location.exists():
            shutil.rmtree(configuration_location)
        pathlib.Path.mkdir(configuration_location, exist_ok=True)
        self.configuration_location = configuration_location
        self.instrument_id = "test_stem_controller"
        self.ronchigram_camera_device_id = "test_ronchigram_camera"
        self.eels_camera_device_id = "test_eels_camera"
        self.instrument = InstrumentDevice.Instrument(self.instrument_id, ValueManager(), AxisManager())
        self.scan_module = ScanModule(self.instrument, "test_scan_device", ScanDataGenerator())
        self.ronchigram_camera_settings = CameraDevice.CameraSettings(self.ronchigram_camera_device_id, 0.005)
        self.eels_camera_settings = CameraDevice.CameraSettings(self.eels_camera_device_id, 0.005)
        self.ronchigram_camera_device = CameraDevice.Camera(self.ronchigram_camera_device_id, "ronchigram", "Ronchigram", CameraSimulator(None), self.instrument)
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "eels", "EELS", CameraSimulator(Geometry.IntSize(256, 1024)), self.instrument)

    def run(self) -> None:
        logging.disable(logging.CRITICAL)
        try:
            Registry.register_component(self.instrument, {"instrument_controller", "stem_controller"})
            component_types = {"camera_module"}  # the set of component types that this component represents
            setattr(self.ronchigram_camera_device, "camera_panel_type", "ronchigram")
            Registry.register_component(CameraDevice.CameraModule("test_stem_controller", self.ronchigram_camera_device, self.ronchigram_camera_settings), component_types)
            setattr(self.eels_camera_device, "camera_panel_type", "eels")
            Registry.register_component(CameraDevice.CameraModule("test_stem_controller", self.eels_camera_device, self.eels_camera_settings), component_types)
            Registry.register_component(self.scan_module, {"scan_module"})
        finally:
            logging.disable(logging.NOTSET)

    def stop(self) -> None:
        for component in Registry.get_components_by_type("camera_module"):
            Registry.unregister_component(component, {"camera_module"})
        Registry.unregister_component(Registry.get_component("scan_module"), {"scan_module"})
        Registry.unregister_component(Registry.get_component("stem_controller"), {"instrument_controller", "stem_controller"})
