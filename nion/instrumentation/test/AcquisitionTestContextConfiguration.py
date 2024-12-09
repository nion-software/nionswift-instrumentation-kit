import math
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


class ScanModule(scan_base.ScanModule):
    def __init__(self, instrument: InstrumentDevice.Instrument, device_id: str) -> None:
        self.stem_controller_id = instrument.instrument_id
        self.device = ScanDevice.Device(device_id, "Scan", instrument)
        setattr(self.device, "priority", 20)
        scan_modes = (
            scan_base.ScanSettingsMode("Fast", "fast", ScanDevice.ScanFrameParameters(pixel_size=(256, 256), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.1)),
            scan_base.ScanSettingsMode("Slow", "slow", ScanDevice.ScanFrameParameters(pixel_size=(512, 512), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 0.4)),
            scan_base.ScanSettingsMode("Record", "record", ScanDevice.ScanFrameParameters(pixel_size=(1024, 1024), pixel_time_us=1, fov_nm=instrument.stage_size_nm * 1.0))
        )
        self.settings = scan_base.ScanSettings(scan_modes, lambda d: ScanDevice.ScanFrameParameters(d), 0, 2)


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
        data = numpy.random.randn(shape.height // binning_shape.height, shape.width // binning_shape.width) * exposure_s
        return DataAndMetadata.new_data_and_metadata(data)


class ScanDataGenerator:
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
        self.instrument_id = "test_stem_controller"
        self.ronchigram_camera_device_id = "test_ronchigram_camera"
        self.eels_camera_device_id = "test_eels_camera"
        self.instrument = InstrumentDevice.Instrument(self.instrument_id, ScanDataGenerator())
        self.scan_module = ScanModule(self.instrument, "test_scan_device")
        self.ronchigram_camera_settings = CameraDevice.CameraSettings(self.ronchigram_camera_device_id)
        self.eels_camera_settings = CameraDevice.CameraSettings(self.eels_camera_device_id)
        self.ronchigram_camera_device = CameraDevice.Camera(self.ronchigram_camera_device_id, "ronchigram", "Ronchigram", CameraSimulator(None), self.instrument)
        self.eels_camera_device = CameraDevice.Camera(self.eels_camera_device_id, "eels", "EELS", CameraSimulator(Geometry.IntSize(256, 1024)), self.instrument)
