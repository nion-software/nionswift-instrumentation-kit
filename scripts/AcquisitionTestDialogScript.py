import asyncio
import dataclasses
import enum
import logging
import time
import typing

from nion.data import DataAndMetadata
from nion.experimental import Wizard
from nion.instrumentation import Acquisition
from nion.instrumentation import camera_base as CameraBase
from nion.instrumentation import scan_base as ScanBase
from nion.instrumentation import stem_controller as STEMController
from nion.swift import Application
from nion.swift import DocumentController
from nion.swift import Facade
from nion.swift.model import PlugInManager
from nionswift_plugin.nion_instrumentation_ui import AcquisitionPanel
from nion.ui import Declarative
from nion.ui import Dialog
from nion.utils import Converter
from nion.utils import Geometry
from nion.utils import Model
from nion.utils import Observable
from nion.utils import Registry
from nion.utils import Stream
from scipy.stats.contingency import margins


class AcquisitionTestState(enum.Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    SUCCEEDED = 2
    FAILED = 3
    NOT_AVAILABLE = 4


class AcquisitionTestResultMessage:
    def __init__(self, status: bool = False, message: str = str(), state: AcquisitionTestState | None = None) -> None:
        self.status = state if state is not None else (AcquisitionTestState.SUCCEEDED if status else AcquisitionTestState.FAILED)
        self.message = message


class AcquisitionTestResult:
    def __init__(self) -> None:
        self.messages = list[AcquisitionTestResultMessage]()
        self.result: AcquisitionPanel._AcquireResult | None = None

    def delete_display_items(self, window: DocumentController.DocumentController) -> None:
        ar = self.result
        assert ar
        data_item_map = ar.data_item_map
        for channel in data_item_map.keys():
            data_item = data_item_map[channel]
            if data_item:
                display_item = window.document_model.get_display_item_for_data_item(data_item)
                if display_item:
                    window.delete_display_items([display_item])


def make_not_available_result(message: str) -> AcquisitionTestResult:
    result = AcquisitionTestResult()
    result.messages.append(AcquisitionTestResultMessage(message=message, state=AcquisitionTestState.NOT_AVAILABLE))
    return result


def check_results(ar: AcquisitionPanel._AcquireResult, logger: logging.Logger, expected_results: typing.Mapping[Acquisition.Channel, tuple[DataAndMetadata.ShapeType, DataAndMetadata.DataDescriptor]]) -> AcquisitionTestResult:
    result = AcquisitionTestResult()
    result.result = ar
    if not ar.success:
        result.messages.append(AcquisitionTestResultMessage(False, "Acquisition failed"))
    for channel, expected_dimension in expected_results.items():
        data_item = ar.data_item_map.get(channel)
        if data_item:
            data_shape = data_item.data_shape
            data_and_metadata = data_item.data_and_metadata
            if data_and_metadata:
                data_descriptor = data_and_metadata.data_descriptor
                result.messages.append(AcquisitionTestResultMessage(data_shape == expected_dimension[0], f"Channel {channel} data shape {data_shape} matches expected {expected_dimension[0]}"))
                result.messages.append(AcquisitionTestResultMessage(data_descriptor == expected_dimension[1], f"Channel {channel} data descriptor {data_descriptor} matches expected {expected_dimension[1]}"))
            else:
                result.messages.append(AcquisitionTestResultMessage(False, f"Channel {channel} data and metadata not found"))
        else:
            result.messages.append(AcquisitionTestResultMessage(False, f"Channel {channel} not found in acquisition result"))
    for channel in ar.data_item_map.keys():
        if channel not in expected_results:
            result.messages.append(AcquisitionTestResultMessage(False, f"Channel {channel} not expected in acquisition result"))
    return result


async def test_ronchigram_view(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    ronchigram_camera_hardware_source = stem_controller.ronchigram_camera
    if ronchigram_camera_hardware_source:
        binning = 2
        ronchigram_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        ronchigram_camera_hardware_source.start_playing(sync_timeout=3.0)
        start = time.time()
        while time.time() - start < 5.0:
            await asyncio.sleep(0.1)
        ronchigram_camera_hardware_source.stop_playing(sync_timeout=3.0)
        ar = AcquisitionPanel._AcquireResult(True, dict())
        expected_results: typing.Mapping[Acquisition.Channel, tuple[DataAndMetadata.ShapeType, DataAndMetadata.DataDescriptor]] = {
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Ronchigram camera not found.")


async def test_eels_view(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        eels_camera_hardware_source.start_playing(sync_timeout=3.0)
        start = time.time()
        while time.time() - start < 5.0:
            await asyncio.sleep(0.1)
        eels_camera_hardware_source.stop_playing(sync_timeout=3.0)
        ar = AcquisitionPanel._AcquireResult(True, dict())
        expected_results: typing.Mapping[Acquisition.Channel, tuple[DataAndMetadata.ShapeType, DataAndMetadata.DataDescriptor]] = {
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_scan_view(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        scan_size = Geometry.IntSize(512, 512)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
        scan_hardware_source.start_playing(sync_timeout=3.0)
        start = time.time()
        while time.time() - start < 5.0:
            await asyncio.sleep(0.1)
        scan_hardware_source.stop_playing(sync_timeout=3.0)
        ar = AcquisitionPanel._AcquireResult(True, dict())
        expected_results: typing.Mapping[Acquisition.Channel, tuple[DataAndMetadata.ShapeType, DataAndMetadata.DataDescriptor]] = {
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller not found.")


async def test_scan_2ch_view(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        scan_size = Geometry.IntSize(512, 512)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index in (0, 1))
        scan_hardware_source.start_playing(sync_timeout=3.0)
        start = time.time()
        while time.time() - start < 5.0:
            await asyncio.sleep(0.1)
        scan_hardware_source.stop_playing(sync_timeout=3.0)
        ar = AcquisitionPanel._AcquireResult(True, dict())
        expected_results: typing.Mapping[Acquisition.Channel, tuple[DataAndMetadata.ShapeType, DataAndMetadata.DataDescriptor]] = {
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller not found.")


async def test_ronchigram(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of ronchigram
    ronchigram_camera_hardware_source = stem_controller.ronchigram_camera
    if ronchigram_camera_hardware_source:
        binning = 2
        ronchigram_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = ronchigram_camera_hardware_source.get_expected_dimensions(binning)
        ronchigram_hardware_source_id = ronchigram_camera_hardware_source.hardware_source_id
        ronchigram_channel = Acquisition.Channel(ronchigram_hardware_source_id)
        ap._activate_basic_acquire()
        ap._activate_camera_acquire(ronchigram_hardware_source_id, "image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            ronchigram_channel: (expected_dimensions, DataAndMetadata.DataDescriptor(False, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Ronchigram camera not found.")


async def test_eels_spectra(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        eels_channel = Acquisition.Channel(eels_hardware_source_id)
        ap._activate_basic_acquire()
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_spectrum", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            eels_channel: ((expected_dimensions[1],), DataAndMetadata.DataDescriptor(False, 0, 1))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_eels_image(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        eels_channel = Acquisition.Channel(eels_hardware_source_id)
        ap._activate_basic_acquire()
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            eels_channel: (expected_dimensions, DataAndMetadata.DataDescriptor(False, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_scan(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test spectrum image
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        scan_size = Geometry.IntSize(8, 8)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
        ap._activate_basic_acquire()
        ap._activate_scan_acquire(scan_hardware_source_id)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel(scan_hardware_source_id, "0"): (tuple(scan_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller or EELS camera not found.")


async def test_scan_2ch(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test spectrum image
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        scan_size = Geometry.IntSize(8, 8)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index in (0, 1))
        ap._activate_basic_acquire()
        ap._activate_scan_acquire(scan_hardware_source_id)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel(scan_hardware_source_id, "0"): (tuple(scan_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
            Acquisition.Channel(scan_hardware_source_id, "1"): (tuple(scan_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller or EELS camera not found.")


async def test_scan_subscan(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test spectrum image
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        binning = 1
        scan_size = Geometry.IntSize(20, 20)
        subscan_pixel_size = Geometry.IntSize(10, 10)
        scan_frame_parameters = ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size, subscan_pixel_size=subscan_pixel_size, subscan_fractional_center=Geometry.FloatPoint(0.5, 0.5), subscan_fractional_size=Geometry.FloatSize(0.5, 0.5))
        scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
        scan_hardware_source.subscan_enabled = True
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
        scan_channel = Acquisition.Channel(scan_hardware_source_id, "0")
        ap._activate_basic_acquire()
        ap._activate_scan_acquire(scan_hardware_source_id)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel(scan_hardware_source_id, "0"): (tuple(subscan_pixel_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller or EELS camera not found.")


async def test_scan_subscan_2ch(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test spectrum image
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        binning = 1
        scan_size = Geometry.IntSize(20, 20)
        subscan_pixel_size = Geometry.IntSize(10, 10)
        scan_frame_parameters = ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size, subscan_pixel_size=subscan_pixel_size, subscan_fractional_center=Geometry.FloatPoint(0.5, 0.5), subscan_fractional_size=Geometry.FloatSize(0.5, 0.5))
        scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
        scan_hardware_source.subscan_enabled = True
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index in (0, 1))
        scan_channel = Acquisition.Channel(scan_hardware_source_id, "0")
        ap._activate_basic_acquire()
        ap._activate_scan_acquire(scan_hardware_source_id)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel(scan_hardware_source_id, "0"): (tuple(subscan_pixel_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
            Acquisition.Channel(scan_hardware_source_id, "1"): (tuple(subscan_pixel_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller or EELS camera not found.")


async def test_scan_eels(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test spectrum image
    scan_hardware_source = stem_controller.scan_controller
    eels_camera_hardware_source = stem_controller.eels_camera
    if scan_hardware_source and eels_camera_hardware_source:
        binning = 1
        scan_size = Geometry.IntSize(8, 8)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
        scan_channel = Acquisition.Channel(scan_hardware_source_id, "0")
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        ap._activate_basic_acquire()
        ap._activate_synchronized_acquire(scan_hardware_source_id, scan_size.width, eels_hardware_source_id, "eels_spectrum", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel(scan_hardware_source_id, "0"): (tuple(scan_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
            Acquisition.Channel(eels_hardware_source_id): (tuple(scan_size) + (expected_dimensions[1],), DataAndMetadata.DataDescriptor(False, 2, 1)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller or EELS camera not found.")


async def test_scan_subscan_eels(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test spectrum image
    scan_hardware_source = stem_controller.scan_controller
    eels_camera_hardware_source = stem_controller.eels_camera
    if scan_hardware_source and eels_camera_hardware_source:
        binning = 1
        scan_size = Geometry.IntSize(20, 20)
        subscan_pixel_size = Geometry.IntSize(10, 10)
        scan_frame_parameters = ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size, subscan_pixel_size=subscan_pixel_size, subscan_fractional_center=Geometry.FloatPoint(0.5, 0.5), subscan_fractional_size=Geometry.FloatSize(0.5, 0.5))
        scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
        scan_hardware_source.subscan_enabled = True
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
        scan_channel = Acquisition.Channel(scan_hardware_source_id, "0")
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        ap._activate_basic_acquire()
        ap._activate_synchronized_acquire(scan_hardware_source_id, subscan_pixel_size.width, eels_hardware_source_id, "eels_spectrum", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel(scan_hardware_source_id, "0"): (tuple(subscan_pixel_size), DataAndMetadata.DataDescriptor(False, 0, 2)),
            Acquisition.Channel(eels_hardware_source_id): (tuple(subscan_pixel_size) + (expected_dimensions[1],), DataAndMetadata.DataDescriptor(False, 2, 1)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller or EELS camera not found.")


async def test_ronchigram_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of ronchigram
    ronchigram_camera_hardware_source = stem_controller.ronchigram_camera
    if ronchigram_camera_hardware_source:
        binning = 2
        count = 4
        ronchigram_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = ronchigram_camera_hardware_source.get_expected_dimensions(binning)
        ronchigram_hardware_source_id = ronchigram_camera_hardware_source.hardware_source_id
        ronchigram_channel = Acquisition.Channel(ronchigram_hardware_source_id)
        ap._activate_sequence_acquire(count)
        ap._activate_camera_acquire(ronchigram_hardware_source_id, "image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            ronchigram_channel: ((count,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Ronchigram camera not found.")


async def test_eels_spectra_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        count = 4
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        eels_channel = Acquisition.Channel(eels_hardware_source_id)
        ap._activate_sequence_acquire(count)
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_spectrum", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            eels_channel: ((count, expected_dimensions[1]), DataAndMetadata.DataDescriptor(True, 0, 1))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_eels_image_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        count = 4
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        eels_channel = Acquisition.Channel(eels_hardware_source_id)
        ap._activate_sequence_acquire(count)
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            eels_channel: ((count,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_scan_one_channel_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        count = 4
        scan_size = Geometry.IntSize(40, 40)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
        scan_channel = Acquisition.Channel(scan_hardware_source_id, "0")
        ap._activate_sequence_acquire(count)
        ap._activate_scan_acquire(scan_hardware_source_id)
        ar = await ap._acquire()
        expected_results = {
            scan_channel: ((count,) + tuple(scan_size), DataAndMetadata.DataDescriptor(True, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller not found.")


async def test_scan_two_channel_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        count = 4
        scan_size = Geometry.IntSize(40, 40)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index in (0, 1))
        scan_channel0 = Acquisition.Channel(scan_hardware_source_id, "0")
        scan_channel1 = Acquisition.Channel(scan_hardware_source_id, "1")
        ap._activate_sequence_acquire(count)
        ap._activate_scan_acquire(scan_hardware_source_id)
        ar = await ap._acquire()
        expected_results = {
            scan_channel0: ((count,) + tuple(scan_size), DataAndMetadata.DataDescriptor(True, 0, 2)),
            scan_channel1: ((count,) + tuple(scan_size), DataAndMetadata.DataDescriptor(True, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller not found.")


async def test_scan_subscan_one_channel_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        count = 4
        scan_size = Geometry.IntSize(40, 40)
        subscan_pixel_size = Geometry.IntSize(20, 20)
        scan_frame_parameters = ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size, subscan_pixel_size=subscan_pixel_size, subscan_fractional_center=Geometry.FloatPoint(0.5, 0.5), subscan_fractional_size=Geometry.FloatSize(0.5, 0.5))
        scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
        scan_hardware_source.subscan_enabled = True
        try:
            scan_hardware_source_id = scan_hardware_source.hardware_source_id
            for channel_index in range(scan_hardware_source.get_channel_count()):
                scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
            scan_channel = Acquisition.Channel(scan_hardware_source_id, "0")
            ap._activate_sequence_acquire(count)
            ap._activate_scan_acquire(scan_hardware_source_id)
            ar = await ap._acquire()
            expected_results = {
                scan_channel: ((count,) + tuple(subscan_pixel_size), DataAndMetadata.DataDescriptor(True, 0, 2))
            }
            return check_results(ar, logger, expected_results)
        finally:
            scan_hardware_source.subscan_enabled = False
    else:
        return make_not_available_result("Scan controller not found.")


async def test_scan_subscan_two_channel_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    scan_hardware_source = stem_controller.scan_controller
    if scan_hardware_source:
        count = 4
        scan_size = Geometry.IntSize(40, 40)
        subscan_pixel_size = Geometry.IntSize(20, 20)
        scan_frame_parameters = ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size, subscan_pixel_size=subscan_pixel_size, subscan_fractional_center=Geometry.FloatPoint(0.5, 0.5), subscan_fractional_size=Geometry.FloatSize(0.5, 0.5))
        scan_hardware_source.set_current_frame_parameters(scan_frame_parameters)
        scan_hardware_source.subscan_enabled = True
        try:
            scan_hardware_source_id = scan_hardware_source.hardware_source_id
            for channel_index in range(scan_hardware_source.get_channel_count()):
                scan_hardware_source.set_channel_enabled(channel_index, channel_index in (0, 1))
            scan_channel0 = Acquisition.Channel(scan_hardware_source_id, "0")
            scan_channel1 = Acquisition.Channel(scan_hardware_source_id, "1")
            ap._activate_sequence_acquire(count)
            ap._activate_scan_acquire(scan_hardware_source_id)
            ar = await ap._acquire()
            expected_results = {
                scan_channel0: ((count,) + tuple(subscan_pixel_size), DataAndMetadata.DataDescriptor(True, 0, 2)),
                scan_channel1: ((count,) + tuple(subscan_pixel_size), DataAndMetadata.DataDescriptor(True, 0, 2))
            }
            return check_results(ar, logger, expected_results)
        finally:
            scan_hardware_source.subscan_enabled = False
    else:
        return make_not_available_result("Scan controller not found.")


async def test_synchronized_eels_sequence(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    scan_hardware_source = stem_controller.scan_controller
    eels_camera_hardware_source = stem_controller.eels_camera
    if scan_hardware_source and eels_camera_hardware_source:
        binning = 1
        count = 4
        scan_size = Geometry.IntSize(8, 8)
        scan_hardware_source.set_current_frame_parameters(ScanBase.ScanFrameParameters(fov_nm=100.0, pixel_time_us=10.0, pixel_size=scan_size))
        scan_hardware_source.subscan_enabled = False
        scan_hardware_source_id = scan_hardware_source.hardware_source_id
        for channel_index in range(scan_hardware_source.get_channel_count()):
            scan_hardware_source.set_channel_enabled(channel_index, channel_index == 0)
        scan_channel = Acquisition.Channel(scan_hardware_source_id, "0")
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        ap._activate_sequence_acquire(count)
        ap._activate_synchronized_acquire(scan_hardware_source_id, scan_size.width, eels_hardware_source_id, "eels_spectrum", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel(scan_hardware_source_id, "0"): ((count,) + tuple(scan_size), DataAndMetadata.DataDescriptor(True, 0, 2)),
            Acquisition.Channel(eels_hardware_source_id): ((count,) + tuple(scan_size) + (expected_dimensions[1],), DataAndMetadata.DataDescriptor(True, 2, 1)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Scan controller or EELS camera not found.")


async def test_ronchigram_series(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test series of ronchigram
    ronchigram_camera_hardware_source = stem_controller.ronchigram_camera
    if ronchigram_camera_hardware_source:
        binning = 2
        count = 4
        ronchigram_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = ronchigram_camera_hardware_source.get_expected_dimensions(binning)
        ronchigram_hardware_source_id = ronchigram_camera_hardware_source.hardware_source_id
        ronchigram_channel = Acquisition.Channel(ronchigram_hardware_source_id)
        ap._activate_series_acquire("defocus", AcquisitionPanel.ControlValues(count, 0e-9, 5e-9))
        ap._activate_camera_acquire(ronchigram_hardware_source_id, "image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            ronchigram_channel: ((count,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Ronchigram camera not found.")


async def test_eels_spectra_series(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        count = 6
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        eels_channel = Acquisition.Channel(eels_hardware_source_id)
        ap._activate_series_acquire("energy_offset", AcquisitionPanel.ControlValues(count, 0, 20))
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_spectrum", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            eels_channel: ((count, expected_dimensions[1]), DataAndMetadata.DataDescriptor(True, 0, 1))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_eels_image_series(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        count = 4
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        eels_channel = Acquisition.Channel(eels_hardware_source_id)
        ap._activate_series_acquire("energy_offset", AcquisitionPanel.ControlValues(count, 0, 20))
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            eels_channel: ((count,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2))
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_ronchigram_multi(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test series of ronchigram
    ronchigram_camera_hardware_source = stem_controller.ronchigram_camera
    if ronchigram_camera_hardware_source:
        binning = 2
        count1 = 4
        count2 = 2
        ronchigram_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = ronchigram_camera_hardware_source.get_expected_dimensions(binning)
        ronchigram_hardware_source_id = ronchigram_camera_hardware_source.hardware_source_id
        ap._activate_multiple_acquire([AcquisitionPanel.MultipleAcquireEntry(0.0, 0.025, count1, True), AcquisitionPanel.MultipleAcquireEntry(5.0, 0.05, count2, False)])
        ap._activate_camera_acquire(ronchigram_hardware_source_id, "image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel("0", ronchigram_hardware_source_id): ((count1,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2)),
            Acquisition.Channel("0", ronchigram_hardware_source_id, "sum"): (expected_dimensions, DataAndMetadata.DataDescriptor(False, 0, 2)),
            Acquisition.Channel("1", ronchigram_hardware_source_id): ((count2,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("Ronchigram camera not found.")


async def test_eels_spectra_multi(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        count1 = 4
        count2 = 2
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        ap._activate_multiple_acquire([AcquisitionPanel.MultipleAcquireEntry(0.0, 0.025, count1, True), AcquisitionPanel.MultipleAcquireEntry(5.0, 0.05, count2, False)])
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_spectrum", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel("0", eels_hardware_source_id): ((count1, expected_dimensions[1]), DataAndMetadata.DataDescriptor(True, 0, 1)),
            Acquisition.Channel("0", eels_hardware_source_id, "sum"): ((expected_dimensions[1],), DataAndMetadata.DataDescriptor(False, 0, 1)),
            Acquisition.Channel("1", eels_hardware_source_id): ((count2, expected_dimensions[1]), DataAndMetadata.DataDescriptor(True, 0, 1)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


async def test_eels_image_multi(ap: AcquisitionPanel.AcquisitionPanel, stem_controller: STEMController.STEMController, logger: logging.Logger) -> AcquisitionTestResult:
    # test sequence of eels
    eels_camera_hardware_source = stem_controller.eels_camera
    if eels_camera_hardware_source:
        binning = 1
        count1 = 4
        count2 = 2
        eels_camera_hardware_source.set_current_frame_parameters(CameraBase.CameraFrameParameters(exposure_ms=10, binning=binning))
        expected_dimensions = eels_camera_hardware_source.get_expected_dimensions(binning)
        eels_hardware_source_id = eels_camera_hardware_source.hardware_source_id
        ap._activate_multiple_acquire([AcquisitionPanel.MultipleAcquireEntry(0.0, 0.025, count1, True), AcquisitionPanel.MultipleAcquireEntry(5.0, 0.05, count2, False)])
        ap._activate_camera_acquire(eels_hardware_source_id, "eels_image", 12e-3)
        ar = await ap._acquire()
        expected_results = {
            Acquisition.Channel("0", eels_hardware_source_id): ((count1,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2)),
            Acquisition.Channel("0", eels_hardware_source_id, "sum"): (expected_dimensions, DataAndMetadata.DataDescriptor(False, 0, 2)),
            Acquisition.Channel("1", eels_hardware_source_id): ((count2,) + expected_dimensions, DataAndMetadata.DataDescriptor(True, 0, 2)),
        }
        return check_results(ar, logger, expected_results)
    else:
        return make_not_available_result("EELS camera not found.")


def acquisition_test_state_to_string(state: AcquisitionTestState | None) -> str:
    if state == AcquisitionTestState.NOT_STARTED:
        return "\N{WHITE CIRCLE}"
    elif state == AcquisitionTestState.IN_PROGRESS:
        return "\N{CIRCLE WITH LEFT HALF BLACK}"
    elif state == AcquisitionTestState.SUCCEEDED:
        return "\N{WHITE HEAVY CHECK MARK}"
    elif state == AcquisitionTestState.FAILED:
        return "\N{CROSS MARK}"
    else:
        return "\N{LARGE BLUE CIRCLE}"


class AcquisitionTest(Observable.Observable):
    def __init__(self, title: str, acquisition_fn: typing.Callable[[AcquisitionPanel.AcquisitionPanel, STEMController.STEMController, logging.Logger], typing.Awaitable[AcquisitionTestResult]]) -> None:
        super().__init__()
        self.title = title
        self.acquisition_fn = acquisition_fn
        self.__state = AcquisitionTestState.NOT_STARTED
        self.status_stream = Model.StreamValueModel(Stream.MapStream[AcquisitionTestState, str](Stream.PropertyChangedEventStream(self, "state"), acquisition_test_state_to_string))
        self.__elapsed_time: float | None = None
        self.tooltip = Model.PropertyModel[str]()
        self.__result: AcquisitionTestResult | None = None

    @property
    def state(self) -> AcquisitionTestState:
        return self.__state

    @state.setter
    def state(self, value: AcquisitionTestState) -> None:
        self.__state = value
        self.notify_property_changed("state")

    @property
    def elapsed_time(self) -> float | None:
        return self.__elapsed_time

    @elapsed_time.setter
    def elapsed_time(self, value: float | None) -> None:
        self.__elapsed_time = value
        self.notify_property_changed("elapsed_time")

    @property
    def has_result(self) -> bool:
        return self.__result is not None

    @property
    def result(self) -> AcquisitionTestResult | None:
        return self.__result

    @result.setter
    def result(self, value: AcquisitionTestResult | None) -> None:
        self.__result = value
        self.notify_property_changed("result")
        self.notify_property_changed("has_result")

    async def run_test_async(self, window: DocumentController.DocumentController, is_stopped_model: Model.ValueModel[bool], delete_results: bool) -> None:
        was_stopped = is_stopped_model.value
        is_stopped_model.value = False
        try:
            self.state = AcquisitionTestState.IN_PROGRESS
            stem_controller = typing.cast(STEMController.STEMController | None, Registry.get_component("stem_controller"))
            assert stem_controller

            ap = typing.cast(AcquisitionPanel.AcquisitionPanel | None, window.find_dock_panel("acquisition-panel"))
            assert ap
            ap.show()

            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)

            logger.info(f"Starting acquisition test: {self.title}")

            result: AcquisitionTestResult | None = None
            exception: Exception | None = None

            try:
                start_time = time.time()
                result = await self.acquisition_fn(ap, stem_controller, logger)
                self.elapsed_time = time.time() - start_time
            except Exception as e:
                import traceback
                traceback.print_exc()
                exception = e

            logger.info(f"Finishing acquisition test: {self.title}")

            result_messages = result.messages if result else list()

            if not result:
                result_messages.append(AcquisitionTestResultMessage(False, f"Acquisition test failed: {exception}"))
                self.state = AcquisitionTestState.FAILED

            self.tooltip.value = "\n".join(f"{acquisition_test_state_to_string(r.status)} {r.message}" for r in result_messages)

            if all(r.status == AcquisitionTestState.SUCCEEDED for r in result_messages):
                self.state = AcquisitionTestState.SUCCEEDED
            elif any(r.status == AcquisitionTestState.NOT_AVAILABLE for r in result_messages):
                self.state = AcquisitionTestState.NOT_AVAILABLE
            else:
                self.state = AcquisitionTestState.FAILED

            self.result = result

            if result and delete_results:
                result.delete_display_items(window)
        finally:
            is_stopped_model.value = was_stopped


all_acquisition_tests = [
    AcquisitionTest("Ronchigram (View)", test_ronchigram_view),
    AcquisitionTest("EELS (View)", test_eels_view),
    AcquisitionTest("Scan (View)", test_scan_view),
    AcquisitionTest("Scan/2Ch (View)", test_scan_2ch_view),
    AcquisitionTest("Ronchigram (Single)", test_ronchigram),
    AcquisitionTest("EELS (Single/Spectra)", test_eels_spectra),
    AcquisitionTest("EELS (Single/Image)", test_eels_image),
    AcquisitionTest("Scan (Single)", test_scan),
    AcquisitionTest("Scan (Single/Subscan)", test_scan_subscan),
    AcquisitionTest("Spectrum Image (Single)", test_scan_eels),
    AcquisitionTest("Spectrum Image (Single/Subscan)", test_scan_subscan_eels),
    AcquisitionTest("Ronchigram (Sequence)", test_ronchigram_sequence),
    AcquisitionTest("EELS (Sequence/Spectra)", test_eels_spectra_sequence),
    AcquisitionTest("EELS (Sequence/Images)", test_eels_image_sequence),
    AcquisitionTest("Scan (Sequence)", test_scan_one_channel_sequence),
    AcquisitionTest("Scan/2Ch (Sequence)", test_scan_two_channel_sequence),
    AcquisitionTest("Subscan (Sequence)", test_scan_subscan_one_channel_sequence),
    AcquisitionTest("Subscan/2Ch (Sequence)", test_scan_subscan_two_channel_sequence),
    AcquisitionTest("SI (Sequence)", test_synchronized_eels_sequence),
    AcquisitionTest("Ronchigram (Series/defocus)", test_ronchigram_series),
    AcquisitionTest("EELS (Series/Spectra/energy_offset)", test_eels_spectra_series),
    AcquisitionTest("EELS (Series/Images/energy_offset)", test_eels_image_series),
    AcquisitionTest("Ronchigram (Multi)", test_ronchigram_multi),
    AcquisitionTest("EELS (Multi/Spectra)", test_eels_spectra_multi),
    AcquisitionTest("EELS (Multi/Images)", test_eels_image_multi),
]


class AcquisitionTestHandler(Declarative.Handler):
    def __init__(self, acquisition_test: AcquisitionTest, window: DocumentController.DocumentController, is_stopped_model: Model.ValueModel[bool]) -> None:
        super().__init__()
        self.acquisition_test = acquisition_test
        self.window = window
        self.is_stopped_model = is_stopped_model
        self.elapsed_time_converter = Converter.PhysicalValueToStringConverter("s", 1, "{:.1f}")
        u = Declarative.DeclarativeUI()
        self.ui_view = u.create_row(
            u.create_label(text="@binding(acquisition_test.status_stream.value)", width=18),
            u.create_label(text=acquisition_test.title, tool_tip="@binding(acquisition_test.tooltip.value)"),
            u.create_stretch(),
            u.create_label(text="@binding(acquisition_test.elapsed_time, converter=elapsed_time_converter)", width=48),
            u.create_push_button(text="Run", enabled="@binding(is_stopped_model.value)", on_clicked="handle_run_clicked", style="minimal"),
            u.create_push_button(text="Delete", enabled="@binding(acquisition_test.has_result)", on_clicked="handle_delete_clicked", style="minimal"),
            spacing=8,
            margin=8
        )

    def handle_run_clicked(self, widget: Declarative.UIWidget) -> None:
        self.window.event_loop.create_task(self.acquisition_test.run_test_async(self.window, self.is_stopped_model, False))

    def handle_delete_clicked(self, widget: Declarative.UIWidget) -> None:
        result = self.acquisition_test.result
        if result:
            result.delete_display_items(self.window)
            self.acquisition_test.result = None
            self.acquisition_test.elapsed_time = None


class AcquisitionTestDialog(Declarative.Handler):
    def __init__(self, window: DocumentController.DocumentController) -> None:
        super().__init__()
        self.window = window
        self.acquisition_tests = all_acquisition_tests
        self.is_stopped_model = Model.PropertyModel[bool](True)

        self.is_run_all_model = Model.PropertyModel[bool](False)
        self.__cancel = False

        def is_enabled(b: typing.Optional[bool]) -> str:
            return "Run All" if not b else "Cancel"

        self.run_all_button_text_model = Model.StreamValueModel(Stream.MapStream(
            Stream.PropertyChangedEventStream(self.is_run_all_model, "value"),
            is_enabled))

        def is_run_all_enabled(b: typing.Optional[bool]) -> str:
            return "Run All" if not b else "Cancel"

        self.run_all_enabled_model: Model.ValueModel[bool] = Model.StreamValueModel(Stream.CombineLatestStream([Stream.PropertyChangedEventStream(self.is_stopped_model, "value"), Stream.PropertyChangedEventStream(self.is_run_all_model, "value")], lambda a, b: a or b))

        u = Declarative.DeclarativeUI()
        ui_view = u.create_column(
            u.create_column(items="acquisition_tests", item_component_id="acquisition_test"),
            u.create_row(
                u.create_stretch(),
                u.create_push_button(text="Reset", enabled="@binding(is_stopped_model.value)", on_clicked="handle_reset_clicked", style="minimal"),
                u.create_push_button(text="@binding(run_all_button_text_model.value)", enabled="@binding(run_all_enabled_model.value)", on_clicked="handle_run_all_clicked", style="minimal"), spacing=8, margin=8),
        )
        dialog = typing.cast(Dialog.ActionDialog, Declarative.construct(window.ui, window, u.create_modeless_dialog(ui_view, title="Acquisition Test"), self))
        dialog.show()

    def create_handler(self, component_id: str, container: typing.Any = None, item: typing.Any = None, **kwargs: typing.Any) -> typing.Optional[Declarative.HandlerLike]:
        if component_id == "acquisition_test":
            return AcquisitionTestHandler(typing.cast(AcquisitionTest, item), self.window, self.is_stopped_model)
        return None

    def handle_reset_clicked(self, widget: Declarative.UIWidget) -> None:
        for acquisition_test in self.acquisition_tests:
            acquisition_test.state = AcquisitionTestState.NOT_STARTED
            acquisition_test.elapsed_time = None
            acquisition_test.result = None

    def handle_run_all_clicked(self, widget: Declarative.UIWidget) -> None:
        if self.is_run_all_model.value:
            self.__cancel = True
            return

        for acquisition_test in self.acquisition_tests:
            acquisition_test.state = AcquisitionTestState.NOT_STARTED

        async def run_all() -> None:
            self.__cancel = False
            self.is_run_all_model.value = True
            self.is_stopped_model.value = False
            try:
                for acquisition_test in self.acquisition_tests:
                    try:
                        await acquisition_test.run_test_async(self.window, self.is_stopped_model, True)
                        if self.__cancel:
                            break
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
            finally:
                self.is_stopped_model.value = True
                self.is_run_all_model.value = False

        self.window.event_loop.create_task(run_all())



def _internal_main(window: DocumentController.DocumentController) -> None:
    # this will be called from run script on a thread. to run on the ui thread, schedule an async task.

    async def main() -> None:
        AcquisitionTestDialog(window)

    window.event_loop.create_task(main())
