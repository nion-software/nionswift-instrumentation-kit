from __future__ import annotations

# system imports
import asyncio
import copy
import datetime
import functools
import gettext
import numpy
import numpy.typing
import typing

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.instrumentation import scan_base
from nion.swift.model import DataItem
from nion.utils import Geometry

if typing.TYPE_CHECKING:
    from nion.swift.model import DocumentModel

_NDArray = numpy.typing.NDArray[typing.Any]

_ = gettext.gettext


class DriftLogger:
    """Drift logger to update the drift log with ongoing drift measurements."""

    def __init__(self, document_model: DocumentModel.DocumentModel, drift_tracker: scan_base.DriftTracker, event_loop: asyncio.AbstractEventLoop):
        self.__document_model = document_model
        self.__drift_tracker = drift_tracker
        self.__event_loop = event_loop
        self.__data_item = next(iter(data_item for data_item in document_model.data_items if data_item.title == "Drift Log"), None)
        self.__drift_changed_event_listener = drift_tracker.drift_changed_event.listen(self.__drift_changed)

    def close(self) -> None:
        self.__drift_changed_event_listener.close()
        self.__drift_changed_event_listener = typing.cast(typing.Any, None)

    def __ensure_drift_log_data_item(self) -> None:
        # must be called on main thread
        if not self.__data_item:
            drift_data_frame = self.__drift_tracker.drift_data_frame
            delta_nm_data = numpy.vstack([drift_data_frame[0], drift_data_frame[1], numpy.hypot(drift_data_frame[0], drift_data_frame[1])])
            data_item = DataItem.DataItem(delta_nm_data)
            data_item.title = f"Drift Log"
            self.__document_model.append_data_item(data_item)
            display_item = self.__document_model.get_display_item_for_data_item(data_item)
            if display_item:
                display_item.display_type = "line_plot"
                display_item.append_display_data_channel_for_data_item(data_item)
                display_item.append_display_data_channel_for_data_item(data_item)
                display_item._set_display_layer_properties(0, label=_("x"), stroke_color=display_item.get_display_layer_property(0, "fill_color"), fill_color=None)
                display_item._set_display_layer_properties(1, label=_("y"), stroke_color=display_item.get_display_layer_property(1, "fill_color"), fill_color=None)
                display_item._set_display_layer_properties(2, label=_("m"), stroke_color=display_item.get_display_layer_property(2, "fill_color"), fill_color=None)
            self.__data_item = data_item

    def __update_drift_log_data_item(self, delta_nm_data: _NDArray) -> None:
        # must be called on main thread
        # check __drift_changed_event_listener to see if the logger has been closed
        if self.__drift_changed_event_listener:
            self.__ensure_drift_log_data_item()
            assert self.__data_item
            offset_nm_xdata = DataAndMetadata.new_data_and_metadata(delta_nm_data, intensity_calibration=Calibration.Calibration(units="nm"))
            self.__data_item.set_data_and_metadata(offset_nm_xdata)

    def __drift_changed(self, offset_nm: Geometry.FloatSize, elapsed_time: float) -> None:
        drift_data_frame = self.__drift_tracker.drift_data_frame
        delta_nm_data = numpy.vstack([drift_data_frame[0], drift_data_frame[1], numpy.hypot(drift_data_frame[0], drift_data_frame[1])])
        self.__event_loop.call_soon_threadsafe(functools.partial(self.__update_drift_log_data_item, delta_nm_data))


class DriftCorrectionBehavior(scan_base.SynchronizedScanBehaviorInterface):
    """Drift correction behavior for updating drift at beginning of each synchronized scan section.

    Take a drift scan from the drift region and send it to the drift compensator.
    """

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, scan_frame_parameters: scan_base.ScanFrameParameters):
        # init with the frame parameters from the synchronized grab
        self.__scan_hardware_source = scan_hardware_source
        self.__scan_frame_parameters = copy.deepcopy(scan_frame_parameters)
        # here we convert those frame parameters to the context
        self.__scan_frame_parameters.subscan_pixel_size = None
        self.__scan_frame_parameters.subscan_fractional_size = None
        self.__scan_frame_parameters.subscan_fractional_center = None
        self.__scan_frame_parameters.subscan_rotation = 0.0
        self.__scan_frame_parameters.channel_override = "drift"

    def prepare_section(self, *, utc_time: typing.Optional[datetime.datetime] = None) -> None:
        # this method must be thread safe
        # start with the context frame parameters and adjust for the drift region
        frame_parameters = copy.deepcopy(self.__scan_frame_parameters)
        context_size = frame_parameters.size.to_float_size()
        drift_channel_id = self.__scan_hardware_source.drift_channel_id
        drift_region = self.__scan_hardware_source.drift_region
        drift_rotation = self.__scan_hardware_source.drift_rotation
        if drift_channel_id is not None and drift_region is not None:
            drift_channel_index = self.__scan_hardware_source.get_channel_index(drift_channel_id)
            assert drift_channel_index is not None
            frame_parameters.subscan_pixel_size = Geometry.IntSize(int(context_size.height * drift_region.height * 4), int(context_size.width * drift_region.width * 4))
            if frame_parameters.subscan_pixel_size[0] >= 8 or frame_parameters.subscan_pixel_size[1] >= 8:
                frame_parameters.subscan_fractional_size = Geometry.FloatSize(drift_region.height, drift_region.width)
                frame_parameters.subscan_fractional_center = Geometry.FloatPoint(drift_region.center.y, drift_region.center.x)
                frame_parameters.subscan_rotation = drift_rotation
                # attempt to keep drift area in roughly the same position by adding in the accumulated correction.
                drift_tracker = self.__scan_hardware_source.drift_tracker
                utc_time = utc_time or datetime.datetime.utcnow()
                delta_nm = drift_tracker.predict_drift(utc_time)
                frame_parameters.center_nm = frame_parameters.center_nm - delta_nm
                xdatas = self.__scan_hardware_source.record_immediate(frame_parameters, [drift_channel_index])
                xdata0 = xdatas[0]
                if xdata0:
                    drift_tracker.submit_image(xdata0, drift_rotation, wait=True)
