from __future__ import annotations

# system imports
import asyncio
import copy
import datetime
import functools
import gettext
import math
import numpy
import numpy.typing
import threading
import typing

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.instrumentation import Acquisition
from nion.instrumentation import AcquisitionPreferences
from nion.swift.model import DataItem
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Registry
from nion.utils import ThreadPool

if typing.TYPE_CHECKING:
    from nion.swift.model import DocumentModel
    from nion.instrumentation import scan_base

_NDArray = numpy.typing.NDArray[typing.Any]

_ = gettext.gettext


class DriftLogger:
    """Drift logger to update the drift log with ongoing drift measurements."""

    def __init__(self, document_model: DocumentModel.DocumentModel, drift_tracker: DriftTracker, event_loop: asyncio.AbstractEventLoop):
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
            delta_nm_data: numpy.typing.NDArray[typing.Any] = numpy.vstack([drift_data_frame[0], drift_data_frame[1], numpy.hypot(drift_data_frame[0], drift_data_frame[1])])
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
        delta_nm_data: numpy.typing.NDArray[typing.Any] = numpy.vstack([drift_data_frame[0], drift_data_frame[1], numpy.hypot(drift_data_frame[0], drift_data_frame[1])])
        self.__event_loop.call_soon_threadsafe(functools.partial(self.__update_drift_log_data_item, delta_nm_data))


class DriftTracker:
    """Track drift state.

    Tracks several properties of drift including the accumulated drift and rate of drift.

    Reset the tracker by calling reset.

    Drift is tracked as a difference from the first data item. This has the obvious limitation of requiring drift to
    be no larger than approximately 1/2 the field of view over the course of acquisition.

    An extension to this class would be to separate the drift algorithm into its own class and allow it to be
    configured.
    """
    def __init__(self) -> None:
        # dispatcher is used to calculate drift offsets on a thread
        self.__dispatcher = ThreadPool.SingleItemDispatcher()

        # the lock controls access to the fields below
        self.__lock = threading.RLock()

        #  The current data item is used for the delta calculation only.
        self.__first_xdata: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.__current_xdata: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.__rotation = 0.0

        self.__drift_data_frame: numpy.typing.NDArray[numpy.float_] = numpy.zeros((4, 0), float)

        self.__total_delta_nm = Geometry.FloatSize()

        self.drift_changed_event = Event.Event()

    def close(self) -> None:
        self.__dispatcher.close()
        self.__dispatcher = typing.cast(typing.Any, None)

    def reset(self) -> None:
        with self.__lock:
            self.__first_xdata = None
            self.__current_xdata = None
            self.__drift_data_frame = numpy.zeros((4, 0), float)
            self.__rotation = 0.0
            self.__total_delta_nm = Geometry.FloatSize()

    # For testing
    @property
    def _first_xdata(self) -> typing.Optional[DataAndMetadata.DataAndMetadata]:
        return self.__first_xdata

    @property
    def measurement_count(self) -> int:
        with self.__lock:
            return typing.cast(int, self.__drift_data_frame.shape[-1])

    @property
    def total_delta_nm(self) -> Geometry.FloatSize:
        return self.__total_delta_nm

    @property
    def drift_data_frame(self) -> _NDArray:
        with self.__lock:
            return numpy.copy(self.__drift_data_frame)  # type: ignore

    @property
    def last_delta_nm(self) -> Geometry.FloatSize:
        with self.__lock:
            if self.measurement_count > 0:
                width = self.__drift_data_frame[1][-1]
                height = self.__drift_data_frame[0][-1]
            else:
                width = 0.0
                height = 0.0
            return Geometry.FloatSize(height=height, width=width)

    @property
    def _last_entry_utc_time(self) -> datetime.datetime:
        if self.__first_xdata:
            # self.__first_xdata.timestamp is utc datetime, so just use 'fromtimestamp' to get utc POSIX timestamp
            return datetime.datetime.fromtimestamp(self.__first_xdata.timestamp.timestamp() + numpy.sum(self.__drift_data_frame[3]))
        else:
            return datetime.datetime.utcnow()

    def get_drift_rate(self, *, n: typing.Optional[int] = None) -> Geometry.FloatSize:
        """Return the drift rate over the last n measurements."""
        n = 3 if n is None else n
        assert n > 0
        with self.__lock:
            if self.measurement_count > 0:
                assert self.__first_xdata
                n = min(n, self.measurement_count)
                offset_v = typing.cast(float, numpy.sum(self.__drift_data_frame[0][-n:]))
                offset_h = typing.cast(float, numpy.sum(self.__drift_data_frame[1][-n:]))
                recent_offset = Geometry.FloatSize(h=offset_v, w=offset_h)
                recent_time = typing.cast(float, numpy.sum(self.__drift_data_frame[3][-n:]))
                return recent_offset / recent_time
            return Geometry.FloatSize()

    def predict_drift(self, utc_time: datetime.datetime, *, n: typing.Optional[int] = None) -> Geometry.FloatSize:
        """Predict total drift (nm) at utc_time."""
        with self.__lock:
            future_delta_nm = Geometry.FloatSize()
            if self.__drift_data_frame.shape[-1] > 0:
                assert self.__first_xdata
                last_entry_timestamp = self.__first_xdata.timestamp.timestamp() + numpy.sum(self.__drift_data_frame[3])
                delta_timestamp = utc_time.timestamp() - last_entry_timestamp
                future_delta_nm = delta_timestamp * self.get_drift_rate(n=n)
            return self.__total_delta_nm + future_delta_nm

    def __append_drift(self, delta_nm: Geometry.FloatSize, delta_time: float) -> None:
        offset_nm_xy = math.sqrt(pow(delta_nm.height, 2) + pow(delta_nm.width, 2))
        with self.__lock:
            self.__drift_data_frame = numpy.hstack(
                [self.__drift_data_frame, numpy.array([delta_nm.height, delta_nm.width, offset_nm_xy, delta_time]).reshape(4, 1)])

    def __calculate(self) -> None:
        with self.__lock:
            first_xdata = self.__first_xdata
            current_xdata = self.__current_xdata
            rotation = self.__rotation

            if first_xdata and current_xdata:
                quality, raw_offset = xd.register_template(first_xdata, current_xdata)
                delta_time = (current_xdata.timestamp - first_xdata.timestamp).total_seconds() - numpy.sum(self.__drift_data_frame[3])
                assert delta_time > 0.0
                offset = Geometry.FloatPoint.make(typing.cast(typing.Tuple[float, float], raw_offset))
                delta_nm = Geometry.FloatSize(
                    h=current_xdata.dimensional_calibrations[0].convert_to_calibrated_size(offset.y),
                    w=current_xdata.dimensional_calibrations[1].convert_to_calibrated_size(offset.x))
                # calculate adjustment (center_nm). if center_nm positive, data shifts up/left.
                # rotate back into context reference frame
                delta_nm = delta_nm.rotate(-rotation)
                # print(f"measured {delta_nm}")
                self.__append_drift(delta_nm, delta_time)
                # add the difference from the last time, but negative since center_nm positive shifts up/left
                self.__total_delta_nm = self.__total_delta_nm + delta_nm

        # this call is not under lock - so recheck the condition upon which we fire the event.
        if first_xdata and current_xdata:
            self.drift_changed_event.fire(delta_nm, delta_time)

    def submit_image(self, xdata: DataAndMetadata.DataAndMetadata, rotation: float, *, wait: bool = False) -> None:
        # set first data if it hasn't been set or if rotation has changed.
        # otherwise, set current data and be ready to measure.
        with self.__lock:
            global _next_image_index
            _next_image_index += 1
            # useful for debugging.
            # numpy.save(f"/Users/cmeyer/Desktop/n{_next_image_index}.npy", xdata.data)
            if self.__first_xdata and math.isclose(self.__rotation, rotation):
                self.__current_xdata = copy.deepcopy(xdata)
            else:
                self.reset()
                self.__first_xdata = copy.deepcopy(xdata)
                self.__rotation = rotation
            if wait:
                self.__calculate()
            else:
                self.__dispatcher.dispatch(self.__calculate)


# used for debugging
_next_image_index = 0


class DriftCorrectionBehavior:
    """Drift correction behavior for updating drift at beginning of each synchronized scan section.

    Take a drift scan from the drift region and send it to the drift compensator.
    """

    def __init__(self,
                 drift_tracker: DriftTracker,
                 scan_hardware_source: scan_base.ScanHardwareSource,
                 scan_frame_parameters: scan_base.ScanFrameParameters,
                 drift_scan_interval: int,
                 *, use_prediction: bool = True) -> None:
        # init with the frame parameters from the synchronized grab
        self.__drift_tracker = drift_tracker
        self.__scan_hardware_source = scan_hardware_source
        self.__scan_frame_parameters = copy.deepcopy(scan_frame_parameters)
        self.__drift_scan_interval = drift_scan_interval
        self.__drift_scan_interval_index = 0
        self.__use_prediction = use_prediction
        # here we convert those frame parameters to the context
        self.__scan_frame_parameters.subscan_pixel_size = None
        self.__scan_frame_parameters.subscan_fractional_size = None
        self.__scan_frame_parameters.subscan_fractional_center = None
        self.__scan_frame_parameters.subscan_rotation = 0.0
        self.__scan_frame_parameters.channel_override = "drift"
        self.__drift_tracker.reset()

    def prepare_section(self, *, utc_time: typing.Optional[datetime.datetime] = None) -> None:
        # if this is called, it means some form of drift-sub-area drift correction has been enabled.
        # if the scan interval is 0, it means every n lines; so do the drift correction here since the
        # each section wil have its own acquisition and this will be called for each section. if the scan
        # interval is non-zero, then only perform drift correction every n scans.
        if self.__drift_scan_interval == 0 or (self.__drift_scan_interval_index % self.__drift_scan_interval) == 0:
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
                aspect_ratio = (context_size.width * drift_region.width) / (context_size.height * drift_region.height)
                # Get the drift scan preferences and change the scan width and dwell time accordingly. The drift
                # preferences can be changed in the preferences panel
                scan_customization = getattr(AcquisitionPreferences.acquisition_preferences, "drift_scan_customization", None)
                TARGET_SIZE = 64
                if scan_customization:
                    frame_parameters.pixel_time_us = scan_customization.dwell_time_us
                    TARGET_SIZE = scan_customization.scan_width_pixels
                if aspect_ratio >= 1.0:
                    if aspect_ratio <= 2.0:
                        shape = Geometry.IntSize(w=TARGET_SIZE, h=int(TARGET_SIZE / aspect_ratio))
                    else:
                        shape = Geometry.IntSize(w=int(TARGET_SIZE // 2 * aspect_ratio), h=TARGET_SIZE // 2)
                else:
                    if aspect_ratio > 0.5:
                        shape = Geometry.IntSize(h=TARGET_SIZE, w=int(TARGET_SIZE * aspect_ratio))
                    else:
                        shape = Geometry.IntSize(h=int(TARGET_SIZE // 2 / aspect_ratio), w=TARGET_SIZE // 2)
                frame_parameters.subscan_pixel_size = shape
                if frame_parameters.subscan_pixel_size[0] >= 8 or frame_parameters.subscan_pixel_size[1] >= 8:
                    frame_parameters.subscan_fractional_size = Geometry.FloatSize(drift_region.height, drift_region.width)
                    frame_parameters.subscan_fractional_center = Geometry.FloatPoint(drift_region.center.y, drift_region.center.x)
                    frame_parameters.subscan_rotation = drift_rotation
                    # attempt to keep drift area in roughly the same position by adding in the accumulated correction.
                    utc_time = utc_time or datetime.datetime.utcnow()
                    delta_nm = self.__drift_tracker.predict_drift(utc_time) if self.__use_prediction else self.__drift_tracker.total_delta_nm
                    frame_parameters.center_nm = frame_parameters.center_nm - delta_nm
                    # print(f"measure with center_nm {frame_parameters.center_nm}")
                    xdatas = self.__scan_hardware_source.record_immediate(frame_parameters, [drift_channel_index])
                    xdata0 = xdatas[0]
                    if xdata0:
                        self.__drift_tracker.submit_image(xdata0, drift_rotation, wait=True)
        self.__drift_scan_interval_index += 1


class DriftCorrectionDataStream(Acquisition.ContainerDataStream):
    """Drift correction data stream.

    The drift correction data stream will capture a scan from the drift region and submit the scan
    to the drift tracker (via the submit_image method).

    The drift tracker can be used to adjust the center_nm of any frame parameters so that the drift
    region is kept in the same relative location in the scan data.
    """

    def __init__(self, drift_correction_behavior: DriftCorrectionBehavior, data_stream: Acquisition.DataStream) -> None:
        super().__init__(data_stream)
        self.__drift_correction_behavior = drift_correction_behavior

    def _prepare_stream(self, stream_args: Acquisition.DataStreamArgs, **kwargs: typing.Any) -> None:
        # during preparation for this section of the scan, let the drift correction behavior capture
        # the drift region and submit it to the drift tracker. the super prepare stream will then call
        # prepare stream of the scan, which can use the drift tracker to adjust the center_nm frame
        # parameter.
        self.__drift_correction_behavior.prepare_section()
        # call this last so that we measure drift before preparing the scan section (which will utilize the measured drift).
        super()._prepare_stream(stream_args, **kwargs)


class DriftCorrectionDataStreamFunctor(Acquisition.DataStreamFunctor):
    """Define a functor to create a drift correction data stream.

    A functor object can be passed to another function and allows the other function to modify a data stream
    it has created with the functor object.
    """

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, scan_frame_parameters: scan_base.ScanFrameParameters, drift_tracker: DriftTracker, drift_scan_interval: int, *, use_prediction: bool = True) -> None:
        self.scan_hardware_source = scan_hardware_source
        self.scan_frame_parameters = scan_frame_parameters
        self.__drift_tracker = drift_tracker
        self.__drift_scan_interval = drift_scan_interval
        self.__use_prediction = use_prediction
        # for testing
        self._drift_correction_data_stream: typing.Optional[DriftCorrectionDataStream] = None

    def apply(self, data_stream: Acquisition.DataStream) -> Acquisition.DataStream:
        assert not self._drift_correction_data_stream
        drift_correction_behavior = DriftCorrectionBehavior(self.__drift_tracker,
                                                            self.scan_hardware_source,
                                                            self.scan_frame_parameters,
                                                            self.__drift_scan_interval,
                                                            use_prediction=self.__use_prediction)
        self._drift_correction_data_stream = DriftCorrectionDataStream(drift_correction_behavior, data_stream)
        return self._drift_correction_data_stream


class DriftUpdaterDataStream(Acquisition.ContainerDataStream):
    """A data stream which watches the first channel (HAADF) and sends its frames to the drift compensator"""

    def __init__(self, data_stream: Acquisition.DataStream, drift_tracker: DriftTracker, drift_rotation: float):
        super().__init__(data_stream)
        self.__drift_tracker = drift_tracker
        self.__drift_rotation = drift_rotation
        self.__channel = data_stream.channels[0]
        self.__framer = Acquisition.Framer(Acquisition.DataAndMetadataDataChannel())

    def _start_stream(self, stream_args: Acquisition.DataStreamArgs) -> None:
        super()._start_stream(stream_args)
        self.__drift_tracker.reset()

    def _fire_data_available(self, data_stream_event: Acquisition.DataStreamEventArgs) -> None:
        if self.__channel == data_stream_event.channel:
            self.__framer.data_available(data_stream_event, typing.cast(Acquisition.FrameCallbacks, self))
        super()._fire_data_available(data_stream_event)

    def _send_data(self, channel: Acquisition.Channel, data_and_metadata: DataAndMetadata.DataAndMetadata) -> None:
        self.__drift_tracker.submit_image(data_and_metadata, self.__drift_rotation)

    def _send_data_multiple(self, channel: Acquisition.Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int) -> None:
        pass


def run() -> None:
    Registry.register_component(DriftTracker(), {"drift_tracker"})

def stop() -> None:
    pass
