from __future__ import annotations

# system imports
import asyncio
import copy
import dataclasses
import datetime
import functools
import gettext
import math
import numpy
import numpy.typing
import threading
import typing
import collections

# local libraries
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.data import xdata_1_0 as xd
from nion.instrumentation import Acquisition
from nion.swift.model import DataItem
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import ThreadPool
from nion.utils import Registry

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


@dataclasses.dataclass
class DriftResult:
    """Result of drift calculation returned by DriftCalculator.calculate

    It can interpreted as: Between 'time_window_start' and 'time_window_end' the average drift rate was estimated to
    be 'drift_rate'.

    'drift_rate' should be in units of m / s.
    """

    drift_rate: Geometry.FloatSize
    time_window_start: datetime.datetime
    time_window_end: datetime.datetime


class DriftCalculator:
    """Calculate drift rate from images with timestamps

    Used by DriftTracker to perform the actual drift calculation from submitted images.

    There can be different algorithms for drift calculation. Each of them should be implemented in its own class
    which should be a subclass of this one.

    DriftTracker discovers available drift calculation algorithms from the registry by looking for the component id
    'drift_calculator'.

    You need to set 'drift_calculator_id' to a unique name that will be used to identify a specifc algorithm.
    'drift_calculator_name' is optional and will be used as the display name in a UI based drift calculator selection.
    """
    drift_calculator_id: str
    drift_calculator_name: typing.Optional[str] = None

    def calculate_drift_result(self, first_xdata: typing.Optional[DataAndMetadata.DataAndMetadata], xdata_history: collections.deque) -> typing.Optional[DriftResult]:
        ...


class SimpleDriftCalculator(DriftCalculator):
    drift_calculator_id = 'simple_drift_calculator'

    def calculate_drift_result(self, first_xdata: typing.Optional[DataAndMetadata.DataAndMetadata], xdata_history: collections.deque) -> typing.Optional[DriftResult]:
        current_xdata = xdata_history[-1] if len(xdata_history) else None

        if first_xdata and current_xdata:
            quality, raw_offset = xd.register_template(first_xdata, current_xdata)
            delta_time = (current_xdata.timestamp - first_xdata.timestamp).total_seconds()
            assert delta_time > 0.0
            offset = Geometry.FloatPoint.make(typing.cast(typing.Tuple[float, float], raw_offset))
            # TODO How do we ensure that the data is actually calibrated in nm? It could also be m or even something completely different like rad.
            delta_nm = Geometry.FloatSize(
                    h=current_xdata.dimensional_calibrations[0].convert_to_calibrated_size(offset.y),
                    w=current_xdata.dimensional_calibrations[1].convert_to_calibrated_size(offset.x))
            # print(f"Raw shift nm: {delta_nm}, raw offset px: {raw_offset}, quality: {quality}, delta time: {delta_time}")

            return DriftResult(drift_rate=delta_nm / delta_time * 1e-9, time_window_start=first_xdata.timestamp, time_window_end=current_xdata.timestamp)

        return None


Registry.register_component(SimpleDriftCalculator(), {"drift_calculator"})


class DriftDataSource:
    """Collects images from a camera or scan device and provides them to DriftTracker for calculating drift.

    Each drift data source needs to define in which axis its images are oriented. In addition it can also define an
    extra rotation, which is useful for example for subscan images.

    Each drift data source needs to have a unique id so that "DriftTracker.submit_image" can append images to the
    correct drift data source.
    """

    def __init__(self, *, drift_data_source_id: str, axis: str, applies_drift_correction: bool, rotation: float=0.0, max_history_size: typing.Optional[int]=10) -> None:
        self.__drift_data_source_id = drift_data_source_id
        self.__axis = axis
        self.__applies_drift_correction = applies_drift_correction
        self.__rotation = rotation
        self.__first_xdata: typing.Optional[DataAndMetadata.DataAndMetadata] = None
        self.__xdata_history = collections.deque(maxlen=max_history_size)

    @property
    def drift_data_source_id(self) -> str:
        return self.__drift_data_source_id

    @property
    def xdata_history(self) -> collections.deque:
        return self.__xdata_history

    @property
    def first_xdata(self) -> DataAndMetadata.DataAndMetadata:
        return self.__first_xdata

    @property
    def axis(self) -> str:
        return self.__axis

    @property
    def rotation(self) -> float:
        return self.__rotation

    @property
    def applies_drift_correction(self) -> bool:
        return self.__applies_drift_correction

    def submit_image(self, xdata: DataAndMetadata.DataAndMetadata):
        if self.__first_xdata is None:
            self.__first_xdata = copy.deepcopy(xdata)
        else:
            self.__xdata_history.append(copy.deepcopy(xdata))


class DriftTracker:
    """Track drift state.

    Tracks several properties of drift including the accumulated drift and rate of drift.

    Reset the tracker by calling reset.

    Drift is tracked as a difference from the first data item. This has the obvious limitation of requiring drift to
    be no larger than approximately 1/2 the field of view over the course of acquisition.

    An extension to this class would be to separate the drift algorithm into its own class and allow it to be
    configured.
    """
    def __init__(self, *, max_history_size: typing.Optional[int] = 10, native_axis: typing.Optional[str]=None) -> None:
        self.__native_axis = native_axis or "StageAxis"
        # dispatcher is used to calculate drift offsets on a thread
        self.__dispatcher = ThreadPool.SingleItemDispatcher()

        # the lock controls access to the fields below
        self.__lock = threading.RLock()

        self.__drift_history: typing.List[DriftResult] = list()

        self.drift_changed_event = Event.Event()
        # Set this to an existing drift calculator id to select a specific algorithm for drift calculation
        # If this is set to None or a non-exisiting name we simply use the first algorithm available in the Registry
        self.drift_calculator_id: typing.Optional[str] = None
        # DriftTracker.submit_image() now requires callers to supply a "drift_data_source_id". This needs to be a string
        # that is unique to a source for images used to calculate drift. For each new "drift_data_source_id" we create
        # a separate DriftDataSource that tracks images from this source separately. The calculated drift rates will
        # be shared across all DriftDataSources though. This allows to have different sources for drift images. As an
        # example, a user might do a coarse drift estimation with the Ronchigram camera before going to scan mode. Then
        # in scan mode there is very likely some residual drift because there you are typically working with much higher
        # resolutions. Of course we cannot cross-correlate ronchigram images with scanned images, so we need to track
        # them separately. DriftTracker also converts all measured drift rates to its native axis, which defaults to
        # "StageAxis".
        self.drift_data_sources: typing.Set[DriftDataSource] = set()

    def close(self) -> None:
        self.__dispatcher.close()
        self.__dispatcher = typing.cast(typing.Any, None)

    def reset(self) -> None:
        with self.__lock:
            self.__drift_history.clear()
            self.drift_data_sources.clear()

    @property
    def measurement_count(self) -> int:
        with self.__lock:
            return len(self.__drift_history)

    @property
    def total_delta_nm(self) -> Geometry.FloatSize:
        with self.__lock:
            if not len(self.__drift_history):
                return Geometry.FloatSize()
            total_delta = Geometry.FloatSize()
            time_window_start = self.__drift_history[0].time_window_start
            for drift_result in self.__drift_history:
                total_delta += drift_result.drift_rate * (drift_result.time_window_end - time_window_start).total_seconds()
                time_window_start = drift_result.time_window_end
            return total_delta * 1e9

    @property
    def drift_data_frame(self) -> _NDArray:
        with self.__lock:
            drift_data_frame = numpy.zeros((4, len(self.__drift_history)))
            if not len(self.__drift_history):
                return drift_data_frame

            time_window_start = self.__drift_history[0].time_window_start
            for i, drift_result in enumerate(self.__drift_history):
                delta_time = (drift_result.time_window_end - time_window_start).total_seconds()
                delta_nm = drift_result.drift_rate * delta_time * 1e9
                offset_nm_xy = math.sqrt(pow(delta_nm.height, 2) + pow(delta_nm.width, 2))
                drift_data_frame[:, i] = (delta_nm.height, delta_nm.width, offset_nm_xy, delta_time)
                time_window_start = drift_result.time_window_end
            return numpy.cumsum(drift_data_frame, axis=1)

    @property
    def last_delta_nm(self) -> Geometry.FloatSize:
        with self.__lock:
            if self.measurement_count > 0:
                drift_result = self.__drift_history[-1]
                time_window_start = drift_result.time_window_start
                if self.measurement_count > 1:
                    time_window_start = self.__drift_history[-2].time_window_end
                delta_time = (drift_result.time_window_end - time_window_start).total_seconds()
                delta_nm = drift_result.drift_rate * delta_time * 1e9
            else:
                delta_nm = Geometry.FloatSize()
            return delta_nm

    @property
    def _last_entry_utc_time(self) -> datetime.datetime:
        if self.measurement_count > 0:
            return self.__drift_history[-1].time_window_end
        else:
            return datetime.datetime.utcnow()

    def get_drift_rate(self, *, n: typing.Optional[int] = None) -> Geometry.FloatSize:
        """Return the drift rate over the last n measurements."""
        n = 1 if n is None else n
        assert n > 0
        with self.__lock:
            drift_rate = Geometry.FloatSize()
            if self.measurement_count > 0:
                n = min(n, self.measurement_count)
                for i in range(1, n + 1):
                    drift_rate += self.__drift_history[-i].drift_rate
                drift_rate /= n
            return drift_rate

    def predict_drift(self, utc_time: datetime.datetime, *, n: typing.Optional[int] = None) -> Geometry.FloatSize:
        """Predict total drift (nm) at utc_time."""
        with self.__lock:
            future_delta_nm = Geometry.FloatSize()
            if self.measurement_count > 0:
                delta_time = utc_time - self.__drift_history[-1].time_window_end
                future_delta_nm = delta_time.total_seconds() * self.get_drift_rate(n=n) * 1e9
            return self.total_delta_nm + future_delta_nm

    def __append_drift(self, drift_result: DriftResult) -> None:
        with self.__lock:
            self.__drift_history.append(drift_result)

    def __calculate(self, drift_data_source_id: str) -> None:
        available_calculators: typing.Set[DriftCalculator] = Registry.get_components_by_type("drift_calculator")

        drift_calculator: typing.Optional[DriftCalculator] = None
        drift_result: typing.Optional[DriftResult] = None
        drift_data_source: typing.Optional[DriftDataSource] = None

        for drift_calculator in available_calculators:
            if drift_calculator.drift_calculator_id == self.drift_calculator_id or self.drift_calculator_id is None:
                break

        for drift_data_source in self.drift_data_sources:
                if drift_data_source.drift_data_source_id == drift_data_source_id:
                    break

        with self.__lock:
            if drift_data_source and drift_calculator:
                drift_result = drift_calculator.calculate_drift_result(drift_data_source.first_xdata, drift_data_source.xdata_history)
            if drift_result:
                # rotate back into context reference frame
                if not math.isclose(drift_data_source.rotation, 0.0):
                    drift_result.drift_rate = drift_result.drift_rate.rotate(-drift_data_source.rotation)
                # We don't have that available yet, but it would be great if we could do something like this
                # to have an easier time supporting different axis
                # drift_result.drift_rate = stem_controller.convert_axis(drift_result.drift_rate, from_axis=drift_data_source.axis, to_axis=self.__native_axis)

                # There are two types of drift tracking/correction: 1) We only track drift, 2) We also use the calculated
                # drift to correct the scan or beam position on the sample. In DriftTracker we want to track the total
                # drift, independent from potential corrections. Each DriftDataSource indicates if its data comes
                # from a source that applies drift correction. In this case, the calculated drift rate will be on top
                # of the corrected drift rate, which is - to the best of our knowledge - the drift rate from the previous
                # measurement.
                if drift_data_source.applies_drift_correction:
                    drift_result.drift_rate += self.get_drift_rate(n=1)
                self.__append_drift(drift_result)

        # this call is not under lock - so recheck the condition upon which we fire the event.
        if drift_result:
            delta_time = (drift_result.time_window_end - drift_result.time_window_start).total_seconds()
            delta_nm = drift_result.drift_rate * delta_time * 1e9
            self.drift_changed_event.fire(delta_nm, delta_time)

    def submit_image(self, xdata: DataAndMetadata.DataAndMetadata, rotation: float, *, drift_data_source_id: str, axis: str, drift_correction_applied: bool, wait: bool = False) -> None:
        # set first data if it hasn't been set or if rotation has changed.
        # otherwise, set current data and be ready to measure.
        with self.__lock:
            global _next_image_index
            _next_image_index += 1
            drift_data_source = None
            for drift_data_source in self.drift_data_sources:
                if drift_data_source.drift_data_source_id == drift_data_source_id:
                    if not math.isclose(drift_data_source.rotation, rotation) or drift_data_source.axis != axis or drift_correction_applied != drift_data_source.applies_drift_correction:
                        self.drift_data_sources.remove(drift_data_source)
                        drift_data_source = None
                    break
            # If we did not find "drift_data_source_id" in the existing sources, set "drift_data_osurce" to None so that
            # a new one will be created below.
            else:
                drift_data_source = None

            if drift_data_source is None:
                drift_data_source = DriftDataSource(drift_data_source_id=drift_data_source_id, axis=axis, rotation=rotation, applies_drift_correction=drift_correction_applied)
                self.drift_data_sources.add(drift_data_source)

            drift_data_source.submit_image(xdata)
            # useful for debugging.
            # numpy.save(f"/Users/cmeyer/Desktop/n{_next_image_index}.npy", xdata.data)
            future = self.__dispatcher.dispatch(functools.partial(self.__calculate, drift_data_source_id))
        if wait:
            future.result()

    def submit_drift_result(self, *, drift_rate: Geometry.FloatSize, time_window_start: datetime.datetime, time_window_end: datetime.datetime, axis: str) -> None:
        """Submit a drift rate that has been calculated externally.

        For example tuning also calculates the drift rate, so it can add it to the "globally known drift" through this method.

        "axis" should be the name of the axis that drift_rate was measured in. DriftTracker will then convert it to its
        native axis.
        """
        # TODO convert drift rate to self.__native_axis before adding the new drift result
        self.__append_drift(DriftResult(drift_rate=drift_rate, time_window_start=time_window_start, time_window_end=time_window_end))


# used for debugging
_next_image_index = 0


class DriftCorrectionBehavior:
    """Drift correction behavior for updating drift at beginning of each synchronized scan section.

    Take a drift scan from the drift region and send it to the drift compensator.
    """

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource,
                 scan_frame_parameters: scan_base.ScanFrameParameters, *, use_prediction: bool = True) -> None:
        # init with the frame parameters from the synchronized grab
        self.__scan_hardware_source = scan_hardware_source
        self.__scan_frame_parameters = copy.deepcopy(scan_frame_parameters)
        self.__use_predition = use_prediction
        # here we convert those frame parameters to the context
        self.__scan_frame_parameters.subscan_pixel_size = None
        self.__scan_frame_parameters.subscan_fractional_size = None
        self.__scan_frame_parameters.subscan_fractional_center = None
        self.__scan_frame_parameters.subscan_rotation = 0.0
        self.__scan_frame_parameters.channel_override = "drift"
        self.__scan_hardware_source.drift_tracker.reset()

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
            aspect_ratio = (context_size.width * drift_region.width) / (context_size.height * drift_region.height)
            TARGET_SIZE = 64
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
                drift_tracker = self.__scan_hardware_source.drift_tracker
                utc_time = utc_time or datetime.datetime.utcnow()
                delta_nm = drift_tracker.predict_drift(utc_time) if self.__use_predition else drift_tracker.total_delta_nm
                frame_parameters.center_nm = frame_parameters.center_nm - delta_nm
                # print(f"measure with center_nm {frame_parameters.center_nm}")
                xdatas = self.__scan_hardware_source.record_immediate(frame_parameters, [drift_channel_index])
                xdata0 = xdatas[0]
                if xdata0:
                    drift_tracker.submit_image(xdata0, drift_rotation, drift_data_source_id="drift_correction_behavior_subscan", axis="Scan", drift_correction_applied=True, wait=True)


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
    """Define a functor to create a drift correction data stream."""

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, scan_frame_parameters: scan_base.ScanFrameParameters, *, use_prediction: bool = True) -> None:
        self.scan_hardware_source = scan_hardware_source
        self.scan_frame_parameters = scan_frame_parameters
        self.__use_prediction = use_prediction
        # for testing
        self._drift_correction_data_stream: typing.Optional[DriftCorrectionDataStream] = None

    def apply(self, data_stream: Acquisition.DataStream) -> Acquisition.DataStream:
        assert not self._drift_correction_data_stream
        self._drift_correction_data_stream = DriftCorrectionDataStream(DriftCorrectionBehavior(self.scan_hardware_source, self.scan_frame_parameters, use_prediction=self.__use_prediction), data_stream)
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
        self.__drift_tracker.submit_image(data_and_metadata, self.__drift_rotation, drift_data_source_id="drift_updater_data_stream_haadf", axis="Scan", drift_correction_applied=False)

    def _send_data_multiple(self, channel: Acquisition.Channel, data_and_metadata: DataAndMetadata.DataAndMetadata, count: int) -> None:
        pass


"""
Architectural Decision Records.

# ADR-002 2022-05-05 AM "Use drift_rate as the native quantiy for tracking drift"

This is a consequence of allowing different drift sources and different cross-correlation algorithms. Compared to an
absolute drift, drift_rate can be interpreted without further knowledge about the algorithm that produces it. For example
cross-correlating the first recorded image with the latest one produces a different absolute drift than cross-correlating
the second to last one with the latest one. However, assuming a constant drift rate both methods would result in the same
drift_rate. Of course drift and drift_rate can be converted into each other if the respective time windows used for their
calculation are known. But overall it is drift_rate that we are actually interested in: The sample will usually drift with
a certain drift_rate that typically decreases over time. The drift rate decrease is usually small compared to the sampling
interval, so assuming a constant drift rate between the sampling points is a good approximation. 

# ADR-001 2022-05-05 AM "Introduce DriftDataSource"

DriftTracker is intended as a global place to track sample drift. But for calculating the global drift we can have different
sources like ronchigram images or scanned images. Only images from one source can be cross-correlated and different sources
also produce data oriented in different coordinate systems. In order to be able to globally track sample drift we need to
track the different sources individually.
Each DriftDataSource contains the information needed to correctly interpret the data it produces: In addition to the axis
the data is oriented it can have an extra rotation which is for example required for subscan images. We also need to track
if a DriftDataSource produces data that has drift correction applied, i.e. if the images are shifted according to the last
known drift rate. DriftTracker tracks the total global drift, so for a DriftDataSource that produces drift corrected images
it adds the drift rate calculated from it to the last known drift rate.

"""
