from __future__ import annotations

# standard libraries
import copy
import gettext
import math
import numpy
import numpy.typing
import time
import typing
import threading

# other plug-ins
from nion.instrumentation import scan_base
from nion.instrumentation import stem_controller
from nion.utils import Geometry

if typing.TYPE_CHECKING:
    from nion.device_kit import InstrumentDevice

_NDArray = numpy.typing.NDArray[typing.Any]
_DataElementType = typing.Dict[str, typing.Any]

_ = gettext.gettext


class Channel:

    def __init__(self, channel_id: int, name: str, enabled: bool):
        self.channel_id = channel_id
        self.name = name
        self.enabled = enabled
        self.data: typing.Optional[_NDArray] = None


class Frame:

    def __init__(self, frame_number: int, channels: typing.List[Channel], frame_parameters: ScanFrameParameters):
        self.frame_number = frame_number
        self.channels = channels
        self.frame_parameters = frame_parameters
        self.complete = False
        self.bad = False
        self.data_count = 0
        self.start_time = time.perf_counter()
        self.scan_data: typing.Optional[typing.List[_NDArray]] = None


class ScanDataGeneratorLike(typing.Protocol):
    def generate_scan_data(self, instrument: InstrumentDevice.Instrument, scan_frame_parameters: ScanFrameParameters) -> numpy.typing.NDArray[numpy.float32]:
        ...


class ScanSimulatorLike(typing.Protocol):
    scan_shape_pixels: Geometry.IntSize
    pixel_size_nm: Geometry.FloatSize
    external_clock: bool

    @property
    def probe_position_pixels(self) -> Geometry.IntPoint:
        raise NotImplementedError()

    @property
    def blanker_signal_condition(self) -> threading.Condition:
        raise NotImplementedError()

    @property
    def current_pixel_flat(self) -> int:
        raise NotImplementedError()

    def reset_frame(self) -> None: ...

    def _advance_pixel(self, n: int) -> None: ...

    def advance_pixel(self) -> None: ...

    def generate_scan_data(self, instrument: InstrumentDevice.Instrument, scan_frame_parameters: ScanFrameParameters) -> numpy.typing.NDArray[numpy.float32]: ...


class ScanFrameParameters(scan_base.ScanFrameParameters):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)

    @property
    def external_clock_wait_time_ms(self) -> int:
        return typing.cast(int, self.get_parameter("external_clock_wait_time_ms", 0))

    @external_clock_wait_time_ms.setter
    def external_clock_wait_time_ms(self, value: int) -> None:
        self.set_parameter("external_clock_wait_time_ms", value)

    @property
    def external_clock_mode(self) -> int:
        # 0=off, 1=on:rising, 2=on:falling
        return typing.cast(int, self.get_parameter("external_clock_mode", 0))

    @external_clock_mode.setter
    def external_clock_mode(self, value: int) -> None:
        # 0=off, 1=on:rising, 2=on:falling
        self.set_parameter("external_clock_mode", value)

    @property
    def external_scan_mode(self) -> int:
        # 0=off, 1=on:rising, 2=on:falling
        return typing.cast(int, self.get_parameter("external_scan_mode", 0))

    @external_scan_mode.setter
    def external_scan_mode(self, value: int) -> None:
        # 0=off, 1=on:rising, 2=on:falling
        self.set_parameter("external_scan_mode", value)

    @property
    def external_scan_ratio(self) -> float:
        return typing.cast(float, self.get_parameter("external_scan_ratio", 1.0))

    @external_scan_ratio.setter
    def external_scan_ratio(self, value: float) -> None:
        self.set_parameter("external_scan_ratio", value)

    @property
    def ac_frame_sync(self) -> bool:
        return typing.cast(bool, self.get_parameter("ac_frame_sync", False))

    @ac_frame_sync.setter
    def ac_frame_sync(self, value: bool) -> None:
        self.set_parameter("ac_frame_sync", value)

    @property
    def flyback_time_us(self) -> float:
        return typing.cast(float, self.get_parameter("flyback_time_us", 30.0))

    @flyback_time_us.setter
    def flyback_time_us(self, value: float) -> None:
        self.set_parameter("flyback_time_us", value)


class Device(scan_base.ScanDevice):

    def __init__(self, device_id: str, device_name: str, instrument: InstrumentDevice.Instrument, scan_simulator: ScanSimulatorLike) -> None:
        self.scan_device_id = device_id
        self.scan_device_name = device_name
        self.__instrument = instrument
        self.__channels = self.__get_channels()
        self.__frame: typing.Optional[Frame] = None
        self.__frame_number = 0
        self.__is_scanning = False
        self.__flyback_pixels = 2
        self.on_device_state_changed = None
        self.__frame_parameters = ScanFrameParameters()
        self.flyback_pixels = 2
        self.__buffer: typing.List[typing.List[typing.Dict[str, typing.Any]]] = list()
        self.__view_buffer_size = 20
        self.__sequence_buffer_size = 0
        self.__scan_simulator = scan_simulator

    def close(self) -> None:
        pass

    def __get_channels(self) -> typing.List[Channel]:
        return [Channel(0, "HAADF", True), Channel(1, "MAADF", False), Channel(2, "X1", False), Channel(3, "X2", False)]

    @property
    def current_frame_parameters(self) -> scan_base.ScanFrameParameters:
        return self.__frame_parameters

    @property
    def channel_count(self) -> int:
        return len(self.__channels)

    @property
    def channels_enabled(self) -> typing.Tuple[bool, ...]:
        return tuple(channel.enabled for channel in self.__channels)

    @property
    def current_probe_position(self) -> Geometry.FloatPoint:
        # The scan box simulator keeps track of where we are in the currently configured scan. Here we calculate the
        # probe position in the context scan from the current position in the ongoing scan.
        current_frame = self.__frame
        if current_frame is None:
            return Geometry.FloatPoint()
        frame_parameters = current_frame.frame_parameters
        h, w = self.__scan_simulator.scan_shape_pixels
        # calculate relative position within sub-area
        probe_position_pixels = self.__scan_simulator.probe_position_pixels
        ry, rx = probe_position_pixels.y / h - 0.5, probe_position_pixels.x / w - 0.5
        # now translate to context:
        # First get the fractional size of the subscan if we are using one, otherwise this is just (1, 1)
        ss = Geometry.FloatSize.make(frame_parameters.subscan_fractional_size) if frame_parameters.subscan_fractional_size else Geometry.FloatSize(h=1.0, w=1.0)
        # We need the offset of the configured scan to calculate the absolute probe position
        # First is the subscan center if one is used, otherwise this is just (0, 0)
        oo = Geometry.FloatPoint.make(frame_parameters.subscan_fractional_center) - Geometry.FloatPoint(y=0.5, x=0.5) if frame_parameters.subscan_fractional_center else Geometry.FloatPoint()
        # Add the offset of the context scan. Since we are working in fractional coordinates here, we calculate the fractional center
        oo += Geometry.FloatSize(h=frame_parameters.center_nm[0] / frame_parameters.fov_nm, w=frame_parameters.center_nm[1] / frame_parameters.fov_nm)
        # Now add the offset to the relative probe position in the scan
        pt = Geometry.FloatPoint(y=ry * ss.height + oo.y, x=rx * ss.width + oo.x)
        # Apply the scan rotation
        pt = pt.rotate(frame_parameters.rotation_rad)
        # And the subscan rotation if there is one
        if frame_parameters.subscan_rotation:
            pt = pt.rotate(-frame_parameters.subscan_rotation, oo)
        return pt + Geometry.FloatPoint(y=0.5, x=0.5)

    @property
    def blanker_signal_condition(self) -> threading.Condition:
        return self.__scan_simulator.blanker_signal_condition

    def advance_pixel(self, do_sync: bool = False) -> None:
        # if do_sync is True, synchronize with the acquisition thread. this is done by
        # waiting for is_scanning to be True. this should occur quickly in all sitations,
        # so the timeout is short (5s). this will typically only be True for the first pixel.
        if do_sync:
            start = time.time()
            while not self.__is_scanning:
                time.sleep(0.03)
                assert time.time() - start < 5.0
        self.__scan_simulator.advance_pixel()

    def set_channel_enabled(self, channel_index: int, enabled: bool) -> bool:
        assert 0 <= channel_index < self.channel_count
        self.__channels[channel_index].enabled = enabled
        if not any(channel.enabled for channel in self.__channels):
            self.cancel()
        return True

    def get_channel_name(self, channel_index: int) -> str:
        return self.__channels[channel_index].name

    # note: channel typing is just for ease of tests. it can be more strict in the future.
    def get_scan_data(self, frame_parameters: scan_base.ScanFrameParameters, channel: typing.Union[int, Channel]) -> _NDArray:
        """Get the simulated data from the sample simulator"""
        size = Geometry.IntSize.make(frame_parameters.subscan_pixel_size if frame_parameters.subscan_pixel_size else frame_parameters.pixel_size)
        fov_size_nm = Geometry.FloatSize.make(frame_parameters.fov_size_nm) if frame_parameters.fov_size_nm else Geometry.FloatSize(frame_parameters.fov_nm, frame_parameters.fov_nm)
        # If we are doing a subscan calculate the actually used fov
        if frame_parameters.subscan_fractional_size:
            subscan_fractional_size = Geometry.FloatSize.make(frame_parameters.subscan_fractional_size)
            used_fov_size_nm = Geometry.FloatSize(height=fov_size_nm.height * subscan_fractional_size.height, width=fov_size_nm.width * subscan_fractional_size.width)
        else:
            used_fov_size_nm = fov_size_nm
        # Get the scan offset, if we are using a subscan we add the subscan offset to the context offset
        center_nm = Geometry.FloatPoint.make(frame_parameters.center_nm)
        if frame_parameters.subscan_fractional_center:
            subscan_fractional_center = Geometry.FloatPoint.make(frame_parameters.subscan_fractional_center) - Geometry.FloatPoint(y=0.5, x=0.5)
            fc = subscan_fractional_center.rotate(frame_parameters.rotation_rad)
            center_nm += Geometry.FloatPoint(y=fc.y * fov_size_nm.height, x=fc.x * fov_size_nm.width)
        total_rotation = frame_parameters.rotation_rad
        # Apply any rotation (context + subscan)
        if frame_parameters.subscan_rotation:
            total_rotation -= frame_parameters.subscan_rotation
        scan_frame_parameters = ScanFrameParameters(size=size, pixel_time_us=frame_parameters.pixel_time_us, fov_nm=used_fov_size_nm[0], center_nm=center_nm, rotation_rad=total_rotation)
        return self.__scan_simulator.generate_scan_data(self.__instrument, scan_frame_parameters)

    def read_partial(self, frame_number: typing.Optional[int], pixels_to_skip: int) -> typing.Tuple[typing.Sequence[typing.Dict[str, typing.Any]], bool, bool, typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]], typing.Optional[int], int]:
        """Read or continue reading a frame.

        The `frame_number` may be None, in which case a new frame should be read.

        The `frame_number` otherwise specifies which frame to continue reading.

        The `pixels_to_skip` specifies where to start reading the frame, if it is a continuation.

        Return values should be a list of dict's (one for each active channel) containing two keys: 'data' and
        'properties' (see below), followed by a boolean indicating whether the frame is complete, a boolean indicating
        whether the frame was bad, a tuple of the form (top, left), (height, width) indicating the valid sub-area
        of the data, the frame number, and the pixels to skip next time around if the frame is not complete.

        The 'data' keys in the list of dict's should contain a ndarray with the size of the full acquisition and each
        ndarray should be the same size. The 'properties' keys are dicts which must contain the frame parameters and
        a 'channel_id' indicating the index of the channel.
        """

        if self.__frame is None:
            self.__start_next_frame()
        current_frame = self.__frame
        assert current_frame is not None
        frame_number = current_frame.frame_number

        frame_parameters = current_frame.frame_parameters
        size = Geometry.IntSize.make(frame_parameters.subscan_pixel_size if frame_parameters.subscan_pixel_size else frame_parameters.pixel_size)
        total_pixels = size.height * size.width
        time_slice = 0.005  # 5 ms

        if current_frame.scan_data is None:
            scan_data = list()
            for channel in current_frame.channels:
                scan_data.append(self.get_scan_data(current_frame.frame_parameters, channel))
            current_frame.scan_data = scan_data

        is_synchronized_scan = frame_parameters.external_clock_mode != 0

        target_count = 0
        if is_synchronized_scan:
            # In synchronized mode, sleep for the update period and check where we are afterwards since the camera will
            # tell us when to move forward.
            time.sleep(time_slice)
            target_count = self.__scan_simulator.current_pixel_flat + 1
        else:
            while self.__is_scanning and target_count <= current_frame.data_count:
                pixels_remaining = min(total_pixels - current_frame.data_count, int(time_slice * 1e6 / frame_parameters.pixel_time_us) + 1)
                pixel_wait = min(pixels_remaining * frame_parameters.pixel_time_us / 1E6, time_slice)
                time.sleep(pixel_wait)
                target_count = min(int((time.perf_counter() - current_frame.start_time) / (frame_parameters.pixel_time_us / 1E6)), total_pixels)
            if (new_pixels := target_count - self.__scan_simulator.current_pixel_flat) > 0:
                self.__scan_simulator._advance_pixel(new_pixels)

        if self.__is_scanning and target_count > current_frame.data_count:
            for channel_index, channel in enumerate(current_frame.channels):
                assert channel.data is not None
                scan_data_flat = current_frame.scan_data[channel_index].reshape((total_pixels,))
                channel_data_flat = channel.data.reshape((total_pixels,))
                channel_data_flat[current_frame.data_count:target_count] = scan_data_flat[current_frame.data_count:target_count]
            current_frame.data_count = target_count
            current_frame.complete = current_frame.data_count >= total_pixels
        elif not self.__is_scanning:
            current_frame.data_count = total_pixels
            current_frame.complete = True

        data_elements = list()

        for channel in current_frame.channels:
            data_element: _DataElementType = dict()
            data_element["data"] = channel.data
            properties = current_frame.frame_parameters.as_dict()
            properties["center_x_nm"] = current_frame.frame_parameters.center_nm[1]
            properties["center_y_nm"] = current_frame.frame_parameters.center_nm[0]
            properties["rotation_deg"] = math.degrees(current_frame.frame_parameters.rotation_rad)
            properties["channel_id"] = channel.channel_id
            properties["flyback_pixels"] = self.__flyback_pixels
            data_element["properties"] = properties
            data_elements.append(data_element)

        current_rows_read = current_frame.data_count // size.width

        if current_frame.complete:
            sub_area = ((0, 0), size.as_tuple())
            pixels_to_skip = 0
            self.__frame = None
        else:
            sub_area = ((pixels_to_skip // size.width, 0), (current_rows_read - pixels_to_skip // size.width, size.width))
            pixels_to_skip = size.width * current_rows_read

        complete = current_frame.complete
        bad_frame = False

        if complete:
            if len(self.__buffer) > 0 and len(self.__buffer[-1]) != len(data_elements):
                self.__buffer = list()
            self.__buffer.append(data_elements)
            # all new frames go to the end of the buffer list
            # the buffer up to buffer_size is the recording buffer
            # the buffer past the buffer_size is the view buffer
            # the buffer elements past the buffer size can be removed during viewing
            while len(self.__buffer) > self.__sequence_buffer_size + self.__view_buffer_size:
                del self.__buffer[self.__sequence_buffer_size]
            self.__is_scanning = False

        return data_elements, complete, bad_frame, sub_area, frame_number, pixels_to_skip

    def open_configuration_interface(self) -> None:
        """Open settings dialog, if any."""
        pass

    def save_frame_parameters(self) -> None:
        """Called when shutting down. Save frame parameters to persistent storage."""
        pass

    def set_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters) -> None:
        """Called just before and during acquisition.

        Device should use these parameters for new acquisition; and update to these parameters during acquisition.
        """
        self.__frame_parameters = ScanFrameParameters(frame_parameters.as_dict())

    def set_scan_context_probe_position(self, scan_context: stem_controller.ScanContext, probe_position: typing.Optional[Geometry.FloatPoint]) -> None:
        self.__instrument._set_scan_context_probe_position(scan_context, probe_position)

    def set_idle_position_by_percentage(self, x: float, y: float) -> None:
        """Set the idle position as a percentage of the last used frame parameters."""
        pass

    def start_frame(self, is_continuous: bool) -> int:
        """Start acquiring. Return the frame number."""
        if not self.__is_scanning:
            self.__buffer = list()
            self.__start_next_frame()
        return self.__frame_number

    def __start_next_frame(self) -> None:
        frame_parameters = copy.deepcopy(self.__frame_parameters)
        channels = [copy.deepcopy(channel) for channel in self.__channels if channel.enabled]
        size = Geometry.IntSize.make(frame_parameters.subscan_pixel_size if frame_parameters.subscan_pixel_size else frame_parameters.pixel_size)
        for channel in channels:
            channel.data = numpy.zeros(tuple(size), numpy.float32)
        self.__frame_number += 1
        self.__frame = Frame(self.__frame_number, channels, frame_parameters)
        self.__scan_simulator.reset_frame()
        self.__scan_simulator.scan_shape_pixels = size
        if frame_parameters.fov_size_nm is not None:
            self.__scan_simulator.pixel_size_nm = Geometry.FloatSize(frame_parameters.fov_size_nm.height / size.height,
                                                                     frame_parameters.fov_size_nm.width / size.width)
        self.__scan_simulator.external_clock = self.__frame_parameters.external_clock_mode != 0
        self.__is_scanning = True
        self.__flyback_pixels = self.calculate_flyback_pixels(frame_parameters)

    def cancel(self) -> None:
        """Cancel acquisition (immediate)."""
        self.__is_scanning = False

    def stop(self) -> None:
        """Stop acquiring."""
        pass

    @property
    def is_scanning(self) -> bool:
        return self.__is_scanning

    def prepare_synchronized_scan(self, scan_frame_parameters: scan_base.ScanFrameParameters, *, camera_exposure_ms: float, **kwargs: typing.Any) -> None:
        # this method modifies scan_frame_parameters in place, so use base functions to update the fields. bad design.
        scan_frame_parameters.set_parameter("pixel_time_us", min(5120000, int(1000 * camera_exposure_ms * 0.75)))
        scan_frame_parameters.set_parameter("external_clock_wait_time_ms", 20000)  # int(camera_frame_parameters["exposure_ms"] * 1.5)
        scan_frame_parameters.set_parameter("external_clock_mode", 1)

    def set_sequence_buffer_size(self, buffer_size: int) -> None:
        self.__sequence_buffer_size = buffer_size
        self.__buffer = list()

    def get_sequence_buffer_count(self) -> int:
        return len(self.__buffer)

    def pop_sequence_buffer_data(self) -> typing.List[typing.Dict[str, typing.Any]]:
        self.__sequence_buffer_size -= 1
        return self.__buffer.pop(0)

    def get_buffer_data(self, start: int, count: int) -> typing.List[typing.List[typing.Dict[str, typing.Any]]]:
        # print(f"get {start=} {count=} {len(self.__buffer)=}")
        # time.sleep(0.1)
        if start < 0:
            return self.__buffer[start: start+count if count < -start else None]
        else:
            return self.__buffer[start: start+count]

    def calculate_flyback_pixels(self, frame_parameters: scan_base.ScanFrameParameters) -> int:
        return 2
