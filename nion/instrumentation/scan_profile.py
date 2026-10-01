from __future__ import annotations

# standard libraries
import copy
import math
import typing

# local libraries
from nion.utils import Geometry
from nion.instrumentation import scan_base

_VectorType = typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float]]


class ScanProfile(scan_base.ParametersBase):
    """High-level scan profile.

    This is the UI-facing representation of scan settings and is intended for use by the UI and layers above the
    low-level scan implementation in ``scan_base``. Profiles can be copied using ``copy.copy`` or created from a
    dictionary using ``settings.get_frame_parameters_from_dict``.

    The low-level scan implementation continues to use :class:`ScanFrameParameters`. Convert between the two with
    :meth:`to_scan_frame_parameters` and :meth:`from_scan_frame_parameters`.
    """

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(*args, **kwargs)
        if self.has_parameter("size") and not self.has_parameter("pixel_size"):
            self.pixel_size = Geometry.IntSize.make(self.get_parameter("size"))

    @property
    def fov_nm(self) -> float:
        return typing.cast(float, self.get_parameter("fov_nm", 8.0))

    @fov_nm.setter
    def fov_nm(self, value: float) -> None:
        self.set_parameter("fov_nm", value)

    @property
    def rotation_rad(self) -> float:
        return typing.cast(float, self.get_parameter("rotation_rad", 0.0))

    @rotation_rad.setter
    def rotation_rad(self, value: float) -> None:
        self.set_parameter("rotation_rad", value)

    @property
    def center_nm(self) -> Geometry.FloatPoint:
        # parameters must be convertible to JSON; center_nm must be stored as a tuple.
        return Geometry.FloatPoint.make(self.get_parameter("center_nm", (0, 0)))

    @center_nm.setter
    def center_nm(self, value: Geometry.FloatPointTuple) -> None:
        # parameters must be convertible to JSON; center_nm must be stored as a tuple.
        self.set_parameter("center_nm", Geometry.FloatPoint.make(value).as_tuple())

    @property
    def pixel_time_us(self) -> float:
        return typing.cast(float, self.get_parameter("pixel_time_us", 10.0))

    @pixel_time_us.setter
    def pixel_time_us(self, value: float) -> None:
        self.set_parameter("pixel_time_us", value)

    @property
    def pixel_size(self) -> Geometry.IntSize:
        # parameters must be convertible to JSON; pixel_size must be stored as a tuple.
        return Geometry.IntSize.make(self.get_parameter("pixel_size", (512, 512)))

    @pixel_size.setter
    def pixel_size(self, value: Geometry.IntSizeTuple) -> None:
        # parameters must be convertible to JSON; center_nm must be stored as a tuple.
        self.set_parameter("pixel_size", Geometry.IntSize.make(value).as_tuple())

    @property
    def size(self) -> Geometry.IntSize:
        return self.pixel_size

    @size.setter
    def size(self, value: Geometry.IntSize) -> None:
        self.pixel_size = value

    @property
    def scan_size(self) -> Geometry.IntSize:
        if self.subscan_pixel_size:
            return self.subscan_pixel_size
        return self.pixel_size

    @property
    def pixel_size_nm(self) -> Geometry.FloatSize:
        return Geometry.FloatSize(height=self.fov_nm / self.pixel_size.height, width=self.fov_nm / self.pixel_size.width)

    @pixel_size_nm.setter
    def pixel_size_nm(self, pixel_size_nm: Geometry.FloatSize) -> None:
        # the free parameter is always the size; keep the fov fixed.
        if pixel_size_nm and pixel_size_nm.height > 0.0 and pixel_size_nm.width > 0.0:
            self.pixel_size = Geometry.IntSize(
                height=max(1, round(self.fov_nm / pixel_size_nm.height)),
                width=max(1, round(self.fov_nm / pixel_size_nm.width))
            )

    # subscan_type_partial is for internal use only. the intention is that sometimes subscan is used for true
    # subscan acquisition while other times it is used to break a larger acquisition into partial sections.
    # in order to know where to route the data (to a context data item or to a subscan data item), this flag
    # is set when doing partial acquisition.

    @property
    def subscan_type_partial(self) -> bool:
        return typing.cast(bool, self.get_parameter("subscan_type_partial", False))

    @subscan_type_partial.setter
    def subscan_type_partial(self, value: bool) -> None:
        self.set_parameter("subscan_type_partial", value)

    @property
    def subscan_pixel_size(self) -> typing.Optional[Geometry.IntSize]:
        # parameters must be convertible to JSON; subscan_pixel_size must be stored as a tuple.
        subscan_pixel_size_tuple = self.get_parameter("subscan_pixel_size", None)
        return Geometry.IntSize.make(subscan_pixel_size_tuple) if subscan_pixel_size_tuple else None

    @subscan_pixel_size.setter
    def subscan_pixel_size(self, value: typing.Optional[Geometry.IntSizeTuple]) -> None:
        # parameters must be convertible to JSON; subscan_pixel_size must be stored as a tuple.
        self.set_parameter("subscan_pixel_size", Geometry.IntSize.make(value).as_tuple() if value else None)

    @property
    def subscan_fractional_size(self) -> typing.Optional[Geometry.FloatSize]:
        # parameters must be convertible to JSON; subscan_fractional_size must be stored as a tuple.
        subscan_fractional_size_tuple = self.get_parameter("subscan_fractional_size", None)
        return Geometry.FloatSize.make(subscan_fractional_size_tuple) if subscan_fractional_size_tuple else None

    @subscan_fractional_size.setter
    def subscan_fractional_size(self, value: typing.Optional[Geometry.FloatSizeTuple]) -> None:
        # parameters must be convertible to JSON; subscan_fractional_size must be stored as a tuple.
        self.set_parameter("subscan_fractional_size", Geometry.FloatSize.make(value).as_tuple() if value else None)

    @property
    def subscan_fractional_center(self) -> typing.Optional[Geometry.FloatPoint]:
        # parameters must be convertible to JSON; subscan_fractional_center must be stored as a tuple.
        subscan_fractional_center_tuple = self.get_parameter("subscan_fractional_center", None)
        return Geometry.FloatPoint.make(subscan_fractional_center_tuple) if subscan_fractional_center_tuple else None

    @subscan_fractional_center.setter
    def subscan_fractional_center(self, value: typing.Optional[Geometry.FloatPointTuple]) -> None:
        # parameters must be convertible to JSON; subscan_fractional_center must be stored as a tuple.
        self.set_parameter("subscan_fractional_center", Geometry.FloatPoint.make(value).as_tuple() if value else None)

    @property
    def subscan_pixel_size_nm(self) -> typing.Optional[Geometry.FloatSize]:
        if self.subscan_pixel_size and self.subscan_fractional_size and self.subscan_pixel_size.height > 0.0 and self.subscan_pixel_size.width > 0.0:
            return Geometry.FloatSize(
                height=self.subscan_fractional_size.height * self.fov_nm / self.subscan_pixel_size.height,
                width=self.subscan_fractional_size.width * self.fov_nm / self.subscan_pixel_size.width
            )
        return None

    @subscan_pixel_size_nm.setter
    def subscan_pixel_size_nm(self, pixel_size_nm: typing.Optional[Geometry.FloatSize]) -> None:
        # the free parameter is always the subscan size; keep the fov fixed.
        if pixel_size_nm and pixel_size_nm.height > 0.0 and pixel_size_nm.width > 0.0 and self.subscan_fractional_size:
            self.subscan_pixel_size = Geometry.IntSize(
                height=max(1, round(self.subscan_fractional_size.height * self.fov_nm / pixel_size_nm.height)),
                width=max(1, round(self.subscan_fractional_size.width * self.fov_nm / pixel_size_nm.width))
            )

    @property
    def subscan_pixel_width_override(self) -> typing.Optional[int]:
        return typing.cast(typing.Optional[int], self.get_parameter("subscan_pixel_width_override", None))

    @subscan_pixel_width_override.setter
    def subscan_pixel_width_override(self, value: typing.Optional[int]) -> None:
        self.set_parameter("subscan_pixel_width_override", value)

    @property
    def subscan_rotation(self) -> float:
        return typing.cast(float, self.get_parameter("subscan_rotation", 0.0))

    @subscan_rotation.setter
    def subscan_rotation(self, value: float) -> None:
        self.set_parameter("subscan_rotation", value)

    @property
    def line_scan_vector(self) -> _VectorType | None:
        return typing.cast(_VectorType | None, self.get_parameter("line_scan_vector", None))

    @line_scan_vector.setter
    def line_scan_vector(self, value: _VectorType | None) -> None:
        if value:
            start = Geometry.FloatPoint.make(value[0])
            end = Geometry.FloatPoint.make(value[1])
            self.set_parameter("line_scan_vector", (start.as_tuple(), end.as_tuple()))
        else:
            self.set_parameter("line_scan_vector", None)

    @property
    def line_scan_pixel_length(self) -> typing.Optional[int]:
        # parameters must be convertible to JSON; line_scan_pixel_length is stored as an int.
        return typing.cast(typing.Optional[int], self.get_parameter("line_scan_pixel_length", None))

    @line_scan_pixel_length.setter
    def line_scan_pixel_length(self, value: typing.Optional[int]) -> None:
        # parameters must be convertible to JSON; line_scan_pixel_length is stored as an int.
        self.set_parameter("line_scan_pixel_length", value)

    @property
    def ac_line_sync(self) -> bool:
        return typing.cast(bool, self.get_parameter("ac_line_sync", False))

    @ac_line_sync.setter
    def ac_line_sync(self, value: bool) -> None:
        self.set_parameter("ac_line_sync", value)

    @property
    def enabled_channel_indexes(self) -> typing.Optional[typing.Sequence[int]]:
        maybe_enabled_channel_indexes = self.get_parameter("enabled_channel_indexes", None)
        if maybe_enabled_channel_indexes is not None:
            return typing.cast(typing.Sequence[int], maybe_enabled_channel_indexes)
        return None

    @enabled_channel_indexes.setter
    def enabled_channel_indexes(self, value: typing.Sequence[int]) -> None:
        self.set_parameter("enabled_channel_indexes", list(value) if value is not None else None)

    @property
    def channel_variant(self) -> typing.Optional[str]:
        return typing.cast(typing.Optional[str], self.get_parameter("channel_variant", None))

    @channel_variant.setter
    def channel_variant(self, value: typing.Optional[str]) -> None:
        self.set_parameter("channel_variant", value)

    # channel_override for backwards compatibility

    @property
    def channel_override(self) -> typing.Optional[str]:
        return self.channel_variant

    @channel_override.setter
    def channel_override(self, value: typing.Optional[str]) -> None:
        self.channel_variant = value

    @property
    def fov_size_nm(self) -> typing.Optional[Geometry.FloatSize]:
        # return the fov size with the same aspect ratio as the size
        # the largest dimension will be the same as the fov_nm
        if self.pixel_size.aspect_ratio > 1.0:  # width > height
            return Geometry.FloatSize(self.fov_nm / self.pixel_size.aspect_ratio, self.fov_nm)
        else:
            return Geometry.FloatSize(self.fov_nm, self.fov_nm * self.pixel_size.aspect_ratio)

    @property
    def rotation_deg(self) -> float:
        return math.degrees(self.rotation_rad)

    def to_scan_frame_parameters(self, frame_parameters_factory: typing.Callable[[typing.Mapping[str, typing.Any]], typing.Any] | None = None) -> typing.Any:
        if isinstance(self, scan_base.ScanFrameParameters):
            return copy.copy(self)
        if frame_parameters_factory:
            scan_frame_parameters = frame_parameters_factory(dict())
        else:
            scan_frame_parameters = scan_base.ScanFrameParameters()
        scan_frame_parameters._copy_parameters_from(self)
        return scan_frame_parameters

    @classmethod
    def from_scan_frame_parameters(cls, scan_frame_parameters: typing.Any) -> ScanProfile:
        profile = cls()
        profile._copy_parameters_from(scan_frame_parameters)
        return profile

