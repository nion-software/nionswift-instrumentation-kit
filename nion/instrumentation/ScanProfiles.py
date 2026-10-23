"""Scan profiles of a scan device.

A scan profile is a named set of values for some of the scan device's parameters, which the user switches between.
ProfileModel is one profile, and ProfileConfigurationModel holds the profiles of a scan device, tracks which of them is
active and follows the scan device's current frame parameters.
"""

from __future__ import annotations

# system imports
import copy
import dataclasses
import gettext
import math
import typing
import uuid

# local libraries
from nion.utils import Event
from nion.utils import Geometry
from nion.utils import Observable
from nion.utils import ReferenceCounting

if typing.TYPE_CHECKING:
    from nion.instrumentation import scan_base


_ = gettext.gettext

# the names of the scan device's hardware profile slots, ordered so that the index of each name is its slot index.
hardware_profile_slot_names: typing.Final[typing.Sequence[str]] = (_("Puma"), _("Rabbit"), _("Frame"))


def _is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_positive_number(value: object) -> bool:
    return _is_finite_number(value) and typing.cast(float, value) > 0


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


# the functions below convert a value read from the saved properties, which may be of any JSON type, so they take
# typing.Any. each returns None for a value which cannot be converted, since a hand edited or damaged file should leave
# the value unchanged rather than prevent the profiles, and so the scan device, from loading.

def _to_float(value: typing.Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _to_int(value: typing.Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _to_bool(value: typing.Any) -> bool | None:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        return None
    return bool(value)


def _has_subscan(frame_parameters: scan_base.ScanFrameParameters) -> bool:
    # the subscan fields are cleared while the subscan is off and hold the line scan while the line scan is on.
    return frame_parameters.subscan_fractional_size is not None and frame_parameters.subscan_fractional_center is not None and not frame_parameters.line_scan_vector


class ProfileModel(Observable.Observable):
    """A named scan profile: values for the scan device's parameters, and which of the parameters it includes.

    A profile has nine parameters: the field width, the size, the pixel time, the rotation, AC line sync, and the subscan
    size, position, pixel width and rotation. Most parameters have one value; the size has three (the width, the height
    and the link between them), and the subscan size and position have two each. A profile includes or excludes a
    parameter as a whole, with one included flag for each parameter.

    Activating a profile applies the values of the parameters it includes to the scan device. A parameter which it does
    not include keeps its current value.

    Setting an invalid value, such as a field width of zero typed in the configuration dialog, is ignored and leaves the
    value unchanged. A line edit bound to the value then shows the unchanged value again.

    While the size is linked, setting the width or the height sets both. The link is saved with the profile and applies
    to every editor of it, so the profile applies it.

    Setting a value or an included flag notifies a change to its property. In addition, values_changed_event fires once
    for each change to the values and included_changed_event once for each change to the included flags, including a
    change to several of them at once, such as reading the values from frame parameters.
    """

    def __init__(self, name: str, profile_uuid: str | None = None) -> None:
        super().__init__()
        self.values_changed_event = Event.Event()
        self.included_changed_event = Event.Event()
        self.__uuid = profile_uuid or str(uuid.uuid4())
        self.__name = str(name)

        # the field of view in nm.
        self.__field_width = 1.0

        # the scan size in pixels.
        self.__width = 256
        self.__height = 256

        self.__size_linked = True
        self.__pixel_time_us = 1.0

        # the scan rotation in degrees.
        self.__rotation = 0.0

        self.__ac_line_sync = False

        # the subscan size is kept in nm rather than as a fraction, so that it keeps its size when the field width changes.
        self.__subscan_field_width_nm = 1.0
        self.__subscan_field_height_nm = 1.0

        # the subscan center as a percentage of the scan.
        self.__subscan_position_x_percent = 50.0
        self.__subscan_position_y_percent = 50.0

        # None is automatic: the subscan pixel width follows the subscan size and the scan size.
        self.__subscan_pixel_width: int | None = None

        # the subscan rotation in degrees.
        self.__subscan_rotation = 0.0

        self.__field_width_included = True
        self.__size_included = True
        self.__pixel_time_us_included = True
        self.__rotation_included = True
        self.__ac_line_sync_included = True
        self.__subscan_field_size_included = True
        self.__subscan_fractional_position_included = True
        self.__subscan_pixel_width_included = True
        self.__subscan_rotation_included = True

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str) -> None:
        if value != self.__name:
            self.__name = value
            self.notify_property_changed("name")

    @property
    def uuid(self) -> str:
        return self.__uuid

    # the values. each private setter validates and sets its value and returns whether the value changed, so that a
    # change to several values can notify once.

    @property
    def field_width(self) -> float:
        return self.__field_width

    @field_width.setter
    def field_width(self, field_width: float) -> None:
        if self.__set_field_width(field_width):
            self.notify_property_changed("field_width")
            self.values_changed_event.fire()

    def __set_field_width(self, field_width: float) -> bool:
        if not _is_positive_number(field_width) or field_width == self.__field_width:
            return False
        self.__field_width = float(field_width)
        return True

    @property
    def width(self) -> int:
        return self.__width

    @width.setter
    def width(self, width: int) -> None:
        if self.__set_size(width, self.__height, self.__size_linked):
            self.__notify_size_changed()

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    def height(self, height: int) -> None:
        # while the size is linked, the width follows the height.
        if self.__set_size(height if self.__size_linked else self.__width, height, self.__size_linked):
            self.__notify_size_changed()

    @property
    def size_linked(self) -> bool:
        return self.__size_linked

    @size_linked.setter
    def size_linked(self, size_linked: bool) -> None:
        if self.__set_size(self.__width, self.__height, size_linked):
            self.__notify_size_changed()

    def __set_size(self, width: int, height: int, size_linked: bool) -> bool:
        if not (_is_positive_int(width) and _is_positive_int(height) and isinstance(size_linked, bool)):
            return False
        # while the size is linked, the height follows the width.
        height = width if size_linked else height
        if (width, height, size_linked) == (self.__width, self.__height, self.__size_linked):
            return False
        self.__width = width
        self.__height = height
        self.__size_linked = size_linked
        return True

    def __notify_size_changed(self) -> None:
        self.notify_property_changed("width")
        self.notify_property_changed("height")
        self.notify_property_changed("size_linked")
        self.values_changed_event.fire()

    @property
    def pixel_time_us(self) -> float:
        return self.__pixel_time_us

    @pixel_time_us.setter
    def pixel_time_us(self, pixel_time_us: float) -> None:
        if self.__set_pixel_time_us(pixel_time_us):
            self.notify_property_changed("pixel_time_us")
            self.values_changed_event.fire()

    def __set_pixel_time_us(self, pixel_time_us: float) -> bool:
        if not _is_positive_number(pixel_time_us) or pixel_time_us == self.__pixel_time_us:
            return False
        self.__pixel_time_us = float(pixel_time_us)
        return True

    @property
    def rotation(self) -> float:
        return self.__rotation

    @rotation.setter
    def rotation(self, rotation: float) -> None:
        if self.__set_rotation(rotation):
            self.notify_property_changed("rotation")
            self.values_changed_event.fire()

    def __set_rotation(self, rotation: float) -> bool:
        if not _is_finite_number(rotation) or rotation == self.__rotation:
            return False
        self.__rotation = float(rotation)
        return True

    @property
    def ac_line_sync(self) -> bool:
        return self.__ac_line_sync

    @ac_line_sync.setter
    def ac_line_sync(self, ac_line_sync: bool) -> None:
        if self.__set_ac_line_sync(ac_line_sync):
            self.notify_property_changed("ac_line_sync")
            self.values_changed_event.fire()

    def __set_ac_line_sync(self, ac_line_sync: bool) -> bool:
        if not isinstance(ac_line_sync, bool) or ac_line_sync == self.__ac_line_sync:
            return False
        self.__ac_line_sync = ac_line_sync
        return True

    @property
    def subscan_field_width_nm(self) -> float:
        return self.__subscan_field_width_nm

    @subscan_field_width_nm.setter
    def subscan_field_width_nm(self, subscan_field_width_nm: float) -> None:
        if self.__set_subscan_field_size(subscan_field_width_nm, self.__subscan_field_height_nm):
            self.__notify_subscan_field_size_changed()

    @property
    def subscan_field_height_nm(self) -> float:
        return self.__subscan_field_height_nm

    @subscan_field_height_nm.setter
    def subscan_field_height_nm(self, subscan_field_height_nm: float) -> None:
        if self.__set_subscan_field_size(self.__subscan_field_width_nm, subscan_field_height_nm):
            self.__notify_subscan_field_size_changed()

    def __set_subscan_field_size(self, subscan_field_width_nm: float, subscan_field_height_nm: float) -> bool:
        if not (_is_positive_number(subscan_field_width_nm) and _is_positive_number(subscan_field_height_nm)):
            return False
        if (subscan_field_width_nm, subscan_field_height_nm) == (self.__subscan_field_width_nm, self.__subscan_field_height_nm):
            return False
        self.__subscan_field_width_nm = float(subscan_field_width_nm)
        self.__subscan_field_height_nm = float(subscan_field_height_nm)
        return True

    def __notify_subscan_field_size_changed(self) -> None:
        self.notify_property_changed("subscan_field_width_nm")
        self.notify_property_changed("subscan_field_height_nm")
        self.values_changed_event.fire()

    @property
    def subscan_position_x_percent(self) -> float:
        return self.__subscan_position_x_percent

    @subscan_position_x_percent.setter
    def subscan_position_x_percent(self, subscan_position_x_percent: float) -> None:
        if self.__set_subscan_position(subscan_position_x_percent, self.__subscan_position_y_percent):
            self.__notify_subscan_position_changed()

    @property
    def subscan_position_y_percent(self) -> float:
        return self.__subscan_position_y_percent

    @subscan_position_y_percent.setter
    def subscan_position_y_percent(self, subscan_position_y_percent: float) -> None:
        if self.__set_subscan_position(self.__subscan_position_x_percent, subscan_position_y_percent):
            self.__notify_subscan_position_changed()

    def __set_subscan_position(self, subscan_position_x_percent: float, subscan_position_y_percent: float) -> bool:
        if not (_is_finite_number(subscan_position_x_percent) and _is_finite_number(subscan_position_y_percent)):
            return False
        if (subscan_position_x_percent, subscan_position_y_percent) == (self.__subscan_position_x_percent, self.__subscan_position_y_percent):
            return False
        self.__subscan_position_x_percent = float(subscan_position_x_percent)
        self.__subscan_position_y_percent = float(subscan_position_y_percent)
        return True

    def __notify_subscan_position_changed(self) -> None:
        self.notify_property_changed("subscan_position_x_percent")
        self.notify_property_changed("subscan_position_y_percent")
        self.values_changed_event.fire()

    @property
    def subscan_pixel_width(self) -> int | None:
        return self.__subscan_pixel_width

    @subscan_pixel_width.setter
    def subscan_pixel_width(self, subscan_pixel_width: int | None) -> None:
        if self.__set_subscan_pixel_width(subscan_pixel_width):
            self.notify_property_changed("subscan_pixel_width")
            self.values_changed_event.fire()

    def __set_subscan_pixel_width(self, subscan_pixel_width: int | None) -> bool:
        if not (subscan_pixel_width is None or _is_positive_int(subscan_pixel_width)) or subscan_pixel_width == self.__subscan_pixel_width:
            return False
        self.__subscan_pixel_width = subscan_pixel_width
        return True

    @property
    def subscan_rotation(self) -> float:
        return self.__subscan_rotation

    @subscan_rotation.setter
    def subscan_rotation(self, subscan_rotation: float) -> None:
        if self.__set_subscan_rotation(subscan_rotation):
            self.notify_property_changed("subscan_rotation")
            self.values_changed_event.fire()

    def __set_subscan_rotation(self, subscan_rotation: float) -> bool:
        if not _is_finite_number(subscan_rotation) or subscan_rotation == self.__subscan_rotation:
            return False
        self.__subscan_rotation = float(subscan_rotation)
        return True

    def __notify_values_changed(self) -> None:
        # a change to several values notifies every value, rather than tracking which of them changed.
        self.notify_property_changed("field_width")
        self.notify_property_changed("width")
        self.notify_property_changed("height")
        self.notify_property_changed("size_linked")
        self.notify_property_changed("pixel_time_us")
        self.notify_property_changed("rotation")
        self.notify_property_changed("ac_line_sync")
        self.notify_property_changed("subscan_field_width_nm")
        self.notify_property_changed("subscan_field_height_nm")
        self.notify_property_changed("subscan_position_x_percent")
        self.notify_property_changed("subscan_position_y_percent")
        self.notify_property_changed("subscan_pixel_width")
        self.notify_property_changed("subscan_rotation")
        self.values_changed_event.fire()

    # the included flags. as with the values, each private setter returns whether the flag changed.

    @property
    def field_width_included(self) -> bool:
        return self.__field_width_included

    @field_width_included.setter
    def field_width_included(self, included: bool) -> None:
        if self.__set_field_width_included(included):
            self.notify_property_changed("field_width_included")
            self.included_changed_event.fire()

    def __set_field_width_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__field_width_included:
            return False
        self.__field_width_included = included
        return True

    @property
    def size_included(self) -> bool:
        return self.__size_included

    @size_included.setter
    def size_included(self, included: bool) -> None:
        if self.__set_size_included(included):
            self.notify_property_changed("size_included")
            self.included_changed_event.fire()

    def __set_size_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__size_included:
            return False
        self.__size_included = included
        return True

    @property
    def pixel_time_us_included(self) -> bool:
        return self.__pixel_time_us_included

    @pixel_time_us_included.setter
    def pixel_time_us_included(self, included: bool) -> None:
        if self.__set_pixel_time_us_included(included):
            self.notify_property_changed("pixel_time_us_included")
            self.included_changed_event.fire()

    def __set_pixel_time_us_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__pixel_time_us_included:
            return False
        self.__pixel_time_us_included = included
        return True

    @property
    def rotation_included(self) -> bool:
        return self.__rotation_included

    @rotation_included.setter
    def rotation_included(self, included: bool) -> None:
        if self.__set_rotation_included(included):
            self.notify_property_changed("rotation_included")
            self.included_changed_event.fire()

    def __set_rotation_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__rotation_included:
            return False
        self.__rotation_included = included
        return True

    @property
    def ac_line_sync_included(self) -> bool:
        return self.__ac_line_sync_included

    @ac_line_sync_included.setter
    def ac_line_sync_included(self, included: bool) -> None:
        if self.__set_ac_line_sync_included(included):
            self.notify_property_changed("ac_line_sync_included")
            self.included_changed_event.fire()

    def __set_ac_line_sync_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__ac_line_sync_included:
            return False
        self.__ac_line_sync_included = included
        return True

    @property
    def subscan_field_size_included(self) -> bool:
        return self.__subscan_field_size_included

    @subscan_field_size_included.setter
    def subscan_field_size_included(self, included: bool) -> None:
        if self.__set_subscan_field_size_included(included):
            self.notify_property_changed("subscan_field_size_included")
            self.included_changed_event.fire()

    def __set_subscan_field_size_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__subscan_field_size_included:
            return False
        self.__subscan_field_size_included = included
        return True

    @property
    def subscan_fractional_position_included(self) -> bool:
        return self.__subscan_fractional_position_included

    @subscan_fractional_position_included.setter
    def subscan_fractional_position_included(self, included: bool) -> None:
        if self.__set_subscan_fractional_position_included(included):
            self.notify_property_changed("subscan_fractional_position_included")
            self.included_changed_event.fire()

    def __set_subscan_fractional_position_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__subscan_fractional_position_included:
            return False
        self.__subscan_fractional_position_included = included
        return True

    @property
    def subscan_pixel_width_included(self) -> bool:
        return self.__subscan_pixel_width_included

    @subscan_pixel_width_included.setter
    def subscan_pixel_width_included(self, included: bool) -> None:
        if self.__set_subscan_pixel_width_included(included):
            self.notify_property_changed("subscan_pixel_width_included")
            self.included_changed_event.fire()

    def __set_subscan_pixel_width_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__subscan_pixel_width_included:
            return False
        self.__subscan_pixel_width_included = included
        return True

    @property
    def subscan_rotation_included(self) -> bool:
        return self.__subscan_rotation_included

    @subscan_rotation_included.setter
    def subscan_rotation_included(self, included: bool) -> None:
        if self.__set_subscan_rotation_included(included):
            self.notify_property_changed("subscan_rotation_included")
            self.included_changed_event.fire()

    def __set_subscan_rotation_included(self, included: bool) -> bool:
        if not isinstance(included, bool) or included == self.__subscan_rotation_included:
            return False
        self.__subscan_rotation_included = included
        return True

    def __notify_included_changed(self) -> None:
        # a change to several included flags notifies every flag, rather than tracking which of them changed.
        self.notify_property_changed("field_width_included")
        self.notify_property_changed("size_included")
        self.notify_property_changed("pixel_time_us_included")
        self.notify_property_changed("rotation_included")
        self.notify_property_changed("ac_line_sync_included")
        self.notify_property_changed("subscan_field_size_included")
        self.notify_property_changed("subscan_fractional_position_included")
        self.notify_property_changed("subscan_pixel_width_included")
        self.notify_property_changed("subscan_rotation_included")
        self.included_changed_event.fire()

    def copy_values_from(self, other: ProfileModel, included_by: ProfileModel | None = None) -> None:
        """Copy the values of the other profile into this one, or only the values of the parameters which included_by includes, if given.

        The included flags of this profile are unchanged.
        """
        changed = False
        if included_by is None or included_by.field_width_included:
            changed |= self.__set_field_width(other.field_width)
        if included_by is None or included_by.size_included:
            changed |= self.__set_size(other.width, other.height, other.size_linked)
        if included_by is None or included_by.pixel_time_us_included:
            changed |= self.__set_pixel_time_us(other.pixel_time_us)
        if included_by is None or included_by.rotation_included:
            changed |= self.__set_rotation(other.rotation)
        if included_by is None or included_by.ac_line_sync_included:
            changed |= self.__set_ac_line_sync(other.ac_line_sync)
        if included_by is None or included_by.subscan_field_size_included:
            changed |= self.__set_subscan_field_size(other.subscan_field_width_nm, other.subscan_field_height_nm)
        if included_by is None or included_by.subscan_fractional_position_included:
            changed |= self.__set_subscan_position(other.subscan_position_x_percent, other.subscan_position_y_percent)
        if included_by is None or included_by.subscan_pixel_width_included:
            changed |= self.__set_subscan_pixel_width(other.subscan_pixel_width)
        if included_by is None or included_by.subscan_rotation_included:
            changed |= self.__set_subscan_rotation(other.subscan_rotation)
        if changed:
            self.__notify_values_changed()

    def copy_included_from(self, other: ProfileModel) -> None:
        """Copy the included flags of the other profile into this one."""
        changed = self.__set_field_width_included(other.field_width_included)
        changed |= self.__set_size_included(other.size_included)
        changed |= self.__set_pixel_time_us_included(other.pixel_time_us_included)
        changed |= self.__set_rotation_included(other.rotation_included)
        changed |= self.__set_ac_line_sync_included(other.ac_line_sync_included)
        changed |= self.__set_subscan_field_size_included(other.subscan_field_size_included)
        changed |= self.__set_subscan_fractional_position_included(other.subscan_fractional_position_included)
        changed |= self.__set_subscan_pixel_width_included(other.subscan_pixel_width_included)
        changed |= self.__set_subscan_rotation_included(other.subscan_rotation_included)
        if changed:
            self.__notify_included_changed()

    def read_from_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters) -> None:
        """Read the values which the frame parameters carry, leaving the others and any invalid ones unchanged.

        The frame parameters carry the subscan values only while the subscan is on.
        """
        width, height = int(frame_parameters.pixel_size.width), int(frame_parameters.pixel_size.height)
        changed = self.__set_field_width(float(frame_parameters.fov_nm))
        # the frame parameters do not carry the size link, so it is kept, unless the width and height differ.
        changed |= self.__set_size(width, height, self.__size_linked and width == height)
        changed |= self.__set_pixel_time_us(float(frame_parameters.pixel_time_us))
        changed |= self.__set_rotation(float(frame_parameters.rotation_rad * 180.0 / math.pi))
        changed |= self.__set_ac_line_sync(bool(frame_parameters.ac_line_sync))
        changed |= self.__set_subscan_pixel_width(frame_parameters.subscan_pixel_width_override)
        subscan_fractional_size = frame_parameters.subscan_fractional_size
        subscan_fractional_center = frame_parameters.subscan_fractional_center
        if subscan_fractional_size and subscan_fractional_center and not frame_parameters.line_scan_vector:
            # the subscan size is a fraction of the field width, so an invalid field width leaves the subscan size unchanged.
            if _is_positive_number(field_width_nm := float(frame_parameters.fov_nm)):
                changed |= self.__set_subscan_field_size(float(subscan_fractional_size.width) * field_width_nm, float(subscan_fractional_size.height) * field_width_nm)
            changed |= self.__set_subscan_position(100.0 * float(subscan_fractional_center.x), 100.0 * float(subscan_fractional_center.y))
            changed |= self.__set_subscan_rotation(float(frame_parameters.subscan_rotation * 180.0 / math.pi))
        if changed:
            self.__notify_values_changed()

    def write_to_frame_parameters(self, frame_parameters: scan_base.ScanFrameParameters) -> None:
        """Write every value into the frame parameters, including the subscan values."""
        frame_parameters.fov_nm = self.__field_width
        frame_parameters.pixel_size = Geometry.IntSize(self.__height, self.__width)
        frame_parameters.pixel_time_us = self.__pixel_time_us
        frame_parameters.rotation_rad = self.__rotation * math.pi / 180.0
        frame_parameters.ac_line_sync = self.__ac_line_sync
        # the subscan size is kept in nm, so its fraction of the scan follows the field width of this profile.
        frame_parameters.subscan_fractional_size = Geometry.FloatSize(height=self.__subscan_field_height_nm / self.__field_width, width=self.__subscan_field_width_nm / self.__field_width)
        frame_parameters.subscan_fractional_center = Geometry.FloatPoint(x=self.__subscan_position_x_percent / 100.0, y=self.__subscan_position_y_percent / 100.0)
        frame_parameters.subscan_pixel_width_override = self.__subscan_pixel_width
        frame_parameters.subscan_rotation = self.__subscan_rotation * math.pi / 180.0

    def write_values_to_dict(self) -> dict[str, typing.Any]:
        """Return the properties from which the values and the included flags can be restored.

        The properties are of the types which a JSON file holds.
        """
        return {
            "field_width": self.__field_width,
            "width": self.__width,
            "height": self.__height,
            "width_height_linked": self.__size_linked,
            "pixel_time_us": self.__pixel_time_us,
            "rotation": self.__rotation,
            "ac_line_sync": self.__ac_line_sync,
            "subscan_field_width_nm": self.__subscan_field_width_nm,
            "subscan_field_height_nm": self.__subscan_field_height_nm,
            "subscan_fractional_position_x_percent": self.__subscan_position_x_percent,
            "subscan_fractional_position_y_percent": self.__subscan_position_y_percent,
            "subscan_pixel_width": self.__subscan_pixel_width,
            "subscan_rotation": self.__subscan_rotation,
            "field_width_included": self.__field_width_included,
            "size_included": self.__size_included,
            "pixel_time_us_included": self.__pixel_time_us_included,
            "rotation_included": self.__rotation_included,
            "ac_line_sync_included": self.__ac_line_sync_included,
            "subscan_field_size_included": self.__subscan_field_size_included,
            "subscan_fractional_position_included": self.__subscan_fractional_position_included,
            "subscan_pixel_width_included": self.__subscan_pixel_width_included,
            "subscan_rotation_included": self.__subscan_rotation_included,
        }

    def read_values_from_dict(self, properties: typing.Mapping[str, typing.Any]) -> None:
        """Read the values and the included flags from the properties, as written by write_values_to_dict.

        A value or an included flag which the properties do not hold, or hold with an invalid value, such as one edited
        by hand in the saved file, is left unchanged, rather than preventing the profiles, and so the scan device, from
        loading.
        """
        # a missing part of a size or a pair is its current value, but an invalid part leaves the whole size or pair unchanged.
        width = _to_int(properties.get("width", self.__width))
        height = _to_int(properties.get("height", self.__height))
        size_linked = _to_bool(properties.get("width_height_linked", self.__size_linked))
        subscan_field_width_nm = _to_float(properties.get("subscan_field_width_nm", self.__subscan_field_width_nm))
        subscan_field_height_nm = _to_float(properties.get("subscan_field_height_nm", self.__subscan_field_height_nm))
        subscan_position_x_percent = _to_float(properties.get("subscan_fractional_position_x_percent", self.__subscan_position_x_percent))
        subscan_position_y_percent = _to_float(properties.get("subscan_fractional_position_y_percent", self.__subscan_position_y_percent))
        values_changed = False
        if (field_width := _to_float(properties.get("field_width"))) is not None:
            values_changed |= self.__set_field_width(field_width)
        if width is not None and height is not None and size_linked is not None:
            values_changed |= self.__set_size(width, height, size_linked)
        if (pixel_time_us := _to_float(properties.get("pixel_time_us"))) is not None:
            values_changed |= self.__set_pixel_time_us(pixel_time_us)
        if (rotation := _to_float(properties.get("rotation"))) is not None:
            values_changed |= self.__set_rotation(rotation)
        if "ac_line_sync" in properties and (ac_line_sync := _to_bool(properties["ac_line_sync"])) is not None:
            values_changed |= self.__set_ac_line_sync(ac_line_sync)
        if subscan_field_width_nm is not None and subscan_field_height_nm is not None:
            values_changed |= self.__set_subscan_field_size(subscan_field_width_nm, subscan_field_height_nm)
        if subscan_position_x_percent is not None and subscan_position_y_percent is not None:
            values_changed |= self.__set_subscan_position(subscan_position_x_percent, subscan_position_y_percent)
        # a subscan pixel width of None is automatic, so it is set rather than ignored.
        if "subscan_pixel_width" in properties:
            saved_subscan_pixel_width = properties["subscan_pixel_width"]
            subscan_pixel_width = _to_int(saved_subscan_pixel_width) if saved_subscan_pixel_width is not None else None
            if saved_subscan_pixel_width is None or subscan_pixel_width is not None:
                values_changed |= self.__set_subscan_pixel_width(subscan_pixel_width)
        if (subscan_rotation := _to_float(properties.get("subscan_rotation"))) is not None:
            values_changed |= self.__set_subscan_rotation(subscan_rotation)
        included_changed = False
        if "field_width_included" in properties and (included := _to_bool(properties["field_width_included"])) is not None:
            included_changed |= self.__set_field_width_included(included)
        if "size_included" in properties and (included := _to_bool(properties["size_included"])) is not None:
            included_changed |= self.__set_size_included(included)
        if "pixel_time_us_included" in properties and (included := _to_bool(properties["pixel_time_us_included"])) is not None:
            included_changed |= self.__set_pixel_time_us_included(included)
        if "rotation_included" in properties and (included := _to_bool(properties["rotation_included"])) is not None:
            included_changed |= self.__set_rotation_included(included)
        if "ac_line_sync_included" in properties and (included := _to_bool(properties["ac_line_sync_included"])) is not None:
            included_changed |= self.__set_ac_line_sync_included(included)
        if "subscan_field_size_included" in properties and (included := _to_bool(properties["subscan_field_size_included"])) is not None:
            included_changed |= self.__set_subscan_field_size_included(included)
        if "subscan_fractional_position_included" in properties and (included := _to_bool(properties["subscan_fractional_position_included"])) is not None:
            included_changed |= self.__set_subscan_fractional_position_included(included)
        if "subscan_pixel_width_included" in properties and (included := _to_bool(properties["subscan_pixel_width_included"])) is not None:
            included_changed |= self.__set_subscan_pixel_width_included(included)
        if "subscan_rotation_included" in properties and (included := _to_bool(properties["subscan_rotation_included"])) is not None:
            included_changed |= self.__set_subscan_rotation_included(included)
        if values_changed:
            self.__notify_values_changed()
        if included_changed:
            self.__notify_included_changed()

    def to_dict(self) -> dict[str, typing.Any]:
        """Return the properties from which the profile, including its identity and name, can be restored."""
        return {"uuid": self.uuid, "name": self.name} | self.write_values_to_dict()

    @classmethod
    def from_dict(cls, properties: typing.Mapping[str, typing.Any]) -> ProfileModel:
        """Return a profile restored from the properties, with a new identity if they have none."""
        profile = cls(str(properties.get("name") or _("Profile")), str(properties.get("uuid") or str()))
        profile.read_values_from_dict(properties)
        return profile


@dataclasses.dataclass(frozen=True)
class SavedScanProfiles:
    """The scan profiles read from a saved profiles file, as the profile configuration model restores them.

    profile_properties_list: the properties of each saved profile, in order. The properties are mappings from JSON,
        which cannot be made immutable, so the model only reads them.
    active_profile_uuid: the identity of the saved active profile, or empty if none is saved.
    previous_profile_uuid: the identity of the saved previous profile, or empty if none is saved.
    current_values_properties: the saved current values, or None if none are saved or they are not a mapping. A
        mapping from JSON, which cannot be made immutable, so the model only reads it.
    has_dropped_properties: whether part or all of the saved properties were dropped because their structure could
        not be used. Invalid individual values are not counted, since a profile keeps its value for each of them.
    """
    # the values of the mappings are any JSON value, so they are typed as typing.Any.
    profile_properties_list: typing.Sequence[typing.Mapping[str, typing.Any]]
    active_profile_uuid: str
    previous_profile_uuid: str
    current_values_properties: typing.Mapping[str, typing.Any] | None
    has_dropped_properties: bool


def parse_saved_scan_profiles(properties: object) -> SavedScanProfiles:
    """Return the scan profiles in the properties read from a saved profiles file.

    The properties are any JSON value, such as from a damaged or hand edited file. Properties of the wrong structure
    are dropped, in whole or for the profile concerned, and the result records that they were dropped.
    """
    if not isinstance(properties, typing.Mapping):
        # a value which is not a mapping is dropped, except null, which is treated as no saved profiles, as if the
        # file were absent.
        return SavedScanProfiles(tuple(), str(), str(), None, properties is not None)
    profile_properties_list = properties.get("profiles")
    if not isinstance(profile_properties_list, list):
        return SavedScanProfiles(tuple(), str(), str(), None, True)
    usable_profile_properties_list = tuple(profile_properties for profile_properties in profile_properties_list if isinstance(profile_properties, typing.Mapping))
    active_profile_uuid = properties.get("active_profile_uuid")
    previous_profile_uuid = properties.get("previous_profile_uuid")
    current_values_properties = properties.get("current_values")
    return SavedScanProfiles(usable_profile_properties_list,
                             str(active_profile_uuid) if active_profile_uuid is not None else str(),
                             str(previous_profile_uuid) if previous_profile_uuid is not None else str(),
                             current_values_properties if isinstance(current_values_properties, typing.Mapping) else None,
                             len(usable_profile_properties_list) != len(profile_properties_list))


class ProfileConfigurationModel(Observable.Observable):
    """The scan profiles of a scan device and which of them is active.

    The profiles are restored from the saved scan profiles, as parsed from the properties of a previous
    profiles_changed_event, if there are any; otherwise they are made from the profile slots of the scan device. A
    restored active profile is applied to the scan device.

    The profiles_changed_event fires with the properties to save whenever the profiles or the active profile change.

    Changes to the current frame parameters and subscan of the scan device are edits to the current values.

    The model is confined to the main thread and does not protect its state against concurrent access. This includes
    the scan device's changes to its current frame parameters and subscan, which the model follows as they happen, so
    the scan device is expected to make those changes on the main thread.
    """

    def __init__(self, scan_hardware_source: scan_base.ScanHardwareSource, saved_scan_profiles: SavedScanProfiles | None = None) -> None:
        super().__init__()
        self.__scan_hardware_source = scan_hardware_source
        self.profiles_changed_event = Event.Event()
        self.__saved_properties: dict[str, typing.Any] | None = None
        self.__profiles: list[ProfileModel] = list()

        # the listeners for each profile, in the same order as the profiles, so that a change to a profile updates the
        # names, is applied if the profile is active, and is saved.
        self.__profile_listeners: list[typing.Sequence[Event.EventListener]] = list()

        # the active and previous profiles are tracked by identity so that they are unaffected by changes to the
        # order of the profiles.
        self.__previous_profile: ProfileModel | None = None

        # the current values, which are what the scan device scans with, including the subscan values while the
        # subscan is off, so that they are saved. they are not a profile, so they have no name, and their included
        # flags are unused.
        self.__current_values = ProfileModel(str())
        self.__current_values.read_from_frame_parameters(self.__get_frame_parameters_with_subscan(scan_hardware_source.get_current_frame_parameters()))

        # set while the active profile and the current values are being made consistent, so that the change which
        # results is not propagated back again.
        self.__is_synchronizing = False

        if saved_scan_profiles and saved_scan_profiles.profile_properties_list:
            for profile_properties in saved_scan_profiles.profile_properties_list:
                profile = ProfileModel.from_dict(profile_properties)
                profile.name = self.make_unique_profile_name(profile.name)
                self.__append_profile(profile)
            profiles_by_uuid = {profile.uuid: profile for profile in self.__profiles}
            self.__active_profile = profiles_by_uuid.get(saved_scan_profiles.active_profile_uuid, self.__profiles[0])
            previous_profile = profiles_by_uuid.get(saved_scan_profiles.previous_profile_uuid)
            self.__previous_profile = previous_profile if previous_profile is not self.__active_profile else None
            # the active profile gives the values of the parameters it includes, and the saved current values give the others.
            if saved_scan_profiles.current_values_properties is not None:
                self.__current_values.read_values_from_dict(saved_scan_profiles.current_values_properties)
            self.__apply_active_profile()
        else:
            for profile_index, profile_name in enumerate(hardware_profile_slot_names):
                frame_parameters = self.__get_frame_parameters_with_subscan(scan_hardware_source.get_frame_parameters(profile_index))
                profile = ProfileModel(profile_name)
                profile.read_from_frame_parameters(frame_parameters)
                self.__append_profile(profile)
            # the hardware's selected profile slot holds the current frame parameters, so its profile is the active one.
            self.__active_profile = self.__profiles[min(max(0, scan_hardware_source.scan_settings.selected_profile_index), len(self.__profiles) - 1)]

        self.__frame_parameters_changed_event_listener: Event.EventListener | None = scan_hardware_source.current_frame_parameters_changed_event.listen(ReferenceCounting.weak_partial(self.__class__.__update_frame_parameters, self))

        # the subscan graphic stays visible and editable while the subscan is off, when its changes reach only the
        # scan device's subscan region and rotation and not the frame parameters.
        self.__scan_hardware_source_property_changed_event_listener: Event.EventListener | None = scan_hardware_source.property_changed_event.listen(ReferenceCounting.weak_partial(self.__class__.__handle_scan_hardware_source_property_changed, self))

    def close(self) -> None:
        """Stop following the scan device and the profiles, so that changes to them are neither applied nor saved."""
        self.__frame_parameters_changed_event_listener = None
        self.__scan_hardware_source_property_changed_event_listener = None
        self.__profile_listeners = list()

    def __get_frame_parameters_with_subscan(self, scan_frame_parameters: scan_base.ScanFrameParameters) -> scan_base.ScanFrameParameters:
        # frame parameters carry the subscan values only while the subscan is on. otherwise the subscan values are
        # the scan device's subscan region and rotation, which turning the subscan on uses.
        if _has_subscan(scan_frame_parameters):
            return scan_frame_parameters
        scan_frame_parameters = copy.copy(scan_frame_parameters)
        subscan_region = self.__scan_hardware_source.subscan_region
        scan_frame_parameters.subscan_fractional_size = subscan_region.size
        scan_frame_parameters.subscan_fractional_center = subscan_region.center
        scan_frame_parameters.subscan_rotation = self.__scan_hardware_source.subscan_rotation
        scan_frame_parameters.line_scan_vector = None
        return scan_frame_parameters

    def __append_profile(self, profile: ProfileModel) -> None:
        self.__profiles.append(profile)
        self.__profile_listeners.append(self.__listen_to_profile(profile))

    def __listen_to_profile(self, profile: ProfileModel) -> typing.Sequence[Event.EventListener]:
        return (
            profile.property_changed_event.listen(ReferenceCounting.weak_partial(ProfileConfigurationModel.__handle_profile_property_changed, self)),
            profile.values_changed_event.listen(ReferenceCounting.weak_partial(ProfileConfigurationModel.__handle_profile_values_changed, self, profile)),
            profile.included_changed_event.listen(ReferenceCounting.weak_partial(ProfileConfigurationModel.__handle_profile_included_changed, self, profile)),
        )

    def __handle_profile_property_changed(self, property_name: str) -> None:
        # the values and the included flags are followed by their own events, which fire once for a change to several of them.
        if property_name == "name":
            self.notify_property_changed("profile_names")
            # a combo box of the names loses its selection when the selected name changes, so restore it.
            self.notify_property_changed("active_profile_index")
            self.__notify_profiles_changed()

    def __handle_profile_values_changed(self, profile: ProfileModel) -> None:
        # a change made while synchronizing is saved once the synchronizing is done.
        if self.__is_synchronizing:
            return
        if profile is self.__active_profile:
            self.__is_synchronizing = True
            try:
                # an edit to the active profile, such as in the configuration dialog, is an edit to the current values.
                self.__apply_active_profile()
            finally:
                self.__is_synchronizing = False
        self.__notify_profiles_changed()

    def __handle_profile_included_changed(self, profile: ProfileModel) -> None:
        # a change made while synchronizing is saved once the synchronizing is done.
        if self.__is_synchronizing:
            return
        if profile is self.__active_profile:
            self.__is_synchronizing = True
            try:
                # a parameter which becomes included keeps the value currently in use.
                profile.copy_values_from(self.__current_values, included_by=profile)
            finally:
                self.__is_synchronizing = False
        self.__notify_profiles_changed()

    def __apply_active_profile(self) -> None:
        # the values of the parameters the active profile includes become the current values. every current value is written to the
        # frame parameters, including the subscan values while the subscan is off, so that the scan device's subscan
        # region and rotation, which turning the subscan on uses, are the current ones.
        self.__current_values.copy_values_from(self.__active_profile, included_by=self.__active_profile)
        frame_parameters = self.__scan_hardware_source.get_current_frame_parameters()
        self.__current_values.write_to_frame_parameters(frame_parameters)
        self.__scan_hardware_source.set_current_frame_parameters(frame_parameters)

    def __notify_profiles_changed(self) -> None:
        # a change to the current values may change no saved properties, such as when it is already saved as a change
        # to the active profile, so only fire when the properties change.
        properties = self.write_to_dict()
        if properties != self.__saved_properties:
            self.__saved_properties = properties
            self.profiles_changed_event.fire(properties)

    def write_to_dict(self) -> dict[str, typing.Any]:
        """Return the properties from which the profiles, the active and previous profiles, and the current values can be restored."""
        return {
            "version": 1,
            "profiles": [profile.to_dict() for profile in self.__profiles],
            "active_profile_uuid": self.__active_profile.uuid,
            "previous_profile_uuid": self.__previous_profile.uuid if self.__previous_profile else None,
            # the current values are not a profile, so they have no identity; their included flags are unused.
            "current_values": self.__current_values.write_values_to_dict(),
        }

    def __update_frame_parameters(self, scan_frame_parameters: scan_base.ScanFrameParameters) -> None:
        self.__current_values.read_from_frame_parameters(self.__get_frame_parameters_with_subscan(scan_frame_parameters))
        # a change made while synchronizing is saved once the synchronizing is done.
        if not self.__is_synchronizing:
            # a change to the current values is a change to the active profile's included values.
            self.__is_synchronizing = True
            try:
                self.__active_profile.copy_values_from(self.__current_values, included_by=self.__active_profile)
            finally:
                self.__is_synchronizing = False
            # the values which the active profile does not include are saved with the profiles, so a change to them is saved too.
            self.__notify_profiles_changed()

    def __handle_scan_hardware_source_property_changed(self, property_name: str) -> None:
        if property_name in ("subscan_region", "subscan_rotation"):
            self.__update_frame_parameters(self.__scan_hardware_source.get_current_frame_parameters())

    @property
    def profiles(self) -> typing.Sequence[ProfileModel]:
        return tuple(self.__profiles)

    @property
    def profile_names(self) -> typing.Sequence[str]:
        return [profile.name for profile in self.__profiles]

    @property
    def active_profile(self) -> ProfileModel:
        return self.__active_profile

    @property
    def active_profile_index(self) -> int:
        return self.__profiles.index(self.__active_profile)

    @active_profile_index.setter
    def active_profile_index(self, value: int | None) -> None:
        """Activate the profile at the index, applying it to the scan and remembering the profile it replaces.

        An index of None, which a combo box reports when it has no selection, leaves the active profile unchanged,
        since there is always an active profile.
        """
        if value is not None and 0 <= value < len(self.__profiles):
            self.__activate_profile(self.__profiles[value])

    def __activate_profile(self, profile: ProfileModel) -> None:
        if profile is not self.__active_profile:
            self.__previous_profile = self.__active_profile
            self.__active_profile = profile
            self.__apply_active_profile()
            self.notify_property_changed("active_profile_index")
            self.notify_property_changed("has_previous_profile")
            self.__notify_profiles_changed()

    def make_frame_parameters(self, profile: ProfileModel) -> scan_base.ScanFrameParameters:
        """Return the current frame parameters with the profile's values written, without activating the profile.

        The subscan values which the scan device derives, such as the subscan pixel size, are not calculated for the
        profile's values. Recording with the frame parameters calculates them.
        """
        frame_parameters = self.__scan_hardware_source.get_current_frame_parameters()
        values = ProfileModel(str())
        values.copy_values_from(self.__current_values)
        values.copy_values_from(profile, included_by=profile)
        values.write_to_frame_parameters(frame_parameters)
        return frame_parameters

    def is_active_profile(self, profile: ProfileModel) -> bool:
        return profile is self.__active_profile

    @property
    def has_previous_profile(self) -> bool:
        return self.__previous_profile is not None

    def activate_previous_profile(self) -> None:
        """Activate the previously active profile, so that repeating this toggles between the last two profiles."""
        if self.__previous_profile is not None:
            self.__activate_profile(self.__previous_profile)

    def make_unique_profile_name(self, base_name: str) -> str:
        """Return the base name, or the base name with the lowest number appended which no other profile uses."""
        names = set(self.profile_names)
        if base_name not in names:
            return base_name
        number = 2
        while f"{base_name} {number}" in names:
            number += 1
        return f"{base_name} {number}"

    def rename_profile(self, profile: ProfileModel, name: str) -> None:
        """Rename the profile, unless the name is blank or the name of another profile, since a profile is chosen by its name."""
        other_names = [other_profile.name for other_profile in self.__profiles if other_profile is not profile]
        if name.strip() and name not in other_names:
            profile.name = name

    def insert_profile(self, index: int, profile: ProfileModel) -> None:
        """Insert the profile at the index, giving it a unique name if its name is already used by another profile."""
        profile.name = self.make_unique_profile_name(profile.name)
        self.__profiles.insert(index, profile)
        self.__profile_listeners.insert(index, self.__listen_to_profile(profile))
        self.notify_insert_item("profiles", profile, index)
        self.notify_property_changed("profiles")
        self.notify_property_changed("profile_names")
        self.notify_property_changed("active_profile_index")
        self.__notify_profiles_changed()

    def insert_new_profile(self, index: int) -> ProfileModel:
        """Insert and return a new profile at the index which includes every parameter at its current value."""
        # the current values include the subscan values while the subscan is off, which the frame parameters do not.
        profile = ProfileModel(_("New Profile"))
        profile.copy_values_from(self.__current_values)
        self.insert_profile(index, profile)
        return profile

    def duplicate_profile(self, index: int) -> ProfileModel:
        """Insert and return a copy of the profile at the index, placing the copy directly after it."""
        source_profile = self.__profiles[index]
        # the copy has its own identity and name, so every value except those is copied.
        profile = ProfileModel(_("{name} Copy").format(name=source_profile.name))
        profile.copy_values_from(source_profile)
        profile.copy_included_from(source_profile)
        self.insert_profile(index + 1, profile)
        return profile

    def move_profile(self, from_index: int, to_index: int) -> None:
        """Move the profile at from_index so that it ends up at to_index, keeping the same profile active."""
        if from_index == to_index or not (0 <= from_index < len(self.__profiles)) or not (0 <= to_index < len(self.__profiles)):
            return
        profile = self.__profiles.pop(from_index)
        profile_listener = self.__profile_listeners.pop(from_index)
        self.notify_remove_item("profiles", profile, from_index)
        self.__profiles.insert(to_index, profile)
        self.__profile_listeners.insert(to_index, profile_listener)
        self.notify_insert_item("profiles", profile, to_index)
        self.notify_property_changed("profiles")
        self.notify_property_changed("profile_names")
        self.notify_property_changed("active_profile_index")
        self.__notify_profiles_changed()

    def can_remove_profile(self, index: int) -> bool:
        """Return whether the profile at the index can be removed; the active profile, and so the last one, cannot."""
        return 0 <= index < len(self.__profiles) and self.__profiles[index] is not self.__active_profile

    def remove_profile(self, index: int) -> ProfileModel:
        """Remove and return the profile at the index.

        Raises ValueError if the profile is the active profile, since there must always be an active profile.
        """
        if self.__profiles[index] is self.__active_profile:
            raise ValueError("The active profile cannot be removed.")
        profile = self.__profiles.pop(index)
        self.__profile_listeners.pop(index)
        if profile is self.__previous_profile:
            self.__previous_profile = None
        self.notify_remove_item("profiles", profile, index)
        self.notify_property_changed("profiles")
        self.notify_property_changed("profile_names")
        self.notify_property_changed("active_profile_index")
        self.notify_property_changed("has_previous_profile")
        self.__notify_profiles_changed()
        return profile
