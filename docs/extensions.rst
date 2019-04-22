Instrumentation Developers Kit
==============================

Overview
--------
The instrumentation developers kit provides user interfaces and abstractions for instrument controllers and devices such as scan controllers, analytic cameras, detectors, and video cameras.

The main instrument controller defines the instrument's major components and function such that custom components can be integrated into the system.

User interface components provide an interface to the instrument controller and its components, working together in a way consistent with operation of the instrument. User interface components can usually be replaced or extended as required.

The only specific instrument controller provided is a STEM microscope instrument controller which specifies a scan controller, analytic cameras and detectors, and video cameras.

Additional acquisition tools are provided that build on the instrument controllers.

Some tools such as cumulative acquisition can be used with nearly all cameras. Others such as dark subtraction reference collection provide a partial implementation that the specific instrument controller must extend. Still others such as tuning are specific to a particular instrument.

STEM Instrument Controller
--------------------------
The STEM controller tracks the instrument state, scanning, and probe state, and provides methods to control the state.

It defines several major components, including a scan controller, a Ronchigram camera, EELS and other detectors, and video cameras.

The scan controller controls the scan state, including the scan area, scan rotation, scan timing, and scan resolution. In addition, it facilitates synchronized scanning in conjunction with another detector such as a camera.

Cameras and detectors provide a user interface and a means of acquiring data into the application.

Acquisition devices provide the ability to acquire data and interact with applications such as a live data viewer or
spectrum imaging via settings describing the data acquisition.

Acquisition Settings
--------------------
The acquisition settings are layered so that various parts of the system can contribute to the settings.

- Default settings - fallback values if not specified elsewhere
- Configuration settings - configuration values, not frequently changed
- Application settings - values specific to the application using the device
- User settings - values configurable by the user
- Automatic settings - values automatically updated depending on the STEM controller

.. TODO: are there any missing layers or capabilities not provided by these layers?

The UI will typically allow the user to save/load user settings and possibly configuration settings.

The settings for a particular acquisition will be constructed starting with default settings and applying any more
specific settings from each layer. Acquisition settings may change (frame by frame) explicitly from the user or
automatically from changes to the state of the STEM controller.

Example of application settings: external trigger mode (spectrum imaging).

Example of user setting: exposure time, binning, dark and gain mode.

Example of automatic setting: calibration, readout area (EELS), horizontal flip (EELS).

Acquisition UI
--------------
The acquisition panel allows the user to display and edit current settings, start/stop acquisition, save/load settings,
open a configuration dialog, and preview the acquisition data or image.

The configuration dialog allows the user to edit configuration settings.

.. TODO: does the configuration dialog ever affect current acquisition exposure/binning?

Acquisition begins with a copy of settings constructed from each of the settings layers. Changes to any of the layers
will result in update settings being passed to the low level device acquisition.

The acquisition panel will reflect the current settings as closely as possible.

The current settings are persistent between launches.

The acquisition panel provides options to load/save settings. Loading previously saved settings will affect the current
acquisition.

.. TODO: which settings are allowed to be changed during acquisition? (currently: anything)
.. TODO: how to prevent user from switching settings while an application is running?
.. TODO: how to prevent multiple applications from running conflicting settings?

Acquisition Settings Model
--------------------------
The acquisition settings model provides storage for acquisition settings, configuration, and other persistent settings.
It can be used to provide a full UI and storage in Python, but it can also provide a common interface when the UI and
storage are implemented at a lower level.

Acquisition devices can have the following optional customizations. If a particular device doesn't customize one or
more of the items below, a default will be used according to the device type (camera, scan).

- Custom acquisition panel
- Custom configuration dialog
- Custom storage

Camera
------
The system provides capabilities for acquisition, dark subtraction, gain normalization, blemish removal,
configuration storage, dark and gain image storage, settings storage, settings model, a standard acquisition panel,
ability to use custom acquisition panel, and ability to use custom configuration panel.

Acquisition styles include viewing, sequence acquisition (SI, ptychography), frame summing, frame averaging (rolling),
single frame, and recording.

Provides a mechanism to link configuration or settings to other objects such as UI or the stem controller. This
allows camera configuration, saved settings, and current settings to be edited and viewed.

Camera Device
^^^^^^^^^^^^^

Data Elements
^^^^^^^^^^^^^
The data elements are a list of data elements ``dict`` describing the data. The data elements can contain the
following keys.

=============================== =============== ===============================================================
Key                             Default         Description
=============================== =============== ===============================================================
version                         none            should be set to 1
data                            required        an ndarray
timestamp                                       default now
is_sequence
collection_dimension_count      0               an integer describing the collection dimension count
datum_dimension_count           data shape      an integer describing the datum dimension count
properties                      none            a dict of properties, see below
properties.frame_number
properties.integration_count
properties.counts_per_electron
intensity_calibration           none            a calibration dict
spatial_calibrations            none            a list of calibration dicts
reader_version
large_format
metadata
datetime_modified
datetime_original
description.timezone (tz, dst)
properties.integration_count
properties.frame_number
=============================== =============== ===============================================================

Data is stored with the fastest varying index last.

Data elements that have height=1 are expected to be returned as 1d data.

Integration count is optional; passed in settings, but should return how many were actually integrated.

Exposure, binning, and signal type will be automatically determined from settings.

frame_number comes from camera; frame_index comes from Swift.

hardware_source_name, hardware_source_id, state, and valid_rows will be set after acquisition.

.. TODO: Explain how counts_per_electron and spatial/intensity calibrations handle binning.

swift processes the data using the following data_element keys:
channel_id
state
sub_area
ImportExportManager.convert_data_element_to_data_and_metadata:
* data
* is_sequence, collection_dimension_count, datum_dimension_count
* spatial_calibrations
* intensity_calibration
* metadata
* properties (get stored into metdata.hardware_source)
* timestamp or datetime_modified
* if datetime_modified (dst, tz) (converted to timestamp; then timezone gets stored into metadata.description.timezone)

Calibration
^^^^^^^^^^^
Providing calibrations directly in the data element.

Providing a set of controls from which to read the calibrations.

Scan
----
Acquire list of regions (rectangle, line, point) with action between each region. The action could be defined
declaratively (i.e. change AS2 setting, etc.). Or we could go full-out and make callbacks to Python.

Control the probe position.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
