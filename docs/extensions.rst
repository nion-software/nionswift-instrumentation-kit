.. _creating-components:

Creating Instrumentation Components
===================================
Create custom components based on classes in the instrumentation kit.

*The information on this page is incomplete. Refer to example implementations for more information.*

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

.. TODO: handling actions, state from STEM events (e.g., flip of EELS camera)

STEM Scan Controller
--------------------
The scan control panel and scan control bar provide a UI for the scan device. A standard basic UI is provided and cannot be extended currently.

The standard scan control panel provides play/pause/abort buttons, record/abort buttons, basic mode selection, access to settings dialog, scan settings (time, field of view, pixel size, rotation, etc.), scan status, channel selection and live data previews, and beam control (when not scanning).

The standard scan control bar provides scan/abort/stop buttons, integrated PMT controls, channel enable/disable.

.. TODO: allow primary and secondary scan controllers

Cameras
-------
The camera control panels and camera control bars provide a UI for each camera instance. A standard basic UI is provided but additional custom UI's can be provided in plug-in packages.

The standard camera control panel provides play/pause/abort buttons, basic mode selection, access to settings dialog, access to a monitor view, binning and exposure controls, and a live data preview.

The standard camera control bar provides play/pause button and a checkbox to display the processing channel if there is one.

A custom camera is defined by implementing a camera module and registering it with the registry. A camera module provides a camera device, optional camera settings, and optional camera panel type. If the camera panel type is not defined, the standard camera panel is used.

A camera device defines several methods and properties that define its behavior.

The camera device should define a `camera_category` property. Although not limited to these, `eels` and `ronchigram` are two possible values.

The camera device should also define a `signal_type` property. Although not limited to these, `eels` and `ronchigram` are two possible values.

.. TODO: Document camera modules.

If the camera device has a property `has_processed_channel` with a value of `True`, then the camera control bar displays a checkbox to decide whether it is showing the original raw data or the processed data.

The camera device acquires images and returns data in a data element.

The data element can directly specify calibrations (using the `intensity_calibration` and `spatial_calibrations` keys in the data element), can directly specify how to read the calibrations from the instrument (using the `calibration_controls` key in the data element), or can indirectly specify how to read the calibrations from the instrument (using the `calibration_controls` camera device property).

The `calibration_controls` data element key or camera device propery returns a `dict` describing how to read the calibrations from the instrument controller. If the calibrations are dependent on the camera device state, the `calibration_controls` should be provided as a key in the data element; otherwise the `calibration_controls` can be specified as a property of the camera device.

.. TODO: dark subtraction, gain normalization, blemish removal
.. TODO: dark reference collection, gain image collection, blemish configuration
.. TODO: readout area, other camera configuration
.. TODO: settings storage, settings dialog
.. TODO: monitor configuration, button
.. TODO: processing channel configuration, settings including readout area
.. TODO: synchronized acquisition
.. TODO: sequence acquisition
.. TODO: handling integrated scan/camera systems
.. TODO: frame summing, averaging (rolling), recording

Video Camera
------------
The video control panel and video control bars provide a UI for each video camera instance. A standard basic UI is provided and can be extended by plug-in packages.

The video control panel provides a play/stop button and live data preview for each video source.

The video control bar provides a play/stop button.

The video control preference panel provides the ability to add/remove video sources and edit their settings. The settings are defined by plug-in packages and can be have a customized UI.

..
    Data Elements
    -------------
    The data elements are a list of data elements ``dict`` describing the data. The data elements can contain the
    following keys.

    =============================== =============== ===============================================================
    Key                             Default         Description
    =============================== =============== ===============================================================
    version                         required        data element version, must be the integer 1
    data                            required        the data, a numpy array
    timestamp                       current time    the timestamp of the data, datetime object
    is_sequence                     False           whether the data represents a sequence
    collection_dimension_count      0               an integer describing the collection dimension count
    datum_dimension_count           data shape      an integer describing the datum dimension count
    properties                      none            a dict of properties, see below
    properties.frame_number
    properties.frame_index
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
    =============================== =============== ===============================================================

    Data is stored with the fastest varying index last (numpy default).

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
    -----------
    Providing calibrations directly in the data element.

    Providing a set of controls from which to read the calibrations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
