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

The standard scan control panel provides play/pause/abort buttons, record/abort buttons, basic mode selection, access to settings dialog, scan settings (time, field of view, pixel size, rotation, etc.), scan status, channel selection and live data previews, sub-scan control, and beam control (when not scanning).

The standard scan control bar provides scan/abort/stop buttons, integrated PMT controls, channel enable/disable.

.. TODO: allow primary and secondary scan controllers

Cameras
-------
The camera control panels and camera control bars provide a UI for each camera instance. A standard basic UI is provided and can be slightly customized but additional custom UI's can be provided in plug-in packages.

The standard camera control panel provides play/pause/abort buttons, basic mode selection, access to configuration dialog, access to a monitor view, binning and exposure controls, and a live data preview.

The standard camera control bar provides play/pause button and a checkbox to display the processing channel if there is one.

A custom camera is defined by implementing a camera module and registering it with the registry. A camera module provides a camera device, optional camera settings, an optional camera panel delegate, or an optional camera panel. If the camera panel is not defined, the standard camera panel is used. The standard camera panel can utilize a delegate to provide configuration and monitor dialogs.

A camera device uses several methods and properties to define its behavior. As part of this behavior, the camera device acquires images and returns data in a data element. Some properties can be specified by the camera object, while others can be or need to be specified in the data element returned from the acquisition methods. When both options are available, the data element version takes precedence.

The camera device should define a `camera_category` property. Although not limited to these, `eels` and `ronchigram` are two possible values.

The camera device may also define a `signal_type` property. Although not limited to these, `eels` and `ronchigram` are two possible values. The `signal_type` can also be returned as an entry in the data element dictionary.

If the camera device may also define a `has_processed_channel` property. If this property is true or if the camera type is 'eels' and this property is not defined, then the camera control bar displays a checkbox to decide whether it is showing the original raw data or the processed data. If the camera type is 'eels' and this checkbox is not desired, then set `has_processed_channel` to false.

The data element can directly specify calibrations (using the `intensity_calibration` and `spatial_calibrations` keys in the data element), or it can directly specify how to read the calibrations from the instrument (using the `calibration_controls` key in the data element), or it can indirectly specify how to read the calibrations from the instrument (using the `calibration_controls` camera device property).

The `calibration_controls` data element key or camera device propery returns a `dict` describing how to read the calibrations from the instrument controller. If the calibrations are dependent on the camera device state, the `calibration_controls` should be provided as a key in the data element; otherwise the `calibration_controls` can be specified as a property of the camera device.

Providing a Camera Panel Delegate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The standard camera control panel provides buttons to open a configuration dialog and a monitor dialog. The camera module can define a delegate to handle these buttons for a camera.

1. Define ``camera_panel_delegate_type`` in the camera module by putting a line similar to this in the ``__init__`` function::

    self.camera_panel_delegate_type = "my_camera_panel_delegate"

2. Register a delegate factory in the registry with a type of ``camera_panel_delegate``. This can be done by calling a function from the ``__init__.py`` file of your package::

    camera_panel_delegate = CameraPanelDelegate()
    Registry.register_component(camera_panel_delegate, {"camera_panel_delegate"})

3. Define a class factory, derived from ``CameraControlPanel.CameraPanelDelegate``, to handle camera panel requests. The class factory must define ``camera_panel_delegate_type`` to match the factory type in step 1::

    class CameraPanelDelegate(CameraControlPanel.CameraPanelDelegate):
        camera_panel_delegate_type = "my_camera_panel_delegate"
        def get_configuration_ui_handler(self, *, api_broker: PlugInManager.APIBroker = None,
                                         event_loop: asyncio.AbstractEventLoop = None,
                                         hardware_source_id: str = None,
                                         camera_device: camera_base.CameraDevice = None,
                                         camera_settings: camera_base.CameraSettings = None,
                                         **kwargs):
            dclui = api_broker.get_ui("~1.0")
            class Handler:
                ui_view = dclui.create_row(dclui.create_label(text="LABEL2"), dclui.create_push_button(text="Push", on_clicked="cancel_clicked"))
                def cancel_clicked(self, widget):
                    print("CLICK")
            return Handler()

Providing a Custom Camera Panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The camera module can define a custom panel to replace the standard camera panel.

1. Define ``camera_panel_type`` in the camera module by putting a line similar to this in the ``__init__`` function::

        self.camera_panel_type = "my_camera_panel"

2. Register a delegate factory in the registry with a type of ``camera_panel``. This can be done by calling a function from the ``__init__.py`` file of your package::

    camera_panel_factory = CameraPanelFactory()
    Registry.register_component(camera_panel_factory, {"camera_panel"})

3. Define a camera panel factory to create the delegate. The class factory must define ``camera_panel_type`` to match the factory type in step 1::

    class CameraPanelFactory:
        camera_panel_type = "my_camera_panel"
        def get_ui_handler(self, api_broker=None,
                           event_loop=None, hardware_source_id=None,
                           camera_device=None, camera_settings=None,
                           **kwargs):
            dclui = api_broker.get_ui("~1.0")
            class Handler:
                ui_view = dclui.create_row(dclui.create_label(text="Camera X"), dclui.create_push_button(text="Push", on_clicked="button_clicked"))
                def button_clicked(self, widget):
                    print("CLICK")
            return Handler()

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

Data Elements
-------------
Methods that return data or lists of data do so by using a data element `dict`.

The following keys are used in the data element.

=============================== =============== ===============================================================
Key                             Default         Description
=============================== =============== ===============================================================
version                         1               data element version, must be the integer 1
data                            *required*      the data, a numpy array
timestamp                       current time    the timestamp of the data, datetime object
is_sequence                     False           whether the data represents a sequence
collection_dimension_count      0               an integer describing the collection dimension count
datum_dimension_count           data shape      an integer describing the datum dimension count
properties                      none            a dict of properties, see below
properties.frame_number
properties.frame_index
properties.integration_count
counts_per_electron             none            *deprecated* use calibration_controls instead
intensity_calibration           none            *deprecated* use calibration_controls instead
spatial_calibrations            none            *deprecated* use calibration_controls instead
calibration_controls            none            see description below
reader_version
large_format
metadata
datetime_modified
datetime_original
description.timezone (tz, dst)
=============================== =============== ===============================================================

Data is stored with the fastest varying index last (numpy default).

Data elements that have height=1 are expected to be returned as 1d data.

frame_number comes from camera; frame_index comes from Swift.

Integration count is optional; passed in settings, but should return how many were actually integrated.

Exposure, binning, and signal type will be automatically determined from settings.

hardware_source_name, hardware_source_id, state, and valid_rows will be set after acquisition.

.. TODO: Explain how counts_per_electron and spatial/intensity calibrations handle binning.

..
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

Calibration Controls
--------------------
To faciliate integration with the instrument controller, calibrations can be passed directly or read from the instrument controller.

The dict should have keys of the form `<axis>_<field>_<type>` where `<axis>` is `x`, `y`, `z`, or `intensity`, `<field>` is `scale`, `offset`, `units`, or `origin_override`, and `<type>` is `control` or `value`. If `<type>` is `control`, then the value for that axis/field will use the value of that key to read the calibration field value from the instrument controller. If `<type>` is `value`, then the calibration field value will be the value of that key.

For example, the dict with the following keys will read `x_scale` and `x_offset` from the instrument controller values `cam_scale` and `cam_offset`, but supply the units directly as "nm". ::

    { "x_scale_control": "cam_scale", "x_offset_control": "cam_offset", "x_units_value": "nm" }

The dict can contain the key `<axis>_origin_override` with the value `center` to indicate that the origin is at the center of the data for that axis. ::

    { "x_scale_control": "cam_scale",
      "x_offset_control": "cam_offset",
      "x_units_value": "nm",
      "x_origin_override": "center" }

In addition to the calibration controls, a `counts_per_electron` control or value can also be specified. ::

    { "counts_per_electron_control": "Camera1_CountsPerElectron" }


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
