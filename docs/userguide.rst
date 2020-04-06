.. _user-guide:

User's Guide
============

.. contents::

.. _camera-panel:

Using the Camera Panel and Control Bar
--------------------------------------

Readout area is always specified in un-binned coordinates. Binning is applied to the data read from the readout area.

.. _scan-panel:

Using the Scan Panel and Control Bar
------------------------------------

.. _video-panel:

Using Video Panels and Control Bars
-----------------------------------

.. _synchronized-acquisition-panel:

Synchronized Acquisition (aka Spectrum Imaging / 4D Acquisition)
----------------------------------------------------------------
Synchronized acquisition is camera acquisition synchronized with scanning. Specific examples include spectrum imaging and 4D STEM.

Your system must be configured properly for synchronized acquisition. This may include physical triggering connections between cameras and the scan device. It may also include software connections to route the trigger connections.

To use synchronized acquisition, open the `Spectrum Imaging / 4D Scan Acquisition` panel using the `Window` menu.

Next, establish a context image by performing a scan with one or more channels enabled. Confirm the scan data is shown in a display panel and click on that display panel to ensure it has keyboard focus.

Next, select the type of data you wish to acquire, spectra or images, using the menu at the top right of the `Spectrum Imaging / 4D Scan Acquisition` panel.

You can perform synchronized acquisition on the full scan, a sub-area rectangle (possibly rotated), or on a line.

.. image:: resources/synchronized_acquisition_panel.png
   :scale: 50 %

Full Context Synchronized Acquisition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform a full scan, click on the context image and make sure no graphics are selected. Enter the desired scan width in the `Spectrum Imaging / 4D Scan Acquisition` panel, check the acquisition estimated time and size, and click `Acquire`. The context image will update during the acquisition process.

Sub Area (Rectangle) Synchronized Acquisition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform a rectangle scan, add a rectangle graphic to the context image (a quick way to do this is click on the `Subscan Enabled` in the Scan Control panel). Adjust the rectangle by dragging and adjust its rotation using the inspector. Now enter the desired scan width in the `Spectrum Imaging / 4D Scan Acquisition` panel, check the acquisition estimated time and size, and click `Acquire`. The *subscan* image will update during the acquisition process. If the *subscan* image is not visible, make it visible by right clicking in an empty display panel and choosing the desired channel with "Subscan" in its title.

Line Scan Synchronized Acquisition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To perform a line scan, add a line graphic to the context image. After adjusting its position, enter the desired scan width in the `Spectrum Imaging / 4D Scan Acquisition` panel, check the acquisition estimated time and size, and click `Acquire`. The *subscan* image will update during the acquisition process. If the *subscan* image is not visible, make it visible by right clicking in an empty display panel and choosing the desired channel with "Subscan" in its title.

The resulting data will be a collection (one dimension) of your selected data type (spectra or image). You can display the data in its default mode (a spectra or an image) and scroll through the collection using the Ctrl-Right-Arrow and Ctrl-Left-Arrow keys. Or you can use the menu item `Processing > Redimension Data > Redimension to a Collection of 1 Image of Shape 400x2048` or similar to display an image with a collection index on one axis and the data dimension (energy) on the other axis.

.. _multi-acquire-panel:

Using the Multi-Acquire Panel
-----------------------------
A Nion Swift plug-in that acquires and displays multiple EEL spectra with different energy offsets.

Main window
+++++++++++
Get the plug-in main window by selecting it from the "Window" menu.

.. image:: resources/multi_acquire_main_window.png

This plugin allows you to acquire series of EEL spectra with multiple energy offsets and exposure settings. The acquisitions can
be set up with the table in the main window. Every spectrum corresponds to one line. Use the "+" and "-" buttons to add or remove
lines. The first column shows the spectrum number, which will also be added to the titles of the result data items.
You have the option to acquire the programmed sequence of spectra as individual spectra or as spectrum images by clicking
either on "Start Multi-Acquire" or "Start Multi-Acquire spectrum image".
Note that for individual spectra, the progress bar will only update once per spectrum (i.e. if only one spectrum is defined it will jump
straight from 0 to 100%). For spectrum images, the progress bar will update once per acquired line of a spectrum image.
The times shown above the two "start" buttons are the estimated acquisition times for the two modes.


Settings Window
+++++++++++++++
You can access the settings menu via the "Settings..." button in the top-right corner of the main window.

.. image:: resources/multi_acquire_settings_window.png

In order to set the energy offsets, the plugin needs to know which control it has to change in AS2. Type the name of
this control into the "X-shift control name" field. If the field is empty, x-shifts are disabled, regardless of what
is configured in the main window.
The checkboxes in the bottom row allow you to configure how the data will be returned:

* "Bin data in y-direction" will sum the images in vertical direction to obtain spectra.
* "Auto dark subtraction" will blank the beam after the acquisition is finished and repeat it (with the exact same settings). This data will be then be used as dark images for the actual data.
  Make sure "Blanker control name" is set correctly, otherwise this mode will fail. Note that this settings has no effect for spectrum images as it will always be deactivated in this mode.
* "Sum frames" will sum all frames that were acquired for each spectrum (as specified by the column "frames" in the main window). If this is off, the plug-in will return a stack for each spectrum.

Output
++++++
The plug-in will create one result data item per spectrum. These data items can be either single spectra, single images,
stacks of spectra or stacks of images, depending on the settings. When you are acquring spectrum images, the result data items
will be either single spectrum images, single 4D images, stacks of spectrum images or stacks of 4d images.
Additionaly the plug-in will create a data item that contains all acquired spectra as multiple line plots.
This last data item will only be created if "Bin data in y-direction"
is selected in the settings window and if you are not acquiring spectrum images.

.. image:: resources/multi_acquire_output_stacked.png

.. _multiple-shift-acquire-panel:

Using the Multiple Shift Acquire Panel
--------------------------------------

.. _acquisition-recorder-panel:

Using the Acquisition Recorder Panel
------------------------------------
