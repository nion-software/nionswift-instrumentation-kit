.. _scan_control:

Scanning
========
The scan device is tightly integrated into the STEM microscope. So some functions such as basic acquisition can be
handled via the device, while other functions such as synchronized acquisition and probe control are handled through
the :ref:`stem-instrument`.

.. contents::

General Info
++++++++++++

How does scanning work?
-----------------------
On a STEM instrument, scanning works by moving the probe electronically over the sample. The scan device allows you to
configure a scan with a physical size and position on the sample. The scan can then be configured with a resolution, and
timing information. In addition, you can control characteristics of the scan such as whether to synchronize to an
external source such as a camera frame.

By choosing appropriate positioning and resolution, you can construct more specific scans, such as a line scan, or a
point. By combining a sequence of scans, you can construct more complex scans such as a polygon or scattered individual
points.

TODO: explain flyback time

How do the scan coordinate systems and rotations work?
------------------------------------------------------
The scan device is typically centered on the axis of the microscope and the magnification and position are specified
using field of view (FoV) combined with rotation.

Once the magnification and position are specified, the scan device is configured so that its full range of scanning
extends over the entire _unrotated_ field of view and then inset so that all rotations will be within the available
scanning range.

Subscans can also be specified within a full scan context. Subscans take the rotation of the enclosing context scan.

What parameters can be controlled on the scan?
----------------------------------------------
An individual scan is configured with frame parameters, which is just a Python dict structure.

The following parameters are supported:

============================    =========   ===========
Name                            Immediate   Description
============================    =========   ===========
fov_size_nm                     yes         the physical field of view, in nm
rotation_rad                    yes         the rotation, in radians
pixel_time_us                   yes         the time to spend at each pixel, in microseconds
flyback_time_us                 yes         the flyback time, in microseconds
size                            no          the size, in pixels, of the scan
external_clock_wait_time_ms     no          the maximum time to wait for external trigger
external_clock_mode             no          the external trigger mode
ac_line_sync                    no          whether to sync each line to power line frequency
ac_frame_sync                   no          whether to sync each frame to power line frequency
============================    =========   ===========

The scan device tracks its current frame parameters during acquisition.

Some parameters (fov_size_nm, rotation_rad, pixel_time_us, and flyback_time_us) take effect immediately, while others
are marked as pending and take effect on the next frame.

More complex scans can also be comprised of more than one individual scan. If parameters are changed during a complex
scan, it applies to the next individual scan. If the parameters to the complex scan are changed during a complex scan,
the complex scan is restarted.

How do scanning applications interact?
--------------------------------------
There is currently no mechanism whereby an application using the scan device can exclude actions from another
application. Access is managed by the user not running conflicting applications simultaneously.

How does scanning interact with Python threads?
-----------------------------------------------
In Nion Swift, the user interface is run on the main UI thread. It is important that function calls made from the main
UI thread return within a short period of time (< 50ms). This ensures that the UI is responsive.

During acquisition, function calls to the scan device can easily take more than 50ms. For this reason, most of the
examples on this page should be run from a thread other than the main UI thread.

In addition, some function calls may require the UI thread to be running separately in order to complete. These
functions must be called from a thread and are noted separately.

Python code run in the Console are run on the main UI thread. It is useful for experimentation and most examples on this
page will run in the Console, although if something goes wrong, there is no way to recover other than restarting Nion
Swift.

Python code run using Run Script is run on a separate thread and the examples on this page can all be run using that
mechanism unless otherwise noteed.

Python code run in plug-ins will need to create its own threads and run these examples from those threads.

It is also possible to define a function in the Console and then launch that function using threading. Here is a short
example::

    import threading

    def fn():
        print("Put code to run on thread here.")

    threading.Thread(target=fn).start()

Basic Scanning
++++++++++++++

How do I access the STEM Controller and scan device?
----------------------------------------------------
You can access the STEM controller and scan device using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller

How do I configure the scan and start view mode?
------------------------------------------------
You can configure an individual scan and start viewing using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller
    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired
    scan.start_playing(frame_parameters)

As the scan starts, output data will be associated with data items in Nion Swift which will be updated in near real
time.

How do I configure the scan and acquire one channel?
----------------------------------------------------
You can configure an individual scan, start viewing, and grab data from the acquisition using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller
    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired
    scan.start_playing(frame_parameters, channels=[0])
    # grab two consecutive frames, with a guaranteed start time after the first call
    frame1 = scan.grab_next_to_start()[0]
    frame2 = scan.grab_next_to_finish()[0]

The `grab_next_to_start` call waits until the next frame starts and then grabs it. The `grab_next_to_finish` call waits
until the current frame ends and then grabs it. Both calls return a list of `xdata` objects with an entry for each
enabled channel. In this case the first element is selected since only a single channel is enabled.

How do I configure the scan and acquire multiple channels?
----------------------------------------------------------
You can configure an individual scan with multiple channels, start viewing, and grab data from the acquisition using the
following code::

    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller
    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired
    scan.start_playing(frame_parameters, channels=[1, 2])
    # grab two consecutive frames, with a guaranteed start time after the first call
    frames1 = scan.grab_next_to_start()
    frames2 = scan.grab_next_to_finish()
    frame1c1, frame1c2 = frames1
    frame2c1, frame2c2 = frames2

The `grab_next_to_start` and `grab_next_to_finish` calls return a list of `xdata` objects with an entry for each enabled
channel. These values are unpacked in the last two lines.

How do I monitor progress (partial scans) during a scan?
--------------------------------------------------------
You can monitor progress during an individual scan. ::

    import time
    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller
    frame_parameters = scan.get_current_frame_parameters()
    frame_time = scan.calculate_frame_time(frame_parameters)
    # adjust frame_parameters here if desired
    scan.start_playing(frame_parameters)
    # monitor progress
    frame_id = scan.get_current_frame_id()
    for i in range(10):
        time.sleep(frame_time / 10)
        print(scan.get_frame_progress(frame_id))

How do I stop or cancel an individual scan?
-------------------------------------------
There are two ways to cancel a scan: stop and abort. Stop waits until the end of the current frame, while abort stops as
soon as possible. Aborting a scan may result in partially acquired data. You can abort a scan that has already been
stopped. ::

    import time
    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller
    frame_parameters = scan.get_current_frame_parameters()
    frame_time = scan.calculate_frame_time(frame_parameters)
    # adjust frame_parameters here if desired
    scan.start_playing(frame_parameters)
    time.sleep(1.0)
    scan.stop_playing(frame_parameters)
    scan.abort_playing(frame_parameters)

How do I configure the scan for acquire a subscan of an existing scan?
-----------------------------------------------------------------------
A subscan can be specified within the context of an individual scan by specifying additional parameters. ::

    import time
    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller
    frame_parameters = scan.get_current_frame_parameters()
    frame_parameters["subscan_pixel_size"] = (100, 100)
    frame_parameters["subscan_fractional_size"] = (0.4, 0.3)
    frame_parameters["subscan_fractional_center"] = (0.5, 0.5)
    # adjust frame_parameters further here if desired
    scan.start_playing(frame_parameters)

============================    =========   ===========
Name                            Immediate   Description
============================    =========   ===========
subscan_pixel_size              yes         the subscan desired size tuple (h, w), in pixels
subscan_fractional_size         yes         the subscan fractional size, relative to field of view
subscan_fractional_center       yes         the subscan fractional center, relative to field of view
============================    =========   ===========

The fractional size and center are relative to the field of view and have the same rotation. The (0, 0) tuple is at the
top left and the (1, 1) tuple is at the bottom right. Coordinates are specified in y-axis, x-axis order.

Changing the rotation will rotate the scan around the microscope axis and the subscan will generally be off axis; so a
rotation will effectively shift a subscan in addition to rotating it.

How do I configure a rectangular scan synchronized with a camera?
-----------------------------------------------------------------
A combined scan produces data from the scan and data from the camera.

..
    from nion.utils import Registry
    stem_controller = Registry.get_compoonent("stem_controller")
    scan = stem_controller.scan_controller
    scan_frame_parameters = scan.get_current_frame_parameters()
    # adjust scan_frame_parameters here if desired
    frame_id = stem_controller.start_combined_record(scan, scan_frame_parameters, camera, camera_frame_parameters)
    combined_data = scan.grab_combined_data(frame_id)
    frames, camera_data_list = combined_data
    frame = frames[0]
    camera_data = camera_data[0]

.. the API needs to handle multiple cameras (eventually)
.. the API needs to handle error conditions or abort
.. the API will generally connect acquisition to channels, which the user can view and cancel

How do I configure a line scan synchronized with a camera?
----------------------------------------------------------

How do I configure complex multi-region scans synchronized with a camera?
-------------------------------------------------------------------------

How do I do multiple acquisitions at each point in a synchronized scan?
-----------------------------------------------------------------------

How do I perform an action between regions in a multi-region synchronized scan?
-------------------------------------------------------------------------------

How do I acquire a sequence of rectangular scans?
-------------------------------------------------

How do I grab recently captured data?
-------------------------------------

How do I find data items associated with view mode?
---------------------------------------------------

How do I determine scan parameters from acquired data's metadata?
-----------------------------------------------------------------

How do I control the probe manually?
------------------------------------

How do I control the state of the probe (scanning, positioned, not-positioned)?
-------------------------------------------------------------------------------

How do I control blanking?
--------------------------
