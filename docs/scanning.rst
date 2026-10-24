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

Scans typically fit into two categories: continuous or non-continuous. Continuous scans repeat when they reach the end
of a frame. Non-continuous scans stop and wait for further function calls or UI initiation to continue.

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
An individual scan is configured with frame parameters, a ``ScanFrameParameters`` object. A parameter with an
attribute, such as ``fov_nm``, is read and set through the attribute. Other parameters, such as those particular to a
scan device, are read with ``get_parameter`` and set with ``set_parameter``.

The following parameters are supported:

============================    =========   ===========
Name                            Immediate   Description
============================    =========   ===========
fov_nm                          yes         the physical field of view, in nm
rotation_rad                    yes         the rotation, in radians
pixel_time_us                   yes         the time to spend at each pixel, in microseconds
flyback_time_us                 yes         the flyback time, in microseconds
pixel_size                      no          the size, in pixels, of the scan
external_clock_wait_time_ms     no          the maximum time to wait for external trigger
external_clock_mode             no          the external trigger mode
ac_line_sync                    no          whether to sync each line to power line frequency
ac_frame_sync                   no          whether to sync each frame to power line frequency
============================    =========   ===========

The scan controller tracks its current frame parameters during acquisition. ``get_current_frame_parameters`` returns a
copy of them, so changing the copy changes nothing until it is passed to the scan.

Some parameters (fov_nm, rotation_rad, pixel_time_us, and flyback_time_us) take effect immediately, while others
are marked as pending and take effect on the next frame.

More complex scans can also be comprised of more than one individual scan. If parameters are changed during a complex
scan, it applies to the next individual scan. If the parameters to the complex scan are changed during a complex scan,
the complex scan is restarted.

How does scanning interact with Nion Swift and plug-ins?
--------------------------------------------------------
Acquisition threads send their data into Nion Swift via a data channel. Unless otherwise configured, a data channel will
feed into a reusable data item which can be displayed in the user interface.

There is currently no mechanism whereby a plug-in or script using the scan device can exclude actions from other
plug-ins or scripts. Access is managed by convention and the user not running conflicting applications simultaneously.

How does scanning interact with Python threads?
-----------------------------------------------
In Nion Swift, the user interface is run on the main UI thread. It is important that function calls made from the main
UI thread return within a short period of time (< 50ms). This ensures that the UI is responsive.

During acquisition, function calls to the scan device can easily take more than 50ms. For this reason, most of the
examples on this page should be run from a thread other than the main UI thread.

In addition, some function calls may _require_ the UI thread to be running separately in order to complete. These
functions must be called from a thread and are noted separately.

Python code run in the Console is run on the main UI thread. It is useful for experimentation and most examples on this
page will run in the Console, although if something goes wrong, there is no way to recover other than restarting Nion
Swift.

Python code run using Run Script is run on a separate thread and the examples on this page can all be run using that
mechanism unless otherwise noted.

Python code run in plug-ins will need to create its own threads and run these examples from those threads.

It is also possible to define a function in the Console and then launch that function using threading. Here is a short
example::

    import threading

    def fn():
        print("Put code to run on thread here.")

    threading.Thread(target=fn).start()

Using the API
+++++++++++++

How do I access the STEM Controller and scan device?
----------------------------------------------------
You can access the STEM controller and scan device using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

How do I configure the scan and start view mode?
------------------------------------------------
You can configure an individual scan and start viewing using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    scan.start_playing(frame_parameters=frame_parameters)

As the scan starts, output data will be associated with data items in Nion Swift which will be updated in near real
time.

How do I configure the scan and acquire one channel?
----------------------------------------------------
You can configure an individual scan, start viewing, and grab data from the acquisition using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    scan.set_enabled_channels([0])
    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    scan.start_playing(frame_parameters=frame_parameters)

    # grab two consecutive frames, with a guaranteed start time after the first call
    frame1 = scan.grab_next_to_start()[0]
    frame2 = scan.grab_next_to_finish()[0]

The ``grab_next_to_start`` call waits until the next frame starts and then grabs it. The ``grab_next_to_finish`` call
waits until the current frame ends and then grabs it. Both calls return a list of ``xdata`` objects with an entry for
each enabled channel. In this case the first element is selected since only a single channel is enabled.

How do I configure the scan and acquire multiple channels?
----------------------------------------------------------
You can configure an individual scan with multiple channels, start viewing, and grab data from the acquisition using the
following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    scan.set_enabled_channels([1, 2])
    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    scan.start_playing(frame_parameters=frame_parameters)

    # grab two consecutive frames, with a guaranteed start time after the first call
    frames1 = scan.grab_next_to_start()
    frames2 = scan.grab_next_to_finish()
    frame1c1, frame1c2 = frames1
    frame2c1, frame2c2 = frames2

The ``grab_next_to_start`` and ``grab_next_to_finish`` calls return a list of ``xdata`` objects with an entry for each
enabled channel. These values are unpacked in the last two lines.

How do I record one frame with its own frame parameters?
--------------------------------------------------------
You can record one frame with frame parameters which differ from the current ones, without changing the current ones,
using the following code::

    from nion.utils import Geometry
    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    frame_parameters = scan.get_current_frame_parameters()
    frame_parameters.pixel_size = Geometry.IntSize(h=100, w=100)
    frame_parameters.rotation_rad = 0.0
    frame_parameters.enabled_channel_indexes = [0]
    # adjust frame_parameters further here if desired

    scan.abort_playing(sync_timeout=5.0)
    frame = scan.record_immediate(frame_parameters)[0]

The ``record_immediate`` call returns once the frame is acquired, with an ``xdata`` object for each recorded channel. The
recording uses the frame parameters as given, and the scan's current frame parameters are unchanged afterwards, except
that they record which channels are enabled, as the scan control panel's channel check boxes do. The channels in
``enabled_channel_indexes`` are enabled for the recording, and the channels enabled before are enabled again when the
recording delivers its data. A recording which fails returns an empty list and leaves the channels as it set them.

A running view is suspended during the recording and resumes afterwards, so the example stops the view first. With no
view running, the scan device stays on the recording's frame parameters afterwards, so a probe position set next refers
to the recorded frame.

How do I determine if the scan is running?
------------------------------------------
You can make a rough determination if a scan is running using the following::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    is_scanning = scan.is_playing

You shouldn't use this technique to synchronize acquisition as it does not handle threads and race conditions in a
predictable manner. For instance, it may not be accurate if called immediately following a call that initiates
acquisition; likewise it may not be accurate if called immediately before acquisition ends.

..
    How do I monitor progress (partial scans) during a scan?
    --------------------------------------------------------
    You can monitor progress during an individual scan. ::

        import time
        from nion.utils import Registry

        stem_controller = Registry.get_component("stem_controller")

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
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    frame_parameters = scan.get_current_frame_parameters()
    frame_time = scan.calculate_frame_time(frame_parameters)
    # adjust frame_parameters here if desired

    scan.start_playing(frame_parameters=frame_parameters)

    time.sleep(frame_time * 0.75)

    scan.stop_playing()
    scan.abort_playing()

How do I configure the scan to acquire a subscan of an existing scan?
---------------------------------------------------------------------
A subscan covers a region of the field of view. To record one frame of a subscan, give ``record_immediate`` frame
parameters with subscan fields of their own. The recording scans them as given, whether the user's subscan is on or off,
so the subscan's size in pixels is given along with its region. The user's subscan is unchanged afterwards. ::

    import math
    from nion.utils import Geometry
    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    frame_parameters = scan.get_current_frame_parameters()
    frame_parameters.subscan_pixel_size = Geometry.IntSize(h=100, w=75)
    frame_parameters.subscan_fractional_size = Geometry.FloatSize(h=0.4, w=0.3)
    frame_parameters.subscan_fractional_center = Geometry.FloatPoint(y=0.5, x=0.5)
    frame_parameters.subscan_rotation = math.radians(15)

    scan.abort_playing(sync_timeout=5.0)
    frame = scan.record_immediate(frame_parameters)[0]

While the user's subscan is on, the current frame parameters carry it in these fields, so a recording made from them
scans the user's subscan unless the fields are replaced, or set to ``None`` to record the whole field of view.

The user's own subscan is set with the scan's ``subscan_region``, ``subscan_rotation`` and ``subscan_enabled``
properties. Setting them changes the subscan graphic, the scan control panel and the active scan profile, as the user's
own change would, and a view scans the subscan while it is on. ::

    import math
    from nion.utils import Geometry
    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    scan.subscan_region = Geometry.FloatRect.from_center_and_size(Geometry.FloatPoint(y=0.5, x=0.5), Geometry.FloatSize(h=0.4, w=0.3))
    scan.subscan_rotation = math.radians(15)
    scan.subscan_enabled = True

    scan.start_playing()

============================    ===========
Name                            Description
============================    ===========
subscan_region                  the subscan's fractional rectangle, a ``Geometry.FloatRect`` relative to the field of view
subscan_rotation                the subscan's rotation around the center of its region, in radians
subscan_enabled                 whether the scan scans the subscan instead of the whole field of view
============================    ===========

The region and rotation are kept while the subscan is off, and are used again when it is turned on.

The scan decides the size in pixels of the user's subscan. Its height and width are the region's fractional height and
width times the height and width of the current frame parameters' ``pixel_size``. When the current frame parameters'
``subscan_pixel_width_override`` is set, the subscan is that many pixels wide and as many pixels high as the region's
aspect ratio gives.

A subscan's region is relative to the field of view and has the same rotation. The (0, 0) point is at the top left and
the (1, 1) point is at the bottom right. Coordinates are specified in y-axis, x-axis order.

Changing the rotation will rotate the scan around the microscope axis and the subscan will generally be off axis; so a
rotation will effectively shift a subscan in addition to rotating it. Positive rotation is counterclockwise.

.. _synced-acquisition:

How do I configure a rectangular scan synchronized with a camera?
-----------------------------------------------------------------
A synchronized acquisition puts a camera producing a trigger signal together with a scan configured to advance on an
external trigger. The camera is asked to acquire a sequence of frames corresponding to the size of the scan plus
overhead required by the scan (flyback). The operation results in scan data and data from the camera.

Although not possible at the moment, we expect future capabilities to include the ability to combine acquisition from
multiple cameras/devices.

The following code will perform a scan synchronized with a camera::

    from nion.utils import Geometry
    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    eels = stem_controller.eels_camera

    scan_frame_parameters = scan.get_current_frame_parameters()
    scan_frame_parameters.pixel_size = Geometry.IntSize(h=16, w=12)

    eels_frame_parameters = eels.get_current_frame_parameters()
    eels_frame_parameters.processing = "sum_project"  # produce 1D spectrum at each scan location
    # further adjust scan_frame_parameters and eels_frame_parameters here if desired

    combined_data = scan.grab_synchronized(
        scan_frame_parameters=scan_frame_parameters,
        camera=eels,
        camera_frame_parameters=eels_frame_parameters)

    frames, camera_data_list = combined_data
    frame = frames[0]
    camera_data = camera_data_list[0]

The scan uses the scan frame parameters it is given, including their rotation. A running view is stopped during the
acquisition. When the acquisition ends, the scan's current frame parameters are set back to what they were before, which
also puts the scan device back on them, and the view is started again if it was running. So the current frame parameters
are unchanged afterwards.

You can use a camera frame parameter to control processing from 2d to 1d data.

============================    =========   ===========
Name                            Immediate   Description
============================    =========   ===========
processing                      no          use "sum_project" to sum and project the data from 2d to 1d
============================    =========   ===========

.. the API will handle multiple cameras (eventually) by passing 'cameras' instead of 'camera', etc.
.. the API needs to handle error conditions or abort
.. the API will generally connect acquisition to channels, which the user can view and cancel

..
    How do I configure a line scan synchronized with a camera?
    ----------------------------------------------------------
    You can configure a scan with a height of one and an appropriate rotation to perform a combined acquisition along an
    arbitrary line. The calculations are tedious so a help routine is provided. ::

        from nion.utils import Registry
        stem_controller = Registry.get_component("stem_controller")

        scan = stem_controller.scan_controller

        ronchigram = stem_controller.ronchigram_camera

        scan_frame_parameters = scan.get_current_frame_parameters()

        ronchigram_frame_parameters = ronchigram.get_current_frame_parameters()
        # further adjust scan_frame_parameters and ronchigram_frame_parameters here if desired

        line_scan_frame_parameters = scan.calculate_line_scan_frame_parameters(scan_frame_parameters, start, end, length)

        combined_data = scan.grab_synchronized(
            scan_frame_parameters=line_scan_frame_parameters,
            camera=ronchigram,
            camera_frame_parameters=ronchigram_frame_parameters)

        frames, camera_data_list = combined_data
        frame = frames[0]
        camera_data = camera_data[0]

    The scan and camera data will be returned with one fewer collection dimension since the data will be squeezed to get rid
    of the extra dimension with size of one.

..
    How do I configure complex multi-region scans synchronized with a camera?
    -------------------------------------------------------------------------

    How do I do multiple acquisitions at each point in a synchronized scan?
    -----------------------------------------------------------------------

    How do I perform an action between regions in a multi-region synchronized scan?
    -------------------------------------------------------------------------------

..
    How do I acquire a sequence of scans?
    -------------------------------------
    You can grab a sequence of scans as long as they have the same pixel size. ::

        from nion.utils import Registry
        stem_controller = Registry.get_component("stem_controller")

        scan = stem_controller.scan_controller

        scan.set_enabled_channels([0, 1])
        frame_parameters = scan.get_current_frame_parameters()
        # adjust frame_parameters here if desired

        scan.start_playing(frame_parameters)

        # grab consecutive frames, with a guaranteed start time after the first call
        if scan.grab_sequence_prepare(10):
            frames_list = scan.grab_sequence(10)
            if frames_list:
                for frames in frames_list:
                    # each frames will have data for each channel
                    frame1, frame2 = frames

How do I grab recently scanned data?
------------------------------------
You can grab recently acquired scans (as long as they each have the same pixel size) by using this code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    scan.set_enabled_channels([0, 1])
    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    scan.start_playing(frame_parameters=frame_parameters)

    # grab buffered frames
    frames_list = scan.grab_buffer(10)
    if frames_list:
        for frames in frames_list:
            # each frames will have data for each channel
            frame1, frame2 = frames

How do I find data items associated with viewing and recording?
---------------------------------------------------------------
The scan device pushes its data through data channels which are connected to data items via data item references in Nion
Swift. To find the associated data item, you must find the associated data item reference key (there will be one for
each individual scan detector and application) and then ask Nion Swift for the associated data item. The default
channel mapping forms the key from the scan's hardware source id and the detector's channel id joined with underscores,
with ``subscan`` added for the subscan. A scan device's data channel delegate can change the mapping, for example to put
the subscan's data into the context data item, and then the key differs. ::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    channel_id = scan.get_channel_state(0).channel_id
    reference_key = "_".join([scan.hardware_source_id, channel_id, "subscan"])

    data_item = api.library.get_data_item_for_reference_key(reference_key)

You can also create or get a data item which will be the target of an acquisition. This is useful if you need to set up
the data item in a particular display panel in a workspace in Nion Swift. ::

    data_item = api.library.get_data_item_for_reference_key(reference_key, create_if_needed=True, large_format=False)

How do I determine scan parameters from acquired data's metadata?
-----------------------------------------------------------------
The scan parameters are saved in the metadata of acquired xdata or data items. You can create new frame parameters from
metadata using the following technique::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    frame_parameters = scan.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    scan.start_playing(frame_parameters=frame_parameters)

    # grab a frame as an example
    frame = scan.grab_next_to_finish()[0]

    new_frame_parameters = scan.get_frame_parameters_from_metadata(frame.metadata)

.. _probe-position:

How do I control the probe when not scanning?
---------------------------------------------
You can determine the probe state and probe position. The probe state will be either "scanning" or "parked". If "parked"
the position will be either None or a fractional position relative to the most recently acquired data. ::

    from nion.utils import Geometry
    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    print(stem_controller.probe_state)
    print(stem_controller.probe_position)

    stem_controller.probe_position = Geometry.FloatPoint(y=0.6, x=0.4)
    stem_controller.probe_position = None  # move to default parked position

.. TODO: observing probe_position, probe_state changes
.. TODO: partial data acquisitions
.. TODO: monitoring changes to current values
.. TODO: get/set named/saved settings
