.. _using-cameras:

Cameras
=======
There can be several cameras on a STEM microscope, but the Ronchigram and optional EELS camera are the primary ones.

.. contents::

General Info
++++++++++++

How does camera acquisition work?
---------------------------------
The cameras for the STEM microscope typically operate in a continuous acquisition mode (also known as view mode). To
get high quality data normally means that the camera will need to already be running in the desired mode by the time
you expose and read data in order to minimize startup or mode change artifacts.

Although the details for a given camera may vary, a camera will typically have a these phases to acquire data: exposure,
readout, corrections, and processing.

Cameras start with an exposure phase. After the exposure phase, and typically overlapping with the next exposure phase,
the data is read from the camera into a frame buffer. Once in the frame buffer, corrections such as blemish removal,
dark subtraction, and gain normalization are performed. Next, processing such as binning to height one or averaging
with the previous frame is applied. Finally, the finished data can be displayed, stored, or further processed.

It is important to understand these phases in order to understand how internal changes to camera settings or external
changes are reflected in the acquired data. Specifically, if you make an external change, such as changing defocus on
the microscope, you will need to ensure that you grab a frame whose _exposure_ has started after the external change has
been applied to guarantee that the grabbed data reflects the external change. This process varies between cameras so it
is recommended to follow the advice on this page to accomplish this.

What parameters can be controlled on the camera?
------------------------------------------------
A camera is configured with frame parameters, which is just a Python `dict` structure.

The following parameters are supported:

============================    ===========
Name                            Description
============================    ===========
exposure_ms                     the exposure time, in milliseconds
binning                         the binning factor, may apply only to one dimension
============================    ===========

The camera tracks its current frame parameters during acquisition.

Frame parameters applied while the camera is playing are marked as pending and take effect on the next frame.

How does camera acquisition interact with Nion Swift and plug-ins?
------------------------------------------------------------------
Acquisition threads send their data into Nion Swift via a data channel. Unless otherwise configured, a data channel will
feed into a reusable data item which can be displayed in the user interface.

There is currently no mechanism whereby a plug-in or script using a camera can exclude actions from other plug-ins or
scripts. Access is managed by convention and the user not running conflicting applications simultaneously.

How does camera acquisition interact with Python threads?
---------------------------------------------------------
In Nion Swift, the user interface is run on the main UI thread. It is important that function calls made from the main
UI thread return within a short period of time (< 50ms). This ensures that the UI is responsive.

During acquisition, function calls to the camera can easily take more than 50ms. For this reason, most of the examples
on this page should be run from a thread other than the main UI thread.

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

How do I access the STEM Controller and cameras?
------------------------------------------------
You can access the STEM controller and cameras using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    ronchigram = stem_controller.ronchigram_camera

    eels = stem_controller.eels_camera

How do I configure the camera and start view mode?
--------------------------------------------------
You can configure a camera and start viewing using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    ronchigram = stem_controller.ronchigram_camera

    frame_parameters = ronchigram.get_current_frame_parameters()
    frame_parameters["binning"] = 4
    frame_parameters["exposure_ms"] = 200

    ronchigram.start_playing(frame_parameters)

How do I determine if the camera is running?
--------------------------------------------
You can make a rough determination if a camera acquisition is running using the following::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    ronchigram = stem_controller.ronchigram_camera

    is_playing = ronchigram.is_playing

You shouldn't use this technique to synchronize acquisition as it does not handle threads and race conditions in a
predictable manner. For instance, it may not be accurate if called immediately following a call that initiates
acquisition; likewise it may not be accurate if called immediately before acquisition ends.

How do I stop or cancel view mode?
----------------------------------
There are two ways to cancel a camera acquisition: stop and abort. Stop waits until the end of the current frame, while
abort stops as soon as possible. Aborting a camera acquisition may result in partially acquired data. You can abort a
camera acquisition that has already been stopped. ::

    import time
    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    ronchigram = stem_controller.ronchigram_camera

    frame_parameters = ronchigram.get_current_frame_parameters()
    frame_parameters["exposure_ms"] = 200
    # adjust frame_parameters here if desired

    ronchigram.start_playing(frame_parameters)

    time.sleep(0.15)

    ronchigram.stop_playing()
    ronchigram.abort_playing()

How do I configure the camera and acquire data?
-----------------------------------------------
You can configure a camera, start viewing, and grab data from the acquisition using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    ronchigram = stem_controller.ronchigram_camera

    frame_parameters = ronchigram.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    ronchigram.start_playing(frame_parameters)

    # grab two consecutive frames, with a guaranteed start time after the first call
    frame1 = ronchigram.grab_next_to_start()[0]
    frame2 = ronchigram.grab_next_to_finish()[0]

The ``grab_next_to_start`` call waits until the next frame starts and then grabs it. The ``grab_next_to_finish`` call
waits until the current frame ends and then grabs it. Both calls return a list of ``xdata`` objects with an entry for
each enabled channel. In this case the first element is selected since only a single channel is enabled.

The ``grab_next_to_start`` will grab the next frame that begins the readout phase after the function call. However, it
will not ensure that the _exposure_ started after the function call. To ensure your code grabs a frame that is exposued
_after_ the call, you should first make a call to ``grab_next_to_start`` followed by a call ``grab_next_to_finish``.

How do I acquire a sequence of frames from a camera?
----------------------------------------------------
You can grab a sequence of frames from a camera acquisition as long as they each have the same pixel size. ::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    eels = stem_controller.eels_camera

    frame_parameters = eels.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    eels.start_playing(frame_parameters)

    # grab consecutive frames, with a guaranteed start time after the first call
    if eels.grab_sequence_prepare(10):
        frames_list = eels.grab_sequence(10)
        if frames_list:
            for frames in frames_list:
                # each frames will have data for each channel
                # eels may have two channels: 2d and 1d data; grab the last one (1d)
                frame = frames[-1]

This capability may not be available on all cameras.

How do I grab recently acquired data?
-------------------------------------
You can grab recently acquired data (as long as they each have the same pixel size) by using this code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    eels = stem_controller.eels_camera

    frame_parameters = eels.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    eels.start_playing(frame_parameters)

    # grab buffered frames
    frames_list = eels.grab_buffer(10)
    if frames_list:
        for frames in frames_list:
            # each frames will have data for each channel
            frame1, frame2 = frames

This capability may not be available on all cameras.

How do I find data items associated with viewing and sequence acquisition?
--------------------------------------------------------------------------
The camera pushes its data through data channels which are connected to data items in Nion Swift. To find the associated
data item, you must find the associated data channel name and then ask Nion Swift for the associated data item. ::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    ronchigram = stem_controller.ronchigram_camera

    frame_parameters = ronchigram.get_current_frame_parameters()

    data_channel_id = ronchigram.get_data_channel_id(frame_parameters)

    data_item = api.library.get_data_item_for_data_channel_id(data_channel_id)

How do I determine camera frame parameters from acquired data's metadata?
-------------------------------------------------------------------------
The camera frame parameters are saved in the metadata of acquired xdata or data items. You can create new frame
parameters from metadata using the following technique::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    ronchigram = stem_controller.ronchigram_camera

    frame_parameters = ronchigram.get_current_frame_parameters()
    # adjust frame_parameters here if desired

    ronchigram.start_playing(frame_parameters)

    # grab a frame as an example
    frame = ronchigram.grab_next_to_finish()[0]

    new_frame_parameters = ronchigram.create_frame_parameters(frame.metadata["hardware_source"])

How do I configure a rectangular scan synchronized with a camera?
-----------------------------------------------------------------
See :ref:`synced-acquisition`

.. TODO: monitoring changes to current values
