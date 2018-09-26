.. _stem-instrument:

STEM Instrument
===============
The STEM controller represents the STEM instrument. The scan device is tightly integrated into the STEM controller and
some functions related to scanning are available in this object too. The scan controller, Ronchigram camera, and EELS
camera are also accessible from this controller.

.. contents::

How do I access the STEM controller?
------------------------------------
You can access the STEM controller with the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

Older plug-ins may also access the STEM controller directly via the Nion Swift API with this code::

    stem_controller = api.get_instrument_by_id("autostem", version="1.0")

How do STEM controller controls work?
-------------------------------------
The STEM controller manages a special set of properties called *controls*.

Controls are special properties that are always represented as float values and may represent combinations of other
controls. Their methods have special features which allow more precise setting within the network of controls.

Controls are characterized as having an internal "local" value added to weighted sum of values from zero or more input
controls. Changing the value of an input control can change the output value of other controls.

How do I access STEM controller controls?
-----------------------------------------
Once you have a STEM controller, you can access properties of the instrument using code like this::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    success, defocus_value = stem_controller.TryGetVal("C10")

    if stem_controller.SetVal("C10", defocus_value - 2000E-9):
        print("Defocus successfully changed.")

    blanked = bool(stem_controller.GetVal("C_Blank"))  # will throw exception if not available

The code above shows the :code:`TryGetVal`, :code:`GetVal` and :code:`SetVal` functions. The :code:`SetVal` function
tries to immediately set the value and returns without delay.

You can also use the :code:`SetValDelta` function which adds. The :code:`SetVal` above can be replaced with this line::

    stem_controller.SetValDelta("C10", -2000E-9)  # decreases C10 by 2000nm

You can also use :code:`SetValWait` and :code:`SetValAndConfirm`. The former waits a specified number of milliseconds
before returning a failure; while the latter waits for the value to be set and confirmed, up to a specified number of
milliseconds. ::

    stem_controller.SetValWait("C10", 1000)  # wait 1 second
    stem_controller.SetValAndConfirm("C10", 500E-9, 1.0, 3000)  # wait 3 seconds for C10 to be set to 500nm

The :code:`SetValAndConfirm` function is useful to be assured that the value has been set before proceeding with
acquisition. The `1.0` parameter is the tolerance factor and `1.0` signifies its nominal value.

Finally, you can adjust a control in such a way that the output values of dependent controls stay constant. This is
useful during setup when you want to change the displayed value without actually changing the dependent outputs,
somewhat like tare function. This function is named :code:`InformControl` for historical reasons but can also be thought
of as keep dependent outputs constant*. ::

    stem_controller.InformControl("C10", 0)  # defocus will now be displayed as 0, but output values won't change

How do I determine if a control exists?
---------------------------------------
When a control doesn't exist, it will return `False` from :code:`TryGetVal`::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")
    exists, _ = stem_controller.TryGetVal("C93")
    assert not exists

How do I access the scan and camera controllers?
------------------------------------------------
The scan controller, Ronchigram camera, and optional EELS camera are integral parts of the STEM instrument. They are
accessible using the following code::

    from nion.utils import Registry
    stem_controller = Registry.get_component("stem_controller")

    scan = stem_controller.scan_controller

    ronchigram = stem_controller.ronchigram_camera

    eels = stem_controller.eels_camera

On systems without an EELS detector, the :code:`eels_camera` property will be :code:`None`.

How do I control the probe position?
------------------------------------
You can determine the probe state and control the probe position using the STEM controller. The probe position is
specified in terms of the last scan.

See :ref:`probe-position` for more information.

.. TODO: how to set the local value of a control
.. TODO: how to get the state of a control (i.e. does it exist)
.. TODO: older functions
.. TODO: change stage position
.. TODO: has_monochromator (add to stem controllers)
.. TODO: defocus (add to stem controllers)
.. TODO: observing control changes
