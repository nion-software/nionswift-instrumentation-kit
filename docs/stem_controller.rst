.. _stem-instrument:

STEM Instrument
===============

Accessing the STEM Instrument
-----------------------------
To do instrument control, you will need to get a versioned ``instrument`` object from the ``api`` object using an
``instrument_id`` (see :ref:`hardware-source-identifiers`). ::

    autostem = api.get_instrument_by_id("autostem", version="1.0")

Instrument Properties
---------------------
Once you have an ``instrument`` object, you can set and get properties on the instrument. ::

    if autostem.get_property_as_bool("ShowTuningImages"):
        show_data()

Properties are typed and the following types are supported:

    - float
    - int
    - str
    - bool
    - float_point

You can also set properties on an instrument. ::

    superscan.set_property_as_float_point("probe_position", (0.5, 0.5))

For more information about these methods, see :py:class:`nion.swift.Facade.Instrument`.

Instrument Controls
-------------------
A set of methods to access a special subset of properties called *controls* is also available.

Controls are special properties that are always represented as float values and may represent combinations of other
controls. Their methods have special features which allow more precise setting within the network of controls.

Autostem controls are characterized as having a internal "local" value added to weighted sum of values from zero or more
input controls. Changing the value of an input control can change the output value of other controls.

Setting Output Values
---------------------
You can set values on controls in such a way as to allow changes to propogate to dependent controls or not.

To set the output value of a control, use the ``set_control`` method with no options. ::

    autostem.set_control_output("d3x", d3x_value)

Confirmation
------------
When setting the absolute output value of a control, you can confirm the value gets set by passing an options dict with
a ``value_type`` key of ``confirm``. ::

    autostem.set_control_output("d3x", 0.0, options={'confirm': True})

You can also add options for tolerance factor when confirming. The tolerance factor default is 1.0 and should be thought
of as the nominal tolerance for that control. Passing a higher tolerance factor (for example 1.5) will increase the
permitted error margin and passing lower tolerance factor (for example 0.5) will decrease the permitted error margin
and consequently make a timeout more likely. The tolerance factor value 0.0 is a special value which removes all
checking and only waits for any change at all and then returns.

To set d3x to within 2% of its nominal target value ::

    autostem.set_control_output("d3x", 0.0, options={'confirm': True, 'confirm_tolerance_factor': 1.02})

You can also add timeout options when confirming. The default timeout is 16 seconds. ::

    autostem.set_control_output("d3x", 0.0, options={'confirm': True, 'confirm_timeout': 16.0})

If the timeout occurs before the value is confirmed, a ``TimeoutException`` will be raised.

Local Values
------------
You can set the *local* value of a control by passing an options dict with a ``value_type`` key of ``local``. ::

    autostem.set_control_output("d3x", 0.0, options={'value_type': 'local'})

Delta Values
------------

You can change a control by a delta value by passing an options dict with a ``value_type`` key of ``delta``. ::

    autostem.set_control_output("d3x", d3x_delta, options={'value_type': 'delta'})

Inform, or Keeping Dependent Outputs Constant
---------------------------------------------

Finally, you can adjust a control in such a way that the output values of dependent controls stay constant. This is
useful during setup when you want to change the displayed value without actually changing the dependent outputs. You do
this by passing an options dict with a ``inform`` key of True. This parameter is named ``inform`` for historical
reasons but can also be thought of as *keep dependent outputs constant*. ::

    autostem.set_control_output("d3x", d3x_value, options={'inform': True})

Control State
-------------

Finally, you can query the state of a control to see if it exists or to see its current state. The only defined
return values at the moment are None and 'undefined' state. ::

    if autostem.get_control_state("dqt") is not None:
        run_dqt_adjustment()

For more information about these methods, see :py:class:`nion.swift.Facade.Instrument`.
