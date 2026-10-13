# STEM Controller

The STEM controller is the interface through which scripts, plug-ins and the instrumentation panels control the microscope. It is the set of operations Nion Swift commits to offering a user, so code written against it doesn't depend on how a particular instrument is implemented. The simulator in nionswift-usim implements the same interface, so the software and its tests run without hardware. The interface, like the implementation for real instruments, is specific to Nion microscopes.

## Whether a control exists

Which controls exist differs between microscopes and configurations. The STEM controller can say whether a control exists, so a caller can check before using one, instead of calling `TryGetVal` and interpreting the failure.

## Control value streams

A caller can follow a control's value as a stream. A stream never raises an exception while it gets a value. Streams deliver values through events, and an event that hits an exception in one listener prints it and stops, so the listeners after it never get the value. A stream that raised would silently cut off its other consumers, and nobody would see the error except in the printed output.

Each value in a control's stream is therefore a `TryValue`: either the control's value, or the exception that prevented getting it. Getting the stream succeeds whether or not the control exists; a missing control shows up as an exception in the stream. Other designs were rejected:
- sending nothing on an error, which would hide that the value is missing;
- sending no value (`None`), which loses why it is missing;
- keeping the last good value, which shows a stale value as if it were current.

With `TryValue`, the consumer decides what to do. A consumer that wants the last good value can build a stream on top that keeps it.

The exception is an `Exception` rather than a `BaseException`, because `BaseException` also covers the cancellation of asyncio tasks, which must not be captured.

The name `TryValue` follows the existing `TryGetVal`. "Result" (Rust) and "expected" (C++) were considered and not chosen.
