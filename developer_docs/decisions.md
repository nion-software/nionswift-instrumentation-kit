# Architectural Decision Records

## ADR-003-Try-Value-Stream

#### Date: 2025-09-05. Authors: Chris Meyer. Matthew Stagg.

Streams must not throw exceptions while retrieving values.

So we introduce a type parameterized `TryValue` with two fields `value: <type> | None` to represent either the expected value result, and `exception: Exception` to represent an unexpected exception. For the exception, we use `Exception` and do not use `BaseException` since that is only used for cancelling asyncio.

We also introduce `get_control_try_value_stream`. This function succeeds whether the desired control exists or not.

Other names considered were "Result" (Rust) and "expected" (C++), however `TryValue` fits more closely with our current code.

In order to avoid bad caller behavior, `TryValue` `value` field will be `None` when `exception` is not `None`. If callers want to maintain a "last value" they can design another latching stream on top of the low level stream to track the "last valid value".

## ADR-002-Does-Control-Exist

#### Date: 2025-09-05. Authors: Chris Meyer. Matthew Stagg.

Add a `does_control_exist -> bool` function which determines whether a control exists at the call to avoid callers having to handle awkward exceptions.

## ADR-001-Stem-Controller-Abstraction

#### Date: 2025-09-05. Authors: Chris Meyer. Matthew Stagg.

`stem_controller` represents what we are willing to commit to allow a user to do. And `stem_controller` represents the abstraction necessary for the simulator.

We are currently resigned to the fact that both the `stem_controller` and the `AutoSTEMController` classes are highly Nion microscope specific.
