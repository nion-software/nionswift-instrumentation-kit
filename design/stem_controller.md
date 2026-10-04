# STEM Controller Design Notes

## Mode Session Lifecycle and Control-Scoped Observation

This section describes the intent, requirements, and API behavior for mode handling in the STEM controller, with emphasis on client workflows that need to react to mode entry, track a specific control while the mode is active, and clean up when the mode ends.

A mode is represented as a session, not only as a type identifier. The design treats each activation as a distinct lifecycle episode with a unique session identifier. This allows clients to correlate start and end events reliably, even when the same mode type is entered multiple times.

### Functional requirements

The controller must support entering and exiting a mode from the controller side. Entering a mode must return a unique session identifier for that activation. Exiting must target a specific activation by session identifier rather than mode type.

The controller must expose a mode stream that reports lifecycle events as shared truth, regardless of whether the mode was entered by the local controller logic or by device-side behavior. Clients should observe a mode as active when entered and inactive when ended, and should receive an end reason when the mode ends.

The controller must support a common client use case where a specific control is observed only while a matching mode session is active. This must include a way to react to mode completion and stop consuming control updates after mode exit.

Control reliability must be preserved. The API must carry both value and error state so that clients can distinguish valid values from unavailable or unreliable states.

### Data model and semantics

The mode lifecycle is modeled with a `ModeSession` record containing mode identifier, payload, active state, unique session identifier, and optional end reason. The payload carries mode-specific parameters, such as the control name associated with a tracking mode.

End reasons are explicit and represent normal completion, cancellation, device abort, or replacement by another activation. This communicates termination intent and avoids ambiguous boolean-only lifecycle semantics.

For control-scoped mode observation, the API uses `ModeControlSession`, which extends mode session data with the observed control name and a control try-value. The try-value preserves both latest value and any exception indicating reliability issues.

A convenience accessor is provided to read a plain control value. This accessor returns the float only when valid and otherwise raises the underlying exception if one exists. If an invalid state is present without a concrete exception, it raises a value error. This keeps the authoritative state in the try-value while still allowing ergonomic access for callers that want exception semantics.

### API behavior

The mode stream API provides mode lifecycle events for a requested mode stream identifier. The mode control stream API composes mode lifecycle state with control try-value updates for a requested control.

Mode-control composition follows lifecycle gating behavior. Control updates are reflected while the associated mode session is active. After the mode exits, later control updates are ignored for that session state, so the final session snapshot remains stable.

A context-managed mode entry API is provided for client ergonomics and correctness. The context manager enters a mode at scope entry, yields the session identifier, and guarantees exit when the scope ends, including exceptional exits.

### Implementation choices

Mode session and mode control session are implemented as frozen dataclasses.

The mode control stream is implemented as a composed stream adapter. It listens to both mode lifecycle and control try-value updates and emits combined snapshots suitable for UI and tool logic.

Listener cleanup relies on event-listener ownership and dereference behavior already used in the codebase. Explicit finalizer-based cleanup in the composed stream was intentionally avoided.

The instrument-side default mode handling supports arbitrary modes for testing and development. Entering a mode creates a new session record, publishes an active lifecycle update, and tracks the session. Exiting a known session publishes an inactive lifecycle update with completion reason.

### Client integration pattern

A practical client pattern is to observe the mode stream and filter for mode payloads that match expected criteria, such as a target control name. On matching entry, the client creates a mode-control stream for that control and updates display overlays while the session is active. On matching exit, the client detaches listeners and removes overlays.

This pattern enables reactive UI behavior for transient adjustment modes and ensures overlays do not outlive their session.

### Test coverage focus

Tests emphasize stream composition, lifecycle transitions, and context-manager safety.

Coverage verifies that mode-control snapshots are emitted when a mode is active and control values change, that inactive mode state is reported with end reason after exit, and that post-exit control updates do not mutate the completed session snapshot.

Coverage also verifies that context-managed mode handling exits reliably when an exception is raised inside the active scope, which is important for preventing stale active-mode state in real client flows.

Additional high-value tests can focus on overlapping sessions and ensuring session identifiers prevent cross-talk between same-mode activations.

### Future extension: target-scoped modes

If future use cases require scoping modes to specific devices (for example a particular camera), this model can be extended without changing the core lifecycle semantics. A target concept can be added as an explicit parameter to mode entry and stream lookup APIs, while keeping the existing STEM-scoped methods as convenience wrappers.

A practical target model should allow global STEM scope, identifier-based scope, type-based scope, and runtime-object scope. The recommended retrofit path is to add target-aware methods first and retain existing methods as compatibility aliases to the global STEM target.

