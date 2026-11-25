# Cameras

## The view during a sequence

A camera has one detector, which the view and sequences share. Each camera device decides what its view shows while a sequence runs, in its `acquire_image` method. The hardware source calls `acquire_image` for the view either way.

Cameras do this in two ways:
- `acquire_image` waits until the sequence is done. The view shows nothing new during the sequence.
- `acquire_image` returns the sequence's frames as they arrive. The view shows what the sequence is acquiring.

The same applies to synchronized acquisitions.

## device_kit cameras

device_kit cameras, used by uSim and the tests, show the sequence's frames in the view, so the user can see what is being acquired. The view gets the raw frames, before processing like `sum_project`, with the sequence's binning and other frame parameters. When the sequence is done, the view goes back to its own frames.

### Threads

The view acquires on the camera's own thread. A sequence acquires on the thread that calls the camera's continue method.

The view thread stops acquiring while a sequence runs, from the sequence's begin to its end. Begin waits for any view frame in progress to finish first. Only one thread at a time calls the simulator or writes the frame buffer, so simulators don't need locks. This assumes only one sequence runs at a time.

During a sequence, each frame advances the scan if the scan takes its clock from the camera and is scanning in external clock mode when the sequence begins. A synchronized acquisition passes its camera's `camera_id` to `prepare_synchronized_scan`, so a frame or a sequence on another camera doesn't advance its scan, as on hardware, where only the camera wired to the scan's external clock input advances it. A scan prepared without a camera takes its clock from any camera. The camera detaches the scan before the view starts acquiring again, so a view frame never advances the scan.

Cancel doesn't restart the view. Cancel comes from another thread and can arrive in the middle of a frame. Restarting the view then would acquire on two threads at once, and in a sequence which clocks the scan it would advance the scan. The view restarts when the sequence ends, or when begin or continue fails. Closing the camera stops the view thread.
