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

In a synchronized acquisition, each frame advances the scan. The camera detaches the scan before the view starts acquiring again, so a view frame never advances the scan.

Cancel doesn't restart the view. Cancel comes from another thread and can arrive in the middle of a frame. Restarting the view then would acquire on two threads at once, and in a synchronized acquisition it would advance the scan. The view restarts when the sequence ends, or when begin or continue fails. Closing the camera stops the view thread.
