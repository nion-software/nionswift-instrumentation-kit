Changelog (nionswift-instrumentation)
=====================================

22.1.0 (2023-10-23)
-------------------
- Add console startup component to define stem_controller.
- Fix #161. Ensures proper scan after low level changes.
- Allow enabled channels to be set with frame parameters.
- Fix issue where acquisition control being set past last index.
- Begin work on new acquisition API (work in progress).
- Python 3.12 compatibility.

22.0.0 (2023-09-14)
-------------------
- Fix issues with intermittent errors during synchronized acquisition.
- Fix #13. Scan recorder allows limited-only-by-memory frame count on scan devices.
- Fix #160. Introduce RecordingTask to allow synchronization of recording with other tasks.
- Eliminate special crop region on summed camera acquisition (EELS). Crop region now determined by camera readout area.
- Many improvements to improve ability to replace default scan device; work in progress.
- Simplify acquisition panel slightly (radio button to select mode); work in progress.
- Improvements to metadata handling during acquisition; work in progress.
- Fix 1D ramp acquisition using FOV; add ability to control rotation, too.
- Raise max scan size to 16384x16384 and correctly reduce it when required.
- Performance improvements (latency and frame rate).
- Performance improvements (scan recorder).
- Add Python 3.11 support. Drop 3.8. This will be the last Python 3.9 release, too.
- Fix some issues with subscan handling during synchronized acquisition.

0.21.1 (2023-02-28)
-------------------
- Fix #151. Properly calibrate subscan/line-scan during synchronized acquisition.
- Improve handling of partial acquisition, directing data to desired data item.
- Fix more acquisition graphic edge cases: graphics only on enabled context displays.
- Fix issue where acquisition graphics would not appear on new channels.

0.21.0 (2022-12-07)
-------------------
- Fix sync issue with SI (prep scan after camera stopped).
- Require scan_module (scan device and settings) rather than scan_device to be registered.
- Improve progress bars for 1D line scans.
- Only remove graphics from active acquisition data items (fix #127).
- Enable graphics properly when switching projects (fix #133).
- Only invalidate context field of view, rotation, or center changes (fix #140).
- Retain probe position when disabling and re-enabling (fix #139).
- Check channel states for changes before rebuilding thumbnails to avoid flashing.
- Allow scan modules to supply their own control panel UI.
- Make fov_size_nm be a computed property representing fov with aspect ratio applied.
- Add pixel_size_nm and subscan_pixel_size_nm computed properties to frame parameters.
- Handle scan data calibrations when scan data is 1D. Pass through 3D.
- Add channel_indexes_enabled to frame parameters for future use.
- Enable support for Python 3.11.

0.20.8 (2022-09-13)
-------------------
- Replace flyback_pixels property with calculate_flyback_pixels method.
- Change 2D ramp (tableau) to use relative control values, like 1D ramp.
- Fix regression in multiple shift EELS acquisition.
- Ensure live view calibrations are correct during synchronized acquisition.

0.20.7 (2022-07-26)
-------------------
- Drift correction improvements, simplified UI. Work in progress.
- Performance and reliability improvements.

0.20.6 (2022-06-06)
-------------------
- Fix calibration issue on view modes.

0.20.5 (2022-05-28)
-------------------
- Improve error handling and reporting.
- Use a target size of 64x64 for drift tracking area.
- Allow camera device to supply its own calibrator object.
- Fix PMT issue when index >= 2.
- Allow camera device to specify desired exposure precision.
- Improve support for axis handling in STEM controller.

0.20.4 (2022-02-18)
-------------------
- Fix sequence/series/tableau when used with a synchronized acquistiion.
- Add optional method to validate camera frame parameters.
- Improve compatibility with older CameraDevice implementations.
- Ensure low level data is not directly used in data items. Fixes phantom data issue.
- Add methods to instrument to get/set configuration parameters.

0.20.3 (2021-12-21)
-------------------
- Fix issue handling partial acquisition during synchronized/sequence acquisition.
- Improve error handling and notifications after errors.

0.20.2 (2021-12-13)
-------------------
- Fix issue assembling scan channels during synchronized acquisition when split into sections (drift).
- Add support for camera device 3 (no prepare methods).
- Fix issue with camera state being incorrect in display panel control bar.
- Make auto drift tracker during synchronized acquisition optional (default off).
- Improve error recovery and notification during acquisition errors.
- Improve handling of subscan/drift graphics when switching projects.
- Fix sequence of spectra acquisition when optimized on device.
- Show acquisition activity in activity panel.
- Enable support for Python 3.10.

0.20.0 (2021-11-12)
-------------------
- Add preliminary acquisition panel for more complex sequence/collection acquisitions.
- Acquisition panel includes 1D ramp (series), 2D ramp (tableau), and other options.
- Add preliminary drift tracker, both manual and automatic. Work in progress.
- Add virtual detector option to MultiAcquire.
- Add option to apply shift for each frame in MultiAcquire.
- Remove unused and incomplete monitor button in camera panel.
- Improve internal documentation by utilizing Python protocols for various interfaces.
- Improve reliability and code quality by enabling Python strict typing.
- Drop support for Python 3.7.

0.19.5 (2021-04-12)
-------------------
- Improve multiple shift EELS acquire by allowing disabling of dark subtraction and alignment.
- Synchronized acquisition now attached session metadata to resulting data items.

Thanks to Isobel Bicket/McMaster University for multiple shift EELS contributions.

0.19.4 (2021-03-12)
-------------------
- Compatibility with nionui 0.5.0, nionswift 0.15.5.
- Fix issue with large SI's in wrong storage format (ndata vs h5py).

0.19.3 (2021-01-17)
-------------------
- Add ability to pass some metadata from camera to final synchronized acquisition data. Temporary.
- Fix issue where record did not work after running synchronized acquisition.
- Add camera base methods for setting and clearing gain reference images.

0.19.2 (2020-12-10)
-------------------
- Fix issue with probe position graphic when multiple channels enabled.

0.19.1 (2020-12-08)
-------------------
- Fix issue with metadata in scan recorder result data items.
- Fix issue with spectrum imaging panel Acquire button not getting enabled.
- Rework MultiAcquire to use new partial data item updates.

0.19.0 (2020-08-31)
-------------------
- Add section-by-section drift correction during synchronized acquisition.
- Add support for specifying drift correction parameters (only used in synchronized acquisition).
- Add record_immediate function for scan devices.
- Add partial updating during synchronized acquisition.
- Add optional help button and ability to register delegate for camera panel to handle.
- Fix numerous issues handling the subscan and beam position graphics.
- Fix issue starting scan record immediately after another one.
- Fix issue with reloading scans with leftover probe/subscan graphics.
- Enable spectrum image acquisition for MultiAcquire.
- Removed y-shift and shifter strength from MultiAcquire.
- Added a time estimate to MultiAcquire.

0.18.3 (2019-11-26)
-------------------
- Fix issue with subscan handling during SI.
- Improve exception handling during camera initialization.
- Add support for time-based initial calibration style for scans.

0.18.2 (2019-07-22)
-------------------
- Fix metadata handling in scan devices (was not copying it to the image).

0.18.1 (2019-06-27)
-------------------
- Fix multi acquire data item calibration handling.

0.18.0 (2019-06-25)
-------------------
- Change camera to use instrument controller (generalized stem controller).
- Add stubs for 2D control methods.

0.17.0 (2019-04-29)
-------------------
- Add synchronized acquisition of sub areas and lines.
- Expand ability of custom devices to specify calibration and processing.
- Add multi-acquire panel for acquiring multiple summed sets of spectra with optional energy offsets.
- Support subscan rotation and subscan resolution.
- Improve handling of default stem_controller for camera, scan modules.
- Change shift output messages to match sign of change.

0.16.3 (2019-02-27)
-------------------
- Change camera exposure time and scan pixel time to have two digits of precision.
- Add 'synchronized state' messages to instrument controller, invoked at start/end of synchronized acquisition.
- Limit scan device pixel time in the case of long camera exposure during synchronized acquisition.

0.16.2 (2018-01-18)
-------------------
- Fix closing bug in state controller leading to errors when closing document window.

0.16.1 (2018-12-21)
-------------------
- Change spectrum and 4d images to go into new data items each acquisition.

0.16.0 (2018-12-12)
-------------------
- Add check mark in context menu to indicate active display panel controller.
- Use new display item capabilities in Nion Swift 0.14.

0.15.1 (2018-10-04)
-------------------
- Fix race condition when scripting probe position.

0.15.0 (2018-10-03)
-------------------
- Improve support for sub-scan.
- Expand API and documentation.
- Improve cancel and error handling in synchronized acquisition.
- Remove limitation of PMT to channels 0, 1.
- Add support for acquisition sequence cancellation.

0.14.1 (2018-06-25)
-------------------
- Add STEM controller methods to access ronchigram camera, eels camera, and scan controller.
- Register all cameras via Registry rather than directly in HardwareSourceManager.
- Improve metadata, calibration, and naming during acquisition recording.

0.14.0 (2018-06-21)
-------------------
- Introduce camera modules to replace camera devices. Allows more control of camera settings.

0.13.3 (2018-06-18)
-------------------
- Minor changes to scan acquisition (ensure size is int).
- Fix handling of Ronchigram when scale calibration missing.

0.13.2 (2018-06-04)
-------------------
- Improve handling of sum/project processing in acquire sequence.
- Improve handling of calibration via calibration controls.
- Fix default handling of dimensional calibrations in acquire sequence.

0.13.1 (2018-05-13)
-------------------
- Fix manifest.

0.13.0 (2018-05-12)
-------------------
- Initial version online.
