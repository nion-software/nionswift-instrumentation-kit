Changelog (nionswift-instrumentation)
=====================================

0.18.4 (UNRELEASED)
-------------------
- Fix issue starting scan record immediately after another one.
- Fix issue with reloading scans with leftover probe/subscan graphics.

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
