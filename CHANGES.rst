Changelog (nionswift-instrumentation)
=====================================

0.20.0 (2011-11-12)
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
