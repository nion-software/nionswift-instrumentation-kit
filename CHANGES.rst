Changelog (nionswift-instrumentation)
=====================================

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
