# Scanning

This is a preliminary design document with minimal content.

## Requirements

Email from ND 2020-06-01:

- A list of basic sample characteristics for scanning, any sample can have any combination of these characteristics.
- Some samples are destroyed by the beam, so being able to set things up without illuminating the sample is essential.
- Some samples charge up or heat up if the dose rate of electrons on the sample changes, so the beam must be on the sample illuminating the same area (or sometimes a nearby area), they can still be beam sensitive.
- Using the stage to move around may cause drift so sometimes all movement is done electrically.
- Some samples move around so one must take a lot of fast images and cross-correlate and add them.
- Sometimes the user knows exactly the scan setup relative to the sample that is needed for SI, like 9 pixels per nm and 2ms/ pixel, and when selecting a scan area would want to scan that area with the given dose/resolution, sometimes they’re less quantitative and base their scan settings on patience and guts.
- Often focus and stigmation is done with different scan parameters than the image acquisition, so these setups have to be easily switched between.
- Surveying is usually done at larger fields of view, with fiddling of scan parameters-trying to balance frame rate with ability to see fine structures and to get enough signal-to-noise in the view, throwing away image data when this is going on is disastrous-all frames are needed and all pixels, time and space averaging can work. Surveying often consists of looking through 1000’s of frames of image data looking for a barely visible feature with a need to temporarily improve the imaging to examine possible candidates.
- When operating the operators view is often focused on the sample and you try to select the proper control by feel or looking through the corner of the eye or a quick glance, which is where colored controls and icons are essential and keyboard shortcuts useful.
- Quantitative information is often essential, but is so seldom provided historically that users would need to get used to it to utilize it.
- There’s a lot of trial and error in imaging, so one routinely starts an image looks at partial data and decides some parameter or the sample location itself is wrong and so abandon it so analysis during collection and proper cancellation are both essential.
- When things work it’s often 3am and you just want to collect a dozen 10min images and go to bed, having some parameter set wrong or not recorded can then be a real bummer when analyzing the data three weeks later, this is the real origin of the desire for keeping all raw data.
- The advanced user needs complete access to all hardware functionality in an easy-to-understand way, so that he can conceive and implement new experiments--if it can be done it should be doable.

## User Story

See [nion-software/nionswift-instrumentation-kit#152](https://github.com/nion-software/nionswift-instrumentation-kit/issues/152).

There are multiple stages of setting up an acquisition. First you want to get oriented to the context surrounding the area of interest. That scan needs to cover a large area like maybe 100 nm on a side. And that scan shouldn't take too long to refresh, because you are going to be navigating around, so the pixel time and number of pixels needs to be modest, maybe 5 us pixel time, 512x512 scan points.

Next you identify your area of interest within the context, and make a sub-scan graphic that picks out that area. You want to make sure you are in focus at your acquisition parameters, which might be something like 50 us per pixel, and 512x512 sub-scan pixels. The region of interested might be 4 nm.

At the context scan resolution, the sub-scan would be approx. 20x20 pixels. So, you need to have a different resolution for the sub-scan compared to the context scan.

## User Story

See (internal link) [nion-software/nion-instrumentation#335](https://github.com/nion-software/nion-instrumentation/issues/335).

Sparse sampling or explanation of why this is not needed or possible.

## User Story

User wants to be able to acquire scans on one or more channels with minimal dead time between frames. Be able to stop on command, data cropped to stopping point; count only limited by available memory. Perhaps seeing the aligned/semi-aligned sum of those acquisitions. Reduces drift, low and high frequency noise, more resilient mode for instrumentation. Enabled by fast detectors.
