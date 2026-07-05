# Scanning

This is a preliminary design document with minimal content.

## User Story

See [GitHub Issue #152](https://github.com/nion-software/nionswift-instrumentation-kit/issues/152).

There are multiple stages of setting up an acquisition. First you want to get oriented to the context surrounding the area of interest. That scan needs to cover a large area like maybe 100 nm on a side. And that scan shouldn't take too long to refresh, because you are going to be navigating around, so the pixel time and number of pixels needs to be modest, maybe 5 us pixel time, 512x512 scan points.

Next you identify your area of interest within the context, and make a sub-scan graphic that picks out that area. You want to make sure you are in focus at your acquisition parameters, which might be something like 50 us per pixel, and 512x512 sub-scan pixels. The region of interested might be 4 nm.

At the context scan resolution, the sub-scan would be approx. 20x20 pixels. So, you need to have a different resolution for the sub-scan compared to the context scan.
