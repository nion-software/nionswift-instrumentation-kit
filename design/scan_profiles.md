# Scan profiles

## What scan profiles are

A scan profile is a named set of values for some of the scan device's parameters, which the user switches between, such as a fast survey scan and a slow acquisition scan.

Each scan device has one profile model, `scan_hardware_source.scan_profiles` in `nion/instrumentation/ScanProfiles.py`, shared by all of its scan panels. It holds an ordered list of profiles. Exactly one is active, and the model remembers the previously active one.

The parameters are the field width, the scan size, the pixel time, the rotation, AC line sync, and the subscan size, position, pixel width and rotation. Most parameters have one value; the scan size has three (the width, the height and the link between them), and the subscan size and position have two each. A profile includes some of the parameters, each as a whole, and has values for them. Activating a profile applies the values of the parameters it includes to the scan device; a parameter it doesn't include keeps its current value. Values a profile keeps for a parameter it doesn't include have no effect while the parameter isn't included.

While the subscan is off, activating a profile applies its included subscan size, position and rotation to the scan device's subscan region and rotation, which the subscan uses when it is turned on.

Whether the subscan or the line scan is on is not part of a profile, so activating a profile turns neither on nor off.

The scan device scans with its current frame parameters. A change to them or to the subscan region and rotation, from the scan panel or the subscan graphic, changes the active profile's value of each parameter it includes. A change made while the scan device is recording or in sequence mode doesn't change the active profile, because those are the acquisition's temporary frame parameters, not edits by the user. A profile can't be activated while the scan device is recording or in sequence mode, because the acquisition restores the frame parameters it started from when it ends, which would write them into the newly active profile.

The active profile follows edits as they happen, without a separate save step. This keeps the behavior of the scan panel before profiles, where an edit went straight into the selected hardware profile slot, and it means there is no unsaved state to show, confirm or lose: switching to another profile and back returns to the values last used. The cost is that there is no saved version to return to. An exploratory change, a mistake, or a change from a script or another scan panel replaces the active profile's value, and there is no undo. To keep a profile's values while experimenting, duplicate it first, or switch to a profile meant for experimenting.

In the dialog, editing a value of a parameter the active profile includes applies it to the scan device immediately; editing any other value changes only the profile. When the active profile starts including a parameter, its value becomes the parameter's current value. An invalid value is ignored, and the profile keeps its previous value.

Profile names are unique. The active profile can't be deleted, so the last profile can't be either.

## Persistence

The profiles, the active and previous profiles, and the current value of each parameter, including the subscan region and rotation while the subscan is off, are saved to `<hardware_source_id>_scan_profiles.json` in the configuration location, separate from the scan settings file, through a temporary file that replaces the old one. Without a saved file, the initial profiles come from the scan device's hardware profile slots. They are ordinary profiles, which can be renamed, edited, reordered and deleted like any other. A file that can't be read, or whose structure the model can't fully use, such as a misspelled `profiles` key or an entry that isn't a profile, is renamed with `.bad` appended, and the profiles are loaded from whatever part of it the model can use; if it can't be renamed, the profiles aren't saved in that session. An invalid individual value is ignored without renaming the file, and the profile keeps its value for it.

## The user interface

The scan panel's top row has a button that opens the profile dialog (the sliders icon), the label "Profile", a combo box that shows and changes the active profile, a button that switches to the previous profile, and at its right end a button for the hardware configuration (the gear icon). The Record button records once with the current frame parameters.

The profile dialog lists the profiles, marking the active one. Selecting a row only shows the profile; the Activate button, a double click or return activates it. Buttons under the list add, delete, duplicate and reorder profiles; a new profile includes every parameter at its current value. For the selected profile, each parameter has a check box for whether the profile includes it and an editor for each of its values. The profile's actions menu has Record Once, which records once applying the values of the parameters the profile includes, without activating it. The dialog remembers its position and size for each scan device.
