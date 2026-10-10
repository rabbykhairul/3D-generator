# Clear-frame boundary investigation

## Engine 18 follow-up: neutral side arm and front transitions

The neutral side fallback now retains colorless material, retries nearby root
seeds when the oblique front blocks tracking, and requires agreement across
three contrast masks. It runs at a common analysis size. The distal tip uses
strong observed perimeter pixels to complete a small clear cap, bounded to
12% of arm length and 8% added arm area. This is inferred cap completion, not
manufacturing geometry. Existing colored-arm detection keeps its original path.

Six actual-reference side tests cover 540/1080/2160, including mirrored left
views: shaft and clear tip retained, rear arm/front rim/shadow excluded, hinge
on the outline. A disposable front-plus-right-arm export test verifies that
the GLB contains the arm and its welded geometry is watertight. Saved merchant
annotations are not changed.

Front faint-boundary recovery extends through the nasal transition, while
remaining above the lower contact-shadow band. Bounded closing/opening removes
narrow reflection notches. Updated source overlays: `after.png` and
`arm-after.png` in `artifacts/clear-boundary`. Both were visually inspected.
These are outline/export checks, not a live AR verification or a guarantee
that invisible surfaces can be recovered from every photograph.

Reproduced the supplied screenshot using the saved original water-clear front.
Stable aperture bounding boxes do not establish a correct external boundary:
the threshold contour follows long internal reflection paths through clear material.

Added bounded morphological closing of the filled outer silhouette (not the
thin rim annulus). This reduces small inward reflection returns; the two lens
apertures remain separately validated. The repair is resolution-relative and
limited by area and filled-region displacement. Saved approvals are untouched.

Engine 17 follow-up: separate faint-upper-silhouette recovery now removes the
long brow return and recovers the observed nasal opening from the image. It uses
smoothed LAB distance thresholds 4/5/6, requires majority boundary agreement,
and combines only the upper region with the stable-aperture mask. The lower
contact-shadow band is excluded from weak-threshold recovery. Analysis is
normalized to 1080 pixels so the same guards apply at different source sizes.
This does not create missing source detail or guarantee arbitrary-photo recovery.

The previous geometric brow completion was discarded because it made an
unsupported straight nasal edge. The retained correction uses actual faint
image evidence instead. Fine lower nasal hardware separation remains approximate;
this is a front-boundary fix, not a claim of perfect 3D reconstruction or AR.

Evidence: `artifacts/clear-boundary/before.png` and `after.png` show the same source
and drawing scale. Eight focused clear-frame tests pass, including resolution,
negative-background and reconstruction checks. New spatial assertions verify
material above the nose, an open nose aperture and excluded contact shadow at
540, 1080 and 2160 source pixels. The existing 4096 test also passes.
