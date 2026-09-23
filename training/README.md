# Eyewear pose training contract

The production landmark detector requires a project-owned Ultralytics pose checkpoint named `eyewear_pose.pt`.

Each YOLO pose label line contains the class and bounding box followed by 16 `(x, y, visibility)` keypoints. Coordinates are normalized to image dimensions. Use visibility `0` for absent, `1` for labeled but occluded, and `2` for visible.

Keypoint order:

1. `rim_left_outer`
2. `rim_left_inner`
3. `rim_left_top`
4. `rim_left_bottom`
5. `rim_right_inner`
6. `rim_right_outer`
7. `rim_right_top`
8. `rim_right_bottom`
9. `bridge_left`
10. `bridge_right`
11. `hinge_left`
12. `hinge_right`
13. `temple_left_end`
14. `temple_right_end`
15. `nose_pad_left`
16. `nose_pad_right`

Keep product-level splits: every view of one SKU belongs to only one of train, validation, or test. Include acetate, metal, rimless, semi-rimless, transparent, dark, reflective, folded, and open-temple examples. The first production candidate must be evaluated on a separately approved merchant-photo set before its immutable checkpoint revision is added to GitHub Actions.

Training is intentionally not part of local service tests and must run on a GPU training environment.

