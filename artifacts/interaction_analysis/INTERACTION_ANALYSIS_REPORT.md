# Pedestrian–Vehicle Interaction Analysis

## Scope and result

This experiment produces same-frame pedestrian–vehicle interaction candidates and image-space movement descriptors. It is not a validated collision predictor and does not assign risk levels.

- Recording filter: sets set01; videos video_0001.
- Pedestrian observations: 5,007 across 173 tracks.
- Vehicle trajectory observations: 13,226 across 429 class-aware tracks.
- Candidate pair observations: 29,564.
- Unique pedestrian–vehicle track pairs: 2,543; persistent pairs (at least two observations): 1,851; single-observation pairs: 692.
- Valid/invalid per-observation movement estimates: 26,698/2,866.
- Valid/invalid persistent-pair movement summaries: 1,487/364.
- Valid/invalid observed closest-approach summaries: 1,487/364.

## Method

Observations are joined only on the exact `(set_id, video_id, frame_index)` key. Pedestrian and vehicle tracker IDs remain separate and are preserved; vehicle identities are class-aware. Every pedestrian and vehicle present in a shared frame forms one candidate pair, with no interpolation or distance-based pair pruning.

The pedestrian footpoint is the bounding-box bottom-center `((x1+x2)/2, y2)`. The vehicle reference point for displacement is the bounding-box center. Per-frame separation is the Euclidean distance from the footpoint to the nearest point on the vehicle rectangle, obtained by clamping each footpoint coordinate to the box bounds: `q=(clamp(px,x1,x2), clamp(py,y1,y2))`, `d_px=sqrt((px-qx)^2+(py-qy)^2)`. The vehicle center is an image-space reference only, not a physical position.

For valid successive pair observations, pedestrian displacement is the change in footpoint; vehicle displacement is the change in box center; relative displacement is `delta_vehicle_center - delta_pedestrian_footpoint`; its image-plane direction is `atan2(relative_dy, relative_dx)`. Approaching/separating is determined by `delta_d = d_current - d_previous`: negative beyond the configured tolerance means approaching, positive means separating, otherwise indeterminate. No movement is computed across nonpositive timestamps, timestamp misalignment, or frame gaps above the configured limit. Positions are never interpolated.

The closest-approach output is the minimum **observed** separation inside the longest consecutive valid pair segment. It is not a projected future closest approach and is not TTC.

## Alignment and data quality

- Shared frames: 1,525; timestamp-consistent: 1,525; mismatched: 0; missing/invalid timestamps: 0.
- Maximum timestamp spread: 0.0 seconds (tolerance 1e-06).
- Vehicle detection-to-trajectory matched observations: 13,226; tracked detections without trajectory rows: 0; trajectory rows without detection rows: 0.
- Missing frame slots within pedestrian track spans: 3,886; within vehicle track spans: 8,734.
- Pair-observation gaps: 3,676, with 24,784 unobserved frame slots. These are gaps in sampled outputs, not proof of object presence or detector misses.

## Interpretation and limitations

The configured `nearby` indicator is a distance threshold of 150.0 pixels. It is an experimental image-space filter, not a physical clearance or risk threshold. No Low/Medium/High categories, collision labels, meters, real-world speeds, or validated TTC values are produced.

PIE is recorded from a moving ego-vehicle camera. Ego motion, perspective, parallax, changing scale, occlusion, detector jitter, and tracker ID switches can change apparent image-space distance and direction. Vehicle annotations are not exhaustive; unmatched detections are not interpreted as false positives. A 2D footpoint/box overlap is not proof of physical contact.

The source trackers provide estimated IDs, not guaranteed persistent identities. Missing observations are counted within observed track spans only; no absent position is fabricated.

Each sample shows all available pedestrian and vehicle boxes/track IDs and pedestrian footpoints, with a distance/status line for only the nearest image-space candidate pair in that frame. This presentation does not filter the saved interaction data.

## Visual samples

- [visual_samples/interaction_frame_000000.jpg](visual_samples/interaction_frame_000000.jpg)
- [visual_samples/interaction_frame_000139.jpg](visual_samples/interaction_frame_000139.jpg)
- [visual_samples/interaction_frame_000277.jpg](visual_samples/interaction_frame_000277.jpg)
- [visual_samples/interaction_frame_000416.jpg](visual_samples/interaction_frame_000416.jpg)
- [visual_samples/interaction_frame_000554.jpg](visual_samples/interaction_frame_000554.jpg)
- [visual_samples/interaction_frame_000693.jpg](visual_samples/interaction_frame_000693.jpg)
- [visual_samples/interaction_frame_000831.jpg](visual_samples/interaction_frame_000831.jpg)
- [visual_samples/interaction_frame_000970.jpg](visual_samples/interaction_frame_000970.jpg)
- [visual_samples/interaction_frame_001119.jpg](visual_samples/interaction_frame_001119.jpg)
- [visual_samples/interaction_frame_001336.jpg](visual_samples/interaction_frame_001336.jpg)
- [visual_samples/interaction_frame_001502.jpg](visual_samples/interaction_frame_001502.jpg)
- [visual_samples/interaction_frame_001799.jpg](visual_samples/interaction_frame_001799.jpg)

## Persistent pair overview

| Set/video | Pedestrian track | Vehicle class/track | Observations | Min separation (px) | Median (px) | Movement | Closest observed frame |
|---|---|---|---:|---:|---:|---|---:|
| set01/video_0001 | 1 | car/1 | 29 | 636.56 | 666.63 | separating | 0 |
| set01/video_0001 | 1 | car/12 | 2 | 588.10 | 593.30 | indeterminate | n/a |
| set01/video_0001 | 1 | car/14 | 10 | 444.46 | 469.40 | approaching | 28 |
| set01/video_0001 | 1 | car/17 | 3 | 1053.29 | 1057.11 | separating | 26 |
| set01/video_0001 | 1 | car/2 | 29 | 0.00 | 172.02 | approaching | 23 |
| set01/video_0001 | 1 | car/3 | 26 | 1070.21 | 1231.41 | separating | 2 |
| set01/video_0001 | 1 | car/4 | 22 | 507.96 | 515.35 | separating | 7 |
| set01/video_0001 | 1 | car/5 | 6 | 226.09 | 230.41 | separating | 1 |
| set01/video_0001 | 1 | truck/12 | 2 | 575.28 | 576.88 | indeterminate | n/a |
| set01/video_0001 | 1 | truck/4 | 4 | 587.09 | 609.39 | separating | 20 |
| set01/video_0001 | 102 | car/1 | 6 | 641.88 | 649.45 | separating | 344 |
| set01/video_0001 | 102 | car/120 | 3 | 1179.43 | 1181.11 | separating | 345 |
| set01/video_0001 | 102 | car/130 | 6 | 1217.78 | 1222.34 | approaching | 352 |
| set01/video_0001 | 102 | car/179 | 4 | 1103.31 | 1118.92 | separating | 344 |
| set01/video_0001 | 102 | car/194 | 2 | 980.95 | 981.91 | indeterminate | n/a |
| set01/video_0001 | 103 | car/1 | 2 | 320.10 | 321.78 | indeterminate | n/a |
| set01/video_0001 | 103 | car/118 | 2 | 816.55 | 818.49 | indeterminate | n/a |
| set01/video_0001 | 103 | car/130 | 2 | 969.78 | 971.98 | indeterminate | n/a |
| set01/video_0001 | 103 | car/134 | 2 | 657.09 | 660.60 | indeterminate | n/a |
| set01/video_0001 | 103 | car/145 | 2 | 912.80 | 914.38 | indeterminate | n/a |

Showing 20 of 1851 persistent pairs.
