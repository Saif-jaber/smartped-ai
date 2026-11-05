from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from smartped_ai.vehicle_tracking.pipeline import (
    VehicleAnnotation,
    VehicleDetection,
    extract_trajectory_rows,
    match_annotations,
    parse_result_detections,
    read_vehicle_annotations,
)


class _FakeBoxes(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.cls)


class VehicleDetectionParsingTests(unittest.TestCase):
    def test_only_configured_coco_vehicle_classes_are_kept(self) -> None:
        boxes = _FakeBoxes(
            cls=[0, 1, 2, 3, 5, 7, 9],
            xyxy=[
                [0, 0, 5, 5],
                [1, 2, 11, 12],
                [2, 3, 12, 13],
                [3, 4, 13, 14],
                [4, 5, 14, 15],
                [5, 6, 15, 16],
                [6, 7, 16, 17],
            ],
            conf=[0.99, 0.81, 0.82, 0.83, 0.85, 0.87, 0.99],
            id=[10, 11, 12, 13, 14, 15, 16],
        )
        result = SimpleNamespace(boxes=boxes)

        detections = parse_result_detections(
            result,
            frame_index=30,
            fps=30.0,
            coco_classes={
                "bicycle": 1,
                "car": 2,
                "motorcycle": 3,
                "bus": 5,
                "truck": 7,
            },
        )

        self.assertEqual(
            [detection.class_name for detection in detections],
            ["bicycle", "car", "motorcycle", "bus", "truck"],
        )
        self.assertEqual(detections[0].track_id, "11")
        self.assertEqual(detections[0].frame_index, 30)
        self.assertEqual(detections[0].timestamp_seconds, 1.0)
        self.assertEqual(detections[0].bbox_xyxy, (1.0, 2.0, 11.0, 12.0))

    def test_tracked_outputs_without_ids_remain_detections(self) -> None:
        result = SimpleNamespace(
            boxes=_FakeBoxes(
                cls=[2],
                xyxy=[[1, 2, 10, 20]],
                conf=[0.7],
                id=None,
            )
        )

        detections = parse_result_detections(
            result, frame_index=4, fps=30, coco_classes={"car": 2}
        )

        self.assertEqual(len(detections), 1)
        self.assertIsNone(detections[0].track_id)


class VehicleTrajectoryTests(unittest.TestCase):
    def test_track_gaps_are_reported_without_interpolating_observations(self) -> None:
        detections = [
            VehicleDetection(
                0, 0.0, 2, "car", 0.8, (0.0, 0.0, 10.0, 10.0), "7"
            ),
            VehicleDetection(
                1, 1 / 30, 2, "car", 0.9, (2.0, 0.0, 12.0, 10.0), "7"
            ),
            VehicleDetection(
                20, 20 / 30, 2, "car", 0.7, (22.0, 0.0, 32.0, 10.0), "7"
            ),
        ]

        rows = extract_trajectory_rows(
            detections,
            fps=30.0,
            max_track_gap_frames=15,
            set_id="set01",
            video_id="video_0001",
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["set_id"], "set01")
        self.assertEqual(rows[0]["video_id"], "video_0001")
        self.assertEqual(rows[0]["center_x"], 5.0)
        self.assertIsNone(rows[0]["displacement_px"])
        self.assertEqual(rows[1]["displacement_x_px"], 2.0)
        self.assertAlmostEqual(rows[1]["image_speed_px_s"] or 0.0, 60.0)
        self.assertEqual(rows[2]["frame_gap"], 19)
        self.assertEqual(rows[2]["missing_frames_between_observations"], 18)
        self.assertTrue(rows[2]["gap_exceeded"])
        self.assertIsNone(rows[2]["image_speed_px_s"])


class VehicleAnnotationMatchingTests(unittest.TestCase):
    def test_matches_one_to_one_by_frame_and_mapped_class(self) -> None:
        detections = [
            VehicleDetection(
                10, 1 / 3, 2, "car", 0.8, (0.0, 0.0, 10.0, 10.0), "p1"
            ),
            VehicleDetection(
                10, 1 / 3, 1, "bicycle", 0.7, (20.0, 20.0, 30.0, 30.0), "p2"
            ),
        ]
        annotations = [
            VehicleAnnotation(
                10, "g1", "car", "car", (0.0, 0.0, 10.0, 10.0), False, True
            ),
            VehicleAnnotation(
                10, "g2", "bike", None, (20.0, 20.0, 30.0, 30.0), False, False
            ),
        ]

        rows, metrics = match_annotations(
            detections, annotations, iou_threshold=0.5
        )

        self.assertEqual(metrics["eligible_annotation_boxes"], 1)
        self.assertEqual(metrics["matched_annotation_boxes"], 1)
        self.assertEqual(metrics["annotation_match_rate"], 1.0)
        self.assertEqual(metrics["mean_iou_matched_pairs"], 1.0)
        self.assertEqual(sum(row["matched"] for row in rows), 1)

    def test_vehicle_annotation_xml_keeps_pie_bike_unmapped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video_0001_annt.xml"
            path.write_text(
                """<annotations><meta><task><original_size><width>100</width>
<height>80</height></original_size></task></meta>
<track label="vehicle">
<box frame="0" outside="0" occluded="1" keyframe="1"
xtl="10" ytl="10" xbr="30" ybr="40">
<attribute name="id">v1</attribute><attribute name="type">bike</attribute></box>
<box frame="1" outside="1" occluded="0" keyframe="0">
<attribute name="id">v1</attribute><attribute name="type">bike</attribute></box>
</track></annotations>""",
                encoding="utf-8",
            )

            annotations = read_vehicle_annotations(
                path,
                frame_limit=2,
                pie_type_to_coco_class={"bike": None},
            )

        self.assertEqual(len(annotations), 2)
        self.assertIsNone(annotations[0].class_name)
        self.assertTrue(annotations[0].occluded)
        self.assertTrue(annotations[1].outside)
        self.assertIsNone(annotations[1].bbox_xyxy)


if __name__ == "__main__":
    unittest.main()
