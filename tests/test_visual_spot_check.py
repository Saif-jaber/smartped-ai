from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from smartped_ai.inspection.visual_spot_check import (
    _read_csv_frames,
    _read_spatial_xml,
    _select_category_frames,
    record_review,
)


class VisualSpotCheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_spatial_xml_reads_selected_boxes_and_all_visible_frame_indices(self) -> None:
        path = self.root / "video_0001_annt.xml"
        path.write_text(
            """<annotations>
<meta><task><original_size><width>100</width><height>80</height>
</original_size></task></meta>
<track label="pedestrian">
<box frame="0" outside="0" occluded="0" xtl="10" ytl="10" xbr="20" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">crossing</attribute></box>
</track>
<track label="vehicle">
<box frame="1" outside="0" occluded="1" xtl="30" ytl="20" xbr="60" ybr="50">
<attribute name="id">v1</attribute></box>
</track></annotations>""",
            encoding="utf-8",
        )

        result = _read_spatial_xml(path, {1})

        self.assertEqual(result.visible_frames, {0, 1})
        self.assertEqual(result.image_width, 100)
        self.assertEqual(result.image_height, 80)
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0].category, "vehicle")
        self.assertEqual(result.boxes[0].track_id, "v1")
        self.assertTrue(result.boxes[0].occluded)

    def test_csv_frame_reader_returns_the_video_frame_indices(self) -> None:
        path = self.root / "set01_annotated_frames.csv"
        path.write_text("video_0001,3,2,4,8\n", encoding="utf-8")

        self.assertEqual(_read_csv_frames(path, "video_0001"), {2, 4, 8})

    def test_category_samples_require_alignment_and_cover_multiple_objects(self) -> None:
        path = self.root / "objects.jsonl.gz"
        records = [
            {
                "set_id": "set01",
                "video_id": "video_0001",
                "frame_index": 0,
                "object_category": "pedestrian",
                "track_id": "p1",
                "bbox_xyxy": {"x1": 1, "y1": 1, "x2": 5, "y2": 10},
                "outside": False,
                "occluded": "0",
                "attributes": {"action": "walking", "cross": "crossing"},
            },
            {
                "set_id": "set01",
                "video_id": "video_0001",
                "frame_index": 0,
                "object_category": "pedestrian",
                "track_id": "p2",
                "bbox_xyxy": {"x1": 6, "y1": 1, "x2": 10, "y2": 10},
                "outside": False,
                "occluded": "1",
                "attributes": {"action": "walking", "cross": "crossing"},
            },
            {
                "set_id": "set01",
                "video_id": "video_0001",
                "frame_index": 0,
                "object_category": "vehicle",
                "track_id": "v1",
                "bbox_xyxy": {"x1": 11, "y1": 1, "x2": 20, "y2": 10},
                "outside": False,
                "occluded": "0",
                "attributes": {},
            },
            {
                "set_id": "set01",
                "video_id": "video_0001",
                "frame_index": 9,
                "object_category": "pedestrian",
                "track_id": "p3",
                "bbox_xyxy": {"x1": 1, "y1": 1, "x2": 5, "y2": 10},
                "outside": False,
                "occluded": "0",
                "attributes": {"action": "standing", "cross": "not-crossing"},
            },
        ]
        with gzip.open(path, "wt", encoding="utf-8") as destination:
            for record in records:
                destination.write(json.dumps(record) + "\n")

        selected = _select_category_frames(
            path, {("set01", "video_0001"): {0}}
        )

        self.assertEqual(
            selected["multiple_pedestrians_and_vehicles"],
            ("set01", "video_0001", 0),
        )
        self.assertEqual(selected["walking"], ("set01", "video_0001", 0))
        self.assertEqual(selected["crossing"], ("set01", "video_0001", 0))
        self.assertEqual(selected["occluded_or_partial"], ("set01", "video_0001", 0))
        self.assertNotIn("standing", selected)

    def test_record_review_updates_manifest_and_report_without_changing_alignment(self) -> None:
        output_dir = self.root / "visual_spot_check"
        output_dir.mkdir()
        item = {
            "set_id": "set06",
            "video_id": "video_0002",
            "frame_index": 5260,
            "sample_categories": ["reported_xml_only_mismatch"],
            "image": "set06_video_0002_frame_005260.jpg",
            "alignment": "XML-only",
            "visible_box_count": 1,
            "missing_bbox_count": 0,
            "invalid_bbox_count": 0,
            "out_of_image_bbox_count": 0,
            "missing_id_count": 0,
            "duplicate_id_count": 0,
            "pedestrian_label_gaps": 0,
            "visual_status": "pending_manual_review",
            "finding": "Initial mismatch note",
            "likely_cause": "Initial likely cause",
        }
        manifest_path = output_dir / "sample_manifest.json"
        manifest_path.write_text(json.dumps([item]), encoding="utf-8")

        record_review(
            output_dir,
            "set06",
            "video_0002",
            5260,
            "uncertain",
            "Vehicle box is plausible but needs closer review.",
            "CSV may exclude vehicle-only frames.",
        )

        updated = json.loads(manifest_path.read_text(encoding="utf-8"))[0]
        self.assertEqual(updated["visual_status"], "reviewed_uncertain")
        self.assertEqual(updated["alignment"], "XML-only")
        self.assertIn("reviewed_uncertain", (output_dir / "visual_spot_check_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
