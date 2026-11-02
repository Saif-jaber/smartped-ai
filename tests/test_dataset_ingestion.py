from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smartped_ai.data.models import VideoMetadata, ValidationIssue
from smartped_ai.data.parsing import (
    compare_frame_sets,
    parse_annotated_frames,
    parse_ego_telemetry,
    parse_pedestrian_attributes,
    parse_spatial_xml,
)
from smartped_ai.data.pipeline import run_pipeline


class DatasetParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_annotated_frame_csv_parses_indices_and_reports_duplicates(self) -> None:
        path = self.root / "set01_annotated_frames.csv"
        path.write_text("video_0001,3,0,1,1\n", encoding="utf-8")
        issues: list[ValidationIssue] = []

        parsed = parse_annotated_frames(path, "set01", issues)

        self.assertEqual(parsed["video_0001"].frame_indices, [0, 1, 1])
        self.assertEqual(parsed["video_0001"].declared_count, 3)
        self.assertIn("duplicate_annotated_frame", {issue.code for issue in issues})

    def test_frame_set_comparison_reports_exact_mismatch_samples(self) -> None:
        issues: list[ValidationIssue] = []
        differences = compare_frame_sets(
            {0, 1, 4},
            {0, 1, 3},
            self.root / "video_0001_annt.xml",
            "video_0001",
            issues,
        )

        self.assertEqual(differences, {"xml_only": 1, "csv_only": 1})
        self.assertEqual(issues[0].severity, "error")
        self.assertIn("XML-only sample: [4]", issues[0].message)
        self.assertIn("CSV-only sample: [3]", issues[0].message)

    def test_spatial_xml_extracts_track_boxes_and_rejects_bad_geometry(self) -> None:
        path = self.root / "video_0001_annt.xml"
        path.write_text(
            """<annotations>
<meta><task><size>4</size><original_size><width>100</width><height>80</height>
</original_size></task></meta>
<track label="pedestrian">
<box frame="0" outside="0" occluded="0" keyframe="1" xtl="10" ytl="10" xbr="20" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">not-crossing</attribute></box>
<box frame="1" outside="0" occluded="0" keyframe="1" xtl="20" ytl="10" xbr="15" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">standing</attribute>
<attribute name="cross">crossing</attribute></box>
</track></annotations>""",
            encoding="utf-8",
        )
        issues: list[ValidationIssue] = []
        output = self.root / "records.jsonl"

        with output.open("w", encoding="utf-8") as destination:
            summary = parse_spatial_xml(
                path, "set01", "video_0001", destination, issues
            )

        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(summary.task_size, 4)
        self.assertEqual(summary.image_width, 100)
        self.assertEqual(summary.visible_frames, {0, 1})
        self.assertEqual(summary.track_ids["pedestrian"], {"p1"})
        self.assertEqual(records[0]["bbox_xyxy"], {"x1": 10.0, "y1": 10.0, "x2": 20.0, "y2": 40.0})
        self.assertIn("nonpositive_bbox", {issue.code for issue in issues})

    def test_attribute_and_ego_xml_are_joinable_by_track_and_frame(self) -> None:
        attributes_path = self.root / "attributes.xml"
        attributes_path.write_text(
            '<ped_attributes><pedestrian id="p1" crossing="1" intention_prob="0.75" '
            'age="adult" /></ped_attributes>',
            encoding="utf-8",
        )
        telemetry_path = self.root / "ego.xml"
        telemetry_path.write_text(
            '<vehicle_info><frame id="0" GPS_speed="12.5" yaw="1.0" />'
            '<frame id="1" GPS_speed="13.0" yaw="1.1" /></vehicle_info>',
            encoding="utf-8",
        )
        issues: list[ValidationIssue] = []
        attributes_output = self.root / "attributes.jsonl"
        telemetry_output = self.root / "telemetry.jsonl"

        with attributes_output.open("w", encoding="utf-8") as destination:
            ids, outcomes, probabilities = parse_pedestrian_attributes(
                attributes_path, "set01", "video_0001", destination, issues
            )
        with telemetry_output.open("w", encoding="utf-8") as destination:
            frame_ids, summary = parse_ego_telemetry(
                telemetry_path, "set01", "video_0001", destination, issues
            )

        attribute_record = json.loads(attributes_output.read_text(encoding="utf-8"))
        telemetry_records = [
            json.loads(line)
            for line in telemetry_output.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(ids, {"p1"})
        self.assertEqual(outcomes, {"1": 1})
        self.assertEqual(probabilities, [0.75])
        self.assertEqual(attribute_record["track_id"], "p1")
        self.assertEqual(frame_ids, {0, 1})
        self.assertEqual(summary["unique_frames"], 2)
        self.assertEqual(telemetry_records[1]["frame_index"], 1)
        self.assertEqual(telemetry_records[1]["fields"]["GPS_speed"], 13.0)
        self.assertEqual(issues, [])

    def test_pipeline_writes_manifest_split_and_streamed_records(self) -> None:
        dataset = self.root / "dataset"
        set_id = "set01"
        video_id = "video_0001"
        (dataset / "PIE_clips" / set_id).mkdir(parents=True)
        (dataset / "annotations" / set_id).mkdir(parents=True)
        (dataset / "annotations_attributes" / set_id).mkdir(parents=True)
        (dataset / "annotations_vehicle" / set_id).mkdir(parents=True)
        (dataset / "PIE_clips" / set_id / f"{video_id}.mp4").write_bytes(b"fixture")
        (dataset / "annotations" / set_id / f"{set_id}_annotated_frames.csv").write_text(
            f"{video_id},2,0,1\n", encoding="utf-8"
        )
        (dataset / "annotations" / set_id / f"{video_id}_annt.xml").write_text(
            """<annotations><meta><task><size>4</size><original_size><width>100</width>
<height>80</height></original_size></task></meta><track label="pedestrian">
<box frame="0" outside="0" occluded="0" keyframe="1" xtl="10" ytl="10" xbr="20" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">not-crossing</attribute></box>
<box frame="1" outside="0" occluded="0" keyframe="1" xtl="11" ytl="10" xbr="21" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">not-crossing</attribute></box></track></annotations>""",
            encoding="utf-8",
        )
        (dataset / "annotations_attributes" / set_id / f"{video_id}_attributes.xml").write_text(
            '<ped_attributes><pedestrian id="p1" crossing="0" intention_prob="0.25" '
            'age="adult" /></ped_attributes>',
            encoding="utf-8",
        )
        (dataset / "annotations_vehicle" / set_id / f"{video_id}_obd.xml").write_text(
            "<vehicle_info>"
            + "".join(f'<frame id="{index}" GPS_speed="0.0" />' for index in range(4))
            + "</vehicle_info>",
            encoding="utf-8",
        )
        configs = self.root / "configs"
        configs.mkdir()
        config_path = configs / "pie_dataset.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_root": "../dataset",
                    "output_dir": "../artifacts",
                    "split_assignments": {
                        "train": ["set01"],
                        "validation": [],
                        "test": [],
                    },
                }
            ),
            encoding="utf-8",
        )
        metadata = VideoMetadata(1920, 1080, 30.0, 4, 4 / 30, "ffprobe")

        with patch(
            "smartped_ai.data.pipeline.probe_video", return_value=(metadata, None)
        ):
            result = run_pipeline(config_path)

        output_dir = self.root / "artifacts"
        manifest = json.loads((output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        split_manifest = json.loads((output_dir / "split_manifest.json").read_text(encoding="utf-8"))
        issues = json.loads((output_dir / "validation_issues.json").read_text(encoding="utf-8"))
        with gzip.open(output_dir / "object_annotations.jsonl.gz", "rt", encoding="utf-8") as source:
            object_records = [json.loads(line) for line in source]

        self.assertEqual(result["error_count"], 0)
        self.assertEqual(result["statistics"]["pedestrian_tracks"], 1)
        self.assertEqual(manifest["videos"][0]["paths"]["video"], "PIE_clips/set01/video_0001.mp4")
        self.assertEqual(split_manifest["assignments"]["train"], ["set01"])
        self.assertEqual(issues["issues"], [])
        self.assertEqual(len(object_records), 2)
        self.assertTrue((output_dir / "validation_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
