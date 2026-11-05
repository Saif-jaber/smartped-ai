from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from smartped_ai.data.pipeline import DatasetConfig
from smartped_ai.data.prepare_detection import (
    PedestrianBox,
    _metadata_record,
    _split_mapping,
    bbox_to_yolo,
    read_pedestrian_xml,
    run_preparation,
    select_frame_indices,
    validate_bbox,
)


class DetectionPreparationTests(unittest.TestCase):
    def test_bbox_to_yolo_normalizes_xyxy_coordinates(self) -> None:
        normalized = bbox_to_yolo((10.0, 20.0, 30.0, 60.0), 100, 100)

        self.assertEqual(normalized, (0.2, 0.4, 0.2, 0.4))

    def test_invalid_boxes_are_rejected_without_clipping(self) -> None:
        self.assertEqual(
            validate_bbox((20.0, 10.0, 20.0, 30.0), 100, 80),
            "non_positive_area",
        )
        self.assertEqual(
            validate_bbox((-1.0, 10.0, 20.0, 30.0), 100, 80),
            "outside_image_bounds",
        )
        self.assertEqual(
            validate_bbox((None, 10.0, 20.0, 30.0), 100, 80),
            "missing_or_non_finite_coordinate",
        )
        self.assertEqual(
            validate_bbox((float("nan"), 10.0, 20.0, 30.0), 100, 80),
            "non_finite_coordinate",
        )
        with self.assertRaisesRegex(ValueError, "outside_image_bounds"):
            bbox_to_yolo((-1.0, 10.0, 20.0, 30.0), 100, 80)

    def test_spatial_xml_preserves_outside_occlusion_and_keyframe_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video_annt.xml"
            path.write_text(
                """<annotations><meta><task><original_size><width>100</width>
<height>80</height></original_size></task></meta>
<track label="pedestrian">
<box frame="0" outside="0" occluded="1" keyframe="1" xtl="10" ytl="10" xbr="20" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">crossing</attribute><attribute name="occlusion">partial</attribute></box>
<box frame="1" outside="1" occluded="0" keyframe="0" />
</track>
<track label="vehicle"><box frame="0" outside="0" xtl="1" ytl="1" xbr="8" ybr="8" /></track>
</annotations>""",
                encoding="utf-8",
            )

            width, height, boxes = read_pedestrian_xml(path)

        self.assertEqual((width, height), (100, 80))
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].track_id, "p1")
        self.assertEqual(boxes[0].attributes["cross"], "crossing")
        self.assertTrue(boxes[0].keyframe)
        self.assertEqual(boxes[0].occluded_raw, "1")
        self.assertTrue(boxes[1].outside)

    def test_outside_and_occluded_boxes_have_distinct_target_policies(self) -> None:
        occluded = PedestrianBox(
            5, "p1", (1.0, 2.0, 10.0, 20.0), ("1", "2", "10", "20"),
            False, "1", True, {"id": "p1", "action": "walking"},
        )
        outside = PedestrianBox(
            6, "p1", (None, None, None, None), (None, None, None, None),
            True, "0", False, {"id": "p1"},
        )

        self.assertIsNone(validate_bbox(occluded.bbox_xyxy, 100, 80))
        self.assertEqual(validate_bbox(outside.bbox_xyxy, 100, 80), "missing_or_non_finite_coordinate")
        record = _metadata_record(occluded, "set01", "video_0001", "train", 100, 80, {5})
        outside_record = _metadata_record(outside, "set01", "video_0001", "train", 100, 80, {6})
        self.assertTrue(record["label_included"])
        self.assertTrue(record["occluded"])
        self.assertEqual(record["action"], "walking")
        self.assertFalse(outside_record["label_included"])
        self.assertEqual(outside_record["exclusion_reason"], "outside")

    def test_sampling_keeps_stride_track_boundaries_and_state_transitions(self) -> None:
        boxes = [
            PedestrianBox(
                frame, "p1", (1.0, 1.0, 10.0, 20.0),
                ("1", "1", "10", "20"), False, "0", frame == 5,
                {"id": "p1", "action": "walking" if frame < 4 else "standing", "cross": "not-crossing"},
            )
            for frame in range(1, 7)
        ]

        selected, _ = select_frame_indices(boxes, 3, 100, 80)
        with_keyframes, _ = select_frame_indices(boxes, 3, 100, 80, preserve_keyframes=True)

        self.assertEqual(set(selected), {1, 3, 4, 6})
        self.assertIn("label_or_visibility_transition", selected[4])
        self.assertIn("keyframe", with_keyframes[5])

    def test_split_mapping_rejects_leakage_and_unassigned_sets(self) -> None:
        config = DatasetConfig(
            config_path=Path("config.json"),
            dataset_root=Path("dataset"),
            output_dir=Path("output"),
            video_dir="PIE_clips",
            spatial_dir="annotations",
            attributes_dir="annotations_attributes",
            telemetry_dir="annotations_vehicle",
            ffprobe_executable="ffprobe",
            ffprobe_timeout_seconds=20,
            split_assignments={
                "train": ["set01"],
                "validation": ["set02"],
                "test": [],
            },
        )
        mapping = _split_mapping(config, {"set01", "set02"})
        self.assertEqual(mapping, {"set01": "train", "set02": "validation"})
        config.split_assignments["test"] = ["set02"]
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            _split_mapping(config, {"set01", "set02"})
        config.split_assignments["test"] = []
        with self.assertRaisesRegex(ValueError, "no split assignment"):
            _split_mapping(config, {"set01", "set03"})

    def test_end_to_end_output_keeps_occlusion_and_records_exclusions(self) -> None:
        class Frame:
            shape = (80, 100, 3)

            def copy(self):
                return Frame()

        class Capture:
            def __init__(self, _path):
                self.frame_index = 0

            def isOpened(self):
                return True

            def set(self, _property, frame_index):
                self.frame_index = int(frame_index)

            def read(self):
                return True, Frame()

            def release(self):
                pass

        class FakeCV2:
            CAP_PROP_POS_FRAMES = 1
            IMWRITE_JPEG_QUALITY = 2
            FONT_HERSHEY_SIMPLEX = 0
            LINE_AA = 1

            @staticmethod
            def VideoCapture(path):
                return Capture(path)

            @staticmethod
            def imwrite(path, _frame, _params):
                Path(path).write_bytes(b"image")
                return True

            @staticmethod
            def rectangle(*_args):
                pass

            @staticmethod
            def putText(*_args):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            (dataset / "PIE_clips" / "set01").mkdir(parents=True)
            (dataset / "annotations" / "set01").mkdir(parents=True)
            (dataset / "PIE_clips" / "set01" / "video_0001.mp4").write_bytes(b"fixture")
            (dataset / "annotations" / "set01" / "set01_annotated_frames.csv").write_text(
                "video_0001,3,1,2,4\n", encoding="utf-8"
            )
            (dataset / "annotations" / "set01" / "video_0001_annt.xml").write_text(
                """<annotations><meta><task><original_size><width>100</width>
<height>80</height></original_size></task></meta>
<track label="pedestrian">
<box frame="1" outside="0" occluded="0" keyframe="1" xtl="10" ytl="10" xbr="20" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">not-crossing</attribute></box>
<box frame="2" outside="0" occluded="1" keyframe="0" xtl="12" ytl="10" xbr="22" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">standing</attribute>
<attribute name="cross">crossing</attribute><attribute name="occlusion">partial</attribute></box>
<box frame="3" outside="1" occluded="0" keyframe="0" />
<box frame="4" outside="0" occluded="0" keyframe="0" xtl="30" ytl="20" xbr="20" ybr="40">
<attribute name="id">p1</attribute></box>
</track></annotations>""",
                encoding="utf-8",
            )
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "pie_dataset.json").write_text(
                json.dumps(
                    {
                        "dataset_root": "../dataset",
                        "output_dir": "../artifacts/validation",
                        "split_assignments": {
                            "train": ["set01"],
                            "validation": [],
                            "test": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (config_dir / "prepare.json").write_text(
                json.dumps(
                    {
                        "dataset_config": "pie_dataset.json",
                        "output_dir": "../artifacts/prepared",
                        "frame_stride": 30,
                        "visual_sample_count": 1,
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "smartped_ai.data.prepare_detection._require_cv2",
                return_value=FakeCV2,
            ):
                result = run_preparation(config_dir / "prepare.json")

            output = result["output_dir"]
            summary = result["summary"]
            with gzip.open(output / "annotation_metadata.jsonl.gz", "rt", encoding="utf-8") as source:
                metadata = [json.loads(line) for line in source]
            labels = list((output / "labels" / "train" / "set01" / "video_0001").glob("*.txt"))
            visual_sample_count = len(list((output / "visual_samples").rglob("*.jpg")))
            has_data_yaml = (output / "data.yaml").is_file()

        self.assertEqual(summary["images"], 2)
        self.assertEqual(summary["labels"], 2)
        self.assertEqual(summary["outside_boxes_excluded"], 1)
        self.assertEqual(summary["invalid_boxes_excluded"], 1)
        self.assertEqual(summary["csv_crosscheck"]["xml_only_frame_count"], 0)
        self.assertEqual(summary["csv_crosscheck"]["csv_only_frame_count"], 0)
        self.assertEqual(len(labels), 2)
        self.assertTrue(any(row["occluded"] and row["label_included"] for row in metadata))
        self.assertTrue(any(row["exclusion_reason"] == "outside" for row in metadata))
        self.assertGreater(visual_sample_count, 0)
        self.assertTrue(has_data_yaml)


if __name__ == "__main__":
    unittest.main()
