from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from smartped_ai.trajectory.features import extract_track_features, summarize_tracks
from smartped_ai.trajectory.models import (
    MOVEMENT_STANDING,
    MOVEMENT_UNKNOWN,
    MOVEMENT_WALKING,
    TrajectoryConfig,
    TrackObservation,
)
from smartped_ai.trajectory.pipeline import run_extraction
from smartped_ai.trajectory.readers import load_observations


def _observation(
    track_id: str,
    frame_index: int,
    x: float,
    y: float = 0.0,
    *,
    video_id: str = "video_0001",
    set_id: str | None = "set01",
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        frame_index=frame_index,
        bbox_xyxy=(x, y, x + 10.0, y + 20.0),
        set_id=set_id,
        video_id=video_id,
    )


def _config(**overrides) -> TrajectoryConfig:
    values = {
        "fps": 30.0,
        "max_gap_frames": 5,
        "standing_speed_threshold_px_s": 10.0,
        "smoothing_window_observations": 5,
        "smoothing_min_observations": 3,
    }
    values.update(overrides)
    return TrajectoryConfig(**values)


class TrajectoryFeatureTests(unittest.TestCase):
    def test_constant_motion_calculates_displacement_velocity_and_direction(self) -> None:
        features = extract_track_features(
            [
                _observation("p1", 0, 0.0),
                _observation("p1", 1, 10.0),
                _observation("p1", 2, 20.0),
            ],
            _config(),
        )

        self.assertEqual(features[0].smoothed_movement_state, MOVEMENT_UNKNOWN)
        self.assertFalse(features[0].velocity_valid)
        self.assertEqual(features[1].frame_gap, 1)
        self.assertAlmostEqual(features[1].time_gap_seconds or 0.0, 1 / 30)
        self.assertAlmostEqual(features[1].displacement_x_px or 0.0, 10.0)
        self.assertAlmostEqual(features[1].speed_px_s or 0.0, 300.0)
        self.assertAlmostEqual(features[1].velocity_x_px_s or 0.0, 300.0)
        self.assertAlmostEqual(features[1].velocity_y_px_s or 0.0, 0.0)
        self.assertAlmostEqual(features[1].direction_deg or 0.0, 0.0)
        self.assertEqual(features[1].direction_label, "right")
        self.assertEqual(features[1].raw_movement_state, MOVEMENT_WALKING)
        self.assertEqual(features[1].smoothed_movement_state, MOVEMENT_WALKING)

    def test_short_gap_is_used_but_long_gap_has_unknown_state(self) -> None:
        features = extract_track_features(
            [
                _observation("p1", 0, 0.0),
                _observation("p1", 1, 10.0),
                _observation("p1", 4, 40.0),
                _observation("p1", 10, 40.0),
                _observation("p1", 11, 50.0),
            ],
            _config(max_gap_frames=5),
        )

        short_gap = features[2]
        self.assertEqual(short_gap.frame_gap, 3)
        self.assertTrue(short_gap.velocity_valid)
        self.assertAlmostEqual(short_gap.speed_px_s or 0.0, 300.0)

        long_gap = features[3]
        self.assertEqual(long_gap.frame_gap, 6)
        self.assertTrue(long_gap.gap_exceeded)
        self.assertFalse(long_gap.velocity_valid)
        self.assertEqual(long_gap.velocity_invalid_reason, "frame_gap_exceeded")
        self.assertIsNone(long_gap.speed_px_s)
        self.assertIsNone(long_gap.raw_movement_state)
        self.assertEqual(long_gap.smoothed_movement_state, MOVEMENT_UNKNOWN)
        self.assertNotEqual(long_gap.smoothed_movement_state, MOVEMENT_STANDING)

        after_gap = features[4]
        self.assertTrue(after_gap.velocity_valid)
        self.assertEqual(after_gap.smoothed_movement_state, MOVEMENT_WALKING)

    def test_direction_uses_image_axes_with_y_pointing_down(self) -> None:
        observations = [
            _observation("right", 0, 0.0, 0.0),
            _observation("right", 1, 10.0, 0.0),
            _observation("down", 0, 0.0, 0.0),
            _observation("down", 1, 0.0, 10.0),
            _observation("left", 0, 10.0, 0.0),
            _observation("left", 1, 0.0, 0.0),
            _observation("up", 0, 0.0, 10.0),
            _observation("up", 1, 0.0, 0.0),
        ]

        features = extract_track_features(observations, _config())
        by_track = {
            feature.track_id: feature
            for feature in features
            if feature.observation_index == 1
        }

        self.assertAlmostEqual(by_track["right"].direction_deg or 0.0, 0.0)
        self.assertEqual(by_track["right"].direction_label, "right")
        self.assertAlmostEqual(by_track["down"].direction_deg or 0.0, 90.0)
        self.assertEqual(by_track["down"].direction_label, "down")
        self.assertAlmostEqual(by_track["left"].direction_deg or 0.0, 180.0)
        self.assertEqual(by_track["left"].direction_label, "left")
        self.assertAlmostEqual(by_track["up"].direction_deg or 0.0, 270.0)
        self.assertEqual(by_track["up"].direction_label, "up")

    def test_stationary_jitter_is_smoothed_and_a_spike_does_not_flip_state(self) -> None:
        features = extract_track_features(
            [
                _observation("p1", 0, 0.0),
                _observation("p1", 1, 0.1),
                _observation("p1", 2, 0.2),
                _observation("p1", 3, 50.0),
                _observation("p1", 4, 50.1),
            ],
            _config(),
        )

        self.assertEqual(features[1].raw_movement_state, MOVEMENT_STANDING)
        self.assertEqual(features[3].raw_movement_state, MOVEMENT_WALKING)
        self.assertEqual(features[3].smoothed_movement_state, MOVEMENT_STANDING)
        self.assertEqual(features[4].smoothed_movement_state, MOVEMENT_STANDING)

    def test_track_ids_are_preserved_and_not_merged_across_videos(self) -> None:
        features = extract_track_features(
            [
                _observation("1", 0, 0.0, video_id="video_0001"),
                _observation("1", 1, 10.0, video_id="video_0001"),
                _observation("1", 0, 100.0, video_id="video_0002"),
                _observation("1", 1, 90.0, video_id="video_0002"),
                _observation("2", 0, 0.0, video_id="video_0001"),
                _observation("2", 1, 0.0, video_id="video_0001"),
            ],
            _config(),
        )

        summaries = summarize_tracks(features)
        identities = {
            (summary.video_id, summary.track_id): summary for summary in summaries
        }

        self.assertEqual(len(summaries), 3)
        self.assertIn(("video_0001", "1"), identities)
        self.assertIn(("video_0002", "1"), identities)
        self.assertIn(("video_0001", "2"), identities)
        self.assertEqual(identities[("video_0001", "1")].observation_count, 2)
        self.assertEqual(identities[("video_0002", "1")].observation_count, 2)


class TrajectoryInputExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_config(self) -> Path:
        path = self.root / "trajectory.json"
        path.write_text(
            json.dumps(
                {
                    "fps": 30.0,
                    "max_gap_frames": 5,
                    "standing_speed_threshold_px_s": 10.0,
                    "smoothing_window_observations": 5,
                    "smoothing_min_observations": 3,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_jsonl_input_exports_csv_jsonl_summary_and_report(self) -> None:
        source = self.root / "tracks.jsonl"
        records = [
            {
                "set_id": "set01",
                "video_id": "video_0001",
                "track_id": "p1",
                "frame_index": frame,
                "bbox_xyxy": [frame * 10.0, 0.0, frame * 10.0 + 10.0, 20.0],
            }
            for frame in range(3)
        ]
        source.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        result = run_extraction(
            source,
            config_path=self._write_config(),
            output_dir=self.root / "out",
        )

        with result["csv_path"].open("r", encoding="utf-8", newline="") as table:
            rows = list(csv.DictReader(table))
        jsonl_rows = [
            json.loads(line)
            for line in result["jsonl_path"].read_text(encoding="utf-8").splitlines()
        ]
        summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
        report = result["report_path"].read_text(encoding="utf-8")

        self.assertEqual(len(rows), 3)
        self.assertEqual(len(jsonl_rows), 3)
        self.assertEqual(rows[1]["track_id"], "p1")
        self.assertEqual(rows[1]["raw_movement_state"], MOVEMENT_WALKING)
        self.assertAlmostEqual(float(rows[1]["speed_px_s"]), 300.0)
        self.assertEqual(jsonl_rows[1]["track_id"], "p1")
        self.assertTrue(jsonl_rows[1]["velocity_valid"])
        self.assertEqual(summary["counts"]["tracks"], 1)
        self.assertEqual(summary["counts"]["observations"], 3)
        self.assertEqual(summary["tracks"][0]["track_id"], "p1")
        self.assertIn("not real-world speed", report)

    def test_csv_reader_accepts_xyxy_and_top_left_xywh(self) -> None:
        source = self.root / "tracks.csv"
        source.write_text(
            "frame_index,track_id,x1,y1,x2,y2\n"
            "0,p1,10,20,30,60\n"
            "1,p1,20,20,40,60\n"
            "0,p2,5,5,15,25\n",
            encoding="utf-8",
        )
        config = _config()

        observations, issues, resolved_format = load_observations(source, config)

        self.assertEqual(resolved_format, "csv")
        self.assertEqual(len(observations), 3)
        self.assertEqual(observations[0].bbox_xyxy, (10.0, 20.0, 30.0, 60.0))
        self.assertEqual(issues, [])

        xywh = self.root / "tracks_xywh.csv"
        xywh.write_text(
            "frame,track_id,x,y,w,h\n0,p1,10,20,20,40\n1,p1,20,20,20,40\n",
            encoding="utf-8",
        )
        observations, issues, _ = load_observations(xywh, config)
        self.assertEqual(observations[0].bbox_xyxy, (10.0, 20.0, 30.0, 60.0))
        self.assertEqual(issues, [])

    def test_missing_track_id_is_skipped_without_inventing_identity(self) -> None:
        source = self.root / "tracks.jsonl"
        source.write_text(
            json.dumps(
                {
                    "frame_index": 0,
                    "bbox_xyxy": [0, 0, 10, 20],
                }
            )
            + "\n"
            + json.dumps(
                {
                    "track_id": "p1",
                    "frame_index": 1,
                    "bbox_xyxy": [10, 0, 20, 20],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        observations, issues, _ = load_observations(source, _config())

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].track_id, "p1")
        self.assertIn("missing_track_id", {issue.code for issue in issues})

    def test_pie_xml_reader_skips_outside_boxes_and_preserves_track_id(self) -> None:
        source = self.root / "video_0001_annt.xml"
        source.write_text(
            """<annotations><meta><task><original_size><width>100</width>
<height>80</height></original_size></task></meta>
<track label="pedestrian">
<box frame="0" outside="0" occluded="0" xtl="10" ytl="10" xbr="20" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">not-crossing</attribute></box>
<box frame="1" outside="1" occluded="0" />
<box frame="2" outside="0" occluded="0" xtl="20" ytl="10" xbr="30" ybr="40">
<attribute name="id">p1</attribute><attribute name="action">walking</attribute>
<attribute name="cross">not-crossing</attribute></box>
</track></annotations>""",
            encoding="utf-8",
        )

        observations, issues, resolved_format = load_observations(
            source, _config(), input_format="pie-xml"
        )

        self.assertEqual(resolved_format, "pie-xml")
        self.assertEqual([item.frame_index for item in observations], [0, 2])
        self.assertEqual({item.track_id for item in observations}, {"p1"})
        self.assertEqual(issues, [])

    def test_duplicate_track_frame_is_rejected_instead_of_reassociated(self) -> None:
        source = self.root / "tracks.jsonl"
        records = [
            {"track_id": "p1", "frame_index": 0, "bbox_xyxy": [0, 0, 10, 20]},
            {"track_id": "p1", "frame_index": 0, "bbox_xyxy": [1, 0, 11, 20]},
        ]
        source.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Duplicate observation"):
            load_observations(source, _config())


if __name__ == "__main__":
    unittest.main()
