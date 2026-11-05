from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from smartped_ai.trajectory.comparison import compare_predictions_to_pie
from smartped_ai.trajectory.features import extract_track_features, summarize_tracks
from smartped_ai.trajectory.models import TrajectoryConfig, TrackObservation
from smartped_ai.trajectory.pipeline import run_extraction
from smartped_ai.trajectory.readers import load_observations


def _observation(
    track_id: str,
    frame_index: int,
    x: float,
    y: float = 0.0,
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        frame_index=frame_index,
        bbox_xyxy=(x, y, x + 10.0, y + 20.0),
        set_id="set01",
        video_id="video_0001",
    )


def _config() -> TrajectoryConfig:
    return TrajectoryConfig(
        fps=30.0,
        max_gap_frames=5,
        standing_speed_threshold_px_s=10.0,
        smoothing_window_observations=5,
        smoothing_min_observations=3,
    )


def _write_pie_xml(path: Path, boxes: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    box_lines = []
    for box in boxes:
        box_lines.append(
            f'<box frame="{box["frame"]}" keyframe="1" occluded="0" '
            f'outside="{box.get("outside", 0)}" xtl="{box["x1"]}" '
            f'ytl="{box["y1"]}" xbr="{box["x2"]}" ybr="{box["y2"]}">'
            f'<attribute name="id">{box["track_id"]}</attribute>'
            f'<attribute name="action">{box.get("action", "walking")}</attribute>'
            "</box>"
        )
    path.write_text(
        "<annotations><meta><task><original_size>"
        "<width>640</width><height>480</height>"
        "</original_size></task></meta>"
        '<track label="pedestrian">'
        + "".join(box_lines)
        + "</track></annotations>",
        encoding="utf-8",
    )


class TrajectorySummaryTests(unittest.TestCase):
    def test_track_summary_includes_duration_and_missing_observations(self) -> None:
        features = extract_track_features(
            [
                _observation("p1", 0, 0.0),
                _observation("p1", 1, 10.0),
                _observation("p1", 4, 40.0),
            ],
            _config(),
        )

        summary = summarize_tracks(features)[0]

        self.assertEqual(summary.observation_count, 3)
        self.assertEqual(summary.expected_observation_count, 5)
        self.assertEqual(summary.missing_observation_count, 2)
        self.assertAlmostEqual(summary.duration_seconds or 0.0, 4 / 30)
        self.assertEqual(summary.first_frame_index, 0)
        self.assertEqual(summary.last_frame_index, 4)


class PredictionAnnotationComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.annotations = self.root / "annotations"
        _write_pie_xml(
            self.annotations / "set01" / "video_0001_annt.xml",
            [
                {
                    "frame": 0,
                    "track_id": "gt1",
                    "x1": 0,
                    "y1": 0,
                    "x2": 10,
                    "y2": 20,
                    "action": "walking",
                },
                {
                    "frame": 1,
                    "track_id": "gt1",
                    "x1": 10,
                    "y1": 0,
                    "x2": 20,
                    "y2": 20,
                    "action": "walking",
                },
                {
                    "frame": 0,
                    "track_id": "gt2",
                    "x1": 200,
                    "y1": 200,
                    "x2": 210,
                    "y2": 220,
                    "action": "standing",
                },
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _predictions(self):
        return extract_track_features(
            [
                _observation("p1", 0, 0.0),
                _observation("p1", 1, 10.0),
                _observation("p2", 0, 100.0, 100.0),
            ],
            _config(),
        )

    def test_comparison_distinguishes_predictions_and_ground_truth(self) -> None:
        comparison = compare_predictions_to_pie(
            self._predictions(),
            self.annotations,
        )

        self.assertFalse(
            comparison["ground_truth"]["is_exhaustive_scene_ground_truth"]
        )
        self.assertFalse(comparison["predictions"]["is_ground_truth"])
        self.assertEqual(comparison["counts"]["matched_boxes"], 2)
        self.assertEqual(comparison["counts"]["unmatched_ground_truth_boxes"], 1)
        self.assertEqual(
            comparison["counts"]["unmatched_prediction_boxes_on_evaluated_frames"],
            1,
        )
        self.assertAlmostEqual(
            comparison["counts"]["ground_truth_box_match_rate"] or 0.0,
            2 / 3,
        )
        self.assertAlmostEqual(
            comparison["counts"]["mean_iou_of_matched_boxes"] or 0.0,
            1.0,
        )
        self.assertEqual(
            comparison["counts"]["ground_truth_tracks_on_evaluated_frames"], 2
        )
        self.assertEqual(
            comparison["counts"]["predicted_tracks_on_evaluated_frames"], 2
        )
        self.assertEqual(comparison["counts"]["all_prediction_tracks"], 2)
        self.assertEqual(
            comparison["counts"][
                "ground_truth_tracks_with_a_match_on_evaluated_frames"
            ],
            1,
        )
        self.assertEqual(
            comparison["counts"][
                "predicted_tracks_with_a_match_on_evaluated_frames"
            ],
            1,
        )
        self.assertAlmostEqual(
            comparison["walking_standing_action_comparison"]["agreement_ratio"]
            or 0.0,
            1.0,
        )
        self.assertEqual(
            comparison["walking_standing_action_comparison"][
                "comparable_matched_observations"
            ],
            1,
        )

    def test_pipeline_writes_comparison_json_and_report(self) -> None:
        source = self.root / "tracks.jsonl"
        records = [
            {
                "set_id": "set01",
                "video_id": "video_0001",
                "track_id": "p1",
                "frame_index": 0,
                "bbox_xyxy": [0, 0, 10, 20],
            },
            {
                "set_id": "set01",
                "video_id": "video_0001",
                "track_id": "p1",
                "frame_index": 1,
                "bbox_xyxy": [10, 0, 20, 20],
            },
        ]
        source.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(_config().to_dict()), encoding="utf-8")

        result = run_extraction(
            source,
            config_path=config_path,
            output_dir=self.root / "out",
            annotations=self.annotations / "set01" / "video_0001_annt.xml",
        )

        comparison_path = result["comparison_path"]
        self.assertIsNotNone(comparison_path)
        assert comparison_path is not None
        self.assertTrue(comparison_path.is_file())
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        self.assertEqual(comparison["comparison_type"], "predictions_vs_pie_ground_truth")
        report = result["report_path"].read_text(encoding="utf-8")
        self.assertIn("Prediction vs PIE ground truth", report)
        self.assertIn("Predictions are YOLO + BoT-SORT tracker outputs, not ground truth", report)


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def numpy(self):
        return self

    def tolist(self):
        return self.values


class _FakeBoxes:
    def __init__(self, ids, coordinates, confidences):
        self.id = _FakeTensor(ids) if ids is not None else None
        self.xyxy = _FakeTensor(coordinates)
        self.conf = _FakeTensor(confidences)
        self._length = len(coordinates)

    def __len__(self):
        return self._length


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    last_options = None

    def __init__(self, weights):
        self.weights = weights

    def track(self, **options):
        _FakeYOLO.last_options = options
        for frame_index in range(5):
            yield _FakeResult(
                _FakeBoxes(
                    ids=[1, 2],
                    coordinates=[
                        [float(frame_index), 0.0, 10.0, 20.0],
                        [100.0, 100.0, 110.0, 120.0],
                    ],
                    confidences=[0.9, 0.8],
                )
            )


class VideoAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.video = self.root / "video_0001.mp4"
        self.model = self.root / "yolo11s.pt"
        self.tracker = self.root / "smartped_botsort.yaml"
        for path in (self.video, self.model, self.tracker):
            path.write_bytes(b"")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_video_adapter_uses_tracker_outputs_and_stops_at_frame_limit(self) -> None:
        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYOLO

        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            with mock.patch(
                "smartped_ai.trajectory.readers._video_fps",
                return_value=30.0,
            ):
                observations, issues, resolved_format = load_observations(
                    self.video,
                    _config(),
                    input_format="video",
                    default_set_id="set01",
                    default_video_id="video_0001",
                    model_path=self.model,
                    tracker_config=self.tracker,
                    confidence=0.4,
                    max_seconds=0.1,
                )

        self.assertEqual(resolved_format, "video")
        self.assertEqual(issues, [])
        self.assertEqual(len(observations), 6)
        self.assertEqual({item.frame_index for item in observations}, {0, 1, 2})
        self.assertEqual({item.track_id for item in observations}, {"1", "2"})
        self.assertEqual(
            {round(item.timestamp_seconds or 0.0, 6) for item in observations},
            {0.0, round(1 / 30, 6), round(2 / 30, 6)},
        )
        self.assertEqual(_FakeYOLO.last_options["tracker"], str(self.tracker))
        self.assertEqual(_FakeYOLO.last_options["conf"], 0.4)
        self.assertEqual(_FakeYOLO.last_options["classes"], [0])
        self.assertTrue(_FakeYOLO.last_options["persist"])
        self.assertFalse(_FakeYOLO.last_options["save"])


if __name__ == "__main__":
    unittest.main()
