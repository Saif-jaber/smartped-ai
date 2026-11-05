from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from smartped_ai.video_runner.pipeline import (
    VideoInfo,
    _draw_compact_label,
    _make_pedestrian_detection_csv,
    _merge_prediction_display_fields,
    _prepare_vehicle_config,
    _render_annotated_video,
    _select_nearby_pairs,
    compute_frame_limit,
    inspect_video,
    timestamps_aligned,
    validate_runner_inputs,
    verify_annotated_video,
)


class VideoInputTests(unittest.TestCase):
    def test_missing_source_and_checkpoints_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "video.mp4"
            model = root / "model.pt"
            pedestrian_tracker = root / "ped.yaml"
            vehicle_tracker = root / "vehicle.yaml"
            source.write_bytes(b"")
            model.write_bytes(b"model")
            pedestrian_tracker.write_text("tracker: botsort\n", encoding="utf-8")
            vehicle_tracker.write_text("tracker: botsort\n", encoding="utf-8")

            validate_runner_inputs(
                source,
                model,
                pedestrian_tracker,
                vehicle_tracker,
                0.25,
                60.0,
            )
            with self.assertRaises(FileNotFoundError):
                validate_runner_inputs(
                    root / "missing.mp4",
                    model,
                    pedestrian_tracker,
                    vehicle_tracker,
                    0.25,
                    60.0,
                )
            with self.assertRaises(FileNotFoundError):
                validate_runner_inputs(
                    source,
                    root / "missing.pt",
                    pedestrian_tracker,
                    vehicle_tracker,
                    0.25,
                    60.0,
                )

    def test_invalid_mp4_confidence_and_duration_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "video.mov"
            paths = [root / name for name in ("model.pt", "ped.yaml", "veh.yaml")]
            source.touch()
            for path in paths:
                path.touch()
            with self.assertRaisesRegex(ValueError, "MP4"):
                validate_runner_inputs(source, *paths, 0.25, 1.0)
            source = root / "video.mp4"
            source.touch()
            with self.assertRaisesRegex(ValueError, "confidence"):
                validate_runner_inputs(source, *paths, 1.1, 1.0)
            with self.assertRaisesRegex(ValueError, "duration_seconds"):
                validate_runner_inputs(source, *paths, 0.25, 0.0)
            with self.assertRaisesRegex(ValueError, "vehicle_confidence"):
                validate_runner_inputs(
                    source,
                    *paths,
                    0.25,
                    1.0,
                    vehicle_model_path=paths[0],
                    vehicle_confidence=1.1,
                )

    def test_vehicle_config_accepts_isolated_model_and_confidence_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "set01" / "video_0001.mp4"
            source.parent.mkdir()
            source.touch()
            vehicle_model = root / "vehicle.pt"
            tracker = root / "vehicle.yaml"
            for path in (vehicle_model, tracker):
                path.touch()
            config_path = root / "vehicle_config.json"
            _prepare_vehicle_config(
                source=source,
                output_dir=root / "vehicle_output",
                model_path=vehicle_model,
                tracker_config=tracker,
                duration_seconds=60.0,
                confidence=0.25,
                set_id="set01",
                video_id="video_0001",
                empty_annotation_path=root / "empty.xml",
                config_path=config_path,
            )
            import json

            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["model_path"], str(vehicle_model.resolve()))
            self.assertEqual(config["confidence_threshold"], 0.25)
            self.assertEqual(config["tracker_config_path"], str(tracker.resolve()))

    def test_unreadable_video_releases_capture(self) -> None:
        capture = mock.Mock()
        capture.isOpened.return_value = False
        with mock.patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(OSError, "Could not open"):
                inspect_video(Path("bad.mp4"))
        capture.release.assert_called_once()


class DurationAndAlignmentTests(unittest.TestCase):
    def test_duration_limit_uses_ceil_and_clamps_to_source_length(self) -> None:
        info = VideoInfo(
            source=Path("video.mp4"),
            fps=30.0,
            width=1920,
            height=1080,
            frame_count=18000,
            duration_seconds=600.0,
        )
        self.assertEqual(compute_frame_limit(info, None), 18000)
        self.assertEqual(compute_frame_limit(info, 60.0), 1800)
        self.assertEqual(compute_frame_limit(info, 0.01), 1)
        self.assertEqual(compute_frame_limit(info, 900.0), 18000)
        with self.assertRaises(ValueError):
            compute_frame_limit(info, -1.0)

    def test_frame_timestamps_must_match_source_fps(self) -> None:
        self.assertTrue(timestamps_aligned(30, 1.0, 1.0, 30.0))
        self.assertFalse(timestamps_aligned(30, 1.0, 1.1, 30.0))
        self.assertFalse(timestamps_aligned(30, None, 1.0, 30.0))
        self.assertFalse(timestamps_aligned(30, "bad", 1.0, 30.0))

    def test_prediction_display_merge_requires_exact_recording_and_tracks(self) -> None:
        interaction = {
            "set_id": "set01",
            "video_id": "video_0001",
            "frame_index": "10",
            "pedestrian_track_id": "p1",
            "vehicle_track_id": "v1",
        }
        unrelated_prediction = {
            **interaction,
            "set_id": "set02",
            "prediction_available": "true",
            "crossing_probability": "0.9",
        }
        merged = _merge_prediction_display_fields([interaction], [unrelated_prediction])
        self.assertEqual(merged[0]["prediction_available"], "false")
        self.assertEqual(merged[0]["crossing_probability"], "")

        valid_prediction = {
            **interaction,
            "prediction_available": "true",
            "crossing_probability": "0.7",
        }
        merged = _merge_prediction_display_fields([interaction], [valid_prediction])
        self.assertEqual(merged[0]["prediction_available"], "true")
        self.assertEqual(merged[0]["crossing_probability"], "0.7")

    def test_nearby_pair_selection_deduplicates_and_prioritizes_smallest_distance(self) -> None:
        rows = [
            {
                "proximity_status": "nearby",
                "distance_pixels": "50",
                "pedestrian_track_id": "p1",
                "vehicle_track_id": "v1",
            },
            {
                "proximity_status": "nearby",
                "distance_pixels": "20",
                "pedestrian_track_id": "p1",
                "vehicle_track_id": "v1",
            },
            {
                "proximity_status": "nearby",
                "distance_pixels": "35",
                "pedestrian_track_id": "p2",
                "vehicle_track_id": "v2",
            },
            {
                "proximity_status": "nearby",
                "distance_pixels": "151",
                "pedestrian_track_id": "p3",
                "vehicle_track_id": "v3",
            },
        ]
        selected = _select_nearby_pairs(rows, 150.0, limit=3)
        self.assertEqual(
            [
                (row["pedestrian_track_id"], row["vehicle_track_id"])
                for row in selected
            ],
            [("p1", "v1"), ("p2", "v2")],
        )
        self.assertEqual(selected[0]["distance_pixels"], "20")


class OutputVerificationTests(unittest.TestCase):
    def test_empty_detection_csv_has_a_stable_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trajectory_path = root / "trajectory.csv"
            trajectory_path.write_text(
                "set_id,video_id,frame_index,timestamp_seconds,track_id,"
                "confidence,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n",
                encoding="utf-8",
            )
            destination = root / "pedestrian_detections.csv"
            count, rows = _make_pedestrian_detection_csv(
                trajectory_path,
                destination,
            )
            self.assertEqual((count, rows), (0, []))
            self.assertIn("track_id", destination.read_text(encoding="utf-8"))

    def test_renderer_handles_empty_detections_and_missing_prediction(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (64, 64),
            )
            if not writer.isOpened():
                writer.release()
                self.skipTest("MP4 writer codec is unavailable")
            for _ in range(2):
                writer.write(np.zeros((64, 64, 3), dtype="uint8"))
            writer.release()
            info = VideoInfo(source, 10.0, 64, 64, 2, 0.2)
            interactions = [
                {
                    "frame_index": "0",
                    "proximity_status": "nearby",
                    "distance_pixels": "12.5",
                    "pedestrian_footpoint_x_px": "20",
                    "pedestrian_footpoint_y_px": "30",
                    "nearest_vehicle_point_x_px": "24",
                    "nearest_vehicle_point_y_px": "33",
                    "pedestrian_track_id": "p1",
                    "vehicle_track_id": "v1",
                    "vehicle_class": "car",
                    "movement_status": "indeterminate",
                }
            ]
            output = root / "annotated.mp4"
            result = _render_annotated_video(
                source=source,
                destination=output,
                video=info,
                frame_limit=2,
                pedestrian_rows=[],
                vehicle_rows=[],
                interaction_rows=interactions,
                pixel_distance_threshold=150.0,
                progress=lambda _: None,
            )
            self.assertEqual(result["frames_written"], 2)
            verified = verify_annotated_video(
                output,
                expected_frames=2,
                expected_fps=10.0,
            )
            self.assertTrue(verified["first_frame_decodes"])

    def test_compact_labels_are_clamped_to_frame_edges(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are unavailable")
        image = np.zeros((64, 64, 3), dtype="uint8")
        rect = _draw_compact_label(
            cv2,
            image,
            "P123 0.94 walking",
            bbox=(0, 0, 12, 12),
            color=(55, 215, 75),
            occupied=[],
            scale=0.4,
        )
        self.assertGreaterEqual(rect[0], 0)
        self.assertGreaterEqual(rect[1], 0)
        self.assertLessEqual(rect[2], 64)
        self.assertLessEqual(rect[3], 64)

    def test_annotated_output_requires_open_and_decodable_video(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is unavailable")
        self.assertTrue(cv2.__version__)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty.mp4"
            capture = mock.Mock()
            capture.isOpened.return_value = False
            with mock.patch("cv2.VideoCapture", return_value=capture):
                with self.assertRaisesRegex(OSError, "cannot be opened"):
                    verify_annotated_video(
                        path,
                        expected_frames=10,
                        expected_fps=30,
                    )
            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
