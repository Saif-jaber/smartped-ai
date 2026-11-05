from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from smartped_ai.trajectory.calibration import (
    choose_speed_threshold,
    classification_metrics,
)
from smartped_ai.trajectory.camera_motion import (
    CameraMotionConfig,
    CameraMotionEstimate,
    _PairMotion,
    estimate_pair_motion,
)
from smartped_ai.trajectory.ego_telemetry import summarize_ego_telemetry
from smartped_ai.trajectory.experiment import load_experiment_config
from smartped_ai.trajectory.features import (
    coverage_diagnostics,
    extract_track_features,
)
from smartped_ai.trajectory.models import (
    MOVEMENT_STANDING,
    MOVEMENT_WALKING,
    TrajectoryConfig,
    TrackObservation,
)


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


def _config(**overrides) -> TrajectoryConfig:
    values = {
        "fps": 30.0,
        "max_gap_frames": 5,
        "standing_speed_threshold_px_s": 5.0,
        "smoothing_window_observations": 5,
        "smoothing_min_observations": 3,
    }
    values.update(overrides)
    return TrajectoryConfig(**values)


class _StubCameraMotion:
    def __init__(self, estimate: CameraMotionEstimate) -> None:
        self.estimate = estimate

    def displacement_at(self, point, frame_from: int, frame_to: int):
        return self.estimate


class CameraMotionFeatureTests(unittest.TestCase):
    def test_compensation_subtracts_background_motion(self) -> None:
        camera_motion = _StubCameraMotion(
            CameraMotionEstimate(
                dx_px=10.0,
                dy_px=0.0,
                method="local_optical_flow",
                local_sample_count=1,
                fallback_step_count=0,
                available=True,
            )
        )

        features = extract_track_features(
            [
                _observation("p1", 0, 100.0),
                _observation("p1", 1, 110.0),
            ],
            _config(),
            camera_motion=camera_motion,
        )

        feature = features[1]
        self.assertAlmostEqual(feature.raw_speed_px_s or 0.0, 300.0)
        self.assertIsNotNone(feature.speed_px_s)
        self.assertAlmostEqual(feature.speed_px_s, 0.0)
        self.assertIsNotNone(feature.displacement_x_px)
        self.assertAlmostEqual(feature.displacement_x_px, 0.0)
        self.assertTrue(feature.camera_motion_applied)
        self.assertEqual(feature.camera_motion_method, "local_optical_flow")
        self.assertEqual(feature.smoothed_movement_state, MOVEMENT_STANDING)

    def test_unavailable_compensation_falls_back_to_raw_displacement(self) -> None:
        camera_motion = _StubCameraMotion(
            CameraMotionEstimate(
                dx_px=0.0,
                dy_px=0.0,
                method="unavailable",
                local_sample_count=0,
                fallback_step_count=0,
                available=False,
                reason="missing_pair_motion",
            )
        )

        features = extract_track_features(
            [
                _observation("p1", 0, 100.0),
                _observation("p1", 1, 110.0),
            ],
            _config(),
            camera_motion=camera_motion,
        )

        feature = features[1]
        self.assertFalse(feature.camera_motion_applied)
        self.assertEqual(
            feature.camera_motion_invalid_reason, "missing_pair_motion"
        )
        self.assertAlmostEqual(feature.speed_px_s or 0.0, 300.0)
        self.assertEqual(feature.smoothed_movement_state, MOVEMENT_WALKING)


class CameraMotionOpticalFlowTests(unittest.TestCase):
    def test_synthetic_translation_is_recovered(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are required for optical-flow tests.")

        width, height = 320, 240
        rng = np.random.default_rng(7)
        previous = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        shift_x, shift_y = 6.0, 4.0
        current = cv2.warpAffine(
            previous,
            np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]]),
            (width, height),
        )
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        config = CameraMotionConfig(
            max_width_px=width,
            feature_max_corners=1500,
            feature_quality_level=0.01,
            feature_min_distance_px=5.0,
            local_radius_px=300.0,
            min_local_flow_samples=10,
        )

        pair = estimate_pair_motion(
            previous_gray,
            current_gray,
            [],
            config,
        )

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertIsInstance(pair, _PairMotion)
        local = pair.sample_local(
            (width / 2, height / 2),
            radius_px=300.0,
            min_samples=10,
        )
        self.assertIsNotNone(local)
        assert local is not None
        self.assertAlmostEqual(local[0], shift_x, delta=1.5)
        self.assertAlmostEqual(local[1], shift_y, delta=1.5)


class ClassificationCalibrationTests(unittest.TestCase):
    def test_classification_metrics_report_confusion_and_per_class_scores(self) -> None:
        metrics = classification_metrics(
            ["standing", "standing", "walking", "walking"],
            ["standing", "walking", "walking", "walking"],
        )

        self.assertEqual(metrics["confusion_matrix"]["matrix"], [[1, 1], [0, 2]])
        self.assertAlmostEqual(metrics["accuracy"] or 0.0, 0.75)
        self.assertAlmostEqual(metrics["balanced_accuracy"] or 0.0, 0.75)
        self.assertAlmostEqual(
            metrics["per_class"]["standing"]["precision"] or 0.0, 1.0
        )
        self.assertAlmostEqual(
            metrics["per_class"]["standing"]["recall"] or 0.0, 0.5
        )
        self.assertAlmostEqual(
            metrics["per_class"]["walking"]["recall"] or 0.0, 1.0
        )

    def test_threshold_selection_maximizes_balanced_accuracy(self) -> None:
        result = choose_speed_threshold(
            [
                ("standing", 10.0),
                ("standing", 20.0),
                ("walking", 100.0),
                ("walking", 110.0),
            ]
        )

        self.assertAlmostEqual(result["threshold_px_s"], 100.0)
        self.assertAlmostEqual(
            result["metrics"]["balanced_accuracy"] or 0.0, 1.0
        )

    def test_threshold_selection_requires_both_classes(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires both walking and standing"):
            choose_speed_threshold([("standing", 1.0), ("standing", 2.0)])


class ExperimentProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_config(self, evaluation: dict[str, str]) -> Path:
        path = self.root / "experiment.json"
        path.write_text(
            json.dumps(
                {
                    "evaluation_features_csv": "detections.csv",
                    "output_dir": "results",
                    "calibration": [
                        {"set_id": "set03", "video_id": "video_0001"},
                        {"set_id": "set05", "video_id": "video_0001"},
                    ],
                    "evaluation": evaluation,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_config_keeps_calibration_and_evaluation_recordings_disjoint(self) -> None:
        config = load_experiment_config(
            self._write_config(
                {"set_id": "set01", "video_id": "video_0001"}
            ),
            root=self.root,
        )

        calibration = {
            (clip.set_id, clip.video_id) for clip in config["calibration"]
        }
        evaluation = (
            config["evaluation"].set_id,
            config["evaluation"].video_id,
        )
        self.assertNotIn(evaluation, calibration)

    def test_config_rejects_evaluation_recording_used_for_calibration(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "evaluation recording cannot also be used for calibration"
        ):
            load_experiment_config(
                self._write_config(
                    {"set_id": "set03", "video_id": "video_0001"}
                ),
                root=self.root,
            )


class CoverageDiagnosticsTests(unittest.TestCase):
    def test_gap_buckets_and_missing_frame_counts(self) -> None:
        features = extract_track_features(
            [
                _observation("p1", 0, 0.0),
                _observation("p1", 1, 0.0),
                _observation("p1", 4, 0.0),
                _observation("p1", 10, 0.0),
                _observation("p1", 30, 0.0),
            ],
            _config(max_gap_frames=100),
        )

        diagnostics = coverage_diagnostics(features)

        self.assertEqual(diagnostics["gap_events"]["frame_gap_2_5"], 1)
        self.assertEqual(diagnostics["gap_events"]["frame_gap_6_15"], 1)
        self.assertEqual(diagnostics["gap_events"]["frame_gap_gt_15"], 1)
        self.assertEqual(diagnostics["missing_frames"]["frame_gap_2_5"], 2)
        self.assertEqual(diagnostics["missing_frames"]["frame_gap_6_15"], 5)
        self.assertEqual(diagnostics["missing_frames"]["frame_gap_gt_15"], 19)
        self.assertEqual(diagnostics["total_missing_frames"], 26)
        self.assertAlmostEqual(
            diagnostics["long_gap_missing_frame_share"] or 0.0, 19 / 26
        )


class EgoTelemetryTests(unittest.TestCase):
    def test_ego_telemetry_summary_is_context_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "video_0001_obd.xml"
            path.write_text(
                "<vehicle_info>"
                '<frame GPS_speed="10" OBD_speed="9" accX="0" accY="0" '
                'accZ="0" gyroX="0" gyroY="0" gyroZ="0" '
                'heading_angle="90" id="0" latitude="0" longitude="0" '
                'pitch="1" roll="2" yaw="3" />'
                '<frame GPS_speed="12" OBD_speed="10" accX="0" accY="0" '
                'accZ="0" gyroX="0" gyroY="0" gyroZ="0" '
                'heading_angle="92" id="1" latitude="0" longitude="0" '
                'pitch="1" roll="2" yaw="3" />'
                "</vehicle_info>",
                encoding="utf-8",
            )

            summary = summarize_ego_telemetry(path)

        self.assertEqual(summary["frame_count"], 2)
        self.assertAlmostEqual(summary["gps_speed_kmh"]["mean"], 11.0)
        self.assertAlmostEqual(
            summary["mean_absolute_heading_change_deg_per_frame"], 2.0
        )
        self.assertFalse(summary["usable_for_pixel_compensation"])


if __name__ == "__main__":
    unittest.main()
