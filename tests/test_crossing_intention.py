from __future__ import annotations

import unittest

from smartped_ai.data.prepare_detection import PedestrianBox
from smartped_ai.intention.pipeline import (
    CATEGORICAL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURE_COLUMNS,
    build_features,
    construct_target,
    split_for_set,
)


def _box(
    frame: int,
    *,
    x: float = 10.0,
    cross: str = "not-crossing",
    outside: bool = False,
    action: str = "walking",
) -> PedestrianBox:
    return PedestrianBox(
        frame_index=frame,
        track_id="p1",
        bbox_xyxy=(x, 20.0, x + 20.0, 60.0),
        bbox_raw_xyxy=("10", "20", "30", "60"),
        outside=outside,
        occluded_raw="0",
        keyframe=True,
        attributes={
            "id": "p1",
            "cross": cross,
            "action": action,
            "look": "looking",
            "gesture": "__undefined__",
        },
    )


def _attributes(
    *,
    outcome: str = "1",
    exp_start: int = 0,
    critical: int = 10,
    crossing_point: int = 15,
) -> dict[str, str]:
    return {
        "crossing": outcome,
        "exp_start_point": str(exp_start),
        "critical_point": str(critical),
        "crossing_point": str(crossing_point),
        "intention_prob": "0.9",
    }


class CrossingTargetTests(unittest.TestCase):
    def test_positive_crossing_on_last_horizon_frame_is_included(self) -> None:
        boxes = {frame: _box(frame) for frame in range(10, 41)}
        boxes[40] = _box(40, cross="crossing")

        target, reason = construct_target(
            _attributes(critical=10, crossing_point=40),
            boxes,
            horizon_frames=30,
        )

        self.assertEqual(target, 1)
        self.assertIsNone(reason)

    def test_crossing_after_horizon_is_not_used_as_positive_label(self) -> None:
        boxes = {frame: _box(frame) for frame in range(10, 42)}
        boxes[41] = _box(41, cross="crossing")

        target, reason = construct_target(
            _attributes(critical=10, crossing_point=41),
            boxes,
            horizon_frames=30,
        )

        self.assertEqual(target, 0)
        self.assertIsNone(reason)

    def test_irrelevant_incomplete_ambiguous_and_already_crossing_are_excluded(self) -> None:
        complete = {frame: _box(frame) for frame in range(10, 41)}
        cases = [
            (_attributes(outcome="-1"), complete, "crossing_irrelevant"),
            (
                _attributes(critical=10, crossing_point=40),
                {frame: box for frame, box in complete.items() if frame != 20},
                "incomplete_or_ambiguous_horizon",
            ),
            (
                _attributes(critical=10, crossing_point=40),
                {**complete, 20: _box(20, cross="crossing-irrelevant")},
                "incomplete_or_ambiguous_horizon",
            ),
            (
                _attributes(critical=10, crossing_point=40),
                {**complete, 10: _box(10, cross="crossing")},
                "already_crossing_by_cutoff",
            ),
        ]
        for attributes, boxes, expected_reason in cases:
            with self.subTest(reason=expected_reason):
                target, reason = construct_target(
                    attributes,
                    boxes,
                    horizon_frames=30,
                )
                self.assertIsNone(target)
                self.assertEqual(reason, expected_reason)


class CrossingFeatureTests(unittest.TestCase):
    def test_features_use_only_observations_at_or_before_cutoff(self) -> None:
        past = {frame: _box(frame, x=frame) for frame in range(0, 11)}
        future_a = {**past, 11: _box(11, x=11), 12: _box(12, x=12)}
        future_b = {**past, 11: _box(11, x=900), 12: _box(12, x=1200)}
        options = {
            "prediction_frame": 10,
            "image_width": 100,
            "image_height": 100,
            "history_window_frames": 10,
            "max_motion_gap_frames": 3,
        }

        features_a = build_features(future_a, **options)
        features_b = build_features(future_b, **options)

        self.assertEqual(features_a, features_b)
        self.assertTrue(
            all(
                not any(token in name.lower() for token in ("cross", "target", "label"))
                for name in FEATURE_COLUMNS
            )
        )
        self.assertNotIn("intention_prob", FEATURE_COLUMNS)
        self.assertNotIn("critical_point", FEATURE_COLUMNS)
        self.assertNotIn("crossing_point", FEATURE_COLUMNS)

    def test_missing_history_is_reported_without_interpolation(self) -> None:
        boxes = {
            0: _box(0, x=0),
            2: _box(2, x=20),
            10: _box(10, x=100),
        }

        features = build_features(
            boxes,
            prediction_frame=10,
            image_width=100,
            image_height=100,
            history_window_frames=10,
            max_motion_gap_frames=3,
        )

        self.assertEqual(features["history_observation_count"], 3)
        self.assertAlmostEqual(features["history_coverage_ratio"], 3 / 11)
        self.assertEqual(features["max_history_gap_frames"], 8)
        self.assertEqual(features["history_motion_pair_count"], 1)
        self.assertAlmostEqual(features["mean_dx_norm_per_frame"], 0.1)
        self.assertAlmostEqual(features["last_dx_norm_per_frame"], 0.1)
        self.assertEqual(
            set(FEATURE_COLUMNS),
            set(NUMERIC_FEATURE_COLUMNS) | set(CATEGORICAL_FEATURE_COLUMNS),
        )


class GroupedSplitTests(unittest.TestCase):
    def test_recording_set_is_mapped_as_a_whole_and_unknown_sets_fail(self) -> None:
        assignments = {
            "train": ["set01", "set02", "set03", "set05"],
            "validation": ["set06"],
            "test": ["set04"],
        }
        self.assertEqual(split_for_set("set03", assignments), "train")
        self.assertEqual(split_for_set("set06", assignments), "validation")
        self.assertEqual(split_for_set("set04", assignments), "test")
        with self.assertRaisesRegex(ValueError, "exactly one configured split"):
            split_for_set("set07", assignments)
        assignments["test"].append("set03")
        with self.assertRaisesRegex(ValueError, "exactly one configured split"):
            split_for_set("set03", assignments)


if __name__ == "__main__":
    unittest.main()
