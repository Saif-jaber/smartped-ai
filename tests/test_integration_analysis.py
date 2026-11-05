from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from smartped_ai.integration.pipeline import (
    _load_predictions,
    _summarize,
    assess_feature_compatibility,
    associate_predictions,
)


THRESHOLD = 0.4019572450275602


def _interaction(
    *,
    set_id: str = "set01",
    video_id: str = "video_0001",
    frame: int = 10,
    pedestrian: str = "tracker-p1",
    vehicle: str = "vehicle-1",
    vehicle_class: str = "car",
    distance: float = 25.0,
    movement_valid: bool = True,
    movement_status: str = "approaching",
) -> dict[str, object]:
    return {
        "set_id": set_id,
        "video_id": video_id,
        "frame_index": frame,
        "timestamp_seconds": frame / 30,
        "interaction_timestamp_status": "consistent",
        "pedestrian_track_id": pedestrian,
        "vehicle_track_id": vehicle,
        "vehicle_class": vehicle_class,
        "distance_pixels": distance,
        "movement_status": movement_status,
        "movement_valid": movement_valid,
        "movement_invalid_reason": "" if movement_valid else "frame_gap_exceeded",
    }


def _prediction(
    *,
    set_id: str = "set01",
    video_id: str = "video_0001",
    track_id: str = "annotation-p1",
    frame: int = 8,
    probability: float = 0.8,
    predicted: int = 1,
) -> dict[str, object]:
    return {
        "set_id": set_id,
        "video_id": video_id,
        "track_id": track_id,
        "prediction_frame": frame,
        "crossing_probability": probability,
        "predicted_crossing": predicted,
        "threshold": THRESHOLD,
        "source": "validation_predictions.csv",
    }


def _predictions_by_identity(*records: dict[str, object]):
    grouped = {}
    for record in records:
        key = (record["set_id"], record["video_id"], record["track_id"])
        grouped.setdefault(key, []).append(record)
    return grouped


def _mapping(
    *,
    set_id: str = "set01",
    video_id: str = "video_0001",
    interaction_id: str = "tracker-p1",
    prediction_id: str = "annotation-p1",
):
    return {
        (set_id, video_id, interaction_id): {
            "prediction_track_id": prediction_id,
            "mapping_source": "human-reviewed synthetic crosswalk",
        }
    }


def _associate(rows, predictions=None, mapping=None, **overrides):
    options = {
        "model_name": "random_forest",
        "frozen_threshold": THRESHOLD,
        "max_age_frames": 30,
        "fps": 30.0,
        "timestamp_tolerance": 1e-6,
        "minimum_observations_for_closest_approach": 3,
    }
    options.update(overrides)
    return associate_predictions(
        rows,
        predictions or {},
        mapping or {},
        **options,
    )


class PredictionAssociationTests(unittest.TestCase):
    def test_associates_only_explicit_same_recording_track(self) -> None:
        result = _associate(
            [_interaction(frame=10)],
            _predictions_by_identity(_prediction(frame=8)),
            _mapping(),
        )[0]
        self.assertTrue(result["prediction_available"])
        self.assertEqual(result["crossing_probability"], 0.8)
        self.assertEqual(result["crossing_prediction"], 1)
        self.assertEqual(result["prediction_frame"], 8)
        self.assertEqual(result["prediction_age_frames"], 2)
        self.assertEqual(result["prediction_source"], "random_forest")
        self.assertEqual(result["prediction_identity_mapping_source"], "human-reviewed synthetic crosswalk")

    def test_recording_set_and_video_are_part_of_prediction_key(self) -> None:
        interactions = [_interaction()]
        predictions = _predictions_by_identity(
            _prediction(set_id="set02"),
            _prediction(video_id="video_0002"),
        )
        result = _associate(interactions, predictions, _mapping())[0]
        self.assertFalse(result["prediction_available"])
        self.assertEqual(result["prediction_status"], "no_prediction_for_mapped_track")

    def test_prediction_before_interaction_is_temporally_aligned(self) -> None:
        result = _associate(
            [_interaction(frame=10)],
            _predictions_by_identity(_prediction(frame=10)),
            _mapping(),
        )[0]
        self.assertEqual(result["prediction_age_frames"], 0)
        self.assertEqual(result["prediction_timestamp_seconds"], 10 / 30)
        self.assertEqual(result["prediction_timestamp_status"], "consistent")

    def test_interaction_timestamp_inconsistent_with_frame_is_invalid(self) -> None:
        interaction = _interaction(frame=10)
        interaction["timestamp_seconds"] = 0.9
        result = _associate(
            [interaction],
            _predictions_by_identity(_prediction(frame=8)),
            _mapping(),
        )[0]
        self.assertFalse(result["prediction_available"])
        self.assertEqual(result["prediction_status"], "interaction_timestamp_frame_mismatch")
        self.assertEqual(result["association_validity"], "invalid_timestamp")

    def test_future_prediction_is_not_associated_with_earlier_interaction(self) -> None:
        result = _associate(
            [_interaction(frame=9)],
            _predictions_by_identity(_prediction(frame=10)),
            _mapping(),
        )[0]
        self.assertFalse(result["prediction_available"])
        self.assertIsNone(result["crossing_probability"])
        self.assertEqual(
            result["prediction_status"], "prediction_is_after_interaction_frame"
        )
        self.assertEqual(result["association_validity"], "invalid_future_prediction")

    def test_prediction_is_unavailable_after_target_horizon(self) -> None:
        result = _associate(
            [_interaction(frame=39)],
            _predictions_by_identity(_prediction(frame=8)),
            _mapping(),
            max_age_frames=30,
        )[0]
        self.assertFalse(result["prediction_available"])
        self.assertEqual(result["prediction_status"], "prediction_expired")
        self.assertEqual(result["prediction_age_frames"], 31)

    def test_missing_identity_mapping_is_explicit_and_probability_null(self) -> None:
        result = _associate(
            [_interaction()],
            _predictions_by_identity(_prediction()),
        )[0]
        self.assertEqual(result["prediction_status"], "no_identity_mapping")
        self.assertFalse(result["prediction_available"])
        self.assertIsNone(result["crossing_probability"])
        self.assertIsNone(result["crossing_prediction"])

    def test_equal_tracker_and_annotation_id_is_not_assumed_to_be_a_mapping(self) -> None:
        result = _associate(
            [_interaction(pedestrian="7")],
            _predictions_by_identity(_prediction(track_id="7")),
        )[0]
        self.assertFalse(result["prediction_available"])
        self.assertEqual(result["prediction_status"], "no_identity_mapping")

    def test_invalid_interaction_timestamp_prevents_association(self) -> None:
        interaction = _interaction()
        interaction["interaction_timestamp_status"] = "invalid_timestamp"
        result = _associate(
            [interaction],
            _predictions_by_identity(_prediction(frame=8)),
            _mapping(),
        )[0]
        self.assertFalse(result["prediction_available"])
        self.assertEqual(result["prediction_status"], "invalid_interaction_timestamp")
        self.assertEqual(result["association_validity"], "invalid_timestamp")

    def test_prediction_age_and_probability_are_not_modified(self) -> None:
        source = _prediction(frame=8, probability=0.402, predicted=1)
        result = _associate(
            [_interaction(frame=9)],
            _predictions_by_identity(source),
            _mapping(),
        )[0]
        self.assertEqual(result["prediction_threshold"], THRESHOLD)
        self.assertEqual(result["crossing_probability"], 0.402)

    def test_duplicate_interactions_are_removed(self) -> None:
        row = _interaction()
        result = _associate([row, dict(row)])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["prediction_available"])


class FeatureCompatibilityTests(unittest.TestCase):
    def test_annotation_categories_and_tracker_semantics_are_incompatible(self) -> None:
        metadata = {
            "feature_columns": [
                "center_x_norm",
                "action_at_prediction",
                "look_at_prediction",
                "gesture_at_prediction",
            ],
            "categorical_features": {
                "action_at_prediction": "PIE frame-level action at critical_point.",
                "look_at_prediction": "PIE frame-level look at critical_point.",
                "gesture_at_prediction": "PIE frame-level gesture at critical_point.",
            },
        }
        compatibility = assess_feature_compatibility(
            metadata,
            {
                "center_x_norm",
                "pedestrian_track_id",
                "frame_index",
                "distance_pixels",
            },
        )
        self.assertFalse(compatibility["compatible_for_tracker_inference"])
        self.assertEqual(
            compatibility["missing_model_feature_columns_from_interaction_output"],
            ["action_at_prediction", "look_at_prediction", "gesture_at_prediction"],
        )
        self.assertFalse(compatibility["inference_performed"])

    def test_empty_feature_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "feature_columns"):
            assess_feature_compatibility({}, [])


class IntegrationOutputTests(unittest.TestCase):
    def test_multiple_vehicles_remain_distinct_for_one_pedestrian(self) -> None:
        rows = _associate(
            [
                _interaction(vehicle="v1", vehicle_class="car", distance=20),
                _interaction(vehicle="v2", vehicle_class="truck", distance=40),
            ]
        )
        summary = _summarize(rows, [])
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(summary), 2)
        self.assertEqual({row["vehicle_track_id"] for row in summary}, {"v1", "v2"})
        self.assertTrue(
            all(row["observations_with_valid_intention_prediction"] == 0 for row in summary)
        )
        self.assertTrue(
            all(row["crossing_probability_mean"] is None for row in summary)
        )

    def test_empty_interactions_produce_empty_output(self) -> None:
        self.assertEqual(_associate([]), [])
        self.assertEqual(_summarize([], []), [])

    def test_frame_causal_closest_approach_does_not_reveal_future_minimum(self) -> None:
        rows = [
            _interaction(frame=0, distance=10.0, movement_valid=False),
            _interaction(frame=1, distance=20.0),
            _interaction(frame=2, distance=5.0),
        ]
        result = _associate(
            rows,
            minimum_observations_for_closest_approach=2,
        )
        by_frame = {row["frame_index"]: row for row in result}
        self.assertFalse(by_frame[0]["closest_approach_valid_as_of_frame"])
        self.assertEqual(
            by_frame[1]["closest_approach_separation_pixels_as_of_frame"], 10.0
        )
        self.assertEqual(
            by_frame[1]["closest_approach_frame_index_as_of_frame"], 0
        )
        self.assertEqual(
            by_frame[2]["closest_approach_separation_pixels_as_of_frame"], 5.0
        )
        self.assertEqual(
            by_frame[2]["closest_approach_frame_index_as_of_frame"], 2
        )


class SavedPredictionValidationTests(unittest.TestCase):
    def test_saved_threshold_is_preserved_and_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            columns = [
                "model",
                "set_id",
                "video_id",
                "track_id",
                "prediction_frame",
                "crossing_probability",
                "predicted_crossing",
                "threshold",
            ]

            def write(threshold: float) -> None:
                with path.open("w", encoding="utf-8", newline="") as destination:
                    writer = csv.DictWriter(destination, fieldnames=columns)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "model": "random_forest",
                            "set_id": "set06",
                            "video_id": "video_0001",
                            "track_id": "gt1",
                            "prediction_frame": 10,
                            "crossing_probability": 0.8,
                            "predicted_crossing": 1,
                            "threshold": threshold,
                        }
                    )

            write(THRESHOLD)
            grouped, metrics = _load_predictions(
                (path,),
                expected_model_name="random_forest",
                frozen_threshold=THRESHOLD,
            )
            self.assertEqual(metrics["unique_prediction_records"], 1)
            self.assertEqual(grouped[("set06", "video_0001", "gt1")][0]["threshold"], THRESHOLD)
            write(0.5)
            with self.assertRaisesRegex(ValueError, "frozen validation threshold"):
                _load_predictions(
                    (path,),
                    expected_model_name="random_forest",
                    frozen_threshold=THRESHOLD,
                )

    def test_null_probability_prediction_row_is_rejected_not_filled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv"
            columns = [
                "model",
                "set_id",
                "video_id",
                "track_id",
                "prediction_frame",
                "crossing_probability",
                "predicted_crossing",
                "threshold",
            ]
            with path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "model": "random_forest",
                        "set_id": "set06",
                        "video_id": "video_0001",
                        "track_id": "gt1",
                        "prediction_frame": 10,
                        "crossing_probability": "",
                        "predicted_crossing": "",
                        "threshold": THRESHOLD,
                    }
                )
            grouped, metrics = _load_predictions(
                (path,),
                expected_model_name="random_forest",
                frozen_threshold=THRESHOLD,
            )
            self.assertEqual(grouped, {})
            self.assertEqual(metrics["invalid_rows"], 1)


if __name__ == "__main__":
    unittest.main()
