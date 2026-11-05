from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smartped_ai.interaction.pipeline import (
    InteractionConfig,
    Observation,
    generate_interaction_rows,
    load_config,
    nearest_point_on_box,
    pedestrian_footpoint,
    pixel_box_distance,
    summarize_interactions,
)


def _config(**overrides) -> InteractionConfig:
    values = {
        "pedestrian_trajectory_path": Path("pedestrians.csv"),
        "vehicle_detection_path": Path("vehicle_detections.csv"),
        "vehicle_trajectory_path": Path("vehicle_trajectories.csv"),
        "video_path": Path("video.mp4"),
        "output_dir": Path("output"),
        "set_ids": (),
        "video_ids": (),
        "max_observation_gap_frames": 3,
        "min_observations_for_movement": 3,
        "min_observations_for_closest_approach": 3,
        "pixel_distance_threshold": 15.0,
        "timestamp_tolerance_seconds": 1e-6,
        "movement_delta_tolerance_pixels": 0.1,
        "visual_sample_count": 0,
    }
    values.update(overrides)
    return InteractionConfig(**values)


def _observation(
    *,
    track_id: str,
    frame: int,
    bbox: tuple[float, float, float, float],
    kind: str,
    set_id: str = "set01",
    video_id: str = "video_0001",
    timestamp: float | None = None,
    timestamp_issue: str | None = None,
    class_name: str | None = None,
) -> Observation:
    return Observation(
        set_id=set_id,
        video_id=video_id,
        track_id=track_id,
        frame_index=frame,
        timestamp_seconds=frame / 30.0 if timestamp is None else timestamp,
        timestamp_issue=timestamp_issue,
        bbox_xyxy=bbox,
        class_name=class_name if kind == "vehicle" else None,
    )


def _pair(
    frame: int,
    vehicle_x: float,
    *,
    pedestrian: Observation | None = None,
    vehicle: Observation | None = None,
    timestamp: float | None = None,
    timestamp_issue: str | None = None,
) -> tuple[Observation, Observation]:
    pedestrian_observation = pedestrian or _observation(
        track_id="p1",
        frame=frame,
        bbox=(0.0, 0.0, 10.0, 20.0),
        kind="pedestrian",
        timestamp=timestamp,
        timestamp_issue=timestamp_issue,
    )
    vehicle_observation = vehicle or _observation(
        track_id="v1",
        frame=frame,
        bbox=(vehicle_x, 10.0, vehicle_x + 10.0, 30.0),
        kind="vehicle",
        timestamp=timestamp,
        timestamp_issue=timestamp_issue,
        class_name="car",
    )
    return pedestrian_observation, vehicle_observation


class InteractionGeometryTests(unittest.TestCase):
    def test_pedestrian_footpoint_is_bottom_center(self) -> None:
        self.assertEqual(pedestrian_footpoint((2.0, 4.0, 12.0, 24.0)), (7.0, 24.0))

    def test_distance_is_to_nearest_vehicle_box_point(self) -> None:
        point = pedestrian_footpoint((0.0, 0.0, 10.0, 20.0))
        box = (30.0, 10.0, 40.0, 30.0)
        self.assertEqual(nearest_point_on_box(point, box), (30.0, 20.0))
        self.assertEqual(pixel_box_distance(point, box), 25.0)

    def test_boxes_outside_image_are_not_clipped_for_geometry(self) -> None:
        point = pedestrian_footpoint((-30.0, -20.0, -10.0, 0.0))
        self.assertEqual(point, (-20.0, 0.0))
        self.assertEqual(
            nearest_point_on_box(point, (10.0, -5.0, 20.0, 5.0)),
            (10.0, 0.0),
        )


class InteractionPairingTests(unittest.TestCase):
    def test_aligns_exact_frame_and_pairs_multiple_objects(self) -> None:
        pedestrians = [
            _observation(track_id="p1", frame=0, bbox=(0, 0, 10, 20), kind="pedestrian"),
            _observation(track_id="p2", frame=0, bbox=(30, 0, 40, 20), kind="pedestrian"),
            _observation(track_id="p1", frame=1, bbox=(0, 0, 10, 20), kind="pedestrian"),
        ]
        vehicles = [
            _observation(track_id="v1", frame=0, bbox=(50, 0, 70, 30), kind="vehicle", class_name="car"),
            _observation(track_id="v2", frame=0, bbox=(80, 0, 100, 30), kind="vehicle", class_name="bus"),
            _observation(track_id="v1", frame=1, bbox=(50, 0, 70, 30), kind="vehicle", class_name="car"),
        ]

        rows = generate_interaction_rows(pedestrians, vehicles, _config())

        self.assertEqual(len(rows), 5)
        self.assertEqual(
            sum(row["frame_index"] == 0 for row in rows),
            4,
        )
        self.assertTrue(
            all(
                row["set_id"] == "set01" and row["video_id"] == "video_0001"
                for row in rows
            )
        )

    def test_no_pairing_across_recording_or_video(self) -> None:
        pedestrian = _observation(
            track_id="same-id",
            frame=5,
            bbox=(0, 0, 10, 20),
            kind="pedestrian",
            set_id="set01",
            video_id="video_0001",
        )
        vehicles = [
            _observation(
                track_id="v1",
                frame=5,
                bbox=(20, 0, 30, 20),
                kind="vehicle",
                set_id="set02",
                video_id="video_0001",
                class_name="car",
            ),
            _observation(
                track_id="v2",
                frame=5,
                bbox=(20, 0, 30, 20),
                kind="vehicle",
                set_id="set01",
                video_id="video_0002",
                class_name="car",
            ),
        ]
        self.assertEqual(generate_interaction_rows([pedestrian], vehicles, _config()), [])

    def test_duplicate_observation_does_not_duplicate_pair_row(self) -> None:
        pedestrian, vehicle = _pair(0, 30.0)
        rows = generate_interaction_rows(
            [pedestrian, pedestrian], [vehicle, vehicle], _config()
        )
        self.assertEqual(len(rows), 1)

    def test_empty_inputs_produce_no_pairs(self) -> None:
        self.assertEqual(generate_interaction_rows([], [], _config()), [])


class InteractionMovementTests(unittest.TestCase):
    def _rows_for_vehicle_positions(self, positions, *, max_gap=3, timestamps=None):
        pedestrians = []
        vehicles = []
        for index, frame in enumerate(positions):
            time_value = timestamps[index] if timestamps is not None else frame / 30.0
            pedestrian, vehicle = _pair(
                frame,
                40.0 - index * 5.0,
                timestamp=time_value,
            )
            pedestrians.append(pedestrian)
            vehicles.append(vehicle)
        return generate_interaction_rows(
            pedestrians,
            vehicles,
            _config(max_observation_gap_frames=max_gap),
        )

    def test_approaching_pair_has_valid_displacements_and_summary(self) -> None:
        rows = self._rows_for_vehicle_positions([0, 1, 2])
        self.assertEqual(
            [row["movement_status"] for row in rows],
            ["indeterminate", "approaching", "approaching"],
        )
        self.assertTrue(rows[1]["movement_valid"])
        self.assertEqual(rows[1]["pedestrian_displacement_x_px"], 0.0)
        self.assertEqual(rows[1]["vehicle_displacement_x_px"], -5.0)
        self.assertEqual(rows[1]["relative_displacement_x_px"], -5.0)
        summaries, _ = summarize_interactions(rows, _config())
        self.assertEqual(summaries[0]["movement_status"], "approaching")
        self.assertTrue(summaries[0]["movement_analysis_valid"])

    def test_separating_pair_is_identified(self) -> None:
        pedestrians = []
        vehicles = []
        for frame, x in enumerate((30.0, 35.0, 40.0)):
            pedestrian, vehicle = _pair(frame, x)
            pedestrians.append(pedestrian)
            vehicles.append(vehicle)
        rows = generate_interaction_rows(pedestrians, vehicles, _config())
        self.assertEqual(rows[-1]["movement_status"], "separating")
        self.assertEqual(summarize_interactions(rows, _config())[0][0]["movement_status"], "separating")

    def test_stationary_pair_is_indeterminate_not_approaching(self) -> None:
        pedestrian, vehicle = _pair(0, 30.0)
        fixed_pedestrians = []
        fixed_vehicles = []
        for frame in range(3):
            fixed_pedestrians.append(
                Observation(**{**pedestrian.__dict__, "frame_index": frame, "timestamp_seconds": frame / 30})
            )
            fixed_vehicles.append(
                Observation(**{**vehicle.__dict__, "frame_index": frame, "timestamp_seconds": frame / 30})
            )
        fixed_rows = generate_interaction_rows(
            fixed_pedestrians, fixed_vehicles, _config()
        )
        self.assertEqual(fixed_rows[-1]["movement_status"], "indeterminate")
        self.assertTrue(fixed_rows[-1]["movement_valid"])
        self.assertEqual(
            summarize_interactions(fixed_rows, _config())[0][0]["movement_status"],
            "indeterminate",
        )

    def test_missing_observation_within_gap_limit_is_not_interpolated(self) -> None:
        pedestrians = []
        vehicles = []
        for frame in (0, 2, 3):
            pedestrian, vehicle = _pair(frame, 30.0 - frame)
            pedestrians.append(pedestrian)
            vehicles.append(vehicle)
        rows = generate_interaction_rows(pedestrians, vehicles, _config(max_observation_gap_frames=2))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["frame_gap_from_previous_pair_observation"], 2)
        self.assertTrue(rows[1]["movement_valid"])

    def test_excessive_gap_invalidates_movement(self) -> None:
        pedestrians = []
        vehicles = []
        for frame in (0, 4):
            pedestrian, vehicle = _pair(frame, 30.0 - frame)
            pedestrians.append(pedestrian)
            vehicles.append(vehicle)
        rows = generate_interaction_rows(pedestrians, vehicles, _config(max_observation_gap_frames=2))
        self.assertFalse(rows[1]["movement_valid"])
        self.assertEqual(rows[1]["movement_invalid_reason"], "frame_gap_exceeded")

    def test_invalid_timestamps_invalidate_movement_but_zero_is_valid(self) -> None:
        pedestrian0, vehicle0 = _pair(0, 30.0, timestamp=0.0)
        pedestrian1, vehicle1 = _pair(1, 29.0, timestamp=0.1)
        rows = generate_interaction_rows(
            [pedestrian0, pedestrian1],
            [vehicle0, vehicle1],
            _config(),
        )
        self.assertEqual(rows[0]["timestamp_validation_status"], "consistent")
        self.assertTrue(rows[1]["movement_valid"])

        invalid_pedestrian = Observation(
            **{**pedestrian1.__dict__, "timestamp_seconds": None, "timestamp_issue": "invalid_timestamp"}
        )
        invalid_rows = generate_interaction_rows(
            [pedestrian0, invalid_pedestrian],
            [vehicle0, vehicle1],
            _config(),
        )
        self.assertEqual(
            invalid_rows[1]["timestamp_validation_status"], "invalid_timestamp"
        )
        self.assertFalse(invalid_rows[1]["movement_valid"])

    def test_timestamp_mismatch_is_reported_and_invalidates_movement(self) -> None:
        ped0, veh0 = _pair(0, 30.0)
        ped1, _ = _pair(1, 29.0)
        veh1 = Observation(
            **{**_pair(1, 29.0)[1].__dict__, "timestamp_seconds": 0.2}
        )
        rows = generate_interaction_rows([ped0, ped1], [veh0, veh1], _config())
        self.assertEqual(rows[1]["timestamp_validation_status"], "timestamp_mismatch")
        self.assertEqual(rows[1]["movement_invalid_reason"], "timestamp_mismatch")

    def test_closest_observed_approach_requires_sufficient_valid_observations(self) -> None:
        rows = self._rows_for_vehicle_positions([0, 1, 2])
        summaries, counts = summarize_interactions(rows, _config())
        self.assertTrue(summaries[0]["closest_approach_valid"])
        self.assertEqual(summaries[0]["closest_approach_type"], "minimum_observed_separation")
        self.assertEqual(summaries[0]["closest_approach_frame_index"], 2)
        self.assertEqual(counts["closest_approach_valid_persistent_pairs"], 1)

        short_config = _config(min_observations_for_closest_approach=4)
        short_summaries, _ = summarize_interactions(rows, short_config)
        self.assertFalse(short_summaries[0]["closest_approach_valid"])
        self.assertEqual(
            short_summaries[0]["closest_approach_invalid_reason"],
            "insufficient_consecutive_valid_observations",
        )

    def test_transient_pair_is_excluded_from_persistent_summary(self) -> None:
        pedestrian, vehicle = _pair(0, 30.0)
        rows = generate_interaction_rows([pedestrian], [vehicle], _config())
        summaries, counts = summarize_interactions(rows, _config())
        self.assertEqual(summaries, [])
        self.assertEqual(counts["transient_pair_count"], 1)


class InteractionInputTests(unittest.TestCase):
    def test_empty_csv_is_a_valid_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty.csv"
            path.write_text("", encoding="utf-8")
            # The configuration reader accepts zero-byte CSVs as empty sources.
            from smartped_ai.interaction.pipeline import _read_csv

            self.assertEqual(_read_csv(path, ("frame_index",)), ([], 0))

    def test_json_config_resolves_existing_experiment_paths(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "interaction_analysis.json"
        config = load_config(config_path)
        self.assertEqual(config.set_ids, ("set01",))
        self.assertEqual(config.video_ids, ("video_0001",))
        self.assertGreater(config.max_observation_gap_frames, 0)


if __name__ == "__main__":
    unittest.main()
