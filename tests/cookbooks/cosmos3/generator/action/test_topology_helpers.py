# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
HELPER_PATH = ROOT / "cookbooks" / "cosmos3" / "generator" / "action" / "topology_helpers.py"
SPEC = importlib.util.spec_from_file_location("topology_helpers", HELPER_PATH)
topology_helpers = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["topology_helpers"] = topology_helpers
SPEC.loader.exec_module(topology_helpers)


def mask(rows):
    return [[1 if char == "#" else 0 for char in row] for row in rows]


class TopologyHelpersTest(unittest.TestCase):
    def test_empty_mask_has_no_components_or_holes(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        m = mask(["....", "....", "...."])

        stats = topology_helpers.label_components(m, cfg)

        self.assertEqual(stats.components, 0)
        self.assertEqual(topology_helpers.count_holes(m, cfg), 0)

    def test_solid_rectangle_has_one_component_and_no_holes(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        m = mask(["....", ".##.", ".##.", "...."])

        stats = topology_helpers.label_components(m, cfg)

        self.assertEqual(stats.components, 1)
        self.assertEqual(stats.area_px, 4)
        self.assertEqual(stats.bbox, (1, 1, 2, 2))
        self.assertEqual(topology_helpers.count_holes(m, cfg), 0)

    def test_two_rectangles_are_two_components(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        m = mask(["#..#", "#..#", "...."])

        stats = topology_helpers.label_components(m, cfg)

        self.assertEqual(stats.components, 2)
        self.assertEqual(stats.area_px, 4)

    def test_ring_counts_as_one_hole(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        m = mask([".....", ".###.", ".#.#.", ".###.", "....."])

        self.assertEqual(topology_helpers.label_components(m, cfg).components, 1)
        self.assertEqual(topology_helpers.count_holes(m, cfg), 1)

    def test_ring_touching_border_does_not_count_as_hole(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        m = mask(["###.", "#.#.", "###.", "...."])

        self.assertEqual(topology_helpers.count_holes(m, cfg), 1)

        open_to_border = mask(["#.#.", "#.#.", "###.", "...."])
        self.assertEqual(topology_helpers.count_holes(open_to_border, cfg), 0)

    def test_noise_removed_by_area_threshold(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=3)
        m = mask(["#...", "....", "..##", "...."])

        stats = topology_helpers.label_components(m, cfg)

        self.assertEqual(stats.components, 0)

    def test_diagonal_connectivity_is_configurable(self):
        m = mask(["#.", ".#"])
        cfg4 = topology_helpers.TopologyConfig(
            min_component_area_px=1, foreground_connectivity=4, background_connectivity=8
        )
        cfg8 = topology_helpers.TopologyConfig(
            min_component_area_px=1, foreground_connectivity=8, background_connectivity=4
        )

        self.assertEqual(topology_helpers.label_components(m, cfg4).components, 2)
        self.assertEqual(topology_helpers.label_components(m, cfg8).components, 1)

    def test_sequence_reports_drift_and_chunk_indices(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1, generated_frame_start=1)
        rollout = topology_helpers.RolloutSpec(
            video_id="robotics_action_cond_stitched",
            domain_name="droid_lerobot",
            fps=15,
            action_chunk_size=2,
        )
        masks = [
            mask(["....", "....", "....", "...."]),
            mask(["....", ".##.", ".##.", "...."]),
            mask(["#..#", "#..#", "....", "...."]),
            mask(["#..#", "#..#", "....", "...."]),
        ]

        report = topology_helpers.evaluate_fd_rollout(masks, rollout, cfg)

        self.assertEqual(report.schema_version, "topology_metrics.v1")
        self.assertEqual(len(report.frames), 3)
        self.assertEqual(report.frames[0].frame_index, 1)
        self.assertEqual(report.frames[0].generated_index, 0)
        self.assertEqual(report.frames[0].chunk_index, 0)
        self.assertEqual(report.frames[1].components, 2)
        self.assertEqual(report.frames[1].topology_delta_prev, 1)
        self.assertEqual(report.frames[2].chunk_index, 1)
        self.assertEqual(report.summary.component_change_count, 1)

    def test_conditioning_reference_compares_to_conditioning_frame(self):
        cfg = topology_helpers.TopologyConfig(
            min_component_area_px=1,
            generated_frame_start=1,
            reference="conditioning",
        )
        rollout = topology_helpers.RolloutSpec(video_id="robotics_action_cond")
        masks = [
            mask(["....", "....", "....", "...."]),
            mask(["....", ".##.", ".##.", "...."]),
        ]

        report = topology_helpers.evaluate_fd_rollout(masks, rollout, cfg)

        self.assertEqual(report.frames[0].topology_delta_ref, 1)

    def test_labeled_sequences_validate_equal_lengths(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        rollout = topology_helpers.RolloutSpec(video_id="x")

        with self.assertRaisesRegex(ValueError, "same number of frames"):
            topology_helpers.evaluate_fd_rollout(
                {"object": [mask(["#"])], "gripper": [mask(["#"]), mask(["."])]},
                rollout,
                cfg,
            )

    def test_json_and_csv_outputs_are_stable(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        rollout = topology_helpers.RolloutSpec(video_id="umi_action_cond_stitched", domain_name="umi")
        report = topology_helpers.evaluate_fd_rollout(
            [mask(["..", "##"]), mask(["..", "##"])],
            rollout,
            cfg,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_path = topology_helpers.write_topology_json(report, tmp_path / "metrics.json")
            csv_path = topology_helpers.write_topology_csv(report, tmp_path / "metrics.csv")

            loaded = json.loads(json_path.read_text())
            self.assertEqual(loaded["schema_version"], "topology_metrics.v1")
            self.assertEqual(loaded["run"]["video_id"], "umi_action_cond_stitched")

            with csv_path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_id"], "umi_action_cond_stitched")
            self.assertIn("topology_delta_ref", rows[0])

    def test_threshold_frame_to_mask_accepts_rgb_rows(self):
        frame = [
            [(0, 0, 0), (255, 255, 255)],
            [(20, 20, 20), (60, 60, 60)],
        ]

        result = topology_helpers.threshold_frame_to_mask(frame, threshold=32)

        self.assertEqual(result, [[False, True], [False, True]])

    def test_state_vector_distance_and_betti_stability(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1)
        rollout = topology_helpers.RolloutSpec(video_id="stable")
        report = topology_helpers.evaluate_fd_rollout(
            [
                mask(["....", ".##.", ".##.", "...."]),
                mask(["....", ".##.", ".##.", "...."]),
                mask(["....", ".##.", ".##.", "...."]),
                mask(["....", ".##.", ".##.", "...."]),
            ],
            rollout,
            cfg,
        )

        first = topology_helpers.frame_to_state_vector(report.frames[0])
        second = topology_helpers.frame_to_state_vector(report.frames[1])

        self.assertEqual(topology_helpers.topology_state_distance(first, second), 0.0)
        self.assertEqual(topology_helpers.betti_stability_score(report.frames, window=3), 1.0)

    def test_fdtd_trace_and_convergence_gate_for_stable_rollout(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1, generated_frame_start=0)
        rollout = topology_helpers.RolloutSpec(video_id="stable", fps=10)
        report = topology_helpers.evaluate_fd_rollout(
            [
                mask(["....", ".##.", ".##.", "...."]),
                mask(["....", ".##.", ".##.", "...."]),
                mask(["....", ".##.", ".##.", "...."]),
                mask(["....", ".##.", ".##.", "...."]),
            ],
            rollout,
            cfg,
        )

        trace = topology_helpers.compute_fdtd_rollout_trace(report.frames)
        result = topology_helpers.evaluate_topological_convergence(report)

        self.assertEqual(len(trace), 3)
        self.assertTrue(all(sample.topology_speed == 0.0 for sample in trace))
        self.assertTrue(result.passed)

    def test_convergence_gate_fails_on_topology_jump(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1, generated_frame_start=0)
        rollout = topology_helpers.RolloutSpec(video_id="jump", fps=10)
        report = topology_helpers.evaluate_fd_rollout(
            [
                mask(["....", ".##.", ".##.", "...."]),
                mask(["#..#", "#..#", "....", "...."]),
                mask(["#..#", "#..#", "....", "...."]),
                mask(["#..#", "#..#", "....", "...."]),
            ],
            rollout,
            cfg,
        )
        gate = topology_helpers.TopologyConvergenceGate(max_topology_speed=0.5)

        result = topology_helpers.evaluate_topological_convergence(report, gate)

        self.assertFalse(result.passed)
        self.assertFalse(result.checks["topology_speed"])

    def test_sparse_topology_specialist_ranking(self):
        query = topology_helpers.TopologyStateVector(
            components=1,
            holes=0,
            euler_characteristic=1,
            area_ratio=0.25,
            centroid_x_norm=0.5,
            centroid_y_norm=0.5,
            topology_delta_prev=0,
            topology_delta_ref=0,
        )
        close = topology_helpers.SpecialistSignal(
            specialist_id="droid-contact",
            capability_tags=("droid", "contact"),
            state=query,
            reliability=0.95,
            cost=0.1,
            utilization=0,
        )
        far = topology_helpers.SpecialistSignal(
            specialist_id="umi-open-space",
            capability_tags=("umi",),
            state=topology_helpers.TopologyStateVector(
                components=4,
                holes=2,
                euler_characteristic=2,
                area_ratio=0.9,
                centroid_x_norm=0.0,
                centroid_y_norm=0.0,
                topology_delta_prev=3,
                topology_delta_ref=3,
            ),
            reliability=1.0,
            cost=0.0,
            utilization=0,
        )

        ranked = topology_helpers.rank_topology_specialists(
            query,
            [far, close],
            task_tags=("droid", "contact"),
            top_k=1,
        )

        self.assertEqual(ranked[0].specialist_id, "droid-contact")
        self.assertEqual(ranked[0].matched_tags, ("contact", "droid"))


class TopologyGroundTruthTest(unittest.TestCase):
    """Known-shape checks: the value a correct implementation must return."""

    CFG = topology_helpers.TopologyConfig(min_component_area_px=1)

    def _euler(self, rows):
        m = mask(rows)
        return (
            topology_helpers.label_components(m, self.CFG).components
            - topology_helpers.count_holes(m, self.CFG)
        )

    def test_separated_blobs_give_one_component_each(self):
        rows = ["#.#.#", ".....", "#.#.#"]
        self.assertEqual(topology_helpers.label_components(mask(rows), self.CFG).components, 6)
        self.assertEqual(self._euler(rows), 6)

    def test_solid_disk_is_euler_one(self):
        self.assertEqual(self._euler([".....", ".###.", ".###.", ".###.", "....."]), 1)

    def test_annulus_is_euler_zero(self):
        self.assertEqual(self._euler([".....", ".###.", ".#.#.", ".###.", "....."]), 0)

    def test_figure_eight_is_euler_minus_one(self):
        rows = [".....", ".###.", ".#.#.", ".###.", ".#.#.", ".###.", "....."]
        self.assertEqual(topology_helpers.label_components(mask(rows), self.CFG).components, 1)
        self.assertEqual(topology_helpers.count_holes(mask(rows), self.CFG), 2)
        self.assertEqual(self._euler(rows), -1)

    def test_diagonal_ring_encloses_a_hole_under_complementary_connectivity(self):
        # 8-connected foreground, 4-connected background is the only pair for which
        # the digital Jordan curve theorem holds: the diagonal ring must enclose
        # exactly one hole, so chi = 1 - 1 = 0.
        self.assertEqual(self._euler([".#.", "#.#", ".#."]), 0)

    def test_non_complementary_connectivity_is_rejected(self):
        # 8/8 reports the diagonal ring above as hole-free and 4/4 splits it into
        # four components; neither is a valid Euler characteristic.
        for foreground, background in ((4, 4), (8, 8)):
            with self.subTest(foreground=foreground, background=background):
                with self.assertRaisesRegex(ValueError, "complementary"):
                    topology_helpers.TopologyConfig(
                        foreground_connectivity=foreground,
                        background_connectivity=background,
                    )

    def test_area_filter_applies_to_holes_as_well_as_components(self):
        # A ring smaller than the area threshold is discarded entirely, so it can
        # contribute neither a component nor a hole: chi must be 0, not -1.
        cfg = topology_helpers.TopologyConfig(min_component_area_px=16)
        m = mask([".....", ".###.", ".#.#.", ".###.", "....."])

        self.assertEqual(topology_helpers.label_components(m, cfg).components, 0)
        self.assertEqual(topology_helpers.count_holes(m, cfg), 0)

    def test_scattered_points_report_no_holes(self):
        # Negative control: isolated points have components but enclose nothing, so
        # a run that reports loops here is finding structure in noise.
        rows = ["#...#", ".....", "..#..", ".....", "#...#"]
        self.assertEqual(topology_helpers.label_components(mask(rows), self.CFG).components, 5)
        self.assertEqual(topology_helpers.count_holes(mask(rows), self.CFG), 0)
        self.assertEqual(self._euler(rows), 5)


class TopologyMetricAxiomTest(unittest.TestCase):
    """topology_state_distance is advertised as a distance; hold it to that."""

    VECTORS = (
        topology_helpers.TopologyStateVector(1, 0, 1, 0.25, 0.5, 0.5, 0.0, 0.0),
        topology_helpers.TopologyStateVector(2, 1, 1, 0.40, 0.1, 0.9, 1.0, 2.0),
        topology_helpers.TopologyStateVector(0, 0, 0, 0.00, 0.0, 0.0, 0.0, 0.0),
        topology_helpers.TopologyStateVector(4, 2, 2, 0.90, 1.0, 0.2, 3.0, 3.0),
    )

    def test_identity_of_indiscernibles(self):
        for vector in self.VECTORS:
            self.assertEqual(topology_helpers.topology_state_distance(vector, vector), 0.0)

    def test_positivity_for_distinct_vectors(self):
        for left, right in combinations(self.VECTORS, 2):
            self.assertGreater(topology_helpers.topology_state_distance(left, right), 0.0)

    def test_symmetry(self):
        for left, right in combinations(self.VECTORS, 2):
            self.assertEqual(
                topology_helpers.topology_state_distance(left, right),
                topology_helpers.topology_state_distance(right, left),
            )

    def test_triangle_inequality(self):
        for left in self.VECTORS:
            for middle in self.VECTORS:
                for right in self.VECTORS:
                    direct = topology_helpers.topology_state_distance(left, right)
                    detour = topology_helpers.topology_state_distance(
                        left, middle
                    ) + topology_helpers.topology_state_distance(middle, right)
                    self.assertLessEqual(direct, detour + 1e-12)

    def test_topology_speed_matches_distance_over_dt(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1, generated_frame_start=0)
        rollout = topology_helpers.RolloutSpec(video_id="speed", fps=10)
        report = topology_helpers.evaluate_fd_rollout(
            [
                mask(["....", ".##.", ".##.", "...."]),
                mask(["#..#", "#..#", "....", "...."]),
            ],
            rollout,
            cfg,
        )

        sample = topology_helpers.compute_fdtd_rollout_trace(report.frames)[0]
        expected = topology_helpers.topology_state_distance(
            topology_helpers.frame_to_state_vector(report.frames[1]),
            topology_helpers.frame_to_state_vector(report.frames[0]),
        )

        self.assertEqual(sample.local_change_ratio, expected)
        self.assertAlmostEqual(sample.topology_speed, expected / sample.dt_s, places=12)


class TopologyDeterminismTest(unittest.TestCase):
    CFG = topology_helpers.TopologyConfig(min_component_area_px=1, generated_frame_start=0)
    ROLLOUT = topology_helpers.RolloutSpec(video_id="determinism", fps=15, action_chunk_size=2)

    def _labeled(self):
        object_masks = [mask(["....", ".##.", ".##.", "...."])] * 3
        gripper_masks = [mask(["#..#", "#..#", "....", "...."])] * 3
        return object_masks, gripper_masks

    def test_report_is_independent_of_label_insertion_order(self):
        object_masks, gripper_masks = self._labeled()

        forward = topology_helpers.evaluate_fd_rollout(
            {"object": object_masks, "gripper": gripper_masks}, self.ROLLOUT, self.CFG
        )
        reverse = topology_helpers.evaluate_fd_rollout(
            {"gripper": gripper_masks, "object": object_masks}, self.ROLLOUT, self.CFG
        )

        self.assertEqual(
            [frame.label for frame in forward.frames],
            [frame.label for frame in reverse.frames],
        )
        self.assertEqual(
            topology_helpers.report_to_dict(forward), topology_helpers.report_to_dict(reverse)
        )

    def test_json_and_csv_bytes_are_independent_of_label_insertion_order(self):
        object_masks, gripper_masks = self._labeled()
        forward = topology_helpers.evaluate_fd_rollout(
            {"object": object_masks, "gripper": gripper_masks}, self.ROLLOUT, self.CFG
        )
        reverse = topology_helpers.evaluate_fd_rollout(
            {"gripper": gripper_masks, "object": object_masks}, self.ROLLOUT, self.CFG
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            forward_json = topology_helpers.write_topology_json(forward, tmp_path / "a.json")
            reverse_json = topology_helpers.write_topology_json(reverse, tmp_path / "b.json")
            forward_csv = topology_helpers.write_topology_csv(forward, tmp_path / "a.csv")
            reverse_csv = topology_helpers.write_topology_csv(reverse, tmp_path / "b.csv")

            self.assertEqual(forward_json.read_bytes(), reverse_json.read_bytes())
            self.assertEqual(forward_csv.read_bytes(), reverse_csv.read_bytes())

    def test_fdtd_trace_order_is_independent_of_frame_order(self):
        object_masks, gripper_masks = self._labeled()
        forward = topology_helpers.evaluate_fd_rollout(
            {"object": object_masks, "gripper": gripper_masks}, self.ROLLOUT, self.CFG
        )

        canonical = [
            (sample.label, sample.frame_index)
            for sample in topology_helpers.compute_fdtd_rollout_trace(forward.frames)
        ]
        shuffled = [
            (sample.label, sample.frame_index)
            for sample in topology_helpers.compute_fdtd_rollout_trace(list(reversed(forward.frames)))
        ]

        self.assertEqual(canonical, shuffled)

    def test_repeated_runs_are_byte_identical(self):
        object_masks, gripper_masks = self._labeled()
        payloads = set()
        for _ in range(3):
            report = topology_helpers.evaluate_fd_rollout(
                {"object": object_masks, "gripper": gripper_masks}, self.ROLLOUT, self.CFG
            )
            payloads.add(json.dumps(topology_helpers.report_to_dict(report), sort_keys=True))

        self.assertEqual(len(payloads), 1)

    def test_specialist_ranking_breaks_score_ties_by_id(self):
        state = topology_helpers.TopologyStateVector(1, 0, 1, 0.25, 0.5, 0.5, 0.0, 0.0)
        alpha = topology_helpers.SpecialistSignal("alpha", ("droid",), state, reliability=0.9, cost=0.1)
        bravo = topology_helpers.SpecialistSignal("bravo", ("droid",), state, reliability=0.9, cost=0.1)

        forward = topology_helpers.rank_topology_specialists(
            state, [alpha, bravo], task_tags=("droid",)
        )
        reverse = topology_helpers.rank_topology_specialists(
            state, [bravo, alpha], task_tags=("droid",)
        )

        self.assertEqual(forward[0].score, forward[1].score)
        self.assertEqual([item.specialist_id for item in forward], ["alpha", "bravo"])
        self.assertEqual([item.specialist_id for item in reverse], ["alpha", "bravo"])


class TopologyValidationTest(unittest.TestCase):
    def test_non_positive_fps_is_rejected(self):
        for fps in (0, -15):
            with self.subTest(fps=fps):
                with self.assertRaisesRegex(ValueError, "fps"):
                    topology_helpers.RolloutSpec(video_id="v", fps=fps)

    def test_persistent_homology_defaults_to_disabled(self):
        summary = topology_helpers.compute_persistent_homology(
            mask(["##", "##"]), topology_helpers.TopologyConfig()
        )

        self.assertEqual(summary.status, "disabled")

    def test_persistent_homology_auto_backend_degrades_with_a_warning(self):
        cfg = topology_helpers.TopologyConfig(
            min_component_area_px=1,
            persistent_homology=topology_helpers.PersistentHomologyConfig(backend="auto"),
        )

        summary = topology_helpers.compute_persistent_homology(mask(["##", "##"]), cfg)

        self.assertIn(summary.status, ("computed", "unavailable"))
        if summary.status == "unavailable":
            self.assertIn("ripser", summary.warning)

    def test_persistent_homology_ripser_backend_is_strict(self):
        cfg = topology_helpers.TopologyConfig(
            min_component_area_px=1,
            persistent_homology=topology_helpers.PersistentHomologyConfig(backend="ripser"),
        )

        try:
            import ripser  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(ImportError, "ripser"):
                topology_helpers.compute_persistent_homology(mask(["##", "##"]), cfg)
        else:
            summary = topology_helpers.compute_persistent_homology(mask(["##", "##"]), cfg)
            self.assertEqual(summary.status, "computed")

    @unittest.skipUnless(
        importlib.util.find_spec("ripser") is not None, "ripser is not installed"
    )
    def test_persistent_homology_finds_the_loop_in_an_annulus(self):
        # Ground truth: an annulus has one long H1 bar. This exercises the whole
        # ripser path, which a list-vs-array mismatch previously broke.
        cfg = topology_helpers.TopologyConfig(
            min_component_area_px=1,
            persistent_homology=topology_helpers.PersistentHomologyConfig(backend="ripser"),
        )
        rows = [
            ".#####.",
            "##...##",
            "#.....#",
            "#.....#",
            "#.....#",
            "##...##",
            ".#####.",
        ]

        summary = topology_helpers.compute_persistent_homology(mask(rows), cfg)

        self.assertEqual(summary.status, "computed")
        self.assertEqual(summary.h0_bars, summary.sampled_points)
        self.assertGreaterEqual(summary.h1_bars, 1)
        self.assertGreater(summary.h1_max_persistence, 1.0)

    def test_convergence_metrics_expose_the_betti_window_count(self):
        cfg = topology_helpers.TopologyConfig(min_component_area_px=1, generated_frame_start=0)
        rollout = topology_helpers.RolloutSpec(video_id="short", fps=10)
        report = topology_helpers.evaluate_fd_rollout([mask(["##", "##"])] * 2, rollout, cfg)

        result = topology_helpers.evaluate_topological_convergence(report)

        # Two frames cannot fill the default window of four, so the Betti check is
        # vacuous. It still reports 1.0, but betti_windows == 0 makes that visible.
        self.assertEqual(result.metrics["betti_stability"], 1.0)
        self.assertEqual(result.metrics["betti_windows"], 0.0)

    def test_conditioning_reference_without_a_conditioning_frame_warns(self):
        cfg = topology_helpers.TopologyConfig(
            min_component_area_px=1, generated_frame_start=0, reference="conditioning"
        )
        rollout = topology_helpers.RolloutSpec(video_id="v")

        report = topology_helpers.evaluate_fd_rollout([mask(["##", "##"])] * 2, rollout, cfg)

        self.assertTrue(any("conditioning" in warning for warning in report.warnings))


if __name__ == "__main__":
    unittest.main()
