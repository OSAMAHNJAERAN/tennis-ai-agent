"""Stdlib-only contract tests: no media decode, rendering, model or GPU use.

All render dependencies are mocked. The retained real sampler is read only as
JSON so that these tests also exercise the exact historical protocol shape.
"""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.data import prepare_yolo26_ball_review_wave as review


FROZEN = review.ROOT / "outputs/ball_data_repair_plan600_v2/frozen_sampler_source_groups.json"
FROZEN_SHA256 = "920be9491390ef62327a19bf8c034a623ee14bd233c7801b93565af279eaf428"


class FrozenPlanContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen_raw = FROZEN.read_bytes()
        cls.plan = json.loads(cls.frozen_raw)

    def test_retained_plan_keeps_exact600_centers96_wave_and16_pending_tiles(self):
        self.assertEqual(hashlib.sha256(self.frozen_raw).hexdigest(), FROZEN_SHA256)
        counts = review.validate_frozen_render_plan(self.plan)
        self.assertEqual(counts["positive_center_proposals"], 600)
        self.assertEqual(counts["complete_tile_proposals"], 120)
        self.assertEqual(counts["wave_positive_proposals"], 96)
        self.assertEqual(counts["wave_complete_tiles"], 16)
        self.assertEqual(set(counts["wave_event_groups"].values()), {6})

    def test_plan_semantics_reject_changed_source_pts_groups_and_accepted_targets(self):
        for field in ("source_pts", "conservative_event_group", "accepted_ground_truth"):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                first = plan["positive_center_proposals"][0]
                first[field] = {"source_pts": first["source_pts"] + 1,
                                "conservative_event_group": "unknown",
                                "accepted_ground_truth": True}[field]
                with self.assertRaises(ValueError):
                    review.validate_frozen_render_plan(plan)

    def test_plan_semantics_reject_wave_reselection_duplicates_and_guard_changes(self):
        for mutation in ("missing_id", "duplicate_tile", "ram_floor", "boolean_pts", "native_path_id"):
            with self.subTest(mutation=mutation):
                plan = copy.deepcopy(self.plan)
                if mutation == "missing_id":
                    plan["wave_positive_ids"][0] = "not_in_the_sampler"
                elif mutation == "duplicate_tile":
                    plan["wave_tile_ids"][0] = plan["wave_tile_ids"][1]
                elif mutation == "ram_floor":
                    plan["resource_guards"]["minimum_available_ram_bytes"] -= 1
                elif mutation == "boolean_pts":
                    plan["positive_center_proposals"][0]["source_pts"] = True
                else:
                    plan["sources"][0]["clip"] = "../escape"
                with self.assertRaises(ValueError):
                    review.validate_frozen_render_plan(plan)

    def test_read_existing_sampler_requires_both_explicit_hash_and_exact_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            frozen = root / "sampler.json"
            frozen.write_bytes(self.frozen_raw)
            sidecar = root / "sampler.json.sha256"
            sidecar.write_text(FROZEN_SHA256 + "  sampler.json\n", encoding="utf-8")
            original = {path.name: path.read_bytes() for path in (frozen, sidecar)}
            with patch.object(review, "ROOT", root):
                plan, binding = review.read_frozen_render_plan(frozen, FROZEN_SHA256)
                self.assertEqual(plan, self.plan)
                self.assertFalse(binding["sampler_recomputed"])
                self.assertFalse(binding["sampler_bytes_rewritten"])
                self.assertEqual(binding["sampler_implementation_sha256"], self.plan["implementation_sha256"])
                with self.assertRaisesRegex(ValueError, "explicitly pinned hash"):
                    review.read_frozen_render_plan(frozen, "0" * 64)
                sidecar.write_text(FROZEN_SHA256 + "  different.json\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "sidecar"):
                    review.read_frozen_render_plan(frozen, FROZEN_SHA256)
                sidecar.write_bytes(original[sidecar.name])
                # Even a plausible change to an allowed tile origin violates
                # the caller's original byte hash; it cannot silently replace it.
                modified = copy.deepcopy(plan)
                modified["negative_tile_proposals"][0]["tile_origin_xy_px"] = [512, 440]
                frozen.write_text(json.dumps(modified), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "explicitly pinned hash"):
                    review.read_frozen_render_plan(frozen, FROZEN_SHA256)


class ResourceObservationContractTests(unittest.TestCase):
    def test_media_guard_rejects_noninteger_nonfinite_boolean_and_negative_readings(self):
        good = {"available_ram_bytes": 3 * 1024 ** 3, "free_disk_bytes": 10 * 1024 ** 3,
                "owned_rss_bytes": 20 * 1024 ** 2}
        for field in good:
            for invalid in (float("nan"), float("inf"), float("-inf"), True, False, -1, 2.0):
                with self.subTest(field=field, invalid=invalid):
                    values = dict(good, **{field: invalid})
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(review.psutil, "virtual_memory", return_value=SimpleNamespace(available=values["available_ram_bytes"])))
                        stack.enter_context(patch.object(review.shutil, "disk_usage", return_value=SimpleNamespace(free=values["free_disk_bytes"])))
                        process = stack.enter_context(patch.object(review.psutil, "Process"))
                        process.return_value.memory_info.return_value = SimpleNamespace(rss=values["owned_rss_bytes"])
                        with self.assertRaisesRegex(RuntimeError, "finite nonnegative integer"):
                            review.guarded_resources()

    def test_media_guard_keeps_exact_original_thresholds(self):
        with ExitStack() as stack:
            memory = stack.enter_context(patch.object(review.psutil, "virtual_memory", return_value=SimpleNamespace(available=1024 ** 3)))
            disk = stack.enter_context(patch.object(review.shutil, "disk_usage", return_value=SimpleNamespace(free=2 * 1024 ** 3)))
            process = stack.enter_context(patch.object(review.psutil, "Process"))
            process.return_value.memory_info.return_value = SimpleNamespace(rss=1)
            self.assertEqual(review.guarded_resources()["available_ram_bytes"], 1024 ** 3)
            memory.return_value.available -= 1
            with self.assertRaisesRegex(RuntimeError, "1 GiB"):
                review.guarded_resources()
            memory.return_value.available += 1
            disk.return_value.free -= 1
            with self.assertRaisesRegex(RuntimeError, "2 GiB"):
                review.guarded_resources()


class FreshOutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen_raw = FROZEN.read_bytes()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.old = self.root / "old_failed_render"
        self.old.mkdir()
        self.frozen = self.old / "frozen_sampler_source_groups.json"
        self.frozen.write_bytes(self.frozen_raw)
        self.sidecar = self.old / "frozen_sampler_source_groups.json.sha256"
        self.sidecar.write_text(FROZEN_SHA256 + "  frozen_sampler_source_groups.json\n", encoding="utf-8")
        (self.old / "old_partial_native.png").write_bytes(b"retained test placeholder; no media codec used")
        self.old_snapshot = {path.name: path.read_bytes() for path in self.old.iterdir()}
        self.output = self.root / "fresh_render"
        self.resources = {"available_ram_bytes": 3 * 1024 ** 3, "free_disk_bytes": 10 * 1024 ** 3,
                          "owned_rss_bytes": 20 * 1024 ** 2}
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(patch.object(review, "ROOT", self.root))

    def assert_old_preserved(self):
        self.assertEqual({path.name: path.read_bytes() for path in self.old.iterdir()}, self.old_snapshot)

    def test_existing_output_is_rejected_without_guard_import_decode_or_write(self):
        self.output.mkdir()
        marker = self.output / "preserve.txt"
        marker.write_bytes(b"must survive")
        guard = self.stack.enter_context(patch.object(review, "guarded_resources"))
        media = self.stack.enter_context(patch.object(review, "import_image_stack"))
        with self.assertRaises(FileExistsError):
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.output)
        guard.assert_not_called()
        media.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"must survive")
        self.assert_old_preserved()

    def test_fresh_output_cannot_be_nested_under_old_failed_artifacts(self):
        with self.assertRaisesRegex(ValueError, "outside the original"):
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.old / "new_child")
        self.assert_old_preserved()

    def test_output_path_traversal_is_rejected_before_guard_import_or_mutation(self):
        escaped = self.root / "child" / ".." / ".." / "escaped_ball_render"
        guard = self.stack.enter_context(patch.object(review, "guarded_resources"))
        media = self.stack.enter_context(patch.object(review, "import_image_stack"))
        with self.assertRaisesRegex(ValueError, "private workspace"):
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, escaped)
        guard.assert_not_called()
        media.assert_not_called()
        self.assert_old_preserved()

    def test_first_guard_failure_precedes_media_import_and_all_output_creation(self):
        self.stack.enter_context(patch.object(review, "guarded_resources", side_effect=RuntimeError("unchanged 1 GiB guard")))
        media = self.stack.enter_context(patch.object(review, "import_image_stack"))
        reader = self.stack.enter_context(patch.object(review, "read_frozen_render_plan"))
        with self.assertRaisesRegex(RuntimeError, "1 GiB"):
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.output)
        media.assert_not_called()
        reader.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assert_old_preserved()

    def test_postimport_guard_failure_still_creates_no_output(self):
        self.stack.enter_context(patch.object(review, "guarded_resources", side_effect=[self.resources, self.resources, RuntimeError("postimport guard")]))
        self.stack.enter_context(patch.object(review, "verify_frozen_render_inputs", return_value={"fixture": "unchanged"}))
        media = self.stack.enter_context(patch.object(review, "import_image_stack"))
        decode = self.stack.enter_context(patch.object(review, "decode"))
        with self.assertRaisesRegex(RuntimeError, "postimport"):
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.output)
        media.assert_called_once_with()
        decode.assert_not_called()
        self.assertFalse(self.output.exists())
        self.assert_old_preserved()

    def test_success_uses_original_plan_without_sampling_and_records_renderer_separately(self):
        order = []
        plan_snapshot = None

        def guard():
            order.append("guard")
            self.assertFalse(self.output.exists())
            return self.resources

        def fake_import():
            order.append("fake media import")
            self.assertFalse(self.output.exists())

        def fake_decode(plan, output):
            nonlocal plan_snapshot
            order.append("fake decode")
            self.assertEqual(plan, json.loads(self.frozen_raw))
            plan_snapshot = copy.deepcopy(plan)
            self.assertTrue((output / "render_launch_manifest.json").is_file())
            self.assertTrue((output / "native").is_dir())
            return {"positive_center_proposals": [], "negative_tile_proposals": [], "special_unannotated_review_cases": []}, [], []

        self.stack.enter_context(patch.object(review, "guarded_resources", side_effect=guard))
        self.stack.enter_context(patch.object(review, "import_image_stack", side_effect=fake_import))
        protected = self.stack.enter_context(patch.object(review, "verify_frozen_render_inputs", return_value={"fixture": "unchanged"}))
        self.stack.enter_context(patch.object(review, "decode", side_effect=fake_decode))
        self.stack.enter_context(patch.object(review, "boards", return_value=[]))
        prepare = self.stack.enter_context(patch.object(review, "prepare", side_effect=AssertionError("sampling must not run")))
        with redirect_stdout(io.StringIO()):
            report = review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.output)
        self.assertEqual(order, ["guard", "guard", "fake media import", "guard", "fake decode"])
        self.assertEqual(plan_snapshot, json.loads(self.frozen_raw))
        self.assertEqual(report["sampler_sha256"], FROZEN_SHA256)
        self.assertEqual(report["sampler_implementation_sha256"], json.loads(self.frozen_raw)["implementation_sha256"])
        self.assertEqual(report["implementation_sha256"], review.digest(Path(review.__file__)))
        self.assertFalse(report["frozen_sampler"]["sampler_recomputed"])
        self.assertEqual(protected.call_count, 2)
        prepare.assert_not_called()
        terminal = json.loads((self.output / "render_execution_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(terminal["status"], "COMPLETED_PENDING_VISUAL_REVIEW_ONLY")
        self.assertEqual(terminal["accepted_visible_boxes"], 0)
        self.assertEqual(terminal["accepted_complete_tile_negatives"], 0)
        self.assert_old_preserved()

    def test_decode_failure_retains_new_partial_artifact_without_complete_review_manifest(self):
        self.stack.enter_context(patch.object(review, "guarded_resources", return_value=self.resources))
        self.stack.enter_context(patch.object(review, "import_image_stack"))
        self.stack.enter_context(patch.object(review, "verify_frozen_render_inputs", return_value={"fixture": "unchanged"}))

        def fake_failure(plan, output):
            (output / "native" / "new_partial.txt").write_text("test fixture only", encoding="utf-8")
            raise RuntimeError("runtime resource guard")

        self.stack.enter_context(patch.object(review, "decode", side_effect=fake_failure))
        with self.assertRaisesRegex(RuntimeError, "runtime resource guard"):
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.output)
        self.assertTrue((self.output / "native" / "new_partial.txt").is_file())
        self.assertFalse((self.output / "review_aids_manifest.json").exists())
        terminal = json.loads((self.output / "render_execution_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(terminal["status"], "TERMINAL_RENDER_FAILURE_PARTIAL_ARTIFACTS_RETAINED")
        self.assertEqual(terminal["failed_phase"], "native_decode_and_exact_context_crops")
        self.assertFalse(terminal["completed_review_manifest_written"])
        self.assertEqual(terminal["accepted_visible_boxes"], 0)
        self.assertEqual(terminal["accepted_complete_tile_negatives"], 0)
        self.assert_old_preserved()

    def test_failure_receipt_write_error_preserves_original_failure_and_old_artifacts(self):
        self.stack.enter_context(patch.object(review, "guarded_resources", return_value=self.resources))
        self.stack.enter_context(patch.object(review, "import_image_stack"))
        self.stack.enter_context(patch.object(review, "verify_frozen_render_inputs", return_value={"fixture": "unchanged"}))
        self.stack.enter_context(patch.object(review, "decode", side_effect=RuntimeError("original resource failure")))
        actual_write = review.write_json

        def refuse_only_failure_receipt(path, value):
            if path.name == "render_execution_receipt.json":
                raise OSError("simulated receipt write refusal")
            return actual_write(path, value)

        self.stack.enter_context(patch.object(review, "write_json", side_effect=refuse_only_failure_receipt))
        with self.assertRaisesRegex(RuntimeError, "original resource failure") as error:
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.output)
        self.assertTrue(any("receipt write refusal" in note for note in error.exception.__notes__))
        self.assertFalse((self.output / "review_aids_manifest.json").exists())
        self.assert_old_preserved()

    def test_changed_protected_source_postflight_prevents_completed_review_manifest(self):
        self.stack.enter_context(patch.object(review, "guarded_resources", return_value=self.resources))
        self.stack.enter_context(patch.object(review, "import_image_stack"))
        self.stack.enter_context(patch.object(review, "verify_frozen_render_inputs", side_effect=[{"fixture": "original"}, ValueError("protected source changed")]))
        result = {"positive_center_proposals": [], "negative_tile_proposals": [], "special_unannotated_review_cases": []}
        self.stack.enter_context(patch.object(review, "decode", return_value=(result, [], [])))
        self.stack.enter_context(patch.object(review, "boards", return_value=[]))
        with self.assertRaisesRegex(ValueError, "protected source changed"):
            review.render_frozen_plan(self.frozen, FROZEN_SHA256, self.output)
        self.assertFalse((self.output / "review_aids_manifest.json").exists())
        terminal = json.loads((self.output / "render_execution_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(terminal["failed_phase"], "postrender_frozen_input_verification")
        self.assertFalse(terminal["completed_review_manifest_written"])
        self.assert_old_preserved()

    def test_manifest_write_failures_are_terminal_and_never_claim_completed_review(self):
        for filename, expected_phase in (("render_launch_manifest.json", "render_launch_manifest_publication"),
                                         ("review_aids_manifest.json", "review_manifest_publication")):
            with self.subTest(filename=filename), ExitStack() as local:
                output = self.root / ("fresh_" + filename.replace(".", "_"))
                local.enter_context(patch.object(review, "guarded_resources", return_value=self.resources))
                local.enter_context(patch.object(review, "import_image_stack"))
                local.enter_context(patch.object(review, "verify_frozen_render_inputs", return_value={"fixture": "original"}))
                result = {"positive_center_proposals": [], "negative_tile_proposals": [], "special_unannotated_review_cases": []}
                local.enter_context(patch.object(review, "decode", return_value=(result, [], [])))
                local.enter_context(patch.object(review, "boards", return_value=[]))
                actual_write = review.write_json

                def refuse_manifest(path, value):
                    if path.name == filename:
                        path.write_text("{ incomplete new artifact", encoding="utf-8")
                        raise OSError("simulated manifest write failure")
                    return actual_write(path, value)

                local.enter_context(patch.object(review, "write_json", side_effect=refuse_manifest))
                with self.assertRaisesRegex(OSError, "manifest write failure"):
                    review.render_frozen_plan(self.frozen, FROZEN_SHA256, output)
                terminal = json.loads((output / "render_execution_receipt.json").read_text(encoding="utf-8"))
                self.assertEqual(terminal["failed_phase"], expected_phase)
                self.assertFalse(terminal["completed_review_manifest_written"])
                self.assertEqual(terminal["partial_review_manifest_file_exists"], filename == "review_aids_manifest.json")
                self.assert_old_preserved()

    def test_render_cli_requires_explicit_frozen_digest_and_rejects_wave_changes(self):
        renderer = self.stack.enter_context(patch.object(review, "render_frozen_plan"))
        commands = [["--render", "--output", str(self.output)],
                    ["--render", "--output", str(self.output), "--frozen-plan", str(self.frozen),
                     "--expected-plan-sha256", FROZEN_SHA256, "--wave-size", "48"],
                    ["--prepare", "--output", str(self.output), "--frozen-plan", str(self.frozen)]]
        for command in commands:
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                review.main(command)
            self.assertEqual(error.exception.code, 2)
        renderer.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
