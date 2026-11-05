import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import study_audit as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def result(delta=-0.02):
    interval = [-0.03, -0.01]
    value = {"accuracy_delta": delta, "intervals": {"accuracy_delta": interval},
             "strict_preservation": audit.confirmation.preservation(delta, interval, 0),
             "one_pp_noninferiority": audit.confirmation.preservation(delta, interval, 1)}
    return {"scope": "confirmation", "samples": 8, "question_count": 2247,
            "domains": {name: dict(value) for name in audit.confirmation.DOMAINS},
            "macro": dict(value), "inference_role": "confirmatory_primary",
            "input_sha256": {"base/results.json": "verified-by-analyzer"}}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_empty_workspace_never_green(self):
        with patch.object(audit.confirmation, "analyze") as analyze:
            report = audit.build_audit(self.root, resamples=200, now=1)
        self.assertFalse(report["study_complete"])
        self.assertEqual(report["state"], "incomplete")
        self.assertFalse(analyze.called)
        self.assertEqual(len([i for i in report["items"] if i["id"].startswith("training:")]), 12)
        self.assertEqual(len([i for i in report["items"] if i["id"].startswith("development:")]), 15)

    def test_corrupt_protocol_is_explicit_contradiction(self):
        path = self.root / "lab/experiments/study/protocol.json"
        path.parent.mkdir(parents=True)
        path.write_text("{broken")
        report = audit.build_audit(self.root, resamples=200, now=1)
        self.assertFalse(report["study_complete"])
        self.assertEqual(report["state"], "contradicted")
        self.assertEqual(next(i for i in report["items"] if i["id"] == "protocol")["state"], "contradicted")

    def test_additional_refinements_add_requirements_without_removing_originals(self):
        scales = audit.study_report.adapter_refinement.SCALES
        protocol = {'budget_total_usd': 200, 'required_approaches': list(audit.APPROACHES),
                    'offline_controls': list(audit.CONTROLS),
                    'initial_refinement_candidates': list(audit.REFINEMENTS),
                    'additional_refinement_candidates': list(scales),
                    'additional_refinement_specs': {
                        n: {'parent_candidate': 'dpo-32', 'strength_multiplier': s,
                            'registered_at': '2026-10-24T02:00:00+00:00'} for n, s in scales.items()}}
        write(self.root / 'lab/experiments/study/protocol.json', protocol)
        report = audit.build_audit(self.root, resamples=200, now=1)
        self.assertFalse(report['study_complete'])
        self.assertEqual(len([i for i in report['items'] if i['id'].startswith('training:')]), 12)
        self.assertEqual(len([i for i in report['items'] if i['id'].startswith('development:')]), 17)
        for name in scales:
            checkpoint = next(i for i in report['items'] if i['id'] == 'checkpoint:' + name)
            self.assertTrue(checkpoint['required'])
            self.assertNotEqual(checkpoint['state'], 'proven')

    def test_complete_flag_does_not_hide_incomplete_rl_budget(self):
        info = {"training_status": {"status": "completed", "completed_groups": 103, "optimizer_updates": 30}}
        self.assertEqual(audit.training_budget("grpo", info)["state"], "contradicted")
        info["training_status"]["status"] = "partial"
        self.assertEqual(audit.training_budget("grpo", info)["state"], "incomplete")

    def test_zero_update_rl_is_not_proven(self):
        info = {"training_status": {"status": "completed", "completed_groups": 128, "optimizer_updates": 0}}
        self.assertEqual(audit.training_budget("rl-fixed", info)["state"], "contradicted")

    def test_mislabeled_rl_variant_is_rejected(self):
        info = {"training_status": {"status": "completed", "completed_groups": 128, "optimizer_updates": 10},
                "config": {"max_steps": 128, "group_size": 4, "max_new_tokens": 2048, "seed": 17,
                           "optimizer": "dr_grpo", "variant": "correctness"}}
        self.assertEqual(audit.training_budget("rl-fixed", info)["state"], "contradicted")
        info["config"]["variant"] = "fixed"
        self.assertEqual(audit.training_budget("rl-fixed", info)["state"], "proven")

    def test_replay_requires_same_measured_references_and_coefficient(self):
        config = {"data_sha256": "data", "references_sha256": "refs", "length_coefficient": 0.1,
                  "learning_rate": 1e-5, "rank": 16, "group_size": 4, "seed": 17,
                  "max_steps": 128, "max_new_tokens": 2048, "optimizer": "dr_grpo", "scale_rewards_resolved": False}
        replay = dict(config, replay_reward="fixed")
        rows = {"rl-fixed": {"training": {"config": config}}, "rl-replay": {"training": {"config": replay}}}
        self.assertEqual(audit.replay_control_item(rows)["state"], "proven")
        replay["references_sha256"] = "different"
        self.assertEqual(audit.replay_control_item(rows)["state"], "contradicted")

    def test_offline_budget_requires_selected_final_checkpoint(self):
        info = {"parent_training_result": {"complete": True, "step": 64, "requested_steps": 64},
                "config": {"steps": 64, "method": "dpo"}, "selected_checkpoint_state": {"step": 48}}
        self.assertEqual(audit.training_budget("dpo", info)["state"], "contradicted")
        info["selected_checkpoint_state"]["step"] = 64
        self.assertEqual(audit.training_budget("dpo", info)["state"], "proven")

    def test_ambiguous_create_blocks_cleanup_even_without_pod_id(self):
        checks = audit.rental_items({"ceiling_usd": 200, "jobs": {
            "ambiguous": {"state": "creating", "created_at": 0, "budget_rate": 1, "reserved_usd": 5}}}, now=3600)
        self.assertEqual(checks[0]["state"], "incomplete")
        self.assertIn("ambiguous", checks[0]["evidence"]["unreleased"])
        self.assertEqual(checks[1]["state"], "proven")

    def test_budget_overage_and_nan_cannot_be_green(self):
        for value in (201, float("nan"), -1):
            checks = audit.rental_items({"ceiling_usd": 200, "jobs": {
                "done": {"released_at": 3600, "created_at": 0, "pod_id": "p", "charged_estimate_usd": value}}}, now=4000)
            self.assertEqual(checks[1]["state"], "contradicted")

    def test_manifest_detects_changed_or_escaping_file(self):
        file = self.root / "artifact"
        file.write_text("original")
        expected = audit.confirmation.digest(file)
        self.assertEqual(audit.verify_manifest(self.root, {"artifact": expected}), 1)
        file.write_text("changed")
        with self.assertRaises(ValueError):
            audit.verify_manifest(self.root, {"artifact": expected})
        with self.assertRaises(ValueError):
            audit.verify_manifest(self.root, {"../escape": expected})

    def confirmation_fixture(self, locked="2026-10-23T20:00:00+00:00", state="complete"):
        data = self.root / "lab/experiments/data/study"
        split = data / "splits.json"
        write(split, {"frozen": "identity metadata only"})
        adapter = data / "adapters/dpo"
        write(adapter / "adapter_config.json", {"r": 16})
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture-weights")
        hashes = audit.confirmation.adapter_hashes(adapter)
        lock = {"candidate": "dpo", "primary_candidate": "dpo", "locked_at": locked,
                "adapter_files_sha256": hashes, "split_manifest_sha256": audit.confirmation.digest(split)}
        write(data / "selected-lock.json", lock)
        output = self.root / "arbitrary-location/study"
        write(output / "study-provenance.json", {
            "subset": "confirmation", "samples": 8, "cohort_total": 2247,
            "candidate_lock": lock, "adapter_files_sha256": {"dpo": hashes},
            "split_manifest_sha256": audit.confirmation.digest(split),
            "config": {"candidates": ["base", "dpo"], "candidate_lock": "selected-lock.json"}})
        for name in ("base", "dpo"):
            folder = output / "runs" / name
            write(folder / "results.json", {"fixture": True})
            (folder / "records.jsonl").write_text('{"fixture":true}\n')
            write(output / "status" / (name + ".json"), {"state": state,
                  "started_at": "2026-10-23T21:00:00+00:00", "responses": 2247 * 8,
                  "results_sha256": audit.confirmation.digest(folder / "results.json"),
                  "records_sha256": audit.confirmation.digest(folder / "records.jsonl")})
        protocol = {"registered_at": "2026-10-23T19:00:00+00:00"}
        return output, split, protocol

    def test_refinement_lineage_is_required_before_confirmation_outcomes_open(self):
        output, split, protocol = self.confirmation_fixture()
        scales = audit.study_report.adapter_refinement.SCALES
        protocol['additional_refinement_candidates'] = list(scales)
        protocol['additional_refinement_specs'] = {
            n: {'parent_candidate': 'dpo-32', 'strength_multiplier': s,
                'registered_at': '2026-10-23T19:30:00+00:00'} for n, s in scales.items()}
        provenance = json.loads((output / 'study-provenance.json').read_text())
        name = 'dpo-32-scale050'
        provenance['candidate_lock'].update(candidate=name, primary_candidate=name)
        write(output / 'study-provenance.json', provenance)
        with patch.object(audit.confirmation, 'analyze') as analyze:
            checks, analysis = audit.confirmation_items(self.root, output, split, protocol, 200)
        self.assertIsNone(analysis)
        self.assertFalse(analyze.called)
        self.assertIn('lineage', checks[0]['detail'])

    def test_late_lock_blocks_confirmation_outcome_read(self):
        output, split, protocol = self.confirmation_fixture(locked="2026-10-23T22:00:00+00:00")
        with patch.object(audit.confirmation, "analyze") as analyze:
            checks, analysis = audit.confirmation_items(self.root, output, split, protocol, 200)
        self.assertEqual(checks[0]["state"], "contradicted")
        self.assertIsNone(analysis)
        self.assertFalse(analyze.called)

    def test_running_confirmation_never_opens_outcomes(self):
        output, split, protocol = self.confirmation_fixture(state="running")
        (output / "runs/dpo/records.jsonl").write_text("Do not parse this running output")
        with patch.object(audit.confirmation, "analyze") as analyze:
            checks, analysis = audit.confirmation_items(self.root, output, split, protocol, 200)
        self.assertEqual(checks[1]["state"], "incomplete")
        self.assertFalse(analyze.called)

    def test_hash_mismatch_blocks_confirmation_analysis(self):
        output, split, protocol = self.confirmation_fixture()
        (output / "runs/dpo/records.jsonl").write_text("tampered")
        with patch.object(audit.confirmation, "analyze") as analyze:
            checks, _ = audit.confirmation_items(self.root, output, split, protocol, 200)
        self.assertEqual(checks[1]["state"], "contradicted")
        self.assertFalse(analyze.called)

    def test_honest_regression_is_completed_science(self):
        output, split, protocol = self.confirmation_fixture()
        with patch.object(audit.confirmation, "analyze", return_value=result()):
            checks, analysis = audit.confirmation_items(self.root, output.parent, split, protocol, 200)
        self.assertEqual([c["state"] for c in checks], ["proven", "proven"])
        self.assertEqual(analysis["macro"]["strict_preservation"]["classification"], "evidence_of_regression_beyond_margin")

    def test_false_preservation_label_is_rejected(self):
        output, split, protocol = self.confirmation_fixture()
        fabricated = result()
        fabricated["macro"]["strict_preservation"]["classification"] = "supported_at_stated_margin"
        with patch.object(audit.confirmation, "analyze", return_value=fabricated):
            checks, analysis = audit.confirmation_items(self.root, output, split, protocol, 200)
        self.assertEqual(checks[1]["state"], "contradicted")
        self.assertIsNone(analysis)


if __name__ == "__main__":
    unittest.main()
