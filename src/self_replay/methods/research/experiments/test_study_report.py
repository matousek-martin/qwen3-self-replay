import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import study_report as s


NAMES = ["sft-short", "sft-compatible", "dpo", "grpo", "dr-grpo", "rl-fixed",
         "rl-adaptive", "rl-difficulty", "rl-progressive", "rl-replay",
         "sft-random", "sft-dpo-matched", "dpo-16", "dpo-32", "dpo-48"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def analysis(delta=0, saving=0.2):
    return {"scope": "tuning", "macro": {"accuracy_delta": delta, "token_reduction": saving,
            "passes_exploratory_point_screen": delta >= -0.01 and saving > 0,
            "strict_preservation": {"classification": "inconclusive"}},
            "domains": {}, "decision_flags": {"domains_with_material_point_decline_over_one_pp": ["mbpp"]}}


class StudyReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / "lab/experiments/study"
        self.cloud = self.root / "lab/experiments/cloud"
        self.data = self.root / "lab/experiments/data/study"
        self.split = self.data / "splits.json"
        write(self.split, {"schema_version": 1, "subsets": {"tuning": {}}})
        write(self.study / "protocol.json", {"registered_at": "before-evaluation", "budget_total_usd": 200,
              "required_approaches": NAMES[:10], "offline_controls": NAMES[10:12],
              "initial_refinement_candidates": NAMES[12:],
              "splits_manifest": "lab/experiments/data/study/splits.json"})
        write(self.study / "evaluation-campaign.json", {"state": "running", "batches": []})
        write(self.study / "training-campaign.json", {"state": "running", "jobs": []})
        write(self.cloud / "ledger.json", {"ceiling_usd": 200, "jobs": {}})
        write(self.data / "adapters.json", {})

    def tearDown(self):
        self.temp.cleanup()

    def batch(self, job_name, names, created=1, gpu="NVIDIA L4", subset="tuning"):
        ledger = json.loads((self.cloud / "ledger.json").read_text())
        ledger["jobs"][job_name] = {"state": "running", "created_at": created,
                                   "budget_rate": 1.3, "reserved_usd": 7.8, "gpu": gpu}
        write(self.cloud / "ledger.json", ledger)
        output = self.cloud / job_name / "output/study"
        batches = json.loads((self.study / "evaluation-campaign.json").read_text())
        batches["batches"].append({"job": job_name, "candidates": names, "output": str(output),
                                  "success": False, "subset": subset})
        write(self.study / "evaluation-campaign.json", batches)
        provenance = {"subset": subset, "split_manifest_sha256": s.digest(self.split),
                      "config": {"candidates": ["base"] + names},
                      "adapter_files_sha256": {name: {"adapter_model.safetensors": name} for name in names},
                      "evaluator_files_sha256": {"evaluate.py": "pinned"}}
        write(output / "study-provenance.json", provenance)
        (output.parent / "packages.txt").write_text("torch==2.8.0\ntransformers==4.57.1\n")
        for name in ["base"] + names:
            run = output / "runs" / name
            write(run / "results.json", {"adapter": None if name == "base" else "/adapter/" + name,
                  "study": {"subset": subset, "candidate": name,
                            "split_manifest_sha256": s.digest(self.split),
                            "adapter_files_sha256": provenance["adapter_files_sha256"].get(name)}})
            (run / "records.jsonl").write_text('{}\n')
            write(output / "status" / (name + ".json"), {
                "state": "complete", "results_sha256": s.digest(run / "results.json"),
                "records_sha256": s.digest(run / "records.jsonl")})
        return output

    def training(self, name):
        job = "offline-train"
        source = self.cloud / job / "output/training" / name / "checkpoint-000064"
        source.mkdir(parents=True)
        (source / "adapter_config.json").write_text('{}')
        (source / "adapter_model.safetensors").write_text(name)
        write(source.parent / "result.json", {"complete": True, "step": 64, "training_tokens": 1234})
        write(source / "state.json", {"step": 64, "training_tokens": 1234})
        adapters = json.loads((self.data / "adapters.json").read_text())
        adapters[name] = {"source": str(source), "checkpoint_sha256": s.digest(source / "adapter_model.safetensors"),
                          "config_sha256": s.digest(source / 'adapter_config.json')}
        write(self.data / "adapters.json", adapters)
        return adapters[name]["checkpoint_sha256"]

    def align_registered_hash(self, output, name, digest):
        provenance = json.loads((output / "study-provenance.json").read_text())
        provenance["adapter_files_sha256"][name]["adapter_model.safetensors"] = digest
        adapters = json.loads((self.data / 'adapters.json').read_text())
        provenance['adapter_files_sha256'][name]['adapter_config.json'] = adapters[name]['config_sha256']
        write(output / "study-provenance.json", provenance)
        meta = json.loads((output / "runs" / name / "results.json").read_text())
        meta["study"]["adapter_files_sha256"] = provenance["adapter_files_sha256"][name]
        write(output / "runs" / name / "results.json", meta)
        status = json.loads((output / "status" / (name + ".json")).read_text())
        status["results_sha256"] = s.digest(output / "runs" / name / "results.json")
        write(output / "status" / (name + ".json"), status)

    def test_all_fifteen_pending_rows_remain_visible(self):
        report = s.build_report(self.root, 200, now=100)
        self.assertEqual(len(report["candidates"]), 15)
        self.assertEqual(report["completed_development_count"], 0)
        self.assertEqual(report["ranking"]["candidates"], [])
        self.assertFalse(report["cost"]["invoice_verified"])
        self.assertIsNone(report["cost"]["invoice_total_usd"])

    def test_complete_candidate_reported_from_running_batch_and_ranked_with_domain_warning(self):
        checkpoint = self.training("dpo")
        out = self.batch("study-eval-00-test", ["dpo"])
        self.align_registered_hash(out, "dpo", checkpoint)
        with patch.object(s, "analyze", return_value=analysis(-0.005, 0.4)) as analyze:
            report = s.build_report(self.root, 200, now=100)
        self.assertEqual(report["completed_development_count"], 1)
        rank = report["ranking"]["candidates"][0]
        self.assertEqual(rank["name"], "dpo")
        self.assertEqual(rank["domain_warnings"]["domains_with_material_point_decline_over_one_pp"], ["mbpp"])
        self.assertEqual(rank["strict_preservation"]["classification"], "inconclusive")
        self.assertEqual(analyze.call_args.args[3], "tuning")

    def test_hash_mismatch_is_invalid_not_a_score(self):
        out = self.batch("study-eval-00-test", ["dpo"])
        (out / "runs/dpo/records.jsonl").write_text('changed\n')
        with patch.object(s, "analyze") as analyze:
            report = s.build_report(self.root, 200, now=100)
        self.assertFalse(analyze.called)
        row = next(row for row in report["candidates"] if row["name"] == "dpo")
        self.assertEqual(row["evaluation_state"], "invalid")
        self.assertIn("hash", row["evaluation_attempts"][0]["error"])

    def test_same_weights_with_wrong_configuration_are_not_reported(self):
        checkpoint = self.training('dpo')
        out = self.batch('study-eval-00-test', ['dpo'])
        self.align_registered_hash(out, 'dpo', checkpoint)
        registry = json.loads((self.data / 'adapters.json').read_text())
        registry['dpo']['config_sha256'] = 'different-registered-configuration'
        write(self.data / 'adapters.json', registry)
        with patch.object(s, 'analyze') as analyze:
            report = s.build_report(self.root, 200, now=100)
        self.assertFalse(analyze.called)
        row = next(r for r in report['candidates'] if r['name'] == 'dpo')
        self.assertEqual(row['evaluation_state'], 'invalid')
        self.assertIn('adapter_config.json', row['evaluation_attempts'][0]['error'])

    def test_additional_refinements_do_not_replace_original_fifteen(self):
        protocol = json.loads((self.study / 'protocol.json').read_text())
        protocol['additional_refinement_candidates'] = list(s.adapter_refinement.SCALES)
        protocol['additional_refinement_specs'] = {
            name: {'parent_candidate': 'dpo-32', 'strength_multiplier': scale,
                   'registered_at': '2026-10-24T02:00:00+00:00'}
            for name, scale in s.adapter_refinement.SCALES.items()}
        write(self.study / 'protocol.json', protocol)
        report = s.build_report(self.root, 200, now=100)
        self.assertEqual(report['original_candidate_count'], 15)
        self.assertEqual(report['planned_candidate_count'], 17)
        self.assertTrue(set(NAMES) <= {r['name'] for r in report['candidates']})
        self.assertFalse(report['all_planned_development_complete'])

    def test_confirmation_batch_never_reads_outcomes(self):
        out = self.batch("confirmation-job", ["dpo"], subset="confirmation")
        (out / "study-provenance.json").write_text("invalid JSON must not be opened")
        with patch.object(s, "analyze") as analyze:
            report = s.build_report(self.root, 200, now=100)
        self.assertFalse(analyze.called)
        self.assertEqual(report["completed_development_count"], 0)
        self.assertFalse(any("confirmation-job" in key for key in report["input_sha256"]))

    def test_cross_gpu_direct_contrast_flagged(self):
        self.batch("study-eval-00-test", ["sft-short"], gpu="NVIDIA L4")
        self.batch("study-eval-01-test", ["sft-random"], created=2, gpu="NVIDIA RTX A6000")
        with patch.object(s, "analyze", return_value=analysis()):
            report = s.build_report(self.root, 200, now=100)
        contrast = next(c for c in report["controlled_contrasts"] if c["candidate"] == "sft-short" and c["reference"] == "sft-random")
        self.assertEqual(contrast["state"], "complete_with_comparability_caveats")
        self.assertTrue(any("different gpu" in warning for warning in contrast["warnings"]))

    def test_separate_batches_flagged_even_with_identical_gpu_and_packages(self):
        self.batch("study-eval-00-test", ["sft-short"])
        self.batch("study-eval-01-test", ["sft-random"], created=2)
        with patch.object(s, "analyze", return_value=analysis()):
            report = s.build_report(self.root, 200, now=100)
        contrast = next(c for c in report["controlled_contrasts"]
                        if c["candidate"] == "sft-short" and c["reference"] == "sft-random")
        self.assertEqual(contrast["state"], "complete_with_comparability_caveats")
        self.assertTrue(any("different evaluation batches" in w for w in contrast["warnings"]))
        repeated = next(c for c in report["candidates"] if c["name"] == "sft-random")
        self.assertTrue(any("separate fresh baselines" in w for w in repeated["warnings"]))

    def test_same_batch_contrast_has_no_cross_batch_warning(self):
        self.batch("study-eval-00-test", ["sft-short", "sft-random"])
        with patch.object(s, "analyze", return_value=analysis()):
            report = s.build_report(self.root, 200, now=100)
        contrast = next(c for c in report["controlled_contrasts"]
                        if c["candidate"] == "sft-short" and c["reference"] == "sft-random")
        self.assertIsNotNone(contrast["analysis"])
        self.assertFalse(any("different evaluation batches" in w for w in contrast["warnings"]))

    def test_earliest_valid_attempt_not_best_score(self):
        self.batch("study-eval-00-test", ["dpo"], created=1)
        self.batch("study-eval-01-test", ["dpo"], created=2)
        with patch.object(s, "analyze", side_effect=[analysis(0, 0.1), analysis(0.1, 0.8)]):
            report = s.build_report(self.root, 200, now=100)
        row = next(row for row in report["candidates"] if row["name"] == "dpo")
        self.assertEqual(row["analysis"]["macro"]["token_reduction"], 0.1)
        self.assertEqual(len(row["evaluation_attempts"]), 2)

    def test_cost_separates_accrual_reservation_and_invoice(self):
        ledger = {"ceiling_usd": 200, "jobs": {
            "done": {"created_at": 0, "released_at": 3600, "charged_estimate_usd": 2, "budget_rate": 2, "rate": 1},
            "live": {"created_at": 0, "budget_rate": 2, "rate": 1, "reserved_usd": 8}}}
        result = s.cost_report(ledger, 3600)
        self.assertEqual(result["released_conservative_estimate_usd"], 2)
        self.assertEqual(result["live_accrued_conservative_estimate_usd"], 2)
        self.assertEqual(result["total_conservative_liability_usd"], 10)
        self.assertEqual(result["allocated_rate_elapsed_estimate_usd"], 2)
        self.assertIsNone(result["invoice_total_usd"])

    def test_pareto_frontier_keeps_tradeoffs(self):
        rows = [{"name": "a", "analysis": analysis(0, 0.1)},
                {"name": "b", "analysis": analysis(0.01, 0.2)},
                {"name": "c", "analysis": analysis(-0.01, 0.5)},
                {"name": "pending", "analysis": None}]
        self.assertEqual(s.pareto_names(rows), ["b", "c"])


if __name__ == "__main__":
    unittest.main()
