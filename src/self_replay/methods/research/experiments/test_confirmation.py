import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments import confirmation as c


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.questions = self.root / "questions"
        self.questions.mkdir()
        self.source = []
        for dataset, count in (("bfcl_simple_python", 8), ("bfcl_multiple", 4), ("bfcl_irrelevance", 4),
                               ("ifeval", 8), ("mbpp", 8), ("svamp", 8)):
            for i in range(count):
                self.source.append({"dataset": dataset, "id": f"{dataset}/{i}", "seed_index": len(self.source),
                                    "messages": [{"role": "user", "content": f"synthetic {i}"}]})
        source = self.questions / "questions.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in self.source))
        (self.questions / "manifest.json").write_text(json.dumps({"cohort_sha256": c.digest(source)}))
        self.manifest = c.create_split(self.questions, 4, 20261023)
        self.split = self.root / "split.json"
        c.write_json(self.split, self.manifest, immutable=True)

    def tearDown(self):
        self.temp.cleanup()

    def make_runs(self, part="tuning", trained_reference=False):
        info = self.manifest["subsets"][part]
        samples = self.manifest["policy"][f"{part}_samples"]
        base, candidate = [], []
        for n, row in enumerate(info["keys"]):
            for sample in range(samples):
                record = {"dataset": row["dataset"], "id": row["id"], "sample_index": sample,
                          "seed": c.SEED + row["seed_index"] * 1000 + sample,
                          "output_tokens": 100, "reasoning_tokens": 80, "truncated": False,
                          "correct": (n + sample) % 3 != 0}
                base.append(record)
                candidate.append({**record, "output_tokens": 75, "reasoning_tokens": 60,
                                  "correct": (n + sample) % 4 != 0})
        meta = {"model": "Qwen/Qwen3-0.6B", "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "cohort_sha256": info["questions_sha256"], "cohort_total": info["total"],
                "samples": samples, "seed": c.SEED, "sampling": c.SAMPLING,
                "max_tokens": {name: c.token_cap(name) for name in info["counts"]},
                "max_model_len": 12288, "benchmarks": list(c.DOMAINS), "failed_benchmarks": {},
                "adapter": "/adapter/control" if trained_reference else None,
                "sandbox": {"mode": "bwrap", "why": "explicit"}, "revisions": {"test": "pinned"},
                "selection": "study-" + part, "smoke": False}
        new_meta = {**meta, "adapter": "/adapter/candidate"}
        directories = []
        for name, metadata, rows in (("base", meta, base), ("candidate", new_meta, candidate)):
            directory = self.root / name
            directory.mkdir(exist_ok=True)
            (directory / "results.json").write_text(json.dumps(metadata))
            (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            directories.append(directory)
        return (*directories, meta, new_meta, base, candidate)

    def test_split_disjoint_complete_deterministic(self):
        self.assertEqual(self.manifest, c.create_split(self.questions, 4, 20261023))
        a, b = (c.subset_keys(self.manifest, part) for part in ("tuning", "confirmation"))
        self.assertFalse(a & b)
        self.assertEqual(len(a | b), len(self.source))
        self.assertEqual(self.manifest["subsets"]["tuning"]["domain_counts"], dict.fromkeys(c.DOMAINS, 4))
        c.load_split(self.split, self.questions / "questions.jsonl")

    def test_split_not_based_on_scores_or_answers(self):
        before = c.subset_keys(self.manifest, "tuning")
        for row in self.source:
            row.update(answer="changed", fake_baseline_accuracy=0.99)
        path = self.questions / "questions.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in self.source))
        (self.questions / "manifest.json").write_text(json.dumps({"cohort_sha256": c.digest(path)}))
        self.assertEqual(before, c.subset_keys(c.create_split(self.questions, 4, 20261023), "tuning"))

    def test_frozen_artifact_cannot_be_replaced(self):
        changed = copy.deepcopy(self.manifest)
        changed["seed"] += 1
        with self.assertRaises(ValueError):
            c.write_json(self.split, changed, immutable=True)

    def test_pairing_rejects_missing_duplicate_ungraded_and_seed(self):
        _, _, meta, new_meta, base, candidate = self.make_runs()
        for broken in (candidate[:-1], candidate + [candidate[0]],
                       [{**candidate[0], "correct": None}] + candidate[1:],
                       [{**candidate[0], "seed": 17}] + candidate[1:]):
            with self.assertRaises(ValueError):
                c.check_runs(meta, new_meta, self.manifest, "tuning", base, broken)

    def test_mismatched_sampling_or_cohort_rejected(self):
        _, _, meta, new_meta, base, candidate = self.make_runs()
        for patch in ({"samples": 8}, {"sampling": {"temperature": 0.7}}, {"cohort_sha256": "old-full-run"}):
            with self.assertRaises(ValueError):
                c.check_runs(meta, {**new_meta, **patch}, self.manifest, "tuning", base, candidate)

    def test_macro_four_domains_and_token_geomean(self):
        base, candidate, *_ = self.make_runs()
        result = c.analyze(base, candidate, self.split, resamples=200)
        self.assertEqual(set(result["domains"]), set(c.DOMAINS))
        self.assertAlmostEqual(result["macro"]["token_reduction"], 0.25)
        self.assertEqual(result["domains"]["bfcl"]["base"]["prompts"], 4)
        self.assertEqual(result["macro"]["intervals"]["groups"], 4)
        self.assertFalse(result["interpretation"]["full_contest_evaluation"])

    def test_trained_control_contrast_supported(self):
        base, candidate, *_ = self.make_runs(trained_reference=True)
        result = c.analyze(base, candidate, self.split, resamples=200)
        self.assertEqual(result["comparison_reference"]["kind"], "trained_control")

    def test_thinking_savings_distinct_from_total_output(self):
        base, candidate, *_, rows = self.make_runs()
        for row in rows:
            row["reasoning_tokens"] = 20
        (candidate / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        result = c.analyze(base, candidate, self.split, resamples=100)
        self.assertAlmostEqual(result["macro"]["token_reduction"], 0.25)
        thinking = result["macro"]["thinking_tokens"]
        self.assertAlmostEqual(thinking["reduction"], 0.75)
        self.assertEqual(thinking["mean_delta"], -60)
        self.assertEqual(thinking["aggregation"], "ratio_of_equal_domain_means")
        self.assertTrue(thinking["includes_incorrect_and_truncated"])

    def test_thinking_counts_cannot_be_missing_or_impossible(self):
        _, _, meta, new_meta, base, candidate = self.make_runs()
        for value in (None, -1, 76, True, 1.5):
            broken = [{**candidate[0], "reasoning_tokens": value}] + candidate[1:]
            with self.assertRaisesRegex(ValueError, "reasoning token"):
                c.check_runs(meta, new_meta, self.manifest, "tuning", base, broken)

    def test_zero_thinking_denominator_is_not_a_saving(self):
        result = c.thinking_comparison(0, 20, "ratio_of_equal_domain_means")
        self.assertIsNone(result["reduction"])
        self.assertEqual(result["mean_delta"], 20)
        self.assertEqual(c.thinking_comparison(80, 0, "domain")["reduction"], 1)

    def test_nonsignificance_not_preservation(self):
        self.assertEqual(c.preservation(0.001, [-0.02, 0.02])["classification"], "inconclusive")
        self.assertEqual(c.preservation(0, [0, 0])["classification"], "inconclusive_degenerate_bootstrap")
        self.assertEqual(c.preservation(0.02, [0.005, 0.04])["classification"], "supported_at_stated_margin")
        self.assertEqual(c.preservation(-0.03, [-0.04, -0.02], 1)["classification"], "evidence_of_regression_beyond_margin")
        self.assertTrue(c.preservation(-0.005, [-0.02, 0.01], 1)["point_estimate_within_margin"])
        self.assertEqual(c.preservation(-0.005, [-0.02, 0.01], 1)["classification"], "inconclusive")

    def test_confirmation_lock_and_output_provenance_required(self):
        base, candidate, *_ = self.make_runs("confirmation")
        with self.assertRaises(ValueError):
            c.analyze(base, candidate, self.split, "confirmation", resamples=200)
        adapter = self.root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
        (adapter / "tokenizer.json").write_text("{}")
        lock = c.lock_candidate(self.split, "candidate", adapter)
        self.assertIn("tokenizer.json", lock["adapter_files_sha256"])
        lock_path = self.root / "lock.json"
        c.write_json(lock_path, lock)
        with self.assertRaises(ValueError):
            c.analyze(base, candidate, self.split, "confirmation", 200, 42, lock_path, adapter)
        metadata = json.loads((candidate / "results.json").read_text())
        metadata["study"] = {k: lock[k] for k in ("candidate", "adapter_files_sha256", "split_manifest_sha256")}
        (candidate / "results.json").write_text(json.dumps(metadata))
        result = c.analyze(base, candidate, self.split, "confirmation", 200, 42, lock_path, adapter)
        self.assertEqual(result["inference_role"], "confirmatory_primary")
        (adapter / "tokenizer.json").write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            c.analyze(base, candidate, self.split, "confirmation", 200, 42, lock_path, adapter)


if __name__ == "__main__":
    unittest.main()
