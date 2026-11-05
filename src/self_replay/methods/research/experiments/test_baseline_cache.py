import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments import baseline_cache as cache, study_eval as study
from experiments.full_replay_eval import COUNTS


class BaselineCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.repo=self.root/'ThinkingCap-Test'
        questions=self.repo/'questions';questions.mkdir(parents=True)
        (self.repo/'evaluate.py').write_text('# frozen evaluator\n');(self.repo/'compare.py').write_text('# comparator\n')
        self.rows=[]
        for domain,n in COUNTS.items():
            for i in range(n):self.rows.append(dict(dataset=domain,id=str(i),seed_index=len(self.rows)))
        (questions/'questions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        study.write_json(questions/'manifest.json',{'version':1})
        self.runtime={p:'test-1' for p in cache.PACKAGES};self.identity='GPU-original, NVIDIA GeForce RTX 4090'
        self.contract=cache.contract(self.repo,questions,'NVIDIA GeForce RTX 4090',self.runtime,compat='patched')
        self.source=self.root/'source';self.source.mkdir()
        records=[]
        for row in self.rows:
            for sample in range(8):
                records.append(dict(dataset=row['dataset'],id=row['id'],sample_index=sample,
                    seed=9001+row['seed_index']*1000+sample,correct=True,output_tokens=1))
        (self.source/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        self.result=dict(model=study.BASE_MODEL,revision=study.BASE_REVISION,adapter=None,selection='full',
            cohort_total=2759,cohort_sha256=self.contract['cohort_sha256'],samples=8,seed=9001,
            sampling=study.SAMPLING,max_tokens=self.contract['max_tokens'],max_model_len=12288,
            smoke=False,sandbox={'mode':'rlimit'},failed_benchmarks={},results={d:{'n':n*8} for d,n in COUNTS.items()})
        study.write_json(self.source/'results.json',self.result)
        self.manifest=dict(schema_version=1,key=cache.key(self.contract),contract=self.contract,
            files_sha256={f:cache.digest(self.source/f) for f in cache.FILES},source={'job':'original','gpu_identity':self.identity})
        study.write_json(self.source/'baseline-cache-entry.json',self.manifest)
        self.catalog=self.root/'lab/experiments/baseline-cache'
        self.out=self.root/'new-evaluation/runs/base'
        self.command=['python',self.repo/'evaluate.py','--questions',questions,'--out',self.out,
                      '--samples','8','--seed','9001','--workers','4']

    def run_cached(self,runner,identity='GPU-different, NVIDIA GeForce RTX 4090'):
        with patch.object(study,'ROOT',self.root),patch.object(cache,'runtime_context',return_value=(identity,'rlimit',self.runtime,'patched')):
            return cache.run_or_reuse(self.command,self.root/'test.log',10,runner)

    def test_different_physical_gpu_same_type_reuses_without_inference(self):
        cache.publish(self.source,self.catalog)
        study.write_json(self.out.parent.parent/'provenance.json',{'matched_baseline_regenerated':True})
        def forbidden(*args):self.fail('Baseline inference must not be launched on a cache hit')
        self.assertTrue(self.run_cached(forbidden))
        self.assertEqual(cache.digest(self.out/'records.jsonl'),cache.digest(self.source/'records.jsonl'))
        proof=cache.read(self.out.parent.parent/'provenance.json')
        self.assertFalse(proof['matched_baseline_regenerated']);self.assertFalse(proof['same_physical_worker_required'])
        self.assertEqual(cache.status_fields(self.out)['gpu_identity'],self.identity)

    def test_gpu_type_protocol_and_software_change_key(self):
        original=cache.key(self.contract)
        for update in [dict(gpu_model='NVIDIA H100'),dict(revision='new'),dict(cohort_sha256='new'),
                       dict(seed=42),dict(samples=4),dict(runtime={'torch':'new'}),dict(loading='local')]:
            self.assertNotEqual(original,cache.key({**self.contract,**update}))

    def test_new_gpu_type_generates_and_publishes_once(self):
        cache.publish(self.source,self.catalog);calls=[]
        def generate(*args):
            calls.append(True);self.out.mkdir(parents=True)
            import shutil
            for f in cache.FILES:shutil.copy2(self.source/f,self.out/f)
        self.assertTrue(self.run_cached(generate,'GPU-new, NVIDIA H100'))
        self.assertEqual(len(calls),1)
        cache.verify_entry(self.out)
        cache.ingest(self.out.parent.parent,self.catalog)
        self.assertEqual(len(list(self.catalog.glob('*/baseline-cache-entry.json'))),2)

    def test_required_cache_miss_never_generates(self):
        study.write_json(self.out.parent.parent/'provenance.json',{'config':{'baseline_cache_required':True}})
        with self.assertRaisesRegex(RuntimeError,'paid regeneration disabled'):
            self.run_cached(lambda *args:self.fail('Must not regenerate required cache'))

    def test_corrupt_cache_fails_without_spending_on_replacement(self):
        cached=cache.publish(self.source,self.catalog);(cached/'records.jsonl').write_text('corrupt')
        def forbidden(*args):self.fail('Do not silently regenerate a corrupt known baseline')
        with self.assertRaisesRegex(ValueError,'checksum'):self.run_cached(forbidden)

    def test_incomplete_records_not_publishable(self):
        (self.source/'records.jsonl').write_text('')
        self.manifest['files_sha256']['records.jsonl']=cache.digest(self.source/'records.jsonl')
        study.write_json(self.source/'baseline-cache-entry.json',self.manifest)
        with self.assertRaisesRegex(ValueError,'Incomplete'):cache.publish(self.source,self.catalog)

    def test_candidates_and_smoke_are_never_baseline_cache_hits(self):
        self.assertIsNone(cache.options([*self.command,'--adapter','adapter']))
        self.assertIsNone(cache.options([*self.command,'--smoke']))
        changed=list(self.command);changed[changed.index('--out')+1]=self.root/'runs/candidate'
        self.assertIsNone(cache.options(changed))

    def test_publish_is_immutable_and_payload_stage_is_verified(self):
        destination=cache.publish(self.source,self.catalog)
        self.assertEqual(destination,cache.publish(self.source,self.catalog))
        release=self.root/'release';(release/'experiments').mkdir(parents=True)
        (release/'experiments/study_eval.py').write_text('from . import baseline_cache')
        (release/'experiments/baseline_cache.py').write_text('# package')
        self.assertEqual(cache.stage(release,self.catalog),1)
        cache.verify_entry(release/'lab/experiments/data/baseline-cache'/destination.name)

    def test_obsolete_and_stale_payloads_fail_before_rental(self):
        release=self.root/'release';(release/'experiments').mkdir(parents=True)
        runner=release/'experiments/study_eval.py';runner.write_text('# old runner')
        with self.assertRaisesRegex(RuntimeError,'Legacy'):
            cache.check_launch(release,'experiments/run-full-eval.sh',self.catalog,'NVIDIA GeForce RTX 4090')
        (release/'experiments/run-isolated-seed17.sh').write_text('python -m experiments.teacher_full_eval')
        with self.assertRaisesRegex(RuntimeError,'Legacy'):
            cache.check_launch(release,'experiments/run-isolated-seed17.sh',self.catalog,'NVIDIA GeForce RTX 4090')
        cache.check_launch(release,'experiments/train.sh',self.catalog,'NVIDIA GeForce RTX 4090')
        runner.write_text('from . import baseline_cache');(release/'experiments/baseline_cache.py').write_text('# module')
        cache.publish(self.source,self.catalog)
        with self.assertRaisesRegex(RuntimeError,'updated after'):
            cache.check_launch(release,'experiments/run-full-eval.sh',self.catalog,'NVIDIA GeForce RTX 4090')
        cache.stage(release,self.catalog)
        cache.check_launch(release,'experiments/run-full-eval.sh',self.catalog,'NVIDIA GeForce RTX 4090')


if __name__=='__main__':unittest.main()
