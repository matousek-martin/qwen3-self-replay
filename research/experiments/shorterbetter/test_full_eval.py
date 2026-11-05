import json
from pathlib import Path
import tempfile
import unittest
from experiments import study_eval as study
from experiments.shorterbetter.full_eval import inputs,digest


class FullModelEvalTests(unittest.TestCase):
    def test_frozen_candidate_and_evaluator_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'candidate';model.mkdir()
            (model/'model.safetensors').write_bytes(b'fixture-only')
            (model/'config.json').write_text(json.dumps({'dtype':'bfloat16','model_type':'qwen3'}))
            q=study.REPO/'questions'
            c={'evaluation_only':True,'training_authorized':False,'samples':8,'evaluation_seed':9001,
               'candidates':['base','shorterbetter-final'],'model':study.BASE_MODEL,'revision':study.BASE_REVISION,
               'cohort_sha256':digest(q/'questions.jsonl'),'questions_manifest_sha256':digest(q/'manifest.json'),
               'evaluator_files_sha256':study.evaluator_hashes(),
               'candidate':{'path':'candidate','files_sha256':{f.name:digest(f) for f in model.iterdir()},
                            'original_weights_sha256':digest(model/'model.safetensors'),'optimizer_step':426,
                            'original_model_config':{'dtype':'float32','model_type':'qwen3'}}}
            path=root/'evaluation.json';path.write_text(json.dumps(c))
            self.assertEqual(len(inputs(path)[2]),2759)
            c['candidate']['optimizer_step']=160;path.write_text(json.dumps(c))
            self.assertEqual(inputs(path)[0]['candidate']['optimizer_step'],160)
            c['candidate']['optimizer_step']=80;path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'authorized checkpoint'):inputs(path)
            c['paired_short_evaluation']=True
            c['candidate']['source_training_job']='shorterbetter-scaled-paired-control-test'
            path.write_text(json.dumps(c));self.assertEqual(inputs(path)[0]['candidate']['optimizer_step'],80)
            c['candidate']['optimizer_step']=81;path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'authorized checkpoint'):inputs(path)
            c['paired_short_evaluation']=False
            c['candidate']['optimizer_step']=426
            c['samples']=1;path.write_text(json.dumps(c))
            with self.assertRaises(ValueError):inputs(path)
            c['samples']=8;path.write_text(json.dumps(c))
            (model/'model.safetensors').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'candidate changed'):inputs(path)

    def test_local_full_weights_records_keep_official_recipe_and_coverage_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);rows=[{'dataset':d,'id':d,'seed_index':i} for i,d in enumerate(sorted(study.DATASETS))]
            records=[{'dataset':r['dataset'],'id':r['id'],'sample_index':s,'seed':9001+r['seed_index']*1000+s,
                      'correct':True,'output_tokens':12} for r in rows for s in range(8)]
            result={'model':'/local/candidate','revision':None,'samples':8,'seed':9001,'sampling':study.SAMPLING,
                    'max_tokens':{d:8192 if d=='mbpp' else 4096 for d in study.DATASETS},'max_model_len':12288,
                    'cohort_sha256':'fixture','cohort_total':len(rows),'smoke':False,'selection':'full',
                    'failed_benchmarks':[],'results':{d:{'n':8} for d in study.DATASETS}}
            (out/'results.json').write_text(json.dumps(result))
            (out/'records.jsonl').write_text('\n'.join(map(json.dumps,records)))
            study.validate_records(out,rows,8,'fixture','full',model='/local/candidate',revision=None)
            with self.assertRaisesRegex(ValueError,'model'):study.validate_records(out,rows,8,'fixture','full')
            (out/'records.jsonl').write_text('\n'.join(map(json.dumps,records[:-1])))
            with self.assertRaisesRegex(ValueError,'coverage'):
                study.validate_records(out,rows,8,'fixture','full',model='/local/candidate',revision=None)

if __name__=='__main__':unittest.main()
