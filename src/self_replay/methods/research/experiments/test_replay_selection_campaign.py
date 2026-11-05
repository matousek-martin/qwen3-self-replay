import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import replay_selection_campaign as batch
from experiments import replay_selection as recipe


class CampaignTests(unittest.TestCase):
    def test_same_worker_evaluation_config_for_each_seed(self):
        p=recipe.load_protocol()
        for seed in p['training_seeds']:
            c=batch.evaluation_config(p,seed)
            self.assertEqual(c['candidates'],['base',f'control-seed{seed}',f'mixed-seed{seed}'])
            self.assertEqual(c['subset'],'tuning')
            self.assertEqual(c['samples'],2)
        self.assertIn('{seed}',p['evaluation']['config']['candidates'][1])

    def test_partial_or_no_updates_training_cannot_be_scored(self):
        p=recipe.load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'status.json'
            for value in ({'status':'partial','completed_groups':128,'optimizer_updates':128},
                          {'status':'completed','completed_groups':127,'optimizer_updates':127},
                          {'status':'completed','completed_groups':128,'optimizer_updates':0}):
                path.write_text(json.dumps(value))
                with self.assertRaises(RuntimeError): batch.training_status(path,p)
            path.write_text(json.dumps({'status':'completed','completed_groups':128,'optimizer_updates':120}))
            self.assertEqual(batch.training_status(path,p)['optimizer_updates'],120)

    def test_existing_jobs_never_duplicate_or_retry_failed_rental(self):
        self.assertEqual(batch.existing_action({}),'run')
        self.assertEqual(batch.existing_action({'pod_id':'p','identity_confirmed':True}),'watch')
        self.assertEqual(batch.existing_action({'released_at':1,'state':'complete'}),'verify')
        for job in ({'state':'allocation_waiting'}, {'pod_id':'p'}, {'released_at':1,'state':'failed'},
                    {'released_at':1,'state':'allocation_unavailable'}):
            with self.assertRaises(RuntimeError): batch.existing_action(job)

    def test_source_change_rejected_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'source.py').write_text('original')
            launch={'source_sha256':{'source.py':batch.digest(root/'source.py')}}
            with patch.object(batch,'ROOT',root):
                batch.verify_sources(launch)
                (root/'source.py').write_text('changed')
                with self.assertRaises(RuntimeError): batch.verify_sources(launch)

    def test_worker_path_mapping_keeps_registered_hashes(self):
        p=recipe.load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rel in p['input_sha256']:
                target=rel.replace('lab/experiments/data/','data/',1)
                if target=='lab/gpu-requirements.lock': target='requirements.lock'
                path=root/target;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((recipe.ROOT/rel).read_bytes())
            with patch.object(recipe,'ROOT',root):
                self.assertEqual(recipe.load_protocol(worker=True),p)

    def test_worker_runs_both_arms_before_one_paired_evaluation(self):
        from experiments import pilot
        p=recipe.load_protocol()
        launch={'seeds':[17], 'protocol_sha256':'locked'}
        commands=[]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); output=root/'output';output.mkdir()
            (root/'data/study').mkdir(parents=True)
            (root/'data/study/splits.json').write_text('{}')
            def run(command, **kwargs):
                command=list(map(str,command));commands.append(command)
                if 'experiments.rl_training.train' in command:
                    target=Path(command[command.index('--out')+1]);(target/'adapter').mkdir(parents=True)
                    (target/'status.json').write_text(json.dumps({'status':'completed','completed_groups':128,'optimizer_updates':128}))
                    (target/'config.json').write_text(json.dumps({'replay_reward':'fixed','replay_admission':command[command.index('--replay-admission')+1]}))
                if 'experiments.study_eval' in command:
                    target=Path(command[command.index('--out')+1]);target.mkdir()
                    (target/'study-result.json').write_text(json.dumps({'complete':True,'completed_candidates':['base','control-seed17','mixed-seed17']}))
            with patch.object(batch,'ROOT',root), patch.object(batch,'read_launch',return_value=launch), \
                 patch.object(recipe,'load_protocol',return_value=p), \
                 patch.object(batch.subprocess,'check_output',return_value='NVIDIA GeForce RTX 4090'), \
                 patch.object(batch.subprocess,'run',side_effect=run), patch.object(pilot,'validate') as validate:
                batch.worker(17)
            training=[c for c in commands if 'experiments.rl_training.train' in c]
            self.assertEqual(len(training),2)
            self.assertEqual([c[c.index('--replay-admission')+1] for c in training],['all_correct','verified_correct'])
            self.assertEqual(validate.call_count,2)
            evaluations=[c for c in commands if 'experiments.study_eval' in c]
            self.assertEqual(len(evaluations),1)
            config=json.loads((output/'evaluation-inputs/evaluation.json').read_text())
            self.assertEqual(config['candidates'],['base','control-seed17','mixed-seed17'])
            self.assertTrue(json.loads((output/'role-result.json').read_text())['development_complete'])

    def test_worker_module_import_does_not_require_controller_only_runpod(self):
        import subprocess
        import sys
        code = '''
import importlib.abc, sys
class RejectController(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('runpod', 'experiments.cloud'):
            raise ImportError('Controller-only module imported on worker: ' + fullname)
sys.meta_path.insert(0, RejectController())
import experiments.replay_selection_campaign
'''
        subprocess.run([sys.executable, '-c', code], cwd=recipe.ROOT, check=True)
