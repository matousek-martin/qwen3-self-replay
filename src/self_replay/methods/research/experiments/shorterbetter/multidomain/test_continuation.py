import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.shorterbetter.stopping import apply_quality_policy, decide, DEFAULTS
from experiments.shorterbetter.test_scaled import point
from experiments.shorterbetter.prepare import sha
from experiments.shorterbetter.multidomain.continuation import validate_lineage, replace_baseline
from experiments.shorterbetter.multidomain.test_full_run import FullRunTests


class ContinuationTests(unittest.TestCase):
    def test_all_quality_stops_monitor_only_but_other_stops_remain(self):
        p={'stopping':{'quality_stopping_enabled':False},'execution':{'multidomain_full_training':True},
           'authorization':{'quality_stopping_disabled_by_user':True}}
        for reason in ('persistent_regression','no_useful_progress_within_window','useful_plateau_needs_confirmation'):
            d=apply_quality_policy({'stop':True,'reason':reason,'best_eligible_step':None},p)
            self.assertFalse(d['stop']);self.assertTrue(d['quality_stop_recommendation']['stop'])
        for reason in ('deadline_or_stop_request','registered_updates_completed'):
            self.assertTrue(apply_quality_policy({'stop':True,'reason':reason},p)['stop'])
        d={'stop':True,'reason':'persistent_regression'}
        self.assertEqual(apply_quality_policy(d,{}),d)
        p['authorization']={}
        with self.assertRaises(ValueError):apply_quality_policy(d,p)
        bad=point(0);bad['accuracy']=float('nan')
        with self.assertRaises(ValueError):decide({'checkpoints':[bad]},DEFAULTS)

    def test_parent_rejects_changed_reward_data_seed_or_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'data/parent';source.mkdir(parents=True)
            original=FullRunTests().protocol();original.update(id='old',stopping={},model={'revision':'fixed'})
            original['data']['manifest_sha256']='data-hash'
            (source/'protocol.json').write_text(json.dumps(original))
            p=copy.deepcopy(original);p['execution']['full_run_continuation']=True
            p['stopping']['quality_stopping_enabled']=False
            p['authorization']['quality_stopping_disabled_by_user']=True
            p['continuation_parent']={'directory':'data/parent','job_name':'old','protocol_sha256':sha(source/'protocol.json'),'optimizer_step':160}
            prior={'protocol_sha256':sha(source/'protocol.json'),'job_name':'old','data_manifest_sha256':'data-hash'}
            self.assertEqual(validate_lineage(p,prior,{'global_step':160},root),source)
            for key, value in [('arms',{}),('seeds',[18]),('model',{'revision':'other'}),('training',{})]:
                bad=copy.deepcopy(p);bad[key]=value
                with self.assertRaises((ValueError,KeyError)):validate_lineage(bad,prior,{'global_step':160},root)
            with self.assertRaises(ValueError):validate_lineage(p,prior,{'global_step':120},root)

    def test_fresh_worker_baseline_replaces_only_old_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'training/rank-000';new=root/'baseline/rank-000'
            old.mkdir(parents=True);new.mkdir(parents=True)
            historical=[{'optimizer_step':0,'split':'development','marker':'old'},
                        {'optimizer_step':0,'split':'train'}, {'optimizer_step':160,'split':'development'}]
            fresh=[{'optimizer_step':0,'split':'development','marker':'new'}]*128
            (old/'rollouts.jsonl').write_text('\n'.join(map(json.dumps,historical)))
            (new/'rollouts.jsonl').write_text('\n'.join(map(json.dumps,fresh)))
            replace_baseline(root/'training',root/'baseline',world=1)
            result=[json.loads(s) for s in (old/'rollouts.jsonl').read_text().splitlines()]
            self.assertEqual(result,fresh+historical[1:])

if __name__=='__main__':unittest.main()

class ContinuationBudgetTests(unittest.TestCase):
    def test_funding_override_requires_released_verified_parent_and_remaining_cap(self):
        from experiments.shorterbetter.budget import ticket
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'authorizations').mkdir()
            parent='shorterbetter-scaled-parent';name='shorterbetter-scaled-continuation'
            auth={'job_name':name,'authorized':True,'scope':'shorterbetter_vast_bounded_v1',
                  'provider':'vast','gpu':'NVIDIA A100-SXM4-80GB','gpu_count':4,'offer_id':123,
                  'disk_gb':65,'max_cost_usd':397.,'max_runtime_hours':60,'max_gpu_rate':1.2,
                  'source_user_instruction':'resume without quality stopping',
                  'full_training_authorized':True,'rollout_batches':300,
                  'continuation_parent_job':parent,'quality_stopping_disabled_by_user':True,
                  'incremental_funding':{'minimum_start_credit_usd':40,'warning_credit_usd':30,'stop_credit_usd':12,'release_credit_usd':3}}
            (root/'authorizations'/f'{name}.json').write_text(json.dumps(auth))
            (root/'authorizations'/f'{parent}.json').write_text(json.dumps({'max_cost_usd':450,'full_training_authorized':True}))
            old={'released_at':1,'backup_verified':True,'provider':'vast','charged_estimate_usd':53}
            def ledger(): (root/'ledger.json').write_text(json.dumps({'jobs':{parent:old}}))
            job={**auth,'entry':'experiments/shorterbetter/run-scaled.sh','hardware_scope':'shorterbetter-scaled-v1'}
            ledger();self.assertIsNotNone(ticket(root,name,job))
            for field,value in [('released_at',None),('backup_verified',False),('charged_estimate_usd',54)]:
                saved=old[field];old[field]=value;ledger()
                self.assertIsNone(ticket(root,name,job));old[field]=saved

class SuccessfulExitTests(unittest.TestCase):
    def test_success_waits_for_cuda_and_all_ranks_before_exit(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from experiments.shorterbetter.train import finish_successfully
        events=[]
        fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('cuda')))
        trainer=SimpleNamespace(accelerator=SimpleNamespace(wait_for_everyone=lambda:events.append('ranks')))
        with patch.dict(sys.modules,{'torch':fake}), patch('os._exit',side_effect=lambda code:events.append(('exit',code))):
            finish_successfully(trainer)
        self.assertEqual(events,['cuda','ranks',('exit',0)])

    def test_failed_synchronization_does_not_claim_success(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch,Mock
        from experiments.shorterbetter.train import finish_successfully
        sync=Mock(side_effect=RuntimeError('CUDA failure'))
        fake=SimpleNamespace(cuda=SimpleNamespace(synchronize=sync))
        trainer=SimpleNamespace(accelerator=SimpleNamespace(wait_for_everyone=Mock()))
        with patch.dict(sys.modules,{'torch':fake}), patch('os._exit') as exit_process:
            with self.assertRaisesRegex(RuntimeError,'CUDA failure'):finish_successfully(trainer)
            exit_process.assert_not_called()
