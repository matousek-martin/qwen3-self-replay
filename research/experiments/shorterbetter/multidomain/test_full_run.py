import copy
import json
from pathlib import Path
import unittest

from experiments.shorterbetter.config import training_arguments, assert_launch_ready
from experiments.shorterbetter.multidomain.full_run import technical_protocol, validate


class FullRunTests(unittest.TestCase):
    def protocol(self):
        p=json.loads((Path(__file__).resolve().parents[3]/'studies/shorterbetter/protocol.json').read_text())
        p['seeds']=[17];p['arms']={'shorterbetter':p['arms']['shorterbetter']}
        p['authorization']={'full_training_authorized':True}
        p['training'].update(rollout_batches=300,optimizer_updates_total=1200,parameter_dtype='float32')
        p['data'].update(cover_all_training_rows=True,monitor_subset_questions=64,monitor_ids=list(range(64)))
        p['monitoring']['every_rollout_batches']=10
        p['execution']={'scaled':True,'multidomain':True,'multidomain_full_training':True,'phase':'training',
                        'gpu':'NVIDIA A100-SXM4-80GB','world_size':4,'deterministic_training':True,
                        'deterministic_ddp':True,'shutdown_reserve_seconds':1200,'terminal_evaluation_required':True}
        return p

    def test_full_exposure_and_validation(self):
        p=self.protocol();a=training_arguments(p,'shorterbetter',17,'/tmp/test')
        self.assertEqual(a['max_steps'],1200);self.assertEqual(a['generation_batch_size'],512)
        self.assertEqual(a['eval_steps'],40);self.assertEqual(a['gradient_accumulation_steps']*4,128)
        for change in ({'world_size':2},{'gpu':'NVIDIA H100 80GB HBM3'},{'phase':'benchmark'},
                       {'terminal_evaluation_required':False}):
            q=copy.deepcopy(p);q['execution'].update(change)
            with self.assertRaises(ValueError):validate(q)

    def test_technical_run_does_not_change_full_registration(self):
        p=self.protocol();q=technical_protocol(p);a=training_arguments(q,'shorterbetter',17,'/tmp/test')
        self.assertEqual(p['training']['rollout_batches'],300)
        self.assertEqual(q['data']['monitor_ids'],[i for start in range(0,64,16) for i in range(start,start+4)])
        self.assertEqual(a['max_steps'],8);self.assertEqual(a['save_total_limit'],2)
        self.assertTrue(a['eval_on_start']);self.assertEqual(a['eval_steps'],4)

    def test_full_run_cannot_use_cpu_only_readiness(self):
        p=self.protocol();p['status']='locked';p['budget']={'cap_authorized':True}
        ready={k:True for k in ('data_audit','reward_parity','backend_cpu_parity','distributed_cpu_check',
                'stopping_tests','budget_reservation','pilot_cap_enforced','backup_release_integration',
                'four_domain_verifiers','balanced_sampler','data_domains')}
        with self.assertRaisesRegex(RuntimeError,'resume_equivalence'):assert_launch_ready(p,ready)


if __name__=='__main__':unittest.main()
