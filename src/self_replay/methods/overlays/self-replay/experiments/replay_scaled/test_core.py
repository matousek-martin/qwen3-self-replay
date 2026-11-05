import json,tempfile,unittest
import time
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
from experiments.replay_scaled import core
from experiments.replay_scaled.metrics import summarize
from experiments.replay_scaled.notify import deliver_once

class CoreTests(unittest.TestCase):
    def test_base_restart_repair_refuses_any_trained_state(self):
        from ops import replay_scaled as controller
        with tempfile.TemporaryDirectory() as tmp:
            reg=Path(tmp)/'reg';state=Path(tmp)/'cloud';name='replay-scaled-training-test'
            core.write(reg/'campaign.json',dict(state='stopped',job_name=name,full_training_authorized=True))
            core.write(state/'ledger.json',dict(jobs={name:{}}))
            for current,updates in [(1,0),(0,1),(32,16)]:
                core.write(state/name/'output/training/progress.json',dict(current=current,optimizer_updates=updates))
                with patch.object(controller,'REG',reg),patch.object(controller.cloud,'STATE',state):
                    with self.assertRaisesRegex(RuntimeError,'Trained state exists'):
                        controller.prepare('training',30,repair=True)
    def test_archive_grace_stays_within_existing_replay_budget(self):
        from experiments import cloud
        probe=dict(worker_status='dead',exit_marker=dict(status='valid'))
        with patch('experiments.shorterbetter.budget.ticket',return_value={'scope':'replay_vast_bounded_v1'}):
            self.assertTrue(cloud.completion_archive_authorized('replay-scaled-test',{'hard_deadline':time.time()+600},probe))
            self.assertFalse(cloud.completion_archive_authorized('replay-scaled-test',{'hard_deadline':time.time()+100},probe))
            self.assertFalse(cloud.completion_archive_authorized('replay-scaled-test',{'hard_deadline':time.time()+600},{'worker_status':'alive'}))
    def test_checkpoints_do_not_trigger_extra_validation(self):
        cfg=dict(save_every=32,validate_every=256,milestones=[128,512,1024,1920,3840])
        self.assertEqual(core.checkpoint_boundaries(31,3840,cfg),(False,False))
        self.assertEqual(core.checkpoint_boundaries(32,3840,cfg),(True,False))
        self.assertEqual(core.checkpoint_boundaries(64,3840,cfg),(True,False))
        for n in [128,256,1920,3840]:self.assertEqual(core.checkpoint_boundaries(n,3840,cfg),(True,True))
        self.assertEqual(core.checkpoint_boundaries(2,4,cfg,pilot=True),(True,True))
        saves=[n for n in range(1,3841) if core.checkpoint_boundaries(n,3840,cfg)[0]]
        validations=[n for n in range(1,3841) if core.checkpoint_boundaries(n,3840,cfg)[1]]
        self.assertEqual(len(saves),120);self.assertEqual(len(validations),17)

    def test_prior_test_budget_requires_release_or_reconciled_absence(self):
        from ops.replay_scaled import previous_pilot_spend
        complete=dict(released_at=1,backup_verified=True,charged_estimate_usd=1.5)
        absent=dict(released_at=2,safely_no_pod=True,state='create_absent_reconciled',charged_estimate_usd=0)
        self.assertEqual(previous_pilot_spend({'a':complete,'b':absent},['a','b']),1.5)
        prelaunch=dict(released_at=3,state='failed',startup_failed_before_launch=True,charged_estimate_usd=.12)
        self.assertEqual(previous_pilot_spend({'a':complete,'b':prelaunch},['a','b']),1.62)
        for bad in [{**absent,'released_at':None},{**absent,'state':'creating'},
                    {**absent,'pod_id':'vast:123'},{**complete,'backup_verified':False},
                    {**prelaunch,'worker_launch_attempted_at':1},{**prelaunch,'released_at':None}]:
            with self.assertRaises(RuntimeError):previous_pilot_spend({'x':bad},['x'])
    def test_no_call_rows_are_never_randomly_dropped(self):
        from experiments.replay_scaled.prepare import select_rows
        rows=[dict(task=d,id=f'{d}{i}',answer=json.dumps({'calls':[] if d=='tools' and i<2 else [{'name':'x'}]})) for d in core.DOMAINS for i in range(10)]
        result=select_rows(rows,5,17)
        self.assertEqual(len(result),20)
        self.assertEqual(sum(r['task']=='tools' and json.loads(r['answer'])['calls']==[] for r in result),2)
    def test_technical_fixtures_cover_domains_and_are_not_benchmarks(self):
        from experiments.replay_scaled.pilot import fixtures
        rows=fixtures();self.assertEqual({r['task'] for r in rows},set(core.DOMAINS))
        self.assertTrue(all(r['id'].startswith('technical-') for r in rows))
    def test_historical_rewards(self):
        r,a=core.policy_rewards([True,False,True,False],[2048,1,100,400],1024)
        self.assertEqual(r[0],.9);self.assertEqual(r[1],0);self.assertGreater(min(r[0],r[2]),max(r[1],r[3]))
        self.assertAlmostEqual(sum(a),0)
        self.assertEqual(core.policy_rewards([False]*4,[1,20,500,2048],1024)[1],[0]*4)
    def test_admission_and_fifo(self):
        b={};bad=[dict(correct=True,ids=[1]),dict(correct=False,ids=[2])]
        core.admission(b,'bad',[0],bad);self.assertFalse(b)
        core.admission(b,'first',[0],[dict(correct=True,ids=[1,2])]*4,1)
        core.admission(b,'next',[0],[dict(correct=True,ids=[1])]*4,1)
        self.assertEqual(list(b),['next'])
    def test_balance_unique(self):
        rows=[dict(task=d,id=f'{d}{i}') for d in core.DOMAINS for i in range(10)]
        a=core.balanced(rows,5,17);self.assertEqual(a,core.balanced(rows,5,17));self.assertEqual(len(a),20)
        for start in range(0,20,4):self.assertEqual({r['task'] for r in a[start:start+4]},set(core.DOMAINS))
    def test_scope_not_global(self):
        a=SimpleNamespace(hardware_scope=core.SCOPE,name='replay-scaled-test',entry=core.ENTRY,gpu=core.GPU,gpu_count=2,max_cost_usd=10)
        self.assertEqual(core.allocation_spec(a)['count'],2)
        for field,value in [('name','unrelated'),('gpu_count',8),('gpu','NVIDIA H100 80GB HBM3')]:
            old=getattr(a,field);setattr(a,field,value)
            with self.assertRaises(ValueError):core.allocation_spec(a)
            setattr(a,field,old)
        lease=dict(hardware_scope=core.SCOPE,job_name='replay-scaled-test',gpu=core.GPU,gpu_count=2)
        self.assertEqual(len(core.verify_worker([core.GPU]*2,lease)),2)
        with self.assertRaises(RuntimeError):core.verify_worker(['NVIDIA GeForce RTX 4090']*2,lease)
    def test_named_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            name='replay-scaled-test';v=dict(job_name=name,authorized=True,scope='replay_vast_bounded_v1',provider='vast',hardware_scope=core.SCOPE,entry=core.ENTRY,gpu=core.GPU,gpu_count=2,offer_id=123,disk_gb=60,max_cost_usd=10,max_runtime_hours=2,max_gpu_rate=1.5,phase='pilot',source_user_instruction='test')
            core.write(Path(tmp)/'authorizations'/(name+'.json'),v)
            self.assertIsNotNone(core.ticket(tmp,name,v))
            self.assertIsNone(core.ticket(tmp,name,{**v,'offer_id':456}))
            self.assertIsNone(core.ticket(tmp,name,{**v,'max_cost_usd':11}))
    def test_monitor_pairing(self):
        rows=[dict(task='math',id='x',sample=i,correct=True,total_tokens=100,truncated=False) for i in range(4)]
        s=summarize(rows,rows)['math'];self.assertEqual(s['accuracy_delta_ci'],[0,0]);self.assertEqual(s['token_saving'],0)
        with self.assertRaises(ValueError):summarize(rows,rows[:-1])
    def test_uncertain_notification_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def fail(message):calls.append(message);raise TimeoutError()
            with self.assertRaises(TimeoutError):deliver_once(Path(tmp)/'n.db','k','text',fail)
            self.assertFalse(deliver_once(Path(tmp)/'n.db','k','text',calls.append));self.assertEqual(calls,['text'])
