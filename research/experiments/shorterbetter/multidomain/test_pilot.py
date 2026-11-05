import collections,json,tempfile,unittest
from pathlib import Path
from experiments.shorterbetter.multidomain.rewards import verify,code_check,instruction_check,tool_check
from experiments.shorterbetter.multidomain.sampler import BalancedRepeatSampler
from experiments.shorterbetter.multidomain.pilot import projection
from experiments.shorterbetter.budget import ticket
from experiments.shorterbetter.config import training_arguments

class Tests(unittest.TestCase):
    def test_math_final_only(self):
        gold=json.dumps({'task':'math','value':'42'})
        self.assertTrue(verify('<think>work</think>\\boxed{42}',gold))
        self.assertFalse(verify('<think>\\boxed{42}</think>\\boxed{41}',gold))
    def test_code_actual_tests(self):
        t={'inputs':['2','8'],'outputs':['3','9']}
        self.assertTrue(code_check('print(int(input())+1)',t))
        self.assertFalse(code_check('print(3)',t))
        self.assertTrue(code_check('def f(x): return x+1',{'fn_name':'f','inputs':[[2],[8]],'outputs':[[3],[9]]}))
    def test_code_sandbox(self):
        t={'inputs':['2'],'outputs':['3']}
        for code in ['open("/etc/passwd").read(); print(3)','import socket; socket.socket(); print(3)','import os; os.fork(); print(3)',
                     'import __main__; print(__main__.payload["outputs"][0])']:
            self.assertFalse(code_check(code,t))
    def test_instruction_constraints(self):
        self.assertTrue(instruction_check('hello world',['lowercase','no_comma']))
        self.assertFalse(instruction_check('Hello, world',['lowercase','no_comma']))
        self.assertFalse(instruction_check('null',['json']))
        self.assertTrue(instruction_check('{"x":1}',['json']))
        self.assertFalse(instruction_check('', ['no_comma']))
    def test_tool_selection_and_irrelevance(self):
        calls=[{'name':'f','arguments':{'x':2}}]
        self.assertTrue(tool_check(json.dumps(calls),calls))
        self.assertFalse(tool_check('[{"name":"g","arguments":{"x":2}}]',calls))
        self.assertTrue(tool_check('[]',[]));self.assertFalse(tool_check(json.dumps(calls),[]))
    def test_balanced_real_grouping(self):
        data={'task':sum(([d]*32 for d in ('math','coding','tools','instructions')),[])}
        s=BalancedRepeatSampler(data,8,64,2,17);a=list(s)
        self.assertEqual(a,list(BalancedRepeatSampler(data,8,64,2,17)))
        for start in range(0,len(a),1024):
            self.assertEqual(a[start:start+512],a[start+512:start+1024])
            ids=a[start:start+512:8]
            self.assertEqual(collections.Counter(data['task'][i] for i in ids),{d:16 for d in data['task'][:1]+['coding','tools','instructions']})
            for k in range(start,start+512,8):self.assertEqual(len(set(a[k:k+8])),1)
    def test_scope_does_not_authorize_other_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);(state/'authorizations').mkdir();name='shorterbetter-scaled-multidomain-test'
            value={'scope':'shorterbetter_multidomain_pilot','authorized':True,'job_name':name,'gpu':'NVIDIA GeForce RTX 4090','gpu_count':1,'max_cost_usd':5,'max_runtime_hours':3,'max_gpu_rate':.89,'pilot_rollouts':3}
            (state/'authorizations'/f'{name}.json').write_text(json.dumps(value))
            job=dict(value,hardware_scope='shorterbetter-scaled-v1',entry='experiments/shorterbetter/run-scaled.sh')
            self.assertTrue(ticket(state,name,job))
            for change in [{'gpu_count':2},{'gpu':'NVIDIA A100-SXM4-80GB'},{'max_cost_usd':6}]:self.assertIsNone(ticket(state,name,{**job,**change}))
    def test_projection_does_not_claim_two_gpu_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'rank-000').mkdir()
            (out/'steps.jsonl').write_text('\n'.join(json.dumps({'optimizer_step':s,'time':t}) for s,t in [(4,200),(8,300),(12,400)]))
            (out/'rollout-starts.jsonl').write_text('\n'.join(json.dumps({'optimizer_step':s,'time':t}) for s,t in [(0,80),(4,200),(8,300)]))
            (out/'rank-000/metrics.jsonl').write_text(json.dumps({'eval_runtime':20}))
            r=projection(out,{'data':{'monitor_subset_questions':16}},{'created_at':0,'rate':.59})
            self.assertFalse(r['two4090_measured']);self.assertEqual(r['steady_rollout_samples'],2)
            self.assertLess(r['two4090_scenarios_hours']['ideal_2x'],r['single4090_hours_range'][0])

    def test_a100_named_ticket_is_bounded_and_specific(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);(state/'authorizations').mkdir();name='shorterbetter-scaled-multidomain-a100x2-test'
            auth={'scope':'shorterbetter_multidomain_a100x2_benchmark','authorized':True,'job_name':name,
                'gpu':'NVIDIA A100-SXM4-80GB','gpu_count':2,'max_cost_usd':10,'max_runtime_hours':2,
                'max_gpu_rate':1.79,'pilot_rollouts':2,'full_training_authorized':False}
            (state/'authorizations'/f'{name}.json').write_text(json.dumps(auth))
            job=dict(auth,hardware_scope='shorterbetter-scaled-v1',entry='experiments/shorterbetter/run-scaled.sh',created_at=100,hard_deadline=7300)
            self.assertTrue(ticket(state,name,job))
            for change in [{'gpu_count':4},{'gpu':'NVIDIA GeForce RTX 4090'},{'max_cost_usd':11},{'hard_deadline':7400},
                           {'completion_budget_scope':'finish_registered_1200_updates'}]:
                self.assertIsNone(ticket(state,name,{**job,**change}))

    def test_a100_config_preserves_exposure_and_reserves_evaluation(self):
        p=json.loads((Path(__file__).resolve().parents[3]/'studies/shorterbetter/protocol.json').read_text())
        p['training']['rollout_batches']=2
        p['execution']={'scaled':True,'phase':'benchmark','multidomain':True,'world_size':2,'gpu':'NVIDIA A100-SXM4-80GB',
            'accelerated_multidomain_benchmark':True,'shutdown_reserve_seconds':1200,'terminal_evaluation_required':True}
        a=training_arguments(p,'shorterbetter',17,'/tmp/test')
        self.assertEqual(a['max_steps'],8);self.assertEqual(a['generation_batch_size'],512)
        self.assertEqual(a['gradient_accumulation_steps']*2,128)
        p['execution']['terminal_evaluation_required']=False
        with self.assertRaises(ValueError):training_arguments(p,'shorterbetter',17,'/tmp/test')

    def test_deadline_requests_final_evaluation_before_save(self):
        from types import SimpleNamespace as S
        import torch,time
        from experiments.shorterbetter.control import ScaledControl
        with tempfile.TemporaryDirectory() as tmp:
            cb=ScaledControl(tmp,{'execution':{'terminal_evaluation_required':True,'shutdown_reserve_seconds':1200}},
                {'hard_deadline':time.time()+100,'job_name':'test'},S(device=torch.device('cpu')))
            state=S(global_step=3,is_world_process_zero=True)
            control=S(should_training_stop=False,should_save=False,should_evaluate=False)
            cb.on_step_end(S(eval_strategy='steps',max_steps=8),state,control)
            self.assertTrue(control.should_training_stop);self.assertTrue(control.should_evaluate);self.assertTrue(control.should_save)
            self.assertEqual(json.loads((Path(tmp)/'decision.json').read_text())['reason'],'deadline_or_stop_request')

    def test_a100_projection_is_actually_measured_for_100_and_300(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'rank-000').mkdir()
            (out/'steps.jsonl').write_text('\n'.join(json.dumps({'optimizer_step':s,'time':t}) for s,t in [(4,200),(8,300)]))
            (out/'rollout-starts.jsonl').write_text('\n'.join(json.dumps({'optimizer_step':s,'time':t}) for s,t in [(0,80),(4,200)]))
            (out/'rank-000/metrics.jsonl').write_text(json.dumps({'eval_runtime':20}))
            p={'data':{'monitor_subset_questions':16},'execution':{'accelerated_multidomain_benchmark':True}}
            r=projection(out,p,{'created_at':0,'gpu':'NVIDIA A100-SXM4-80GB','gpu_count':2,'budget_rate':4.625})
            self.assertEqual(r['measured_gpu_count'],2);self.assertEqual(r['steady_rollout_samples'],1)
            self.assertEqual(set(r['targets']),{'100','300'});self.assertNotIn('two4090_scenarios_hours',r)
            self.assertLess(r['targets']['100']['hours_range'][1],r['targets']['300']['hours_range'][0])

if __name__=='__main__':unittest.main()
