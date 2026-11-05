import copy
import json
from pathlib import Path
import tempfile
import unittest

from .hardware_benchmark import SETUPS, SCOPE, validate, measure, compare
from .config import training_arguments
from .budget import ticket
from .hardware import verify_worker
from .train import monitor_rows
from .throughput import read_rows
from ops.shorterbetter_hardware_benchmark import build_protocol, ROOT


class HardwareBenchmarkTests(unittest.TestCase):
    def test_vast_four_a100_is_separate_bounded_followup(self):
        p = self.protocol('h100x4')
        p['execution'].update(hardware_comparison='vast_a100x4', gpu='NVIDIA A100-SXM4-80GB')
        p['budget']['proposed_total_pilot_cap_usd'] = 15.
        self.assertEqual(validate(p)[:2], ('NVIDIA A100-SXM4-80GB', 4))
        self.assertEqual(training_arguments(p, 'shorterbetter', 17, '/tmp/test')['max_steps'], 16)
        p['training']['rollout_batches'] = 100
        with self.assertRaises(ValueError): validate(p)

    def protocol(self, setup='a100x2'):
        data = ROOT/'lab/experiments/shorterbetter-multidomain/data-v5-full'
        return build_protocol(setup, data, json.loads((data/'manifest.json').read_text()))

    def test_three_configurations_keep_same_global_work_and_full_dataset(self):
        protocols = [self.protocol(s) for s in SETUPS]
        for p in protocols:
            validate(p)
            args = training_arguments(p, 'shorterbetter', 17, '/tmp/test')
            self.assertEqual(args['generation_batch_size'], 512)
            self.assertEqual(args['gradient_accumulation_steps']*p['execution']['world_size'], 128)
            self.assertEqual(args['max_steps'], 16)
            self.assertTrue(p['data']['cover_all_training_rows'])
            self.assertEqual(p['data']['train_questions'], 41857)
            self.assertEqual(p['data'], protocols[0]['data'])
            self.assertEqual(p['reward'], protocols[0]['reward'])

    def test_rejects_changed_hardware_exposure_precision_and_scope(self):
        p = self.protocol()
        for section, key, value in [('execution','world_size',4), ('execution','gpu','NVIDIA H200'),
                ('training','rollout_batches',100), ('training','generations_per_prompt',4),
                ('training','parameter_dtype','bfloat16'), ('data','cover_all_training_rows',False),
                ('execution','terminal_evaluation_required',False), ('authorization','full_training_authorized',True)]:
            changed = copy.deepcopy(p); changed[section][key] = value
            with self.assertRaises(ValueError): validate(changed)

    def test_budget_is_named_bounded_and_has_no_completion_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'authorizations').mkdir()
            for setup, (gpu, count, cap, rate) in SETUPS.items():
                name = 'shorterbetter-scaled-hardware-'+setup+'-test'
                auth = dict(scope=SCOPE, setup=setup, authorized=True, job_name=name, gpu=gpu,
                            gpu_count=count, max_cost_usd=cap, max_runtime_hours=3,
                            max_gpu_rate=rate, pilot_rollouts=4, full_training_authorized=False)
                (root/'authorizations'/f'{name}.json').write_text(json.dumps(auth))
                job = dict(auth, hardware_scope='shorterbetter-scaled-v1', entry='experiments/shorterbetter/run-scaled.sh',
                           created_at=100, hard_deadline=10900)
                self.assertIsNotNone(ticket(root, name, job))
                verify_worker([gpu]*count, job)
                for changes in [{'gpu_count':8}, {'gpu':'NVIDIA H200'}, {'max_cost_usd':cap+1},
                                {'hard_deadline':10902}, {'completion_budget_scope':'finish_registered_1200_updates'}]:
                    self.assertIsNone(ticket(root, name, dict(job, **changes)))
                with self.assertRaises(RuntimeError): verify_worker([gpu], job)

    def test_monitor_panel_is_balanced_and_ids_cannot_be_silently_substituted(self):
        p = self.protocol()
        path = ROOT/p['data']['directory']/'monitor.jsonl'
        rows = monitor_rows(path, p)
        self.assertEqual(len(rows), 16)
        self.assertEqual({r['task'] for r in rows}, {'math','coding','tools','instructions'})
        p['data']['monitor_ids'][0] = 'absent'
        with self.assertRaises(ValueError): monitor_rows(path, p)

    def test_unicode_outputs_use_real_jsonl_record_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'rows.jsonl'
            row = {'completion':'first\u2028second\u2029third'}
            path.write_text(json.dumps(row, ensure_ascii=False)+'\n')
            self.assertEqual(read_rows(path), [row])

    def test_measure_excludes_warmup_and_evaluation_gaps_and_handles_old_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            starts = [(0,0),(4,1000),(8,2000),(12,3000)]
            durations = [900,100,110,120]
            (out/'rollout-starts.jsonl').write_text(''.join(json.dumps({'optimizer_step':s,'time':t})+'\n' for s,t in starts))
            (out/'steps.jsonl').write_text(''.join(json.dumps({'optimizer_step':s+4,'time':t+d})+'\n' for (s,t),d in zip(starts,durations)))
            for rank in range(2):
                folder = out/f'rank-{rank:03d}'; folder.mkdir()
                rows = [{'split':'train','optimizer_step':s,'prompt_id':str(i),'total_tokens':100}
                        for s,_ in starts for i in range(rank*32,(rank+1)*32) for _ in range(8)]
                (folder/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            lease = {'gpu':'NVIDIA A100-SXM4-80GB','gpu_count':2,'rate':3.58}
            r = measure(out, lease)
            self.assertEqual(r['status'], 'measured')
            self.assertEqual(r['median_rollout_seconds'], 110)
            self.assertAlmostEqual(r['iteration_cost_usd'], 110/3600*3.58)
            del lease['rate']
            self.assertIsNone(measure(out, lease)['iteration_cost_usd'])
            (out/'rank-001/rollouts.jsonl').write_text('')
            with self.assertRaises(ValueError): measure(out, lease)

    def test_compare_refuses_different_prompts_and_flags_close_results(self):
        def result(gpu, count, duration):
            return {'status':'measured','gpu':gpu,'gpu_count':count,'median_rollout_seconds':duration,
                    'min_rollout_seconds':duration-10, 'max_rollout_seconds':duration+10,
                    'iteration_cost_usd':duration*count, 'prompt_ids_by_step':{'0':['same']},
                    'total_generated_tokens':1000, 'comparison_identity':{'model':'fixture','code':'same'}}
        rows = [result('A100',2,100), result('H100',2,95), result('H100',4,90)]
        self.assertEqual(compare(rows)['status'], 'needs_more_measurement')
        rows[-1] = result('H100',4,50)
        for row in rows:
            row['min_rollout_seconds'] = row['median_rollout_seconds']-1
            row['max_rollout_seconds'] = row['median_rollout_seconds']+1
        self.assertEqual(compare(rows)['status'], 'initial_comparison')
        rows[-1]['comparison_identity']['code'] = 'different'
        with self.assertRaises(ValueError): compare(rows)
        rows[-1]['comparison_identity']['code'] = 'same'
        rows[-1]['prompt_ids_by_step'] = {'0':['different']}
        with self.assertRaises(ValueError): compare(rows)
        self.assertEqual(compare([])['status'], 'incomplete')

    def test_archive_grace_keeps_new_scope_within_deadline(self):
        from experiments import cloud
        from unittest.mock import patch
        job = {'hard_deadline':1000}
        probe = {'worker_status':'dead','exit_marker':{'status':'valid'}}
        with patch('experiments.shorterbetter.budget.ticket', return_value={'scope':SCOPE}), patch.object(cloud.time,'time',return_value=600):
            self.assertTrue(cloud.completion_archive_authorized('test',job,probe))
        with patch('experiments.shorterbetter.budget.ticket', return_value={'scope':SCOPE}), patch.object(cloud.time,'time',return_value=800):
            self.assertFalse(cloud.completion_archive_authorized('test',job,probe))


if __name__ == '__main__': unittest.main()
