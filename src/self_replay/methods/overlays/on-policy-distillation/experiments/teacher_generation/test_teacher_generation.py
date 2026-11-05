"""CPU-only safety/verification tests. Never access real API, ledger, or credentials."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from experiments import cloud
from .budget import Budget
from .common import canonical, digest, durable_json, file_hash, read_key
from .runner import CONFIG, complete_record, execute, make_plan, usage_cost
from .verify import verify_response, verify_run


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()  # Unit fixture only; real-tokenizer smoke is separate.


def response(request, text='Put opened jelly in the refrigerator to slow spoilage.\nFinal answer: E', status='completed'):
    return {'id': 'resp_fixture', 'model': request['payload']['model'], 'status': status,
            'service_tier': 'default', 'usage': {'input_tokens': 100, 'output_tokens': 40,
                'input_tokens_details': {'cached_tokens': 20}, 'output_tokens_details': {'reasoning_tokens': 10}},
            'output': [{'type': 'reasoning', 'summary': []}, {'type': 'message', 'role': 'assistant',
                'status': 'completed', 'content': [{'type': 'output_text', 'text': text}]}]}


class TeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full_plan = make_plan(json.loads(CONFIG.read_text()))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = copy.deepcopy(self.full_plan)
        self.plan['requests'] = self.plan['requests'][:2]
        self.plan['reservation_usd'] = sum(r['reserved_usd'] for r in self.plan['requests'])
        self.evidence = self.root / 'user-approval.txt'
        self.evidence.write_text('TEST FIXTURE: approves this test plan only; no real spending.')
        self.approval = {'approved': True, 'scope': 'teacher-pilot-openai', 'plan_sha256': digest(self.plan),
            'max_usd': self.plan['reservation_usd'], 'expires_at': time.time() + 3600,
            'evidence_path': str(self.evidence), 'evidence_sha256': file_hash(self.evidence)}
        self.budget = Budget(self.root / 'cloud')
        durable_json(self.budget.path, {'ceiling_usd': 200, 'jobs': {}})
        self.calls = 0

    def api(self, key, path, payload, client_id):
        self.assertEqual(key, 'fixture-only')
        self.calls += 1
        request = next(r for r in self.plan['requests'] if r['request_key'] == client_id)
        saved = json.loads((self.root / 'run/requests' / (client_id + '.json')).read_text())
        self.assertEqual(saved['state'], 'dispatching')
        state = json.loads(self.budget.path.read_text())
        self.assertGreater(cloud.liability(state['jobs']), 0)
        return {'http_status': 200, 'request_id': 'req_fixture', 'body': canonical(response(request))}

    def run_plan(self, **kwargs):
        return execute(self.plan, self.root / 'run', self.approval, api_call=self.api,
                       key_loader=lambda: 'fixture-only', budget=self.budget, **kwargs)

    def test_plan_exact_matched_cohort_and_no_gold(self):
        requests = self.full_plan['requests']
        self.assertEqual(len(requests), 512)
        self.assertEqual(len({r['request_key'] for r in requests}), 512)
        cohorts = {}
        for r in requests:
            cohorts.setdefault((r['payload']['model'], r['variant']), set()).add(r['prompt_id'])
            self.assertNotIn('answer', r['payload'])
            self.assertNotIn('seed', r['payload'])
        self.assertEqual(len(cohorts), 4)
        self.assertTrue(all(len(ids) == 128 for ids in cohorts.values()))
        self.assertTrue(all(ids == next(iter(cohorts.values())) for ids in cohorts.values()))

    def test_denied_approval_has_no_api_calls_or_reservation(self):
        self.approval['approved'] = False
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual(self.calls, 0)
        self.assertEqual(json.loads(self.budget.path.read_text())['jobs'], {})

    def test_shared_gpu_reservation_blocks_api(self):
        durable_json(self.budget.path, {'ceiling_usd': 200, 'jobs': {'gpu': {
            'created_at': time.time(), 'budget_rate': 1.3, 'reserved_usd': 179.999}}})
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual(self.calls, 0)

    def test_overrun_is_counted_and_observation_is_idempotent(self):
        name = self.budget.authorize(self.plan, self.approval, self.root/'run')
        request = self.plan['requests'][0]
        cost = {'accounted_usd': request['reserved_usd'] + 1, 'nominal_usd': 0.5}
        self.budget.observe(name, request, cost)
        self.budget.observe(name, request, cost)
        state=json.loads(self.budget.path.read_text())
        self.assertAlmostEqual(cloud.liability(state['jobs']), self.plan['reservation_usd'] + 1)
        self.assertAlmostEqual(state['jobs'][name]['charged_estimate_usd'],cost['accounted_usd'])

    def test_api_reservation_counts_for_gpu_but_is_not_a_pod(self):
        name = self.budget.authorize(self.plan, self.approval, self.root / 'run')
        jobs = json.loads(self.budget.path.read_text())['jobs']
        self.assertAlmostEqual(cloud.liability(jobs), self.plan['reservation_usd'])
        self.assertEqual(cloud.potentially_live_pods(jobs), 0)
        with patch.object(cloud, 'api', side_effect=AssertionError('No provider access')):
            cloud.guard_job(name, jobs[name])

    def test_resume_sends_only_remaining_and_settles_conservatively(self):
        result = self.run_plan(max_requests=1)
        self.assertEqual(result['status'], 'paused')
        state = json.loads(self.budget.path.read_text())
        self.assertAlmostEqual(cloud.liability(state['jobs']), self.plan['reservation_usd'])
        result = self.run_plan()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.calls, 2)
        self.assertGreater(result['accounted_usd'], result['nominal_usd'])
        self.assertAlmostEqual(cloud.liability(json.loads(self.budget.path.read_text())['jobs']), result['accounted_usd'])

    def test_timeout_retains_reservation_and_never_resends(self):
        def timeout(*args):
            self.calls += 1
            raise TimeoutError('MUST NOT LEAK transport secrets')
        with self.assertRaisesRegex(RuntimeError, 'evidence') as caught:
            execute(self.plan, self.root/'run', self.approval, api_call=timeout,
                    key_loader=lambda: 'fixture-only', budget=self.budget)
        self.assertNotIn('LEAK', str(caught.exception))
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual(self.calls, 1)
        self.assertAlmostEqual(cloud.liability(json.loads(self.budget.path.read_text())['jobs']), self.plan['reservation_usd'])

    def test_saved_response_crash_recovers_without_resending(self):
        self.budget.authorize(self.plan, self.approval, self.root/'run')
        req = self.plan['requests'][0]
        durable_json(self.root/'run/requests'/ (req['request_key'] + '.json'), {
            'state': 'response_saved', 'request': req,
            'wire': {'http_status': 200, 'request_id': 'req_fixture', 'body': canonical(response(req))}})
        self.run_plan()
        self.assertEqual(self.calls, 1)

    def test_dispatch_crash_never_replays(self):
        self.budget.authorize(self.plan, self.approval, self.root/'run')
        req = self.plan['requests'][0]
        durable_json(self.root/'run/requests'/ (req['request_key'] + '.json'), {'state': 'dispatching', 'request': req})
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual(self.calls, 0)

    def test_duplicate_plan_directory_and_changed_plan_are_blocked(self):
        self.budget.authorize(self.plan, self.approval, self.root/'run')
        with self.assertRaises(ValueError):
            self.budget.authorize(self.plan, self.approval, self.root/'duplicate')
        plan = copy.deepcopy(self.plan); plan['extra'] = 'new plan, same requests'
        approval = dict(self.approval, plan_sha256=digest(plan))
        with self.assertRaises(ValueError):
            self.budget.authorize(plan, approval, self.root/'duplicate')

    def test_expiry_and_evidence_and_price_fail_closed(self):
        for change in ({'expires_at': 1}, {'evidence_sha256': 'wrong'}, {'plan_sha256': 'wrong'}, {'max_usd': float('nan')}):
            with self.assertRaises(ValueError):
                self.budget.authorize(self.plan, dict(self.approval, **change), self.root/'run')
        bad = json.loads(CONFIG.read_text()); bad['models'][0]['output_per_million'] = -1
        with self.assertRaises(ValueError):
            make_plan(bad)

    def test_usage_cache_reasoning_and_bounds(self):
        req = self.plan['requests'][0]; body = response(req)
        price = req['pricing']; cost = usage_cost(body, req)
        self.assertAlmostEqual(cost['nominal_usd'], (80*price['input_per_million'] + 20*price['cached_input_per_million'] + 40*price['output_per_million'])/1e6)
        body['usage']['input_tokens'] = -1
        with self.assertRaises(ValueError): usage_cost(body, req)
        body = response(req); body['usage']['output_tokens'] = 2049
        with self.assertRaises(ValueError):
            complete_record({'wire': {'http_status': 200, 'body': canonical(body)}}, req)

    def test_bad_response_model_tier_missing_usage_http(self):
        req = self.plan['requests'][0]
        for changes in ({'model': 'unrequested'}, {'service_tier': 'fast'}, {'usage': None}, {'status': 'in_progress'}):
            body = dict(response(req), **changes)
            with self.assertRaises(ValueError):
                complete_record({'wire': {'http_status': 200, 'body': canonical(body)}}, req)
        with self.assertRaises(ValueError):
            complete_record({'wire': {'http_status': 429, 'body': '{}'}}, req)

    def test_real_grader_rejects_wrong_truncated_malformed_and_no_work(self):
        req = self.plan['requests'][0]
        prompt = {'answer': 'E', 'task': 'multiple_choice'}
        for text, status in [('Reason.\nFinal answer: D','completed'), ('Reason.\nFinal answer: E','incomplete'),
                             ('Final answer: E','completed'), ('<think>Reason</think>\nFinal answer: E','completed'),
                             ('Reason.\nFinal answer: E\ntrailing','completed')]:
            result = verify_response(response(req,text,status), prompt, Tokenizer())
            self.assertFalse(result['accepted_answer'])
        result = verify_response(response(req), prompt, Tokenizer())
        self.assertTrue(result['answer_correct'])
        self.assertFalse(result['replay_eligible'])
        self.assertIn('</think>', result['completion'])

    def test_numeric_grading_and_refusal(self):
        req = self.plan['requests'][0]; prompt={'answer':'0.5','task':'math'}
        self.assertTrue(verify_response(response(req,'Divide one by two.\nFinal answer: 1/2'),prompt,Tokenizer())['accepted_answer'])
        body=response(req); body['output'][1]['content']=[{'type':'refusal','refusal':'No'}]
        self.assertEqual(verify_response(body,prompt,Tokenizer())['rejection'],'refusal_or_nontext')

    def test_dotenv_is_literal_and_errors_do_not_expose_secret(self):
        path=self.root/'env'; path.write_text('OTHER=hidden\nexport OPENAI_API_KEY="literal-$(do-not-run)" # comment\n')
        self.assertEqual(read_key(path),'literal-$(do-not-run)')
        path.write_text('OPENAI_API_KEY="SECRET\n')
        with self.assertRaises(ValueError) as caught: read_key(path)
        self.assertNotIn('SECRET',str(caught.exception))

    @unittest.skipUnless(importlib.util.find_spec('transformers'), 'Run in .venv-teacher for pinned-tokenizer smoke')
    def test_offline_end_to_end_verification_with_pinned_tokenizer(self):
        self.run_plan()
        result = verify_run(self.root/'run', self.root/'verify-pending')
        self.assertEqual(result['answer_verified'], 2)
        self.assertEqual(result['replay_eligible'], 0)
        records = [json.loads(p.read_text()) for p in (self.root/'run/requests').glob('*.json')]
        reviews = self.root/'reviews.jsonl'
        reviews.write_text(''.join(canonical({'request_key':r['request']['request_key'],
            'response_sha256':r['response_sha256'],'reviewer':'independent test fixture',
            'evidence':'Synthetic fixture: refrigerator prevents food spoilage.', 'verdict':'pass'})+'\n' for r in records))
        result = verify_run(self.root/'run', self.root/'verify-reviewed', reviews)
        self.assertEqual(result['replay_eligible'], 2)
        data=json.loads((self.root/'verify-reviewed/verified.json').read_text())
        self.assertGreater(data[0]['total_tokens'], 5)
        self.assertTrue((self.root/'run/source-manifest.json').is_file())
        self.assertFalse(any(p.name.startswith('.env') for p in (self.root/'run/source').rglob('*')))


if __name__ == '__main__':
    unittest.main()
