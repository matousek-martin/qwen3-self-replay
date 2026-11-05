import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import round3_api as api
from round3_pipeline import clean_teacher
from round3_report import compare
from round3_format import normalize_completion, validate_tokens


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.state = self.root / 'round3'; self.state.mkdir()
        (self.state / 'authorization.json').write_text(json.dumps({'user_message':'pls udelej','shared_cap_usd':200}))
        self.cloud = self.root / 'cloud'; self.cloud.mkdir()
        (self.cloud / 'ledger.json').write_text(json.dumps({'ceiling_usd':200,'jobs':{}}))
        self.p1 = patch.object(api,'STATE',self.state); self.p1.start(); self.addCleanup(self.p1.stop)
        self.p2 = patch.object(api.cloud,'STATE',self.cloud); self.p2.start(); self.addCleanup(self.p2.stop)
        self.messages = [{'role':'user','content':'Independent test'}]

    def good(self, request):
        return {'model':request['model'],'provider':'DeepInfra','usage':{'cost':.0001},
                'choices':[{'finish_reason':'stop','message':{'content':'Verified content'}}]},200

    def test_success_is_accounted_and_not_repeated(self):
        calls=[]
        def transport(r): calls.append(r); return self.good(r)
        first=api.call('qwen3_32b',self.messages,512,'test',transport)
        second=api.call('qwen3_32b',self.messages,512,'test',transport)
        self.assertEqual(len(calls),1);self.assertEqual(first,second)
        self.assertEqual(api.extract(first,'qwen3_32b'),'Verified content')
        jobs=json.loads((self.cloud/'ledger.json').read_text())['jobs']
        self.assertEqual(list(jobs.values())[0]['charged_estimate_usd'],.0001)
        self.assertTrue(list(jobs.values())[0]['released_at'])

    def test_unknown_transport_never_retried(self):
        calls=[]
        def fail(r): calls.append(r); raise TimeoutError('uncertain')
        with self.assertRaises(TimeoutError):api.call('qwen3_32b',self.messages,512,'test',fail)
        with self.assertRaises(RuntimeError):api.call('qwen3_32b',self.messages,512,'test',fail)
        self.assertEqual(len(calls),1)
        job=list(json.loads((self.cloud/'ledger.json').read_text())['jobs'].values())[0]
        self.assertNotIn('released_at',job);self.assertGreater(job['reserved_usd'],0)

    def test_wrong_model_or_provider_rejected(self):
        r=api.call('qwen3_32b',self.messages,512,'test',self.good)
        for key,value in [('provider','Another provider'),('model','Another model')]:
            modified=copy.deepcopy(r);modified['response'][key]=value
            with self.assertRaises(ValueError):api.extract(modified,'qwen3_32b')

    def test_budget_blocks_before_transport(self):
        (self.cloud/'ledger.json').write_text(json.dumps({'ceiling_usd':0,'jobs':{}}))
        with self.assertRaises(RuntimeError):api.call('qwen3_32b',self.messages,512,'test',self.good)
        self.assertEqual(list((self.state/'api').glob('*.json')),[])

    def test_payload_no_fallback_no_key_and_no_mutation(self):
        original=copy.deepcopy(self.messages);p=api.payload('qwen38_27b',self.messages,2048)
        self.assertEqual(self.messages,original)
        self.assertEqual(p['provider']['only'],['deepinfra'])
        self.assertFalse(p['provider']['allow_fallbacks'])
        self.assertTrue(p['messages'][-1]['content'].endswith('/no_think'))


class DataTests(unittest.TestCase):
    def test_only_complete_worked_solution_normalized(self):
        self.assertEqual(clean_teacher('3+2=5.\nFinal answer: 5'),'<think>\n3+2=5.\n</think>\nFinal answer: 5')
        for text in (None,'Final answer: 5','<think>3+2=5</think>\nFinal answer: 5','No final answer'):
            self.assertIsNone(clean_teacher(text))

    def test_framing_is_idempotent_and_rejects_nested_or_empty_blocks(self):
        raw='3+2=5.\n</think>\nFinal answer: 5'
        fixed=normalize_completion(raw)
        self.assertEqual(fixed, '<think>\n'+raw)
        self.assertEqual(normalize_completion(fixed), fixed)
        for bad in ('answer', '<think></think>answer', '<think>x</think>',
                    'x<think>y</think>z', '<think><think>x</think>y', 'x</think>y</think>z'):
            with self.assertRaises(ValueError): normalize_completion(bad)

    def test_pinned_tokenizer_full_training_sequence(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B',
            revision='c1899de289a04d12100db370d81485cdf75e47ca',local_files_only=True)
        prompt=tok.apply_chat_template([{'role':'user','content':'Add 3 and 2.'}],
            tokenize=True,add_generation_prompt=True,enable_thinking=True)
        old=tok.encode('3+2=5.\n</think>\nFinal answer: 5',add_special_tokens=False)+[tok.eos_token_id]
        new=tok.encode(normalize_completion('3+2=5.\n</think>\nFinal answer: 5'),add_special_tokens=False)+[tok.eos_token_id]
        validate_tokens(tok,prompt,new)
        with self.assertRaises(ValueError): validate_tokens(tok,prompt,old)
        with self.assertRaises(ValueError): validate_tokens(tok,prompt+[tok.convert_tokens_to_ids('<think>')],new)
        with self.assertRaises(ValueError): validate_tokens(tok,prompt,new[:-1])

    def test_paired_cluster_metric_direction_and_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            base=[];candidate=[]
            for source in ('gsm8k','commonsense_qa','arc_easy'):
                for q in range(64):
                    for a in range(8):
                        r={'prompt_id':f'{source}-{q}','source':source,'attempt':a,'seed':q*8+a,
                           'correct':True,'total_tokens':100,'completion':'base'}
                        base.append(r);candidate.append(dict(r,total_tokens=70,completion='candidate'))
            p=Path(d)/'base.jsonl';c=Path(d)/'candidate.jsonl'
            p.write_text(''.join(json.dumps(r)+'\n' for r in base));c.write_text(''.join(json.dumps(r)+'\n' for r in candidate))
            result=compare(p,c)
            self.assertAlmostEqual(result['average']['token_reduction'],.3)
            self.assertEqual(result['average']['accuracy_delta'],0)
            c.write_text(''.join(json.dumps(r)+'\n' for r in candidate[:-1]))
            with self.assertRaises(ValueError):compare(p,c)


if __name__=='__main__':unittest.main()
