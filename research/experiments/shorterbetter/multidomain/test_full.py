import copy
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest

from .prepare_full import make_row, assign_split, split_family, validate_reference
from .sampler import BalancedRepeatSampler


def apps():
    return {'id':42,'question':'Print the input integer plus one.','difficulty':'competition',
            'input_output':json.dumps({'inputs':['2','9'],'outputs':['3','10']}),
            'solutions':json.dumps(['print(3)','print(int(input())+1)']),
            'url':'https://example.invalid/problem/42','starter_code':''}


class FullDataTests(unittest.TestCase):
    def test_all_apps_difficulties_and_all_tests_preserved(self):
        for difficulty in ('introductory','interview','competition'):
            r=apps();r['difficulty']=difficulty
            tests={'inputs':[str(i) for i in range(101)],'outputs':[str(i+1) for i in range(101)]}
            r['input_output']=json.dumps(tests)
            row=make_row('apps',0,r)
            self.assertEqual(json.loads(row['answer'])['tests'],tests)
            self.assertEqual(row['difficulty'],difficulty)

    def test_solution_search_runs_tests_and_records_passing_source_index(self):
        r=make_row('apps',0,apps());result=validate_reference(r)
        self.assertTrue(result['passed']);self.assertEqual(result['solution_index'],1)
        self.assertEqual(result['all_tests'],2)
        r['reference']=['print(3)'];self.assertFalse(validate_reference(r)['passed'])

    def test_unaligned_tests_and_missing_reference_rejected(self):
        r=apps();r['input_output']=json.dumps({'inputs':['1','2'],'outputs':['3']})
        with self.assertRaisesRegex(ValueError,'unaligned'):make_row('apps',0,r)
        r=apps();r['solutions']='[]'
        with self.assertRaisesRegex(ValueError,'missing'):make_row('apps',0,r)

    def test_instruction_checks_cannot_drop_unsupported_constraints(self):
        raw={'id':'a','prompt':'Write lowercase words.','constraints':['case:in english and lowercase'],
             'messages':[{'content':'hello world'}]}
        self.assertTrue(validate_reference(make_row('tulu',0,raw))['passed'])
        raw['constraints'].append('include keywords')
        with self.assertRaisesRegex(ValueError,'unsupported'):make_row('tulu',0,raw)

    def test_numeric_templates_respect_historical_holdout(self):
        a=make_row('gsm8k',0,{'question':'There are 12 marbles. How many?','answer':'#### 12'})
        b=make_row('gsm8k',1,{'question':'There are 25 marbles. How many?','answer':'#### 25'})
        self.assertEqual(split_family(a),split_family(b))
        self.assertEqual(assign_split(b,{split_family(a):{'confirmation'}}),'confirmation')
        with self.assertRaisesRegex(ValueError,'conflict'):assign_split(b,{split_family(a):{'train','development'}})

    def test_full_sampler_covers_imbalanced_domains_unique_batches_and_repetition(self):
        tasks=sum(([d]*n for d,n in zip(('math','coding','tools','instructions'),(5,7,23,9))),[])
        data={'task':tasks};args=dict(data_source=data,mini_repeat_count=3,batch_size=8,repeat_count=2,seed=17,cover_all=True)
        a=list(BalancedRepeatSampler(**args));self.assertEqual(a,list(BalancedRepeatSampler(**args)))
        self.assertEqual(set(a),set(range(len(tasks))))
        self.assertEqual(len(a),len(BalancedRepeatSampler(**args)))
        for start in range(0,len(a),48):
            first=a[start:start+24];self.assertEqual(first,a[start+24:start+48])
            ids=first[::3];self.assertEqual(len(set(ids)),8)
            self.assertEqual(Counter(tasks[i] for i in ids),{d:2 for d in set(tasks)})
        legacy=list(BalancedRepeatSampler(data,3,8,2,17))
        self.assertLess(len(legacy),len(a))

    def test_full_sampler_rejects_insufficient_unique_domain(self):
        data={'task':['math','coding','tools','instructions']}
        with self.assertRaisesRegex(ValueError,'enough unique'):BalancedRepeatSampler(data,1,8,1,17,cover_all=True)

    def test_new_dataset_worker_hash_enforcement(self):
        from experiments.shorterbetter.train import check_data
        from experiments.shorterbetter.prepare import sha
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for f in ('train.jsonl','monitor.jsonl'):(p/f).write_text('{}\n')
            m={'id':'shorterbetter-four-domains-full-v1','files':{f:sha(p/f) for f in ('train.jsonl','monitor.jsonl')},
               'verifier_sha256':{f:sha(Path(__file__).with_name(f)) for f in ('rewards.py','code_worker.py')}}
            def save():
                (p/'manifest.json').write_text(json.dumps(m));(p/'audit.json').write_text(json.dumps({'status':'passed','manifest_sha256':sha(p/'manifest.json')}))
            save();check_data(p)
            m['verifier_sha256']['code_worker.py']='0'*64;save()
            with self.assertRaisesRegex(ValueError,'Verifier changed'):check_data(p)

    def test_full_data_cannot_silently_use_legacy_shortened_epoch(self):
        from experiments.shorterbetter.train import check_dataset_protocol
        m={'id':'shorterbetter-four-domains-full-v1','counts':{'train.jsonl':12345}}
        p={'data':{'train_questions':12345}}
        with self.assertRaisesRegex(ValueError,'cover_all'):check_dataset_protocol(m,p)
        p['data']['cover_all_training_rows']=True;check_dataset_protocol(m,p)
        p['data']['train_questions']=4000
        with self.assertRaisesRegex(ValueError,'count differs'):check_dataset_protocol(m,p)


class JsonLinesTests(unittest.TestCase):
    def test_unicode_line_separators_inside_strings_are_not_records(self):
        from experiments.shorterbetter.jsonl import read_rows
        expected=[{'problem':'a\u2028b\u2029c\x85d'}, {'problem':'ordinary'}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl'
            path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in expected)+'\n')
            self.assertEqual(read_rows(path),expected)
