import copy
import json
import unittest
import tempfile
from pathlib import Path
from .prepare_xlam import api_family, family_registry, family_split, type_tree, type_matches, xlam_row
from .rewards import tool_check


def example():
    return {'id':1,'query':'Compute both requested operations.',
        'tools':[{'name':'arithmetic.add','description':'Add integers.','parameters':{'a':{'type':'int'},'b':{'type':'int','default':0}}},
                 {'name':'arithmetic.scale','description':'Scale numbers.','parameters':{'values':{'type':'List[Union[int, float]]'}}}],
        'answers':[{'name':'arithmetic.add','arguments':{'a':2}},
                   {'name':'arithmetic.scale','arguments':{'values':[1,2.5]}}]}


class XlamTests(unittest.TestCase):
    def test_parallel_calls_preserve_source_and_exact_multiset(self):
        raw=example();row=xlam_row(raw)
        self.assertEqual(row['tools'],raw['tools']);self.assertEqual(row['tool_case'],'multiple_calls')
        calls=json.loads(row['answer'])['calls']
        self.assertTrue(tool_check(json.dumps(list(reversed(calls))),calls))
        self.assertFalse(tool_check(json.dumps(calls[:1]),calls))
        self.assertFalse(tool_check(json.dumps(calls+[calls[0]]),calls))

    def test_invalid_call_names_arguments_and_types_are_rejected(self):
        for change in ('name','extra','missing','type'):
            r=example();a=r['answers'][0]
            if change=='name':a['name']='unknown'
            elif change=='extra':a['arguments']['not_in_schema']=1
            elif change=='missing':a['arguments']={}
            else:a['arguments']['a']=True
            with self.assertRaises(ValueError):xlam_row(r)

    def test_nested_types_and_unsupported_callable(self):
        self.assertTrue(type_matches([[1,2],[3,4]],type_tree('List[List[int]]')))
        self.assertFalse(type_matches([[1,True]],type_tree('List[List[int]]')))
        self.assertTrue(type_matches([1,2.5],type_tree('Tuple[float, float]')))
        self.assertFalse(type_matches([1,2,3],type_tree('Tuple[float, float]')))
        with self.assertRaises(ValueError):type_tree('Callable[[float], float]')
        with self.assertRaises(ValueError):type_tree('__import__("os").system("echo unsafe")')

    def test_namespace_and_historical_binding_prevent_api_leakage(self):
        a=api_family('calculator.add');b=api_family('calculator.subtract');self.assertEqual(a,b)
        registry=family_registry({'train':[{'tools':[{'name':'calculator.add'}]}], 'development':[], 'confirmation':[]})
        self.assertEqual(family_split([b],registry),'train')
        with self.assertRaisesRegex(ValueError,'historical_api_split_conflict'):
            family_split([a],{a:{'train','development'}})

    def test_cross_split_bundle_cannot_drop_tools_to_fit(self):
        with self.assertRaisesRegex(ValueError,'cross_partition_api_bundle'):
            family_split(['a','b'],{'a':{'train'},'b':{'confirmation'}})
        self.assertEqual(family_split(['a'],{}),family_split(['a'],{}))

    def test_source_strings_and_optional_defaults(self):
        r=example();r['tools']=json.dumps(r['tools']);r['answers']=json.dumps(r['answers'])
        self.assertEqual(xlam_row(r)['tool_case'],'multiple_calls')
        r=example();r['tools'][0]['parameters']['b']={'type':'int, optional'}
        self.assertEqual(xlam_row(r)['source'],'xlam')
        r['tools'][0]['parameters']['b']['required']=True
        with self.assertRaises(ValueError):xlam_row(r)

    def test_worker_enforces_new_dataset_verifier_hashes(self):
        from experiments.shorterbetter.train import check_data
        from experiments.shorterbetter.prepare import sha
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for name in ('train.jsonl','monitor.jsonl'):(out/name).write_text('{}\n')
            m={'id':'shorterbetter-four-domains-xlam-v1','files':{name:sha(out/name) for name in ('train.jsonl','monitor.jsonl')},
               'verifier_sha256':{name:sha(Path(__file__).with_name(name)) for name in ('rewards.py','code_worker.py')}}
            def save():
                (out/'manifest.json').write_text(json.dumps(m))
                (out/'audit.json').write_text(json.dumps({'status':'passed','manifest_sha256':sha(out/'manifest.json')}))
            save();self.assertEqual(check_data(out)['id'],m['id'])
            m['verifier_sha256']['rewards.py']='0'*64;save()
            with self.assertRaisesRegex(ValueError,'Verifier changed'):check_data(out)


if __name__=='__main__':unittest.main()
