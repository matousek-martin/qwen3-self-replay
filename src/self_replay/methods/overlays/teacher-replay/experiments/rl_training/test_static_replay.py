import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from .static_replay import StaticReplay, sha


class Tokenizer:
    eos_token_id=9
    def apply_chat_template(self,*a,**k):return [1,2]
    def encode(self,*a,**k):return [3,4]


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)
        r={'prompt_id':'p','prompt':[],'completion':'worked\n</think>\nFinal answer: 2','answer':'2','task':'math',
           'replay_eligible':True,'review':{'verdict':'pass','evidence':'checked arithmetic'},
           'training_prompt_ids':[1,2],'training_completion_ids':[3,4,9]}
        r['completion_sha256']=hashlib.sha256(r['completion'].encode()).hexdigest()
        (self.p/'pool.jsonl').write_text(json.dumps(r)+'\n');(self.p/'schedule.json').write_text(json.dumps(['p']*128))
        self.m={'prompt_ids':['p'],'pools':{a:{'file':'pool.jsonl','sha256':sha(self.p/'pool.jsonl')} for a in ['student','teacher']},
                'schedules':{'17':{'file':'schedule.json','sha256':sha(self.p/'schedule.json')}}}
        self.save()
    def save(self):(self.p/'manifest.json').write_text(json.dumps(self.m))
    def test_same_schedule_and_no_mutation(self):
        pools=[StaticReplay(self.p/'manifest.json',a,17,Tokenizer(),lambda *a:True) for a in ['student','teacher']]
        for group in range(128):self.assertEqual(pools[0].sample(group),pools[1].sample(group))
        with self.assertRaises(ValueError):pools[0].sample(128)
    def test_tamper_rejected(self):
        (self.p/'pool.jsonl').write_text('{}')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):StaticReplay(self.p/'manifest.json','teacher',17,Tokenizer(),lambda *a:True)
    def test_wrong_target_rejected(self):
        with self.assertRaisesRegex(ValueError,'regrade'):StaticReplay(self.p/'manifest.json','teacher',17,Tokenizer(),lambda *a:False)


if __name__=='__main__':unittest.main()
