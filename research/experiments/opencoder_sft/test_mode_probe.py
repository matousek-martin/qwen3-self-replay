import unittest
from types import SimpleNamespace
from .mode_probe import pair_check,nonthinking_complete,metrics
from .mode_budget import SCOPE,PREFIX,ENTRY,GPU,allocation_spec,verify_worker
from experiments.shorterbetter.multidomain.rewards import verify

class ModeProbeTests(unittest.TestCase):
    def test_nonthinking_does_not_require_nonempty_reasoning(self):
        text='\\boxed{42}'
        self.assertTrue(nonthinking_complete(text));self.assertTrue(verify(text,'{"task":"math","value":"42"}'))
        for text in ['', '   ', '<think>unfinished', 'reasoning</think>answer']:
            self.assertFalse(nonthinking_complete(text))
    def test_pairs_reject_duplicates_and_cohort_mixups(self):
        q=[dict(id='q',task='math',cohort='original-development')]
        r=[dict(q[0],sample=s) for s in range(2)];pair_check(r,q)
        for bad in [r[:1],r+[r[0]],[r[0],dict(r[1],cohort='opencoder-development')]]:
            with self.assertRaises(ValueError):pair_check(bad,q)
    def test_opencoder_excluded_from_original_overall(self):
        records=[dict(id='original',sample=0,task='coding',cohort='original-development',correct=False,total_tokens=100,truncated=False),dict(id='oc',sample=0,task='coding',cohort='opencoder-development',correct=True,total_tokens=10,truncated=False)]
        m=metrics(records,records);self.assertEqual(m['overall']['accuracy'],0);self.assertEqual(m['opencoder']['accuracy'],1)
    def test_scope_cannot_launch_training_or_wrong_gpu(self):
        a=dict(hardware_scope=SCOPE,name=PREFIX+'test',entry=ENTRY,gpu=GPU,gpu_count=1,provider='vast',max_cost_usd=4)
        allocation_spec(SimpleNamespace(**a));verify_worker([GPU],dict(a,job_name=a['name']))
        for k,v in [('entry','experiments/opencoder_sft/run.sh'),('gpu_count',2),('max_cost_usd',5)]:
            with self.assertRaises(ValueError):allocation_spec(SimpleNamespace(**dict(a,**{k:v})))

if __name__=='__main__':unittest.main()
