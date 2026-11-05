"""Synthetic tensor/backbone gradient checks, never pretrained model training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from .integration import Auxiliary, validate_args
from .loss import freeze_teacher


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(16, 4)
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.model=TinyBackbone();self.lm_head=nn.Linear(4,16,bias=False)
        self.config=SimpleNamespace(vocab_size=16)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        torch.manual_seed(8)
        self.aux=object.__new__(Auxiliary)
        self.aux.model=Tiny();self.aux.teacher=freeze_teacher(Tiny())
        self.aux.eos={15};self.aux.tokenizer_hash='synthetic'
        self.aux.trace_path=Path(self.tmp.name)/'trace.jsonl'
        self.aux.cache={'entries':{'p':{'accepted':True,'prompt_ids':[1,2],'completion_ids':[3,4,15]}}}

    def test_soft_arms_backprop_to_student_only_on_current_tokens(self):
        for arm in ['offline-soft','on-policy-soft']:
            with self.subTest(arm=arm):
                self.aux.args=SimpleNamespace(distill_arm=arm,device='cpu')
                teacher_before={n:p.detach().clone() for n,p in self.aux.teacher.named_parameters()}
                self.aux.model.zero_grad(set_to_none=True)
                result=self.aux.backward({'id':'p'},torch.tensor([[1,2]]),[{'ids':[5,6,15], 'policy_update':7}],7,None)
                self.assertTrue(result['active']);self.assertEqual(result['tokens'],3)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in self.aux.model.parameters()))
                self.assertTrue(all(p.grad is None and torch.equal(p,teacher_before[n]) for n,p in self.aux.teacher.named_parameters()))
                record=json.loads(self.aux.trace_path.read_text().splitlines()[-1])
                self.assertEqual(record['completion_ids'],[5,6,15] if arm=='on-policy-soft' else [3,4,15])
                self.assertEqual(record['policy_update'],record['scoring_step'])

    def test_hard_loss_and_missing_cache(self):
        self.aux.args=SimpleNamespace(distill_arm='offline-hard',device='cpu')
        def logps(model,prefix,completion,*args):
            ids=torch.cat([prefix,completion],1)
            logits=model.lm_head(model.model(input_ids=ids).last_hidden_state)[:,prefix.shape[1]-1:-1]
            lp=logits.log_softmax(-1).gather(-1,completion.unsqueeze(-1)).squeeze()
            return lp,None
        r=self.aux.backward({'id':'p'},torch.tensor([[1,2]]),[],0,logps)
        self.assertTrue(r['active']);self.assertGreater(r['loss'],0)
        self.aux.cache['entries']['p']['accepted']=False
        for arm in ['offline-hard','offline-soft','on-policy-soft']:
            self.aux.args.distill_arm=arm
            self.assertFalse(self.aux.backward({'id':'p'},None,[],0,None)['active'])

    def test_stale_policy_rejected(self):
        self.aux.args=SimpleNamespace(distill_arm='on-policy-soft',device='cpu')
        with self.assertRaisesRegex(ValueError,'Stale'):
            self.aux.backward({'id':'p'},torch.tensor([[1,2]]),[{'ids':[3], 'policy_update':4}],5,None)


if __name__=='__main__': unittest.main()
