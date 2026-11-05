"""Opt-in integration of the registered three-arm conditional KL pilot."""
import hashlib
import json
from pathlib import Path
import time

TEACHER = 'Qwen/Qwen3-4B'
TEACHER_REVISION = '1cfa9a7208912126459214e8b04321603b3df60c'
STUDENT_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'


def hashed(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def add_arguments(parser):
    parser.add_argument('--distill-arm', choices=['offline-hard', 'offline-soft', 'on-policy-soft'])
    parser.add_argument('--distill-cache')
    parser.add_argument('--distill-cache-sha256')


def validate_args(args):
    if bool(args.distill_arm) != bool(args.distill_cache):
        raise ValueError('Distillation arm and cache must be provided together')
    if args.distill_arm and (args.variant != 'replay' or args.replay_reward != 'fixed'
                             or args.replay_admission != 'all_correct'):
        raise ValueError('Distillation preserves fixed reward and historical replay admission')
    if args.distill_arm:
        actual = hashlib.sha256(Path(args.distill_cache).read_bytes()).hexdigest()
        if actual != args.distill_cache_sha256:
            raise ValueError('Distillation cache hash mismatch')


class Auxiliary:
    def __init__(self, args, model, tokenizer, rows, out, eos):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .loss import assert_tokenizer_alignment, freeze_teacher
        self.args, self.model, self.tokenizer, self.eos = args, model, tokenizer, eos
        self.cache = json.loads(Path(args.distill_cache).read_text())
        if self.cache['teacher_revision'] != TEACHER_REVISION or self.cache['student_revision'] != STUDENT_REVISION:
            raise ValueError('Cache model revision mismatch')
        if self.cache['data_sha256'] != hashlib.sha256(Path(args.data).read_bytes()).hexdigest():
            raise ValueError('Cache train data mismatch')
        if set(self.cache['entries']) != {r['id'] for r in rows}:
            raise ValueError('Cache prompt coverage mismatch')
        if not any(e['accepted'] for e in self.cache['entries'].values()):
            raise ValueError('No valid teacher cache; pilot cannot test teacher supervision')
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            teacher_tok = AutoTokenizer.from_pretrained(TEACHER, revision=TEACHER_REVISION)
            teacher_tok.pad_token = teacher_tok.eos_token
            self.teacher = freeze_teacher(AutoModelForCausalLM.from_pretrained(
                TEACHER, revision=TEACHER_REVISION, torch_dtype=torch.bfloat16,
                attn_implementation='sdpa').to(args.device))
        assert_tokenizer_alignment(tokenizer, teacher_tok, model.config.vocab_size, self.teacher.config.vocab_size)
        self.teacher.config.use_cache = False
        self.tokenizer_hash = hashed(tokenizer.backend_tokenizer.to_str())
        if self.tokenizer_hash != self.cache['tokenizer_sha256']:
            raise ValueError('Cache tokenizer fingerprint mismatch')
        for row in rows:
            e = self.cache['entries'][row['id']]
            expected = tokenizer.apply_chat_template(row['prompt'], tokenize=True, add_generation_prompt=True, enable_thinking=True)
            if e['prompt_ids'] != expected:
                raise ValueError('Cache prefix identity mismatch')
        self.trace_path = out / 'distillation.jsonl'
        self.smoke(out)

    def smoke(self, out):
        """Full-length GPU forward/backward, no optimizer step or RNG change."""
        import torch
        from .loss import completion_mask, qwen_forward_kl
        started = time.monotonic()
        self.model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        ids = torch.full((1, 4096), 100, dtype=torch.long, device=self.args.device)
        attention = torch.ones_like(ids)
        mask = completion_mask(ids, attention, [3072], self.eos)
        self.model.train()
        loss = qwen_forward_kl(self.model, self.teacher, ids, attention, mask, chunk_size=32)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite GPU smoke loss')
        loss.backward()
        torch.cuda.synchronize()
        if not any(p.grad is not None for p in self.model.parameters() if p.requires_grad):
            raise RuntimeError('Auxiliary failed to reach student parameters')
        if any(p.grad is not None or p.requires_grad for p in self.teacher.parameters()):
            raise RuntimeError('Teacher is not frozen')
        evidence = {'sequence_length':4096, 'target_tokens':1024, 'loss':float(loss.detach()),
                    'seconds':time.monotonic()-started, 'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                    'gpu':torch.cuda.get_device_name(), 'optimizer_steps':0,
                    'teacher_revision':TEACHER_REVISION, 'teacher_frozen':True}
        (out / 'distillation-gpu-smoke.json').write_text(json.dumps(evidence,indent=2)+'\n')
        self.model.zero_grad(set_to_none=True)
        del ids, attention, mask, loss
        torch.cuda.empty_cache()

    def backward(self, row, prompt_ids, group, policy_update, token_logprobs):
        import torch
        from .loss import completion_mask, qwen_forward_kl
        entry = self.cache['entries'][row['id']]
        if not entry['accepted']:
            return {'loss':0.0, 'tokens':0, 'active':False}
        on_policy = self.args.distill_arm == 'on-policy-soft'
        prefix = prompt_ids[0].tolist() if on_policy else entry['prompt_ids']
        completion = (group[0]['ids'] if on_policy else entry['completion_ids'])[:1024]
        if not completion:
            raise ValueError('Empty selected completion')
        if on_policy and group[0]['policy_update'] != policy_update:
            raise ValueError('Stale student rollout')
        ids = torch.tensor([prefix + completion], device=self.args.device)
        attention = torch.ones_like(ids)
        mask = completion_mask(ids, attention, [len(prefix)], self.eos)
        if self.args.distill_arm == 'offline-hard':
            lp, _ = token_logprobs(self.model, torch.tensor([prefix],device=self.args.device),
                                   torch.tensor([completion],device=self.args.device),32,1.0)
            loss = -0.1 * lp.mean()
        else:
            loss = 0.1 * qwen_forward_kl(self.model,self.teacher,ids,attention,mask,chunk_size=32)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite teacher auxiliary')
        loss.backward()
        tokens = int(mask.sum())
        evidence = {'prompt_id':row['id'],'arm':self.args.distill_arm,'policy_update':policy_update,
                    'scoring_step':policy_update,'prompt_ids':prefix,'completion_ids':completion,
                    'input_ids_sha256':hashed(ids.tolist()),'attention_sha256':hashed(attention.tolist()),
                    'target_mask_sha256':hashed(mask.tolist()),'prompt_length':len(prefix),
                    'student_revision':STUDENT_REVISION,'teacher_revision':TEACHER_REVISION,
                    'teacher_frozen':True,'tokenizer_sha256':self.tokenizer_hash,
                    'source': 'current_sample0' if on_policy else 'frozen_teacher_cache',
                    'scored_token_count':tokens,'weighted_loss':float(loss.detach())}
        with self.trace_path.open('a') as f:
            f.write(json.dumps(evidence)+'\n')
        return {'loss':float(loss.detach()),'tokens':tokens,'active':True}
