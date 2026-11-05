"""Exercise actual HF/Qwen decoding, independent sampling and saved-state replay."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import (Qwen3Config, Qwen3ForCausalLM, LogitsProcessorList,
                          TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)
from peft import LoraConfig, get_peft_model

from .generation import (RowSeededSampler, batches, generate_batch, settings,
                         trim_completion, validate_settings)
from .check import equal


def tiny_model():
    torch.manual_seed(123)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=256, eos_token_id=63, pad_token_id=0,
        attention_dropout=0., tie_word_embeddings=False, use_cache=False))
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.,
                                          target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))


def decode(model, prompts, seeds, **kwargs):
    return generate_batch(model, prompts, seeds, pad_token_id=0, eos_token_ids={63},
                          max_new_tokens=12, **kwargs)


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_bucketing_limits_and_exact_membership(self):
        lengths=[80,5,100,10,200,10,120]
        chunks=list(batches(lengths,3,450,20))
        self.assertEqual(sorted(i for batch in chunks for i in batch),list(range(len(lengths))))
        for batch in chunks:
            self.assertLessEqual(len(batch),3)
            self.assertLessEqual((max(lengths[i] for i in batch)+20)*len(batch),450)
        self.assertEqual(chunks,list(batches(lengths,3,450,20)))
        with self.assertRaises(ValueError):list(batches([500],16,450,20))

    def test_legacy_or_unsafe_config_rejected(self):
        self.assertEqual(validate_settings(settings()),settings())
        for bad in [None,{}, {**settings(),'evaluation_batch_size':65},
                    {**settings(),'max_padded_tokens':1000000}, {**settings(),'padding_side':'right'}]:
            with self.assertRaises(ValueError):validate_settings(bad)

    def test_large_validation_batches_obey_token_budget(self):
        lengths=[2048]*130
        cfg=settings()
        self.assertEqual([len(x) for x in batches(lengths,cfg['evaluation_batch_size'],cfg['max_padded_tokens'],2048)],[64,64,2])
        self.assertEqual([len(x) for x in batches(lengths,64,65536,2048)],[16]*8+[2])

    def test_sampler_matches_hf_filters_and_independent_generators(self):
        scores=torch.randn(3,64,generator=torch.Generator().manual_seed(12))
        ids=torch.ones((3,4),dtype=torch.long)
        seeds=[10,11,12];rngs=[torch.Generator().manual_seed(s) for s in seeds]
        sampler=RowSeededSampler(seeds,'cpu',.6,20,.95)
        filters=LogitsProcessorList([TemperatureLogitsWarper(.6),TopKLogitsWarper(20),TopPLogitsWarper(.95)])
        for _ in range(5):
            expected=torch.stack([torch.multinomial(p,1,generator=g) for p,g in
                                  zip(filters(ids,scores.clone()).softmax(-1),rngs)]).flatten()
            self.assertTrue(torch.equal(sampler(ids,scores.clone()).argmax(-1),expected))

    def test_first_eos_kept_padding_removed_cap_not_mislabeled(self):
        self.assertEqual(trim_completion([1,2,63,0,0],{63}),[1,2,63])
        self.assertEqual(trim_completion([1,62,63,0],{62,63}),[1,62])
        self.assertEqual(trim_completion([1,2,3],{63}),[1,2,3])
        # EOS in the last allowed position is a completed answer, not truncation.
        self.assertEqual(trim_completion([1,2,63],{63}),[1,2,63])

    def test_actual_qwen_left_padding_batch_serial_and_permutation(self):
        model=tiny_model();model.train()
        prompts=[[5,6,7],[8],[9,10,11,12,13],[14,15]];seeds=[17,18,19,20]
        state=torch.get_rng_state().clone()
        together=decode(model,prompts,seeds)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        separate=[decode(model,[p],[s])[0] for p,s in zip(prompts,seeds)]
        self.assertEqual(together,separate)
        permutation=[2,0,3,1]
        shuffled=decode(model,[prompts[i] for i in permutation],[seeds[i] for i in permutation])
        self.assertEqual(shuffled,[together[i] for i in permutation])
        model.eval();self.assertEqual(decode(model,prompts,seeds),together);self.assertFalse(model.training)

    def test_actual_qwen_eos_completion_stops_at_one(self):
        model=tiny_model()
        # Every token is EOS: exercise HF's real per-row stopping and extraction.
        result=generate_batch(model,[[1],[1,2,3]],[17,18],pad_token_id=0,
                              eos_token_ids=set(range(64)),max_new_tokens=12)
        self.assertEqual([len(x) for x in result],[1,1])

    def test_forward_call_count_scales_with_steps_not_answers(self):
        model=tiny_model();calls=[]
        # Model-level defaults must not disable caching or cause a second sampling pass.
        model.generation_config.use_cache=False
        model.generation_config.do_sample=True
        model.generation_config.temperature=.1
        def observe(module,args,kwargs):calls.append((kwargs['input_ids'].shape,kwargs.get('use_cache')))
        handle=model.base_model.model.register_forward_pre_hook(observe,with_kwargs=True)
        try:
            # Exclude EOS from the model vocabulary to force a fixed 12-step decode.
            opts=dict(pad_token_id=0,eos_token_ids={64},max_new_tokens=12)
            generate_batch(model,[[1,2],[3,4]],[17,18],**opts);batched=len(calls)
            self.assertTrue(all(use_cache for _,use_cache in calls))
            self.assertEqual([shape[-1] for shape,_ in calls],[2]+[1]*11)
            calls.clear()
            for prompt,seed in [([1,2],17),([3,4],18)]:generate_batch(model,[prompt],[seed],**opts)
            self.assertEqual(batched,12);self.assertEqual(len(calls),24)
        finally:handle.remove()

    def test_lora_optimizer_save_resume_and_next_batched_samples(self):
        model=tiny_model();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        def step(m,o,seed):
            answers=decode(m,[[3,4],[3,4]],[seed,seed+1])
            o.zero_grad();m.train()
            for answer in answers:
                ids=torch.tensor([[3,4]+answer]);loss=m(input_ids=ids,labels=ids,use_cache=False).loss/2
                loss.backward()
            o.step()
            return answers
        step(model,opt,17)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),path)
            uninterrupted=step(model,opt,21)
            expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
            restored=tiny_model();restored_opt=torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad],lr=.001)
            state=torch.load(path,weights_only=False)
            restored.load_state_dict(state['model']);restored_opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
            self.assertEqual(step(restored,restored_opt,21),uninterrupted)
            self.assertTrue(equal(restored.state_dict(),expected));self.assertTrue(equal(restored_opt.state_dict(),expected_opt))


if __name__=='__main__':unittest.main()
