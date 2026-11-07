"""Single4090 matched pilot; full native coding context, correctness-only reward."""
import argparse,copy,hashlib,json,random,signal,time
from pathlib import Path
from experiments.opencoder_sft.runtime import read,write,sha,rows
from experiments.replay_scaled.core import MODEL,REVISION
from experiments.replay_scaled.generation import generate_batch,batches
from experiments.replay_scaled.metrics import summarize
from experiments.coding_abc.budget import verify_worker,SCOPE
from experiments.coding_abc.logic import rewards,choose_correct,check_contract
DATA=Path('data/coding-abc');OUT=Path('output/training')

def assess(q,text,truncated):
    from experiments.rl_training.objectives import reasoning_is_complete
    from experiments.shorterbetter.multidomain.rewards import verify,code_check
    from experiments.opencoder_sft.format_audit import first_code_block
    if truncated or not reasoning_is_complete(text):return False
    if q.get('spec'):
        from experiments.coding_frontier.grading import grade
        return grade(text,'stop',q['spec'])['correct']
    if q['task']=='coding':return bool(code_check(first_code_block(text),json.loads(q['answer'])['tests']))
    return bool(verify(text,q['answer']))

def main():
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from experiments.rl_training.train import token_logprobs
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path);parser.add_argument('--technical',action='store_true');args=parser.parse_args()
    cfg=read(DATA/'plan.json');train=rows(DATA/'train.jsonl');dev=rows(DATA/'development.jsonl')
    check_contract(cfg,train,dev)
    if args.technical != (cfg.get('technical_only') is True):raise ValueError('Technical mode and data plan differ')
    for file,h in cfg['files_sha256'].items():
        if sha(DATA/file)!=h:raise RuntimeError('Input hash changed: '+file)
    lease=read('worker-lease.json');verify_worker([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],lease)
    if cfg['scope']!=SCOPE:raise RuntimeError('Wrong scope')
    if args.technical and lease.get('hardware_scope')!='submission-method-smoke-v1':raise RuntimeError('Technical run requires its own smoke lease')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'progress.json').exists() and not args.resume:raise RuntimeError('Existing output requires explicit complete-state resume')
    torch.manual_seed(17);np.random.seed(17);random.seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    tok=AutoTokenizer.from_pretrained(DATA/'adapter',local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    if args.resume:
        manifest=read(args.resume/'complete.json')
        if manifest['contract_sha256']!=sha(DATA/'plan.json') or manifest['arm']!=cfg['arm']:raise RuntimeError('Resume contract mismatch')
        for f,h in manifest['files'].items():
            if sha(args.resume/f)!=h:raise RuntimeError('Corrupt resume file')
    model=PeftModel.from_pretrained(base,(args.resume/'adapter') if args.resume else DATA/'adapter',is_trainable=True)
    model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
    params=[p for p in model.parameters() if p.requires_grad]
    if any(p.dtype!=torch.float32 for p in params):raise RuntimeError('ExpectedFP32LoRA')
    optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0)
    eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
    stop=[False];done=0;updates=0;started=time.time();curve=[];buffer={};cache={}
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def live(phase,**extra):
        write(OUT/'progress.json',dict(phase=phase,current=done,total=cfg['steps'],optimizer_updates=updates,arm=cfg['arm'],
            elapsed=time.time()-started,updated_at=time.time(),gpu_count=1,**extra))
    def permitted():
        l=read('worker-lease.json')
        return not (stop[0] or l.get('stop_requested') or Path('output/stop-request.json').exists() or time.time()>min(l['hard_deadline']-1200,cfg.get('training_deadline',l['hard_deadline']-1200)))
    def prompt(q):
        if q['id'] not in cache:cache[q['id']]=tok.apply_chat_template(q['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        if not 0<len(cache[q['id']])<40960:raise RuntimeError('Prompt exceeds native context')
        return cache[q['id']]
    def answer(q,a):
        ids=a.get('generated_token_ids') or tok.encode(a['text'],add_special_tokens=False)
        if ids[-1] not in eos:ids=ids+[tok.eos_token_id]
        if len(prompt(q))+len(ids)>40960:raise RuntimeError('Verified answer exceeds native context; no silent clipping')
        return ids
    def loss(q,ids):
        lp,_=token_logprobs(model,torch.tensor([prompt(q)],device='cuda'),torch.tensor([ids],device='cuda'))
        if not torch.isfinite(lp).all():raise RuntimeError('Nonfinite log probabilities')
        return lp
    def step():
        norm=torch.nn.utils.clip_grad_norm_(params,1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        return float(norm)
    def checkpoint():
        dest=OUT/'checkpoints'/f'group-{done:06d}'
        if (dest/'complete.json').exists():return
        dest.mkdir(parents=True,exist_ok=True);model.save_pretrained(dest/'adapter');tok.save_pretrained(dest/'adapter')
        torch.save(dict(done=done,updates=updates,optimizer=optimizer.state_dict(),buffer=buffer,
            python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),curve=curve),dest/'state.pt')
        write(dest/'complete.json',dict(group=done,arm=cfg['arm'],contract_sha256=sha(DATA/'plan.json'),
            files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))
        write(OUT/'latest-checkpoint.json',dict(group=done,path=str(dest),manifest_sha256=sha(dest/'complete.json')))
    def record(q,s,ids):
        truncated=not ids or ids[-1] not in eos;text=tok.decode(ids,skip_special_tokens=True)
        return dict(id=q['id'],task=q['task'],sample=s,text=text,ids=ids,total_tokens=len(ids),truncated=truncated,correct=assess(q,text,truncated))
    def generate(requests,cap):
        seqs=generate_batch(model,[prompt(q) for q,s,seed in requests],[seed for q,s,seed in requests],
            pad_token_id=tok.pad_token_id,eos_token_ids=eos,max_new_tokens=cap,temperature=.6,top_k=20,top_p=.95)
        return [record(q,s,seq) for (q,s,seed),seq in zip(requests,seqs)]
    byid={q['id']:q for q in dev}
    baseline=rows(DATA/'baseline.jsonl')
    for r in baseline:r['correct']=assess(byid[r['id']],r['text'],r['truncated'])
    def publish(records):
        metrics=summarize(records,baseline)
        point=dict(iteration=done,completed_at=time.time(),metrics=metrics,official_evaluation=False,
            protocol='ABC-development64x2-HF-2048-format-aware-v1',baseline_cached=True,baseline_provenance=cfg['baseline_provenance'])
        curve.append(point);write(OUT/'validation-curve.json',dict(checkpoints=curve))
        dest=OUT/'validations'/f'{done:06d}.jsonl';dest.parent.mkdir(exist_ok=True)
        dest.write_text(''.join(json.dumps({k:v for k,v in r.items() if k!='ids'})+'\n' for r in records))
        write(dest.with_suffix('.json'),point)
    def validate():
        live('validation',validation_completed=0,validation_total=len(dev)*2);records=[]
        requests=[(q,s,int(hashlib.sha256(f"validation:{q['id']}:{s}:9127".encode()).hexdigest()[:8],16)) for q in dev for s in range(2)]
        for indices in batches([len(prompt(q)) for q,s,seed in requests],16,65536,2048):
            if not permitted():return False
            records+=generate([requests[i] for i in indices],2048)
            live('validation',validation_completed=len(records),validation_total=len(dev)*2)
        publish(records);return True
    try:
        live('technical_check')
        # Validate local verifier wiring, then full native-context backward memory.
        from experiments.coding_frontier.grading import check
        spec=dict(kind='io',inputs=[''],outputs=['2'])
        if check('print(2)',spec)['status']!='passed' or check('print(3)',spec)['status']=='passed':raise RuntimeError('Sandbox preflight failed')
        if not args.resume:
            model.train();saved=[p.detach().cpu().clone() for p in params]
            testq=train[0];testids=[tok.encode('x',add_special_tokens=False)[0]]*(40960-len(prompt(testq))-1)+[tok.eos_token_id]
            (-loss(testq,testids).mean()).backward();step()
            changed=any(not torch.equal(p.detach().cpu(),v) for p,v in zip(params,saved))
            if not changed:raise RuntimeError('Preflight did not update adapter')
            # Disk roundtrip must reproduce an additional optimizer update exactly.
            proof=OUT/'technical-resume.pt'
            torch.save(dict(parameters=[p.detach().cpu() for p in params],optimizer=optimizer.state_dict()),proof)
            (-loss(testq,testids[:127]+[tok.eos_token_id]).mean()).backward();step()
            expected=[p.detach().cpu().clone() for p in params]
            restored=torch.load(proof,map_location='cpu',weights_only=False)
            with torch.no_grad():
                for p,v in zip(params,restored['parameters']):p.copy_(v)
            optimizer.load_state_dict(restored['optimizer'])
            (-loss(testq,testids[:127]+[tok.eos_token_id]).mean()).backward();step()
            if any(not torch.equal(p.detach().cpu(),v) for p,v in zip(params,expected)):raise RuntimeError('Optimizer resume proof differs')
            with torch.no_grad():
                for p,v in zip(params,saved):p.copy_(v)
            optimizer=torch.optim.AdamW(params,lr=1e-5,weight_decay=0);del saved,testids,expected,restored
            proof.unlink();torch.cuda.empty_cache()
            write(OUT/'preflight.json',dict(passed=True,native_context_backward=40960,adapter_restored=True,optimizer_resume_bitwise=True,
                parent_adapter_sha256=sha(DATA/'adapter/adapter_model.safetensors'),gpu=torch.cuda.get_device_name(),arm=cfg['arm']))
            publish(baseline)
        if cfg['arm']=='c' and not args.resume:
            for q in train:
                a=choose_correct(q['own_answers'],'initial:'+q['id']);buffer[q['id']]=dict(q=q,ids=answer(q,a))
        if args.resume:
            state=torch.load(args.resume/'state.pt',map_location='cpu',weights_only=False)
            done=state['done'];updates=state['updates'];buffer=state['buffer'];curve=state['curve'];optimizer.load_state_dict(state['optimizer'])
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
            if done and done%16==0 and (not curve or curve[-1]['iteration']!=done):validate()
        checkpoint()
        for index in range(done,len(train)):
            q=train[index]
            if not permitted():break
            live('training',question=q['id']);began=time.time();model.train();optimizer.zero_grad(set_to_none=True)
            group=[];objective=0.;active=False
            if cfg['arm'] in ('a','c'):
                cap=min(cfg.get('generation_cap',40960),40960-len(prompt(q))) if q['task']=='coding' else 2048
                group=generate([(q,s,int(hashlib.sha256(f"abc:{q['id']}:{s}:17".encode()).hexdigest()[:8],16)) for s in range(4)],cap)
                _,adv=rewards([r['correct'] for r in group])
                for r,a in zip(group,adv):
                    if a:
                        l=-a*loss(q,r['ids']).sum()/(4*2048);l.backward();objective+=float(l.detach());active=True
                if cfg['arm']=='c' and any(r['correct'] for r in group):
                    chosen=choose_correct(group,'online:'+q['id']);buffer[q['id']]=dict(q=q,ids=chosen['ids'])
            if cfg['arm']=='b':
                selected=q['own_answers'] if cfg.get('all_sft_answers') else [choose_correct(q['own_answers'],'sft:'+q['id'])]
                for chosen in selected:
                    ids=answer(q,chosen);l=-loss(q,ids).mean()/len(selected);l.backward();objective+=float(l.detach())
                active=True
            elif cfg['arm']=='c':
                selected=random.Random(17+index).choice(list(buffer.values()))
                l=-.1*loss(selected['q'],selected['ids']).mean();l.backward();objective+=float(l.detach());active=True
            norm=step() if active else 0.;updates+=int(active);done=index+1
            with (OUT/'steps.jsonl').open('a') as f:f.write(json.dumps(dict(step=done,id=q['id'],task=q['task'],loss=objective,
                gradient_norm=norm,updated=active,seconds=time.time()-began,correct=sum(r['correct'] for r in group),answers=len(group)))+'\n')
            with (OUT/'training-records.jsonl').open('a') as f:
                for r in group:f.write(json.dumps(dict(r,step=done))+'\n')
            if done%8==0:checkpoint()
            if (done%16==0 or done==cfg['steps']) and not validate():break
        checkpoint();complete=done==cfg['steps'] and curve[-1]['iteration']==cfg['steps']
        write(OUT/'completion.json',dict(complete=complete,steps=done,optimizer_updates=updates,reason='finished' if complete else 'lease_or_user_stop'))
        live('complete' if complete else 'stopped')
    except Exception as e:
        checkpoint();write(OUT/'failure.json',dict(error=str(e),step=done,detected_at=time.time()));raise

if __name__=='__main__':main()
