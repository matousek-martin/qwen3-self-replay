"""Hash-locked static target pools with identical precomputed prompt schedules."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StaticReplay:
    def __init__(self, manifest_path, arm, seed, tokenizer, grader):
        p=Path(manifest_path); manifest=json.loads(p.read_text())
        if arm not in ('student','teacher'):
            raise ValueError('Static arm must be student or teacher')
        self.manifest=manifest
        pool_path=p.parent/manifest['pools'][arm]['file']
        schedule_path=p.parent/manifest['schedules'][str(seed)]['file']
        if sha(pool_path)!=manifest['pools'][arm]['sha256'] or sha(schedule_path)!=manifest['schedules'][str(seed)]['sha256']:
            raise ValueError('Static pool/schedule hash mismatch')
        pool=[json.loads(line) for line in pool_path.read_text().splitlines()]
        if [r['prompt_id'] for r in pool] != manifest['prompt_ids'] or len(set(manifest['prompt_ids']))!=len(pool):
            raise ValueError('Static pool prompt IDs differ from matched registration')
        self.schedule=json.loads(schedule_path.read_text())
        if len(self.schedule)!=128 or any(i not in manifest['prompt_ids'] for i in self.schedule):
            raise ValueError('Incomplete static replay schedule')
        self.targets={}
        for r in pool:
            if not r.get('replay_eligible') or r['review']['verdict']!='pass' or not r['review'].get('evidence'):
                raise ValueError('Unreviewed static replay target')
            if hashlib.sha256(r['completion'].encode()).hexdigest()!=r['completion_sha256']:
                raise ValueError('Completion hash mismatch')
            if not grader(r['completion'],r['answer'],r['task']):
                raise ValueError('Static target failed independent final-answer regrade')
            prompt=tokenizer.apply_chat_template(r['prompt'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
            completion=tokenizer.encode(r['completion'],add_special_tokens=False)+[tokenizer.eos_token_id]
            if len(completion)>2048 or prompt!=r['training_prompt_ids'] or completion!=r['training_completion_ids']:
                raise ValueError('Static target token IDs/limit changed')
            self.targets[r['prompt_id']]=(prompt,completion)

    def sample(self, group):
        if not 0<=group<len(self.schedule):
            raise ValueError('Static schedule exhausted')
        key=self.schedule[group]
        return key,self.targets[key]
