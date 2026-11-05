"""Pure contract checks and objectives; never select by answer length."""
import hashlib,random
from experiments.rl_training.objectives import advantages

def rewards(correct):
    r=[float(x) for x in correct]
    return r,advantages(r,scale=False)

def choose_correct(records,key):
    choices=[r for r in records if r['correct'] and not r.get('truncated',False)]
    if not choices:raise ValueError('No verified own answer')
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest(),16)).choice(choices)

def check_contract(cfg,train,dev):
    if cfg['arm'] not in ('a','b','c') or cfg['parent_group']!=768 or cfg['length_coefficient']!=0 or (cfg['steps']!=48 and cfg.get('variant')!='expanded-v2'):
        raise ValueError('ABC scientific contract differs')
    if len(train)!=cfg['steps'] or len({r['id'] for r in train})!=cfg['steps'] or {r['id'] for r in train}&{r['id'] for r in dev}:
        raise ValueError('Duplicate or overlapping data')
    if sum(r['task']=='coding' for r in train)!=cfg.get('coding_questions',36):raise ValueError('Coding count differs')
    if cfg.get('variant')=='expanded-v2':
        if cfg.get('generation_cap')!=16384 or sum(len(r['own_answers']) for r in train if r['task']=='coding')!=436:
            raise ValueError('Expanded436-answer/16384-token contract differs')
    for row in train:
        if row.get('split')!='train' or not row.get('own_answers') or any(not a.get('correct') for a in row['own_answers']):
            raise ValueError('Only verified self answers allowed')
