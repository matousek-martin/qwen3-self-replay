"""Post-hoc CPU diagnostic: first code block only, never choose a passing block."""
import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .runtime import read,rows,sha,write
from .mode_probe import pair_check,nonthinking_complete
from experiments.rl_training.objectives import reasoning_is_complete
from experiments.shorterbetter.multidomain.rewards import final_text,unfence,code_check
from experiments.replay_scaled.metrics import summarize

def first_code_block(text):
    answer=final_text(text)
    match=re.search(r'```(?:python|py)?[ \t]*\n(.*?)```',answer,re.S|re.I)
    return match.group(1).strip() if match else unfence(answer)

def main():
    p=argparse.ArgumentParser();p.add_argument('registration',type=Path);a=p.parse_args();launch=read(a.registration/'launch.json')
    data=Path(launch['release'])/'lab/experiments/data/opencoder-mode-probe'
    output=Path('lab/experiments/cloud')/launch['job_name']/'output/opencoder-mode-probe'
    questions=rows(data/'questions.jsonl');byid={r['id']:r for r in questions}
    sets={};paths={}
    for label in ['parent768-on','parent768-off','sft16-on','sft16-off']:
        paths[label]=(data if label.endswith('-on') else output)/(label+'.jsonl')
        sets[label]=rows(paths[label]);pair_check(sets[label],questions)
    def grade(item):
        condition,r=item
        acceptable=(reasoning_is_complete(r['text']) if condition.endswith('-on') else nonthinking_complete(r['text'])) and not r['truncated']
        code=first_code_block(r['text']);passed=bool(acceptable and code_check(code,json.loads(byid[r['id']]['answer'])['tests']))
        return dict(condition=condition,id=r['id'],sample=r['sample'],task='coding',cohort=r['cohort'],correct=passed,
            strict_correct=r['correct'],truncated=r['truncated'],total_tokens=r['total_tokens'],
            wrapper_removed=code!=unfence(final_text(r['text'])))
    candidates=[(label,r) for label,records in sets.items() for r in records if r['task']=='coding']
    with ThreadPoolExecutor(max_workers=8) as pool:graded=list(pool.map(grade,candidates))
    grouped={k:[r for r in graded if r['condition']==k] for k in sets}
    result=dict(post_hoc=True,official_evaluation=False,rule='First Python/py/unlabelled fenced code block from final answer; else unchanged raw code; same completion/truncation gate and sandbox tests. Never choose among blocks by test success.',
        sources_sha256={k:sha(p) for k,p in paths.items()},conditions={},comparisons={})
    for cohort in ['original-development','opencoder-development']:
        result['comparisons'][cohort]={}
        for mode in ['on','off']:
            arows=[r for r in grouped['sft16-'+mode] if r['cohort']==cohort];brows=[r for r in grouped['parent768-'+mode] if r['cohort']==cohort]
            result['comparisons'][cohort][mode]=summarize(arows,brows)['coding']
        for label,records in grouped.items():
            r=[x for x in records if x['cohort']==cohort]
            result['conditions'].setdefault(label,{})[cohort]=dict(answers=len(r),strict_correct=sum(x['strict_correct'] for x in r),code_correct=sum(x['correct'] for x in r),wrappers_removed=sum(x['wrapper_removed'] for x in r))
    target=a.registration/'format-audit';target.mkdir(exist_ok=True)
    (target/'records.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in graded));write(target/'results.json',result)
    print(json.dumps(result['conditions'],indent=2))

if __name__=='__main__':main()
