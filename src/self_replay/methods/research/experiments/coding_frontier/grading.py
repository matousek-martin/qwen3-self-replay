import ast,json,os,signal,subprocess,sys,tempfile
from pathlib import Path
from experiments.opencoder_sft.format_audit import first_code_block
from experiments.rl_training.objectives import reasoning_is_complete

ROOT=Path(__file__).resolve().parents[2]

def check(code,spec):
    try:ast.parse(code)
    except (SyntaxError,ValueError):return dict(status='syntax_error')
    if not code.strip():return dict(status='empty_code')
    with tempfile.TemporaryDirectory(prefix='frontier-sandbox-') as tmp:
        os.chmod(tmp,0o755)
        # -I excludes current directory; explicitly load only our trusted harness.
        script='import sys;sys.path.insert(0,'+repr(str(ROOT))+');from runpy import run_module;run_module("experiments.coding_frontier.sandbox_worker",run_name="__main__")'
        p=subprocess.Popen([sys.executable,'-I','-c',script,tmp],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},start_new_session=True)
        payload=dict(spec,code=code)
        if spec['kind']=='io':payload.pop('outputs') # Expected answers stay in trusted parent.
        try:out,err=p.communicate(json.dumps(payload),timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL);p.communicate();return dict(status='sandbox_timeout')
        if p.returncode:
            if p.returncode<0:return dict(status='sandbox_resource_limit',signal=-p.returncode)
            if 'FRONTIER_SANDBOX_READY' in err:return dict(status='execution_error',exit_code=p.returncode)
            raise RuntimeError('Sandbox infrastructure failed: '+err[-600:])
        try:result=json.loads(out)
        except ValueError:
            if 'FRONTIER_SANDBOX_READY' in err:return dict(status='execution_error',detail='invalid_candidate_output')
            raise RuntimeError('Sandbox returned invalid result')
        if result.get('status')=='executed':
            import math
            from experiments.shorterbetter.multidomain.code_worker import equal
            actual=result['outputs'];expected=spec['outputs'];oks=[]
            for a,b in zip(actual,expected):
                if spec.get('fn_name'):ok=equal(a,b) or (isinstance(b,list) and len(b)==1 and equal(a,b[0]))
                else:
                    av=a.split();bv=('\n'.join(map(str,b)) if isinstance(b,list) else str(b)).split();ok=av==bv
                    if not ok:
                        try:ok=len(av)==len(bv) and all(math.isclose(float(x),float(y),rel_tol=1e-6,abs_tol=1e-8) for x,y in zip(av,bv))
                        except (ValueError,TypeError):ok=False
                oks.append(ok)
            return dict(status='passed' if len(actual)==len(expected) and all(oks) else 'test_failure',tests_passed=sum(oks),tests_total=len(expected))
        return result

def grade(text,finish_reason,spec):
    if finish_reason!='stop':return dict(correct=False,status='context_limit',tests_passed=None,tests_total=None)
    if not reasoning_is_complete(text):return dict(correct=False,status='incomplete_reasoning',tests_passed=None,tests_total=None)
    value=check(first_code_block(text),spec)
    return dict(value,correct=value['status']=='passed')
