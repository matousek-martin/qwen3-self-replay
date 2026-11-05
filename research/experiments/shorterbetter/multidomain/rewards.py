"""Verifiable domain correctness; SOL length reward stays in original objective."""
import ast,json,math,os,re,subprocess,tempfile
from pathlib import Path
from decimal import Decimal,InvalidOperation

IF_TYPES={'case:in english and lowercase':'lowercase','in english and capital':'uppercase',
 'punctuation:use no comma':'no_comma','format:use json format':'json','format:title':'title',
 'content:include a postscript':'postscript','use quotation':'quotation'}

def final_text(text):
    return text.rsplit('</think>',1)[-1].strip() if '</think>' in text else text.strip()

def unfence(text):
    m=re.fullmatch(r'```(?:python|json)?\s*\n(.*?)\n```',text.strip(),re.S)
    return m.group(1).strip() if m else text.strip()

def code_check(code,spec):
    if not code.strip() or len(code)>50000:return False
    with tempfile.TemporaryDirectory(prefix='sb-code-') as tmp:
        os.chmod(tmp,0o755)
        try:
            p=subprocess.run(['/usr/bin/python3','-I','-S',str(Path(__file__).with_name('code_worker.py')),tmp],
                input=json.dumps({'code':code,'inputs':spec['inputs'],'fn_name':spec.get('fn_name')}),text=True,capture_output=True,timeout=5,
                env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},start_new_session=True)
        except subprocess.TimeoutExpired:return False
        if p.returncode!=0:
            if 'isolate' in p.stderr or 'seccomp' in p.stderr or 'chroot' in p.stderr:raise RuntimeError('Code sandbox setup failed: '+p.stderr[-400:])
            return False
        # Hidden expected answers never enter the untrusted process.
        try:results=json.loads(p.stdout)['outputs']
        except (ValueError,KeyError,TypeError):return False
        if not isinstance(results,list) or len(results)!=len(spec['outputs']):return False
        from .code_worker import equal
        for actual,expected in zip(results,spec['outputs']):
            if spec.get('fn_name'):
                if not (equal(actual,expected) or (isinstance(expected,list) and len(expected)==1 and equal(actual,expected[0]))):return False
            else:
                if not isinstance(actual,str):return False
                a=actual.split();b=('\n'.join(map(str,expected)) if isinstance(expected,list) else str(expected)).split()
                if a!=b:
                    try:
                        if len(a)!=len(b) or not all(math.isclose(float(x),float(y),rel_tol=1e-6,abs_tol=1e-8) for x,y in zip(a,b)):return False
                    except ValueError:return False
        return True

def instruction_check(text,checks):
    if not text.strip():return False
    for c in checks:
        if c=='lowercase' and (text!=text.lower() or not any(x.isalpha() for x in text)):return False
        if c=='uppercase' and (text!=text.upper() or not any(x.isalpha() for x in text)):return False
        if c=='no_comma' and ',' in text:return False
        if c=='title' and not re.search(r'<<[^<>\n]+>>',text):return False
        if c=='postscript' and not re.search(r'(?im)^\s*P\.?\s*S\.?\s*:',text):return False
        if c=='quotation' and not (text.startswith('"') and text.endswith('"')):return False
        if c=='json':
            try:
                v=json.loads(text)
                if not isinstance(v,(dict,list)) or not v:return False
            except (ValueError,TypeError):return False
        if c not in IF_TYPES.values():raise ValueError('Unknown instruction verifier')
    return bool(checks)

def tool_check(text,gold):
    try:values=json.loads(unfence(text))
    except ValueError:return False
    if not isinstance(values,list):return False
    normalized=[]
    for v in values:
        if not isinstance(v,dict) or set(v)!={'name','arguments'} or not isinstance(v['name'],str) or not isinstance(v['arguments'],dict):return False
        normalized.append(v)
    return sorted((json.dumps(v,sort_keys=True,separators=(',',':')) for v in normalized))==sorted(json.dumps(v,sort_keys=True,separators=(',',':')) for v in gold)

def verify(text,answer):
    spec=json.loads(answer);text=final_text(text);kind=spec['task']
    if kind=='math':
        matches=re.findall(r'\\boxed\{\s*([^{}]+)\s*\}',text)
        if not matches:return False
        try:return Decimal(matches[-1].replace(',','').strip())==Decimal(spec['value'].replace(',','').strip())
        except InvalidOperation:return False
    if kind=='coding':return code_check(unfence(text),spec['tests'])
    if kind=='tools':return tool_check(text,spec['calls'])
    if kind=='instructions':return instruction_check(text,spec['checks'])
    raise ValueError('Unknown task')
