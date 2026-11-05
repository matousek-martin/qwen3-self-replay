"""Untrusted candidate AND supplied tests execute only after empty chroot/drop/seccomp.

Function-only pytest tests supported, including parametrize/raises/approx;
fixtures, classes and skip/xfail are explicitly unsupported, never silently passed.
"""
import ast, contextlib, inspect, io, itertools, json, os, sys, types
import pytest
import numpy
import unittest, unittest.mock, textwrap, csv, fractions, decimal, collections, heapq
from experiments.shorterbetter.multidomain.code_worker import isolate, run, equal

def python_tests(payload):
    tree=ast.parse(payload['test'])
    if any(isinstance(n,ast.ClassDef) for n in tree.body):
        return dict(status='unsupported_test',detail='test_class')
    if 'pytest.fixture' in payload['test']:
        return dict(status='unsupported_test',detail='fixture')
    mod=types.ModuleType('solution');sys.modules['solution']=mod
    exec(compile(payload['code'],'<candidate>','exec'),mod.__dict__)
    ns={'__name__':'frontier_tests'}
    exec(compile(tree,'<supplied_tests>','exec'),ns)
    tests=[v for k,v in ns.items() if k.startswith('test_') and inspect.isfunction(v)]
    if not tests:return dict(status='unsupported_test',detail='no_tests')
    passed=failed=0;errors=[]
    for fn in tests:
        cases=[{}]
        for mark in getattr(fn,'pytestmark',[]):
            if mark.name!='parametrize':return dict(status='unsupported_test',detail='pytest_mark:'+mark.name)
            if mark.kwargs.get('indirect'):return dict(status='unsupported_test',detail='indirect')
            names=mark.args[0];names=[n.strip() for n in names.split(',')] if isinstance(names,str) else names
            values=[]
            for v in mark.args[1]:
                if hasattr(v,'marks'):
                    if v.marks:return dict(status='unsupported_test',detail='marked_parameter')
                    v=v.values
                elif len(names)==1:v=(v,)
                values.append(dict(zip(names,v)))
            cases=[{**a,**b} for a in cases for b in values]
        for args in cases:
            try:inspect.signature(fn).bind(**args)
            except TypeError:return dict(status='unsupported_test',detail='fixture_argument')
            try:fn(**args);passed+=1
            except BaseException as e:failed+=1;errors.append(type(e).__name__)
    if not passed+failed:return dict(status='unsupported_test',detail='zero_test_cases')
    return dict(status='passed' if failed==0 else 'test_failure',tests_passed=passed,tests_total=passed+failed,error_types=sorted(set(errors)))

def evaluate(payload):
    if payload['kind']=='python_tests':return python_tests(payload)
    return dict(status='executed',outputs=run(payload))

if __name__=='__main__':
    payload=json.load(sys.stdin)
    isolate(sys.argv[1])
    os.write(2,b'FRONTIER_SANDBOX_READY\n')
    try:
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):result=evaluate(payload)
        encoded=(json.dumps(result,allow_nan=False)+'\n').encode()
    except BaseException as e:encoded=(json.dumps(dict(status='execution_error',error_type=type(e).__name__))+'\n').encode()
    os.write(1,encoded)
