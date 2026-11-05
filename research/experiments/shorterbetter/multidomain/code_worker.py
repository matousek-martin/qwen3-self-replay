"""Standalone untrusted-code harness: empty chroot, unprivileged UID, seccomp, limits.
No credentials, host filesystem, networking, or child processes are available.
Run using system Python, not the GPU virtual environment.
"""
import ast,bisect,collections,contextlib,copy,ctypes,datetime,decimal,enum,errno,fractions,functools,heapq,io,itertools,json,math,operator,os,random,re,resource,signal,statistics,string,sys,time,types,typing

def isolate(root):
    if os.geteuid()!=0: raise RuntimeError('Code sandbox requires root for chroot then drops privileges')
    sec=ctypes.CDLL('libseccomp.so.2');libc=ctypes.CDLL(None,use_errno=True)
    sec.seccomp_init.argtypes=[ctypes.c_uint32];sec.seccomp_init.restype=ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];sec.seccomp_syscall_resolve_name.restype=ctypes.c_int
    sec.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    sec.seccomp_load.argtypes=[ctypes.c_void_p];sec.seccomp_release.argtypes=[ctypes.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)
    for name in ('socket','socketpair','connect','bind','listen','accept','accept4','execve','execveat','fork','vfork','clone','clone3','ptrace','process_vm_readv','process_vm_writev','mount','umount2','pivot_root','unshare','setns','bpf','perf_event_open','keyctl','add_key','request_key','io_uring_setup','io_uring_enter','io_uring_register','open_by_handle_at'):
        nr=sec.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and sec.seccomp_rule_add(ctx,0x00050000|errno.EPERM,nr,0)!=0: raise RuntimeError('seccomp rule failed')
    os.chdir(root);os.chroot(root);os.chdir('/')
    os.setgroups([]);os.setgid(65534);os.setuid(65534)
    for which,value in [(resource.RLIMIT_CPU,3),(resource.RLIMIT_AS,512*1024**2),(resource.RLIMIT_FSIZE,1024**2),(resource.RLIMIT_NPROC,0),(resource.RLIMIT_NOFILE,32),(resource.RLIMIT_CORE,0)]:resource.setrlimit(which,(value,value))
    if libc.prctl(38,1,0,0,0)!=0 or sec.seccomp_load(ctx)!=0:raise RuntimeError('seccomp activation failed')
    sec.seccomp_release(ctx)

def equal(actual,expected):
    if isinstance(actual,tuple):actual=list(actual)
    if isinstance(actual,list) and isinstance(expected,list):return len(actual)==len(expected) and all(equal(a,b) for a,b in zip(actual,expected))
    if isinstance(actual,dict) and isinstance(expected,dict):return actual.keys()==expected.keys() and all(equal(actual[k],expected[k]) for k in actual)
    if isinstance(actual,(int,float)) and isinstance(expected,(int,float)):return math.isclose(actual,expected,rel_tol=1e-6,abs_tol=1e-8)
    return actual==expected

def run(payload):
    code=compile(payload['code'],'<candidate>','exec');fn=payload.get('fn_name');outputs=[]
    for inp in payload['inputs']:
        namespace={'__name__':'__main__'}
        if isinstance(inp,list) and not fn:inp='\n'.join(str(v) for v in inp)
        sys.stdin=io.TextIOWrapper(io.BytesIO((str(inp)+'\n').encode()))
        with contextlib.redirect_stdout(io.StringIO()) as stdout,contextlib.redirect_stderr(io.StringIO()):
            try:exec(code,namespace)
            except SystemExit as exc:
                if exc.code not in (None,0):raise ValueError('Nonzero candidate exit')
            if fn:
                target=getattr(namespace['Solution'](),fn) if 'Solution' in namespace else namespace[fn]
                actual=target(*inp) if isinstance(inp,list) else target(inp)
                outputs.append(actual)
            else:
                outputs.append(stdout.getvalue())
    return outputs

if __name__=='__main__':
    payload=json.load(sys.stdin);root=sys.argv[1]
    isolate(root) # Failure is a broken verifier, never an incorrect model answer.
    try:result={'outputs':run(payload)};encoded=json.dumps(result,allow_nan=False).encode()
    except BaseException:encoded=b'{"candidate_error":true}'
    os.write(1,encoded+b'\n')
