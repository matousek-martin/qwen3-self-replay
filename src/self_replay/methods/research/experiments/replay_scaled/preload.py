"""Fetch the pinned base into the Hub cache, checking every file before use."""
import hashlib,json,os,urllib.request
from pathlib import Path
from experiments.study_eval import BASE_MODEL,BASE_REVISION

def main():
    source=Path('data/replay-evaluation/model-preload.json')
    if not source.exists():return
    m=json.loads(source.read_text())
    if m['model']!=BASE_MODEL or m['revision']!=BASE_REVISION:raise ValueError('Unpinned preload')
    folder=Path(os.environ['HF_HOME'])/'hub/models--Qwen--Qwen3-0.6B/snapshots'/BASE_REVISION
    folder.mkdir(parents=True,exist_ok=True)
    for name,expected in m['files_sha256'].items():
        if Path(name).name!=name:raise ValueError('Invalid preload filename')
        target=folder/name
        if target.exists():
            with target.open('rb') as f:
                if hashlib.file_digest(f,'sha256').hexdigest()==expected:continue
        temp=target.with_suffix(target.suffix+'.preload')
        with urllib.request.urlopen(f'https://huggingface.co/{BASE_MODEL}/resolve/{BASE_REVISION}/{name}',timeout=120) as response,temp.open('wb') as out:
            while block:=response.read(8*1024*1024):out.write(block)
        with temp.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=expected:raise ValueError('Model preload SHA mismatch: '+name)
        temp.replace(target)
    Path('output/model-preload-verification.json').write_text(json.dumps(dict(m,verified_on_worker=True),indent=2)+'\n')
    print('Pinned base model files verified in Hub cache',flush=True)

if __name__=='__main__':main()
