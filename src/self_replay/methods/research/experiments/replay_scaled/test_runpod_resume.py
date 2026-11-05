import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import core
from .resume_stage import stage

class RunpodResumeTests(unittest.TestCase):
    def test_1024_continuation_requires_1280_boundary_and_restores_complete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';source=data/'resume-training'
            core.write(data/'plan.json',{'seed':17,'total':3840})
            core.write(source/'checkpoints/group-001024/complete.json',dict(group=1024,world_size=2,pilot=False,contract=core.sha(data/'plan.json')))
            core.write(source/'state-evidence.json',{'optimizer_updates':1023,'cursor':1024})
            core.write(data/'resume-files.json',{str(p.relative_to(source)):core.sha(p) for p in source.rglob('*') if p.is_file()})
            for invalid in ({'resume_group':1024},{'resume_group':1024,'stop_after':1536}):
                core.write(data/'launch-mode.json',invalid)
                with self.assertRaisesRegex(ValueError,'Unregistered'):stage(data,root/'output')
                self.assertFalse((root/'output').exists())
            core.write(data/'launch-mode.json',{'resume_group':1024,'stop_after':1280})
            with patch('experiments.replay_scaled.resume_stage.write') as recorded:
                stage(data,root/'output')
                self.assertEqual(recorded.call_args.args[1]['group'],1024)
            self.assertEqual((root/'output/state-evidence.json').read_bytes(),(source/'state-evidence.json').read_bytes())
            self.assertEqual(core.checkpoint_boundaries(1280,3840,{'save_every':32,'validate_every':256,'milestones':[]},stop_after=1280),(True,True))

    def test_ticket_only_allows_named_verified_two_rank_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);name='replay-scaled-runpod-resume-test';parent='replay-scaled-training-20261101t122440z'
            manifest=state/parent/'output/training/checkpoints/group-000850/complete.json'
            core.write(manifest,{'group':850})
            core.write(state/'ledger.json',{'jobs':{parent:{'released_at':1,'backup_verified':True}}})
            v=dict(job_name=name,scope='replay_runpod_resume_v1',authorized=True,provider='runpod',hardware_scope=core.SCOPE,
                entry=core.ENTRY,gpu=core.GPU,gpu_count=2,phase='training',resume_group=850,resume_parent=parent,
                resume_manifest_sha256=core.sha(manifest),max_cost_usd=65,max_runtime_hours=24,max_gpu_rate=1.79,
                source_user_instruction='Continue unchanged850 on RunPod')
            path=state/'authorizations'/(name+'.json');core.write(path,v)
            self.assertIsNotNone(core.ticket(state,name,v))
            waiting={**v,'requested_gpu':core.GPU};waiting.pop('gpu')
            self.assertIsNotNone(core.ticket(state,name,waiting))
            for change in [dict(gpu_count=4),dict(provider='vast'),dict(max_cost_usd=66),dict(resume_group=832),dict(resume_manifest_sha256='wrong')]:
                core.write(path,{**v,**change})
                self.assertIsNone(core.ticket(state,name,{**v,**change}))
            core.write(path,v);core.write(state/'ledger.json',{'jobs':{parent:{'released_at':None,'backup_verified':True}}})
            self.assertIsNone(core.ticket(state,name,v))

    def test_restore_preserves_files_and_refuses_corruption_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'data';output=root/'output/training';source=data/'resume-training'
            core.write(data/'plan.json',{'seed':17});core.write(data/'launch-mode.json',{'resume_group':850})
            core.write(source/'checkpoints/group-000850/complete.json',dict(group=850,world_size=2,pilot=False,contract=core.sha(data/'plan.json')))
            core.write(source/'history.json',{'previous':[128,256,768]})
            (source/'history.json').chmod(0o444)
            hashes={str(p.relative_to(source)):core.sha(p) for p in source.rglob('*') if p.is_file()}
            core.write(data/'resume-files.json',hashes)
            with patch('experiments.replay_scaled.resume_stage.write'):
                stage(data,output)
                self.assertEqual((source/'history.json').read_bytes(),(output/'history.json').read_bytes())
                self.assertTrue((output/'history.json').stat().st_mode & stat.S_IWUSR)
                self.assertFalse((source/'history.json').stat().st_mode & stat.S_IWUSR)
                with self.assertRaisesRegex(ValueError,'overwrite'):stage(data,output)
                core.write(source/'history.json',{'changed':True})
                with self.assertRaisesRegex(ValueError,'changed'):stage(data,root/'another')

    def test_sealed_logs_can_be_appended_without_root_privileges(self):
        with tempfile.TemporaryDirectory(dir='/dev/shm') as tmp:
            root=Path(tmp);root.chmod(0o755);data=root/'data';source=data/'resume-training'
            core.write(data/'plan.json',{'seed':17});core.write(data/'launch-mode.json',{'resume_group':850})
            core.write(source/'checkpoints/group-000850/complete.json',dict(group=850,world_size=2,pilot=False,contract=core.sha(data/'plan.json')))
            (source/'training-records.jsonl').write_text('{"group":850}\n')
            core.write(data/'resume-files.json',{str(p.relative_to(source)):core.sha(p) for p in source.rglob('*') if p.is_file()})
            for p in data.rglob('*'):
                if p.is_file():p.chmod(0o444)
            (root/'output').mkdir(mode=0o777);(root/'output').chmod(0o777)
            repo=Path(__file__).resolve().parents[2]
            code=f"import sys;sys.path.insert(0,{str(repo)!r});from pathlib import Path;from experiments.replay_scaled.resume_stage import stage;stage(Path('data'),Path('output/training'));f=Path('output/training/training-records.jsonl');f.open('a').write('appended\\n')"
            kwargs=dict(user=65534,group=65534,extra_groups=[]) if os.geteuid()==0 else {}
            result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,**kwargs)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((root/'output/training/training-records.jsonl').read_text(),'{"group":850}\nappended\n')
            self.assertEqual((source/'training-records.jsonl').read_text(),'{"group":850}\n')

    def test_repair_deducts_charge_and_rejects_newer_progress(self):
        from ops import replay_runpod_resume as r
        with tempfile.TemporaryDirectory() as tmp:
            state=Path(tmp);name='failed'
            core.write(state/name/'output/training/progress.json',{'current':850,'optimizer_updates':849})
            jobs={name:dict(state='failed',released_at=1,backup_verified=True,charged_estimate_usd=.8202)}
            with patch.object(r.cloud,'STATE',state):
                self.assertEqual(r.repair_budget(jobs,[name]),64.1798)
                core.write(state/name/'output/training/progress.json',{'current':864,'optimizer_updates':863})
                with self.assertRaisesRegex(RuntimeError,'New training state'):r.repair_budget(jobs,[name])

    def test_low_backup_space_shortens_lease_without_resetting_state(self):
        from experiments import cloud
        from types import SimpleNamespace
        name='replay-scaled-runpod-resume-test'
        job=dict(provider='runpod',hardware_scope=core.SCOPE,hard_deadline=10000)
        with patch.object(cloud.os,'statvfs',return_value=SimpleNamespace(f_bfree=1,f_bavail=1,f_frsize=4096)),patch.object(cloud.time,'time',return_value=1000),patch.object(cloud,'remote') as remote,patch.object(cloud,'update') as update:
            cloud.protect_replay_backup_space({},name,job)
            remote.assert_called_once()
            self.assertIn('worker-lease.json',remote.call_args.args[1])
            self.assertEqual(update.call_args.kwargs['hard_deadline'],1900)
            self.assertNotIn('max_cost_usd',update.call_args.kwargs)
            cloud.protect_replay_backup_space({},name,{**job,'storage_stop_requested_at':1000})
            self.assertEqual(remote.call_count,1)

    def test_empty_capacity_does_not_launch_anything(self):
        from ops import replay_runpod_resume as r
        with tempfile.TemporaryDirectory() as tmp:
            reg=Path(tmp)/'reg';state=Path(tmp)/'state'
            core.write(reg/'campaign.json',dict(state='queued_runpod',queue_expires_at=10**12))
            core.write(reg/'launch.json',dict(job_name='queued',max_gpu_rate=1.79))
            core.write(state/'ledger.json',{'jobs':{}})
            with patch.object(r,'REG',reg),patch.object(r.cloud,'STATE',state),patch.object(r.runpod,'account',return_value={'clientBalance':69.5,'currentSpendPerHr':0}),patch.object(r.runpod,'api',return_value={'availability':'NONE','price':{'secure':1.79}}),patch.object(r.subprocess,'run') as launch:
                r.check();launch.assert_not_called()
                self.assertEqual(r.read(reg/'campaign.json')['state'],'queued_runpod')

if __name__=='__main__':unittest.main()
