import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
import subprocess
import isolated_cloud


class IsolationTests(unittest.TestCase):
    def test_bootstrap_retries_only_idempotent_transport_failures(self):
        cloud=Mock();sync=Mock()
        cloud.remote.side_effect=[subprocess.CalledProcessError(255,'ssh'),None]
        with patch.object(isolated_cloud.time,'sleep'):
            isolated_cloud.probe_uv_bootstrap(cloud,sync,{},'uv','/workspace/project/uv','pass',{},retry_transport=True)
        self.assertEqual(cloud.remote.call_count,2);sync.assert_not_called()
        cloud.remote.reset_mock();cloud.remote.side_effect=[subprocess.CalledProcessError(1,'python'),None]
        sync.side_effect=[subprocess.CalledProcessError(255,'rsync'),None]
        with patch.object(isolated_cloud.time,'sleep'):
            isolated_cloud.probe_uv_bootstrap(cloud,sync,{},'uv','/workspace/project/uv','pass',{},retry_transport=True)
        self.assertEqual(sync.call_count,2)
        self.assertEqual(cloud.remote.call_count,2)

    def test_upload_retry_is_bounded_and_only_transient(self):
        sync=Mock(side_effect=[subprocess.CalledProcessError(255,'rsync'),None])
        with patch.object(isolated_cloud.time,'sleep') as sleep:
            isolated_cloud.retry_upload(sync,{},'source','target',upload=True,timeout=90)
        self.assertEqual(sync.call_count,2);sleep.assert_called_once_with(5)
        self.assertLessEqual(sync.call_args.kwargs['timeout'],90)
        for upload,code in [(False,255),(True,23)]:
            sync=Mock(side_effect=subprocess.CalledProcessError(code,'rsync'))
            with self.assertRaises(subprocess.CalledProcessError):
                isolated_cloud.retry_upload(sync,{},'source','target',upload=upload)
            self.assertEqual(sync.call_count,1)
        sync=Mock(side_effect=subprocess.CalledProcessError(255,'rsync'))
        with patch.object(isolated_cloud.time,'monotonic',side_effect=[0,0,90]):
            with self.assertRaises(subprocess.CalledProcessError):
                isolated_cloud.retry_upload(sync,{},'source','target',upload=True,timeout=90)
        self.assertEqual(sync.call_count,1)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ['experiments','ThinkingCap-Test','lab/experiments/data']:(self.root/name).mkdir(parents=True)
        (self.root/'experiments/entry.sh').write_text('exit 0\n')
        (self.root/'lab/experiments/data/pilot.jsonl').write_text('{}\n')
        (self.root/'lab/gpu-requirements.lock').write_text('torch==2.8.0\n')

    def test_source_and_data_tamper_fail_closed(self):
        isolated_cloud.seal(self.root);isolated_cloud.verify(self.root)
        for relative in ['experiments/entry.sh','lab/experiments/data/pilot.jsonl']:
            p=self.root/relative;before=p.read_text();p.chmod(0o644);p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):isolated_cloud.verify(self.root)
            p.write_text(before)
        isolated_cloud.verify(self.root)

    def test_no_secret_or_mutable_symlink(self):
        (self.root/'experiments/.env').write_text('NOT_A_REAL_SECRET')
        with self.assertRaisesRegex(ValueError,'Private'):isolated_cloud.seal(self.root)
        (self.root/'experiments/.env').unlink()
        (self.root/'experiments/link').symlink_to(self.root/'lab/experiments/data/pilot.jsonl')
        with self.assertRaisesRegex(ValueError,'symlinks'):isolated_cloud.seal(self.root)

    def test_only_payload_origin_changes_shared_ledger_and_gpu_policy_preserved(self):
        from experiments import cloud
        isolated_cloud.seal(self.root)
        state=cloud.STATE;old_root=cloud.ROOT;capacity=cloud.capacity_rejected
        try:
            with patch('sys.argv',['isolated_cloud.py','run','--release',str(self.root),'--name','test','--entry','experiments/entry.sh']),patch.object(cloud,'run') as run:
                isolated_cloud.main()
                self.assertEqual(run.call_args.args[0].gpu,'NVIDIA GeForce RTX 4090')
                self.assertEqual(cloud.ROOT,self.root)
                self.assertEqual(cloud.STATE,state)
                self.assertTrue(cloud.capacity_rejected(RuntimeError('Runpod HTTP 400: This machine does not have the resources to deploy your pod')))
                self.assertFalse(cloud.capacity_rejected(RuntimeError('Runpod HTTP 500: uncertain')))
        finally:cloud.ROOT=old_root;cloud.capacity_rejected=capacity


class TeacherSelectionTests(unittest.TestCase):
    def test_minimum_source_yield_precedes_total_and_length(self):
        from teacher_pipeline import select_teacher,SOURCES
        rows=[]
        for model,counts in [('unbalanced',[6,6,6,1]),('balanced',[3,3,3,3])]:
            for source,count in zip(SOURCES,counts):
                for i in range(count):rows.append({'teacher_model':model,'variant':'concise','source':source,
                    'prompt_id':source+str(i),'replay_eligible':True,'total_tokens':5,'cost':{'accounted_usd':.01}})
        self.assertEqual(select_teacher(rows)[0][0],'balanced')

    def test_matched_yield_gate_and_pool_token_contract(self):
        import teacher_pipeline as pipeline
        from types import SimpleNamespace
        class Tokenizer:
            eos_token_id=9
            def encode(self,*a,**k):return [3,4]
            def apply_chat_template(self,*a,**k):return [1,2]
        cohort={};teacher=[];student=[];reviews=[]
        for source in pipeline.SOURCES:
            for i in range(16):
                pid=source+str(i);cohort[pid]={'source':source,'prompt':[],'answer':'2','task':'math'}
                rev={'verdict':'pass','evidence':'checked','request_key':pid}
                teacher.append({'prompt_id':pid,'replay_eligible':True,'completion':'solution',
                                'response_sha256':pid,'explanation_review':rev})
                student.append({'prompt_id':pid,'accepted_answer':True,'completion':'solution',
                                'response_sha256':pid,'request_key':pid})
                reviews.append(rev)
        with tempfile.TemporaryDirectory() as td,patch.object(pipeline,'STATE',Path(td)):
            ids,pools=pipeline.build_pools(teacher,student,cohort,Tokenizer(),reviews,SimpleNamespace(REVISION='pinned'))
            self.assertEqual(len(ids),64)
            self.assertEqual([r['prompt_id'] for r in pools['teacher']],[r['prompt_id'] for r in pools['student']])
            self.assertTrue(all(r['training_completion_ids']==[3,4,9] for r in pools['teacher']))
            reviews[0]['verdict']='fail'
            self.assertIsNone(pipeline.build_pools(teacher,student,cohort,Tokenizer(),reviews,SimpleNamespace(REVISION='pinned')))

    def test_completion_archive_is_verified_and_tampering_detected(self):
        import teacher_pipeline as pipeline
        with tempfile.TemporaryDirectory() as td,patch.object(pipeline,'STATE',Path(td)):
            data=Path(td)/'api';data.mkdir();(data/'response.json').write_text('{}')
            (data/'.env').write_text('fake excluded value')
            pipeline.backup_api(data,'test')
            manifest=json.loads((Path(td)/'backups/test.verified.json').read_text())
            self.assertNotIn('.env',manifest['files'])
            (data/'response.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'checksum'):pipeline.backup_api(data,'test')


if __name__=='__main__':unittest.main()
