import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]


class CompletionTests(unittest.TestCase):
    def test_parallel_reviews_use_disjoint_items_and_preserve_order(self):
        import teacher_review_completion as recovery
        with tempfile.TemporaryDirectory() as td,patch.object(recovery.previous,'STATE',Path(td)),patch.object(recovery.previous,'backup_api'):
            items=[{'request_key':str(i)} for i in range(12)]
            def evaluate(items,label,runner,common):
                return [{'request_key':r['request_key'],'verdict':'pass'} for r in items]
            with patch.object(recovery,'serial_review',side_effect=evaluate) as api:
                result=recovery.review(items,'new',None,None)
                self.assertEqual([r['request_key'] for r in result],[str(i) for i in range(12)])
                observed=[r['request_key'] for call in api.call_args_list for r in call.args[0]]
                self.assertEqual(sorted(observed),sorted(str(i) for i in range(12)))
                self.assertEqual(len(observed),len(set(observed)))
                self.assertEqual(api.call_count,4)

    def test_placement_policy_never_falls_back_to_unlisted_datacenter(self):
        from experiments import cloud
        with tempfile.TemporaryDirectory() as td,patch.object(cloud,'STATE',Path(td)):
            (Path(td)/'placement-policy.json').write_text(json.dumps({'job_prefixes':['teacher-completion-'],'data_center_ids':['EU-CZ-1']}))
            quote={'dataCenters':[{'id':'US-IL-1','availability':'HIGH'},{'id':'EU-CZ-1','availability':'NONE'}]}
            self.assertEqual(cloud.placement_constraints('teacher-completion-test',quote),[])
            quote['dataCenters'][1]['availability']='LOW'
            self.assertEqual(cloud.placement_constraints('teacher-completion-test',quote),['EU-CZ-1'])
            self.assertIsNone(cloud.placement_constraints('other-existing-study',quote))

    def test_server_error_continuation_never_resends_saved_requests(self):
        import teacher_review_completion as recovery
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'review';(out/'run/requests').mkdir(parents=True)
            items=[{'prompt_id':str(i),'request_key':'sample'+str(i),'response_sha256':'answer'+str(i)} for i in range(3)]
            requests=[{'prompt_id':str(i),'request_key':'request'+str(i),'target_sha256':'answer'+str(i)} for i in range(3)]
            (out/'registered-plan.json').write_text(json.dumps({'requests':requests}))
            receipts=[{'request':requests[0],'state':'completed','response_sha256':'response0',
                       'response':{'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"verdict":"pass","evidence":"Checked."}'}]}]}},
                      {'request':requests[1],'state':'needs_reconciliation','wire':{'http_status':503}}]
            for i,r in enumerate(receipts):(out/f'run/requests/request{i}.json').write_text(json.dumps(r))
            before=(out/'run/requests/request1.json').read_bytes()
            def fresh(items,label,runner,common):
                self.assertEqual([r['request_key'] for r in items],['sample2'])
                return [{'request_key':'sample2','verdict':'pass'}]
            with patch.object(recovery.previous,'STATE',root),patch.object(recovery.previous,'review',side_effect=fresh) as api,patch.object(recovery.previous,'backup_api'),patch.object(recovery.time,'sleep'):
                result=recovery.review(items,'review',None,SimpleNamespace(digest=lambda x:'digest'))
                self.assertEqual([r['verdict'] for r in result],['pass','fail','pass'])
                self.assertEqual(api.call_count,1)
                recovery.review(items,'review',None,None)
                self.assertEqual(api.call_count,1)
            self.assertEqual((out/'run/requests/request1.json').read_bytes(),before)

    def test_unknown_transport_failure_is_not_automatically_continued(self):
        import teacher_review_completion as recovery
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'review';(out/'run/requests').mkdir(parents=True)
            item={'prompt_id':'p','request_key':'sample','response_sha256':'answer'}
            request={'prompt_id':'p','request_key':'request','target_sha256':'answer'}
            (out/'registered-plan.json').write_text(json.dumps({'requests':[request]}))
            (out/'run/requests/request.json').write_text(json.dumps({'request':request,'state':'dispatching'}))
            with patch.object(recovery.previous,'STATE',root),patch.object(recovery.previous,'review') as api:
                with self.assertRaisesRegex(RuntimeError,'Unclassified'):recovery.review([item],'review',None,None)
                api.assert_not_called()

    def test_nltk_resource_path_reaches_child_grader(self):
        from experiments import study_eval
        from types import SimpleNamespace
        fake=SimpleNamespace(data=SimpleNamespace(path=[],find=lambda name: True))
        with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'nltk':fake}),patch.dict(os.environ,{'NLTK_DATA':td}):
            study_eval.preflight_nltk()
            actual=subprocess.check_output(['/usr/bin/python3','-c','import os; print(os.environ["NLTK_DATA"])'],text=True).strip()
            self.assertEqual(actual,td)
            self.assertEqual(Path(td).stat().st_mode&0o777,0o700)
    def test_student_normalization_does_not_rewrite_solution(self):
        from teacher_student_samples import normalize
        self.assertEqual(normalize('<think>\n2+2=4\n</think>\nFinal answer: 4'), '2+2=4\n</think>\nFinal answer: 4')
        self.assertEqual(normalize('2+2=4\n</think>\nFinal answer: 4'), '2+2=4\n</think>\nFinal answer: 4')

    def test_final_answer_cap_preserves_only_generated_prefix(self):
        from teacher_matched_vllm import truncate_final
        self.assertEqual(truncate_final([1,2,99,3,4,5],99,2),([1,2,99,3,4],True))
        self.assertEqual(truncate_final([1,2,3],99,2),([1,2,3],False))
        self.assertEqual(truncate_final([1,99,3,4],99,2),([1,99,3,4],False))

    def test_paired_report_checks_coverage_and_token_ratio(self):
        from completion_report import paired_summary
        with tempfile.TemporaryDirectory() as td:
            left=[];right=[]
            for dataset in ['bfcl_simple_python','ifeval','mbpp','svamp']:
                for i in range(128):
                    for sample in range(2):
                        r={'dataset':dataset,'id':str(i),'sample_index':sample,'correct':True,'output_tokens':100}
                        left.append(r);right.append(dict(r,output_tokens=80))
            a=Path(td)/'a.jsonl';b=Path(td)/'b.jsonl'
            a.write_text(''.join(json.dumps(r)+'\n' for r in left));b.write_text(''.join(json.dumps(r)+'\n' for r in right))
            result=paired_summary([(17,a,b)],resamples=50)
            self.assertAlmostEqual(result['token_ratio'],.8);self.assertEqual(result['accuracy_delta'],0)
            heterogeneous=[dict(r,output_tokens=50 if r['dataset'] in ['bfcl_simple_python','ifeval'] else 200) for r in right]
            b.write_text(''.join(json.dumps(r)+'\n' for r in heterogeneous))
            self.assertAlmostEqual(paired_summary([(17,a,b)],resamples=5)['token_ratio'],1.0)
            b.write_text(''.join(json.dumps(dict(r,output_tokens=50))+'\n' for r in right))
            c=Path(td)/'c.jsonl';d=Path(td)/'d.jsonl'
            c.write_text(''.join(json.dumps(dict(r,output_tokens=400))+'\n' for r in left))
            d.write_text(''.join(json.dumps(dict(r,output_tokens=800))+'\n' for r in right))
            pairs=[(17,a,b),(29,c,d)]
            self.assertAlmostEqual(paired_summary(pairs,resamples=5,aggregation='seed_domain_geometric')['token_ratio'],1.0)
            self.assertAlmostEqual(paired_summary(pairs,resamples=5)['token_ratio'],1.7)
            b.write_text(''.join(json.dumps(r)+'\n' for r in right[:-1]))
            with self.assertRaisesRegex(ValueError,'coverage'):paired_summary([(17,a,b)],resamples=10)

    def test_parallel_campaign_preserves_seed_identity_and_completed_work(self):
        import parallel_campaign
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ledger=root/'cloud';ledger.mkdir()
            config={'id':'test','seeds':[17,29,43],'hours':6}
            (ledger/'ledger.json').write_text(json.dumps({'jobs':{f'test-seed{s}-hash':{'pod_id':str(s)} for s in config['seeds']}}))
            def fake_job(release,name,entry,hours):
                seed=int(name.split('seed')[1].split('-')[0]);output=root/name;output.mkdir()
                self.assertEqual(entry,f'experiments/run-isolated-seed{seed}.sh')
                (output/'role-result.json').write_text(json.dumps({'seed':seed,'development_complete':True}))
                return output
            with patch.object(parallel_campaign.cloud,'STATE',ledger),patch.object(parallel_campaign,'run_job',side_effect=fake_job) as jobs:
                parallel_campaign.run(root,root/'campaign',config,'hash',3)
                self.assertEqual(jobs.call_count,3)
                state=json.loads((root/'campaign/campaign.json').read_text());self.assertEqual(state['state'],'development_complete')
                parallel_campaign.run(root,root/'campaign',config,'hash',3)
                self.assertEqual(jobs.call_count,3)

    def test_clean_training_capacity_refusal_stays_queued_with_audited_new_identity(self):
        import parallel_campaign
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ledger=root/'cloud';ledger.mkdir()
            original='test-seed17-hash'
            (ledger/'ledger.json').write_text(json.dumps({'jobs':{original:{'state':'allocation_unavailable',
                'safely_no_pod':True,'released_at':1}}}))
            called=[]
            def fake(release,name,entry,hours):
                called.append(name)
                if name==original:raise RuntimeError('No capacity, no pod')
                output=root/name;output.mkdir();(output/'role-result.json').write_text(json.dumps({'seed':17,'development_complete':True}));return output
            with patch.object(parallel_campaign.cloud,'STATE',ledger),patch.object(parallel_campaign,'run_job',side_effect=fake),patch.object(parallel_campaign.time,'sleep'):
                parallel_campaign.run(root,root/'campaign',{'id':'test','seeds':[17],'hours':6},'hash',2)
            self.assertEqual(called,[original,original+'-capacity2'])
            state=json.loads((root/'campaign/campaign.json').read_text())
            self.assertEqual(state['trials']['17']['capacity_refusals'],[original])
            self.assertEqual(state['state'],'development_complete')

    def test_failed_or_ambiguous_training_is_not_treated_as_capacity_refusal(self):
        import parallel_campaign
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);ledger=root/'cloud';ledger.mkdir()
            (ledger/'ledger.json').write_text(json.dumps({'jobs':{'test-seed17-hash':{
                'state':'failed','pod_id':'old','released_at':1}}}))
            with patch.object(parallel_campaign.cloud,'STATE',ledger),patch.object(parallel_campaign,'run_job',side_effect=RuntimeError('Worker failed')) as job:
                with self.assertRaisesRegex(RuntimeError,'inspection'):
                    parallel_campaign.run(root,root/'campaign',{'id':'test','seeds':[17],'hours':6},'hash',2)
                self.assertEqual(job.call_count,1)

    def test_readonly_questions_stage_as_unprivileged_user(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);root.chmod(0o777)
            source=root/'source';source.mkdir();source.chmod(0o755)
            for name,body in [('questions.jsonl','original'),('manifest.json','{}')]:
                p=source/name;p.write_text(body);p.chmod(0o444)
            script=f'''import sys
sys.path.insert(0,{str(ROOT)!r})
from experiments.study_eval import stage_questions
from pathlib import Path
p=Path({td!r})
stage_questions(p/'source',p/'out',{{'cohort_sha256':'source'}},[{{'dataset':'test'}}],b'new subset','tuning','split')
assert (p/'out/questions.jsonl').read_bytes()==b'new subset'
assert (p/'source/questions.jsonl').read_bytes()==b'original'
'''
            def unprivileged():
                os.setgroups([]);os.setgid(65534);os.setuid(65534)
            subprocess.run(['/usr/bin/python3','-c',script],check=True,
                           preexec_fn=unprivileged if os.getuid()==0 else None)
            self.assertEqual((source/'questions.jsonl').stat().st_mode&0o777,0o444)


if __name__=='__main__':unittest.main()
