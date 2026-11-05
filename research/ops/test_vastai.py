"""Offline provider-contract and canonical lifecycle regression tests."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ops import vastai
from experiments import cloud
from experiments.shorterbetter.budget import ticket


class VastTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        p = patch.object(cloud, 'STATE', self.state)
        p.start()
        self.addCleanup(p.stop)

    def authorize(self):
        args = SimpleNamespace(name='shorterbetter-scaled-vast-test',
            entry='experiments/shorterbetter/run-scaled.sh', provider='vast',
            gpu='NVIDIA A100-SXM4-80GB', gpu_count=4, hardware_scope='shorterbetter-scaled-v1',
            max_cost_usd=100, hours=1, max_runtime_hours=24,
            vast_offer_id=123, vast_disk_gb=100)
        auth = dict(authorized=True, job_name=args.name, scope='shorterbetter_vast_bounded_v1',
            provider='vast', gpu=args.gpu, gpu_count=4, offer_id=123, disk_gb=100,
            max_cost_usd=100, max_runtime_hours=24, max_gpu_rate=1.2,
            source_user_instruction='Test fixture only')
        cloud.write(self.state/'authorizations'/f'{args.name}.json', auth)
        return args, auth

    def allocate(self, args, create=None):
        q = {'hourly_ceiling': 4.2, 'offer_id': 123, 'disk_gb': 100}
        if create is None:
            create = lambda quote, name: {'id': 'vast:456', 'name': name, 'cost': 4.2}
        with patch.object(vastai, 'api', return_value={'credit': 100, 'can_pay': True}), \
             patch.object(vastai, 'ensure_ssh_key'), patch.object(vastai, 'quote', return_value=q), \
             patch.object(vastai, 'create', side_effect=create), patch.object(cloud, 'runpod_api') as rp:
            result = cloud.allocate(args)
            rp.assert_not_called()
            return result

    def job(self, name):
        return json.loads((self.state/'ledger.json').read_text())['jobs'][name]

    def test_pagination_requires_complete_unique_instances(self):
        pages = [dict(instances=[{'id': 1, 'label': 'a'}], next_token='b', total_instances=2),
                 dict(instances=[{'id': 2, 'label': 'b'}], next_token=None, total_instances=2)]
        with patch.object(vastai, 'api', side_effect=pages) as request:
            self.assertEqual([x['id'] for x in vastai.list_pods()], ['vast:1', 'vast:2'])
            self.assertEqual(request.call_args.kwargs['params'], {'next_token': 'b'})
        for page in [dict(instances=[], total_instances=0),
                     dict(instances=[], next_token=None, total_instances=1)]:
            with patch.object(vastai, 'api', return_value=page):
                with self.assertRaises(RuntimeError): vastai.list_pods()

    def test_stopped_instance_is_not_released(self):
        with patch.object(vastai, 'list_pods', return_value=[vastai.normalize({'id': 7, 'actual_status': 'stopped'})]):
            self.assertEqual(vastai.pod_api('pods/vast:7')['status'], 'STOPPED')

    def test_ssh_setup_recognizes_provider_public_key_field(self):
        key = self.state/'public.pub'
        key.write_text('ssh-ed25519 TESTPUBLIC project\n')
        with patch.object(vastai, 'SSH_KEY', key), \
             patch.object(vastai, 'api', return_value=[{'public_key': 'ssh-ed25519 TESTPUBLIC other-comment'}]) as api:
            self.assertEqual(vastai.ensure_ssh_key(), {'ssh_key_registered': True, 'created': False})
            api.assert_called_once_with('v0/ssh/')

    def test_provider_identity_routing(self):
        with patch.object(vastai, 'pod_api', return_value={}) as va, patch.object(cloud, 'runpod_api') as rp:
            cloud.api('pods/vast:42', 'DELETE')
            va.assert_called_once_with('pods/vast:42', 'DELETE', None)
            rp.assert_not_called()
            cloud.api('pods/runpod-id')
            rp.assert_called_once()

    def test_quote_rejects_wrong_hardware_short_lifetime_and_missing_prices(self):
        offer = dict(id=123, gpu_name='A100 SXM4', num_gpus=4, gpu_ram=81920,
            disk_space=150, duration=100000, verification='verified', rentable=True, rented=False,
            reliability2=.999, cuda_max_good=13., dph_total=4., storage_cost=.2,
            inet_up_cost=0., inet_down_cost=0.)
        with patch.object(vastai, 'api', return_value={'offers': [offer]}):
            q = vastai.quote('NVIDIA A100-SXM4-80GB', 4, min_duration=90000)
            self.assertGreater(q['hourly_ceiling'], 4)
            self.assertIsNone(vastai.quote('NVIDIA A100-SXM4-80GB', 4, min_duration=110000))
        with patch.object(vastai, 'api', return_value={'offers': [offer]}) as api:
            self.assertEqual(vastai.quote('NVIDIA A100-SXM4-80GB', 4, offer_id=123)['offer_id'], 123)
            self.assertNotIn('id', json.loads(api.call_args.kwargs['params']['q']))
            self.assertEqual(json.loads(api.call_args.kwargs['params']['q'])['ask_contract_id'],{'eq':123})
            self.assertEqual(json.loads(api.call_args.kwargs['params']['q'])['limit'],1000)
            self.assertIsNone(vastai.quote('NVIDIA A100-SXM4-80GB', 4, offer_id=999))
        with patch.object(vastai, 'api', return_value={'offers':[offer,dict(offer,id=456,dph_total=4.5)]}):
            self.assertEqual(vastai.quote('NVIDIA A100-SXM4-80GB',4,exclude_offer_ids=[123])['offer_id'],456)
            self.assertIsNone(vastai.quote('NVIDIA A100-SXM4-80GB',4,offer_id=123,exclude_offer_ids=[123]))
        with patch.object(vastai,'api',return_value={'offers':[dict(offer,machine_id=1),dict(offer,id=456,machine_id=2,dph_total=4.5)]}):
            self.assertEqual(vastai.quote('NVIDIA A100-SXM4-80GB',4,exclude_machine_ids=[1])['offer_id'],456)
        with patch.object(vastai,'api',return_value={'offers':[dict(offer,geolocation='Jiangsu, CN'),dict(offer,id=456,geolocation='Poland, PL',dph_total=4.5)]}):
            self.assertEqual(vastai.quote('NVIDIA A100-SXM4-80GB',4,allowed_countries=['PL'])['offer_id'],456)
            self.assertIsNone(vastai.quote('NVIDIA A100-SXM4-80GB',4,allowed_countries=['DE']))
        for changes in ({'gpu_ram': 40960}, {'num_gpus': 2}, {'disk_space': 70}, {'rented': True}):
            with patch.object(vastai, 'api', return_value={'offers': [dict(offer, **changes)]}):
                self.assertIsNone(vastai.quote('NVIDIA A100-SXM4-80GB', 4))
        del offer['storage_cost']
        with patch.object(vastai, 'api', return_value={'offers': [offer]}):
            with self.assertRaises(KeyError): vastai.quote('NVIDIA A100-SXM4-80GB', 4)

    def test_create_returns_id_before_another_provider_call(self):
        with patch.object(vastai, 'api', return_value={'success': True, 'new_contract': 456}) as request:
            pod = vastai.create({'offer_id': 123, 'disk_gb': 100, 'hourly_ceiling': 4.2}, 'nonce')
            self.assertEqual(pod['id'], 'vast:456')
            request.assert_called_once()
            self.assertTrue(request.call_args.args[2]['cancel_unavail'])

    def test_named_vast_authorization_reserves_and_retains_provider(self):
        args, _ = self.authorize()
        self.allocate(args)
        job = self.job(args.name)
        self.assertEqual(job['provider'], 'vast')
        self.assertEqual(job['reserved_usd'], 100)
        self.assertEqual(job['pod_id'], 'vast:456')
        self.assertTrue(ticket(self.state, args.name, job))
        self.assertFalse(ticket(self.state, args.name, dict(job, provider='runpod')))
        self.assertFalse(ticket(self.state, args.name, dict(job, offer_id=999)))

    def test_missing_authorization_cannot_rent(self):
        args, _ = self.authorize()
        args.name += '-other'
        with patch.object(vastai, 'api') as api, patch.object(vastai, 'create') as create:
            with self.assertRaisesRegex(ValueError, 'named bounded'): cloud.allocate(args)
            api.assert_not_called()
            create.assert_not_called()

    def test_insufficient_credit_cannot_rent(self):
        args, _ = self.authorize()
        with patch.object(vastai, 'api', return_value={'credit': 90, 'can_pay': True}), \
             patch.object(vastai, 'create') as create:
            with self.assertRaisesRegex(RuntimeError, 'credit'): cloud.allocate(args)
            create.assert_not_called()

    def incremental_authorize(self):
        args, auth = self.authorize()
        auth.update(full_training_authorized=True, rollout_batches=300,
                    incremental_funding={'minimum_start_credit_usd':60, 'warning_credit_usd':30,
                                         'stop_credit_usd':12, 'release_credit_usd':3})
        cloud.write(self.state/'authorizations'/f'{args.name}.json', auth)
        return args, auth

    def test_incremental_credit_requires_named_full_run_and_keeps_entire_cap(self):
        args, auth = self.incremental_authorize()
        with patch.object(vastai, 'api', return_value={'credit':70, 'can_pay':True}), \
             patch.object(vastai, 'ensure_ssh_key'), \
             patch.object(vastai, 'quote', return_value={'hourly_ceiling':4.2}), \
             patch.object(vastai, 'create', side_effect=lambda quote,name: {'id':'vast:456','name':name,'cost':4.2}):
            cloud.allocate(args)
        job = self.job(args.name)
        self.assertEqual(job['reserved_usd'], 100)
        self.assertEqual(job['initial_account_credit'], 70)
        auth['full_training_authorized'] = False
        cloud.write(self.state/'authorizations'/f'{args.name}.json', auth)
        self.assertIsNone(ticket(self.state,args.name,job))

    def test_low_credit_saves_before_emergency_release_even_if_ssh_fails(self):
        args, _ = self.incremental_authorize()
        self.allocate(args)
        with patch.object(vastai,'api',return_value={'credit':11}), \
             patch.object(cloud,'api',return_value={}), patch.object(cloud,'remote') as remote:
            self.assertFalse(cloud.vast_spending_exhausted(args.name,self.job(args.name)))
            self.assertIn('stop-request.json',remote.call_args.args[1])
        self.assertTrue(self.job(args.name)['credit_stop_requested_at'])
        with patch.object(vastai,'api',return_value={'credit':2}), \
             patch.object(cloud,'remote',side_effect=RuntimeError('ssh down')) as remote:
            self.assertTrue(cloud.vast_spending_exhausted(args.name,self.job(args.name)))
            remote.assert_not_called()

    def test_ambiguous_create_keeps_reservation_and_guard_reconciles_vast(self):
        args, _ = self.authorize()
        calls = []
        def fail(q, name):
            calls.append(name)
            raise RuntimeError('Vast transport failure')
        with self.assertRaises(RuntimeError): self.allocate(args, fail)
        job = self.job(args.name)
        self.assertEqual(len(calls), 1)
        self.assertEqual(job['reserved_usd'], 100)
        self.assertFalse(job.get('released_at'))
        self.assertFalse(job['safely_no_pod'])
        pod = {'id': 'vast:456', 'name': job['pod_name']}
        with patch.object(vastai, 'list_pods', return_value=[pod]), \
             patch.object(cloud, 'release') as release, patch.object(cloud, 'runpod_api') as rp:
            cloud.guard_job(args.name, job)
            release.assert_called_once_with(args.name, 'abandoned_create_reconciled')
            rp.assert_not_called()

    def test_delete_needs_absence_confirmation(self):
        args, _ = self.authorize()
        self.allocate(args)
        with patch.object(vastai, 'api', return_value={'success': True}), \
             patch.object(vastai, 'list_pods', return_value=[{'id': 'vast:456', 'status': 'STOPPED'}]):
            cloud.release(args.name, 'finished')
            self.assertFalse(self.job(args.name).get('released_at'))
        with patch.object(vastai, 'list_pods', return_value=[]):
            self.assertTrue(cloud.confirm_release(args.name))

    def test_account_spend_detects_transfer_and_does_not_reset_on_topup(self):
        args, _ = self.authorize()
        self.allocate(args)
        for credit, stop in [(50, False), (150, False), (100, True)]:
            with patch.object(vastai, 'api', return_value={'credit': credit}):
                self.assertEqual(cloud.vast_spending_exhausted(args.name, self.job(args.name)), stop)
        self.assertEqual(self.job(args.name)['account_spend_observed'], 100)

    def test_concurrent_evaluation_uses_joint_account_cap_but_keeps_own_budget(self):
        args, auth = self.authorize()
        self.allocate(args)
        now = time.time()
        campaign = self.state.parent/'shorterbetter-paired'/self.state.name/'campaign.json'
        self.addCleanup(lambda: __import__('shutil').rmtree(campaign.parent, ignore_errors=True))
        cloud.write(campaign, dict(authorized=True, cap_usd=105, evaluation_reserve_usd=10,
            arms={'control': {'job_name': 'training-a'}, 'accuracy': {'job_name': 'training-b'}},
            early_evaluations=[{'job_name': args.name}]))
        auth.update(max_cost_usd=8, evaluation_only=True, full_training_authorized=False,
                    paired_evaluation_account_guard=True, paired_campaign_path=str(campaign))
        cloud.write(self.state/'authorizations'/f'{args.name}.json', auth)
        with cloud.ledger() as state:
            state['jobs'][args.name].update(max_cost_usd=8, created_at=now,
                initial_account_credit=80, last_account_credit=80)
            state['jobs']['training-a'] = dict(released_at=now-10, charged_estimate_usd=38)
            state['jobs']['training-b'] = dict(created_at=now-3600, budget_rate=5)
        # $10 account usage must not exhaust an $8 evaluator when training shares it.
        with patch.object(vastai, 'api', return_value={'credit': 70}):
            self.assertFalse(cloud.vast_spending_exhausted(args.name, self.job(args.name)))
        self.assertEqual(self.job(args.name)['account_spending_limit_usd'], 62)
        self.assertEqual(self.job(args.name)['max_cost_usd'], 8)
        from experiments.shorterbetter.budget import evaluation_account_limit
        with cloud.ledger() as state:
            state['jobs']['training-a']['charged_estimate_usd'] = 106
        self.assertEqual(evaluation_account_limit(self.state,args.name,self.job(args.name)),0)
        with cloud.ledger() as state:
            state['jobs']['training-a']['charged_estimate_usd'] = 38
        # A deposit cannot erase the joint cap; accumulated spending still trips it.
        for credit, stop in [(100, False), (48, True)]:
            with patch.object(vastai, 'api', return_value={'credit': credit}):
                self.assertEqual(cloud.vast_spending_exhausted(args.name, self.job(args.name)), stop)
        # Missing campaign membership reverts to the strict per-job account guard.
        value=json.loads(campaign.read_text());value['early_evaluations']=[];cloud.write(campaign,value)
        self.assertEqual(evaluation_account_limit(self.state,args.name,self.job(args.name)),8)

    def test_api_failure_does_not_prevent_deadline_cleanup(self):
        args, _ = self.authorize()
        self.allocate(args)
        with patch.object(vastai, 'api', side_effect=RuntimeError('network')):
            self.assertFalse(cloud.vast_spending_exhausted(args.name, self.job(args.name)))
        self.assertIn('account_check_error', self.job(args.name))


if __name__ == '__main__':
    unittest.main()


class ContainerStartupQuarantineTests(unittest.TestCase):
 def test_cdi_failure_quarantines_only_its_machine(self):
  from experiments import cloud
  import time
  gpu='NVIDIA GeForce RTX 4090'
  jobs={'failed':{'provider':'vast','gpu':gpu,'released_at':time.time(),'created_at':time.time()-240,
   'release_reason':'provider_startup_failure','error':'unresolved GPU CDI device','provider_quote':{'offer_id':1,'offer':{'machine_id':28421}}}}
  self.assertTrue(cloud.unhealthy_placement(jobs,gpu,'vast',{'offer_id':2,'offer':{'machine_id':28421}}))
  self.assertFalse(cloud.unhealthy_placement(jobs,gpu,'vast',{'offer_id':3,'offer':{'machine_id':148193}}))
  self.assertFalse(cloud.unhealthy_placement(jobs,gpu,'vast'))

class DirectSSHTests(unittest.TestCase):
 def test_frontier_uses_relay_and_preserves_published_endpoint(self):
  row={'id':1,'label':'thinkingcap-pilot-coding-frontier-apps-fixture','actual_status':'running','image_runtype':'ssh_direc ssh_proxy',
       'ssh_host':'ssh.vast.ai','ssh_port':1234,'public_ipaddr':'192.0.2.1','ports':{'22/tcp':[{'HostPort':'2345'}]}}
  ssh=vastai.normalize(row)['ssh']
  self.assertEqual(ssh['direct']['host'],'ssh.vast.ai')
  self.assertEqual(ssh['published_direct']['host'],'192.0.2.1')
  self.assertEqual(ssh['transport'],'vast_ssh_relay')
  row['label']='unrelated-job'
  self.assertEqual(vastai.normalize(row)['ssh']['direct']['host'],'192.0.2.1')
 def test_published_ssh_selected_without_changing_proxy_only_instances(self):
  row={'id':1,'actual_status':'running','ssh_host':'ssh.vast.ai','ssh_port':1234,'public_ipaddr':'192.0.2.1'}
  self.assertEqual(vastai.normalize(row)['ssh']['direct']['host'],'ssh.vast.ai')
  row['image_runtype']='ssh_direc ssh_proxy'
  self.assertEqual(vastai.normalize(row)['ssh'],{})
  row['ports']={'22/tcp':[{'HostPort':'2345'}]}
  self.assertEqual(vastai.normalize(row)['ssh']['direct'],{'host':'192.0.2.1','port':2345,'username':'root'})
 def test_direct_mode_is_opt_in_and_keeps_proxy_available(self):
  with patch.object(vastai,'api',return_value={'success':True,'new_contract':1}) as api:
   vastai.create({'offer_id':1,'disk_gb':60,'hourly_ceiling':.5},'test',direct_ssh=True)
   self.assertEqual(api.call_args.args[2]['runtype'],'ssh_direc ssh_proxy')
