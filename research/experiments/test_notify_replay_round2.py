import copy
import unittest
from experiments.notify_replay_round2 import completion_ready


class CompletionNotificationTests(unittest.TestCase):
    def fixtures(self):
        plan = {'trials': [{'name': str(i)} for i in range(9)]}
        campaign = {'state': 'development_complete', 'trials': {
            str(i): {'state': 'complete', 'job': 'replay2-' + str(i)} for i in range(9)}}
        ledger = {'jobs': {'replay2-' + str(i): {'state': 'complete', 'released_at': 1,
                   'backup_verified': True, 'gpu': 'NVIDIA GeForce RTX 4090'} for i in range(9)}}
        return plan, campaign, ledger

    def test_all_nine_verified_and_released_are_required(self):
        plan, campaign, ledger = self.fixtures()
        self.assertTrue(completion_ready(plan, campaign, ledger))
        for field, value in [('released_at', None), ('backup_verified', False),
                             ('state', 'failed'), ('gpu', 'NVIDIA L4')]:
            bad = copy.deepcopy(ledger)
            bad['jobs']['replay2-0'][field] = value
            self.assertFalse(completion_ready(plan, campaign, bad))

    def test_partial_campaign_or_orphan_rental_cannot_notify_completion(self):
        plan, campaign, ledger = self.fixtures()
        campaign['state'] = 'needs_inspection'
        self.assertFalse(completion_ready(plan, campaign, ledger))
        campaign['state'] = 'development_complete'
        ledger['jobs']['replay2-orphan'] = {'state': 'running'}
        self.assertFalse(completion_ready(plan, campaign, ledger))
        del ledger['jobs']['replay2-orphan']
        del campaign['trials']['0']
        self.assertFalse(completion_ready(plan, campaign, ledger))
