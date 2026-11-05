"""Checkpoint staging must preserve the configuration that gives weights meaning."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import study_evaluation_campaign as campaign


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.data = root / 'data'
        self.study = root / 'study'
        (self.data / 'adapters').mkdir(parents=True)
        (self.study / 'staging').mkdir(parents=True)
        (self.data / 'adapters.json').write_text('{}')
        self.output = root / 'output'
        self.source = self.output / 'training/grpo/adapter'
        self.source.mkdir(parents=True)
        (self.source.parent / 'status.json').write_text(json.dumps(
            {'status': 'completed', 'completed_groups': 128}))
        (self.source / 'adapter_model.safetensors').write_bytes(b'unchanged test weights')
        (self.source / 'adapter_config.json').write_text(json.dumps({'r': 16, 'lora_alpha': 32}))
        self.job = {'arm': 'grpo', 'job': 'training-job', 'output': str(self.output)}
        self.addCleanup(patch.stopall)
        patch.object(campaign, 'DATA', self.data).start()
        patch.object(campaign, 'STUDY', self.study).start()

    def test_config_digest_survives_idempotent_staging(self):
        campaign.stage(self.job)
        campaign.stage(self.job)
        registration = json.loads((self.data / 'adapters.json').read_text())['grpo']
        config = self.source / 'adapter_config.json'
        self.assertEqual(registration['config_sha256'], hashlib.sha256(config.read_bytes()).hexdigest())
        self.assertEqual(config.read_bytes(), (self.data / 'adapters/grpo/adapter_config.json').read_bytes())

    def test_changed_scaling_rejected_even_when_weights_match(self):
        campaign.stage(self.job)
        config = self.data / 'adapters/grpo/adapter_config.json'
        config.write_text(json.dumps({'r': 16, 'lora_alpha': 16}))
        with self.assertRaisesRegex(RuntimeError, 'configuration differs'):
            campaign.stage(self.job)

    def test_partial_training_cannot_be_staged_as_final(self):
        (self.source.parent / 'status.json').write_text(json.dumps(
            {'status': 'completed', 'completed_groups': 64}))
        with self.assertRaisesRegex(RuntimeError, 'partial training'):
            campaign.stage(self.job)
        self.assertFalse((self.data / 'adapters/grpo').exists())


if __name__ == '__main__':
    unittest.main()
