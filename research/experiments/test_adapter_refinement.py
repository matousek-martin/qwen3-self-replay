import json
from pathlib import Path
import tempfile
import unittest
import shutil
from datetime import datetime, timezone

from adapter_refinement import MODEL, REVISION, prepare, verify, verify_registration, digest, registered_specs


class RefinementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = self.root / 'parent'
        self.parent.mkdir()
        self.config = {'peft_type': 'LORA', 'bias': 'none', 'lora_alpha': 32, 'r': 16,
                       'base_model_name_or_path': MODEL, 'revision': REVISION}
        self.write_config()
        (self.parent / 'adapter_model.safetensors').write_bytes(b'original weights')
        (self.parent / 'state.json').write_text(json.dumps({'step': 32}))
        (self.parent / 'tokenizer.json').write_text('{}')

    def write_config(self):
        (self.parent / 'adapter_config.json').write_text(json.dumps(self.config))

    def test_only_alpha_changes_and_parent_is_preserved(self):
        original = (self.parent / 'adapter_config.json').read_bytes()
        for scale, alpha in [(0.5, 16), (0.75, 24)]:
            destination = self.root / str(alpha)
            manifest = prepare(self.parent, destination, 'variant', scale)
            verify(self.parent, destination / 'adapter', manifest)
            config = json.loads((destination / 'adapter/adapter_config.json').read_text())
            self.assertEqual(config, dict(self.config, lora_alpha=alpha))
            self.assertEqual((self.parent / 'adapter_config.json').read_bytes(), original)

    def test_changed_weights_cannot_pass_with_rewritten_manifest(self):
        destination = self.root / 'variant'
        manifest = prepare(self.parent, destination, 'variant', .5)
        (destination / 'adapter/adapter_model.safetensors').write_bytes(b'changed weights')
        from adapter_refinement import adapter_hashes
        manifest['adapter_files_sha256'] = adapter_hashes(destination / 'adapter')
        with self.assertRaisesRegex(ValueError, 'changed weights'):
            verify(self.parent, destination / 'adapter', manifest)

    def test_additional_trainable_parameters_are_rejected(self):
        self.config['modules_to_save'] = ['lm_head']
        self.write_config()
        with self.assertRaisesRegex(ValueError, 'additional trainable'):
            prepare(self.parent, self.root / 'variant', 'variant', .5)

    def test_existing_refinement_is_not_overwritten(self):
        destination = self.root / 'variant'
        prepare(self.parent, destination, 'variant', .5)
        with self.assertRaises(FileExistsError):
            prepare(self.parent, destination, 'variant', .75)

    def registered_fixture(self):
        name = 'dpo-32-scale050'
        data = self.root / 'lab/experiments/data/study/adapters'
        parent = data / 'dpo-32'
        shutil.copytree(self.parent, parent)
        destination = self.root / 'refinement'
        manifest = prepare(parent, destination, name, .5)
        shutil.copytree(destination / 'adapter', data / name)
        registration = {'source': str(destination / 'adapter'), 'parent_candidate': 'dpo-32',
                        'refinement_manifest': str(destination / 'manifest.json'),
                        'checkpoint_sha256': manifest['adapter_files_sha256']['adapter_model.safetensors'],
                        'config_sha256': manifest['adapter_files_sha256']['adapter_config.json']}
        parent_registration = {'source': str(self.parent),
                               'checkpoint_sha256': manifest['parent_files_sha256']['adapter_model.safetensors'],
                               'config_sha256': manifest['parent_files_sha256']['adapter_config.json']}
        spec = {'parent_candidate': 'dpo-32', 'strength_multiplier': .5,
                'registered_at': datetime.now(timezone.utc).isoformat(),
                'manifest': str(destination / 'manifest.json'),
                'manifest_sha256': digest(destination / 'manifest.json')}
        return name, registration, parent_registration, spec

    def test_registration_proves_parent_and_both_copies(self):
        args = self.registered_fixture()
        proof = verify_registration(self.root, *args)
        self.assertEqual(proof['strength_multiplier'], .5)
        staged = self.root / 'lab/experiments/data/study/adapters' / args[0]
        (staged / 'tokenizer.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'changed after preparation'):
            verify_registration(self.root, *args)

    def test_registration_timestamp_cannot_precede_preparation(self):
        name, registration, parent, spec = self.registered_fixture()
        spec['registered_at'] = '2020-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'Preparation occurred after'):
            verify_registration(self.root, name, registration, parent, spec)

    def test_omitting_one_promised_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly the two'):
            registered_specs({'additional_refinement_candidates': ['dpo-32-scale050'],
                              'additional_refinement_specs': {}})


if __name__ == '__main__':
    unittest.main()
