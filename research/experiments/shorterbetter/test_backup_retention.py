import json
from pathlib import Path
import tempfile
import unittest
from .backup_retention import digest, prune


class RetentionTests(unittest.TestCase):
    def test_preserves_previous_until_successor_complete_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'output';old=out/'training/checkpoint-40';old.mkdir(parents=True)
            (old/'optimizer.pt').write_bytes(b'old checkpoint')
            new=out/'training/checkpoint-80';new.mkdir()
            names=['model.safetensors','optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(4)]
            for n in names:(new/n).write_bytes(b'valid')
            (new/'trainer_state.json').write_text(json.dumps({'global_step':80}))
            snapshot={'name':new.name,'hashes':{n:digest(new/n) for n in names}}
            (new/'optimizer.pt').write_bytes(b'incomplete')
            self.assertEqual(prune(out,snapshot),[]);self.assertTrue(old.exists())
            (new/'optimizer.pt').write_bytes(b'valid')
            self.assertEqual(prune(out,snapshot),['training/checkpoint-40'])
            self.assertFalse(old.exists());self.assertTrue(new.exists())

    def test_technical_compaction_requires_unchanged_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'output';p=out/'verification-resumed/checkpoint-8/model.safetensors';p.parent.mkdir(parents=True);p.write_bytes(b'x')
            (out/'technical-retention.json').write_text(json.dumps({'removed_artifact_sha256':{str(p.relative_to(out)):digest(p)}}))
            p.write_bytes(b'y')
            with self.assertRaises(RuntimeError):prune(out,None)
            self.assertTrue(p.exists())
