import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import teacher_full_eval_worker as worker


class FullTeacherTests(unittest.TestCase):
    def test_relative_cli_config_survives_child_evaluator_working_directory(self):
        class ReachedSmoke(Exception):
            pass
        with tempfile.TemporaryDirectory() as td:
            root = Path(td);stage = root/'data/teacher-full';stage.mkdir(parents=True)
            (stage/'evaluation.json').write_text('{}')
            smoke = stage/'smoke-student-seed17.json'
            smoke.write_text(json.dumps({'candidates': ['base', 'student-seed17']}))
            def process(command, log, timeout):
                if 'vllm_lora_smoke.py' not in str(command[1]):
                    return
                path = Path(command[command.index('--config')+1])
                self.assertTrue(path.is_absolute())
                # Real child process, with the same different cwd as run_process.
                subprocess.run([sys.executable, '-c',
                    'import json,sys; assert json.load(open(sys.argv[1]))["candidates"][1]=="student-seed17"',
                    str(path)], cwd=worker.study.REPO, check=True)
                raise ReachedSmoke()
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(sys, 'argv', ['worker', '--config', 'data/teacher-full/evaluation.json',
                         '--seed', '17', '--out', str(root/'output/full-evaluation')]), \
                     patch.object(worker, 'inputs', return_value=({}, root, [], {'student-seed17':stage})), \
                     patch.object(worker.study, 'preflight_nltk'), \
                     patch.object(worker.study, 'write_json'), \
                     patch.object(worker.subprocess, 'check_output', return_value='GPU test'), \
                     patch.object(worker.study, 'run_process', side_effect=process):
                    with self.assertRaises(ReachedSmoke):
                        worker.main()
            finally:
                os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
