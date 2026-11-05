import unittest
from experiments.gpu_policy import verify_gpu_names, REQUIRED_GPU


class GpuPolicyTests(unittest.TestCase):
    def test_4090_is_accepted(self):
        self.assertEqual(verify_gpu_names(REQUIRED_GPU + '\n'), [REQUIRED_GPU])

    def test_other_or_missing_hardware_is_rejected(self):
        for value in ('', 'NVIDIA L4\n', 'NVIDIA RTX A6000\n',
                      REQUIRED_GPU + '\nNVIDIA L4\n'):
            with self.assertRaises(RuntimeError):
                verify_gpu_names(value)
