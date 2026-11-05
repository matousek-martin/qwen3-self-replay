import ast
from pathlib import Path
from types import SimpleNamespace as NS
from typing import cast
import unittest

from experiments import vllm_compat as compat


class CompatibilityTests(unittest.TestCase):
    def source(self):
        return (Path(__file__).parent / 'fixtures/vllm-0.11.0-output_processor.py').read_text()

    def request_state_class(self, source):
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'RequestState')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        namespace = {
            'length_from_prompt_token_ids_or_embeds': lambda ids, embeds: len(ids),
            'LogprobsProcessor': NS(from_new_request=lambda **kw: NS(prompt_logprobs=None)),
            'IncrementalDetokenizer': NS(from_new_request=lambda **kw: None),
            'RequestOutputKind': NS(DELTA='delta'), 'PoolingOutput': type('PoolingOutput', (), {}),
            'CompletionOutput': type('CompletionOutput', (), {}), 'cast': cast,
            'RequestOutput': lambda **kw: NS(lora_request=kw.pop('lora_request', None), **kw),
        }
        exec(compile(ast.fix_missing_locations(module), '<actual-vllm-request-state>', 'exec'), namespace)
        return namespace['RequestState']

    def output(self, cls, lora):
        request = NS(request_id='request', lora_request=lora, prompt_token_ids=[1], prompt_embeds=None,
                     arrival_time=0, sampling_params=NS(detokenize=False, output_kind='final', max_tokens=1,
                                                       top_p=1, n=1, temperature=0))
        state = cls.from_new_request(None, request, None, None, 0, None, False)
        completion = NS(token_ids=[10, 20])
        return state._new_request_output('request', [completion], True)

    def test_reproduces_original_omission_and_preserves_actual_request_identity(self):
        original = self.request_state_class(self.source())
        repaired = self.request_state_class(compat.patch_source(self.source()))
        for identity in (1, 2):
            adapter = NS(name='adapter-' + str(identity), lora_int_id=identity)
            self.assertIsNone(self.output(original, adapter).lora_request)
            result = self.output(repaired, adapter)
            self.assertIs(result.lora_request, adapter)
            self.assertEqual(result.outputs[0].token_ids, [10, 20])
        self.assertIsNone(self.output(repaired, None).lora_request)

    def test_patch_is_pinned_and_idempotent(self):
        source = compat.patch_source(self.source())
        self.assertEqual(compat.digest(source), compat.PATCHED_SHA)
        self.assertEqual(compat.patch_source(source), source)
        with self.assertRaises(ValueError):
            compat.patch_source(self.source() + '# unexpected change\n')


if __name__ == '__main__':
    unittest.main()
