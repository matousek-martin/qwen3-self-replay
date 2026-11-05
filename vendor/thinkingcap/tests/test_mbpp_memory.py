"""The reference-output cache must not accumulate across coding tasks."""
import unittest
import weakref

from utils.eval.mbpp_worker import oracle_for


class OracleMemory(unittest.TestCase):
    def test_reuses_samples_but_frees_previous_task_before_allocating(self):
        class Output:
            pass

        problems = {task: dict(prompt='', canonical_solution=task,
                              base_input=[], plus_input=[], entry_point='f')
                    for task in ('a', 'b')}
        references, calls, cache = [], [], {}

        def execute(code, inputs, entry_point, **kwargs):
            if code == 'b':
                self.assertTrue(all(ref() is None for ref in references))
            value = Output()
            if code == 'a':
                references.append(weakref.ref(value))
            calls.append(code)
            return value, [0.01]

        oracle_for('a', problems, cache, execute, set())
        oracle_for('a', problems, cache, execute, set())
        self.assertEqual(calls, ['a', 'a'])  # base and plus, once per task
        oracle_for('b', problems, cache, execute, set())
        self.assertEqual(calls, ['a', 'a', 'b', 'b'])


if __name__ == '__main__':
    unittest.main()
