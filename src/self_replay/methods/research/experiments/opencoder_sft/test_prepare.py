import unittest
from .prepare import literals, code_key, candidate, balanced


class PreparationTests(unittest.TestCase):
    def test_literal_contract(self):
        self.assertEqual(literals('assert f([1,2], "x") == {"a":3}', 'f'), ([[1,2],'x'],{'a':3}))

    def test_never_execute_assertion_expressions(self):
        for s in ['assert f(__import__("os").system("true")) == 0',
                  'assert f(1) != 2', 'assert f(1) == 2\nassert f(2) == 3',
                  'assert f(x=1) == 2', 'assert g(1) == 2', 'assert f((1,2)) == 3']:
            with self.subTest(s=s), self.assertRaises((ValueError,SyntaxError)): literals(s,'f')

    def test_renamed_programs_share_family(self):
        self.assertEqual(code_key('def f(a): return a+1'),code_key('def g(b): return b+1'))
        self.assertNotEqual(code_key('def f(a): return a+1'),code_key('def f(a): return a+2'))

    def test_all_source_tests_required(self):
        row={'seq_id':1,'entry_point':'f','instruction':'Return x', 'code':'def f(x): return x',
             'testcase':['assert f(1) == 1','assert f(2) == 2','assert f(3) != 4']}
        with self.assertRaises(ValueError):candidate(row)

    def test_balanced_prefix_is_nested(self):
        rows=[{'id':str(i),'category':str(i%3)} for i in range(24)]
        self.assertEqual(balanced(rows,6),balanced(rows,18)[:6])


if __name__=='__main__':unittest.main()
