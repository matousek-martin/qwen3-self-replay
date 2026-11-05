"""Expose public entry points, never reference bodies or hidden test values."""
import ast

def interface_stub(reference, test):
    tree=ast.parse(reference)
    required={a.name for n in ast.walk(ast.parse(test)) if isinstance(n,ast.ImportFrom) and n.module=='solution' for a in n.names}
    definitions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
    if not required or '*' in required or required-definitions.keys():
        raise ValueError('unresolved_public_interface')
    def stub(node):
        if isinstance(node,ast.ClassDef):
            # Only expose class and method signatures, never attributes or bodies.
            methods=[stub(n) for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
            return ast.ClassDef(name=node.name,bases=[],keywords=[],body=methods or [ast.Pass()],decorator_list=[])
        node.body=[ast.Pass()]
        node.decorator_list=[d for d in node.decorator_list if isinstance(d,ast.Name) and d.id in ('staticmethod','classmethod')]
        node.returns=None
        for arg in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs,node.args.vararg,node.args.kwarg]:
            if arg is not None:arg.annotation=None
        return node
    return '\n\n'.join(ast.unparse(ast.fix_missing_locations(stub(definitions[name]))) for name in sorted(required))
