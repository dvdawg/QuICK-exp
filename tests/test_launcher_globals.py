import ast
import builtins
from pathlib import Path


LAUNCHERS = sorted(
    (Path(__file__).resolve().parents[1] / "experiments").glob("*.py")
)
IMPLICIT = {"__file__", "__name__", "__doc__"}


def _argument_names(arguments):
    names = {
        argument.arg
        for argument in (
            arguments.posonlyargs + arguments.args + arguments.kwonlyargs
        )
    }
    if arguments.vararg:
        names.add(arguments.vararg.arg)
    if arguments.kwarg:
        names.add(arguments.kwarg.arg)
    return names


class _Scope(ast.NodeVisitor):
    def __init__(self):
        self.loads = set()
        self.stores = set()

    def visit_Name(self, node):
        target = self.loads if isinstance(node.ctx, ast.Load) else self.stores
        target.add(node.id)

    def visit_Lambda(self, node):
        self.stores.update(_argument_names(node.args))
        self.visit(node.body)

    def _visit_comprehension(self, node):
        for generator in node.generators:
            self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension
    visit_DictComp = _visit_comprehension


def _assigned_names(target):
    return {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _module_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(_assigned_names(target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((item.asname or item.name).split(".")[0] for item in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def test_every_launcher_function_references_only_defined_globals():
    problems = []
    for path in LAUNCHERS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = _module_names(tree) | set(dir(builtins)) | IMPLICIT
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            scope = _Scope()
            for statement in node.body:
                scope.visit(statement)
            arguments = _argument_names(node.args)
            undefined = scope.loads - defined - scope.stores - arguments
            if undefined:
                problems.append(
                    f"{path.name}:{node.name}(): {sorted(undefined)}"
                )
    assert not problems, "\n".join(problems)
