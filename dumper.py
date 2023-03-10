import ast
import hashlib
import inspect
import os
import string
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence
from types import FunctionType, ModuleType
from typing import Any, Optional

from numba.core.bytecode import ByteCode


class Block(Sequence):
    def __init__(self, *args):
        self.args = list(args)

    def __str__(self) -> str:
        if len(self.args) == 0:
            return ""
        final = "\n".join(map(str, self.args))
        if self.args[-1] == Line.new_line():
            return final.removesuffix("\n")
        return final

    def to_string(self):
        return str(self)

    def __getitem__(self, idx):
        return self.args[idx]

    def __len__(self):
        return len(self.args)

    def __add__(self, other):
        assert isinstance(other, (Line, Block))
        if isinstance(other, Line):
            self.args.append(other)
        elif isinstance(other, Block):
            self.args.extend(other.args)
        return self


class Line(str):
    """ """

    @classmethod
    def new_line(cls):
        return cls("\n")

    def __init__(self, line: str) -> None:
        self._line = line

    def __str__(self) -> str:
        return self._line

    def __add__(self, other):
        return f"{str(self)}\n{str(other)}"


class Comment(Line):
    @classmethod
    def new_line(cls):
        return cls("\n")

    def __init__(self, line: str) -> None:
        super().__init__(line)

    def __str__(self) -> str:
        return f"# {self._line}"


class Property:
    FUNCTION_NAME = "function_name"
    CO_CODE = "co_code"
    SIGNATURES = "signatures"
    IMPORTS = "imports"


class StopPipelineIteration(Exception):
    """ """


class Pipeline:
    def __init__(self, *steps) -> None:
        if not isinstance(steps, Iterable):
            steps = (steps,)
        self._steps = tuple(steps)
        self.properties: dict[str, Any] = {}

    def abort(self):
        raise StopPipelineIteration()

    def set_property(self, key, value, /):
        self.properties[key] = value
        return self

    def get_property(self, key, /) -> Any:
        return self.properties.get(key, None)

    def run_step(self, step: "Step", source: str) -> str:
        if issubclass(step.__class__, AnalysisStep):
            step.apply(self, source)
        else:
            source = step.apply(self, source)
        return source

    def _run(self, source: str) -> None:
        for step in self._steps:
            source = self.run_step(step, source)

    def run(self, source: str) -> None:
        try:
            self._run(source)
        except StopPipelineIteration:
            return


class Step(ABC):
    """
    Base class for all steps
    """


class TransformationStep(Step):
    """
    A transformation step modifies the source code
    """

    @abstractmethod
    def apply(self, pipeline: Pipeline, source: str) -> str:
        pass


class AnalysisStep(Step):
    """
    Analysis step only performs an analysis but keep the source code intact
    """

    @abstractmethod
    def apply(self, pipeline: Pipeline, source: str) -> None:
        pass


class AddSignature(AnalysisStep):
    def __init__(self, sig) -> None:
        if not isinstance(sig, (tuple, list)):
            sig = (sig,)
        self.signatures = tuple(sig)

    def apply(self, pipeline, source):
        set_ = set(self.signatures)
        sigs = pipeline.get_property(Property.SIGNATURES)
        if sigs is not None:
            set_.update(sigs)
        pipeline.set_property(Property.SIGNATURES, set_)


class Compile(AnalysisStep):
    def __init__(self, *, filename="<unknown>", mode="exec"):
        self.filename = filename
        self.mode = mode

    def apply(self, pipeline: Pipeline, source: str) -> None:
        code = compile(source, self.filename, self.mode)
        pipeline.set_property(Property.CO_CODE, code)
        pipeline.set_property(Property.FUNCTION_NAME, code.co_names[-1])


class AddMissingGlobalsVariables(TransformationStep):
    """ """

    def __init__(self, globals) -> None:
        self.globals = globals

    class _FunctionIdentity:
        """Mimics a Numba FunctionIdentity class. Code below if the minimum
        required by `_compute_used_globals` to work
        """

        # mimics a Numba FunctionIdentity
        def __init__(self, code, globals):
            self.code = code
            self.globals = globals

        @property
        def __globals__(self):
            return self.globals

    def get_unused_globals(self, code) -> dict:
        fi = self._FunctionIdentity(code, self.globals)
        bc = ByteCode(fi)
        unused_globals = ByteCode._compute_used_globals(
            fi, bc.table, bc.co_consts, bc.co_names
        )
        return unused_globals

    def format(self, unused: dict) -> str:
        decls = Block()
        for k, v in unused.items():
            if not (isinstance(v, ModuleType) or callable(v)):
                decls += Line(f"{k} = {v}")
        return decls.to_string()

    def apply(self, pipeline, source: str) -> str:
        # TODO: Remove Numba dependency here
        # Should be doable to get this information from AST or Bytecode
        # See: https://stackoverflow.com/questions/33160744/detect-all-global-variables-within-a-python-function
        code = pipeline.get_property(Property.CO_CODE)
        unused = self.get_unused_globals(code)
        decls = self.format(unused)
        return "\n".join([decls, source])


class IncludeImports(AnalysisStep):
    def __init__(self, *args) -> None:
        self.imports: list[str] = list(args)

    def apply(self, pipeline: Pipeline, source: str) -> None:
        imports = pipeline.get_property(Property.IMPORTS)
        if imports is None:
            imports = []
        imports.extend(self.imports)
        pipeline.set_property(Property.IMPORTS, imports)


class ReplacePlaceholders(TransformationStep):
    class Formatter(string.Formatter):
        def __init__(self, mapping):
            super().__init__()
            self._mapping = mapping

        def _format_value(self, value):
            name = getattr(value, "__name__", str(value))
            return name

        def get_value(self, key, args, kwargs):
            try:
                return super().get_value(key, args, kwargs)
            except KeyError as e:
                value = self._mapping.get(key, None)
                if value is None:
                    raise KeyError(e)

                value = self._format_value(value)
                # logger.info(f"Replacing '{key}' by '{value}'")
                return value

    def __init__(self, placeholders_map):
        self.placeholders_map = placeholders_map

    def apply(self, pipeline: Pipeline, source: str) -> str:
        formatter = self.Formatter(self.placeholders_map)
        return formatter.vformat(source, [], globals())


class Node:
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def is_builtin(self, func_name) -> bool:
            builtins = globals()["__builtins__"]
            return builtins.get(func_name) is not None

        def visit_Call(self, node: ast.Call) -> Any:
            if not isinstance(node.func, ast.Name):
                return

            func_name = node.func.id
            if self.is_builtin(func_name):
                return

            self.names.add(func_name)

    @classmethod
    def from_function_object(cls, func: FunctionType, parent: Optional["Node"]):
        func_name = func.__name__
        mod = inspect.getmodule(func)
        assert mod is not None, func_name
        func_str = inspect.getsource(func)
        self = cls(func_name, func_str, mod, parent)
        return self

    @classmethod
    def from_function_name(cls, func_name: str, parent: Optional["Node"]):
        func = getattr(parent.mod, func_name)
        mod = inspect.getmodule(func)
        assert mod is not None, func_name
        func_str = inspect.getsource(func)
        self = cls(func_name, func_str, mod, parent)
        return self

    def __init__(
        self, func_name: str, func_str: str, mod: ModuleType, parent: Optional["Node"]
    ):
        self.func_name: str = func_name
        self.func_str: str = func_str
        self.parent: Optional["Node"] = parent
        self.mod = mod

    def __str__(self) -> str:
        return f'Node("{self.func_name}")'

    __repr__ = __str__

    def get_called_functions(self) -> set[str]:
        t = ast.parse(self.func_str)
        v = self.Visitor()
        v.visit(t)
        return v.names


class AddMissingInformation(TransformationStep):
    def __init__(
        self, globs: dict[str, Any], ns: dict[str, Any], mod: ModuleType
    ) -> None:
        self.ns = ns
        self.globs = globs
        self.mod = mod
        self.visited_funcs: dict[str, bool] = dict()
        self.source_map: dict[str, str] = dict()

    def replace_function_name(self, new_func_name: str, func_str: str) -> str:
        # Split the function string into lines
        lines = func_str.split("\n")

        # Find the line that contains the function definition
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                # Replace the function name in the function definition line
                old_func_name = line.split("(")[0][4:]
                if old_func_name != new_func_name:
                    new_line = line.replace(old_func_name, new_func_name)
                    lines[i] = new_line
                break

        # Join the modified lines and return the new function string
        new_func_str = "\n".join(lines)
        return new_func_str

    def add_new_source_function(self, func_name: str, source: str) -> None:
        assert func_name not in self.source_map

        # if the given func_name does not match the source declaration, one must
        # replace it before serializing the string to source_map
        decl = f"def {func_name}"
        if decl not in source:
            source = self.replace_function_name(func_name, source)
        self.source_map[func_name] = source

    def apply(self, pipeline: Pipeline, source: str) -> str:
        func_name = pipeline.get_property(Property.FUNCTION_NAME)

        self.add_new_source_function(func_name, source)
        root = Node(func_name, source, self.mod, None)
        new_funcs = root.get_called_functions()

        v: set[str] = set()
        q = deque([(name, root) for name in new_funcs])

        while len(q) > 0:
            func_name, parent = q.popleft()

            if func_name in v:
                continue

            v.add(func_name)

            if func_name in self.ns:
                obj = self.ns.get(func_name)
                node = Node.from_function_object(obj, parent)
            else:
                node = Node.from_function_name(func_name, parent)
            self.add_new_source_function(func_name, node.func_str)
            new_names = node.get_called_functions()

            for name in new_names:
                if name not in v:
                    q.append((name, node))

        return "\n".join(self.source_map.values())


class FormatSourceCodeMixin:
    def format_signature(self, pipeline: Pipeline) -> tuple[str, str, str]:
        name = pipeline.get_property(Property.FUNCTION_NAME)
        signatures = pipeline.get_property(Property.SIGNATURES)

        if signatures is None or len(signatures) == 0:
            comments = ""
            imports = ""
            sig_stmts = ""
            return comments, imports, sig_stmts

        comments = Block(
            Comment(
                f"This file contains a reproducer for function `{name}`. It consist of all the"
            ),
            Comment(
                "necessary code: imports, globals, Numba types, etc - to reproduce the issue."
            ),
            Comment(
                f"To execute, uncomment the last line: `{name}.compile(sig)`, and replace `sig`"
            ),
            Comment("by one of the available signatures"),
            Line.new_line(),
        ).to_string()

        imports = Block(
            Comment("imports for signature to be eval"),
            Line("from numba.core import types"),
            Line("from numba.core.types import *"),
            Line("from numba.core.typing import signature"),
            Line.new_line(),
        ).to_string()

        sigs_cmd = Block()
        for idx, sig in enumerate(signatures):
            retty = sig.return_type
            argtys = ", ".join(map(lambda arg: f"types.{arg!r}", sig.args))
            sigs_cmd += Line(f'sig{idx} = eval("signature({retty!r}, {argtys})", globals(), vars(types))')

        sig_stmts = Block(sigs_cmd, Comment(f"{name}.compile(sig.args)")).to_string()
        return comments, imports, sig_stmts

    def format_include_imports(self, pipeline: Pipeline) -> str:
        imports = pipeline.get_property(Property.IMPORTS)
        if imports is None:
            return ""
        imps = Block(Comment("Mandatory imports"))
        for imp in imports:
            imps += Line(imp)
        imps += Line.new_line()
        return imps.to_string()

    def format(self, pipeline: Pipeline, source: str) -> str:
        comment, imports, sig_stmts = self.format_signature(pipeline)
        include_imports = self.format_include_imports(pipeline)
        return "\n".join([comment, include_imports, imports, source, sig_stmts])


class Save(FormatSourceCodeMixin, AnalysisStep):
    def __init__(self, path: os.PathLike, ext: str = "py") -> None:
        self.path = path
        self.ext = ext

    def compute_hash(self, source):
        return hashlib.sha1(source.encode()).hexdigest()

    def save(self, filepath, content):
        with open(filepath, "w") as f:
            f.write(content)
        return content

    def compute_filepath(self, pipeline, source):
        func_name = pipeline.get_property(Property.FUNCTION_NAME)
        hash_ = self.compute_hash(source)
        filename = f"{func_name}_{hash_}.{self.ext}"
        filepath = self.path / filename
        return filepath

    def apply(self, pipeline: Pipeline, source: str) -> None:
        filepath = self.compute_filepath(pipeline, source)
        formatted_source = self.format(pipeline, source)
        self.save(filepath, formatted_source)


class ConfigChecker(AnalysisStep):
    """
    Check config file and abort pipeline execution if needed
    """

    def config_check(self):
        # TODO: Implement this method
        return True

    def apply(self, pipeline: Pipeline, source: str) -> None:
        if self.config_check() == False:
            pipeline.abort()


class Abort(AnalysisStep):
    def apply(self, pipeline: Pipeline, source: str) -> None:
        pipeline.abort()


class Debug(FormatSourceCodeMixin, AnalysisStep):
    def apply(self, pipeline: Pipeline, source: str) -> None:
        print(self.format(pipeline, source))
