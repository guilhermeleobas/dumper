
# __all__ = ['Pipeline', 'addSignature', 'Compile', 'FixMissingImports',
#            'FixMissingGlobals', 'ReplacePlaceholders', 'Save']

import hashlib
import os
import string
from abc import ABC, abstractclassmethod, abstractmethod
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .line import Block, Comment, Line
from numba.core.bytecode import ByteCode
from pyflyby import PythonBlock, find_missing_imports
from pyflyby._imports2s import fix_unused_and_missing_imports


class Property:
    FUNCTION_NAME = "function_name"
    CO_CODE = "co_code"
    SIGNATURE = "signature"


class StopPipelineIteration(Exception):
    """
    """


class Pipeline:
    def __init__(self, *steps) -> None:
        if not isinstance(steps, Iterable):
            steps = (steps,)
        self._steps = tuple(steps)
        self.properties: dict[str, Any] = {}
        self.hooks: list[Hook] = []

    def abort(self):
        raise StopPipelineIteration()

    def set_property(self, key, value, /):
        self.properties[key] = value
        return self

    def get_property(self, key, /) -> Any:
        return self.properties.get(key, None)

    def run_hook(self, step: 'Step', source: str, method_name: str) -> str:
        for hook in self.hooks:
            method = getattr(hook, method_name)
            source = method(self, source, step)
        return source

    def run_step(self, step: 'Step', source: str) -> str:
        if issubclass(step.__class__, AnalysisStep):
            step.apply(self, source)
        else:
            source = step.apply(self, source)
        return source

    def _run(self, source: str) -> None:
        for step in self._steps:
            source = self.run_hook(step, source, 'before_each')
            source = self.run_step(step, source)
            source = self.run_hook(step, source, 'after_each')

    def run(self, source: str) -> None:
        try:
            self._run(source)
        except StopPipelineIteration:
            return

    def add_hook(self, hook) -> None:
        if hook not in self.hooks:
            self.hooks.append(hook)


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


class Hook(ABC):
    """
    A step can register a hook to be execute before or after each step
    """

    @abstractclassmethod
    def before_each(self, pipeline, source, current_step):
        ...

    @abstractclassmethod
    def after_each(self, pipeline, source, current_step):
        ...


class AddSignature(AnalysisStep):
    class SignatureHook(Hook):
        """
        Before saving, a signature hook mutates the source code to include
        types
        """

        @classmethod
        def before_each(cls, pipeline, source, current_step):
            if not isinstance(current_step, Save):
                return source

            name = pipeline.get_property(Property.FUNCTION_NAME)
            comment = Block(
                Comment(f"This file contains a reproducer for function `{name}`. It consist of all the"),
                Comment("necessary code: imports, globals, Numba types, etc - to reproduce the issue."),
                Comment(f"To execute, uncomment the last line: `{name}.compile(sig)`, and"),
                Comment("replace `sig` by one of the available signatures"),
            ).resolve()

            imports = Block(
                Comment("imports for signature to be eval"),
                Line("from numba.core import types"),
                Line("from numba.core.types import *"),
                Line("from numba.core.typing import signature"),
            ).resolve()

            sig_prop = pipeline.get_property(Property.SIGNATURE)

            sigs_cmd = Block()
            for idx, sig in enumerate(sig_prop):
                retty = sig.return_type
                argtys = ", ".join(map(lambda arg: f"types.{arg!r}", sig.args))
                sigs_cmd += Line(f'sig{idx} = eval("signature({retty!r}, {argtys})")')

            signatures = f"{sigs_cmd}" f"# {name}.compile(sig.args)"

            return "\n".join([comment, imports, source, signatures])

        @classmethod
        def after_each(cls, pipeline, source, current_step):
            return source

    def __init__(self, sig) -> None:
        if not isinstance(sig, (tuple, list)):
            sig = (sig,)
        self.signatures = tuple(sig)

    def apply(self, pipeline, source):
        pipeline.add_hook(self.SignatureHook)

        set_ = set(self.signatures)
        sigs = pipeline.get_property(Property.SIGNATURE)
        if sigs is not None:
            set_.update(sigs)
        pipeline.set_property(Property.SIGNATURE, set_)


class Compile(AnalysisStep):
    def apply(self, pipeline: Pipeline, source: str) -> None:
        code = PythonBlock(source).compile()
        pipeline.set_property(Property.CO_CODE, code)
        pipeline.set_property(Property.FUNCTION_NAME, code.co_names[-1])


class FixMissingGlobals(TransformationStep):
    """
    """

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
        return decls.resolve()

    def apply(self, pipeline, source: str) -> str:
        # TODO: Remove Numba dependency here
        # Should be doable to get this information from AST or Bytecode
        # See: https://stackoverflow.com/questions/33160744/detect-all-global-variables-within-a-python-function
        code = pipeline.get_property(Property.CO_CODE)
        unused = self.get_unused_globals(code)
        decls = self.format(unused)
        return "\n".join([decls, source])


class FixMissingImports(TransformationStep):
    def __init__(self):
        self.dbs = []
        for p in Path("./dbs").iterdir():
            self.dbs.append(p)

    @contextmanager
    def set_env(self, **environ):
        """
        Temporarily set the process environment variables.
        """
        old_environ = dict(os.environ)
        os.environ.update(environ)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old_environ)

    def apply(self, pipeline, source: str) -> str:
        """
        Pyflyby has a tool that find and include missing imports in snippets
        of code.
        """
        pyflyby_path = ":".join(map(lambda p: str(p.resolve()), self.dbs)) + ":-"
        with self.set_env(PYFLYBY_PATH=pyflyby_path):
            block = fix_unused_and_missing_imports(source, remove_unused=False)
            return block.text.joined


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


class Save(AnalysisStep):
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
        fn_name = pipeline.get_property(Property.FUNCTION_NAME)
        hash_ = self.compute_hash(source)
        filename = f"{fn_name}_{hash_}.{self.ext}"
        filepath = self.path / filename
        return filepath

    def apply(self, pipeline: Pipeline, source: str) -> None:
        filepath = self.compute_filepath(pipeline, source)
        self.save(filepath, source)


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
