"""Checks that need no model and no database: they catch a refactor that leaves a name undefined or breaks an import."""

import ast
import builtins
import importlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIPPED_DIRS = {".venv", ".git", "benchmark_results", "__pycache__", "sample_images"}
MODULES = [
    "config", "vlm.response_format", "install", "scan_and_index", "search", "pipeline.uninstall",
    "db.connection", "db.operations", "pipeline.console", "pipeline.discovery", "pipeline.locking",
    "pipeline.progress", "pipeline.validation", "pipeline.error_logger", "pipeline.serialisation",
    "image_processing.metadata", "image_processing.ocr", "image_processing.preprocess", "image_processing.features",
    "text_processing.formatters", "vlm.analyzer", "vlm.client",
]


def project_files():
    return [path for path in ROOT.rglob("*.py") if not SKIPPED_DIRS & set(path.relative_to(ROOT).parts)]


def undefined_calls(source: str) -> set:
    """Names that are called but are never defined, imported, assigned or a builtin anywhere in the file."""
    tree = ast.parse(source)
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    return called - defined


class StaticTests(unittest.TestCase):
    def test_no_file_calls_an_undefined_function(self):
        for path in project_files():
            self.assertEqual(undefined_calls(path.read_text(encoding="utf-8")), set(), str(path.relative_to(ROOT)))

    def test_the_check_would_have_caught_a_function_lost_in_a_refactor(self):
        broken = "def pull(client):\n    _pull_with_progress(client)\n"
        self.assertEqual(undefined_calls(broken), {"_pull_with_progress"})

    def test_every_module_imports(self):
        for name in MODULES:
            importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()
