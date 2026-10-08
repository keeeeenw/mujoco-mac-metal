"""Protect physics regressions from silent Python name shadowing."""
import ast
from pathlib import Path


def test_test_modules_have_no_shadowing_top_level_definitions():
  # Pytest cannot discover an earlier test overwritten by a later definition.
  # Helpers matter too: replacing a fixture can change every collected oracle.
  duplicates = []
  for path in sorted(Path(__file__).parent.glob("test_*.py")):
    seen = {}
    for node in ast.parse(path.read_text(), filename=path.name).body:
      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.name in seen:
          duplicates.append((path.name, node.name, seen[node.name], node.lineno))
        seen[node.name] = node.lineno
  assert not duplicates, duplicates
