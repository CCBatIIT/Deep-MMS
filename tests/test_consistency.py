"""
Consistency and style audits run as pytest tests.

These catch regressions in documentation coverage, export completeness,
and import hygiene without requiring any data files.
"""

import ast
import glob
import importlib
import os
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Names that are intentionally undocumented (inner helpers / boilerplate)
_SKIP = frozenset({
    "setup", "log", "combine", "basis_for_feature", "fitness_one",
    "loss_fn", "schedule", "weighted_atom_rmsd", "TrainState",
    "safe_diff", "parse_field", "_check_and_grow", "_grow",
    "_es_step", "_bspline_basis",
})

_PY_FILES = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), "..", "deepmms", "**", "*.py"), recursive=True)
    + glob.glob(os.path.join(os.path.dirname(__file__), "..", "scripts", "*.py"))
)


def _public_nodes(filepath):
    """Yield (node, filepath) for every public function/class in a .py file."""
    tree = ast.parse(open(filepath).read())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not node.name.startswith("_") and node.name not in _SKIP:
                yield node, filepath


def _has_docstring(node):
    return (
        len(node.body) > 0
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    )


# ---------------------------------------------------------------------------
# Collect (node, file) pairs for parametrisation
# ---------------------------------------------------------------------------

_UNDOC_CASES = []
for fp in _PY_FILES:
    for node, path in _public_nodes(fp):
        if not _has_docstring(node):
            relpath = os.path.relpath(path)
            _UNDOC_CASES.append(pytest.param(
                relpath, node.name, node.lineno,
                id=f"{relpath}:{node.lineno}:{node.name}"
            ))


@pytest.mark.parametrize("filepath,name,lineno", _UNDOC_CASES)
def test_public_symbol_has_docstring(filepath, name, lineno):
    """Every public function and class must have a docstring."""
    pytest.fail(f"{filepath}:{lineno}  '{name}' is missing a docstring")


# ---------------------------------------------------------------------------
# Export completeness
# ---------------------------------------------------------------------------

_PACKAGES = [
    "deepmms",
    "deepmms.models",
    "deepmms.training",
    "deepmms.analysis",
]


@pytest.mark.parametrize("pkg", _PACKAGES)
def test_all_exports_are_importable(pkg):
    """Every symbol in __all__ must be accessible as an attribute."""
    mod = importlib.import_module(pkg)
    all_list = getattr(mod, "__all__", [])
    missing = [s for s in all_list if not hasattr(mod, s)]
    assert not missing, f"{pkg}.__all__ contains inaccessible: {missing}"


# ---------------------------------------------------------------------------
# No old-style pyscripts imports in deepmms/
# ---------------------------------------------------------------------------

_DEEPMMS_FILES = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), "..", "deepmms", "**", "*.py"), recursive=True)
)


@pytest.mark.parametrize("filepath", _DEEPMMS_FILES,
                          ids=[os.path.relpath(f) for f in _DEEPMMS_FILES])
def test_no_pyscripts_imports(filepath):
    """deepmms/ modules must not import from the old pyscripts/ package."""
    src = open(filepath).read()
    # Check only actual import statements, not docstring mentions
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "pyscripts" not in node.module, \
                    f"{filepath}: 'from pyscripts' import found at line {node.lineno}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "pyscripts" not in alias.name, \
                        f"{filepath}: 'import pyscripts' found at line {node.lineno}"


# ---------------------------------------------------------------------------
# All scripts have if __name__ == '__main__' guard
# ---------------------------------------------------------------------------

_SCRIPT_FILES = sorted(
    glob.glob(os.path.join(os.path.dirname(__file__), "..", "scripts", "*.py"))
)


@pytest.mark.parametrize("filepath", _SCRIPT_FILES,
                          ids=[os.path.relpath(f) for f in _SCRIPT_FILES])
def test_scripts_have_main_guard(filepath):
    """Every script must have an if __name__ == '__main__' guard."""
    src = open(filepath).read()
    assert "__name__" in src, \
        f"{os.path.relpath(filepath)} is missing an if __name__ == '__main__' guard"


# ---------------------------------------------------------------------------
# hidden_layers_from_config defined on every concrete model
# ---------------------------------------------------------------------------

_MODEL_CLASSES = [
    "BatchNorm_VAE", "BetaVAE", "VQVAE", "EquivariantVAE", "PerceiverVAE",
    "HierarchicalVAE", "SE3TransformerVAE", "MambaVAE", "RealNVPFlow",
    "MaskedAutoencoder", "KANVAE", "TransformerVAE", "NEATAutoencoder",
]


@pytest.mark.parametrize("classname", _MODEL_CLASSES)
def test_model_has_hidden_layers_from_config(classname):
    """Every concrete model class must define hidden_layers_from_config."""
    import deepmms.models as m
    cls = getattr(m, classname)
    assert "hidden_layers_from_config" in cls.__dict__ or \
           any("hidden_layers_from_config" in c.__dict__ for c in cls.__mro__[1:]), \
           f"{classname} does not define hidden_layers_from_config"
