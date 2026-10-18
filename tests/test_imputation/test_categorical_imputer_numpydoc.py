"""Regression test for feature-engine/feature_engine issue #1059."""

import ast
import re
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2] / "feature_engine/imputation/categorical.py"
)


@pytest.mark.parametrize(
    "member, parameters",
    [
        (
            "class",
            (
                "imputation_method",
                "fill_value",
                "variables",
                "return_object",
                "ignore_format",
            ),
        ),
        ("fit", ("X", "y")),
    ],
)
def test_categorical_imputer_numpydoc_parameter_headings(member, parameters):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CategoricalImputer"
    )
    node = (
        cls
        if member == "class"
        else next(
            item
            for item in cls.body
            if isinstance(item, ast.FunctionDef) and item.name == member
        )
    )
    doc = ast.get_docstring(node)
    assert doc is not None
    for parameter in parameters:
        assert re.search(rf"^\s*{parameter} : \S+", doc, flags=re.MULTILINE), parameter
        assert not re.search(rf"^\s*{parameter}:", doc, flags=re.MULTILINE), parameter
