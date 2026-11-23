"""
Property-based tests of malformed and unusual input.

- File readers: any text, and every small corruption of a valid file, is
  either read as a context or refused with ``ValueError``, never another
  error; files with Windows line endings or a byte-order mark read the same.
- Cells: one table is read the same way whether it is passed as a list, a
  NumPy array or a DataFrame, cell by cell and independently of the other
  cells; missing and complex values are refused with ``ValueError``.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from conceptflow import FormalContext  # noqa: E402
from conceptflow.io import (  # noqa: E402
    context_from_json,
    context_to_json,
    read_csv,
    read_cxt,
    write_csv,
    write_cxt,
)
from conceptflow.validation import check_binary_context_input  # noqa: E402
from tests.property_helpers import named_contexts  # noqa: E402

# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------


@st.composite
def corrupted(draw, text):
    """``text`` with lines deleted, duplicated, replaced or garbled."""
    lines = text.split("\n")
    for _ in range(draw(st.integers(1, 3), label="corruptions")):
        if not lines:
            break
        i = draw(st.integers(0, len(lines) - 1), label="line")
        action = draw(st.sampled_from(["delete", "duplicate", "replace", "garble", "blank"]))
        if action == "delete":
            del lines[i]
        elif action == "duplicate":
            lines.insert(i, lines[i])
        elif action == "replace":
            lines[i] = draw(st.sampled_from(["", "B", "-1", "0", "99", "X", ".", "XX.", "?"]))
        elif action == "garble":
            lines[i] = lines[i] + draw(st.text(max_size=3), label="garbage")
        else:
            lines.insert(i, "")
    return "\n".join(lines)


def read_or_refuse(read, path):
    """A reader may only refuse with ValueError (UnicodeDecodeError is one)."""
    try:
        return read(path)
    except ValueError:
        return None


@given(st.data(), named_contexts())
def test_cxt_reader_reads_or_refuses_corrupted_files(tmp_path_factory, data, context):
    folder = tmp_path_factory.mktemp("cxt")
    if any("\n" in n or "\r" in n for n in context.objects + context.attributes):
        return
    write_cxt(context, folder / "valid.cxt")
    text = (folder / "valid.cxt").read_text(encoding="utf-8")
    broken = data.draw(corrupted(text), label="corrupted")
    (folder / "broken.cxt").write_text(broken, encoding="utf-8")
    read_or_refuse(read_cxt, folder / "broken.cxt")


@given(st.text(max_size=60))
def test_cxt_reader_reads_or_refuses_any_text(tmp_path_factory, text):
    path = tmp_path_factory.mktemp("cxt") / "any.cxt"
    path.write_text("B\n" + text, encoding="utf-8")
    read_or_refuse(read_cxt, path)


@given(named_contexts(), st.booleans(), st.booleans())
def test_cxt_files_from_windows_read_the_same(tmp_path_factory, context, crlf, bom):
    if any("\n" in n or "\r" in n for n in context.objects + context.attributes):
        return
    folder = tmp_path_factory.mktemp("cxt")
    write_cxt(context, folder / "plain.cxt")
    text = (folder / "plain.cxt").read_text(encoding="utf-8")
    if crlf:
        text = text.replace("\n", "\r\n")
    data = text.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    (folder / "windows.cxt").write_bytes(data)
    assert read_cxt(folder / "windows.cxt") == context


@given(st.data(), named_contexts())
def test_csv_reader_reads_or_refuses_corrupted_files(tmp_path_factory, data, context):
    folder = tmp_path_factory.mktemp("csv")
    write_csv(context, folder / "valid.csv")
    text = (folder / "valid.csv").read_text(encoding="utf-8")
    broken = data.draw(corrupted(text), label="corrupted")
    (folder / "broken.csv").write_text(broken, encoding="utf-8")
    read_or_refuse(read_csv, folder / "broken.csv")


@given(st.text(max_size=80))
def test_csv_reader_reads_or_refuses_any_text(tmp_path_factory, text):
    path = tmp_path_factory.mktemp("csv") / "any.csv"
    path.write_text(text, encoding="utf-8")
    read_or_refuse(read_csv, path)


@given(named_contexts())
def test_csv_files_from_windows_read_the_same(tmp_path_factory, context):
    folder = tmp_path_factory.mktemp("csv")
    write_csv(context, folder / "plain.csv")
    data = b"\xef\xbb\xbf" + (folder / "plain.csv").read_bytes()
    (folder / "windows.csv").write_bytes(data)
    assert read_csv(folder / "windows.csv") == context


json_values = st.recursive(
    st.none() | st.booleans() | st.integers(-3, 5) | st.floats(allow_nan=False)
    | st.text(max_size=4),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.sampled_from(["ObjNames", "Params", "AttrNames", "Count", "Data",
                                       "Inds", "objects", "attributes", "incidence"]),
                      children, max_size=4),
    max_leaves=12,
)


@given(json_values)
def test_json_reader_reads_or_refuses_any_json(value):
    try:
        context_from_json(json.dumps(value))
    except ValueError:
        pass


@given(st.data(), named_contexts())
def test_json_reader_reads_or_refuses_altered_contexts(data, context):
    document = json.loads(context_to_json(context))
    # Change one value somewhere in the document.
    path = []
    node = document
    while isinstance(node, (list, dict)) and node and data.draw(st.booleans(), label="deeper"):
        key = data.draw(st.sampled_from(list(range(len(node))) if isinstance(node, list)
                                        else list(node)), label="key")
        path.append(key)
        node = node[key]
    if not path:
        return
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = data.draw(json_values, label="new value")
    try:
        context_from_json(json.dumps(document))
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Cells, whatever container they come in
# ---------------------------------------------------------------------------

READABLE = [True, False, 1, 0, 2, -1, 0.5, np.int64(3), np.float32(0.25), "1", "0", "x", "X",
            ".", "", "true", "False", " 1 ", "0.0", "×"]
UNREADABLE = ["yes", "?", "2", "1.5", "-1", "1e3", "nan", "inf", "no"]
MISSING = [np.nan, None, pd.NA, float("inf"), -float("inf")]


def containers(rows):
    width = len(rows[0])
    yield "list", rows
    yield "object array", np.array(rows, dtype=object)
    yield "DataFrame", pd.DataFrame(rows, columns=[f"m{j}" for j in range(width)])
    if all(isinstance(v, str) for row in rows for v in row):
        yield "string array", np.array(rows)


def read(X):
    try:
        return np.asarray(check_binary_context_input(X).incidence).tolist()
    except ValueError as error:
        return ("ValueError", str(error)[:40])


@given(st.lists(st.lists(st.sampled_from(READABLE + UNREADABLE), min_size=2, max_size=2),
                min_size=1, max_size=3))
def test_a_table_is_read_the_same_in_every_container(rows):
    expected = None
    try:
        expected = np.asarray(FormalContext.from_array(rows).incidence).tolist()
    except ValueError:
        pass
    for name, X in containers(rows):
        got = read(X)
        if expected is None:
            assert isinstance(got, tuple), (name, rows, got)
        else:
            assert got == expected, (name, rows, got)


@given(st.lists(st.lists(st.sampled_from(READABLE + MISSING + [1j]), min_size=2, max_size=2),
                min_size=1, max_size=3))
def test_missing_and_complex_cells_are_refused_with_value_error(rows):
    bad = any(
        v is None or v is pd.NA or isinstance(v, complex)
        or (isinstance(v, float) and not np.isfinite(v))
        for row in rows for v in row
    )
    for name, X in containers(rows):
        if bad:
            with pytest.raises(ValueError):
                check_binary_context_input(X)
        else:
            check_binary_context_input(X)
