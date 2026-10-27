"""
JSON and CSV input/output for formal contexts.

JSON uses the context format of FCART, which fcapy also reads and writes:

    [{"ObjNames": [...], "Params": {"AttrNames": [...]}},
     {"Count": n_objects, "Data": [{"Count": k, "Inds": [attribute indices]}, ...]}]

``read_json`` also accepts a plain ``{"objects": [...], "attributes": [...],
"incidence": [[0, 1, ...], ...]}`` object.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from conceptflow.core import FormalContext

TRUE_VALUES = ("1", "1.0", "true", "True", "TRUE", "x", "X", "×")
FALSE_VALUES = ("0", "0.0", "false", "False", "FALSE", ".", "")


def context_to_json(context: FormalContext) -> str:
    """Serialize a context in the FCART/fcapy JSON format."""
    data = [
        {"ObjNames": list(context.objects), "Params": {"AttrNames": list(context.attributes)}},
        {
            "Count": context.n_objects,
            "Data": [
                {"Count": int(row.sum()), "Inds": np.flatnonzero(row).tolist()}
                for row in context.incidence
            ],
        },
    ]
    return json.dumps(data, ensure_ascii=False)


def context_from_json(text: str) -> FormalContext:
    """Parse a context from FCART/fcapy JSON or the plain dict layout."""
    data = json.loads(text)

    if isinstance(data, dict):
        return FormalContext(
            objects=data["objects"],
            attributes=data["attributes"],
            incidence=data["incidence"],
        )

    try:
        header, body = data
        objects = header["ObjNames"]
        attributes = header["Params"]["AttrNames"]
        rows = [row["Inds"] for row in body["Data"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            "Unrecognised JSON context: expected the FCART/fcapy layout or "
            '{"objects", "attributes", "incidence"}.'
        ) from exc

    if len(rows) != len(objects):
        raise ValueError(
            f"The JSON context names {len(objects)} objects but has {len(rows)} rows."
        )

    incidence = np.zeros((len(objects), len(attributes)), dtype=bool)
    for i, indices in enumerate(rows):
        for m in indices:
            if isinstance(m, bool) or not isinstance(m, int) or not 0 <= m < len(attributes):
                raise ValueError(
                    f"Object {objects[i]!r}: attribute index {m!r} is not in "
                    f"0..{len(attributes) - 1}."
                )
            incidence[i, m] = True

    return FormalContext(objects=objects, attributes=attributes, incidence=incidence)


def write_json(context: FormalContext, path: str | Path) -> None:
    """Write a context as FCART/fcapy JSON."""
    Path(path).write_text(context_to_json(context), encoding="utf-8")


def read_json(path: str | Path) -> FormalContext:
    """Read a context from FCART/fcapy JSON or the plain dict layout."""
    return context_from_json(Path(path).read_text(encoding="utf-8"))


def write_csv(
    context: FormalContext,
    path: str | Path,
    true_value: str = "1",
    false_value: str = "0",
    delimiter: str = ",",
) -> None:
    """
    Write a context as CSV: a header row of attributes, then one row per
    object starting with its name.
    """
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerow(["", *context.attributes])
        for name, row in zip(context.objects, context.incidence):
            writer.writerow([name, *(true_value if v else false_value for v in row)])


def read_csv(
    path: str | Path,
    true_values=TRUE_VALUES,
    false_values=FALSE_VALUES,
    delimiter: str = ",",
) -> FormalContext:
    """
    Read a context from CSV (header row of attributes, first column of
    object names).

    Cells must be one of ``true_values`` or ``false_values`` (surrounding
    whitespace is ignored); anything else raises ``ValueError`` naming the
    cell, instead of being guessed.
    """
    true_values, false_values = set(true_values), set(false_values)

    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file, delimiter=delimiter))

    if not rows:
        raise ValueError("Empty CSV file.")

    attributes = rows[0][1:]
    objects, incidence = [], []

    for line_number, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        objects.append(row[0])
        cells = [cell.strip() for cell in row[1:]]
        if len(cells) > len(attributes):
            raise ValueError(
                f"Line {line_number} has {len(cells)} values but the header names "
                f"{len(attributes)} attributes."
            )
        cells += [""] * (len(attributes) - len(cells))

        parsed = []
        for attribute, cell in zip(attributes, cells):
            if cell in true_values:
                parsed.append(True)
            elif cell in false_values:
                parsed.append(False)
            else:
                raise ValueError(
                    f"Line {line_number}, attribute {attribute!r}: cannot read "
                    f"{cell!r} as a cross or non-cross. Pass true_values / "
                    "false_values to say how it should be read."
                )
        incidence.append(parsed)

    return FormalContext(
        objects=objects,
        attributes=attributes,
        incidence=np.array(incidence, dtype=bool).reshape(len(objects), len(attributes)),
    )
