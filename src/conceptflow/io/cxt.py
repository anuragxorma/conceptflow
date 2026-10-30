"""
Burmeister .cxt file input/output.

The .cxt format is a common plain-text format for binary formal contexts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conceptflow.core import FormalContext


def read_cxt(path: str | Path) -> FormalContext:
    """
    Read a Burmeister .cxt file as a FormalContext.

    Notes
    -----
    Both uppercase ``X`` and lowercase ``x`` are accepted as positive
    incidences. Dots ``.`` are interpreted as negative incidences.
    """
    path = Path(path)

    # utf-8-sig also reads files that start with a byte-order mark.
    with path.open("r", encoding="utf-8-sig") as file:
        lines = [line.rstrip("\n") for line in file]

    if not lines or lines[0] != "B":
        raise ValueError("Invalid .cxt file: expected first line to be 'B'.")

    if len(lines) < 4:
        raise ValueError("Invalid .cxt file: file is too short.")

    try:
        n_objects = int(lines[2])
        n_attributes = int(lines[3])
    except ValueError as exc:
        raise ValueError("Invalid .cxt file: object/attribute counts invalid.") from exc

    # The Burmeister format places a blank separator line before the object
    # list, but some files omit it. Disambiguate using the total line count
    # (which is fully determined by n_objects/n_attributes) rather than by
    # sniffing whether line 4 is blank -- that heuristic misfires when the
    # first object's name legitimately IS the empty string.
    expected_without_separator = 4 + n_objects + n_attributes + n_objects
    expected_with_separator = expected_without_separator + 1

    # Blank lines after the cross table are not part of the context. (Rows
    # of a context without attributes are blank too, so only lines beyond
    # the expected count are dropped.)
    while len(lines) > expected_with_separator and not lines[-1].strip():
        lines.pop()
    if (
        len(lines) == expected_with_separator
        and len(lines) > 4
        and lines[4] != ""
        and not lines[-1].strip()
        and n_attributes > 0
    ):
        lines.pop()  # no separator line, one trailing blank line

    if len(lines) == expected_with_separator:
        object_start = 5
    elif len(lines) == expected_without_separator:
        object_start = 4
    else:
        raise ValueError("Invalid .cxt file: file ended too early.")

    attribute_start = object_start + n_objects
    incidence_start = attribute_start + n_attributes

    objects = tuple(lines[object_start:attribute_start])
    attributes = tuple(lines[attribute_start:incidence_start])

    incidence_lines = lines[incidence_start:incidence_start + n_objects]

    incidence: list[list[bool]] = []

    for row in incidence_lines:
        normalized_row = row.strip()

        if len(normalized_row) != n_attributes:
            raise ValueError(
                "Invalid .cxt file: incidence row has wrong length."
            )

        invalid_symbols = {
            char
            for char in normalized_row
            if char not in {"X", "x", "."}
        }

        if invalid_symbols:
            raise ValueError(
                "Invalid .cxt file: incidence rows may only contain "
                "'X', 'x', or '.'."
            )

        incidence.append(
            [
                char.upper() == "X"
                for char in normalized_row
            ]
        )

    if incidence:
        incidence_array = np.array(incidence, dtype=bool)
    else:
        incidence_array = np.empty((n_objects, n_attributes), dtype=bool)

    return FormalContext(
        objects=objects,
        attributes=attributes,
        incidence=incidence_array,
    )


def write_cxt(context: FormalContext, path: str | Path) -> None:
    """
    Write a FormalContext to a Burmeister .cxt file (missing folders are
    created).
    """
    path = Path(path)

    for kind, names in (("object", context.objects), ("attribute", context.attributes)):
        for name in names:
            if "\n" in name or "\r" in name:
                raise ValueError(
                    f"The {kind} name {name!r} contains a line break, which the "
                    ".cxt format cannot store (one name per line)."
                )

    lines: list[str] = [
        "B",
        "",
        str(context.n_objects),
        str(context.n_attributes),
        "",
    ]

    lines.extend(context.objects)
    lines.extend(context.attributes)

    for row in context.incidence:
        lines.append("".join("X" if value else "." for value in row))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
