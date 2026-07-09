"""
Burmeister .cxt file input/output.

The .cxt format is a common plain-text format for binary formal contexts.
"""

from __future__ import annotations

from pathlib import Path

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

    with path.open("r", encoding="utf-8") as file:
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

    expected_min_lines = 4 + n_objects + n_attributes + n_objects
    if len(lines) < expected_min_lines:
        raise ValueError("Invalid .cxt file: file ended too early.")

    object_start = 4

    if len(lines) > object_start and lines[object_start] == "":
        object_start += 1
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

    return FormalContext(
        objects=objects,
        attributes=attributes,
        incidence=incidence,
    )


def write_cxt(context: FormalContext, path: str | Path) -> None:
    """
    Write a FormalContext to a Burmeister .cxt file.
    """
    path = Path(path)

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

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")