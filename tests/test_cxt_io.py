import numpy as np
import pytest

from conceptflow import FormalContext
from conceptflow.io import read_cxt, write_cxt


def test_write_and_read_cxt_roundtrip(tmp_path):
    ctx = FormalContext.from_array(
        np.array([
            [1, 0, 1],
            [0, 1, 1],
        ]),
        objects=["g1", "g2"],
        attributes=["m1", "m2", "m3"],
    )

    path = tmp_path / "context.cxt"

    write_cxt(ctx, path)
    loaded = read_cxt(path)

    assert loaded.objects == ctx.objects
    assert loaded.attributes == ctx.attributes
    assert loaded.incidence.tolist() == ctx.incidence.tolist()


def test_read_cxt_rejects_invalid_header(tmp_path):
    path = tmp_path / "bad.cxt"
    path.write_text("X\n\n0\n0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_cxt(path)


def test_read_cxt_rejects_short_file(tmp_path):
    path = tmp_path / "short.cxt"
    path.write_text("B\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_cxt(path)


def test_read_cxt_rejects_bad_counts(tmp_path):
    path = tmp_path / "bad_counts.cxt"
    path.write_text("B\n\nabc\n2\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_cxt(path)


def test_read_cxt_rejects_wrong_incidence_width(tmp_path):
    path = tmp_path / "bad_width.cxt"
    path.write_text(
        "\n".join([
            "B",
            "",
            "1",
            "2",
            "g1",
            "m1",
            "m2",
            "X",
        ]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        read_cxt(path)

def test_read_cxt_accepts_lowercase_x(tmp_path):
    path = tmp_path / "lowercase.cxt"

    path.write_text(
        "\n".join(
            [
                "B",
                "",
                "2",
                "2",
                "g1",
                "g2",
                "m1",
                "m2",
                "x.",
                ".x",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    ctx = read_cxt(path)

    assert ctx.incidence.tolist() == [
        [True, False],
        [False, True],
    ]


def test_writers_create_missing_folders(tmp_path):
    # Like the TikZ, Graphviz and HTML figure writers.
    from conceptflow.io import read_csv, read_json, write_csv, write_json

    context = FormalContext.from_array([[1, 0], [0, 1]], objects=["g", "h"], attributes=["m", "n"])
    for write, read, suffix in (
        (write_cxt, read_cxt, "cxt"), (write_json, read_json, "json"), (write_csv, read_csv, "csv")
    ):
        path = tmp_path / "new" / suffix / f"context.{suffix}"
        write(context, path)
        assert read(path) == context


def test_read_csv_pads_short_rows_and_rejects_long_ones(tmp_path):
    from conceptflow.io import read_csv

    path = tmp_path / "short.csv"
    path.write_text(",m,n\na,1\n", encoding="utf-8")
    assert read_csv(path).incidence.tolist() == [[True, False]]
    path.write_text(",m\na,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Line 2 has 2 values"):
        read_csv(path)
