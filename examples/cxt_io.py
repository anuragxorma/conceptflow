"""
Burmeister .cxt I/O example.

Run with:

    python examples/cxt_io.py
"""

import tempfile
from pathlib import Path

import numpy as np

import conceptflow as cf
from conceptflow.io import read_cxt, write_cxt


def main():
    context = cf.FormalContext.from_array(
        np.array([
            [1, 0, 1],
            [0, 1, 1],
        ]),
        objects=["g1", "g2"],
        attributes=["m1", "m2", "m3"],
    )

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "example_context.cxt"
        write_cxt(context, path)
        loaded = read_cxt(path)

    print("Original:")
    print(context.to_dataframe())

    print("\nLoaded:")
    print(loaded.to_dataframe())


if __name__ == "__main__":
    main()
