"""
Loaders for bundled and downloadable formal contexts.
"""

from __future__ import annotations

import hashlib
import io
import json
import numbers
import os
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from conceptflow.core import FormalContext, ManyValuedContext
from conceptflow.io import read_cxt

_REPOSITORY_URL = "https://github.com/fcatools/contexts/raw/main/contexts/{name}.cxt"
_MUSHROOM_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
    "https://archive.ics.uci.edu/static/public/73/mushroom.zip",
)
_MUSHROOM_SHA256 = "e65d082030501a3ebcbcd7c9f7c71aa9d28fdfff463bf4cf4716a3fe13ac360e"

# Attribute names and value codes, from the data set's agaricus-lepiota.names.
_MUSHROOM_ATTRIBUTES: dict[str, dict[str, str]] = {
    "class": {"e": "edible", "p": "poisonous"},
    "cap-shape": {
        "b": "bell", "c": "conical", "x": "convex", "f": "flat", "k": "knobbed", "s": "sunken",
    },
    "cap-surface": {"f": "fibrous", "g": "grooves", "y": "scaly", "s": "smooth"},
    "cap-color": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "r": "green",
        "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow",
    },
    "bruises": {"t": "bruises", "f": "no"},
    "odor": {
        "a": "almond", "l": "anise", "c": "creosote", "y": "fishy", "f": "foul",
        "m": "musty", "n": "none", "p": "pungent", "s": "spicy",
    },
    "gill-attachment": {"a": "attached", "d": "descending", "f": "free", "n": "notched"},
    "gill-spacing": {"c": "close", "w": "crowded", "d": "distant"},
    "gill-size": {"b": "broad", "n": "narrow"},
    "gill-color": {
        "k": "black", "n": "brown", "b": "buff", "h": "chocolate", "g": "gray", "r": "green",
        "o": "orange", "p": "pink", "u": "purple", "e": "red", "w": "white", "y": "yellow",
    },
    "stalk-shape": {"e": "enlarging", "t": "tapering"},
    "stalk-root": {
        "b": "bulbous", "c": "club", "u": "cup", "e": "equal",
        "z": "rhizomorphs", "r": "rooted", "?": "missing",
    },
    "stalk-surface-above-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-surface-below-ring": {"f": "fibrous", "y": "scaly", "k": "silky", "s": "smooth"},
    "stalk-color-above-ring": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
        "p": "pink", "e": "red", "w": "white", "y": "yellow",
    },
    "stalk-color-below-ring": {
        "n": "brown", "b": "buff", "c": "cinnamon", "g": "gray", "o": "orange",
        "p": "pink", "e": "red", "w": "white", "y": "yellow",
    },
    "veil-type": {"p": "partial", "u": "universal"},
    "veil-color": {"n": "brown", "o": "orange", "w": "white", "y": "yellow"},
    "ring-number": {"n": "none", "o": "one", "t": "two"},
    "ring-type": {
        "c": "cobwebby", "e": "evanescent", "f": "flaring", "l": "large",
        "n": "none", "p": "pendant", "s": "sheathing", "z": "zone",
    },
    "spore-print-color": {
        "k": "black", "n": "brown", "b": "buff", "h": "chocolate", "r": "green",
        "o": "orange", "u": "purple", "w": "white", "y": "yellow",
    },
    "population": {
        "a": "abundant", "c": "clustered", "n": "numerous",
        "s": "scattered", "v": "several", "y": "solitary",
    },
    "habitat": {
        "g": "grasses", "l": "leaves", "m": "meadows", "p": "paths",
        "u": "urban", "w": "waste", "d": "woods",
    },
}


def get_data_home(data_home: str | os.PathLike | None = None) -> Path:
    """
    The folder for downloaded data sets, created if needed.

    ``data_home`` if given, else the ``CONCEPTFLOW_DATA`` environment
    variable, else ``~/conceptflow_data``.
    """
    if data_home is None:
        data_home = os.environ.get("CONCEPTFLOW_DATA", Path.home() / "conceptflow_data")
    path = Path(data_home).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "conceptflow"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        file.write(content)
    Path(file.name).replace(path)


# ---------------------------------------------------------------------------
# FCA repository
# ---------------------------------------------------------------------------


def _bundled_metadata() -> dict[str, dict]:
    text = resources.files("conceptflow.datasets").joinpath("data", "fca_repository.json")
    return json.loads(text.read_text(encoding="utf-8"))


def list_contexts() -> pd.DataFrame:
    """
    The bundled FCA-repository contexts: one row per name, with title,
    size, language, description and source.
    """
    rows = {
        name: {**info, "source": "; ".join(info["source"])}
        for name, info in _bundled_metadata().items()
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "name"
    return frame[["title", "objects", "attributes", "language", "description", "source"]]


def load_context(name: str) -> FormalContext:
    """
    One of the 20 FCA-repository contexts shipped with ConceptFlow.

    ``name`` is a name from ``list_contexts()``, such as ``"livingbeings_en"``
    (Ganter & Wille's "living beings and water") or ``"planets_en"``. The
    contexts are copies of fcarepository.org (CC0); ``fetch_fca_repository``
    downloads the current version.
    """
    name = str(name).removesuffix(".cxt")
    if name not in _bundled_metadata():
        raise ValueError(
            f"Unknown context {name!r}. Available: {', '.join(sorted(_bundled_metadata()))}. "
            "Use fetch_fca_repository() for contexts added to the repository later."
        )
    resource = resources.files("conceptflow.datasets").joinpath(
        "data", "fca_repository", f"{name}.cxt"
    )
    with resources.as_file(resource) as path:
        return read_cxt(path)


def fetch_fca_repository(
    name: str,
    data_home: str | os.PathLike | None = None,
    refresh: bool = False,
) -> FormalContext:
    """
    Download a context from the FCA repository (fcarepository.org).

    The file is cached in ``get_data_home(data_home) / "fca_repository"``;
    ``refresh=True`` downloads it again. ``name`` is the file name without
    ``.cxt``, e.g. ``"livingbeings_en"``.
    """
    name = str(name).removesuffix(".cxt")
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", name):
        raise ValueError(f"Invalid context name {name!r}.")

    path = get_data_home(data_home) / "fca_repository" / f"{name}.cxt"
    if refresh or not path.exists():
        url = _REPOSITORY_URL.format(name=name)
        try:
            content = _download(url)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise ValueError(f"The FCA repository has no context {name!r}.") from None
            raise OSError(f"Could not download {url}: {error}") from error
        except urllib.error.URLError as error:
            raise OSError(f"Could not download {url}: {error.reason}") from error
        _write_atomically(path, content)
    return read_cxt(path)


# ---------------------------------------------------------------------------
# Mushroom
# ---------------------------------------------------------------------------


def _mushroom_data(data_home) -> bytes:
    path = get_data_home(data_home) / "mushroom" / "agaricus-lepiota.data"
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == _MUSHROOM_SHA256:
        return path.read_bytes()

    errors = []
    for url in _MUSHROOM_URLS:
        try:
            content = _download(url)
        except (urllib.error.URLError, OSError) as error:
            errors.append(f"{url}: {error}")
            continue
        if url.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    content = archive.read("agaricus-lepiota.data")
            except (zipfile.BadZipFile, KeyError):
                errors.append(f"{url}: not the expected zip archive")
                continue
        if hashlib.sha256(content).hexdigest() != _MUSHROOM_SHA256:
            errors.append(f"{url}: the downloaded file differs from the expected data")
            continue
        _write_atomically(path, content)
        return content

    raise OSError("Could not download the Mushroom data set:\n" + "\n".join(errors))


def load_mushroom(
    scaled: bool = True,
    data_home: str | os.PathLike | None = None,
) -> FormalContext | ManyValuedContext:
    """
    The UCI Mushroom data set: 8124 mushrooms, 22 attributes and the class
    (edible or poisonous).

    Downloaded from the UCI Machine Learning Repository on first use (about
    370 KB, CC BY 4.0) and cached in ``get_data_home(data_home)``.

    Parameters
    ----------
    scaled:
        ``True`` (default): the standard 8124 x 119 formal context used in
        FCA benchmarks, nominally scaled, with attributes such as
        ``"odor=almond"`` and ``"class=edible"``; unknown stalk roots are the
        attribute ``"stalk-root=missing"``. ``False``: the raw data as a
        ManyValuedContext with readable values (``"almond"``) and NaN for
        unknown stalk roots, to scale yourself.

    data_home:
        Folder for the download (see ``get_data_home``).
    """
    raw = pd.read_csv(
        io.BytesIO(_mushroom_data(data_home)),
        header=None,
        names=list(_MUSHROOM_ATTRIBUTES),
        dtype=str,
    )
    raw.index = [f"mushroom{i}" for i in range(len(raw))]
    readable = raw.apply(lambda column: column.map(_MUSHROOM_ATTRIBUTES[column.name]))

    if not scaled:
        return ManyValuedContext.from_dataframe(readable.replace("missing", np.nan))

    columns = {}
    for attribute, codes in _MUSHROOM_ATTRIBUTES.items():
        for value in codes.values():
            present = (readable[attribute] == value).to_numpy()
            if present.any():
                columns[f"{attribute}={value}"] = present
    return FormalContext.from_dataframe(pd.DataFrame(columns, index=readable.index))


# ---------------------------------------------------------------------------
# Random contexts
# ---------------------------------------------------------------------------


def make_random_context(
    n_objects: int,
    n_attributes: int,
    density: float = 0.3,
    random_state=None,
) -> FormalContext:
    """
    A random formal context: each object has each attribute independently
    with probability ``density``.

    Objects are named ``g0, g1, ...`` and attributes ``m0, m1, ...``.
    ``random_state`` is an int, None, a NumPy ``RandomState`` (as in
    scikit-learn) or a NumPy ``Generator`` (``np.random.default_rng``).
    """
    for name, value in (("n_objects", n_objects), ("n_attributes", n_attributes)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if isinstance(density, bool) or not isinstance(density, numbers.Real) or not 0 <= density <= 1:
        raise ValueError("density must be a number in [0, 1].")

    shape = (n_objects, n_attributes)
    if isinstance(random_state, np.random.Generator):
        values = random_state.random(shape)
    else:
        values = check_random_state(random_state).random_sample(shape)
    return FormalContext.from_array(values < density)
