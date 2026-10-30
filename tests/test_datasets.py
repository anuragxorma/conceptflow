"""
Datasets. Downloads are replaced by local data, so the tests run offline.
"""

import hashlib
import io
import urllib.error
import zipfile

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext, ManyValuedContext
from conceptflow.datasets import (
    _loaders,
    fetch_fca_repository,
    get_data_home,
    list_contexts,
    load_context,
    load_mushroom,
    make_random_context,
)
from conceptflow.io import write_cxt

# Two mushrooms in the UCI format: class first, then the 22 attributes.
_MUSHROOM_ROWS = (
    b"p,x,s,n,t,p,f,c,n,k,e,e,s,s,w,w,p,w,o,p,k,s,u\n"
    b"e,x,s,y,t,a,f,c,b,k,e,?,s,s,w,w,p,w,o,p,n,n,g\n"
)


@pytest.fixture
def fake_mushroom(monkeypatch):
    monkeypatch.setattr(_loaders, "_MUSHROOM_SHA256", hashlib.sha256(_MUSHROOM_ROWS).hexdigest())
    calls = []

    def download(url):
        calls.append(url)
        return _MUSHROOM_ROWS

    monkeypatch.setattr(_loaders, "_download", download)
    return calls


# ---------------------------------------------------------------------------
# Bundled FCA-repository contexts
# ---------------------------------------------------------------------------


def test_list_contexts_describes_all_bundled_contexts():
    table = list_contexts()
    assert len(table) == 20
    assert {"livingbeings_en", "planets_en", "triangles_en"} <= set(table.index)
    assert list(table.columns) == [
        "title", "objects", "attributes", "language", "description", "source",
    ]


@pytest.mark.parametrize("name", list_contexts().index)
def test_every_bundled_context_loads_with_its_listed_size(name):
    context = load_context(name)
    row = list_contexts().loc[name]
    assert (context.n_objects, context.n_attributes) == (row["objects"], row["attributes"])


def test_living_beings_and_water():
    # Ganter & Wille, Figure 1.1: 8 living beings, 9 attributes, 19 concepts.
    context = load_context("livingbeings_en.cxt")
    assert "leech" in context.objects
    assert ConceptLattice.from_context(context).n_concepts == 19


def test_unknown_context_names_the_available_ones():
    with pytest.raises(ValueError, match="livingbeings_en"):
        load_context("living_beings")


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def test_fetch_fca_repository_downloads_once(tmp_path, monkeypatch):
    buffer = tmp_path / "source.cxt"
    write_cxt(FormalContext.from_array([[1, 0], [0, 1]]), buffer)
    calls = []

    def download(url):
        calls.append(url)
        return buffer.read_bytes()

    monkeypatch.setattr(_loaders, "_download", download)
    first = fetch_fca_repository("tiny", data_home=tmp_path)
    second = fetch_fca_repository("tiny.cxt", data_home=tmp_path)
    assert first == second
    assert calls == ["https://github.com/fcatools/contexts/raw/main/contexts/tiny.cxt"]

    fetch_fca_repository("tiny", data_home=tmp_path, refresh=True)
    assert len(calls) == 2


def test_fetch_fca_repository_unknown_name(tmp_path, monkeypatch):
    def missing(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    monkeypatch.setattr(_loaders, "_download", missing)
    with pytest.raises(ValueError, match="no context 'nothing'"):
        fetch_fca_repository("nothing", data_home=tmp_path)
    with pytest.raises(ValueError, match="Invalid context name"):
        fetch_fca_repository("../escape", data_home=tmp_path)


def test_data_home_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CONCEPTFLOW_DATA", str(tmp_path / "data"))
    assert get_data_home() == tmp_path / "data"
    assert (tmp_path / "data").is_dir()


def test_load_mushroom_scaled_and_raw(tmp_path, fake_mushroom):
    context = load_mushroom(data_home=tmp_path)
    assert isinstance(context, FormalContext)
    assert context.n_objects == 2
    assert "class=poisonous" in context.attributes
    assert "odor=almond" in context.attributes
    assert "stalk-root=missing" in context.attributes
    # Every mushroom has exactly one value per attribute.
    assert context.incidence.sum(axis=1).tolist() == [23, 23]

    raw = load_mushroom(scaled=False, data_home=tmp_path)
    assert isinstance(raw, ManyValuedContext)
    assert raw.data.loc["mushroom0", "odor"] == "pungent"
    assert raw.data["stalk-root"].isna().tolist() == [False, True]
    assert len(fake_mushroom) == 1  # downloaded once, then cached


def test_load_mushroom_falls_back_to_the_zip_archive(tmp_path, monkeypatch, fake_mushroom):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("agaricus-lepiota.data", _MUSHROOM_ROWS)

    def download(url):
        if url.endswith(".data"):
            raise urllib.error.URLError("unavailable")
        return archive.getvalue()

    monkeypatch.setattr(_loaders, "_download", download)
    assert load_mushroom(data_home=tmp_path).n_objects == 2


def test_load_mushroom_rejects_unexpected_data(tmp_path, monkeypatch):
    monkeypatch.setattr(_loaders, "_download", lambda url: b"not the data\n")
    with pytest.raises(OSError, match="differs from the expected data") as error:
        load_mushroom(data_home=tmp_path)
    assert "not the expected zip archive" in str(error.value)


# ---------------------------------------------------------------------------
# Random contexts
# ---------------------------------------------------------------------------


def test_make_random_context():
    context = make_random_context(200, 10, density=0.25, random_state=0)
    assert (context.n_objects, context.n_attributes) == (200, 10)
    assert 0.2 < context.incidence.mean() < 0.3
    assert context == make_random_context(200, 10, density=0.25, random_state=0)
    assert make_random_context(3, 2, density=0).incidence.sum() == 0
    assert make_random_context(0, 0).n_objects == 0
    generator = np.random.RandomState(1)
    assert make_random_context(4, 4, random_state=generator).n_attributes == 4


@pytest.mark.parametrize(
    "arguments", [(-1, 3), (3, 2.5), (True, 3)]
)
def test_make_random_context_rejects_bad_sizes(arguments):
    with pytest.raises(ValueError, match="non-negative integer"):
        make_random_context(*arguments)


def test_make_random_context_rejects_bad_density():
    with pytest.raises(ValueError, match="density"):
        make_random_context(3, 3, density=1.5)


def test_make_random_context_accepts_numpy_generators():
    # As documented; scikit-learn's check_random_state alone rejects them.
    first = make_random_context(6, 5, 0.4, random_state=np.random.default_rng(3))
    again = make_random_context(6, 5, 0.4, random_state=np.random.default_rng(3))
    assert first == again and first.n_objects == 6 and first.n_attributes == 5
    # Ints and RandomState give the same contexts as before.
    assert make_random_context(6, 5, 0.4, random_state=3) == make_random_context(
        6, 5, 0.4, random_state=np.random.RandomState(3)
    )
