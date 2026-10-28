# Datasets

`conceptflow.datasets` has data to try things on: the classic contexts of the
FCA repository, the UCI Mushroom data and random contexts.

## FCA repository contexts

The 20 contexts of the [FCA repository](https://fcarepository.org) (CC0) are
shipped with ConceptFlow, so they work offline:

```python
from conceptflow import ConceptLattice
from conceptflow.datasets import list_contexts, load_context

table = list_contexts()
print(len(table))  # 20
print(table.loc["livingbeings_en", "title"])  # Living beings and water

water = load_context("livingbeings_en")
print(water)  # FormalContext(n_objects=8, n_attributes=9)
print(ConceptLattice.from_context(water).n_concepts)  # 19
```

`list_contexts()` gives each context's title, size, language, description and
source. Among them are Ganter & Wille's "living beings and water"
(`livingbeings_en`), `planets_en`, `animals_en`, `triangles_en` and
`bodiesofwater_en`; several also come in German (`_de`).

`fetch_fca_repository(name)` downloads the current version of a context from
the repository, including contexts added after this release, and caches it:

<!-- no-test -->
```python
from conceptflow.datasets import fetch_fca_repository

water = fetch_fca_repository("livingbeings_en")
```

## Mushroom

The [UCI Mushroom](https://archive.ics.uci.edu/dataset/73/mushroom) data set
(8124 mushrooms, 22 attributes and the class edible/poisonous, CC BY 4.0) is
a standard FCA benchmark. It is downloaded on first use (about 370 KB) and
cached:

<!-- no-test -->
```python
from conceptflow.datasets import load_mushroom

mushrooms = load_mushroom()
print(mushrooms)  # FormalContext(n_objects=8124, n_attributes=119)
print(mushrooms.attributes[:3])  # ('class=edible', 'class=poisonous', 'cap-shape=bell')

raw = load_mushroom(scaled=False)  # ManyValuedContext: odor = "almond", ...
```

The default is the nominally scaled formal context used in the FCA
literature (238,710 concepts, a canonical basis of 2,323 implications); an
unknown stalk root is the attribute `stalk-root=missing`. With
`scaled=False` you get the raw values, with NaN for unknown stalk roots, to
scale yourself with `ConceptualScaler` (see
[Conceptual scaling](scaling.md)).

Downloads go to `~/conceptflow_data`, or to the folder in the
`CONCEPTFLOW_DATA` environment variable, or to `data_home=...`;
`get_data_home()` returns it.

## Random contexts

`make_random_context` gives each object each attribute independently with
probability `density`, reproducibly with `random_state`:

```python
from conceptflow.datasets import make_random_context

random_context = make_random_context(100, 15, density=0.3, random_state=0)
print(random_context)  # FormalContext(n_objects=100, n_attributes=15)
```
