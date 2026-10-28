# Benchmarks

ConceptFlow compared with the two other Python FCA libraries,
[fcapy](https://github.com/EgorDudyrev/FCApy) and
[concepts](https://github.com/xflr6/concepts). The script is
[`benchmarks/run_benchmarks.py`](https://github.com/anuragxorma/conceptflow/blob/main/benchmarks/run_benchmarks.py);
run it yourself with

```bash
pip install conceptflow fcapy concepts
python benchmarks/run_benchmarks.py
```

## Setup

- Every measurement runs in its own process with a limit of 120 s, on
  contexts written to `.cxt` files (reading the file is not timed). The
  fastest of up to three runs is shown.
- Each library is used with its own default or fastest available method:
  ConceptFlow's Close-by-One and LinCbO, fcapy's Close-by-One (concepts) and
  its default `ConceptLattice.from_context` (lattice). fcapy's LCM option
  could not be measured: it fails with current scikit-learn (its dependency
  scikit-mine uses a removed scikit-learn function).
- All libraries report the same numbers of concepts and cover edges wherever
  more than one finished, e.g. 238,710 concepts and 1,370,991 edges for
  Mushroom (ConceptFlow only; the others hit the limit).
- Measured on 2026-10-28 with ConceptFlow 0.4.0, fcapy 0.1.4.5, concepts
  0.9.2, Python 3.12, NumPy 2.5 and scikit-learn 1.9.1, on an AMD Ryzen AI 7
  PRO 350 laptop (Ubuntu 26.04), single-threaded.

**Tasks**

- *concepts*: all formal concepts;
- *lattice*: the concepts and the cover relation (Hasse diagram);
- *stability*: exact stability of every concept;
- *stability bounds*: lower and upper stability bounds;
- *canonical basis*: the Duquenne-Guigues basis.

For both stability tasks the lattice is built first and not timed.
"—" means the library does not offer the task.

## Results

| Data set | Concepts | Task | ConceptFlow | fcapy | concepts |
|---|---|---|---|---|---|
| livingbeings_en (8×9) | 19 | concepts | < 0.001 s | < 0.001 s | — |
|  |  | lattice | < 0.001 s | < 0.001 s | < 0.001 s |
|  |  | stability | < 0.001 s | < 0.001 s | — |
|  |  | stability bounds | < 0.001 s | < 0.001 s | — |
|  |  | canonical basis | < 0.001 s | — | — |
| seasoningplanner_de (56×37) | 532 | concepts | 0.003 s | 0.090 s | — |
|  |  | lattice | 0.006 s | 1.5 s | 0.065 s |
|  |  | stability | 0.006 s | > 120 s | — |
|  |  | stability bounds | 0.003 s | 0.001 s | — |
|  |  | canonical basis | 0.027 s | — | — |
| random 100×15, 30% | 688 | concepts | 0.002 s | 0.078 s | — |
|  |  | lattice | 0.006 s | 1.1 s | 0.227 s |
|  |  | stability | 0.007 s | > 120 s | — |
|  |  | stability bounds | 0.003 s | 0.002 s | — |
|  |  | canonical basis | 0.008 s | — | — |
| random 200×50, 15% | 2,797 | concepts | 0.018 s | 1.6 s | — |
|  |  | lattice | 0.039 s | 17.4 s | 3.4 s |
|  |  | stability | 0.056 s | > 120 s | — |
|  |  | stability bounds | 0.017 s | 0.007 s | — |
|  |  | canonical basis | 0.322 s | — | — |
| random 500×25, 30% | 22,184 | concepts | 0.184 s | 16.1 s | — |
|  |  | lattice | 0.562 s | > 120 s | > 120 s |
|  |  | stability | 0.804 s | > 120 s | — |
|  |  | stability bounds | 0.167 s | > 120 s | — |
|  |  | canonical basis | 1.1 s | — | — |
| random 100×20, 60% | 37,103 | concepts | 0.320 s | 5.5 s | — |
|  |  | lattice | 0.972 s | > 120 s | 15.8 s |
|  |  | stability | 1.6 s | > 120 s | — |
|  |  | stability bounds | 0.250 s | > 120 s | — |
|  |  | canonical basis | 0.580 s | — | — |
| random 2000×30, 20% | 34,610 | concepts | 0.358 s | > 120 s | — |
|  |  | lattice | 1.0 s | > 120 s | > 120 s |
|  |  | stability | 1.9 s | > 120 s | — |
|  |  | stability bounds | 0.365 s | > 120 s | — |
|  |  | canonical basis | 4.0 s | — | — |
| Mushroom (8124×119) | 238,710 | concepts | 14.4 s | > 120 s | — |
|  |  | lattice | 30.2 s | > 120 s | > 120 s |
|  |  | stability | 38.0 s | > 120 s | — |
|  |  | stability bounds | 10.2 s | > 120 s | — |
|  |  | canonical basis | 16.9 s | — | — |

## Reading the results

- **Concepts and lattices:** ConceptFlow is 17–89× faster than fcapy on
  concepts and 180–450× faster on lattices, and 11–87× faster than concepts
  on lattices. From about 20,000 concepts on, fcapy and concepts mostly hit
  the limit; ConceptFlow computes the 238,710 concepts of Mushroom in 14 s
  and its whole lattice in 30 s.
- **Stability:** fcapy's exact stability did not finish within 120 s even
  for 532 concepts. ConceptFlow computes exact stability concept by concept
  from the lower neighbours, so it finished everywhere, Mushroom included
  (38 s). fcapy computes stability *bounds* 2–3× faster on small lattices,
  but did not finish them from about 20,000 concepts on.
- **Canonical basis:** only ConceptFlow computes it.

All three libraries are written in Python. Compiled FCA tools are faster
still: the R package fcaR (C++) computes Mushroom's concepts in about 0.5 s
and its canonical basis in about 1 s.
