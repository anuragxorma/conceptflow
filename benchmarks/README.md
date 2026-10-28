# Benchmarks

`run_benchmarks.py` compares ConceptFlow with the Python FCA libraries
[fcapy](https://github.com/EgorDudyrev/FCApy) and
[concepts](https://github.com/xflr6/concepts) on concepts, lattices,
stability and the canonical basis, using bundled, random and UCI Mushroom
contexts. Results and the setup are in the documentation
(<https://conceptflow.readthedocs.io/en/latest/benchmarks/>).

```bash
pip install conceptflow fcapy concepts
python benchmarks/run_benchmarks.py --quick            # small data sets, ~1 min
python benchmarks/run_benchmarks.py --timeout 120      # all, ~40 min
python benchmarks/run_benchmarks.py --output results.json
```

Each measurement runs in a separate process with a time limit; the script
checks that the libraries agree on every count they report and exits with
status 1 if they do not. Mushroom is downloaded once (see
`conceptflow.datasets.load_mushroom`).
