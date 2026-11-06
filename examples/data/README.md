# Example data

`eurovision_winners.csv` is used by `examples/eurovision_nested_diagram.py`:
one row per Eurovision Song Contest winner, 1975–2025 without 2020, with the
winner's country, whether each of four country clusters (regional, cultural,
historical, political) gave them on average at least 8 points, the song's
tempo (bpm) and its key (major or minor).

It is derived from the
[Unofficial Eurovision Song Contest Dataset](https://github.com/EurovisionAPI/dataset),
which collects its data from ESC Home, Eurovision World, Eurovision LOD,
Logopedia, Ogaespain, Tunebat, Musicstax and Chordify. The cluster columns are
computed by the example's `build_dataframe`; rebuild the file from a clone of
the dataset with

    python -m examples.eurovision_nested_diagram --from-dataset <clone>
