# Contributing to ConceptFlow

Thank you for your interest in ConceptFlow. Questions, bug reports, ideas and
code are all welcome.

## Getting help and reporting problems

Use the [GitHub issues](https://github.com/anuragxorma/conceptflow/issues)
for questions, bug reports and feature requests. For a bug, please include:

- what you did, ideally as a few lines of code that reproduce it (a small
  context is enough; `conceptflow.datasets` has ready-made ones);
- what happened and what you expected;
- your ConceptFlow, Python and scikit-learn versions
  (`python -c "import conceptflow, sklearn, sys; print(conceptflow.__version__, sklearn.__version__, sys.version)"`).

## Setting up

```bash
git clone https://github.com/anuragxorma/conceptflow.git
cd conceptflow
pip install -e ".[dev,docs]"
```

ConceptFlow needs Python 3.12 or newer.

## Checks

Before proposing a change, run:

```bash
pytest                    # the test suite, including scikit-learn's estimator checks
ruff check src tests examples benchmarks .github/scripts
mkdocs build --strict     # the documentation
```

Changes to the interactive (D3) pages should also pass the browser tests,
which open the pages in headless Firefox. They are skipped unless
Playwright is installed, and need internet access for the D3 and jsPDF
scripts:

```bash
pip install playwright && playwright install firefox
pytest tests/test_browser.py
```

The same checks, browser tests included, run automatically on every push and
pull request.

## Guidelines

- **Tests:** every fix comes with a test that fails without it, and every
  feature with tests of its behaviour. Algorithms are best tested against a
  simple reference implementation (the `*_bruteforce` functions, or a direct
  transcription of the definition) on many small random contexts.
- **scikit-learn compatibility:** estimators follow the scikit-learn API and
  are added to `tests/test_check_estimator.py`.
- **Documentation:** public functions and classes have numpy-style
  docstrings. User-facing changes are described in the guide (`docs/`) and
  in `CHANGELOG.md`. Code examples in `docs/` are run by the tests; a line
  `print(x)  # value` checks that `x` prints as `value`.
- **API stability:** changes to the public API follow the
  [deprecation policy](https://conceptflow.readthedocs.io/en/latest/api-stability/).
- **Style:** follow the surrounding code; `ruff` enforces the basics.

## Pull requests

Fork the repository, make your change on a branch, and open a pull request
describing what it changes and why. Small, focused pull requests are easier
to review. By contributing you agree that your contribution is released
under the project's BSD 3-Clause license.
