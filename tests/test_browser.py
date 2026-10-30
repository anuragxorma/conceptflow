"""
The interactive pages in a real browser (headless Firefox via Playwright).

Skipped unless Playwright and its Firefox are installed:

    pip install playwright && playwright install firefox
    pytest tests/test_browser.py

The pages load D3 and jsPDF from CDNs, so these tests also need internet
access; without it they are skipped. With CONCEPTFLOW_BROWSER_TESTS=required
(as in CI) every reason to skip is a failure instead.
"""

import os

import numpy as np
import pandas as pd
import pytest

REQUIRED = os.environ.get("CONCEPTFLOW_BROWSER_TESTS") == "required"


def _skip(reason):
    if REQUIRED:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=True)


try:
    from playwright import sync_api
except ImportError:
    _skip("Playwright is not installed")

from conceptflow import (  # noqa: E402
    ConceptLattice,
    ExplorationBuilder,
    FormalContext,
    ManyValuedContext,
    plot_lattice,
)
from conceptflow.datasets import load_context  # noqa: E402
from conceptflow.preprocessing import (  # noqa: E402
    ContranominalScale,
    NominalScale,
    ThresholdScale,
)
from conceptflow.visualization import plot_nested  # noqa: E402

pytestmark = pytest.mark.browser

ODD_NAMES = ["", "a b", "ä€", "</script>", "'q'", "${x}", "x,y"]


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.firefox.launch()
        except Exception as exc:  # Playwright installed, its Firefox not
            _skip(f"Firefox for Playwright is not installed: {exc}")
        yield browser
        browser.close()


@pytest.fixture
def open_page(browser, tmp_path):
    """Open an HTML string; return the page and the list of its JS errors."""
    pages = []

    def open_(html, **options):
        path = tmp_path / f"page{len(pages)}.html"
        path.write_text(html, encoding="utf-8")
        page = browser.new_page(**options)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.as_uri(), wait_until="load")
        if page.evaluate("typeof d3") == "undefined":
            _skip("D3 could not be loaded from the CDN (no internet?)")
        pages.append(page)
        return page, errors

    yield open_
    for page in pages:
        page.close()


def _reduced_labels(lattice):
    """(attributes, objects) shown at each concept, by brute force."""
    ctx = lattice.context
    return [
        (
            ", ".join(sorted(a for j, a in enumerate(ctx.attributes)
                             if lattice.attribute_concept(j) == concept)) or None,
            ", ".join(sorted(g for i, g in enumerate(ctx.objects)
                             if lattice.object_concept(i) == concept)) or None,
        )
        for concept in lattice.concepts
    ]


LINE_CASES = {
    "living beings, dimflux": (load_context("livingbeings_en"), "dimflux"),
    "animals, layered": (load_context("animals_en"), "layered"),
    "odd names, dimflux": (
        FormalContext.from_array(
            np.random.default_rng(0).random((7, 4)) < 0.5,
            objects=ODD_NAMES,
            attributes=["<i>m</i>", "`t`", "${y}", "p"],
        ),
        "dimflux",
    ),
    "music, layered (163 concepts)": (load_context("music_en"), "layered"),
}


@pytest.mark.parametrize("case", LINE_CASES)
def test_line_diagram_page(open_page, case):
    context, layout = LINE_CASES[case]
    lattice = ConceptLattice.from_context(context)
    page, errors = open_page(
        plot_lattice(lattice, layout=layout, max_label_items=None).html
    )
    drawn = page.evaluate("""() => [...document.querySelectorAll('g.node-group')].map(g => {
        const c = g.querySelector('circle').getBoundingClientRect();
        const hit = document.elementFromPoint(c.x + c.width / 2, c.y + c.height / 2);
        return {
            attrs: g.querySelector('text.attr-label')?.textContent ?? null,
            objs: g.querySelector('text.obj-label')?.textContent ?? null,
            reachable: hit === g.querySelector('circle'),
        };
    })""")
    assert errors == []
    assert len(drawn) == lattice.n_concepts
    assert page.locator("line.edge").count() == len(lattice.edges)
    # Reduced labelling, and every node is the topmost element at its centre.
    assert [(d["attrs"], d["objs"]) for d in drawn] == _reduced_labels(lattice)
    assert all(d["reachable"] for d in drawn)
    page.locator("circle.node").first.click(force=True)
    assert "Extent" in page.inner_text("#conceptflow-details")


def _three_level_view():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "color": rng.choice(["red", "blue", "green"], 30),
        "size": rng.uniform(0, 100, 30).round(),
        "shape": rng.choice(["round", "square"], 30),
    }, index=[f"item{i}" for i in range(30)])
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(df))
    root = builder.root("color", [NominalScale("color")])
    builder.expand_all(root, scales=[ThresholdScale("size", [33, 66])])
    for child in list(root.children.values()):
        builder.expand_all(child, scales=[ContranominalScale("shape")])
    return root


def _wait_until_still(page, timeout=10_000):
    """Wait until the zoom transition has ended: no node moves for 100 ms."""
    page.wait_for_function(
        """() => {
            const now = [...document.querySelectorAll('svg circle')]
                .map(c => { const b = c.getBoundingClientRect(); return b.x + ',' + b.y; })
                .join(';');
            const still = now === window.__lastPositions;
            window.__lastPositions = now;
            return still;
        }""",
        polling=100,
        timeout=timeout,
    )


def _click_centre(page, element):
    _wait_until_still(page)
    box = element.bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    _wait_until_still(page)


def test_nested_page_navigation(open_page):
    page, errors = open_page(plot_nested(_three_level_view()).html)
    crumb = page.inner_text("#conceptflow-breadcrumb")
    bubbles = page.locator("#conceptflow-nested-svg g.node-group > circle.node-with-child")
    assert bubbles.count() > 0
    for i in range(bubbles.count()):
        _click_centre(page, bubbles.nth(i))
        inside = page.inner_text("#conceptflow-breadcrumb")
        assert inside != crumb
        deeper = page.locator("#conceptflow-nested-svg g.node-group > circle.node-with-child")
        if deeper.count():
            _click_centre(page, deeper.first)
            assert page.inner_text("#conceptflow-breadcrumb") != inside
            page.click("#conceptflow-root")
        else:
            page.click("#conceptflow-back")
        _wait_until_still(page)
        assert page.inner_text("#conceptflow-breadcrumb") == crumb
    # The small preview dots inside a bubble do not swallow the click.
    _click_centre(page, page.locator("circle.inner-node").first)
    assert page.inner_text("#conceptflow-breadcrumb") != crumb
    assert errors == []


def _nested(context, outer, inner, **options):
    """``plot_nested`` of ``context`` split into two attribute ranges."""
    return plot_nested(
        context.subcontext(attributes=list(outer)),
        inner=context.subcontext(attributes=list(inner)),
        **options,
    )


def test_nested_page_shows_names_and_saves_files(open_page):
    outer = FormalContext.from_array(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0]], objects=["p", "q", "r"],
        attributes=["color!=red", "a=yes", "has_limbs"],
    )
    inner = FormalContext.from_array([[1], [0], [1]], objects=["p", "q", "r"], attributes=["b=yes"])
    page, errors = open_page(plot_nested(outer, inner=inner).html, accept_downloads=True)
    labels = page.evaluate(
        "() => [...document.querySelectorAll('text.attr-label, text.inner-attr')]"
        ".map(t => t.textContent)"
    )
    assert sorted(labels) == ["a=yes", "b=yes", "color!=red", "has_limbs"]
    sizes = {}
    for button in ("svg", "png", "pdf"):
        with page.expect_download(timeout=15000) as download:
            page.click(f"#conceptflow-save-{button}")
        sizes[button] = download.value.path().read_bytes()
    assert sizes["svg"].startswith(b"<svg")
    assert sizes["png"][:4] == b"\x89PNG"
    # Latin-1 names: a vector PDF of a few KB, not a large image.
    assert sizes["pdf"].startswith(b"%PDF") and len(sizes["pdf"]) < 50_000
    assert errors == []


def test_nested_bubbles_do_not_overlap(open_page):
    animals = load_context("animals_en")
    page, errors = open_page(_nested(animals, range(7), range(7, 11)).html)
    bubbles = page.evaluate("""() => [...document.querySelectorAll(
        '#conceptflow-nested-svg g.node-group > circle')].map(c => {
        const b = c.getBoundingClientRect();
        return [b.x + b.width / 2, b.y + b.height / 2, b.width / 2];
    })""")
    for i, (x, y, r) in enumerate(bubbles):
        for x2, y2, r2 in bubbles[i + 1:]:
            assert np.hypot(x - x2, y - y2) >= r + r2 - 0.5
    assert errors == []


def test_narrow_pages_put_the_panel_below(open_page):
    living = load_context("livingbeings_en")
    for figure in (
        plot_lattice(ConceptLattice.from_context(living), width=400, height=400),
        _nested(living, range(5), range(5, 9), width=400, height=400),
    ):
        page, errors = open_page(figure.html)
        svg = page.locator("svg").first.bounding_box()
        panel = page.locator("#conceptflow-details").bounding_box()
        assert panel["y"] >= svg["y"] + svg["height"] - 1
        inside = page.evaluate("""() => {
            const s = document.querySelector('svg').getBoundingClientRect();
            return [...document.querySelectorAll('svg g.node-group circle')].every(c => {
                const r = c.getBoundingClientRect();
                return r.left >= s.left && r.right <= s.right
                    && r.top >= s.top && r.bottom <= s.bottom;
            });
        }""")
        assert inside and errors == []


def test_several_figures_in_one_notebook_page(open_page):
    """Notebook outputs: every figure in its own frame, none interfering."""
    living = load_context("livingbeings_en")
    figures = [
        plot_lattice(ConceptLattice.from_context(living)),
        plot_lattice(ConceptLattice.from_context(load_context("animals_en")), layout="layered"),
        _nested(living, range(5), range(5, 9)),
    ]
    notebook = (
        "<html><head><script src='https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js'>"
        "</script><style>svg { background: red; }</style></head><body>"
        "<h1 id='conceptflow-svg'>notebook</h1>"
        + "".join(f"<div>{figure._repr_html_()}</div>" for figure in figures)
        + "</body></html>"
    )
    page, errors = open_page(notebook)
    frames = page.frames[1:]
    count_nodes = "() => document.querySelectorAll('svg g.node-group').length"
    for frame in frames:  # each frame loads its own scripts
        frame.wait_for_load_state("load")
        frame.wait_for_function(f"({count_nodes})() > 0", timeout=15_000)
    page.wait_for_function(
        "() => [...document.querySelectorAll('iframe')].every(f => "
        "Math.abs(f.getBoundingClientRect().height"
        " - f.contentDocument.documentElement.scrollHeight) <= 8)",
        timeout=15_000,
    )
    counts = [frame.evaluate(count_nodes) for frame in frames]
    assert counts == [19, 35, 10]
    heights = [f.evaluate("() => document.documentElement.scrollHeight") for f in frames]
    boxes = [e.bounding_box()["height"] for e in page.query_selector_all("iframe")]
    assert all(abs(box - height) <= 8 for box, height in zip(boxes, heights, strict=True))
    assert page.inner_text("h1") == "notebook" and errors == []
