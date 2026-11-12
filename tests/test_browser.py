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


def _three_level_view(df=None):
    if df is None:
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


_NODE_STATES = """() => [...document.querySelectorAll(
    '#conceptflow-nested-svg g.node-group > circle')].map(c => {
    const d = d3.select(c).datum();
    return [c.getAttribute('class'), d.is_filled === true, d.has_child === true];
})"""


def _check_drawn_states(page):
    """Each drawn node shows its data: entered iff it has a child, else filled/hollow."""
    states = page.evaluate(_NODE_STATES)
    for css, filled, has_child in states:
        classes = set(css.split())
        assert ("node-with-child" in classes) == has_child
        assert has_child <= filled  # only filled nodes can be entered
        if not has_child:
            assert ("filled" in classes) == filled
    return states


def test_nested_levels_are_drawn_as_their_data(open_page):
    # All green items are small, so "green and large" is hollow and must not
    # be entered, although the green items' own view has a "large" concept.
    df = pd.DataFrame({
        "color": ["red", "red", "blue", "blue", "green", "green"],
        "size": [10, 80, 50, 90, 10, 20],
        "shape": ["round", "square", "round", "square", "round", "square"],
    }, index=[f"item{i}" for i in range(6)])
    page, errors = open_page(plot_nested(_three_level_view(df)).html)
    outer = page.locator("#conceptflow-nested-svg g.node-group > circle.node-with-child")
    seen = {"level 2": 0, "level 3": 0, "hollow": 0}
    for i in range(outer.count()):
        _click_centre(page, outer.nth(i))
        states = _check_drawn_states(page)
        seen["level 2"] += len(states)
        seen["hollow"] += sum(not filled for _, filled, _ in states)
        middle = page.locator("#conceptflow-nested-svg g.node-group > circle.node-with-child")
        for j in range(middle.count()):
            _click_centre(page, middle.nth(j))
            seen["level 3"] += len(_check_drawn_states(page))
            page.click("#conceptflow-back")
            _wait_until_still(page)
        page.click("#conceptflow-root")
        _wait_until_still(page)
    assert min(seen.values()) > 0, seen
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
    # The pages fill the window when saved, but in a notebook (whose frame
    # grows to the page) the drawing is 700 px high and as wide as the cell.
    for frame, iframe in zip(frames, page.query_selector_all("iframe"), strict=True):
        svg = frame.evaluate("() => { const b = document.querySelector('svg')"
                             ".getBoundingClientRect(); return [b.width, b.height]; }")
        assert svg[1] == pytest.approx(700, abs=1)
        assert svg[0] == pytest.approx(iframe.bounding_box()["width"] - 282, abs=2)
    assert page.inner_text("h1") == "notebook" and errors == []


# A drawing in its window: its size, its nodes' centres, and whether every
# node lies inside the drawing.
_DRAWING = """() => {
    const svg = document.querySelector('svg.conceptflow-drawing');
    const s = svg.getBoundingClientRect();
    const nodes = [...svg.querySelectorAll('g.node-group > circle')].map(c => {
        const r = c.getBoundingClientRect();
        return [r.x + r.width / 2, r.y + r.height / 2, r.width / 2];
    });
    return {
        width: s.width, height: s.height,
        drawnWidth: Number(svg.getAttribute('width')),
        drawnHeight: Number(svg.getAttribute('height')),
        pageHeight: document.documentElement.scrollHeight,
        nodes: nodes,
        inside: nodes.every(([x, y, r]) => x - r >= s.left - 0.5 && x + r <= s.right + 0.5
            && y - r >= s.top - 0.5 && y + r <= s.bottom + 0.5),
    };
}"""


def _drawn_for_current_size(page):
    """Wait until the drawing has been fitted to the current size of its area."""
    page.wait_for_function(
        """() => {
            const svg = document.querySelector('svg.conceptflow-drawing');
            const s = svg.getBoundingClientRect();
            return Math.abs(Number(svg.getAttribute('width')) - s.width) <= 1
                && Math.abs(Number(svg.getAttribute('height')) - s.height) <= 1;
        }""",
        timeout=5_000,
    )
    _wait_until_still(page)
    return page.evaluate(_DRAWING)


def _window_figures():
    living = load_context("livingbeings_en")
    return {
        "line diagram": plot_lattice(ConceptLattice.from_context(living)),
        "nested diagram": _nested(living, range(5), range(5, 9)),
    }


@pytest.mark.parametrize("name", ["line diagram", "nested diagram"])
def test_pages_fill_the_window_and_follow_its_size(open_page, name):
    page, errors = open_page(_window_figures()[name].html,
                             viewport={"width": 1600, "height": 900})
    drawing = _drawn_for_current_size(page)
    # All the window but the details panel (and the frame's border) and the
    # header, without scrolling.
    assert drawing["width"] == pytest.approx(1600 - 282, abs=2)
    assert drawing["height"] > 700 and drawing["pageHeight"] <= 900
    assert drawing["inside"]

    page.set_viewport_size({"width": 1100, "height": 650})
    smaller = _drawn_for_current_size(page)
    assert smaller["width"] == pytest.approx(1100 - 282, abs=2)
    assert smaller["pageHeight"] <= 650 and smaller["inside"]

    # A view the reader has zoomed stays as it is when the window changes,
    # until they reset the zoom.
    page.mouse.move(400, 300)
    page.mouse.wheel(0, -400)
    _wait_until_still(page)
    zoomed = page.evaluate(_DRAWING)["nodes"]
    page.set_viewport_size({"width": 1300, "height": 800})
    page.wait_for_timeout(400)
    _wait_until_still(page)
    assert page.evaluate(_DRAWING)["nodes"] == zoomed
    page.click("#conceptflow-reset-zoom")
    reset = _drawn_for_current_size(page)
    assert reset["width"] == pytest.approx(1300 - 282, abs=2) and reset["inside"]
    assert errors == []


def test_fixed_size_pages_are_centred(open_page):
    living = load_context("livingbeings_en")
    for figure in (
        plot_lattice(ConceptLattice.from_context(living), width=900, height=600),
        _nested(living, range(5), range(5, 9), width=900, height=600),
    ):
        page, errors = open_page(figure.html, viewport={"width": 1600, "height": 900})
        box = page.locator(".conceptflow-container").bounding_box()
        assert box["width"] == pytest.approx(900, abs=1)
        assert box["x"] == pytest.approx((1600 - 900) / 2, abs=1)
        drawing = _drawn_for_current_size(page)
        assert drawing["width"] == pytest.approx(900 - 282, abs=2)
        assert drawing["height"] == pytest.approx(600, abs=1)
        assert drawing["inside"] and errors == []


def test_narrow_windows_put_the_panel_below(open_page):
    for figure in _window_figures().values():
        page, errors = open_page(figure.html, viewport={"width": 420, "height": 800})
        drawing = _drawn_for_current_size(page)
        svg = page.locator("svg.conceptflow-drawing").bounding_box()
        panel = page.locator("#conceptflow-details").bounding_box()
        assert panel["y"] >= svg["y"] + svg["height"] - 1
        assert drawing["width"] == pytest.approx(420 - 2, abs=2)
        assert drawing["inside"] and errors == []


def _many_objects(n=300):
    """Contexts in which each concept introduces dozens of long object names."""
    rng = np.random.default_rng(1)
    names = [f"Object number {i} with a rather long name" for i in range(n)]
    field = np.eye(4, dtype=bool)[rng.integers(0, 4, n)]
    kind = np.eye(3, dtype=bool)[rng.integers(0, 3, n)]
    outer = FormalContext.from_array(field, objects=names,
                                     attributes=[f"field {k}" for k in range(4)])
    inner = FormalContext.from_array(kind, objects=names,
                                     attributes=[f"kind {k}" for k in range(3)])
    return outer, inner


_LABELS = """() => Object.fromEntries(['attr-label', 'obj-label', 'inner-attr'].map(c => [c,
    [...document.querySelectorAll('text.' + c)].map(t => ({
        text: t.textContent, shown: t.getAttribute('display') !== 'none' }))]))"""


def _check_label_buttons(page):
    labels = page.evaluate(_LABELS)
    objects = labels["obj-label"]
    assert objects and all(label["shown"] for label in objects)
    # At most 6 names, then how many more there are.
    for label in objects:
        names = label["text"].split(", ")
        assert len(names) <= 7
        if len(names) == 7:
            assert names[-1].startswith("+") and int(names[-1][1:]) > 0
    page.click("#conceptflow-object-labels")
    labels = page.evaluate(_LABELS)
    assert not any(label["shown"] for label in labels["obj-label"])
    assert all(label["shown"] for label in labels["attr-label"] + labels["inner-attr"])
    assert page.get_attribute("#conceptflow-object-labels", "aria-pressed") == "false"
    page.click("#conceptflow-attribute-labels")
    labels = page.evaluate(_LABELS)
    assert not any(label["shown"] for group in labels.values() for label in group)
    page.click("#conceptflow-object-labels")
    labels = page.evaluate(_LABELS)
    assert all(label["shown"] for label in labels["obj-label"])
    assert not any(label["shown"] for label in labels["attr-label"] + labels["inner-attr"])


def test_label_buttons_and_shortened_labels(open_page):
    outer, inner = _many_objects()
    page, errors = open_page(plot_lattice(ConceptLattice.from_context(outer)).html)
    _check_label_buttons(page)
    assert errors == []

    page, errors = open_page(plot_nested(outer, inner=inner).html, accept_downloads=True)
    _check_label_buttons(page)
    # Attribute labels are hidden now: the saved drawing leaves them out.
    with page.expect_download(timeout=15000) as download:
        page.click("#conceptflow-save-svg")
    saved = download.value.path().read_text(encoding="utf-8")
    assert 'class="obj-label"' in saved and 'class="attr-label"' not in saved
    # The labels are shortened; the details list every object.
    page.click("#conceptflow-attribute-labels")
    first = page.evaluate(_LABELS)["obj-label"][0]["text"]
    assert first.split(", ")[-1].startswith("+")
    assert errors == []


def test_label_options_set_the_start(open_page):
    outer, inner = _many_objects(40)
    page, errors = open_page(
        plot_nested(outer, inner=inner, show_object_labels=False, max_label_items=2).html
    )
    labels = page.evaluate(_LABELS)
    assert labels["obj-label"] and not any(label["shown"] for label in labels["obj-label"])
    assert all(len(label["text"].split(", ")) <= 3 for label in labels["obj-label"])
    assert page.get_attribute("#conceptflow-object-labels", "aria-pressed") == "false"
    assert errors == []
