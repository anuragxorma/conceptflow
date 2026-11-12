"""
Argument checks and page layout shared by the renderers.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral, Real

from sklearn.utils import check_scalar

# Width of the details panel in the interactive pages; it sits beside the
# drawing from PANEL_BESIDE_FROM pixels on, below it in narrower pages.
DETAILS_PANEL_WIDTH = 280
PANEL_BESIDE_FROM = 560

# Height of the drawing with height="auto" in a notebook, whose frame grows to
# the page (so the page cannot fill the frame), and the least height of a
# drawing that fills a small window.
NOTEBOOK_DRAWING_HEIGHT = 700
MIN_DRAWING_HEIGHT = 300

# Attribute on <html> that HTMLFigure._repr_html_ sets: the page is notebook
# output. Saved pages fill the window, also in previews that show them in a
# frame (VS Code, JupyterLab).
NOTEBOOK_MARKER = "data-conceptflow-notebook"

_BESIDE = (
    ".conceptflow-body {{ grid-template-columns: minmax(0, 1fr) {panel}px; "
    "height: var(--drawing-height); }}\n"
    "  .conceptflow-details {{ border-left: 1px solid #ddd; }}"
).format(panel=DETAILS_PANEL_WIDTH)

_BELOW = (
    ".conceptflow-body { grid-template-columns: minmax(0, 1fr); "
    "grid-template-rows: var(--drawing-height) auto; height: auto; }\n"
    "  .conceptflow-details { border-left: none; border-top: 1px solid #ddd; }"
)

# Keeps the drawing at the size of its area: with height="auto" in a window
# (not in a notebook), the drawing fills the window below the header. The
# page's draw(size) is called when the area has a size, and again when it
# changes, unless the reader has zoomed or panned (sizer.held): then their
# view is kept until they reset the zoom.
SIZE_SCRIPT = """
function conceptflowSizer(svgNode, autoHeight, draw) {
  const container = svgNode.closest(".conceptflow-container");
  const header = container.querySelector(".conceptflow-header");
  const notebook = document.documentElement.hasAttribute("%(marker)s");
  const fillWindow = autoHeight && !notebook;
  let drawnSize = null;
  let timer = null;

  function fitHeight() {
    if (!fillWindow) return;
    const borders = container.offsetHeight - container.clientHeight;
    const height = Math.max(
      %(min_height)d, Math.floor(window.innerHeight - header.offsetHeight - borders)
    );
    container.style.setProperty("--drawing-height", height + "px");
  }

  function measure() {
    const box = svgNode.getBoundingClientRect();
    return { width: box.width, height: box.height };
  }

  const sizer = {
    held: false,
    // Draw the current view now, fitted to the drawing area.
    draw() {
      fitHeight();
      const size = measure();
      if (size.width < 1 || size.height < 1) {
        drawnSize = null;  // hidden: drawn once it has a size
        return;
      }
      drawnSize = size;
      svgNode.setAttribute("width", Math.round(size.width));
      svgNode.setAttribute("height", Math.round(size.height));
      draw(size);
    },
    // Draw again if the area has changed and the reader has not moved the view.
    resized() {
      fitHeight();
      const size = measure();
      if (sizer.held || size.width < 1 || size.height < 1) return;
      if (drawnSize && Math.abs(size.width - drawnSize.width) < 0.5
          && Math.abs(size.height - drawnSize.height) < 0.5) return;
      sizer.draw();
    },
    size() {
      return drawnSize || measure();
    },
  };

  window.addEventListener("resize", fitHeight);
  if (window.ResizeObserver) {
    new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(sizer.resized, 100);
    }).observe(svgNode);
  }
  return sizer;
}
""" % {"marker": NOTEBOOK_MARKER, "min_height": MIN_DRAWING_HEIGHT}


def check_page_size(value, name: str) -> None:
    """``"auto"`` or a positive, finite number of pixels."""
    if isinstance(value, str):
        if value != "auto":
            raise TypeError(f'{name} must be "auto" or a number of pixels, got {value!r}.')
        return
    if isinstance(value, bool):
        raise TypeError(f'{name} must be "auto" or a number of pixels, got {value!r}.')
    check_scalar(value, name, Real, min_val=0, include_boundaries="neither")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of pixels, got {value!r}.")


def html_page_layout(width, height) -> dict:
    """
    Check the size of an interactive page and return its CSS.

    ``width="auto"`` fills the window or notebook cell; ``height="auto"``
    fills the window below the header (the page's script sets it), and is
    700 px in a notebook. A number fixes that size in pixels; a fixed width
    includes the details panel, which moves below the drawing in pages
    narrower than 560 px. Returns ``css``, ``auto_height`` and
    ``figure_height`` (the drawing height a notebook frame starts with).
    """
    check_page_size(width, "width")
    check_page_size(height, "height")
    auto_height = isinstance(height, str)
    drawing_height = NOTEBOOK_DRAWING_HEIGHT if auto_height else height

    if isinstance(width, str):
        container = "width: 100%;"
        layout = (
            f"{_BESIDE}\n"
            f"  @media (max-width: {PANEL_BESIDE_FROM - 0.02}px) {{\n  {_BELOW}\n  }}"
        )
    else:
        container = f"width: {width}px; max-width: 100%; margin: 0 auto;"
        layout = _BESIDE if width >= PANEL_BESIDE_FROM else _BELOW

    css = (
        f".conceptflow-container {{ {container} box-sizing: border-box; "
        f"--drawing-height: {drawing_height}px; }}\n"
        f"  .conceptflow-drawing {{ display: block; width: 100%; height: 100%; }}\n"
        f"  {layout}"
    )
    return {"css": css, "auto_height": auto_height, "figure_height": drawing_height}


def check_max_label_items(max_label_items) -> None:
    """``None`` (show every label) or how many to show before "+n"."""
    if max_label_items is not None:
        check_scalar(max_label_items, "max_label_items", Integral, min_val=0)


def object_formatter(object_label: Callable | None) -> Callable[[str], str]:
    """``object_label`` with its result turned into text, as every backend needs."""
    if object_label is None:
        return str
    return lambda name: str(object_label(name))
