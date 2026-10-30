"""
Argument checks shared by the renderers.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real

from sklearn.utils import check_scalar

# Width of the details panel in the interactive pages; it sits beside the
# drawing from PANEL_BESIDE_FROM pixels on, below it in narrower pages.
DETAILS_PANEL_WIDTH = 280
PANEL_BESIDE_FROM = 560


def html_page_layout(width, height) -> dict:
    """
    Check the size of an interactive page and split it into drawing and
    details panel: the drawing width and the CSS for the panel's place.
    """
    check_scalar(width, "width", Real, min_val=0, include_boundaries="neither")
    check_scalar(height, "height", Real, min_val=0, include_boundaries="neither")
    if width >= PANEL_BESIDE_FROM:
        return {
            "drawing_width": width - DETAILS_PANEL_WIDTH,
            "columns": f"1fr {DETAILS_PANEL_WIDTH}px",
            "panel_border": "border-left",
        }
    return {"drawing_width": width, "columns": "1fr", "panel_border": "border-top"}


def check_max_label_items(max_label_items) -> None:
    """``None`` (show every label) or how many to show before "+n"."""
    if max_label_items is not None:
        check_scalar(max_label_items, "max_label_items", Integral, min_val=0)


def object_formatter(object_label: Callable | None) -> Callable[[str], str]:
    """``object_label`` with its result turned into text, as every backend needs."""
    if object_label is None:
        return str
    return lambda name: str(object_label(name))
