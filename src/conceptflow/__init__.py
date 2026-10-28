"""
ConceptFlow.

A scikit-learn compatible Formal Concept Analysis framework.
"""

from conceptflow._version import __version__
from conceptflow.core import (
    Concept,
    ConceptLattice,
    FormalContext,
    IncompleteContext,
    ManyValuedContext,
)
from conceptflow.exploration import (
    ExplorationBuilder,
    ExplorationView,
)
from conceptflow.visualization import (
    GraphData,
    GraphEdge,
    GraphNode,
    lattice_to_graph_data,
    plot_lattice,
)

__all__ = [
    "__version__",
    "Concept",
    "FormalContext",
    "IncompleteContext",
    "ManyValuedContext",
    "ConceptLattice",
    "GraphData",
    "GraphEdge",
    "GraphNode",
    "lattice_to_graph_data",
    "plot_lattice",
    "ExplorationBuilder",
    "ExplorationView",
]
