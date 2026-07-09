import conceptflow as cf


def test_top_level_public_api_exports_core_objects():
    assert hasattr(cf, "__version__")
    assert hasattr(cf, "Concept")
    assert hasattr(cf, "FormalContext")
    assert hasattr(cf, "ManyValuedContext")
    assert hasattr(cf, "ConceptLattice")
    assert hasattr(cf, "GraphNode")
    assert hasattr(cf, "GraphEdge")
    assert hasattr(cf, "GraphData")
    assert hasattr(cf, "lattice_to_graph_data")
    assert hasattr(cf, "plot_lattice")
    assert hasattr(cf, "ExplorationBuilder")
    assert hasattr(cf, "ExplorationView")


def test_preprocessing_public_api_imports():
    from conceptflow.preprocessing import (
        ConceptualScaler,
        ContranominalScale,
        DichotomicScale,
        GeneralScale,
        InterordinalScale,
        NominalScale,
        OrdinalScale,
        Scale,
        ThresholdScale,
    )

    assert ConceptualScaler is not None
    assert Scale is not None
    assert NominalScale is not None
    assert ContranominalScale is not None
    assert DichotomicScale is not None
    assert OrdinalScale is not None
    assert ThresholdScale is not None
    assert InterordinalScale is not None
    assert GeneralScale is not None


def test_cluster_public_api_imports():
    from conceptflow.cluster import ConceptLattice as ClusterConceptLattice

    assert ClusterConceptLattice is not None


def test_feature_extraction_public_api_imports():
    from conceptflow.feature_extraction import ConceptMembershipEncoder

    assert ConceptMembershipEncoder is not None


def test_decomposition_public_api_imports():
    from conceptflow.decomposition import (
        ExactOrdinalTwoFactorizer,
        OrdinalFactor,
        OrdinalTwoFactorization,
        bipartite_coloring,
        complement_context,
        incompatibility_graph,
        incidence_pairs,
        is_ferrers_relation,
    )
    from conceptflow.decomposition.ordinal_factorization import Ord2Factor

    assert ExactOrdinalTwoFactorizer is not None
    assert Ord2Factor is not None
    assert OrdinalFactor is not None
    assert OrdinalTwoFactorization is not None
    assert incidence_pairs is not None
    assert is_ferrers_relation is not None
    assert incompatibility_graph is not None
    assert bipartite_coloring is not None
    assert complement_context is not None


def test_io_public_api_imports():
    from conceptflow.io import read_cxt, write_cxt

    assert read_cxt is not None
    assert write_cxt is not None


def test_metrics_public_api_imports():
    from conceptflow.metrics import (
        attribute_set_support,
        concept_support,
        implication_confidence,
    )

    assert concept_support is not None
    assert attribute_set_support is not None
    assert implication_confidence is not None


def test_algorithms_public_api_imports():
    from conceptflow.algorithms import (
        SUPPORTED_ENUMERATION_ALGORITHMS,
        attribute_closure,
        attribute_derivation,
        compute_hasse_edges,
        enumerate_concepts,
        enumerate_concepts_bruteforce,
        enumerate_concepts_closebyone,
        enumerate_concepts_nextclosure,
        is_cover,
        normalize_enumeration_algorithm,
        object_derivation,
        strict_subconcept_of,
        subconcept_of,
    )

    assert object_derivation is not None
    assert attribute_derivation is not None
    assert attribute_closure is not None
    assert enumerate_concepts is not None
    assert enumerate_concepts_bruteforce is not None
    assert enumerate_concepts_nextclosure is not None
    assert enumerate_concepts_closebyone is not None
    assert subconcept_of is not None
    assert strict_subconcept_of is not None
    assert is_cover is not None
    assert compute_hasse_edges is not None
    assert normalize_enumeration_algorithm is not None
    assert "nextclosure" in SUPPORTED_ENUMERATION_ALGORITHMS
    assert "closebyone" in SUPPORTED_ENUMERATION_ALGORITHMS
    assert "bruteforce" in SUPPORTED_ENUMERATION_ALGORITHMS


def test_validation_public_api_imports():
    from conceptflow.validation import check_binary_context_input

    assert check_binary_context_input is not None

def test_visualization_public_api_imports():
    from conceptflow.visualization import (
        GraphData,
        GraphEdge,
        GraphNode,
        HTMLFigure,
        exploration_view_to_nested_data,
        lattice_to_graph_data,
        lattice_to_graph_data_dimflux,
        plot_lattice,
        plot_nested,
        render_nested_html,
        graph_data_to_d3_data,
        render_graph_data_html,
        render_with_d3,
    )

    assert GraphNode is not None
    assert GraphEdge is not None
    assert GraphData is not None
    assert HTMLFigure is not None
    assert lattice_to_graph_data is not None
    assert lattice_to_graph_data_dimflux is not None
    assert exploration_view_to_nested_data is not None
    assert render_nested_html is not None
    assert plot_lattice is not None
    assert plot_nested is not None
    assert graph_data_to_d3_data is not None
    assert render_graph_data_html is not None
    assert render_with_d3 is not None
    
def test_exploration_public_api_imports():
    from conceptflow.exploration import (
        ExplorationBuilder,
        ExplorationView,
    )

    assert ExplorationBuilder is not None
    assert ExplorationView is not None