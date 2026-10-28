from conceptflow._deprecation import warn_deprecated
from conceptflow.cluster.concept_lattice import ConceptLatticeEstimator


class ConceptLattice(ConceptLatticeEstimator):
    """
    Deprecated. Use ``ConceptLatticeEstimator`` instead; removed in 1.0.
    """

    def __init__(self, **kwargs):
        warn_deprecated(
            "conceptflow.cluster.ConceptLattice",
            "conceptflow.cluster.ConceptLatticeEstimator",
            removal="1.0",
        )
        super().__init__(**kwargs)


__all__ = ["ConceptLatticeEstimator", "ConceptLattice"]
