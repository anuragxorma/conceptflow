import warnings

from conceptflow.cluster.concept_lattice import ConceptLatticeEstimator


class ConceptLattice(ConceptLatticeEstimator):
    """
    Deprecated. Use ``ConceptLatticeEstimator`` instead.
    """

    def __init__(self, **kwargs):
        warnings.warn(
            "conceptflow.cluster.ConceptLattice has been renamed to "
            "ConceptLatticeEstimator. "
            "The old name will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


__all__ = ["ConceptLatticeEstimator", "ConceptLattice"]
