from z3 import *

from conceptflow.core.context import FormalContext
from conceptflow.core.lattice import ConceptLattice

from conceptflow.visualization.dimflux.src.fca.lattice import *


class AdditiveRealizer:
    """
    Compute an additive realizer for a given concept lattice.

    This version uses ConceptFlow concepts as graph nodes, but uses safe
    internal variable names for Z3. Object and attribute labels may contain
    spaces, punctuation, or other characters, so they should not be used
    directly in SMT variable names.
    """

    def __init__(self, context: FormalContext):
        self.context = context
        self.lattice = ConceptLattice.from_context(context)
        self.G = self.lattice.to_networkx()

        self.concepts = list(self.G.nodes)

        self.concept_to_id = {
            concept: i
            for i, concept in enumerate(self.concepts)
        }
        self.id_to_concept = {
            i: concept
            for concept, i in self.concept_to_id.items()
        }

        self.relations = {
            (self.concept_to_id[lower], self.concept_to_id[upper])
            for lower, upper in self.G.edges
        }

        self.bottom_id = self.concept_to_id[self.lattice.bottom()]
        self.top_id = self.concept_to_id[self.lattice.top()]
        self.max_rank = len(self.concepts) - 1

        self.objects = tuple(context.objects)
        self.features = tuple(context.attributes)

        self.object_to_var = {
            obj: f"g_{index}"
            for index, obj in enumerate(self.objects)
        }
        self.feature_to_var = {
            attr: f"m_{index}"
            for index, attr in enumerate(self.features)
        }

        self.incomparable_pairs = {
            (self.concept_to_id[a], self.concept_to_id[b])
            for a, b in incomparability_graph(self.lattice).edges
        }

        self.extents = {
            self.concept_to_id[concept]: set(concept.extent)
            for concept in self.concepts
        }
        self.intents = {
            self.concept_to_id[concept]: set(concept.intent)
            for concept in self.concepts
        }

        self.base_vectors = {}

        self.solver = Solver()
        # Adaptive timeout: more objects require more time even with few attributes.
        # 200ms/object + 500ms/attr, clamped to [3s, 30s].
        n_elements = len(context.objects) + len(context.attributes)
        timeout_ms = min(30000, max(3000, n_elements * 200))
        self.solver.set("timeout", timeout_ms)
        self.dimension = 2
        self.dimensions = ["x", "y"]

        self._setup_smt_variables()
        self._setup_relations()

    def realizer(self):
        """
        Compute an additive realizer using the z3 SMT solver.

        Returns
        -------
        list
            2D additive realizer as two linear extensions, or an empty list
            if none exists.
        """
        if self.solver.check() != sat:
            return []

        self.model = self.solver.model()

        realizer = {
            d: [None for _ in self.concepts]
            for d in self.dimensions
        }

        for obj in self.objects:
            var_name = self.object_to_var[obj]

            self.base_vectors[obj] = [
                float(
                    self.model.eval(
                        Real(f"x_{var_name}"),
                        model_completion=True,
                    ).as_fraction()
                ),
                float(
                    self.model.eval(
                        Real(f"y_{var_name}"),
                        model_completion=True,
                    ).as_fraction()
                ),
            ]

        for attr in self.features:
            var_name = self.feature_to_var[attr]

            self.base_vectors[attr] = [
                float(
                    -self.model.eval(
                        Real(f"x_{var_name}"),
                        model_completion=True,
                    ).as_fraction()
                ),
                float(
                    -self.model.eval(
                        Real(f"y_{var_name}"),
                        model_completion=True,
                    ).as_fraction()
                ),
            ]

        for d in self.dimensions:
            for concept_id in self.id_to_concept:
                pos = self.model.eval(
                    Int(f"{d}_{concept_id}"),
                    model_completion=True,
                ).as_long()

                realizer[d][pos] = self.id_to_concept[concept_id]

        return [le for le in realizer.values()]

    def _setup_smt_variables(self):
        """
        Define SMT variables for all concepts and base vectors.
        """
        variable_names = (
            list(self.object_to_var.values())
            + list(self.feature_to_var.values())
        )

        self.smt_variables = {
            (d, var_name): Real(f"{d}_{var_name}")
            for d in self.dimensions
            for var_name in variable_names
        }

        for d in self.dimensions:
            for concept_id in self.id_to_concept:
                vec_G = (
                    self.smt_variables[d, self.object_to_var[self.objects[index]]]
                    for index in self.extents[concept_id]
                )

                vec_M = (
                    self.smt_variables[d, self.feature_to_var[self.features[index]]]
                    for index in range(len(self.features))
                    if index not in self.intents[concept_id]
                )

                self.solver.add(
                    Int(f"{d}_{concept_id}") == sum(vec_G) + sum(vec_M)
                )

    def _setup_relations(self):
        """
        Define SMT clauses for additivity.
        """
        for lower_id, upper_id in self.relations:
            for d in self.dimensions:
                self.solver.add(
                    Int(f"{d}_{upper_id}") > Int(f"{d}_{lower_id}")
                )

        for a_id, b_id in self.incomparable_pairs:
            a_vars = [Int(f"{d}_{a_id}") for d in self.dimensions]
            b_vars = [Int(f"{d}_{b_id}") for d in self.dimensions]

            a_lt_b = [
                a_vars[i] < b_vars[i]
                for i in range(self.dimension)
            ]
            a_gt_b = [
                a_vars[i] > b_vars[i]
                for i in range(self.dimension)
            ]

            self.solver.add(And(Or(*a_lt_b), Or(*a_gt_b)))

            for d in self.dimensions:
                self.solver.add(Int(f"{d}_{a_id}") != Int(f"{d}_{b_id}"))

        for d in self.dimensions:
            self.solver.add(Int(f"{d}_{self.bottom_id}") == 0)
            self.solver.add(Int(f"{d}_{self.top_id}") == self.max_rank)