import numpy as np

from typing import Optional, Dict
from dataclasses import dataclass

from conceptflow.core.context import FormalContext
from conceptflow.core.lattice import ConceptLattice

from conceptflow.visualization.dimflux.src.fca.context import *
from conceptflow.visualization.dimflux.src.fca.lattice import *


@dataclass
class Args:
    plot_si_graph: bool = False
    si_graph_annotations: bool = False
    plot_initial_layout: bool = False
    initial_layout_annotations: bool = False
    plot_optimized_layout: bool = False
    optimized_layout_annotations: bool = False
    plot_individual_forces: bool = False
    plot_combined_forces: bool = False
    plot_gradients: bool = False
    plot_origin: bool = False


class Variables():
    '''
    A container class to store the formal context, concept lattice,
    and parameters required for lattice drawing and force-directed layouts.

    Parameters
    ----------
    cxt : str
        The filename or path of the formal context.
    context : FormalContext
        The formal context decoded from the input file.
    lattice : ConceptLattice
        The concept lattice derived from the context.
    args : Args
        Configuration object for visualization settings.
    objects : list[str]
        List of object names in the context.
    object_map : dict[str, int]
        Mapping of object names to their integer indices.
    object_closures : dict[str, set[str]]
        Mapping of objects to their closure sets.
    N_g : int
        The number of objects in the context.
    G : set[str]
        The set of all object names.
    attributes : list[str]
        List of attribute names in the context.
    attribute_map : dict[str, int]
        Mapping of attribute names to their integer indices.
    attribute_closures : dict[str, set[str]]
        Mapping of attributes to their closure sets.
    N_m : int
        The number of attributes in the context.
    M : set[str]
        The set of all attribute names.
    elements : list[str]
        Concatenated list of objects and attributes.
    element_map : dict[str, int]
        Mapping of element names to their integer indices.
    N_e : int
        Total number of elements (objects + attributes).
    E : set[str]
        The set of all elements.
    concepts : list
        List of ConceptFlow concept objects from the lattice.
    N_c : int
        Total number of concepts in the lattice.
    extents : dict
        Mapping of concepts to their extents.
    intents : dict
        Mapping of concepts to their intents.
    w_rep : float
        Repulsive force weight.
    w_att : float
        Attractive force weight.
    w_grav : float
        Gravitational force weight.
    order : list
        The processing order for layout optimization.
    scalars : np.ndarray
        Array of scalar values associated with each vector.
    d_si_points : list
        Points used for sup-inf graph calculations.
    n_1 : int
        Left point of f_max.
    n_2 : int
        Right point of f_max.
    base_vectors : dict
        Dictionary storing the base vectors for elements.
    coordinates : dict
        Dictionary mapping concepts to their computed coordinates.
    final_forces : dict
        Dictionary storing the resultant forces after optimization.
    '''

    def __init__(self, cxt: str, args: Optional[Dict[str, bool]]):

        self.cxt = cxt

        if cxt.endswith('.cxt'):
            self.context = FormalContext.from_cxt(cxt)
        else:
            self.context = FormalContext.from_cxt(f'data/{cxt}.cxt')

        self.lattice = ConceptLattice.from_context(self.context)
        self.args: Args = Args(**(args or {}))

        # objects
        self.objects = list(self.context.objects)
        self.object_map = {
            g: i
            for i, g in enumerate(self.objects)
        }
        self.object_closures = {
            g: object_closure(self.context, {g})
            for g in self.objects
        }
        self.N_g = self.context.n_objects
        self.G = set(self.objects)

        # attributes
        self.attributes = list(self.context.attributes)
        self.attribute_map = {
            m: i
            for i, m in enumerate(self.attributes)
        }
        self.attribute_closures = {
            m: attribute_closure(self.context, {m})
            for m in self.attributes
        }
        self.N_m = self.context.n_attributes
        self.M = set(self.attributes)

        # elements
        self.elements = self.objects + self.attributes
        self.element_map = {
            v: i
            for i, v in enumerate(self.elements)
        }
        self.N_e = self.N_g + self.N_m
        self.E = set(self.elements)

        # concepts
        self.concepts = list(self.lattice.concepts)
        self.N_c = len(self.concepts)
        raw_extents = all_extents(self.lattice)
        raw_intents = all_intents(self.lattice)

        self.extents = {
            concept: {
                self.objects[index] if isinstance(index, int) else index
                for index in extent
            }
            for concept, extent in raw_extents.items()
        }

        self.intents = {
            concept: {
                self.attributes[index] if isinstance(index, int) else index
                for index in intent
            }
            for concept, intent in raw_intents.items()
}

        # weights
        self.w_rep = 50.0
        self.w_att = 1.0
        self.w_grav = 30.0

        # global variables
        self.order = []
        self.scalars = np.zeros(self.N_e)
        self.d_si_points = []
        self.n_1 = 0
        self.n_2 = 0
        self.base_vectors = dict({})
        self.coordinates = dict({})
        self.final_forces = dict({})