import networkx as nx

from typing import Tuple, Set, Dict
from collections import deque

from conceptflow.core.concept import Concept
from conceptflow.core.lattice import ConceptLattice


def cover_relations(concept_lattice: ConceptLattice) -> Set[Tuple[Concept, Concept]]:
    '''
    Get the cover relations of a concept lattice.

    Parameters
    ----------
    concept_lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    cover_relations : Set[Tuple[Concept, Concept]]
        A set of tuples representing the cover relations of the lattice.
    '''
    return set(nx.transitive_reduction(concept_lattice.to_networkx()).edges)


def all_extents(
        lattice: ConceptLattice
    ) -> Dict[Concept, Set[str]]:
    '''
    Compute the extents for all concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice

    Returns
    -------
    extents: Dict[Concept, Set[str]]
        A dictionary mapping concepts to their full extents
    '''
    queue = deque({lattice.bottom()})
    extents: Dict[Concept, Set[str]] = dict({})

    while queue:
        concept = queue.popleft()

        # all children processed?
        if lattice.children(concept) <= extents.keys():
            # new_extent ∪ extents(children)
            child_extents = [extents[c] for c in lattice.children(concept)]
            extents[concept] = set(concept.extent)

            if child_extents:
                # keep the structure conceptually close to the original
                extents[concept] = set().union(*child_extents, set(concept.extent))

            # add parents to queue
            queue.extend(lattice.parents(concept) - extents.keys())

        # re-add to queue
        else:
            queue.append(concept)

    return extents


def all_intents(
        lattice: ConceptLattice
    ) -> Dict[Concept, Set[str]]:
    '''
    Compute the intents for all concepts in the lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice

    Returns
    -------
    intents: Dict[Concept, Set[str]]
        A dictionary mapping concepts to their full intents
    '''
    queue = deque({lattice.top()})
    intents: Dict[Concept, Set[str]] = dict({})

    while queue:
        concept = queue.popleft()

        # all parents processed?
        if lattice.parents(concept) <= intents.keys():
            # new_intent ∪ intents(parents)
            parent_intents = [intents[p] for p in lattice.parents(concept)]
            intents[concept] = set(concept.intent)

            if parent_intents:
                # keep the structure conceptually close to the original
                intents[concept] = set().union(*parent_intents, set(concept.intent))

            # add children to queue
            queue.extend(lattice.children(concept) - intents.keys())

        # re-add to queue
        else:
            queue.append(concept)

    return intents


def incomparability_graph(lattice: ConceptLattice) -> nx.Graph:
    '''
    Get the incomparability graph of a concept lattice.

    Parameters
    ----------
    lattice : ConceptLattice
        The concept lattice.

    Returns
    -------
    incomparability_graph : nx.Graph
        The incomparability graph of the lattice.
    '''
    return nx.complement(nx.transitive_closure(lattice.to_networkx()).to_undirected())