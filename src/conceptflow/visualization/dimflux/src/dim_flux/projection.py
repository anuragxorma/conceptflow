import numpy as np

from conceptflow.visualization.dimflux.src.utils.variables import Variables


class Projection():
    def __init__(self,
            variables: Variables
        ):
        '''
        Project a line diagram into the space of additive line diagrams.

        Parameters
        ----------
        variables: Variables
            The storage of variables
        '''
        self.vars = variables
        self.lectic_order = self._compute_lectic_order()
        self._set_representation()
        self._orthonormal_basis()
        self._additive_coordinates()

    def _lectically_smaller(self, intent_a: set, intent_b: set) -> bool:
        r'''
        Check if concept A is lectically smaller than concept B.

        A <L B iff there exists an attribute m in M such that:
            - m is the smallest attribute in (intent_b - intent_a) \cup (intent_a - intent_b)
            - m \in intent_b (B contains it, A does not)

        Parameters
        ----------
        intent_a : set
            Intent of concept A
        intent_b : set
            Intent of concept B

        Returns
        -------
        bool
            True if A is lectically smaller than B
        '''
        if intent_a == intent_b:
            return False

        for m in self.vars.attributes:
            if m in intent_a and m in intent_b:
                continue
            if m not in intent_a and m not in intent_b:
                continue
            # m is the smallest differing element
            # A <L B iff A does NOT contain m
            return m not in intent_a

        return False

    def _compute_lectic_order(self) -> list:
        '''
        Sort all concepts by the lectic order on their intents.

        Returns
        -------
        list
            Concepts sorted lectically
        '''
        concepts = list(self.vars.concepts)

        for i in range(1, len(concepts)):
            key = concepts[i]
            key_intent = self.vars.intents[key]
            j = i - 1
            while j >= 0 and self._lectically_smaller(key_intent, self.vars.intents[concepts[j]]):
                concepts[j + 1] = concepts[j]
                j -= 1
            concepts[j + 1] = key

        return concepts

    def _set_representation(self):
        '''
        Compute the Set Representation Matrix (SRM).

        Each row corresponds to a concept in lectic order. Each column
        corresponds to an element (object or attribute) in the order defined
        by objects + attributes.

        An element is flagged (1) if it is:
            - an object belonging to the extent of c, or
            - an attribute NOT belonging to the intent of c.
        '''
        self.srm = np.array([
            [
                1 if e in (self.vars.extents[c] | (self.vars.M - self.vars.intents[c])) else 0
                for e in self.vars.elements
            ]
            for c in self.lectic_order
        ])

    def _orthonormal_basis(self):
        '''
        Compute an orthonormal basis for the column space of the SRM.

        Applies Gram-Schmidt orthogonalisation column-wise to self.srm,
        discarding linearly dependent columns (norm below 1e-5).

        Sets
        ----
        self.basis : np.ndarray of shape (N_c, k)
            Matrix whose columns form an orthonormal basis for the column
            space of self.srm, where k <= N_e is the rank of the SRM.
        '''
        self.basis = np.zeros((len(self.srm), 0))
        for srm_col_n in range(self.srm.shape[1]):
            projection = np.zeros((len(self.srm), 1))
            srm_col = self.srm[:, srm_col_n:srm_col_n + 1]
            if self.basis.shape[1] > 0:
                for bcolnum in range(self.basis.shape[1]):
                    bcol = self.basis[:, bcolnum:bcolnum + 1]
                    projection += np.dot(srm_col.T, bcol) * bcol
            newcol = srm_col - projection
            norm = np.linalg.norm(newcol)
            if norm > 1.e-05:
                newcol = newcol / norm
                self.basis = np.column_stack((self.basis, newcol))

    def _additive_coordinates(self):
        '''
        Compute the closest additive placement to the current coordinates.
        '''
        # (N_c, 2) position matrix in lectic order
        xy = np.array([self.vars.coordinates[c] for c in self.lectic_order])
        # project onto column space of SRM
        self.projected_xy = self.basis @ (self.basis.T @ xy)
        self.coordinates = {}
        for i, c in enumerate(self.lectic_order):
            self.coordinates[c] = self.projected_xy[i]