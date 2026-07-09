import numpy as np

from conceptflow.core.lattice import ConceptLattice

from conceptflow.visualization.dimflux.src.utils.variables import Variables
from conceptflow.visualization.dimflux.src.dim_flux.dim_draw import DimDraw
from conceptflow.visualization.dimflux.src.dim_flux.projection import Projection
from conceptflow.visualization.dimflux.src.dim_flux.lgs import LinearEquationSolver
from conceptflow.visualization.dimflux.src.dim_flux.additive_realizer import AdditiveRealizer


class Realizer():
    '''
    Disclaimer
    ----------
    The original version is integrated into the tool conexp-clj
    [see https://github.com/tomhanika/conexp-clj].

    Reference
    ---------
    @misc{dürrschnabel2019dimdrawnoveltool,
        title={DimDraw -- A novel tool for drawing concept lattices},
        author={Dominik Dürrschnabel and Tom Hanika and Gerd Stumme},
        year={2019},
        eprint={1903.00686},
        archivePrefix={arXiv},
        primaryClass={cs.CG},
        url={https://arxiv.org/abs/1903.00686}
    }
    '''
    def __init__(self,
            variables: Variables
        ):
        self.vars = variables
        self.context = variables.context
        self.lattice = ConceptLattice.from_context(variables.context)

        # check whether an additive 2D realizer exists
        additive_realizer = AdditiveRealizer(self.context)
        self.realizer = additive_realizer.realizer()
        self.base_vectors = additive_realizer.base_vectors

        if self.realizer:
            self._store_coordinates()
        else:
            dim_draw = DimDraw(self.vars)
            self.realizer = dim_draw.two_dimensional_extension()
            self._store_coordinates()
            projection = Projection(self.vars)
            self.vars.coordinates = projection.coordinates
            self._derive_base_vectors()

    def _store_coordinates(self):
        '''
        Store coordinates after turning the diagram by 45 degrees to the left
        and stretching the diagram horizontally by sqrt(2) and squeezing the
        diagram vertically by 1/sqrt(2).

        Rotating, stretching and squeezing are linear transformations and
        therefore do not change the status of a diagram of being additive or
        not.
        '''
        coords_array = np.array([
            [self.realizer[0].index(c), self.realizer[1].index(c)]
            for c in self.vars.concepts
        ])

        theta = np.radians(45)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        rotated_coords = (coords_array @ R.T) * np.array([np.sqrt(2), 1 / np.sqrt(2)])

        self.coordinates = {
            concept: rotated_coords[i].tolist()
            for i, concept in enumerate(self.vars.concepts)
        }
        self.vars.coordinates = self.coordinates

    def _derive_base_vectors(self):
        '''
        Derive base vectors by solving the system of linear equations.
        '''
        lgs = LinearEquationSolver(self.vars, self.coordinates)
        success, vector_vars = lgs.solve_linear_equations()

        if success:
            self.base_vectors = dict({})
            for v in self.vars.elements:
                if v in self.vars.G:
                    self.base_vectors[v] = np.array(
                        [vector_vars[f'x_{v}'], vector_vars[f'y_{v}']]
                    )
                else:
                    self.base_vectors[v] = np.array(
                        [-vector_vars[f'x_{v}'], -vector_vars[f'y_{v}']]
                    )