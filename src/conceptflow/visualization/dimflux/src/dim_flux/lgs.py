import numpy as np

from typing import Dict
from collections import deque
from sympy import symbols, Eq, solve, sympify, linear_eq_to_matrix

from conceptflow.visualization.dimflux.src.utils.variables import Variables


class LinearEquationSolver:
    '''
    Solve a system of linear equations derived from a concept lattice structure.

    Parameters
    ----------
    variables : Variables
        The underlying concept lattice.
    coordinates : Dict
        A mapping from concepts to their coordinates.
    '''
    def __init__(self,
            variables: Variables,
            coordinates: Dict
        ):
        self.vars = variables
        self.coordinates = coordinates
        self.dimensions = ['x', 'y']

        self.variables = [
            f'{dim}_{v}'
            for dim in self.dimensions
            for v in self.vars.objects + self.vars.attributes
        ]
        self.symbols = symbols(' '.join(self.variables))

        self._construct_equations()
        self.solution = solve(self.equations, self.symbols, dict=True)
        if not self.solution:
            self._solve_approximate()

    def _solve_approximate(self):
        A, b = linear_eq_to_matrix(self.equations, self.symbols)
        sol_matrix = A.pinv() * b
        self.solution = [{self.symbols[i]: sol_matrix[i] for i in range(len(self.symbols))}]

    def _construct_equations(self):
        self.equations = []

        for c in self.vars.concepts:
            elements = self.vars.extents[c] | (self.vars.M - self.vars.intents[c])
            for i, dim in enumerate(self.dimensions):
                l, r = tuple((
                    (' + '.join(f'{dim}_{v}' for v in elements) if elements else '0'),
                    f'{self.coordinates[c][i]}'
                ))
                if f'{l} = {r}' != '0 = 0.0':
                    self.equations.append(Eq(sympify(l), sympify(r)))

    def _solve(self, dim: str, node, expected: int):
        '''
        Solve the equation for a specific node and dimension.

        Parameters
        ----------
        dim : str
            The dimension for which to solve the equation.
        node :
            The lattice concept.
        expected : int
            The expected value for the equation.
        '''
        eq = []

        for var in self.vars.extents[node] | (self.vars.M - self.vars.intents[node]):
            var = f'{dim}_{var}'

            if var in self.vector_variables:
                eq.append(str(self.vector_variables[var]))
            elif symbols(var) in self.free_vars:
                eq.append(str(var))
            else:
                sub_eq = str(self.solution[0][symbols(var)])

                for k, v in self.vector_variables.items():
                    sub_eq = sub_eq.replace(str(k), str(v))

                if not any(var_name in sub_eq for var_name in self.variables):
                    self.vector_variables[var] = float(eval(sub_eq, {"__builtins__": None}))
                    sub_eq = str(self.vector_variables[var])

                eq.append(f'({sub_eq})')

        eq = ' + '.join(eq)

        if any(var_name in eq for var_name in self.variables):
            expr_vars = [v for v in self.variables if eq.find(v) != -1]
            expr_symbols = symbols(' '.join(expr_vars))
            solution = solve(Eq(sympify(eq), expected), expr_symbols, dict=True)

            if solution:
                for k, v in solution[0].items():
                    self.vector_variables[str(k)] = float(v)
            else:
                for var in expr_vars:
                    if str(var) not in list(self.vector_variables.keys()):
                        self.vector_variables[str(var)] = 0

    def solve_linear_equations(self):
        '''
        Solve the system of linear equations.

        Returns
        -------
        success : bool
            True if the equations were successfully solved, False otherwise.
        vector_variables : Dict[str, int]
            A mapping from variable names to their solved values.
        '''
        if not self.solution:
            return False, None

        self.free_vars = [v for v in self.symbols if v not in self.solution[0]]

        self.vector_variables = {}
        for k, v in self.solution[0].items():
            try:
                self.vector_variables[str(k)] = float(v)
            except TypeError:
                continue

        visited = set()
        queue = deque([self.vars.lattice.bottom()])

        while queue:
            node = queue.popleft()

            for i, dim in enumerate(self.dimensions):
                while not all(
                    f'{dim}_{var}' in self.vector_variables.keys()
                    for var in self.vars.extents[node] | (self.vars.M - self.vars.intents[node])
                ):
                    self._solve(dim, node, self.coordinates[node][i])

            visited.add(node)
            for p in self.vars.lattice.parents(node):
                if p not in queue and p not in visited:
                    queue.append(p)

        return True, self.vector_variables