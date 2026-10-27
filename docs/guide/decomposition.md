# Ordinal factorization

`ExactOrdinalTwoFactorizer` implements the exact construction of Algorithm 1
in Dürrschnabel & Stumme's work on ordinal factorization: for a context whose
complement's concept lattice has order dimension at most 2, it finds two
linear extensions of that lattice and builds the two Ferrers factors from
them.

```python
import conceptflow as cf
from conceptflow.decomposition import ExactOrdinalTwoFactorizer

ctx = cf.FormalContext.from_array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
model = ExactOrdinalTwoFactorizer().fit(ctx)

first_factor, second_factor = model.factors_
print(model.coverage_)  # 1.0
```

`coverage_` is the share of crosses covered by the two factors. Finding the
pair of linear extensions is a brute-force search intended for small
contexts.

!!! warning "Experimental"
    The full `Ord2Factor` algorithm for maximal ordinal two-factorizations
    (arbitrary order dimension) is not implemented yet: its `fit` raises
    `NotImplementedError`. It is importable only from
    `conceptflow.decomposition.ordinal_factorization`.
