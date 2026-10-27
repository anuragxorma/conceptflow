# Data quality

FCA gives precise answers to everyday data-quality questions: which records
break rules that should hold, which columns are redundant, and why an object
or a rule looks the way it does.

```python
import conceptflow as cf
from conceptflow.quality import check_implications, redundancy_report, explain_object, explain_rule

animals = cf.FormalContext.from_array(
    [
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
    ],
    objects=["cat", "otter", "dog", "trout", "sparrow", "platypus?"],
    attributes=["mammal", "warm-blooded", "flies", "swims"],
)
```

## Rules that should hold

Give the rules you expect (as `(premise, conclusion)` pairs of names or
indices, or `Implication` objects) and get every violation:

```python
report = check_implications(animals, [(["mammal"], ["warm-blooded"]), (["flies"], ["warm-blooded"])])
print(report[["rule", "object", "missing"]])
print(report["object"].tolist())  # ['platypus?']
```

## Redundancy

`redundancy_report` lists duplicate objects and attributes, attributes every
object has or none has, empty and full objects, and *derivable* attributes
and objects — columns (rows) that are the intersection of others and add no
information to the concept lattice:

```python
table = cf.FormalContext.from_array(
    [[1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
    objects=["r1", "r1_copy", "r2", "r3"],
    attributes=["a", "b", "a_and_b", "c"],
)
redundancy = redundancy_report(table)
print(redundancy.duplicate_objects)  # [('r1', 'r1_copy')]
print(redundancy.derivable_attributes)  # {'a_and_b': ('a', 'b')}
print(redundancy)
```

## Explanations

```python
lattice = cf.ConceptLattice.from_context(animals)
print(explain_object(lattice, "otter"))
print(explain_rule(animals, (["mammal"], ["warm-blooded"])))
```

`explain_object` reports an object's attributes, how many concepts contain
it, which objects are indistinguishable from it, and what it has beyond each
more general neighbouring concept. `explain_rule` reports support,
confidence, and every exception with the attributes it lacks.

## Outliers

To find unusual objects without listing rules yourself, use
[`ImplicationOutlierDetector`](sklearn.md#outlier-detection-with-implications):
it learns the rules from the data and explains every flag.
