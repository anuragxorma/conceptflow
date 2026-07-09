"""
Experimental Toscana-style nested diagram example for ConceptFlow.

This example uses synthetic many-valued data inspired by nested line diagrams:

    coma
        -> pH level
            -> symptom duration

Run from the project root:

    python -m examples.nested_diagram

Output:

    nested_diagram_diabetes.html
"""

import pandas as pd

from conceptflow import ExplorationBuilder, ManyValuedContext
from conceptflow.preprocessing import DichotomicScale, GeneralScale
from conceptflow.visualization import plot_nested


def build_many_valued_context() -> ManyValuedContext:
    """
    Build a synthetic many-valued context.
    """
    df = pd.DataFrame(
        {
            "coma": [
                "no", "no", "no", "no", "no", "no",
                "no", "no", "no", "no", "no", "no",
                "no", "no", "no", "no", "no", "no",
                "no", "no", "no", "no", "no", "no",
                "yes", "yes", "yes", "yes", "yes", "yes",
                "yes", "yes", "yes", "yes", "yes", "yes",
                "yes", "yes", "yes", "yes", "yes", "yes",
            ],
            "pH_level": [
                "normal", "normal", "normal", "normal",
                "pathological", "pathological", "pathological", "pathological",
                "dangerous", "dangerous", "dangerous", "dangerous",
                "normal", "normal", "pathological", "pathological",
                "dangerous", "dangerous", "normal", "pathological",
                "dangerous", "normal", "pathological", "dangerous",
                "dangerous", "dangerous", "dangerous", "dangerous",
                "pathological", "pathological", "pathological", "pathological",
                "normal", "normal", "normal", "normal",
                "dangerous", "dangerous", "pathological", "pathological",
                "normal", "dangerous",
            ],
            "symptom_duration": [
                "none", "month", "week", "year",
                "month", "month", "week", "year",
                "week", "month", "year", "none",
                "month", "year", "week", "none",
                "month", "week", "none", "year",
                "month", "week", "month", "year",
                "month", "week", "year", "none",
                "month", "year", "week", "none",
                "month", "week", "year", "none",
                "month", "year", "month", "week",
                "none", "week",
            ],
        },
        index=[f"C{i}" for i in range(1, 43)],
    )

    return ManyValuedContext.from_dataframe(df)


def main() -> None:
    mvc = build_many_valued_context()

    builder = ExplorationBuilder(
        mvc,
        algorithm="nextclosure",
    )

    coma_scale = DichotomicScale(
        "coma",
        true_value="yes",
        false_value="no",
    )

    ph_scale = GeneralScale(
        "pH_level",
        mapping={
            "dangerous": {
                "dangerous pH",
                "pathological pH",
            },
            "pathological": {
                "pathological pH",
            },
            "normal": {
                "normal pH",
            },
        },
        attribute_order=[
            "dangerous pH",
            "pathological pH",
            "normal pH",
        ],
    )

    symptom_duration_scale = GeneralScale(
        "symptom_duration",
        mapping={
            "none": {
                "no symptoms",
                "s.d.: none or year",
            },
            "year": {
                "s.d.: none or year",
                "s.d.: year",
                "s.d.: month-year",
                "s.d.: week-year",
            },
            "month": {
                "s.d.: month-year",
                "s.d.: week-year",
                "s.d.: week-month",
                "s.d.: month",
            },
            "week": {
                "s.d.: week-year",
                "s.d.: week-month",
                "s.d.: week",
            },
        },
        attribute_order=[
            "no symptoms",
            "s.d.: none or year",
            "s.d.: year",
            "s.d.: month-year",
            "s.d.: week-year",
            "s.d.: week-month",
            "s.d.: month",
            "s.d.: week",
        ],
    )

    root = builder.root(
        name="Diabetes example: coma",
        scales=[coma_scale],
    )

    ph_views = builder.expand_all(
        parent=root,
        scales=[ph_scale],
        name_template="pH level view: {label}",
        min_extent_size=0,
        include_top=True,
        include_bottom=True,
    )

    for ph_view in ph_views:
        builder.expand_all(
            parent=ph_view,
            scales=[symptom_duration_scale],
            name_template="symptom duration view: {label}",
            min_extent_size=0,
            include_top=True,
            include_bottom=True,
        )

    fig_default = plot_nested(
        root,
        width=1150,
        height=780,
        title="Diabetes nested exploration: child previews",
    )
    path_default = fig_default.open("nested_diagram_diabetes_previews.html")

    print(f"Wrote and opened {path_default}")



if __name__ == "__main__":
    main()