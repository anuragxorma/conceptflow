"""
LaTeX in attribute and object names (e.g. ``1,\\sqrt{3}`` in the triangles
data) shown as the symbols it stands for.
"""

from __future__ import annotations

import re

SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω", "cdot": "·", "times": "×", "pm": "±", "div": "÷",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "infty": "∞", "to": "→", "rightarrow": "→", "leftarrow": "←",
    "in": "∈", "notin": "∉", "subseteq": "⊆", "subset": "⊂", "cup": "∪",
    "cap": "∩", "emptyset": "∅", "degree": "°", "circ": "∘", "ldots": "…",
}
_SUPERSCRIPTS = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")

# One LaTeX fragment: \sqrt{...}, a known command, or a simple superscript
# (not subscripts: names such as item_1 are common and meant literally).
_NAMES = "|".join(sorted(SYMBOLS, key=len, reverse=True))
_COMMAND = rf"\\(?:sqrt\{{[^{{}}]*\}}|(?:{_NAMES})(?![A-Za-z]))"
_SCRIPT = r"\^(?:\{[0-9+\-n]+\}|[0-9n])"
LATEX_FRAGMENT = re.compile(rf"{_COMMAND}|{_SCRIPT}")


def _fragment_to_unicode(fragment: str) -> str:
    if fragment.startswith("\\sqrt"):
        argument = display_text(fragment[6:-1])
        return "√" + (argument if argument.isalnum() else f"({argument})")
    if fragment[0] == "^":
        return fragment[1:].strip("{}").translate(_SUPERSCRIPTS)
    return SYMBOLS[fragment[1:]]


def display_text(name) -> str:
    """
    ``name`` with common LaTeX shown as symbols: ``\\sqrt{3}`` → √3,
    ``\\alpha`` → α, ``x^2`` → x², ``$\\leq 5$`` → ≤ 5. Everything else is
    kept as it is, including ``$`` that does not enclose LaTeX.
    """
    text = str(name)
    if "\\" not in text and "^" not in text:
        return text
    text = re.sub(r"\$([^$]*\\[^$]*)\$", r"\1", text)
    return LATEX_FRAGMENT.sub(lambda m: _fragment_to_unicode(m.group(0)), text)
