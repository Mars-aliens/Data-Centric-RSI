from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SynthesisStrategy:
    name: str
    objective: str
    required_evidence: str
    output_mode: str


STRATEGIES = {
    "coverage": SynthesisStrategy(
        "coverage",
        "fill an underrepresented requirement or input type",
        "requirement hypotheses and examples of the missing coverage",
        "new labeled example",
    ),
    "revision": SynthesisStrategy(
        "revision",
        "repair a known failure while preserving the useful task behavior",
        "a seed example and an explicit failure or edit description",
        "revised labeled example",
    ),
    "contrastive": SynthesisStrategy(
        "contrastive",
        "create a labeled task example near a requirement or class boundary",
        "neighboring requirement descriptions and a seed example near their boundary",
        "labeled boundary example",
    ),
}


def strategy_for(name: str) -> SynthesisStrategy:
    try:
        return STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown synthesis strategy {name!r}; choose from {sorted(STRATEGIES)}"
        ) from exc


def strategy_instruction(name: str) -> str:
    strategy = strategy_for(name)
    return (
        f"Strategy objective: {strategy.objective}. "
        f"Use {strategy.required_evidence}. "
        f"The requested output mode is {strategy.output_mode}. "
        "Do not claim that a generated score is human feedback."
    )
