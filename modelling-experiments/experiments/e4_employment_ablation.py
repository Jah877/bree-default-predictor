"""E4: automatically ablate the best *validation-selected* E1/E2/E3 model.

The runner chooses the parent without looking at the test labels, then calls
make_ablation(parent). This file owns the ablation protocol rather than a
separately, arbitrarily retuned set of model parameters.
"""
from specs import ExperimentSpec, without_employment

PARENT_CANDIDATES = ("E1", "E2", "E3")
REMOVED_FEATURE = "employment_status"


def make_ablation(parent: ExperimentSpec) -> ExperimentSpec:
    if parent.key not in PARENT_CANDIDATES:
        raise ValueError(f"E4 parent must be E1, E2 or E3, got {parent.key}")
    return without_employment(parent)
