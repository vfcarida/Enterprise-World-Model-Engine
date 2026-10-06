"""Pure-NumPy discrete bounded-future Signal Temporal Logic (STL) monitor.

Conforms to Track T3: Core Restricted STL Fragment (Zero External Dependencies).
Evaluates Always, Eventually, Until, Implies, And, Or, and Not over discrete-time
WorldState and Trajectory timeseries, reporting exact Boolean verdicts and
quantitative robustness margins.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ewm_engine.simulation.trajectory import Trajectory


@dataclass(frozen=True)
class STLVerdict:
    """Outcome of evaluating an STL formula over a trajectory timeseries."""

    satisfied: bool
    robustness: float
    boolean_trace: tuple[bool, ...]
    robustness_trace: tuple[float, ...]
    violation_steps: tuple[int, ...]


class STLFormula(ABC):
    """Abstract base class for all bounded-future discrete STL formulas."""

    @abstractmethod
    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate formula over signals.

        Returns:
            tuple of (boolean_verdict_array, robustness_margin_array)
        """
        ...


class PredicateFormula(STLFormula):
    """Atomic predicate checking a variable against a threshold: x ~ threshold."""

    def __init__(
        self,
        variable: str,
        operator: Literal[">=", "<=", ">", "<", "==", "!="],
        threshold: float,
    ) -> None:
        self.variable = variable
        self.operator = operator
        self.threshold = float(threshold)

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if self.variable not in signals:
            raise KeyError(f"Variable '{self.variable}' not found in trajectory signals.")
        x = np.asarray(signals[self.variable], dtype=np.float64)

        if self.operator == ">=":
            rob = x - self.threshold
            verdict = rob >= 0.0
        elif self.operator == ">":
            rob = x - self.threshold
            verdict = rob > 0.0
        elif self.operator == "<=":
            rob = self.threshold - x
            verdict = rob >= 0.0
        elif self.operator == "<":
            rob = self.threshold - x
            verdict = rob > 0.0
        elif self.operator == "==":
            rob = -np.abs(x - self.threshold)
            verdict = np.isclose(x, self.threshold)
        elif self.operator == "!=":
            rob = np.abs(x - self.threshold)
            verdict = ~np.isclose(x, self.threshold)
        else:
            raise ValueError(f"Unsupported predicate operator: {self.operator}")

        return verdict, rob


class NotFormula(STLFormula):
    """Negation: !phi."""

    def __init__(self, child: STLFormula) -> None:
        self.child = child

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        b, r = self.child.evaluate(signals)
        return ~b, -r


class AndFormula(STLFormula):
    """Conjunction: phi & psi."""

    def __init__(self, left: STLFormula, right: STLFormula) -> None:
        self.left = left
        self.right = right

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        b_l, r_l = self.left.evaluate(signals)
        b_r, r_r = self.right.evaluate(signals)
        verdict = b_l & b_r
        rob = np.minimum(r_l, r_r)
        return verdict, rob


class OrFormula(STLFormula):
    """Disjunction: phi | psi."""

    def __init__(self, left: STLFormula, right: STLFormula) -> None:
        self.left = left
        self.right = right

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        b_l, r_l = self.left.evaluate(signals)
        b_r, r_r = self.right.evaluate(signals)
        verdict = b_l | b_r
        rob = np.maximum(r_l, r_r)
        return verdict, rob


class ImpliesFormula(STLFormula):
    """Implication: phi -> psi equivalent to (!phi | psi)."""

    def __init__(self, antecedent: STLFormula, consequent: STLFormula) -> None:
        self.antecedent = antecedent
        self.consequent = consequent

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        b_a, r_a = self.antecedent.evaluate(signals)
        b_c, r_c = self.consequent.evaluate(signals)
        verdict = (~b_a) | b_c
        rob = np.maximum(-r_a, r_c)
        return verdict, rob


class AlwaysFormula(STLFormula):
    """Globally / Always in bounded window [k1, k2]: G_[k1, k2] phi."""

    def __init__(self, child: STLFormula, interval: tuple[int, int] = (0, 0)) -> None:
        self.child = child
        self.k1 = interval[0]
        self.k2 = interval[1]
        if self.k1 < 0 or self.k2 < self.k1:
            raise ValueError(f"Invalid interval [{self.k1}, {self.k2}]: must be 0 <= k1 <= k2")

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        b_child, r_child = self.child.evaluate(signals)
        T = len(b_child)
        b_out = np.zeros(T, dtype=bool)
        r_out = np.full(T, fill_value=-np.inf, dtype=np.float64)

        for t in range(T):
            start = t + self.k1
            end = min(t + self.k2 + 1, T)
            if start >= T:
                # Outside future horizon; vacuously true or clamp to boundary
                b_out[t] = True
                r_out[t] = r_child[-1]
            else:
                window_b = b_child[start:end]
                window_r = r_child[start:end]
                b_out[t] = np.all(window_b)
                r_out[t] = np.min(window_r)

        return b_out, r_out


class EventuallyFormula(STLFormula):
    """Finally / Eventually in bounded window [k1, k2]: F_[k1, k2] phi."""

    def __init__(self, child: STLFormula, interval: tuple[int, int] = (0, 0)) -> None:
        self.child = child
        self.k1 = interval[0]
        self.k2 = interval[1]
        if self.k1 < 0 or self.k2 < self.k1:
            raise ValueError(f"Invalid interval [{self.k1}, {self.k2}]: must be 0 <= k1 <= k2")

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        b_child, r_child = self.child.evaluate(signals)
        T = len(b_child)
        b_out = np.zeros(T, dtype=bool)
        r_out = np.full(T, fill_value=-np.inf, dtype=np.float64)

        for t in range(T):
            start = t + self.k1
            end = min(t + self.k2 + 1, T)
            if start >= T:
                b_out[t] = False
                r_out[t] = -np.inf
            else:
                window_b = b_child[start:end]
                window_r = r_child[start:end]
                b_out[t] = np.any(window_b)
                r_out[t] = np.max(window_r)

        return b_out, r_out


class UntilFormula(STLFormula):
    """Bounded Until: phi U_[k1, k2] psi."""

    def __init__(
        self, left: STLFormula, right: STLFormula, interval: tuple[int, int] = (0, 0)
    ) -> None:
        self.left = left
        self.right = right
        self.k1 = interval[0]
        self.k2 = interval[1]
        if self.k1 < 0 or self.k2 < self.k1:
            raise ValueError(f"Invalid interval [{self.k1}, {self.k2}]: must be 0 <= k1 <= k2")

    def evaluate(self, signals: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        b_l, r_l = self.left.evaluate(signals)
        b_r, r_r = self.right.evaluate(signals)
        T = len(b_l)
        b_out = np.zeros(T, dtype=bool)
        r_out = np.full(T, fill_value=-np.inf, dtype=np.float64)

        for t in range(T):
            candidate_robs: list[float] = []
            cand_b = False
            start = t + self.k1
            end = min(t + self.k2 + 1, T)

            for target_k in range(start, end):
                # psi must hold at target_k
                psi_r = r_r[target_k]
                psi_b = b_r[target_k]

                # phi must hold for all steps from start up to target_k
                phi_b: bool
                if target_k > start:
                    phi_r = float(np.min(r_l[start:target_k]))
                    phi_b = bool(np.all(b_l[start:target_k]))
                else:
                    phi_r = np.inf
                    phi_b = True

                step_b = bool(psi_b and phi_b)
                step_r = min(psi_r, phi_r)
                if step_b:
                    cand_b = True
                candidate_robs.append(step_r)

            b_out[t] = cand_b
            r_out[t] = max(candidate_robs) if candidate_robs else -np.inf

        return b_out, r_out


def extract_trajectory_signals(trajectory: Trajectory) -> dict[str, np.ndarray]:
    """Extract named 1D numerical signals from a Trajectory."""
    signals: dict[str, list[float]] = {}

    for step_rec in trajectory.steps:
        # Resource values
        child_state = (
            step_rec.transition_result.next_state
            if step_rec.transition_result is not None
            else None
        )
        if child_state:
            for r_id, r in child_state.resources.items():
                signals.setdefault(r_id, []).append(float(r.current))
                signals.setdefault(f"resource_{r_id}", []).append(float(r.current))
            for m_id, m_val in child_state.memory.items():
                if isinstance(m_val, (int, float)):
                    signals.setdefault(m_id, []).append(float(m_val))

        # Step metrics
        for k, v in step_rec.step_metrics.items():
            signals.setdefault(k, []).append(float(v))

    return {k: np.array(v, dtype=np.float64) for k, v in signals.items()}


def evaluate_stl_bounded(
    formula: STLFormula,
    trajectory_or_signals: Trajectory | Mapping[str, Sequence[float] | np.ndarray],
) -> STLVerdict:
    """Evaluate a bounded-future discrete STL formula over a trajectory or signal dictionary.

    Returns:
        STLVerdict containing satisfaction boolean and quantitative robustness margin.
    """
    if isinstance(trajectory_or_signals, Trajectory):
        signals = extract_trajectory_signals(trajectory_or_signals)
    else:
        signals = {k: np.asarray(v, dtype=np.float64) for k, v in trajectory_or_signals.items()}

    b_arr, r_arr = formula.evaluate(signals)
    violations = [int(i) for i, satisfied in enumerate(b_arr) if not satisfied]

    return STLVerdict(
        satisfied=bool(b_arr[0]),
        robustness=float(r_arr[0]),
        boolean_trace=tuple(bool(x) for x in b_arr),
        robustness_trace=tuple(float(x) for x in r_arr),
        violation_steps=tuple(violations),
    )


# Convenience builders
def predicate(
    variable: str, operator: Literal[">=", "<=", ">", "<", "==", "!="], threshold: float
) -> PredicateFormula:
    return PredicateFormula(variable=variable, operator=operator, threshold=threshold)


def always(child: STLFormula, k1: int = 0, k2: int = 0) -> AlwaysFormula:
    return AlwaysFormula(child=child, interval=(k1, k2))


def eventually(child: STLFormula, k1: int = 0, k2: int = 0) -> EventuallyFormula:
    return EventuallyFormula(child=child, interval=(k1, k2))


def until(left: STLFormula, right: STLFormula, k1: int = 0, k2: int = 0) -> UntilFormula:
    return UntilFormula(left=left, right=right, interval=(k1, k2))


def implies(antecedent: STLFormula, consequent: STLFormula) -> ImpliesFormula:
    return ImpliesFormula(antecedent=antecedent, consequent=consequent)
