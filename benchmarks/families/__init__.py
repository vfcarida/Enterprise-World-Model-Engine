"""The five canonical scientific benchmark families for EWM Engine."""

from __future__ import annotations

from benchmarks.families.constraint_stress import ConstraintStressBenchmark
from benchmarks.families.intervention_shift import InterventionShiftBenchmark
from benchmarks.families.long_horizon import LongHorizonBenchmark
from benchmarks.families.multi_agent_cascade import MultiAgentCascadeBenchmark
from benchmarks.families.rule_shift import RuleShiftBenchmark

__all__ = [
    "ConstraintStressBenchmark",
    "InterventionShiftBenchmark",
    "LongHorizonBenchmark",
    "MultiAgentCascadeBenchmark",
    "RuleShiftBenchmark",
]
