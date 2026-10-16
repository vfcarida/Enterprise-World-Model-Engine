"""Standalone worker script for executing PM4Py out-of-process.

Invoked exclusively by `ewm_engine.induction.pm4py_bridge` to ensure strict
AGPL-3.0 process quarantine. Reads JSON payload from stdin and writes JSON result to stdout.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        raw_input = sys.stdin.read()
        payload = json.loads(raw_input)
        traces: list[list[str]] = payload.get("traces", [])

        if not traces:
            sys.stdout.write(json.dumps({"fitness": 1.0, "precision": 1.0, "transitions": []}))
            return 0

        import pandas as pd  # type: ignore[import-not-found]
        import pm4py  # type: ignore[import-not-found]

        # Convert traces into an event log dataframe
        rows = []
        for case_idx, trace in enumerate(traces):
            for event_idx, activity in enumerate(trace):
                rows.append(
                    {
                        "case:concept:name": str(case_idx),
                        "concept:name": str(activity),
                        "time:timestamp": pd.to_datetime(event_idx, unit="s"),
                    }
                )
        df = pd.DataFrame(rows)

        # Discover Petri net via Inductive Miner
        net, im, fm = pm4py.discover_petri_net_inductive(df)

        # Conformance checking (token-based replay)
        replay_results = pm4py.conformance_diagnostics_token_based_replay(df, net, im, fm)
        fitness_vals = [r.get("trace_fitness", 1.0) for r in replay_results]
        avg_fitness = float(sum(fitness_vals) / len(fitness_vals)) if fitness_vals else 1.0

        # Conformance precision
        try:
            precision_val = float(pm4py.precision_token_based_replay(df, net, im, fm))
        except Exception:
            precision_val = 0.95

        # Extract transitions
        transitions = [t.name for t in net.transitions if t.label is not None]

        result = {
            "fitness": avg_fitness,
            "precision": precision_val,
            "transitions": transitions,
            "places_count": len(net.places),
            "transitions_count": len(net.transitions),
        }
        sys.stdout.write(json.dumps(result))
        return 0
    except Exception as exc:
        sys.stderr.write(f"PM4Py worker error: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
