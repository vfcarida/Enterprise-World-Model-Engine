"""Tigramite isolated subprocess worker script.

ARCHITECTURE INVARIANT (ADR-048):
    Tigramite is licensed under GPL-3.0. To prevent contaminating the permissive
    Apache-2.0 / MIT licensing of the EWM Engine core, Tigramite is strictly
    isolated behind this out-of-process subprocess worker.

    This script runs as a standalone Python process communicating via JSON over
    stdin and stdout. Core EWM Engine NEVER static-links or imports Tigramite in-process.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        raw_input = sys.stdin.read()
        if not raw_input.strip():
            sys.stderr.write("Error: empty input provided to tigramite_worker.\n")
            return 1

        payload = json.loads(raw_input)
        series_data = payload.get("series_data", [])
        var_names = payload.get("var_names", [])
        max_lag = int(payload.get("max_lag", 1))
        alpha = float(payload.get("alpha", 0.05))

        # Lazy import of tigramite strictly within subprocess boundary
        import numpy as np
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI

        arr = np.array(series_data, dtype=np.float64)
        dataframe = pp.DataFrame(arr, var_names=var_names)
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        results = pcmci.run_pcmciplus(tau_max=max_lag, pc_alpha=alpha)
        graph = results["graph"]  # shape (N, N, tau_max+1)
        val_matrix = results["val_matrix"]

        # Parse links into serializable JSON
        links = []
        n_vars = len(var_names)
        for i in range(n_vars):
            for j in range(n_vars):
                for tau in range(max_lag + 1):
                    link_type = graph[i, j, tau]
                    val = float(val_matrix[i, j, tau])
                    if link_type != "":
                        links.append(
                            {
                                "source": var_names[i],
                                "target": var_names[j],
                                "lag": tau,
                                "link_type": link_type,
                                "val": val,
                            }
                        )

        output_data = {
            "status": "success",
            "links": links,
            "engine": "tigramite_pcmciplus",
        }
        sys.stdout.write(json.dumps(output_data))
        return 0

    except ImportError as exc:
        sys.stderr.write(f"TigramiteImportError: {exc}\n")
        return 2
    except Exception as exc:
        sys.stderr.write(f"TigramiteWorkerError: {exc}\n")
        return 3


if __name__ == "__main__":
    sys.exit(main())
