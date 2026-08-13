#!/usr/bin/env python3
"""Rerun NLFT for saved odd Figure 7 fits without repeating optimization."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from degree_scaling_data_odd import (
    _critical_error,
    _critical_max_abs,
    _qsp_chebyshev_coefficients,
)
from qsppack.solver import solve


DEFAULT_INPUT = "data/degree_scaling_uniform_sv_amp_npts19_solver2_odd.csv"
DEFAULT_OUTPUT = "data/degree_scaling_uniform_sv_amp_npts19_solver2_odd_Nweiss16.csv"


def rerun(input_csv: str, output_csv: str, N_weiss: int) -> None:
    if N_weiss < 1:
        raise ValueError("N_weiss must be positive")
    data = pd.read_csv(input_csv).sort_values("degree").reset_index(drop=True)
    required = {"degree", "parity", "a", "epsilon", "coef", "N_weiss"}
    # Older corrected CSVs encode the target parameters in the documented run.
    if "a" not in data.columns:
        data["a"] = 0.2
    if "epsilon" not in data.columns:
        data["epsilon"] = 0.0
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"input CSV is missing columns: {sorted(missing)}")
    if not (data["parity"].astype(int) == 1).all():
        raise ValueError("all input rows must have odd parity")

    data["source_N_weiss"] = data["N_weiss"].astype(int)
    data["retraction_reused_fit"] = True
    completed: dict[int, dict[str, object]] = {}
    if os.path.exists(output_csv):
        previous = pd.read_csv(output_csv)
        if "N_weiss" in previous and not (previous["N_weiss"].astype(int) == N_weiss).all():
            raise ValueError("existing output CSV was generated with a different N_weiss")
        completed = {
            int(row["degree"]): row.to_dict() for _, row in previous.iterrows()
        }

    rows: list[dict[str, object]] = []
    for _, source_row in data.iterrows():
        degree = int(source_row["degree"])
        if degree in completed:
            print(f"Skipping completed degree {degree}")
            rows.append(completed[degree])
            continue

        parity = int(source_row["parity"])
        coefficients = np.asarray(json.loads(source_row["coef"]), dtype=float)
        opts = {
            "N": int(N_weiss),
            "method": "NLFT",
            "targetPre": False,
            "typePhi": "reduced",
        }
        print(f"Retracting degree {degree} with N_weiss={N_weiss}")
        started = time.perf_counter()
        phi_proc, _out = solve(coefficients, parity, opts)
        runtime = time.perf_counter() - started
        phi_proc = np.asarray(phi_proc)
        imaginary_max = float(np.max(np.abs(phi_proc.imag)))
        if imaginary_max > 1.0e-10:
            raise RuntimeError(f"phase imaginary part is {imaginary_max:.3e}")

        qsp_coefficients, reconstruction_residual = _qsp_chebyshev_coefficients(
            degree, phi_proc, parity
        )
        a = float(source_row["a"])
        epsilon = float(source_row["epsilon"])
        qsp_error = _critical_error(
            qsp_coefficients, (1.0 - epsilon) / a, 0.0, a
        )
        qsp_bound = _critical_max_abs(qsp_coefficients, -1.0, 1.0)
        if qsp_bound > 1.0 + 1.0e-10:
            raise RuntimeError(
                f"degree {degree} retracted polynomial violates bound: {qsp_bound:.16g}"
            )

        row = source_row.to_dict()
        row.update(
            {
                "N_weiss": int(N_weiss),
                "time_qsp": runtime,
                "max_error_qsp": qsp_error,
                "max_error_qsp_uniform_1000": np.nan,
                "qsp_constraint_max_abs": qsp_bound,
                "qsp_reconstruction_residual": reconstruction_residual,
                "phi_proc": json.dumps(phi_proc.real.tolist()),
                "source_N_weiss": int(source_row["source_N_weiss"]),
                "retraction_reused_fit": True,
            }
        )
        rows.append(row)
        pd.DataFrame(rows).sort_values("degree").to_csv(output_csv, index=False)
        print(
            f"  error={qsp_error:.6e}, max|q|={qsp_bound:.12f}, "
            f"runtime={runtime:.2f}s"
        )
    print(f"Data written to: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--N", type=int, default=2**16)
    args = parser.parse_args()
    rerun(args.input, args.output, args.N)


if __name__ == "__main__":
    main()
