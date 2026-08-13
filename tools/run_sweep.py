#!/usr/bin/env python3
"""Run a reproducible GEMM shape sweep and combine the emitted CSV rows."""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path


DEFAULT_SHAPES = (
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (4096, 11008, 4096),
    (8192, 4096, 14336),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/gemm_benchmark"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.csv"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--kernel", default="tiled,register,cublas")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    with tempfile.TemporaryDirectory(prefix="gemm-sweep-") as temp_dir:
        for index, (m, n, k) in enumerate(DEFAULT_SHAPES):
            result_path = Path(temp_dir) / f"result-{index}.csv"
            command = [
                str(args.binary),
                "--m",
                str(m),
                "--n",
                str(n),
                "--k",
                str(k),
                "--kernel",
                args.kernel,
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--csv",
                str(result_path),
            ]
            if args.no_validate:
                command.append("--no-validate")
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True)
            with result_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows.extend(reader)

    if fieldnames is None:
        raise RuntimeError("sweep produced no rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
