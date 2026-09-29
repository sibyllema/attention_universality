"""Test m >= ceil(d/r), below/at/above the boundary, varying N and n."""

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import torch

from lie_rank import sample_blocks
from experiment_utils import configuration, run_case


def pair(text):
    values = tuple(map(int, text.split(":")))
    if len(values) != 2 or min(values) < 1:
        raise argparse.ArgumentTypeError("expected two positive integers, e.g. 3:1")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="+", type=pair,
                        default=[(3, 1), (3, 2), (4, 1), (4, 2), (4, 3),
                                 (6, 1), (6, 2), (6, 3)], help="d:r pairs; use r < d")
    parser.add_argument("--shapes", nargs="+", type=pair,
                        default=[(1, 2), (2, 2), (1, 4), (2, 3)], help="N:n pairs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--offsets", nargs="+", type=int, choices=[-1, 0, 1],
                        default=[-1, 0, 1])
    parser.add_argument("--initializations", nargs="+", choices=["standard", "fan_in"],
                        default=["standard", "fan_in"])
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-fields", type=int, default=4096)
    parser.add_argument("--decision-rtol", type=float, default=1e-8)
    parser.add_argument("--rtols", nargs="+", type=float, default=[1e-6, 1e-8, 1e-10])
    parser.add_argument("--no-plateau", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results_threshold")
    args = parser.parse_args()
    if any(d < 3 or r >= d for d, r in args.pairs) or any(n < 2 for _, n in args.shapes):
        parser.error("this threshold sweep requires d >= 3, r < d and n >= 2")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "complete": False, "python": platform.python_version(), "torch": torch.__version__,
        "dtype": "float64", "device": "cpu", "threads": 1,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "hypothesis": "generic full rank for m >= ceil(d/r), with r < d, d >= 3, n >= 2",
        "initializations": {"standard": "projection entries N(0,1)",
                            "fan_in": "projection entries N(0,1/d); same base draws rescaled"},
        "sampling": "frozen dictionary prefixes in m, shared across shapes; independent configuration seed stream",
        "rank_rule": "column-normalized SVD; max(1e-12, rtol*sigma_max); column floor 1e-14",
        "records": [],
    }
    path = args.output / "results.json"
    for initialization in args.initializations:
        for d, r in args.pairs:
            threshold = math.ceil(d / r)
            std = 1.0 if initialization == "standard" else d ** -0.5
            for N, n in args.shapes:
                shape = (N, n, d)
                for seed in args.seeds:
                    points = configuration(shape, seed)
                    fields, outputs = sample_blocks(shape, r, threshold + 1,
                                                    10_000 + seed, std=std)
                    for offset in args.offsets:
                        m = threshold + offset
                        row = run_case(shape, r, m, seed, points, fields, outputs, args)
                        row.update(initialization=initialization, projection_std=std,
                                   threshold=threshold, threshold_offset=m-threshold)
                        payload["records"].append(row)
                        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
                        last = row["history"][-1]
                        print(f"{initialization} d={d} r={r} N={N} n={n} m={m} seed={seed}: "
                              f"k={last['step']} rank={last['ranks'][0]}/{row['D']} "
                              f"{row['status']} ({row['seconds']:.2f}s)", flush=True)
    payload["complete"] = True
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    keys = ["initialization", "d", "r", "N", "n", "D", "m", "threshold_offset", "seed",
            "upper_bound", "status", "stopping_step", "tolerance_agreement", "seconds"]
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + ["last_step", "last_ranks"])
        writer.writeheader()
        for row in payload["records"]:
            last = row["history"][-1]
            writer.writerow({**{key: row[key] for key in keys},
                             "last_step": last["step"], "last_ranks": last["ranks"][0]})
    print(f"Saved {len(payload['records'])} cases to {path}")


if __name__ == "__main__":
    main()
