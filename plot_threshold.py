"""Plot below/at/above ceil(d/r): plot_threshold.py results.json [--output DIR]."""
import argparse
import json
from collections import Counter
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    data = json.loads(path.read_text())
    seen = {}
    for row in data["records"]:
        key = tuple(row[k] for k in ("initialization", "d", "r", "N", "n", "m", "seed"))
        if key in seen and seen[key] != row:
            raise ValueError(f"Conflicting duplicate: {key}")
        threshold = ceil(row["d"] / row["r"])
        if (row["threshold"] != threshold or row["threshold_offset"] not in (-1, 0, 1)
                or row["m"] != threshold + row["threshold_offset"]):
            raise ValueError(f"Inconsistent threshold: {key}")
        if row["D"] != row["N"] * row["n"] * row["d"]:
            raise ValueError(f"Inconsistent dimension: {key}")
        seen[key] = row
    data["records"] = list(seen.values())
    return data


def primary_rank(row, config):
    rtols = row.get("rtols", config.get("rtols", [1e-6, 1e-8, 1e-10]))
    index = rtols.index(row.get("decision_rtol", config.get("decision_rtol", 1e-8)))
    return row["history"][-1]["ranks"][0][index]


def summarize(cell, seeds, config):
    """Color uses all observed primary ranks, including unresolved runs."""
    ranks = [primary_rank(row, config) / row["D"] for row in cell]
    if any(not 0 <= rank <= 1 for rank in ranks):
        raise ValueError("A recorded rank lies outside [0, D]")
    missing = len(set(seeds) - {row["seed"] for row in cell})
    extra = {row["seed"] for row in cell} - set(seeds)
    if extra:
        raise ValueError(f"Unplanned seeds: {sorted(extra)}")
    sensitive = any(len(set(row["history"][-1]["ranks"][0])) > 1 for row in cell)
    flags = Counter()
    for row in cell:
        status = row["status"]
        if status in {"max_steps", "max_fields"}:
            flags["B"] += 1
        elif status == "generic_plateau":
            flags["P"] += 1
        elif status == "numerically_unstable":
            flags["!"] += 1
        elif status not in {"full_rank", "structural_cap", "formal_closure"}:
            flags["?"] += 1
    label = f"{sum(rank == 1 for rank in ranks)}/{len(cell)}" + ("*" if sensitive else "")
    details = [f"{flag}{count}" for flag, count in flags.items()]
    if missing:
        details.append(f"{missing} pending")
    if details:
        label += "\n" + ", ".join(details)
    return np.mean(ranks) if ranks else np.nan, label


def plot(data, destination):
    cfg, rows = data["config"], data["records"]
    pairs = [tuple(pair) for pair in cfg["pairs"]]
    shapes = [tuple(shape) for shape in cfg["shapes"]]
    initializations, seeds = cfg["initializations"], cfg["seeds"]
    if not pairs or not shapes or not seeds:
        raise ValueError("Need configured pairs, shapes and seeds")
    for row in rows:
        if ((row["d"], row["r"]) not in pairs or (row["N"], row["n"]) not in shapes
                or row["initialization"] not in initializations):
            raise ValueError("A record falls outside the configured plotting grid")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e3e3e3")
    destination.mkdir(parents=True, exist_ok=True)
    for initialization in initializations:
        selected = [row for row in rows if row["initialization"] == initialization]
        fig, axes = plt.subplots(1, 3, figsize=(14, max(6, .58 * len(pairs) + 2.6)))
        for ax, offset, title in zip(axes, (-1, 0, 1),
                                    ("Below threshold: m = t − 1", "At threshold: m = t", "Above threshold: m = t + 1")):
            colors = np.full((len(pairs), len(shapes)), np.nan)
            labels = {}
            for i, (d, r) in enumerate(pairs):
                for j, (N, n) in enumerate(shapes):
                    cell = [row for row in selected if (row["d"], row["r"], row["N"], row["n"],
                            row["threshold_offset"]) == (d, r, N, n, offset)]
                    colors[i, j], labels[i, j] = summarize(cell, seeds, cfg)
            im = ax.imshow(colors, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for (i, j), label in labels.items():
                color = "white" if np.isfinite(colors[i, j]) and colors[i, j] < .48 else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=9, color=color)
            ax.set(title=title, xlabel="(N, n)", xticks=range(len(shapes)),
                   xticklabels=[f"({N}, {n})" for N, n in shapes], yticks=range(len(pairs)),
                   yticklabels=[f"({d}, {r}); t={ceil(d/r)}" for d, r in pairs])
            ax.set_xticks(np.arange(len(shapes) + 1) - .5, minor=True)
            ax.set_yticks(np.arange(len(pairs) + 1) - .5, minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="both", length=0)
            ax.set_ylabel("(d, r); threshold t" if ax is axes[0] else "")
            if ax is not axes[0]:
                ax.set_yticklabels([])
        std = "1" if initialization == "standard" else "1/√d" if initialization == "fan_in" else "recorded"
        fig.suptitle(f"Generic-rank threshold probe: t = ⌈d/r⌉ — projection std = {std}", fontsize=13)
        fig.subplots_adjust(left=.115, right=.91, bottom=.24, top=.87, wspace=.12)
        bar = fig.colorbar(im, cax=fig.add_axes((.93, .25, .015, .61)))
        bar.set_label("Mean final primary rank / D")
        total = len(pairs) * len(shapes) * 3 * len(seeds)
        fig.text(.115, .15, "Cell: primary full-rank runs / observed trials. * Final ranks disagree across tolerances.\n"
                 "B: budget reached; P: deficient numerical plateau; !: numerical instability; ?: other unresolved status. Numbers count runs.", fontsize=9)
        fig.text(.115, .07, "Below t, mr < d: full rank is structurally excluded. Deficient plateaus at/above t remain numerically unresolved.\n"
                 "Colors include every observed run; gray means no observations. Float64 evidence does not prove generic sufficiency.\n"
                 f"D = Nnd; {len(seeds)} planned seeds/cell. Observed: {len(selected)}/{total} runs. "
                 f"Primary rtol = {cfg.get('decision_rtol', 1e-8):g}.", fontsize=9)
        for suffix in ("png", "pdf"):
            path = destination / f"threshold_{initialization}.{suffix}"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print(path.resolve())
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(load(args.results), args.output or args.results.parent)


if __name__ == "__main__":
    main()
