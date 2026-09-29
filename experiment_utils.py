"""helpers used only by the threshold experiment."""

import torch

from lie_rank import trace


def first_step(history, target, tolerance_index=None):
    """First target rank at every probe (all tolerances unless specified)."""
    for row in history:
        ranks = row["ranks"]
        if tolerance_index is not None:
            ranks = [[r[tolerance_index]] for r in ranks]
        if all(value == target for r in ranks for value in r):
            return row["step"]
    return None


def configuration(shape, seed):
    """Independent Gaussian X in the full ambient space, without rescaling."""
    D = shape[0] * shape[1] * shape[2]
    X = torch.randn(D, generator=torch.Generator().manual_seed(20_000 + seed),
                    dtype=torch.float64)
    return X.unsqueeze(0)


def run_case(shape, r, m, seed, points, fields, outputs, args):
    N, n, d = shape
    # One autonomous field has Lie rank <= 1; otherwise use the output-space bound.
    # Neither bound is a numerical estimate of rank(O).
    cap = 1 if m == 1 else N * n * min(d, m * r)
    result = trace(fields[:m], points, max_steps=args.max_steps,
                   upper_bound=cap, rtols=args.rtols, max_fields=args.max_fields,
                   decision_rtol=args.decision_rtol, stop_on_plateau=not args.no_plateau)
    evaluations = result.pop("evaluations")
    output_matrix = outputs[:, :m*r]
    U, singular, _ = torch.linalg.svd(output_matrix, full_matrices=False)
    basis = U[:, :min(d, m*r)]
    vectors = evaluations.reshape(len(points), N*n, d, -1)
    transverse = torch.einsum("ij,ptjc->ptic", torch.eye(d) - basis @ basis.T, vectors)
    total_norm = vectors.norm()
    residual = float(transverse.norm() / total_norm) if total_norm else 0.0
    history = result["history"]
    X = points[0].reshape(-1, d)
    unit = X / X.norm(dim=1, keepdim=True)
    cosines = (unit @ unit.T).abs()
    cosines.fill_diagonal_(0)
    return {
        "N": N, "n": n, "d": d, "D": N*n*d, "r": r, "m": m,
        "seed": seed, "projection_seed": 10_000 + seed,
        "configuration_seed": 20_000 + seed,
        "structurally_excludes_full_rank": cap < N*n*d,
        "output_singular_values": singular.tolist(),
        "transverse_relative_residual": residual,
        "minimum_token_norm": float(X.norm(dim=1).min()),
        "minimum_projective_sine": float((1 - cosines.max()**2).clamp_min(0).sqrt()),
        "first_bound_step": first_step(history, cap),
        "first_full_step": first_step(history, N*n*d),
        "first_full_steps_by_tolerance": [first_step(history, N*n*d, i)
                                         for i in range(len(args.rtols))],
        **result,
    }
