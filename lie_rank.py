"""Evaluate W_k(X) = (W_{k-1} + [W_0, W_{k-1}])(X), using PyTorch JVPs.

W_0 contains the original fields: step k means bracket length <= k+1.
No field is discarded because its evaluation is zero or dependent.
"""

from itertools import combinations
from time import perf_counter

import torch
from torch.func import jvp, vmap


def bracket(f, g):
    """Differentiable field [f,g](x) = Dg(x) f(x) - Df(x) g(x)."""
    def h(x):
        gx, dg_f = jvp(g, (x,), (f(x),))
        _, df_g = jvp(f, (x,), (gx,))
        return dg_f - df_g
    return h


def lift(field, shape):
    """Apply the SAME field separately to N samples; accept a flat state."""
    def lifted(x):
        return vmap(field)(x.reshape(shape)).reshape(-1)
    return lifted


def attention(Q, K, U, O):
    """One noncausal head, on an (n,d) sequence; projections stay frozen."""
    W, V = K.T @ Q / Q.shape[0] ** 0.5, O @ U

    def field(X):
        weights = torch.softmax((X @ W.T) @ X.T, dim=-1)
        return weights @ (X @ V.T)
    return field


def sample_blocks(shape, r, m, seed, std=1.0):
    """Independent Gaussian projections; prefix invariant when m increases."""
    if min(*shape, r, m) < 1 or std <= 0:
        raise ValueError("dimensions and projection standard deviation must be positive")
    d = shape[-1]
    rng = torch.Generator().manual_seed(seed)
    fields, outputs = [], []
    for _ in range(m):
        Q, K, U = [std * torch.randn(r, d, generator=rng, dtype=torch.float64)
                   for _ in range(3)]
        O = std * torch.randn(d, r, generator=rng, dtype=torch.float64)
        fields.append(lift(attention(Q, K, U, O), shape))
        outputs.append(O)
    return fields, torch.cat(outputs, dim=1)


def layers(fields, max_steps):
    """Right-nested words spanning each layer; remove only antisymmetry."""
    frontier = [(f"g{i}", f) for i, f in enumerate(fields)]
    yield frontier
    if max_steps == 0:
        return
    frontier = [(f"[g{i},g{j}]", bracket(fields[i], fields[j]))
                for i, j in combinations(range(len(fields)), 2)]
    for k in range(1, max_steps + 1):
        yield frontier
        if k < max_steps:
            frontier = [(f"[g{i},{name}]", bracket(f, h))
                        for i, f in enumerate(fields) for name, h in frontier]


def spectrum(matrix, rtols, atol=1e-12, column_atol=1e-14):
    """SVD after column normalization, performed AFTER differentiation."""
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("non-finite bracket evaluation")
    norms = torch.linalg.vector_norm(matrix, dim=0)
    keep = norms > column_atol
    normalized = matrix[:, keep] / norms[keep]
    singular = torch.linalg.svdvals(normalized)
    top = float(singular[0]) if singular.numel() else 0.0
    return {
        "ranks": [int((singular > max(atol, t * top)).sum()) for t in rtols],
        "singular_values": singular.tolist(),
        "column_norms": norms.tolist(),
        "discarded_columns": int((~keep).sum()),
    }


def trace(fields, points, max_steps=5, upper_bound=None,
          rtols=(1e-6, 1e-8, 1e-10), max_fields=4096,
          decision_rtol=1e-8, stop_on_plateau=True):
    """Return evaluated words and ranks; default to generic plateau stopping.

    points: (D,) or (P,D), with the decision point X first (normally P=1).
    For fixed analytic fields and an independent ambient-density draw X,
    exact ranks are almost surely generic: a plateau implies local closure.
    Numerical tolerances remain a diagnostic, not a rigorous certificate.
    Set stop_on_plateau=False to examine exceptional points. upper_bound
    must be a MATHEMATICAL bound, never a numerically estimated rank.
    """
    if points.ndim == 1:
        points = points.unsqueeze(0)
    if not fields or points.ndim != 2 or min(points.shape) < 1 or max_steps < 0:
        raise ValueError("need fields, nonempty (P,D) points, and max_steps >= 0")
    if points.dtype != torch.float64 or points.device.type != "cpu":
        raise ValueError("use CPU float64 points")
    if not rtols or min(rtols) <= 0 or max_fields < len(fields):
        raise ValueError("positive tolerances and enough space for generators required")
    if decision_rtol not in rtols:
        raise ValueError("decision_rtol must belong to rtols")
    decision_index = list(rtols).index(decision_rtol)
    D = points.shape[1]
    cap = D if upper_bound is None else upper_bound
    if not 1 <= cap <= D:
        raise ValueError("upper_bound must be between 1 and D")
    values, names, history = [], [], []
    status, first_plateau, stopping_step = "max_steps", None, None
    start = perf_counter()
    for k, layer in enumerate(layers(fields, max_steps)):
        if not layer:
            status = "formal_closure"
            stopping_step = k - 1
            break
        if len(values) + len(layer) > max_fields:
            status = "max_fields"
            break
        for name, field in layer:
            value = vmap(field)(points)
            if value.shape != points.shape:
                raise ValueError(f"{name} does not map the state space to itself")
            values.append(value.detach())
            names.append(name)
        matrix = torch.stack(values, dim=-1)  # P x D x number of words
        diagnostics = [spectrum(M, rtols) for M in matrix]
        ranks = [p["ranks"] for p in diagnostics]
        rank = ranks[0][decision_index]
        plateau = k > 0 and rank == history[-1]["ranks"][0][decision_index]
        if plateau and first_plateau is None:
            first_plateau = k - 1
        history.append({
            "step": k, "length": k + 1, "words": len(values),
            "ranks": ranks, "points": diagnostics,
            "pointwise_plateau": plateau, "seconds": perf_counter() - start,
        })
        if (any(value > cap for row in ranks for value in row)
                or (k > 0 and rank < history[-2]["ranks"][0][decision_index])):
            status = "numerically_unstable"
            break
        if rank == D:
            status = "full_rank"
            stopping_step = k
            break
        if plateau and stop_on_plateau:
            status = "structural_cap" if rank == cap else "generic_plateau"
            stopping_step = k - 1
            break
    return {
        "evaluations": torch.stack(values, dim=-1), "words": names,
        "history": history, "status": status, "upper_bound": cap,
        "rtols": list(rtols), "decision_rtol": decision_rtol,
        "stopping_step": stopping_step, "first_plateau_step": first_plateau,
        "tolerance_agreement": len(set(history[-1]["ranks"][0])) == 1,
        "seconds": perf_counter() - start,
    }
