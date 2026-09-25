"""Release-aware 5-core wavefront neighborhood for schedule candidates."""

from __future__ import annotations

import math

from model import WAIT_CROSS, WAIT_SAME, DELAY_B


def _diagnostics(ctx, sol):
    sol = sol.clone()
    mk, added, info = ctx.model.evaluate(
        sol.sg_of_block, sol.core_of_sg, ctx.scene, ctx.num_cores,
        use_cache=False, diagnostics=True)
    return sol, mk, added, info


def generate_candidates(ctx, root_sol, max_actions=24):
    """Generate proxy-gated migration/swap candidates from a wavefront view.

    The generator is intentionally restricted to N=5. It only proposes moves
    from a late, overloaded core to an earlier, lighter core and leaves the
    official candidate fallback untouched when the proxy does not improve.
    """
    if ctx.num_cores != 5:
        return [], {"status": "skipped", "reason": "requires_5_cores"}
    sol, base_mk, base_added, info = _diagnostics(ctx, root_sol)
    durs = info.get("durs", [])
    ends = info.get("end_time", [])
    preds = info.get("sg_preds", [])
    orders = info.get("orders", [[] for _ in range(5)])
    if not durs or len(ends) != len(durs):
        return [], {"status": "skipped", "reason": "missing_diagnostics"}

    loads = [0.0] * 5
    tails = [0.0] * 5
    pos = {}
    for c, row in enumerate(orders):
        for i, s in enumerate(row):
            if s < len(durs):
                loads[c] += max(0.0, durs[s])
                tails[c] = max(tails[c], ends[s])
                pos[s] = (c, i)
    mean = sum(loads) / 5.0
    if mean <= 0.0 or max(loads) <= 1.05 * mean:
        return [], {"status": "balanced", "loads": loads}

    cross_wait = WAIT_CROSS if ctx.scene == "A" else DELAY_B
    same_wait = WAIT_SAME if ctx.scene == "A" else 0.0
    rank = []
    for s, duration in enumerate(durs):
        c, index = pos.get(s, (None, None))
        if c is None:
            continue
        release = 0.0
        for p in preds[s] if s < len(preds) else ():
            pc = sol.core_of_sg[p]
            release = max(release, ends[p] + (cross_wait if pc != c else 0.0))
        slack = max(0.0, ends[s] - release - duration)
        rank.append((max(0.0, duration) + 0.25 * slack, s, c, index, release))
    rank.sort(reverse=True)
    heavy = [c for c in range(5) if loads[c] > mean * 1.05]
    light = [c for c in range(5) if loads[c] < mean * 0.95]
    actions = []
    for score, s, source, index, release in rank:
        if source not in heavy:
            continue
        targets = sorted(light, key=lambda c: (max(tails[c], release), loads[c], c))
        for target in targets[:3]:
            if target == source:
                continue
            child = sol.clone()
            child.move_sg_core(s, target)
            child.compact()
            if not child.validate(ctx.model, 5):
                continue
            mk, added, _ = ctx.model.evaluate(
                child.sg_of_block, child.core_of_sg, ctx.scene, 5,
                use_cache=False)
            if (mk, added) < (base_mk, base_added):
                actions.append((mk, added, child, ("move_core", s, source, target)))
        if len(actions) >= max_actions:
            break

    actions.sort(key=lambda row: (row[0], row[1]))
    unique = []
    seen = set()
    for row in actions:
        sig = (tuple(row[2].sg_of_block), tuple(row[2].core_of_sg))
        if sig not in seen:
            seen.add(sig)
            unique.append(row)
    return unique[:max_actions], {"status": "ok", "base_makespan": base_mk,
                                  "loads": loads, "heavy": heavy,
                                  "candidates": len(unique)}


__all__ = ["generate_candidates"]
