"""范式、粒度、场景和精化层驱动的构造池。"""
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any

from solution import Sol
from construct import heft_construct, strip_construct, chain_construct
from construct_v2 import netbenefit_construct


PARADIGMS = ("heft", "strip", "chain", "netbenefit")
GRANULARITIES = ("coarse", "balanced", "fine")
REFINE_LEVELS = ("R0", "R1", "R2", "R3")

GRANULARITY_PROFILES = {
    "coarse": {"tasks_per_core": 1.5, "min_ops": 256, "max_ops": 2048,
               "min_strips_per_core": 1, "max_strips_per_core": 2},
    "balanced": {"tasks_per_core": 3.0, "min_ops": 128, "max_ops": 1024,
                 "min_strips_per_core": 2, "max_strips_per_core": 4},
    "fine": {"tasks_per_core": 6.0, "min_ops": 64, "max_ops": 512,
             "min_strips_per_core": 4, "max_strips_per_core": 8},
}

SCENE_CONFIG = {
    "A": {"cross_wait": 1000.0, "same_wait": 100.0, "copy_delay": 0.0,
          "traffic_weight": 1.0, "pipe_weight": 0.5, "memory_weight": 0.3,
          "memory_budget": 0.85, "l2_weight": 0.0},
    "B": {"cross_wait": 500.0, "same_wait": 0.0, "copy_delay": 500.0,
          "traffic_weight": 0.7, "pipe_weight": 1.0, "memory_weight": 1.0,
          "memory_budget": 0.85, "l2_weight": 0.0},
    "C": {"cross_wait": 500.0, "same_wait": 0.0, "copy_delay": 500.0,
          "traffic_weight": 0.6, "pipe_weight": 0.8, "memory_weight": 0.8,
          "memory_budget": 0.85, "l2_weight": 1.0},
}


@dataclass(frozen=True)
class GranularityProfile:
    name: str
    target_ops: int
    target_strips: int


@dataclass
class ConstructCandidate:
    paradigm: str
    granularity: str
    scene: str
    refine_level: str
    sol: Sol
    metrics: dict | None = None
    parent_id: str | None = None
    signature: str = ""
    reason: str = ""
    target_ops: int | None = None
    target_strips: int | None = None


def resolve_granularity(model, num_cores, granularity):
    if granularity not in GRANULARITY_PROFILES:
        raise ValueError(f"未知粒度: {granularity}")
    profile = GRANULARITY_PROFILES[granularity]
    n_ops = len(getattr(model, "eligible", ()))
    ops_per_core = n_ops / max(1, num_cores)
    raw_target = math.ceil(n_ops / (num_cores * profile["tasks_per_core"]))
    target_ops = max(profile["min_ops"], min(profile["max_ops"], raw_target))
    blocks = getattr(model, "blocks", ())
    if blocks:
        target_ops = max(target_ops, max(len(block) for block in blocks))
    target_ops = min(target_ops, max(1, n_ops))
    target_strips = math.ceil(num_cores * profile["tasks_per_core"])
    target_strips = max(num_cores, min(target_strips,
                                       profile["max_strips_per_core"] * num_cores))
    # Keep this calculation explicit for callers inspecting the profile.
    _ = ops_per_core
    return GranularityProfile(granularity, target_ops, target_strips)


def make_signature(sol):
    payload = {"sg": list(sol.sg_of_block), "core": list(sol.core_of_sg)}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(raw).hexdigest()


def _candidate(paradigm, granularity, scene, sol, level="R0", parent=None,
               target_ops=None, target_strips=None):
    sol = sol.clone()
    sol.compact()
    return ConstructCandidate(paradigm, granularity, scene, level, sol,
                              parent_id=parent, signature=make_signature(sol),
                              target_ops=target_ops, target_strips=target_strips)


def generate_base_candidates(model, num_cores, scene):
    result = []
    cfg = SCENE_CONFIG[scene]
    # 中图的边界流量与精化评估成本随块数快速上升，保留四类范式的
    # balanced 代表；小图仍完整展开 12 个粒度变体。
    granularities = GRANULARITIES
    if len(getattr(model, "eligible", ())) > 2000:
        granularities = ("balanced",)
    for paradigm in PARADIGMS:
        for granularity in granularities:
            profile = resolve_granularity(model, num_cores, granularity)
            if paradigm == "heft":
                raw = heft_construct(model, num_cores, scene,
                                     max_sg_ops=profile.target_ops,
                                     balance=cfg["traffic_weight"])
            elif paradigm == "strip":
                raw = strip_construct(model, num_cores, scene,
                                      num_strips=profile.target_strips)
            elif paradigm == "chain":
                raw = chain_construct(model, num_cores, scene,
                                      max_sg_ops=profile.target_ops,
                                      max_subgraphs=profile.target_strips)
            else:
                raw = netbenefit_construct(model, num_cores, scene,
                                           max_sg_ops=profile.target_ops,
                                           mem_budget=cfg["memory_budget"])
            raw_sol = _coarsen_solution(model, Sol(raw[0], raw[1]),
                                        profile.target_strips)
            result.append(_candidate(paradigm, granularity, scene, raw_sol,
                                     target_ops=profile.target_ops,
                                     target_strips=profile.target_strips))
    return result


def _score(item):
    if not item.metrics:
        return 0.0
    m = item.metrics
    if "fitness" in m:
        return float(m["fitness"])
    return (float(m.get("makespan", 0.0))
            + float(m.get("added_copy_bytes", 0.0)) / 60.0)


def _spill_value(info):
    """兼容旧评估结果，并把 spill 信号统一成一个可比较指标。"""
    info = info or {}
    if "spill_bytes" in info:
        return float(info["spill_bytes"])
    sig = info.get("spill_sig", {})
    if isinstance(sig, dict):
        return sum(float(sig.get(k, 0.0)) for k in
                   ("L1p", "L1w", "UBp", "UBw", "L1l", "UBl"))
    return 0.0


def _iter_block_edges(model):
    """兼容模型的 (u,v,w) 与测试/轻量模型的 (u,v) 边表示。"""
    for edge in getattr(model, "block_edges", ()):
        if len(edge) == 2 and isinstance(edge[0], (tuple, list)):
            yield int(edge[0][0]), int(edge[0][1])
        else:
            yield int(edge[0]), int(edge[1])


def _coarsen_solution(model, sol, max_subgraphs):
    """沿真实商图边合并子图，统一限制所有范式的最终任务数。"""
    child = sol.clone()
    child.compact()
    limit = max(1, int(max_subgraphs))
    while child.num_used_sg() > limit:
        candidates = []
        for u, v in _iter_block_edges(model):
            su, sv = child.sg_of_block[u], child.sg_of_block[v]
            if su == sv:
                continue
            same_core = child.core_of_sg[su] == child.core_of_sg[sv]
            work = sum(max(model.block_work_m[b], model.block_work_v[b])
                       for b in child.blocks_in_sg[su] | child.blocks_in_sg[sv])
            candidates.append((0 if same_core else 1, work, su, sv))
        if candidates:
            _, _, su, sv = min(candidates)
        else:
            used = [s for s, blocks in enumerate(child.blocks_in_sg) if blocks]
            if len(used) < 2:
                break
            su, sv = used[0], used[1]
        child.merge_sg(su, sv)
        child.compact()
    return child


def _evaluate(items, ctx):
    if ctx is None:
        return items
    for item in items:
        mk, added, info = ctx.evaluate(item.sol)
        spill = _spill_value(info)
        fitness_fn = getattr(ctx, "fitness", None)
        fitness = (fitness_fn(mk, added) if fitness_fn is not None
                   else float(mk) + float(added) / 60.0)
        item.metrics = {"makespan": mk, "added_copy_bytes": added,
                        "spill_bytes": spill,
                        "fitness": fitness,
                        "num_subgraphs": item.sol.num_used_sg()}
    return items


def _deduplicate(items, keep, preserve_paradigm=False, preserve_granularity=False):
    unique = {}
    for item in items:
        key = (item.signature,
               item.paradigm if preserve_paradigm else "*",
               item.granularity if preserve_granularity else "*")
        old = unique.get(key)
        if old is None or _score(item) < _score(old):
            unique[key] = item
    items = sorted(unique.values(), key=_score)
    selected = []
    if preserve_paradigm:
        for paradigm in PARADIGMS:
            item = next((x for x in items if x.paradigm == paradigm), None)
            if item is not None and len(selected) < keep:
                selected.append(item)
    if preserve_granularity:
        for granularity in GRANULARITIES:
            item = next((x for x in items if x.granularity == granularity and x not in selected), None)
            if item is not None and len(selected) < keep:
                selected.append(item)
    for item in items:
        if len(selected) >= keep:
            break
        if item not in selected:
            selected.append(item)
    return selected[:keep]


def _valid_solution(model, sol, max_subgraphs=None):
    """检查覆盖、核编号和商图无环，避免精化制造隐性非法方案。"""
    if not sol.sg_of_block or len(sol.sg_of_block) != len(model.blocks):
        return False
    if not sol.core_of_sg or any(s < 0 or s >= len(sol.core_of_sg)
                                 for s in sol.sg_of_block):
        return False
    if max_subgraphs is not None and sol.num_used_sg() > max_subgraphs:
        return False
    succs = [set() for _ in sol.core_of_sg]
    for bi, bj in _iter_block_edges(model):
        si, sj = sol.sg_of_block[bi], sol.sg_of_block[bj]
        if si != sj:
            succs[si].add(sj)
    indeg = [0] * len(succs)
    for edges in succs:
        for v in edges:
            indeg[v] += 1
    stack = [i for i, d in enumerate(indeg) if d == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return seen == len(succs)


def _candidate_limit(item, num_cores):
    if item.target_strips:
        return max(num_cores, int(item.target_strips))
    return None


def _refine_core(item, model, scene, ctx):
    base = item.sol
    ncores = max(1, getattr(ctx, "num_cores", max(base.core_of_sg) + 1
                             if base.core_of_sg else 1))
    variants = []
    for sg, old_core in enumerate(base.core_of_sg):
        for target in range(ncores):
            if target == old_core:
                continue
            child = base.clone()
            child.core_of_sg[sg] = target
            child.compact()
            if _valid_solution(model, child, _candidate_limit(item, ncores)):
                variants.append(_candidate(item.paradigm, item.granularity,
                                           scene, child, "R1", item.signature,
                                           item.target_ops, item.target_strips))
                if len(variants) >= 8:
                    break
        if len(variants) >= 8:
            break
    if not variants:
        return _candidate(item.paradigm, item.granularity, scene, base, "R1",
                          item.signature, item.target_ops, item.target_strips)
    _evaluate(variants, ctx)
    return min(variants, key=_score)


def _refine_boundary(item, model, scene, ctx):
    # 只沿真实块边 (u,v) 处理子图边界，避免按 block 编号误判相邻关系。
    variants = []
    seen = set()
    for edge_index, (u, v) in enumerate(_iter_block_edges(model)):
        if edge_index >= 64 or len(variants) >= 8:
            break
        su, sv = item.sol.sg_of_block[u], item.sol.sg_of_block[v]
        if su == sv or item.sol.core_of_sg[su] != item.sol.core_of_sg[sv]:
            continue
        for source, target, block in ((su, sv, u), (sv, su, v)):
            child = item.sol.clone()
            child.move_block(block, target)
            child.compact()
            sig = make_signature(child)
            if sig in seen or not _valid_solution(model, child,
                                                   _candidate_limit(item, getattr(ctx, "num_cores", 1))):
                continue
            seen.add(sig)
            variants.append(_candidate(item.paradigm, item.granularity, scene,
                                       child, "R2", item.signature,
                                       item.target_ops, item.target_strips))
    if not variants:
        return _candidate(item.paradigm, item.granularity, scene, item.sol,
                          "R2", item.signature, item.target_ops,
                          item.target_strips)
    _evaluate(variants, ctx)
    return min(variants, key=_score)


def _refine_order(item, model, scene, ctx):
    K = len(item.sol.core_of_sg)
    succ = [set() for _ in range(K)]
    for u, v in _iter_block_edges(model):
        su, sv = item.sol.sg_of_block[u], item.sol.sg_of_block[v]
        if su != sv:
            succ[su].add(sv)
    variants = []
    checked = 0
    for a in range(K):
        for b in range(a + 1, K):
            checked += 1
            if checked > 4096 or len(variants) >= 8:
                break
            if item.sol.core_of_sg[a] == item.sol.core_of_sg[b]:
                continue
            # 直接边先排除；间接依赖由最终商图无环检查兜底。
            if b in succ[a] or a in succ[b]:
                continue
            child = item.sol.clone()
            child.core_of_sg[a], child.core_of_sg[b] = (child.core_of_sg[b],
                                                        child.core_of_sg[a])
            child.compact()
            if _valid_solution(model, child,
                               _candidate_limit(item, getattr(ctx, "num_cores", 1))):
                variants.append(_candidate(item.paradigm, item.granularity,
                                           scene, child, "R3", item.signature,
                                           item.target_ops, item.target_strips))
                if len(variants) >= 8:
                    break
        if checked > 4096 or len(variants) >= 8:
            break
    if not variants:
        return _candidate(item.paradigm, item.granularity, scene, item.sol,
                          "R3", item.signature, item.target_ops,
                          item.target_strips)
    _evaluate(variants, ctx)
    return min(variants, key=_score)


def _keep_non_regressions(children, parents):
    kept = []
    for child, parent in zip(children, parents):
        if child.metrics is not None and parent.metrics is not None and _score(child) > _score(parent):
            parent.reason = f"{child.refine_level}指标未改善"
            kept.append(parent)
        else:
            kept.append(child)
    return kept


def build_construct_candidates(model, num_cores, scene, ctx=None, seed=0,
                               max_base=12, max_r1=8, max_r2=6, max_r3=4,
                               final_seeds=8):
    del seed
    if scene not in SCENE_CONFIG:
        raise ValueError(f"未知场景: {scene}")
    base = _evaluate(generate_base_candidates(model, num_cores, scene), ctx)
    if ctx is None:
        return base
    base = _deduplicate(base, max_base, True, True)
    base_r1 = base[:max_r1]
    r1 = _evaluate([_refine_core(x, model, scene, ctx) for x in base_r1], ctx)
    r1 = _keep_non_regressions(r1, base_r1)
    r1 = _deduplicate(r1, max_r1, True, True)
    r1_r2 = r1[:max_r2]
    r2 = _evaluate([_refine_boundary(x, model, scene, ctx) for x in r1_r2], ctx)
    r2 = _keep_non_regressions(r2, r1_r2)
    r2 = _deduplicate(r2, max_r2, True, False)
    r2_r3 = r2[:max_r3]
    r3 = _evaluate([_refine_order(x, model, scene, ctx) for x in r2_r3], ctx)
    r3 = _keep_non_regressions(r3, r2_r3)
    r3 = _deduplicate(r3, max_r3)
    all_items = base + r1 + r2 + r3
    return _deduplicate(all_items, min(final_seeds, 8), preserve_paradigm=True)


__all__ = ["ConstructCandidate", "SCENE_CONFIG", "GRANULARITY_PROFILES",
           "resolve_granularity", "make_signature", "generate_base_candidates",
           "build_construct_candidates"]
