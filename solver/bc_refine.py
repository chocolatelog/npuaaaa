"""第二问/第三问的方案级候选精化。

该模块只处理可交给官方评估器的完整方案，不改变官方事件、缓存或带宽语义。
它把原算子依赖、张量寿命和缓存复用信号转换为少量确定性候选：

* B：同核子图微块拆分、生命周期压力驱动的核内换序、热点子图迁核；
* C：在同一合法表示上优先保持可复用张量相邻，并把复用潜力写入候选档案。

候选生成不读取官方真值，最终选择必须由调用方逐个官方评估并保留父方案保底。
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import heapq
import json


COPY_OPS = {"COPY_IN", "COPY_OUT"}


def _eligible(graph):
    return {int(op["id"]) for op in graph.get("ops", ())
            if op.get("op") not in COPY_OPS}


def _op_edges(graph, eligible):
    ops = {int(op["id"]): op for op in graph.get("ops", ())}
    tensors = {int(t["id"]): t for t in graph.get("tensors", ())}
    producers = defaultdict(set)
    consumers = defaultdict(set)
    direct = []
    for edge in graph.get("edges", ()):
        src, dst = int(edge["source"]), int(edge["target"])
        if src in eligible and dst in eligible:
            direct.append((src, dst))
        elif src in eligible and dst in tensors:
            producers[dst].add(src)
        elif src in tensors and dst in eligible:
            consumers[src].add(dst)
    edges = set(direct)
    for tid, ps in producers.items():
        for p in ps:
            for c in consumers.get(tid, ()):
                if p != c:
                    edges.add((p, c))
    return sorted(edges), producers, consumers, tensors, ops


def _topological(eligible, edges):
    succ = {op: set() for op in eligible}
    indeg = {op: 0 for op in eligible}
    for src, dst in edges:
        if src not in succ or dst not in indeg or dst in succ[src]:
            continue
        succ[src].add(dst)
        indeg[dst] += 1
    ready = [op for op, degree in indeg.items() if degree == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        op = heapq.heappop(ready)
        order.append(op)
        for nxt in sorted(succ[op]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                heapq.heappush(ready, nxt)
    if len(order) != len(eligible):
        raise ValueError("原算子依赖图不是有向无环图")
    return order


def _copy_plan(plan):
    return {
        "node_to_subgraph": {str(k): int(v)
                             for k, v in plan["node_to_subgraph"].items()},
        "core_schedules": [[int(x) for x in row]
                           for row in plan["core_schedules"]],
    }


def _signature(plan):
    normalized = {"node_to_subgraph": {
        str(k): int(v) for k, v in plan["node_to_subgraph"].items()},
        "core_schedules": [[int(x) for x in row]
                           for row in plan["core_schedules"]]}
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_plan(graph, plan):
    """验证覆盖、商图无环和同核依赖顺序，不修改输入。"""
    eligible = _eligible(graph)
    mapping = {int(k): int(v) for k, v in plan.get("node_to_subgraph", {}).items()}
    if set(mapping) != eligible or any(v < 0 for v in mapping.values()):
        return False
    schedules = [[int(x) for x in row] for row in plan.get("core_schedules", ())]
    all_sg = set(mapping.values())
    listed = [sg for row in schedules for sg in row]
    if set(listed) != all_sg or len(listed) != len(set(listed)):
        return False
    core_of = {sg: core for core, row in enumerate(schedules) for sg in row}
    pos = {sg: i for row in schedules for i, sg in enumerate(row)}
    edges, *_ = _op_edges(graph, eligible)
    succ = {sg: set() for sg in all_sg}
    indeg = {sg: 0 for sg in all_sg}
    for src, dst in edges:
        a, b = mapping[src], mapping[dst]
        if a == b:
            continue
        if b not in succ[a]:
            succ[a].add(b)
            indeg[b] += 1
        if core_of[a] == core_of[b] and pos[a] >= pos[b]:
            return False
    for row in schedules:
        for a, b in zip(row, row[1:]):
            if b not in succ[a]:
                succ[a].add(b)
                indeg[b] += 1
    ready = sorted(sg for sg, d in indeg.items() if d == 0)
    seen = 0
    while ready:
        sg = ready.pop(0)
        seen += 1
        for nxt in sorted(succ[sg]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    return seen == len(all_sg)


def _group_state(graph, plan):
    eligible = _eligible(graph)
    edges, producers, consumers, tensors, ops = _op_edges(graph, eligible)
    topo = _topological(eligible, edges)
    rank = {op: i for i, op in enumerate(topo)}
    mapping = {int(k): int(v) for k, v in plan["node_to_subgraph"].items()}
    schedules = [[int(x) for x in row] for row in plan["core_schedules"]]
    core_of = {sg: core for core, row in enumerate(schedules) for sg in row}
    members = defaultdict(list)
    for op, sg in mapping.items():
        members[sg].append(op)
    for sg in members:
        members[sg].sort(key=lambda op: rank[op])
    op_core = {op: core_of[mapping[op]] for op in eligible}
    core_pos = {}
    for core, row in enumerate(schedules):
        for index, sg in enumerate(row):
            for op in members.get(sg, ()):
                core_pos[op] = index
    pressure = defaultdict(float)
    reuse = defaultdict(float)
    for tid, cons in consumers.items():
        ps = producers.get(tid, set())
        size = max(0, int(tensors.get(tid, {}).get("size", 0)))
        if not size or not cons:
            continue
        ppos = min((core_pos.get(p, 0) for p in ps), default=0)
        cpos = sorted(core_pos.get(c, ppos) for c in cons)
        span = max(cpos) - ppos
        for op in ps | cons:
            sg = mapping[op]
            pressure[sg] += size * max(0, span)
        distinct_cores = len({op_core[c] for c in cons})
        reuse[tid] = size * max(0, distinct_cores - 1)
    group_reuse = defaultdict(float)
    for tid, value in reuse.items():
        for op in producers.get(tid, set()) | consumers.get(tid, set()):
            group_reuse[mapping[op]] += value
    cycles = {int(op["id"]): max(1, int(op.get("cycles", 1)))
              for op in graph.get("ops", ())}
    load = [0.0] * len(schedules)
    for op, core in op_core.items():
        load[core] += cycles.get(op, 1)
    return {
        "eligible": eligible, "edges": edges, "rank": rank, "members": members,
        "mapping": mapping, "schedules": schedules, "core_of": core_of,
        "pressure": pressure, "group_reuse": group_reuse, "load": load,
        "producers": producers, "consumers": consumers, "tensors": tensors,
    }


def _with_split(state, sg, cut):
    plan = {"node_to_subgraph": {str(k): int(v) for k, v in state["mapping"].items()},
            "core_schedules": [list(row) for row in state["schedules"]]}
    ops = state["members"][sg]
    if not (1 <= cut < len(ops)):
        return None
    new_sg = max(state["core_of"], default=-1) + 1
    # 子图编号和 core_of 的编号空间相同；新编号必须避开已有编号。
    new_sg = max(state["mapping"].values(), default=-1) + 1
    for op in ops[cut:]:
        plan["node_to_subgraph"][str(op)] = new_sg
    core = state["core_of"][sg]
    row = plan["core_schedules"][core]
    at = row.index(sg)
    row[at:at + 1] = [sg, new_sg]
    return plan


def _with_swap(state, core, index):
    row = state["schedules"][core]
    if index < 0 or index + 1 >= len(row):
        return None
    a, b = row[index], row[index + 1]
    # 交换后仍需通过全局依赖检查；这里只生成状态。
    schedules = [list(x) for x in state["schedules"]]
    schedules[core][index:index + 2] = [b, a]
    return {"node_to_subgraph": {str(k): int(v) for k, v in state["mapping"].items()},
            "core_schedules": schedules}


def _with_move(state, sg, target_core):
    source = state["core_of"][sg]
    if source == target_core:
        return None
    schedules = [list(x) for x in state["schedules"]]
    schedules[source].remove(sg)
    # 先放到目标核尾部，后续合法性检查会拒绝依赖逆序。
    schedules[target_core].append(sg)
    return {"node_to_subgraph": {str(k): int(v) for k, v in state["mapping"].items()},
            "core_schedules": schedules}


def _metrics(graph, plan, scene):
    """廉价、可解释的候选指标；不模拟官方事件。"""
    state = _group_state(graph, plan)
    pressure = sum(state["pressure"].values())
    reuse = sum(state["group_reuse"].values())
    imbalance = max(state["load"], default=0.0) - (sum(state["load"]) /
                                                    max(1, len(state["load"])))
    proxy_error = None
    try:
        from scene_b_features import SceneBFeatures
        info = SceneBFeatures(graph).evaluate(plan, scene)
        lower = float(info.get("lower_bound", 0.0))
        boundary = float(info.get("boundary_service_cycles", 0.0))
    except Exception as exc:
        proxy_error = repr(exc)
        lower, boundary = 0.0, 0.0
    try:
        from resource_state_bc import build_resource_table
        resource = build_resource_table(graph, plan, scene)
        cache = resource.get('cache', {})
        cache_hits = float(cache.get('hit_bytes', 0))
        cache_misses = float(cache.get('miss_bytes', 0))
        l1_peak = float(resource.get('pool_peaks', {}).get('L1', 0))
        ub_peak = float(resource.get('pool_peaks', {}).get('UB', 0))
    except Exception:
        cache_hits = cache_misses = l1_peak = ub_peak = 0.0
    return {
        "lifetime_pressure": float(pressure),
        "cache_reuse_bytes": float(reuse),
        "cache_hit_potential_bytes": cache_hits,
        "cache_miss_bytes": cache_misses,
        "l1_peak_bytes": l1_peak,
        "ub_peak_bytes": ub_peak,
        "load_imbalance": float(imbalance),
        "proxy_lower_bound": lower,
        "proxy_valid": proxy_error is None,
        "proxy_error": proxy_error,
        "proxy_boundary_cycles": boundary,
        "proxy_score": lower + boundary + imbalance,
    }


def generate_bc_candidates(graph, plan, scene, max_candidates=8):
    """生成 B/C 候选及审计信息；列表首项始终是原方案。"""
    if scene not in {"B", "C"}:
        raise ValueError("场景必须是B或C")
    if not validate_plan(graph, plan):
        raise ValueError("输入方案不满足原算子覆盖或拓扑约束")
    state = _group_state(graph, plan)
    candidates = [_copy_plan(plan)]
    actions = [{"action": "parent", "plan": candidates[0]}]
    seen = {_signature(candidates[0])}
    split_count = swap_count = move_count = 0

    # C先做缓存复用聚簇：对复用潜力高的相邻候选尝试换序和迁核，
    # 让第三问真正拥有独立于B的候选来源；仍由全局合法性检查裁决。
    if scene == 'C':
        reuse_groups = sorted(state["members"],
                              key=lambda sg: (-state["group_reuse"].get(sg, 0.0), sg))
        for core, row in enumerate(state["schedules"]):
            for index in range(max(0, len(row) - 1)):
                if len(candidates) >= max_candidates:
                    break
                candidate = _with_swap(state, core, index)
                if candidate is None or not validate_plan(graph, candidate):
                    continue
                signature = _signature(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(candidate)
                actions.append({'action': 'cache_cluster_swap', 'plan': candidate,
                                'core': int(core), 'index': int(index)})
                swap_count += 1
            if len(candidates) >= max_candidates:
                break
        if len(candidates) < max_candidates and len(state["schedules"]) > 1:
            for sg in reuse_groups[:3]:
                target_order = sorted(range(len(state["schedules"])),
                                      key=lambda c: (state["load"][c], c))
                for target in target_order:
                    if len(candidates) >= max_candidates:
                        break
                    candidate = _with_move(state, sg, target)
                    if candidate is None or not validate_plan(graph, candidate):
                        continue
                    signature = _signature(candidate)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    candidates.append(candidate)
                    actions.append({'action': 'cache_cluster_move', 'plan': candidate,
                                    'source_sg': int(sg), 'target_core': int(target)})
                    move_count += 1
                if len(candidates) >= max_candidates:
                    break

    # 高压力大块优先细化；用 16/4/1 的层级边界选择切点。
    groups = sorted(state["members"],
                    key=lambda sg: (-state["pressure"].get(sg, 0.0), sg))
    for sg in groups:
        ops = state["members"][sg]
        if len(ops) < 2:
            continue
        cuts = sorted({max(1, len(ops) // 2), min(len(ops) - 1, 16),
                       min(len(ops) - 1, 4)})
        for cut in cuts:
            candidate = _with_split(state, sg, cut)
            if candidate is None or not validate_plan(graph, candidate):
                continue
            signature = _signature(candidate)
            if signature in seen:
                continue
            seen.add(signature)
            candidates.append(candidate)
            actions.append({"action": "split", "plan": candidate,
                            "source_sg": int(sg), "cut": int(cut)})
            split_count += 1
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    # B优先尝试压力相邻换序；C优先尝试复用潜力相邻换序。
    if len(candidates) < max_candidates:
        for core, row in enumerate(state["schedules"]):
            ordered = sorted(range(max(0, len(row) - 1)),
                             key=lambda i: -(
                                 state["group_reuse"].get(row[i], 0.0)
                                 if scene == "C" else
                                 state["pressure"].get(row[i], 0.0)))
            for index in ordered:
                candidate = _with_swap(state, core, index)
                if candidate is None or not validate_plan(graph, candidate):
                    continue
                signature = _signature(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(candidate)
                actions.append({"action": "reuse_swap" if scene == "C" else "pressure_swap",
                                "plan": candidate, "core": int(core),
                                "index": int(index)})
                swap_count += 1
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

    # 热点迁核是静态的“临时大核”候选：每个子图只归属一个真实核心，
    # 通过拆分和迁移组合产生并行分支，不引入运行时屏障或虚构核心。
    if len(candidates) < max_candidates and len(state["schedules"]) > 1:
        for sg in groups[:3]:
            for target in sorted(range(len(state["schedules"])),
                                 key=lambda c: (state["load"][c], c)):
                candidate = _with_move(state, sg, target)
                if candidate is None or not validate_plan(graph, candidate):
                    continue
                signature = _signature(candidate)
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(candidate)
                actions.append({"action": "move_group", "plan": candidate,
                                "source_sg": int(sg), "target_core": int(target)})
                move_count += 1
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

    metrics = []
    for item in actions[:len(candidates)]:
        row = {k: v for k, v in item.items() if k != "plan"}
        row.update(_metrics(graph, item["plan"], scene))
        row["plan_id"] = _signature(item["plan"])
        metrics.append(row)
    return candidates[:max_candidates], {
        "scene": scene,
        "generated": len(actions),
        "returned": len(candidates[:max_candidates]),
        "split_candidates": split_count,
        "swap_candidates": swap_count,
        "move_candidates": move_count,
        "candidate_metrics": metrics,
        "parent_plan_id": _signature(plan),
    }
