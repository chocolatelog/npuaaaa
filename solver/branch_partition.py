"""基于反链种子可达集合分离独占分支及共享后缀；独立 E05 候选源。"""
from collections import defaultdict
from itertools import combinations, permutations


def generate_branch_candidates(model, parent, tasks, num_cores,
                               max_levels=3, max_seed_sets=6):
    """保持原块与外部任务不变；候选必须通过完整商图验证。

    依赖边上种子可达掩码只增不减；多个种子共同可达的节点收为后缀，
    防止把汇合算子绑进单一分支，使潜在并行再次串行化。
    """
    if num_cores not in (2, 3, 4, 5) or max_levels < 1 or max_seed_sets < 1:
        raise ValueError('核数或候选预算非法')
    if not parent.validate(model, num_cores):
        raise ValueError('分支精化要求合法父分区')
    succ, pred = defaultdict(set), defaultdict(set)
    for (a, b), _ in model.block_edges:
        succ[a].add(b); pred[b].add(a)
    rows, seen = [], set()
    audit = {'generated': 0, 'legal': 0, 'invalid': 0, 'duplicates': 0,
             'no_wide_level': 0, 'single_block_tasks': 0, 'tasks': []}
    for task in tasks:
        members = parent.blocks_in_sg[task]
        if len(members) < 2:
            audit['single_block_tasks'] += 1
            continue
        topo = sorted(members, key=model.block_pos.get)
        depth, rank, levels = {}, {}, defaultdict(list)
        for b in topo:
            depth[b] = 1 + max((depth[p] for p in pred[b] if p in members), default=-1)
            levels[depth[b]].append(b)
        for b in reversed(topo):
            rank[b] = max(model.block_work_m[b], model.block_work_v[b]) + max(
                (rank[n] for n in succ[b] if n in members), default=0)
        wide = sorted((d for d in levels if len(levels[d]) > 1),
                      key=lambda d: (-len(levels[d]), -sum(rank[b] for b in levels[d]), d))[:max_levels]
        audit['tasks'].append({'task': task, 'blocks': len(members),
                               'max_layer_width': max(map(len, levels.values())), 'levels': wide})
        if not wide:
            audit['no_wide_level'] += 1
            continue
        closures = {}
        def descendants(seed):
            if seed not in closures:
                found, stack = {seed}, [seed]
                while stack:
                    b = stack.pop()
                    for child in sorted(succ[b]):
                        if child in members and child not in found:
                            found.add(child); stack.append(child)
                closures[seed] = found
            return closures[seed]
        for level in wide:
            seeds = sorted(levels[level], key=lambda b: (-rank[b], b))[:4]
            seed_sets = [tuple(seeds[:min(len(seeds), num_cores)])]
            for pair in combinations(seeds, 2):
                if pair not in seed_sets:
                    seed_sets.append(pair)
            for selected in seed_sets[:max_seed_sets]:
                reach = [descendants(b) for b in selected]
                groups = defaultdict(set)
                for b in sorted(members):
                    mask = sum(1 << i for i, group in enumerate(reach) if b in group)
                    key = -1 if mask.bit_count() > 1 else mask
                    groups[key].add(b)
                branches = [1 << i for i in range(len(selected))]
                # 同层种子互不可达，各独占组至少包含自己的种子。
                if any(not groups[key] for key in branches):
                    raise ValueError('反链种子独占组为空，内部拓扑假设被破坏')
                work = {key: (sum(model.block_work_m[b] for b in block_set),
                              sum(model.block_work_v[b] for b in block_set))
                        for key, block_set in groups.items()}
                old_core = parent.core_of_sg[task]
                wm, wv = [0.0] * num_cores, [0.0] * num_cores
                for b, sg in enumerate(parent.sg_of_block):
                    if b not in members:
                        c = parent.core_of_sg[sg]
                        wm[c] += model.block_work_m[b]; wv[c] += model.block_work_v[b]
                for key in (0, -1):
                    if key in work:
                        wm[old_core] += work[key][0]; wv[old_core] += work[key][1]
                placements = []
                for cores in permutations(range(num_cores), len(branches)):
                    mm, vv = list(wm), list(wv)
                    for key, c in zip(branches, cores):
                        mm[c] += work[key][0]; vv[c] += work[key][1]
                    placements.append((max(max(mm), max(vv)), cores))
                for load, cores in sorted(placements)[:2]:
                    child = parent.clone()
                    labels = ([0] if 0 in groups else []) + branches + ([-1] if -1 in groups else [])
                    ids = {labels[0]: task}
                    for label in labels[1:]:
                        ids[label] = len(child.core_of_sg)
                        child.core_of_sg.append(old_core); child.blocks_in_sg.append(set())
                    for key, blocks in groups.items():
                        for b in sorted(blocks):
                            child.move_block(b, ids[key])
                    for key, c in zip(branches, cores):
                        child.core_of_sg[ids[key]] = c
                    child.compact(); audit['generated'] += 1
                    if not child.validate(model, num_cores):
                        audit['invalid'] += 1
                        continue
                    signature = tuple(child.sg_of_block), tuple(child.core_of_sg)
                    if signature in seen:
                        audit['duplicates'] += 1
                        continue
                    seen.add(signature); audit['legal'] += 1
                    source = {'kind': 'branch_regions', 'task': task, 'level': level,
                        'seeds': list(selected), 'branch_cores': list(cores),
                        'compute_load': load, 'region_count': len(groups),
                        'exclusive_work': sum(max(work[k]) for k in branches),
                        'prefix_work': max(work.get(0, (0, 0))),
                        'shared_work': max(work.get(-1, (0, 0)))}
                    rows.append({'sol': child, 'source': source})
    return rows, audit
