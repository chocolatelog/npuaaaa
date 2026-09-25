"""溢出事件附近的完整块闭合集切分；不改变默认块化或原始算子。"""
from collections import defaultdict

from scene_a_event import step1_schedule
from spill_events import count_spill_events


def closed_prefix(model, members, positions, cut, predecessors):
    prefix = {b for b in members if all(positions[op] < cut for op in model.blocks[b])}
    crossed = sum(min(positions[op] for op in model.blocks[b]) < cut <=
                  max(positions[op] for op in model.blocks[b]) for b in members)
    initial = len(prefix)
    pending = sorted(prefix)
    while pending:
        b = pending.pop()
        for p in sorted(predecessors.get(b, ())):
            if p in members and p not in prefix:
                prefix.add(p); pending.append(p)
    return prefix, {'straddling_blocks': crossed, 'closure_added_blocks': len(prefix)-initial,
                    'prefix_blocks': len(prefix)}


def generate_peak_candidates(model, parent, plan, num_cores, evaluator, estimate,
                             max_tasks=3, event_limit=3):
    if not parent.validate(model, num_cores):
        raise ValueError('峰值切分要求合法父方案')
    targets = sorted((s for s in estimate['tasks'] if estimate['tasks'][s]['spill_bytes'] > 0),
                     key=lambda s: (-estimate['tasks'][s]['spill_bytes'], -estimate['tasks'][s]['duration'], s))[:max_tasks]
    audit = {'target_tasks': targets, 'generated': 0, 'legal': 0, 'invalid': 0,
             'duplicates': 0, 'empty_or_full': 0, 'boundary_notes': []}
    if not targets:
        return [], audit
    preds = defaultdict(set)
    for (a,b), _ in model.block_edges:
        preds[b].add(a)
    rows, seen = [], set()
    for task, local in evaluator.local_graphs(plan):
        if task not in targets:
            continue
        sequence = step1_schedule(local)
        positions = {op:i for i,op in enumerate(sequence)}
        stats = count_spill_events(local, sequence, evaluator.capacity, event_limit=event_limit)
        members = parent.blocks_in_sg[task]
        old_core = parent.core_of_sg[task]
        for pool, events in stats['spill_events'].items():
            for event in events:
                for cut in (event['step'], event['step']+1, event['next_use_step']):
                    prefix, note = closed_prefix(model, members, positions, cut, preds)
                    note = {'task': task, 'pool': pool, 'tensor': event['tensor_id'], 'cut': cut, **note}
                    audit['boundary_notes'].append(note)
                    if not prefix or prefix == members:
                        audit['empty_or_full'] += 1
                        continue
                    suffix = members-prefix
                    wm, wv = [0.0]*num_cores, [0.0]*num_cores
                    for b, group in enumerate(parent.sg_of_block):
                        if b not in suffix:
                            c = parent.core_of_sg[group]
                            wm[c] += model.block_work_m[b]; wv[c] += model.block_work_v[b]
                    sm = sum(model.block_work_m[b] for b in suffix)
                    sv = sum(model.block_work_v[b] for b in suffix)
                    placements = sorted((max(max(wm[c]+(sm if c==target else 0), wv[c]+(sv if c==target else 0))
                                              for c in range(num_cores)), target) for target in range(num_cores))[:2]
                    for load, target in placements:
                        child = parent.clone(); new = len(child.core_of_sg)
                        child.core_of_sg.append(target); child.blocks_in_sg.append(set())
                        for b in sorted(suffix):
                            child.move_block(b, new)
                        child.compact(); audit['generated'] += 1
                        if not child.validate(model, num_cores):
                            audit['invalid'] += 1
                            continue
                        key = tuple(child.sg_of_block), tuple(child.core_of_sg)
                        if key in seen:
                            audit['duplicates'] += 1
                            continue
                        seen.add(key); audit['legal'] += 1
                        rows.append({'sol': child, 'source': {'kind': 'spill_boundary',
                            'task': task, 'event': event, 'cut': cut, 'boundary': note,
                            'target_core': target, 'original_core': old_core,
                            'compute_load': load, 'parent_spill': stats['spill_bytes'],
                            'parent_peak_alloc': stats['peak_alloc_bytes']}})
    return rows, audit
