"""累计重复搬运驱动的完整消费者块移动；不假定跨任务缓存复用。"""
from collections import defaultdict

from scene_a_event import step1_schedule
from spill_events import count_spill_events


def audit_tensor_aggregation(evaluator, plan):
    """用原官方逐出记录验证每个逻辑张量，诊断成本单列。"""
    from schedule_step2 import step2_spill_insertion
    result = []
    for task,local in evaluator.local_graphs(plan):
        seq = step1_schedule(local)
        stats = count_spill_events(local,seq,evaluator.capacity,tensor_limit=max(1,len(local['tensors'])))
        official = step2_spill_insertion(local,seq,evaluator.capacity)
        truth = {}
        for event in official['spill_records']:
            tid = event['logical_tid']
            row = truth.setdefault(tid, {'read_bytes':0,'write_bytes':0,'evictions':0,'reload_by_op':{}})
            row['read_bytes'] += event['size']
            row['write_bytes'] += event['size']*int(event['spill_out_copies_data'])
            row['evictions'] += 1
            oid = event['next_use_op']
            row['reload_by_op'][oid] = row['reload_by_op'].get(oid,0)+1
        predicted = {r['tensor_id']:{k:r[k] for k in ('read_bytes','write_bytes','evictions','reload_by_op')}
                     for rows in stats['spill_tensors'].values() for r in rows}
        if predicted != truth:
            raise ValueError(f'任务{task}累计逻辑张量与原官方不一致')
        result.append({'task':task,'tensors':len(truth),'evictions':stats['eviction_count'],
                       'spill_bytes':stats['spill_bytes'],'equal':True})
    return result


def move_region(model, parent, region, target, num_cores):
    """同时移动全部块后才判商图；空源压缩，不进行自动修环。"""
    if not region or target < 0 or target >= len(parent.blocks_in_sg) or not parent.blocks_in_sg[target]:
        return None
    if any(b < 0 or b >= len(model.blocks) for b in region):
        return None
    sources = {parent.sg_of_block[b] for b in region}
    if len(sources) != 1 or target in sources:
        return None
    child = parent.clone()
    for b in sorted(region):
        child.move_block(b, target)
    child.compact()
    if not child.validate(model, num_cores):
        return None
    assert all({b for b,s in enumerate(child.sg_of_block) if s == task} == members
               for task,members in enumerate(child.blocks_in_sg))
    return child


def necessary_region(model, parent, seeds, target, adjacency=None):
    """补齐种子与目标成员间所有有向路径；第三任务中间块只报告，不吸收。"""
    if adjacency is None:
        pred, succ = defaultdict(set), defaultdict(set)
        for (a,b), _ in model.block_edges:
            succ[a].add(b); pred[b].add(a)
    else:
        pred, succ = adjacency
    anchors = set(seeds) | parent.blocks_in_sg[target]
    def reach(edges):
        found, stack = set(anchors), sorted(anchors)
        while stack:
            for b in sorted(edges.get(stack.pop(), ())):
                if b not in found:
                    found.add(b); stack.append(b)
        return found
    hull = reach(pred) & reach(succ)
    sources = {parent.sg_of_block[b] for b in seeds}
    foreign = {b for b in hull if parent.sg_of_block[b] not in sources | {target}}
    return {b for b in hull if parent.sg_of_block[b] in sources}, foreign


def generate_hotspot_candidates(model, parent, plan, num_cores, evaluator, estimate,
                                max_tasks=3, tensor_limit=3, max_blocks=8, max_targets=4):
    if not parent.validate(model,num_cores):
        raise ValueError('热点移动要求合法父方案')
    tasks = sorted((s for s,t in estimate['tasks'].items() if t['spill_bytes'] > 0),
                   key=lambda s: (-estimate['tasks'][s]['spill_bytes'], s))[:max_tasks]
    audit = {'target_tasks': tasks, 'generated': 0, 'legal': 0, 'invalid': 0,
             'duplicates': 0, 'foreign_closure': 0, 'no_target': 0, 'hotspots': []}
    if not tasks:
        return [], audit
    pred, succ = defaultdict(set), defaultdict(set)
    for (a,b), _ in model.block_edges:
        pred[b].add(a); succ[a].add(b)
    rows, seen = [], set()
    for task, local in evaluator.local_graphs(plan):
        if task not in tasks:
            continue
        sequence = step1_schedule(local)
        stats = count_spill_events(local, sequence, evaluator.capacity, tensor_limit=tensor_limit)
        for pool, hot in stats['spill_tensors'].items():
            for tensor in hot:
                tid = tensor['tensor_id']
                # 补边界仅新建DDR张量；片上受害者仍保留原逻辑张量编号。
                if tid not in evaluator.tensors:
                    raise ValueError('局部热点缺少原始逻辑张量映射')
                cb = {model.block_of_op[o] for o in evaluator.consumers[tid] if o in model.block_of_op}
                pb = {model.block_of_op[o] for o in evaluator.producers[tid] if o in model.block_of_op}
                consumers = cb & parent.blocks_in_sg[task]
                reloads = defaultdict(int)
                for oid,count in tensor['reload_by_op'].items():
                    if oid in model.block_of_op:
                        reloads[model.block_of_op[oid]] += count
                blocks = sorted(consumers, key=lambda b: (-reloads[b], model.block_pos[b], b))[:max_blocks]
                shared = {parent.sg_of_block[b] for b in cb | pb} - {task}
                adjacent = {parent.sg_of_block[b] for a in consumers for b in pred[a] | succ[a]} - {task}
                targets = sorted(shared | adjacent,
                    key=lambda s: (-len(cb & parent.blocks_in_sg[s]), -len(pb & parent.blocks_in_sg[s]), s))[:max_targets]
                note = {'task': task, 'pool': pool, **tensor, 'producer_blocks': sorted(pb),
                        'consumer_blocks': sorted(cb), 'selected_blocks': blocks, 'targets': targets,
                        'omitted_consumer_blocks': len(consumers)-len(blocks),
                        'omitted_targets': len(shared | adjacent)-len(targets)}
                audit['hotspots'].append(note)
                if not targets:
                    audit['no_target'] += 1
                seeds = [('single', {b}) for b in blocks]
                if len(blocks)>1:
                    seeds.append(('aggregate', set(blocks)))
                for target in targets:
                    for kind, seed in seeds:
                        region, foreign = necessary_region(model,parent,seed,target,(pred,succ))
                        variants = [(kind, seed)]
                        if foreign:
                            audit['foreign_closure'] += 1
                        elif region != seed:
                            variants.append((kind+'_closed', region))
                        for action, moved in variants:
                            audit['generated'] += 1
                            child = move_region(model,parent,moved,target,num_cores)
                            if child is None:
                                audit['invalid'] += 1
                                continue
                            key = tuple(child.sg_of_block), tuple(child.core_of_sg)
                            if key in seen:
                                audit['duplicates'] += 1
                                continue
                            seen.add(key); audit['legal'] += 1
                            remaining = consumers-moved
                            rows.append({'sol':child, 'source':{
                                'kind':'hotspot_'+action, 'task':task, 'target_task':target,
                                'tensor_id':tid, 'pool':pool, 'moved_blocks':sorted(moved),
                                'seed_blocks':sorted(seed), 'closure_added':len(moved-seed),
                                'hotspot_bytes':tensor['total_bytes'], 'hotspot_evictions':tensor['evictions'],
                                'remaining_consumer_blocks':len(remaining),
                                'remaining_original_reload_count':sum(reloads[b] for b in remaining),
                                'moved_original_reload_count':sum(reloads[b] for b in moved),
                                'moved_matrix_work':sum(model.block_work_m[b] for b in moved),
                                'moved_vector_work':sum(model.block_work_v[b] for b in moved)}})
    return rows,audit
