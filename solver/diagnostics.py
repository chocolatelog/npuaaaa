"""Lower bounds and schedule diagnostics for one submitted plan."""

from model import WAIT_CROSS, WAIT_SAME, DELAY_B


def assignments_from_plan(model, plan, num_cores):
    mapping = plan['node_to_subgraph']
    sg_of_block = []
    for block in model.blocks:
        groups = {int(mapping[str(op)]) for op in block}
        if len(groups) != 1:
            raise ValueError('a model block is split across subgraphs')
        sg_of_block.append(groups.pop())
    groups = set(sg_of_block)
    if groups != set(range(len(groups))):
        raise ValueError('subgraph ids must be contiguous')
    orders = [list(row) for row in plan['core_schedules']]
    if len(orders) != num_cores:
        raise ValueError('wrong number of core schedules')
    flat = [s for row in orders for s in row]
    if len(flat) != len(groups) or set(flat) != groups:
        raise ValueError('each subgraph must appear once in a core schedule')
    core_of_sg = [0] * len(groups)
    for core, row in enumerate(orders):
        for sg in row:
            core_of_sg[sg] = core
    return sg_of_block, core_of_sg, orders


def schedule_diagnostics(model, plan, scene, num_cores, real_makespan=None):
    sg_of_block, core_of_sg, orders = assignments_from_plan(
        model, plan, num_cores)
    estimate, _added, info = model.evaluate(
        sg_of_block, core_of_sg, scene, num_cores,
        use_cache=False, orders_override=orders, diagnostics=True)
    end = info['end_time']
    durations = info['durs']
    if estimate == float('inf'):
        raise ValueError('invalid core order')
    start = [end[s] - durations[s] for s in range(len(end))]
    previous = {}
    idle = [0.0] * num_cores
    for core, row in enumerate(orders):
        earlier = None
        for sg in row:
            previous[sg] = earlier
            idle[core] += max(0.0, start[sg] -
                              (end[earlier] if earlier is not None else 0.0))
            earlier = sg

    parent = {}
    for sg in range(len(end)):
        causes = []
        prev = previous[sg]
        if prev is not None:
            causes.append((end[prev] + (WAIT_SAME if scene == 'A' else 0),
                           prev))
        for pred in info['sg_preds'][sg]:
            delay = (0 if core_of_sg[pred] == core_of_sg[sg] else
                     (WAIT_CROSS if scene == 'A' else DELAY_B))
            causes.append((end[pred] + delay, pred))
        if causes:
            best_time, best_pred = max(causes)
            if best_time >= start[sg] - 1e-6:
                parent[sg] = best_pred
    chain = []
    if end:
        sg = max(range(len(end)), key=lambda s: end[s])
        while sg not in chain:
            chain.append(sg)
            if sg not in parent:
                break
            sg = parent[sg]
        chain.reverse()

    lower = model.lower_bounds(num_cores)
    result = {
        'lower_bounds': {k: round(v, 2) for k, v in lower.items()},
        'proxy_makespan': round(estimate, 2),
        'proxy_schedule_makespan': round(info['schedule_makespan'], 2),
        'proxy_bandwidth_bound': round(info['bandwidth_bound'], 2),
        'proxy_core_idle': [round(x, 2) for x in idle],
        'critical_subgraphs': chain,
    }
    if real_makespan is not None and lower['overall'] > 0:
        result['official_to_bound'] = round(real_makespan /
                                            lower['overall'], 3)
    return result
