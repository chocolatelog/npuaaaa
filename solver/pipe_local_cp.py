"""Bounded CP-SAT proposals; aggregate objectives are NOT official bounds."""
from collections import defaultdict
import copy
import heapq
import json
import time

from bound_certificate import build_compute_dag, PIPES


def _topo(groups, edges):
    succ = {g: set() for g in groups}
    indegree = dict.fromkeys(groups, 0)
    for u,v in edges:
        if u != v and v not in succ[u]:
            succ[u].add(v)
            indegree[v] += 1
    ready = [g for g in groups if not indegree[g]]
    heapq.heapify(ready)
    order = []
    while ready:
        u=heapq.heappop(ready)
        order.append(u)
        for v in sorted(succ[u]):
            indegree[v]-=1
            if not indegree[v]:
                heapq.heappush(ready,v)
    if len(order)!=len(groups):
        raise ValueError('cyclic group quotient or core order')
    return order


def _view(plan, ops, preds, num_cores):
    raw=plan['node_to_subgraph']
    mapping={int(k):v for k,v in raw.items()}
    if len(mapping)!=len(raw) or set(mapping)!=set(ops):
        raise ValueError('plan must cover each compute op exactly once')
    if any(type(v) is not int or v<0 for v in mapping.values()):
        raise ValueError('invalid subgraph id')
    rows=plan['core_schedules']
    if len(rows)!=num_cores or any(not isinstance(r,list) for r in rows):
        raise ValueError('invalid core schedules')
    flat=[g for row in rows for g in row]
    if any(type(g) is not int for g in flat) or len(flat)!=len(set(flat)) or set(flat)!=set(mapping.values()):
        raise ValueError('subgraphs must be scheduled exactly once')
    cores={g:c for c,row in enumerate(rows) for g in row}
    groups=defaultdict(list)
    for i,g in mapping.items():
        groups[g].append(i)
    edges={(mapping[p],mapping[i]) for i in ops for p in preds[i] if mapping[p]!=mapping[i]}
    order=_topo(groups,edges)
    core_edges={(a,b) for row in rows for a,b in zip(row,row[1:])}
    _topo(groups, edges|core_edges)
    return mapping,dict(groups),cores,edges,order


def _split(plan, ops, preds, topo, durations, num_cores, parts, count):
    result=copy.deepcopy(plan)
    mapping,groups,cores,edges,order=_view(result,ops,preds,num_cores)
    ranking=sorted(groups,key=lambda g:(-sum(durations[i] for i in groups[g]),g))
    pos={i:p for p,i in enumerate(topo)}
    next_id=max(groups,default=-1)+1
    for g in ranking[:count]:
        ids=sorted(groups[g],key=pos.get)
        if len(ids)<2:
            continue
        pieces=min(parts,len(ids))
        # Ordered slices preserve the original DAG; no original edge is deleted.
        slices=[ids[len(ids)*k//pieces:len(ids)*(k+1)//pieces] for k in range(pieces)]
        new=[g]+list(range(next_id,next_id+pieces-1))
        next_id+=pieces-1
        for sid,part in zip(new,slices):
            for i in part:
                result['node_to_subgraph'][str(i)]=sid
        row=result['core_schedules'][cores[g]]
        at=row.index(g)
        row[at:at+1]=new
    _view(result,ops,preds,num_cores)
    return result


def _solve(plan, ops, preds, durations, n, seconds, movable_limit, delay, seed):
    from ortools.sat.python import cp_model
    mapping,groups,cores,edges,order=_view(plan,ops,preds,n)
    load={g:{p:sum(durations[i] for i in ids if ops[i]['pipe']==p) for p in PIPES}
          for g,ids in groups.items()}
    length={g:max(load[g].values()) for g in groups}
    core_load=[max(sum(load[g][p] for g in groups if cores[g]==c) for p in PIPES) for c in range(n)]
    ranking=sorted(groups,key=lambda g:(-core_load[cores[g]],-length[g],g))
    movable=set(ranking[:movable_limit])
    horizon=sum(length.values())+(len(groups)+1)*delay+1
    cp=cp_model.CpModel()
    starts={g:cp.new_int_var(0,horizon,f's_{g}') for g in groups}
    ends={g:cp.new_int_var(0,horizon,f'e_{g}') for g in groups}
    assignments={}
    resources=defaultdict(list)
    for g in groups:
        cp.add(ends[g]==starts[g]+length[g])
        assignments[g]=[cp.new_bool_var(f'x_{g}_{c}') for c in range(n)]
        cp.add_exactly_one(assignments[g])
        for c in range(n):
            cp.add_hint(assignments[g][c],int(c==cores[g]))
            if g not in movable:
                cp.add(assignments[g][c]==int(c==cores[g]))
            for pipe,work in load[g].items():
                if work:
                    resources[c,pipe].append(cp.new_optional_fixed_size_interval_var(
                        starts[g],work,assignments[g][c],f'i_{g}_{c}_{pipe}'))
    for intervals in resources.values():
        cp.add_no_overlap(intervals)
    for u,v in sorted(edges):
        cp.add(starts[v]>=ends[u])
        if delay:
            for c in range(n):
                cp.add(starts[v]>=ends[u]+delay).only_enforce_if(
                    [assignments[u][c],assignments[v][c].Not()])
    # Outside this bounded neighborhood the original group order is retained.
    for row in plan['core_schedules']:
        fixed=[g for g in row if g not in movable]
        for u,v in zip(fixed,fixed[1:]):
            cp.add(starts[v]>=ends[u])
    finish=cp.new_int_var(0,horizon,'finish')
    cp.add_max_equality(finish,list(ends.values()))
    cp.minimize(finish)
    solver=cp_model.CpSolver()
    solver.parameters.max_time_in_seconds=seconds
    solver.parameters.num_search_workers=1
    solver.parameters.random_seed=seed%2147483647
    status=solver.solve(cp)
    stats=dict(status=solver.status_name(status),
               bound_scope='restricted aggregate surrogate, not official',
               objective=solver.objective_value if status in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None,
               surrogate_best_bound=solver.best_objective_bound,
               wall_seconds=solver.wall_time, groups=len(groups), movable=len(movable),
               communication_heuristic=delay)
    if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        return None,stats
    candidate=copy.deepcopy(plan)
    candidate['core_schedules']=[[] for _ in range(n)]
    for g in sorted(groups,key=lambda g:(solver.value(starts[g]),g)):
        c=next(c for c in range(n) if solver.value(assignments[g][c]))
        candidate['core_schedules'][c].append(g)
    _view(candidate,ops,preds,n)
    return candidate,stats


def generate_local_candidates(graph_json, plan, num_cores=5, seconds_per_solve=2.0,
                              max_solves=6, movable_limit=8, seed=0):
    started=time.monotonic()
    if seconds_per_solve<=0 or max_solves<=0 or movable_limit<=0:
        return [],dict(status='no_budget',solves=0)
    if num_cores!=5:
        return [],dict(status='requires_5_cores',solves=0)
    from ortools.sat.python import cp_model
    ops,preds,topo,durations=build_compute_dag(graph_json)
    _view(plan,ops,preds,num_cores)
    candidates,records=[],[]
    seen={json.dumps(plan,sort_keys=True)}
    configs=[(1,0,500),(2,2,500),(4,2,500),(8,1,500),(4,4,0),(2,8,500)]
    def append(candidate,source,stats):
        sig=json.dumps(candidate,sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            candidates.append(dict(plan=candidate,source=source,stats=stats))
    for index in range(max_solves):
        parts,count,delay=configs[index%len(configs)]
        variant=_split(plan,ops,preds,topo,durations,num_cores,parts,count) if parts>1 else copy.deepcopy(plan)
        if len(set(variant['node_to_subgraph'].values()))>512:
            records.append(dict(status='group_cap',parts=parts))
            continue
        # Splits change partition priorities too; submit them independently.
        if parts>1:
            append(variant,f'split_{parts}_{count}',dict(scope='partition_candidate'))
        result,stats=_solve(variant,ops,preds,durations,num_cores,seconds_per_solve,
                            movable_limit,delay,seed+index)
        stats.update(parts=parts,split_groups=count)
        records.append(stats)
        if result is not None:
            append(result,f'pipe_cp_split{parts}_groups{count}_delay{delay}',stats)
    return candidates,dict(status='ok',solves=sum(r['status']!='group_cap' for r in records),
                           records=records,candidates=len(candidates),
                           elapsed=time.monotonic()-started)
