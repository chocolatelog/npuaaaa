"""Conservative global compute-only lower bounds, independent of coarsening."""
import heapq
import math
from collections import defaultdict

PIPES = ('PIPE_M', 'PIPE_V', 'PIPE_MTE2', 'PIPE_MTE3')
COPY_TYPES = {'COPY_IN', 'COPY_OUT'}


def build_compute_dag(graph_json):
    raw_ops, tensors = graph_json['ops'], graph_json.get('tensors', [])
    ops = {op['id']: op for op in raw_ops}
    tensor_ids = {t['id'] for t in tensors}
    if len(ops) != len(raw_ops) or len(tensor_ids) != len(tensors):
        raise ValueError('duplicate node IDs')
    if set(ops) & tensor_ids:
        raise ValueError('op and tensor IDs overlap')
    nodes = set(ops) | tensor_ids
    if any(type(n) is not int for n in nodes):
        raise ValueError('IDs must be integers')
    succ = {n: set() for n in nodes}
    indegree = dict.fromkeys(nodes, 0)
    for edge in graph_json.get('edges', []):
        u, v = edge['source'], edge['target']
        if type(u) is not int or type(v) is not int or u not in nodes or v not in nodes:
            raise ValueError('invalid edge endpoint')
        if u == v:
            raise ValueError('self-loop')
        if v not in succ[u]:
            succ[u].add(v)
            indegree[v] += 1
    ready = [n for n in nodes if indegree[n] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        u = heapq.heappop(ready)
        order.append(u)
        for v in sorted(succ[u]):
            indegree[v] -= 1
            if not indegree[v]:
                heapq.heappush(ready, v)
    if len(order) != len(nodes):
        raise ValueError('input graph must be a DAG')
    compute = {i: op for i, op in ops.items() if op['op'] not in COPY_TYPES}
    durations = {}
    for i, op in compute.items():
        if op['pipe'] not in PIPES:
            raise ValueError('unsupported pipe')
        value = op.get('cycles', 1)
        if type(value) is not int:
            raise ValueError('certificates require integer operation cycles')
        durations[i] = max(1, value)
    # Original precedence survives contraction. Removed COPY/tensor nodes cost
    # zero in this relaxation; optional cross-core communication is never added.
    frontier = defaultdict(set)
    preds = {i: set() for i in compute}
    for u in order:
        if u in compute:
            preds[u] = set(frontier[u])
            passing = {u}
        else:
            passing = frontier[u]
        for v in succ[u]:
            frontier[v].update(passing)
    topo = [i for i in order if i in compute]
    return compute, preds, topo, durations


def compute_certificate(graph_json, num_cores, singlecore_makespan=None,
                        incumbent_makespan=None):
    if type(num_cores) is not int or num_cores < 1:
        raise ValueError('num_cores must be a positive integer')
    ops, preds, topo, durations = build_compute_dag(graph_json)
    loads = {pipe: sum(durations[i] for i in ops if ops[i]['pipe'] == pipe)
             for pipe in PIPES}
    work = {pipe: (value + num_cores - 1) // num_cores
            for pipe, value in loads.items()}
    paths, parent = {}, {}
    for i in topo:
        p = max(sorted(preds[i]), key=lambda p: paths[p]) if preds[i] else None
        parent[i] = p
        paths[i] = durations[i] + (paths[p] if p is not None else 0)
    tail = max(topo, key=lambda i: paths[i]) if topo else None
    longest = paths[tail] if tail is not None else 0
    witness = []
    while tail is not None:
        witness.append(tail)
        tail = parent[tail]
    witness.reverse()
    lower = max([longest] + list(work.values()))
    result = dict(scope='global_compute_only_relaxation', certified=True,
                  num_cores=num_cores, pipe_slots_per_core=1,
                  pipe_work_cycles=loads, pipe_work_bounds=work,
                  dependency_path=longest, critical_path_ops=witness,
                  overall=lower, attainment_proven=False,
                  omitted=['copy durations', 'cross-core synchronization',
                           'DDR contention', 'cache spills', 'partition constraints'])
    for name, value in [('singlecore', singlecore_makespan), ('incumbent', incumbent_makespan)]:
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int,float))
                                  or not math.isfinite(value) or value <= 0):
            raise ValueError(name + ' makespan must be finite and positive')
    if incumbent_makespan is not None:
        if lower > incumbent_makespan:
            raise ValueError('certificate contradicts official incumbent')
        result['incumbent_makespan'] = incumbent_makespan
        result['incumbent_consistent'] = True
        result['relative_gap_upper_bound'] = (incumbent_makespan/lower-1) if lower else None
    if singlecore_makespan is not None:
        result['speedup_upper_bound'] = singlecore_makespan/lower if lower else None
    return result
