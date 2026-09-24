import json, glob, sys, io
from collections import Counter, defaultdict, deque
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze(path):
    d = json.load(open(path))
    ops = {o['id']: o for o in d['ops']}
    tensors = {t['id']: t for t in d['tensors']}
    prod = {}
    cons = defaultdict(list)
    for e in d['edges']:
        s, t = e['source'], e['target']
        if s in ops:
            prod[t] = s
        else:
            cons[s].append(t)
    op_pred = defaultdict(set)
    for t, lst in cons.items():
        for o in lst:
            if t in prod:
                op_pred[o].add(prod[t])
    op_succ = defaultdict(set)
    for o, ps in op_pred.items():
        for p in ps:
            op_succ[p].add(o)
    indeg = {o: len(op_pred.get(o, ())) for o in ops}
    q = deque(o for o in ops if indeg[o] == 0)
    depth = {o: 0 for o in ops}
    while q:
        u = q.popleft()
        for v in op_succ[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    cops = Counter(ops[o]['op'] for o in ops)
    total_cycles = sum(ops[o]['cycles'] for o in ops)
    name = path.replace('\\', '/').split('/')[-1]
    print(f'{name}: ops={len(ops)} tensors={len(tensors)} depth={max(depth.values())} '
          f'total_cycles={total_cycles} types={dict(cops)}')

for f in sorted(glob.glob('data/case_*.json')):
    analyze(f)
