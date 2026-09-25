"""可选保守收缩：拓扑安全、负载与搬运联合判断、软任务数量。

只作为独立候选来源，不替换生产收缩。输入须已经合法。
接受条件是计算负载增加不超过边界搬运节省；这是候选启发式，
不是官方耗时保证，不含依赖等待和溢出。必须保留原始候选并官方复核。
"""
from batch_features import scalar_features


def admissible_pairs(model, sol):
    """有向无环商图中的边，若有替代路径则收缩后成环，必须排除。"""
    outgoing = {s:set() for s in sorted(set(sol.sg_of_block))}
    for (u,v),_w in model.block_edges:
        a,b=sol.sg_of_block[u],sol.sg_of_block[v]
        if a!=b:
            outgoing[a].add(b)
    pairs=[]
    for a in sorted(outgoing):
        for b in sorted(outgoing[a]):
            pending=list(outgoing[a]-{b}); seen=set(); alternate=False
            while pending:
                x=pending.pop()
                if x==b:
                    alternate=True
                    break
                if x in seen:
                    continue
                seen.add(x);pending.extend(outgoing[x]-seen)
            if not alternate:
                pairs.append((a,b))
    return pairs


def safe_coarsen(model, raw, num_cores, target, ops_cap, merge_limit=256,
                 evaluator=None, max_candidates_per_step=512):
    """对每个合法合并尝试两端核心；显卡仅计算同一批候选的相同特征。

    候选过多时按省通信字节、原任务编号确定性筛至预算以内。
    记录筛选数量，不把预算约束下未找到改进解释成全邻域无潜力。
    单次最多 merge_limit 次成功合并，目标不可达时允许保持更多任务。
    """
    if min(target,ops_cap,merge_limit,max_candidates_per_step)<1:
        raise ValueError('目标数量、粒度和搜索上限必须为正')
    sol=raw.clone();sol.compact()
    if not sol.validate(model,num_cores):
        raise ValueError('输入构造不合法，请先单独核对输入')
    trace=[]; candidate_count=0; dropped=0
    initial=scalar_features(model,sol.sg_of_block,sol.core_of_sg,num_cores)
    while sol.num_used_sg()>target and len(trace)<merge_limit:
        current=scalar_features(model,sol.sg_of_block,sol.core_of_sg,num_cores)
        edge_weight={}
        for (u,v),weight in model.block_edges:
            a,b=sol.sg_of_block[u],sol.sg_of_block[v]
            if a!=b:
                edge_weight[(a,b)]=edge_weight.get((a,b),0)+weight
        candidates=[]
        for a,b in sorted(admissible_pairs(model,sol),key=lambda x:(-edge_weight.get(x,0),x)):
            members=sol.blocks_in_sg[a]|sol.blocks_in_sg[b]
            if sum(len(model.blocks[x]) for x in members)>ops_cap:
                continue
            for core in sorted({sol.core_of_sg[a],sol.core_of_sg[b]}):
                if len(candidates)>=max_candidates_per_step:
                    dropped+=1
                    continue
                child=sol.clone();child.merge_sg(a,b)
                owner=child.sg_of_block[min(members)]
                child.core_of_sg[owner]=core;child.compact()
                candidates.append((a,b,core,child))
        if not candidates:
            break
        aa=[x[3].sg_of_block for x in candidates]
        cc=[x[3].core_of_sg for x in candidates]
        values=(evaluator.evaluate(aa,cc) if evaluator is not None else
                [scalar_features(model,a,c,num_cores) for a,c in zip(aa,cc)])
        candidate_count+=len(candidates)
        accepted=[]
        for candidate,features in zip(candidates,values):
            delta=(features['compute_load']-current['compute_load']
                   -(current['boundary_bytes']-features['boundary_bytes'])/60.0)
            if delta<=0:
                accepted.append((delta,features['compute_load'],features['boundary_bytes'],
                                 *candidate[:3],candidate[3]))
        if not accepted:
            break
        best=min(accepted,key=lambda x:x[:6]);sol=best[-1]
        if not sol.validate(model,num_cores):
            raise AssertionError('合并后完整商图检查失败')
        trace.append({'pair':list(best[3:5]),'core':best[5],'delta':best[0],
                      'compute_load':best[1],'boundary_bytes':best[2],'tasks':sol.num_used_sg()})
    audit={'initial_features':initial,
           'final_features':scalar_features(model,sol.sg_of_block,sol.core_of_sg,num_cores),
           'trace':trace,'candidate_count':candidate_count,'candidates_dropped_by_budget':dropped,
           'target_reached':sol.num_used_sg()<=target,'merge_limit_hit':len(trace)>=merge_limit,
           'backend':str(evaluator.device) if evaluator is not None else 'scalar_cpu'}
    return sol,audit
