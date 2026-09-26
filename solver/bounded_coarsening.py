"""按原拓扑顺序增量粗化；在合并前拒绝成环，保留算子数与工作量上限。"""
from collections import defaultdict
import math


def coarsen(topological, predecessors, affinity, *, max_ops, work,
            max_work=None, reach_budget=2048, pack_frontiers=False,frontier_width=None,preserve_entries=False,layer_window=None,entry_ancestors=False):
    if isinstance(max_ops,bool) or not isinstance(max_ops,int) or max_ops<1:
        raise ValueError('块算子上限必须为正整数')
    if isinstance(reach_budget,bool) or not isinstance(reach_budget,int) or reach_budget<1:
        raise ValueError('可达检查预算必须为正整数')
    if max_work is not None and (not math.isfinite(max_work) or max_work<=0):
        raise ValueError('块工作量上限必须有限且大于0')
    if frontier_width is not None and (type(frontier_width) is not int or frontier_width<1):
        raise ValueError('同层并行宽度必须为正整数')
    if layer_window is not None and (type(layer_window) is not int or layer_window<1 or frontier_width is None):
        raise ValueError('层窗口必须为正整数且必须提供并行宽度')
    if type(entry_ancestors) is not bool or (entry_ancestors and not preserve_entries):
        raise ValueError('祖先入口放宽必须与入口保持同时使用')
    frontier_caps={};window_caps={}
    if frontier_width is not None:
        known={};layer_work=defaultdict(float);largest=defaultdict(float)
        for op in topological:
            ps=predecessors.get(op,())
            if op in known or any(p not in known for p in ps):raise ValueError('输入顺序不是无重复拓扑序')
            value=work[op]
            if not math.isfinite(value) or value<0:raise ValueError('工作量必须非负有限')
            layer=max((known[p] for p in ps),default=-1)+1;known[op]=layer
            layer_work[layer]+=value;largest[layer]=max(largest[layer],value)
        frontier_caps={l:max(largest[l],total/frontier_width) for l,total in layer_work.items()}
        if layer_window is not None:
            totals=defaultdict(float);maxima=defaultdict(float)
            for op,level in known.items():
                band=level//layer_window;totals[band]+=work[op];maxima[band]=max(maxima[band],work[op])
            window_caps={b:max(maxima[b],total/frontier_width) for b,total in totals.items()}
    blocks, loads, owner, successors = [], [], {}, defaultdict(set)
    levels, group_levels, frontier_groups = {}, [], defaultdict(set)
    group_inputs=[];group_bands=[];ancestor_masks=[]
    audit={'policy':'bounded_cycle_safe','cycle_rejections':0,'reach_budget_rejections':0,
           'max_ops':max_ops,'max_work':max_work,'reach_budget':reach_budget,
           'oversize_singletons':[]}
    audit.update(pack_frontiers=bool(pack_frontiers),frontier_merges=0)
    if frontier_width is not None:audit.update(frontier_width=frontier_width,frontier_caps=frontier_caps)
    if preserve_entries:audit.update(preserve_entries=True,entry_rejections=0)
    if layer_window is not None:audit.update(layer_window=layer_window,layer_window_caps=window_caps,layer_boundary_rejections=0)
    if entry_ancestors:audit.update(entry_ancestors=True,entry_ancestor_relaxations=0)

    def entry_allowed(group, others):
        if not entry_ancestors:return others<=group_inputs[group]
        return all(ancestor_masks[group] & (1<<p) for p in others)

    def safe_merge(group, other_predecessors):
        # 新操作是已处理子图的汇点。收缩入group会成环，当且仅当
        # group已能到达某个其他前驱组；超预算时保守拒绝，不猜测可达性。
        if not other_predecessors:
            return True
        frontier=list(successors[group]); visited=set()
        while frontier:
            current=frontier.pop()
            if current in other_predecessors:
                audit['cycle_rejections']+=1
                return False
            if current in visited:
                continue
            visited.add(current)
            if len(visited)>reach_budget:
                audit['reach_budget_rejections']+=1
                return False
            frontier.extend(sorted(successors[current],reverse=True))
        return True

    for op in topological:
        if op in owner:
            raise ValueError('原拓扑顺序重复算子')
        value=work[op]
        if not math.isfinite(value) or value<0:
            raise ValueError('工作量必须为非负有限数')
        predecessors_here=predecessors.get(op,())
        if any(p not in owner for p in predecessors_here):
            raise ValueError('输入顺序不是拓扑序')
        level=max((levels[p] for p in predecessors_here),default=-1)+1
        levels[op]=level
        band=level//layer_window if layer_window is not None else None
        work_limit=max_work
        if layer_window is not None:work_limit=min(max_work,window_caps[band]) if max_work is not None else window_caps[band]
        pred_groups={owner[p] for p in predecessors_here}
        traffic=defaultdict(float)
        for p in sorted(predecessors_here):
            traffic[owner[p]]+=affinity.get((p,op),0)
        candidates=sorted((g for g in pred_groups if traffic[g]>0),
                          key=lambda g:(-traffic[g],loads[g],g))
        chosen=None
        for group in candidates:
            if layer_window is not None and group_bands[group]!=band:
                audit['layer_boundary_rejections']+=1;continue
            if preserve_entries and not entry_allowed(group,pred_groups-{group}):
                audit['entry_rejections']+=1
                continue
            if len(blocks[group])>=max_ops:
                continue
            if work_limit is not None and loads[group]+value>work_limit:
                continue
            if safe_merge(group,pred_groups-{group}):
                chosen=group
                break
        if chosen is None and pack_frontiers:
            # 同原始依赖层的操作互不依赖；优先填满同层未关闭的块。
            # 商图可能已含其他跨层合并，仍执行完整的防成环检查。
            eligible=frontier_groups[level]
            if frontier_width is not None:
                eligible=[g for g in eligible if loads[g]+value<=frontier_caps[level]]
            if preserve_entries:
                filtered=[g for g in eligible if entry_allowed(g,pred_groups-{g})]
                audit['entry_rejections']+=len(eligible)-len(filtered)
                eligible=filtered
            for group in sorted(eligible,key=lambda g:(-loads[g],g))[:8]:
                if len(blocks[group])>=max_ops:
                    continue
                if work_limit is not None and loads[group]+value>work_limit:
                    continue
                if safe_merge(group,pred_groups-{group}):
                    chosen=group;audit['frontier_merges']+=1
                    break
        if chosen is None:
            chosen=len(blocks);blocks.append([]);loads.append(0.0)
            group_inputs.append(set(pred_groups))
            group_bands.append(band)
            if entry_ancestors:
                # 所有新增外部前驱均须已在此闭包内，因此该闭包在块增长时不扩张。
                mask=0
                for p in pred_groups:mask|=(1<<p)|ancestor_masks[p]
                ancestor_masks.append(mask)
            group_levels.append(level)
            frontier_groups[level].add(chosen)
        elif group_levels[chosen] is not None and group_levels[chosen]!=level:
            frontier_groups[group_levels[chosen]].discard(chosen)
            group_levels[chosen]=None
        if entry_ancestors and not pred_groups-{chosen}<=group_inputs[chosen]:
            audit['entry_ancestor_relaxations']+=1
        blocks[chosen].append(op);loads[chosen]+=value;owner[op]=chosen
        if group_levels[chosen] is not None and (len(blocks[chosen])>=max_ops or
                 (work_limit is not None and loads[chosen]>=work_limit)):
            frontier_groups[group_levels[chosen]].discard(chosen)
        for p in pred_groups-{chosen}:
            successors[p].add(chosen)
        if max_work is not None and value>max_work:
            audit['oversize_singletons'].append(op)
    audit.update(block_count=len(blocks),max_actual_ops=max(map(len,blocks),default=0),
                 max_actual_work=max(loads,default=0))
    if entry_ancestors:audit['ancestor_bit_payload_bytes']=sum((m.bit_length()+7)//8 for m in ancestor_masks)
    return blocks,audit
