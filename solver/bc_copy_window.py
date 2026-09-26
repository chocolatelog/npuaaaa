"""B/C固定分核的关键搬运前移窗口；用真实展开图检验合法性后官方取优。"""
from collections import defaultdict
from copy import deepcopy

from bc_observed_graph import observe_parent,expand,operation_tags,replay_expanded
from official_protocol import object_digest
from dag_priority import build_compute_dag


def generate_critical_copy_candidates(graph,parent,raw,*,max_candidates=6,max_expanded=12,max_span=64,dependency_closure=False,expander=None,candidate_scorer=None):
    if type(max_candidates) is not int or not 1<=max_candidates<=6:raise ValueError('关键搬运候选预算必须1～6')
    if type(max_expanded) is not int or max_expanded<max_candidates:raise ValueError('展开预算不得小于官方候选预算')
    if type(dependency_closure) is not bool:raise ValueError('依赖闭包开关必须为布尔值')
    state=observe_parent(graph,parent,raw)
    positions={g:(c,i) for c,row in enumerate(parent['core_schedules']) for i,g in enumerate(row)}
    group_preds=defaultdict(set)
    if dependency_closure:
        _,preds,_,_=build_compute_dag(graph)
        mapping={int(k):v for k,v in parent['node_to_subgraph'].items()}
        for i,ps in preds.items():
            for p in ps:
                if mapping[p]!=mapping[i]:group_preds[mapping[i]].add(mapping[p])
    pipe_groups=defaultdict(set)
    for (core,_),o in state['observed'].items():
        if o['subgraph_id'] in positions:pipe_groups[core,o['pipe']].add(o['subgraph_id'])
    actions={}
    for key in state['replay']['critical']:
        op=state['observed'][key];g=op['subgraph_id'];core=key[0]
        if op['op'] not in ('COPY_IN','COPY_OUT') or g not in positions:continue
        c,at=positions[g]
        if c!=core:raise ValueError('官方搬运子图所属核心异常')
        before=sorted((positions[h][1] for h in pipe_groups[core,op['pipe']] if positions[h][1]<at and at-positions[h][1]<=max_span),reverse=True)
        for distance in (1,4,8):
            if len(before)<distance:continue
            to=before[distance-1];action_key=(core,g,to)
            action=dict(core=core,subgraph=g,at=at,to=to,pipe=op['pipe'],op_kind=op['op'],
                observed_copy_duration=op['duration'],crossed_pipe_groups=distance,span=at-to,
                observed_start=op['start'],op_id=op['op_id'])
            if action_key not in actions or op['duration']>actions[action_key]['observed_copy_duration']:actions[action_key]=action
    # 读/写热点交替，避免仅大写出或大读入占满展开预算。
    ranked=sorted(actions.values(),key=lambda a:(-a['observed_copy_duration'],a['span'],a['core'],a['subgraph'],a['to']))
    families={k:[a for a in ranked if a['op_kind']==k] for k in ('COPY_OUT','COPY_IN')}
    ordered=[]
    for i in range(max((len(v) for v in families.values()),default=0)):
        for values in families.values():
            if i<len(values):ordered.append(values[i])
    seen={object_digest(parent)};proposals=[];rejected=[];attempted=0;closure_rejected=0
    for action in ordered:
        if attempted>=max_expanded:break
        plan=deepcopy(parent);row=plan['core_schedules'][action['core']]
        moved={action['subgraph']}
        if dependency_closure:
            pending=list(moved);invalid=False
            while pending:
                g=pending.pop()
                for p in group_preds[g]:
                    core,position=positions[p]
                    if core!=action['core'] or position<action['to'] or p in moved:continue
                    if position>action['at']:invalid=True;break
                    moved.add(p);pending.append(p)
                if invalid or len(moved)>16:invalid=True;break
            if invalid:closure_rejected+=1;continue
        moving=[g for g in row if g in moved]
        row[:]=row[:action['to']]+moving+[g for g in row[action['to']:] if g not in moved]
        key=object_digest(plan)
        if key in seen:continue
        seen.add(key);attempted+=1
        try:tasks,links=(expand if expander is None else expander)(graph,plan,raw)
        except (ValueError,RuntimeError) as exc:rejected.append(dict(action=action,error=str(exc)));continue
        compatible=operation_tags(tasks)==state['tags']
        replay=replay_expanded(tasks,links,state['durations'],raw['cross_core_copy_delay_cycles']) if compatible else None
        estimate=replay['makespan'] if replay else None
        proposals.append(dict(plan=plan,source=dict(module='critical_copy_window',**action,
            dependency_closure=dependency_closure,moved_subgraphs=moving,
            observed_duration_compatible=compatible,observed_duration_estimate=estimate,
            scope='真实展开合法性；固定父时长评分仅为代理，原官方决定总时间')))
        if candidate_scorer is not None:
            proposals[-1]['source'].update(candidate_scorer(plan,tasks,links))
    def key(row):
        s=row['source'];estimate=s['observed_duration_estimate']
        score=s.get('closed_loop_makespan',estimate if estimate is not None else raw['makespan'])
        return (score,s['span'],s['core'],s['subgraph'],s['to'])
    ranked=sorted(proposals,key=key);selected=[];selected_ids=set()
    def add(row):
        ident=object_digest(row['plan'])
        if ident not in selected_ids and len(selected)<max_candidates:selected.append(row);selected_ids.add(ident)
    for row in ranked[:min(3,max_candidates)]:add(row)
    for family in ('COPY_OUT','COPY_IN'):
        candidate=next((r for r in ranked if r['source']['op_kind']==family),None)
        if candidate is not None:add(candidate)
    uncertain=next((r for r in ranked if not r['source']['observed_duration_compatible']),None)
    if uncertain is not None:add(uncertain)
    for row in ranked:add(row)
    return selected,dict(status='ok',critical_ops=len(state['replay']['critical']),actions=len(actions),
        expanded=attempted,legal_expanded=len(proposals),rejected=rejected,candidates=len(selected),
        compatible_expanded=sum(r['source']['observed_duration_compatible'] for r in proposals),
        dependency_closure=dependency_closure,closure_rejected=closure_rejected,
        scored_proposals=[dict(plan_id=object_digest(r['plan']),**r['source']) for r in proposals] if candidate_scorer is not None else [],
        max_span=max_span,max_expanded=max_expanded,scope='固定切分/核心；每流水线队首和跨核复制释放，不使用A整任务串行约束')
