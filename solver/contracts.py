"""跨场景不可变数据契约；不依赖求解器、设备或文件系统。"""
from dataclasses import dataclass
import hashlib
import json


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PlanState:
    graph_digest: str
    config_digest: str
    assignments: tuple[tuple[int, int], ...]
    orders: tuple[tuple[int, ...], ...]

    def to_plan(self):
        return {'node_to_subgraph': {str(o): s for o, s in self.assignments},
                'core_schedules': [list(order) for order in self.orders]}

    @property
    def mapping(self):
        return dict(self.assignments)

    @property
    def sg_members(self):
        members = {}
        for op, sg in self.assignments:
            members.setdefault(sg, []).append(op)
        return {sg: tuple(ops) for sg, ops in members.items()}

    @property
    def core_of(self):
        return {sg: c for c, order in enumerate(self.orders) for sg in order}

    @property
    def plan_id(self):
        plan = self.to_plan()
        return content_digest({'n': plan['node_to_subgraph'], 'c': plan['core_schedules']})

    @property
    def identity(self):
        return content_digest((self.graph_digest, self.config_digest, self.plan_id))

    @property
    def structure_id(self):
        # 仅用于结构多样性；不能复用官方结果。
        members = self.sg_members
        return content_digest([[members[s] for s in row] for row in self.orders])


@dataclass(frozen=True)
class Action:
    kind: str
    source_plan_id: str
    groups: tuple[int, ...] = ()
    target_core: int | None = None
    slot: int | None = None
    touched_ops: tuple[int, ...] = ()
    granularity: int | None = None


@dataclass(frozen=True)
class Candidate:
    plan: PlanState
    parent_id: str
    family: str
    actions: tuple[Action, ...]
    generation_index: int


@dataclass(frozen=True)
class Evaluation:
    level: str
    scenario: str
    plan_id: str
    contract_id: str
    status: str
    metrics: tuple[tuple[str, float | int | None], ...]
    evidence_ref: str | None = None
    wall_seconds: float = 0.0

