"""新结构候选的共享资源重放适配；仍须原官方确认后才输出结果。"""
from pathlib import Path
import time
from local_template_cache import TemplateCache
from official_protocol import compact_result, digest
from scene_a_replay import build_resource_evaluator
from scene_a_fast import official
from evaluation_validation import read_evaluation_config
from scenario_contract import validate_plan


class SceneACandidateScorer:
    def __init__(self,graph,config_path,cache_bytes=64*1024*1024):
        self.graph=graph;self.config_path=Path(config_path)
        self.config_sha=digest(self.config_path)
        config=read_evaluation_config(str(self.config_path))
        scene=official.read_scene_a_config(str(self.config_path))
        self.config=config;self.scene=scene
        self.cache=TemplateCache(max_entries=128,max_bytes=cache_bytes)
        self.replay=build_resource_evaluator(unified=True,template_cache=self.cache)

    def evaluate(self,plan):
        if digest(self.config_path)!=self.config_sha:raise ValueError('评分期间配置改变')
        validate_plan(self.graph,plan,'A');start=time.perf_counter()
        raw=self.replay(self.graph,plan,self.config['bandwidth'],self.config['capacity'],
            self.scene['task_cross_core_wait_cycles'],self.scene['task_same_core_wait_cycles'])
        return dict(real=compact_result(raw),elapsed=time.perf_counter()-start,
            shared_ddr_peak=max((e['active_count'] for e in raw['ddr_contention_log']),default=0),
            busy_by_core=[sum(t['duration'] for t in c['tasks']) for c in raw['per_core_timeline']],
            cache=dict(self.cache.stats),scope='candidate_replay_requires_official_confirmation')
