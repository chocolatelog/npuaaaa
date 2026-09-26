"""任务日历插空遵守两侧等待与前驱，不能套用流水线重叠语义。"""
import sys
from pathlib import Path
import random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'solver'))


def test_earliest_slot_preserves_both_neighbour_waits():
    from task_calendar import earliest_task_slot
    assert earliest_task_slot([(10,20),(40,50)],0,5,3)==(0,0)
    assert earliest_task_slot([(10,20),(40,50)],8,14,3)==(23,1)
    assert earliest_task_slot([(10,20),(40,50)],8,15,3)==(53,2)
    assert earliest_task_slot([(10,20),(40,50)],0,5,3,min_position=1)==(23,1)


def test_calendar_schedule_all_dependencies_and_core_waits():
    from dag_priority import schedule_ready
    for seed in range(20):
        rng=random.Random(seed)
        preds={i:{j for j in range(i) if rng.random()<.15} for i in range(20)}
        durations={i:rng.randrange(1,20) for i in preds}
        for n in (2,3,4,5):
            result=schedule_ready(preds,durations,{},n,same_wait=3,cross_wait=15,placement='insert')
            assert result==schedule_ready(preds,durations,{},n,same_wait=3,cross_wait=15,placement='insert')
            assert sorted(i for row in result['orders'] for i in row)==list(preds)
            for row in result['orders']:
                for p,i in zip(row,row[1:]):assert result['starts'][i]>=result['ends'][p]+3
            for i,ps in preds.items():
                for p in ps:
                    wait=0 if result['core_of'][p]==result['core_of'][i] else 15
                    assert result['starts'][i]>=result['ends'][p]+wait


def test_zero_length_dependency_cannot_be_inserted_before_its_predecessor():
    from dag_priority import schedule_ready
    r=schedule_ready({0:set(),1:{0},2:{1}},{0:0,1:0,2:0},{},1,same_wait=0,cross_wait=0,placement='insert')
    assert r['orders']==[[0,1,2]]
