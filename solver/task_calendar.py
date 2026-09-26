"""整任务占用的确定性插空；只返回候选位置，不修改日历或官方语义。"""
import math


def earliest_task_slot(calendar,release,duration,wait,*,min_position=0):
    """在有序不重叠区间中找首个双侧间隔均满足的位置。

    首任务无前置切换等待，末任务无后继等待；内部空隙两边都保留等待。
    min_position用于零时长退化输入，仍须排在所有同核前驱之后。
    """
    if any(not math.isfinite(v) or v<0 for v in (release,duration,wait)):
        raise ValueError('释放、时长和等待必须非负有限')
    if type(min_position) is not int or not 0<=min_position<=len(calendar):
        raise ValueError('最早位置超出日历')
    for position in range(min_position,len(calendar)+1):
        start=max(release,calendar[position-1][1]+wait if position else 0)
        if position==len(calendar) or start+duration+wait<=calendar[position][0]:
            return start,position
    raise AssertionError('日历末尾应总能插入')
