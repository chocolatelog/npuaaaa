"""整数服务工作、动态公平共享与整数退休；仅负责一个独立带宽池。"""
import math

EPSILON=1e-9


class SharedServicePool:
    """剩余工作以独占周期计，耗尽服务不等于执行器已退休。

    调用方先退休同刻完成指令，再发射新指令。缓存命中、依赖、核心与
    流水线发射顺序均由调用方决定；本类不把观测的发射序列冒充预测。
    """
    def __init__(self):
        self.remaining={}
        self.now=0.0
        self.version=0

    def _time(self,now):
        if isinstance(now,bool) or not isinstance(now,(int,float)) or not math.isfinite(now) or now<self.now:
            raise ValueError('事件时间必须单调且为有限非负数')
        return float(now)

    def advance(self,now):
        now=self._time(now);elapsed=now-self.now
        if not elapsed:return
        while elapsed>EPSILON:
            active=[item for item,work in self.remaining.items() if work>EPSILON]
            if not active:break
            minimum=min(self.remaining[item] for item in active)
            finish_delta=minimum*len(active)
            if finish_delta>=elapsed-EPSILON:
                share=elapsed/len(active)
                for item in active:self.remaining[item]=max(0.0,self.remaining[item]-share)
                break
            for item in active:self.remaining[item]=max(0.0,self.remaining[item]-minimum)
            elapsed-=finish_delta
        self.now=now;self.version+=1

    def finish_times(self):
        # 编号不参加浮点分组，只以插入序稳定处理同工作量请求。
        ordered=sorted(self.remaining.items(),key=lambda pair:pair[1])
        result={};cursor=self.now;previous=0.0;active=len(ordered);i=0
        while i<len(ordered):
            work=max(0.0,ordered[i][1]);cursor+=(work-previous)*active
            j=i
            while j<len(ordered) and abs(ordered[j][1]-work)<=EPSILON:
                result[ordered[j][0]]=int(math.ceil(cursor-EPSILON));j+=1
            active-=j-i;previous=work;i=j
        return result

    def issue(self,request_id,service_cycles,now):
        self._time(now)
        try:duplicate=request_id in self.remaining
        except TypeError as exc:raise ValueError('请求编号必须可哈希') from exc
        if duplicate:raise ValueError('请求编号重复')
        if type(service_cycles) is not int or service_cycles<1:
            raise ValueError('独占服务工作必须为正整数周期')
        self.advance(now)
        self.remaining[request_id]=float(service_cycles)
        self.version+=1
        return self.finish_times()

    def retire(self,request_ids,now):
        now=self._time(now);request_ids=list(request_ids)
        if len(set(request_ids))!=len(request_ids):raise ValueError('退休请求重复')
        expected=self.finish_times()
        if any(item not in expected or expected[item]>now for item in request_ids):
            raise ValueError('请求不存在或尚未到退休时间')
        self.advance(now)
        for item in request_ids:self.remaining.pop(item)
        if request_ids:self.version+=1
        return self.finish_times()
