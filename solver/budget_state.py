"""可恢复的动作、唯一评分与逻辑精评预算，缓存复用不赠送名额。"""
class BudgetState:
    def __init__(self, max_attempts=1024, max_states=64, max_exact=6):
        for v in (max_attempts,max_states,max_exact):
            if type(v) is not int or v<0: raise ValueError('预算必须为非负整数')
        self.max_attempts,self.max_states,self.max_exact=max_attempts,max_states,max_exact
        self.attempts=0;self.scored=set();self.exact_ids=[]

    def attempt(self):
        if self.attempts>=self.max_attempts:return False
        self.attempts+=1;return True

    def score(self,plan_id):
        if plan_id in self.scored:return True
        if len(self.scored)>=self.max_states:return False
        self.scored.add(plan_id);return True

    def exact(self,plan_id):
        if len(self.exact_ids)>=self.max_exact:return False
        self.exact_ids.append(plan_id);return True

    def to_dict(self):
        return {'max_attempts':self.max_attempts,'max_states':self.max_states,
                'max_exact':self.max_exact,'attempts':self.attempts,
                'scored':sorted(self.scored),'exact_ids':list(self.exact_ids)}

    @classmethod
    def from_dict(cls,value):
        obj=cls(value['max_attempts'],value['max_states'],value['max_exact'])
        if type(value['attempts']) is not int or not 0<=value['attempts']<=obj.max_attempts:
            raise ValueError('检查点动作计数非法')
        obj.attempts=value['attempts'];obj.scored=set(value['scored']);obj.exact_ids=list(value['exact_ids'])
        if len(obj.scored)>obj.max_states or len(obj.exact_ids)>obj.max_exact:
            raise ValueError('检查点已用额度超出预算')
        return obj
