"""候选计算负载、分区边界特征的分批计算；不替代官方事件评估。"""


def scalar_features(model, assignment, owners, num_cores):
    matrix, vector = [0.0]*num_cores, [0.0]*num_cores
    for b, group in enumerate(assignment):
        c = owners[group]
        matrix[c] += model.block_work_m[b]
        vector[c] += model.block_work_v[b]
    boundary = 0.0
    for producers, consumers, has_out, _pos, size in model.tg:
        pp = {assignment[b] for b in producers}
        cc = {assignment[b] for b in consumers}
        boundary += size*(sum(bool(has_out or cc-{s}) for s in pp)+len(cc-pp))
    load = max(max(matrix), max(vector))
    return {'compute_load': float(load), 'boundary_bytes': float(boundary),
            'coarse_score': float(load + .2*boundary/60.0)}


class BatchFeatures:
    """两层分批：候选批次×张量分块，按显存预算约束临时存在矩阵。

    计算相同张量在相同分区只算一次。矩阵/向量管道独立求和。
    粗分数仅用于候选诊断，不包含等待、关键路径、溢出或共享带宽事件。
    """
    def __init__(self, model, num_cores, device='cuda', batch_size=32,
                 tensor_chunk=64, memory_mb=256):
        import torch
        if batch_size < 1 or tensor_chunk < 1 or memory_mb < 1:
            raise ValueError('候选批次、张量分块和内存预算必须为正')
        self.torch, self.model, self.num_cores = torch, model, num_cores
        self.device = torch.device(device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('请求显卡计算，但当前环境的显卡不可用')
        self.batch_size, self.tensor_chunk = batch_size, tensor_chunk
        self.memory_bytes = memory_mb*1024*1024
        self.work_m = torch.tensor(model.block_work_m, dtype=torch.float64, device=self.device)
        self.work_v = torch.tensor(model.block_work_v, dtype=torch.float64, device=self.device)
        self.last_stats = {}

    def evaluate(self, assignments, owners):
        torch, model = self.torch, self.model
        if len(assignments) != len(owners):
            raise ValueError('候选映射与分核数量不一致')
        result, batches, max_elements = [], 0, 0
        for offset in range(0, len(assignments), self.batch_size):
            aa = assignments[offset:offset+self.batch_size]
            cc = owners[offset:offset+self.batch_size]
            if any(len(a) != len(model.blocks) or not c or any(s < 0 or s >= len(c) for s in a)
                   or any(x < 0 or x >= self.num_cores for x in c) for a,c in zip(aa,cc)):
                raise ValueError('候选覆盖、任务编号或核心编号非法')
            batch = len(aa)
            slots = max(len(c) for c in cc)
            chunk = min(self.tensor_chunk, self.memory_bytes//max(1, 32*batch*slots))
            if chunk < 1:
                raise ValueError('单张量批次超出预算，请减少候选批次')
            groups = torch.tensor(aa, dtype=torch.long, device=self.device)
            block_cores = torch.tensor([[c[s] for s in a] for a,c in zip(aa,cc)],
                                      dtype=torch.long, device=self.device)
            wm = torch.zeros((batch,self.num_cores),dtype=torch.float64,device=self.device)
            wv = torch.zeros_like(wm)
            wm.scatter_add_(1,block_cores,self.work_m.expand(batch,-1))
            wv.scatter_add_(1,block_cores,self.work_v.expand(batch,-1))
            load = torch.maximum(wm.max(dim=1).values,wv.max(dim=1).values)
            boundary = torch.zeros(batch,dtype=torch.float64,device=self.device)
            for start in range(0,len(model.tg),chunk):
                tensors = model.tg[start:start+chunk]
                count = len(tensors)
                max_elements = max(max_elements,batch*count*slots)
                present = []
                for side in (0,1):
                    blocks, ids = [], []
                    for i,t in enumerate(tensors):
                        for block in set(t[side]):
                            blocks.append(block); ids.append(i*slots)
                    mask = torch.zeros((batch,count*slots),dtype=torch.bool,device=self.device)
                    if blocks:
                        bi = torch.tensor(blocks,dtype=torch.long,device=self.device)
                        ti = torch.tensor(ids,dtype=torch.long,device=self.device)
                        mask.scatter_(1,groups[:,bi]+ti[None,:],True)
                    present.append(mask.view(batch,count,slots))
                pp, consumers = present
                has_out = torch.tensor([bool(t[2]) for t in tensors],device=self.device)[None,:,None]
                consumer_count = consumers.sum(dim=2,keepdim=True)
                outgoing = pp & (has_out | (consumer_count-consumers.to(torch.long)>0))
                copies = outgoing.sum(dim=2)+(consumers & ~pp).sum(dim=2)
                sizes = torch.tensor([t[4] for t in tensors],dtype=torch.float64,device=self.device)
                boundary += (copies*sizes[None,:]).sum(dim=1)
            # 各设备除法可能用不同的末位舍入。昂贵统计在设备执行，
            # 回传两个标量后按同一表达式组合，避免近分候选排序随设备变化。
            values = torch.stack((load,boundary),dim=1).cpu().tolist()
            result.extend({'compute_load': wm, 'boundary_bytes': traffic,
                           'coarse_score': wm + .2*traffic/60.0}
                          for wm, traffic in values)
            batches += 1
        self.last_stats = {'device':str(self.device),'candidate_batches':batches,
                           'max_presence_elements':max_elements,'memory_budget_mb':self.memory_bytes//1024//1024}
        return result
