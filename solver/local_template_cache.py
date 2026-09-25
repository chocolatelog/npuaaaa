"""有容量上限的进程内局部模板缓存；不读取外部序列化文件。"""
from collections import OrderedDict
import hashlib
import pickle
import time


class TemplateCache:
    """键含完整输入顺序、参数与源码版本；结果每次解码得到独立副本。

    字节上限针对已序列化有效载荷，不是进程内存硬上限。
    缓存只保存在本进程；不将不可信文件交给 pickle（对象序列化）加载。
    """
    def __init__(self,max_entries=128,max_bytes=64*1024*1024,source_fingerprint=''):
        if max_entries<1 or max_bytes<1:
            raise ValueError('缓存数量和字节上限必须为正')
        self.max_entries,self.max_bytes=max_entries,max_bytes
        self.source_fingerprint=source_fingerprint
        self.entries=OrderedDict()
        self.stats={'hits':0,'misses':0,'evictions':0,'payload_bytes':0,
                    'peak_payload_bytes':0,'key_seconds':0.0,'decode_seconds':0.0,
                    'compute_seconds':0.0,'oversized':0}

    def wrap(self,name,function):
        def cached(*args,**kwargs):
            start=time.perf_counter()
            raw=pickle.dumps((self.source_fingerprint,name,args,kwargs),protocol=5)
            key=hashlib.sha256(raw).digest()
            self.stats['key_seconds']+=time.perf_counter()-start
            if key in self.entries:
                self.stats['hits']+=1
                value=self.entries.pop(key);self.entries[key]=value
                start=time.perf_counter()
                result=pickle.loads(value)
                self.stats['decode_seconds']+=time.perf_counter()-start
                return result
            self.stats['misses']+=1
            start=time.perf_counter();result=function(*args,**kwargs)
            self.stats['compute_seconds']+=time.perf_counter()-start
            value=pickle.dumps(result,protocol=5)
            if len(value)>self.max_bytes:
                self.stats['oversized']+=1
                return result
            while self.entries and (len(self.entries)>=self.max_entries or
                                    self.stats['payload_bytes']+len(value)>self.max_bytes):
                _,old=self.entries.popitem(last=False)
                self.stats['payload_bytes']-=len(old);self.stats['evictions']+=1
            self.entries[key]=value;self.stats['payload_bytes']+=len(value)
            self.stats['peak_payload_bytes']=max(self.stats['peak_payload_bytes'],self.stats['payload_bytes'])
            return result
        return cached
