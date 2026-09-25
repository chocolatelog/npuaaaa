"""可复现性基础设施：稳定种子、内容哈希、代码版本、结果清单。

设计目标（验收标准）：
1. 同一环境连续运行两次 → 计划/指标/汇总完全一致；
2. 结果清单是唯一事实源：已完成的任务按（种子+输入哈希+配置哈希+
   代码版本）命中即跳过，不重复求解、不覆盖；
3. 官方评估结果按"方案哈希"判失效——文件存在但哈希不匹配时重评。
"""
import hashlib
import json
import os
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, '..', 'results'))
SOLVE_MANIFEST = os.path.join(RESULTS, 'solve_manifest.jsonl')
EVAL_MANIFEST = os.path.join(RESULTS, 'eval_manifest.jsonl')


def stable_seed(*parts):
    """跨进程/跨版本稳定的种子：crc32(拼接串)。
    替换 hash((case, scene, n))（Python hash 每进程随机化，不可复现）。"""
    return zlib.crc32('-'.join(str(p) for p in parts).encode('utf-8')) & 0x7fffffff


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def code_version() -> str:
    """solver/*.py 内容哈希（按文件名排序后拼接）——代码变更即版本变更。"""
    h = hashlib.sha256()
    for name in sorted(os.listdir(HERE)):
        if name.endswith('.py'):
            h.update(name.encode())
            with open(os.path.join(HERE, name), 'rb') as f:
                h.update(f.read())
    return h.hexdigest()[:16]


def load_manifest(path):
    """读取 JSONL 清单 -> dict[key] = record（保留最新一条）。"""
    out = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    out[r['_key']] = r
                except Exception:
                    pass
    return out


def append_manifest(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')


def solve_key(case, scene, n, phase='base'):
    return f'{phase}|{case}|{scene}|{n}'


def eval_key(case, prob, n):
    return f'{case}|{prob}|{n}'
