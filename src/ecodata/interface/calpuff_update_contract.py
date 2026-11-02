"""Only allow the requested point-source/receptor group to change on reconnect."""
import hashlib,re
from ecodata.models.adapters.calpuff_receptors import TOKEN


def _parts(raw,mode):
    text=raw.decode('ascii');tokens=list(TOKEN.finditer(text))
    keys=[]
    for token in tokens:
        m=re.fullmatch(r'\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*',token[1],re.S)
        keys.append(m[1].upper() if m else token[1].strip().upper())
    def unique(key):
        indexes=[i for i,k in enumerate(keys) if k==key]
        if len(indexes)!=1:raise ValueError('更新范围不能明确定位：'+key)
        return indexes[0]
    if mode=='constant_points':
        begin,end=unique('NPT1'),unique('NAR1')
        if begin>=end:raise ValueError('点源组顺序错误')
        mutable=set(range(begin,end))
        allowed={'NPT1','NPT2','IPTU','NSPT1','SRCNAM','X','SIGYZI','FMFAC','ZPLTFM','END'}
        if any(keys[i] not in allowed for i in mutable):raise ValueError('点源替换组含不支持的活动参数')
    elif mode=='receptors':
        begin=unique('NREC')
        if keys[begin+1:begin+3]!=['NRGRP','END']:raise ValueError('受体组需明确NRGRP与END')
        if not re.fullmatch(r'\s*NRGRP\s*=\s*0\s*',tokens[begin+1][1],re.I):raise ValueError('当前不支持受体分组变化')
        mutable={begin,*range(begin+3,len(tokens))}
        if any(keys[i] not in {'X','END'} for i in range(begin+3,len(tokens))):raise ValueError('受体替换组含其他活动参数')
    else:raise ValueError('更新类别未支持，不替换完整控制')
    immutable=[t[1].strip() for i,t in enumerate(tokens) if i not in mutable]
    return text.splitlines()[0],immutable


def validate_update(before,after,mode):
    if _parts(before,mode)!=_parts(after,mode):
        raise ValueError('转换控制改变了目标组以外的科学设置或文件引用；拒绝替换，不覆盖原工程')
    return dict(mode=mode,before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),
                unrelated_active_controls_preserved=True,scope='Requested constant-point or discrete-receptor group only; native validation still required')
