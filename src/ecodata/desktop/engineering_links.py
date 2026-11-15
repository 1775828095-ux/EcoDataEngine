"""Local evidence linking by upload-relative path and document context."""
from collections import defaultdict
from pathlib import PurePosixPath
import re

from .document_content import classify

ROLE_PATTERNS = {
    "background":r"背景场|background(?: field)?|first[- ]guess",
    "observations":r"站点观测|实测(?:资料|数据)|station observations?|observations?|in situ",
    "target":r"目标网格|target grid|target coordinates",
    "forcing":r"气象驱动|降水驱动|meteorological forcing",
    "validation":r"独立验证|用于验证|留出验证|independent validation|validation (?:data|reference)",
    "comparison":r"对照数据|对照来源|comparison source|用于对照",
}
_ROLE_REGEX = {role:re.compile(rule,re.I) for role,rule in ROLE_PATTERNS.items()}


def collect(rows,docs):
    lookup=defaultdict(list)
    for row in rows:
        name=row["name"].casefold()
        basename=PurePosixPath(name).name
        lookup[name].append(row)
        if basename!=name:
            lookup[basename].append(row)
    if not lookup:
        return
    matcher=re.compile(r"(?<![\w./-])(?:" + "|".join(re.escape(name) for name in
                       sorted(lookup,key=len,reverse=True)) + r")(?![\w./-])",re.I)
    for document,text in docs:
        description=classify(document,text)
        if not description.text:
            continue
        parent=PurePosixPath(document).parent
        current=None
        for line in description.text.replace("\\","/").splitlines():
            if not line.strip():
                current=None
                continue
            mentions=[]
            for match in matcher.finditer(line):
                key=match[0].casefold()
                local=lookup.get((parent/key).as_posix().casefold(),[])
                hits=local if len(local)==1 and local[0]["name"].casefold()==(parent/key).as_posix().casefold() else lookup[key]
                if len(hits)==1 and hits[0]["name"]!=document:
                    mentions.append((match,hits[0]))
            if mentions:
                current=mentions[0][1] if len({id(row) for _,row in mentions})==1 else None
            elif current is None or not re.match(r"\s|[-*]",line):
                continue
            for i,(match,row) in enumerate(mentions or [(None,current)]):
                if len(mentions)>1:
                    before=mentions[i-1][0].end() if i else 0
                    after=mentions[i+1][0].start() if i+1<len(mentions) else len(line)
                    # A multi-file row needs a clause boundary before assigning a role.
                    left=max(line.rfind(sep,before,match.start()) for sep in "；;。")
                    right=min((pos for sep in "；;。" if (pos:=line.find(sep,match.end(),after))>=0), default=-1)
                    if i and left<0 or i+1<len(mentions) and right<0:
                        fragment=""
                    else:
                        fragment=line[left+1:right if right>=0 else len(line)]
                else:
                    fragment=line
                roles=[role for role,rule in _ROLE_REGEX.items() if rule.search(fragment)]
                evidence=dict(document=document,line=line.strip()[:420],roles=roles)
                if len(row["evidence"])<12 and evidence not in row["evidence"]:
                    row["evidence"].append(evidence)
                for role in roles:
                    if role not in row["roles"]:
                        row["roles"].append(role)
