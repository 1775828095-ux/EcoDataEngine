"""Associate all inventoried engineering text with the sources it describes."""
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from secrets import randbelow
import re

from .document_content import classify

PRIVATE = re.compile(
    r'https?://[^\s<>"\x27]+|(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|'
    r'(?:[A-Za-z]:[\\/]|\\\\|/(?:home|mnt|tmp|Users|data)/)[^\n"<>|]+|'
    r'\b(?:sk|pk)-[A-Za-z0-9_-]{8,}|\b[0-9a-f]{32,}\b|'
    r'\b\d{4}-\d{2}-\d{2}(?:T[\d:.]+(?:Z|[+-]\d{2}:\d{2})?)?\b|'
    r'(?P<label>密钥|密码|令牌|联系人|电话|邮箱|项目编号|项目名称|项目名|地名|流域名称|站点名|姓名|地址|'
    r'API[_ -]?KEY|password|token|author|contact|project[_ -]?name|site[_ -]?name|location|address|coordinates?|lat(?:itude)?|lon(?:gitude)?)'
    r'(?P<sep>["\x27]?\s*[:：=]\s*)'
    r'(?P<value>"[^"\n]*"|\x27[^\x27\n]*\x27|\[[^\]\n]*\]|[^,\n;；|}]+)', re.I
)
FILE = re.compile(r'(?<![\w.%+~@/\\()-])[\w.%+~@/\\()-]*[\w-]\.(?:nc4?|csv|tsv|xlsx?|parquet|zip|7z|tiff?|json|ya?ml|md|txt|shp|gz)\b', re.I)
NUMBER = re.compile(r'(?<![\w.])[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?(?![\w.]|\d)')
PRODUCTS = {
    "CMORPH": re.compile(r"\bCMORPH\b|(?<![A-Za-z])cmorph(?=_)",re.I),
    "NASA POWER": re.compile(r"\bNASA\s+POWER\b|\bPRECTOTCORR\b|(?<![A-Za-z])POWER_regional",re.I),
}

def perturb(match):
    try:
        number=Decimal(match[0])
        if not -20<=number.adjusted()<=20:
            return "[数值]"
        return format(number+Decimal(randbelow(999)+1)/1000,"f")
    except InvalidOperation:
        return "[数值]"

def redact_private(match):
    if match["label"] is None:
        return "[敏感信息]"
    value=match["value"]
    replacement='"[敏感信息]"' if value.startswith('"') else "'[敏感信息]'" if value.startswith("'") else "[敏感信息]"
    return match["label"]+match["sep"]+replacement

def associate_documents(scan, source_ids):
    rows=scan["rows"]
    names=defaultdict(set)
    for index,row in enumerate(rows,1):
        name=row["name"].casefold()
        names[name].add(index)
        names[PurePosixPath(name).name].add(index)
    pattern=re.compile(r"(?<![\w.-])(?:"+"|".join(re.escape(x) for x in sorted(names,key=len,reverse=True))+r")(?![\w.-])",re.I) if names else None
    refs=defaultdict(list)
    documents=[]
    row_docs={row["name"]:row for row in rows if "engineering_text_status" in row}
    read_docs=dict(scan.get("engineering_docs",()))
    for index,name in enumerate(dict.fromkeys([*row_docs,*read_docs]),1):
        content=read_docs.get(name)
        status=row_docs.get(name,{}).get("engineering_text_status","complete")
        doc_id=f"E{index}"
        links=defaultdict(set)
        document=classify(name,content) if content is not None else None
        if content is not None:
            association_text=document.text if document.text is not None else ""
            own_index=next(iter(names.get(name.casefold(),())),None)
            if own_index in source_ids:
                links[source_ids[own_index]].add("本文件")
            for linked,source in source_ids.items():
                if any(e.get("document")==name for e in rows[linked-1].get("evidence",[])):
                    links[source].add("工程说明引用或上下文关联")
            for match in pattern.finditer(association_text) if pattern else ():
                hits=names[match[0].casefold()]
                if len(hits)==1:
                    linked=next(iter(hits))
                    if linked in source_ids:
                        links[source_ids[linked]].add("文件引用")
            if PurePosixPath(name).suffix.casefold() in {".md",".rst",".txt"}:
                directories={match.casefold() for match in re.findall(r"\x60([^\x60\r\n]+/)\x60",association_text)}
                for linked,source in source_ids.items():
                    if rows[linked-1].get("engineering_probe") and any(
                            rows[linked-1]["name"].casefold().startswith(folder) for folder in directories):
                        links[source].add("工程说明引用所在目录")
            for name_suffix in (".receipt.json",".inventory.json"):
                if name.casefold().endswith(name_suffix):
                    hits=names.get(name[:-len(name_suffix)].casefold(),set())
                    for linked in hits:
                        if linked in source_ids:
                            links[source_ids[linked]].add("同路径清单或凭据")
            for product,rule in PRODUCTS.items():
                if rule.search(association_text):
                    for linked,source in source_ids.items():
                        row=rows[linked-1]
                        if any(rule.search(str(value)) for value in (row["name"],*(f["name"] for f in row["fields"]))):
                            links[source].add(f"文档提及产品 {product}")
            status={"full":"complete","mixed":"data_sections_local","data":"data_payload_local",
                    "unknown":"classification_uncertain"}[document.mode]
            masked=None
            if document.text is not None:
                # Invalid Unicode code units cannot be included in the UTF-8 API request.
                association_text=re.sub(r"[\ud800-\udfff]","[无效字符]",association_text)
                masked=PRIVATE.sub(redact_private,association_text)
            tokens=[]
            def mask_file(match):
                hits=names[match[0].casefold()]
                token=source_ids.get(next(iter(hits)),"文件") if len(hits)==1 else "文件"
                tokens.append(token)
                return chr(0xF0000+len(tokens)-1)
            if masked is not None:
                masked=pattern.sub(mask_file,masked) if pattern else masked
                masked=NUMBER.sub(perturb,FILE.sub("[文件]",masked))
                if tokens:
                    masked=re.sub(r"[\U000f0000-\U000ffffd]",lambda m:f"[{tokens[ord(m[0])-0xF0000]}]"
                                  if ord(m[0])-0xF0000<len(tokens) else "[文件]",masked)
        else:
            masked=None
        for source in links:
            refs[source].append(doc_id)
        documents.append(dict(id=doc_id,read_status=status,scope="source_reference" if links else "project_context",
                              source_ids=sorted(links),link_evidence={key:sorted(value) for key,value in links.items()},
                              text=masked, sharing=document.mode if document else "unread",
                              classification_reason=document.reason if document else "未读取正文",
                              **({"data_structure":document.structure} if document and document.structure else {}),
                              **({"replaced_sections":list(document.removed)} if document and document.removed else {})))
    return documents,dict(refs)
