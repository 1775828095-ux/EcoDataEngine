"""Discover cross-source combinations from the approved field catalogue."""
from collections import defaultdict, deque
from itertools import combinations
import json

from . import discovery, request_batches, service
from .decision_checks import CHECKS, LocalChecks
from .scan import assert_current


def fields(sources):
    return [field for source in sources for field in source["fields"]]


def prepare(summary, *, rejected=None):
    sources=summary["sources"]
    ids=[source["id"] for source in sources]
    field_ids=[field["id"] for field in fields(sources)]
    if len(set(ids))!=len(ids) or len(set(field_ids))!=len(field_ids):
        raise ValueError("数据来源或字段编号重复")
    docs=request_batches.unique(summary.get("engineering_documents",[]))
    clean=dict(summary,candidates=[],engineering_documents=docs)

    def draft(block, documents):
        payload=dict(clean,sources=block,engineering_documents=documents)
        if evidence:=payload.pop("final_evidence",None):
            ids={field["id"] for field in fields(block)}
            relevant=dict(evidence,results=[row for row in evidence["results"] if set(row["fields"])<=ids],
                          previous_decisions=[row for row in evidence["previous_decisions"] if set(row["fields"])<=ids])
            payload["final_evidence"]=relevant
            request=service.preview(payload,local_evidence=relevant)
        else:
            request=service.preview(payload)
        hint=service.retry_capacity_hint(request,rejected) if rejected is not None else ""
        return request_batches.Batch(request,tuple(row["id"] for row in block),hint)

    first=draft(sources,docs)
    if not first.blocked:
        return [first]

    def count(block):
        return sum(len(row["fields"]) for row in block)

    def split(block):
        if len(block)>1:
            middle=len(block)//2
            return block[:middle],block[middle:]
        row=block[0]
        middle=len(row["fields"])//2
        return [dict(row,fields=row["fields"][:middle])],[dict(row,fields=row["fields"][middle:])]

    def pack(left,right=(),*,documents=docs):
        by_id={}
        for source in [*left,*right]:
            by_id.setdefault(source["id"],dict(source,fields=[]))["fields"].extend(source["fields"])
        refs={ref for source in by_id.values() for ref in source.get("engineering_documents",[])}
        attached=[doc for doc in documents if doc["id"] in refs or doc.get("scope")=="project_context"
                  or by_id.keys() & set(doc.get("source_ids",[]))]
        batch=draft(list(by_id.values()),attached)
        if not batch.blocked:
            return [batch]
        if not right and count(left)>1:
            a,b=split(left)
            return [*pack(a,documents=documents),*pack(b,documents=documents),*pack(a,b,documents=documents)]
        if right and count(left)>=count(right) and count(left)>1:
            a,b=split(left)
            return [*pack(a,right,documents=documents),*pack(b,right,documents=documents)]
        if count(right)>1:
            a,b=split(right)
            return [*pack(left,a,documents=documents),*pack(left,b,documents=documents)]
        if attached:
            # When even the smallest field block cannot share one request
            # with the documents, send every section with that same block.
            return [draft(list(by_id.values()),part) for part in split_documents(attached,list(by_id.values()))]
        return [batch]

    def split_documents(documents,block):
        if not draft(block,documents).blocked:
            return [documents]
        if len(documents)>1:
            middle=len(documents)//2
            return [*split_documents(documents[:middle],block),*split_documents(documents[middle:],block)]
        doc=documents[0]
        text=doc.get("text") or ""
        if len(text)<2 or draft(block,[]).blocked:
            return [documents]
        middle=len(text)//2
        boundary=text.rfind("\n",max(1,middle//2),middle)
        if boundary>0:
            middle=boundary+1
        # Offsets and the document ID preserve the ordering of every section.
        offset=doc.get("text_offset",0)
        prefix=dict(doc,text=text[:middle],text_offset=offset)
        suffix=dict(doc,text=text[middle:],text_offset=offset+middle)
        return [*split_documents([prefix],block),*split_documents([suffix],block)]

    if docs and draft([],docs).blocked:
        # Avoid fragmenting every field individually when documents alone
        # exceed capacity. Reserve room for fields in each document batch.
        document_parts=split_documents(docs,[])
        if sources:
            return [batch for part in document_parts for batch in pack(sources,documents=part)]
        return [draft([],part) for part in document_parts]
    return pack(sources) if sources else [first]


def simulated_pairs(summary):
    groups=defaultdict(list)
    for field in fields(summary["sources"]):
        if field.get("physical_family") and not field.get("axis") and not field.get("raw_companion"):
            groups[field["physical_family"]].append(field["id"])
    return dict(decisions=[
        dict(fields=[a,b],action="fusion",goal="evaluate_gain",reason="模拟：相同物理量")
        for group in groups.values() for a,b in combinations(group,2) if a.split(".")[0]!=b.split(".")[0]],
        questions=[])


def local_fields(scan,summary):
    approved={field["id"]:dict(field,engineering_documents=source.get("engineering_documents",[]))
              for source in summary["sources"] for field in source["fields"]}
    originals={}
    for index,row in enumerate(scan["rows"]):
        for offset,field in enumerate(row["fields"],1):
            fid=f"D{index+1}.F{offset}"
            if fid in approved:
                detail=approved[fid]
                if detail.get("header")!=field.get("original_header",field["name"]) or detail.get("variable_name")!=field["name"]:
                    raise ValueError(f"{fid} 的原字段名与扫描结果不一致，请重新扫描")
                originals[fid]=(index,fid,field)
    return originals,approved


def read_catalogue(batches,cache,mock,key,progress,feedback,current,reply,*,sample_field=None):
    responses=[]

    pending=deque(batches)
    attempted=set()
    index=0
    while pending:
        batch=pending.popleft()
        if batch.request.digest in attempted:
            continue
        attempted.add(batch.request.digest)
        index+=1
        total=index+len(pending)
        current()
        progress(f"DeepSeek 正在分析字段目录 · 第 {index}/{total} 批")
        draft=batch.request
        markers=[]
        rows=[]
        failure=None
        cached=False
        try:
            if batch.blocked:
                raise ValueError(batch.blocked)
            sent=json.loads(draft.summary)
            allowed={field["id"] for field in fields(sent["sources"])}
            if draft.digest in cache:
                answer=cache[draft.digest]
                reply["reused_batches"]+=1
                cached=True
            else:
                answer=service.send(draft,draft.digest,mock,api_key=key,on_receipt=markers.append)
                if sample_field is not None:
                    answer=answer_single_field_questions(sent,allowed,answer,sample_field,
                                                        mock,key,progress,current,markers.append)
                cache[draft.digest]=answer
            reply.setdefault("local_checks",[]).extend(answer.get("field_checks",[]))
            rows=answer.get("decisions",[]) or answer.get("questions",[])
            responses.append((allowed,answer))
        except service.RequestCapacityError as exc:
            if request_batches.enqueue_smaller(pending,batch,attempted):
                progress("DeepSeek 确认请求超限，正在自动分批重试…")
                feedback(dict(batch=index,total=index+len(pending),request_batch=batch,rows=[],failure=None,
                              notice=str(exc)+"，将分批重试",api_receipts=markers))
                reply["api_receipts"].extend(markers)
                continue
            failure=dict(batch=index,ids=list(batch.ids),reason=str(exc),blocked=True)
            reply["failures"].append(failure)
        except (ValueError,OSError,TypeError,KeyError,IndexError) as exc:
            failure=dict(batch=index,ids=list(batch.ids),reason=str(exc),blocked=bool(batch.blocked))
            reply["failures"].append(failure)
        reply["api_receipts"].extend(markers)
        feedback(dict(batch=index,total=total,request_batch=batch,rows=rows,failure=failure,api_receipts=markers,
                      local_cache=cached,local_checks=[]))
    reply["batch_count"]=index
    return responses


def answer_single_field_questions(sent,allowed,answer,sample_field,mock,key,progress,current,on_receipt):
    from .decision_checks import MAX_QUESTIONS, MAX_ROUNDS
    evidence={}
    for turn in range(MAX_ROUNDS):
        questions=[]
        for question in answer.get("questions",[]):
            if not isinstance(question,dict) or question.get("check")!="numeric_samples":
                continue
            refs=question.get("fields")
            if (isinstance(refs,list) and len(refs)==1 and isinstance(refs[0],str) and refs[0] in allowed
                    and refs[0] not in evidence and refs[0] not in questions):
                questions.append(refs[0])
        questions=questions[:MAX_QUESTIONS-len(evidence)]
        if not questions:
            break
        progress("正在抽取字段数值并回传扰动样本…")
        for fid in questions:
            current()
            evidence[fid]=dict(candidate=fid,check="numeric_samples",**sample_field(fid))
        context=dict(results=list(evidence.values()),rounds_completed=turn+1,
                     final_round=turn+1==MAX_ROUNDS or len(evidence)>=MAX_QUESTIONS,
                     previous_decisions=answer.get("decisions",[]))
        continuation=service.preview(sent,local_evidence=context)
        try:
            followup=service.send(continuation,continuation.digest,mock,api_key=key,on_receipt=on_receipt)
        except service.RequestCapacityError:
            lean=dict(sent,engineering_documents=[
                {k:v for k,v in doc.items() if k!="text"} for doc in sent.get("engineering_documents",[])])
            if lean["engineering_documents"]==sent.get("engineering_documents",[]):
                raise
            continuation=service.preview(lean,local_evidence=context)
            followup=service.send(continuation,continuation.digest,mock,api_key=key,on_receipt=on_receipt)
        previous=[row for row in answer.get("questions",[]) if isinstance(row,dict)
                  and isinstance(row.get("fields"),list) and len(row["fields"])==2]
        answer=dict(followup,decisions=followup.get("decisions") or answer.get("decisions",[]),
                    questions=[*previous,*followup.get("questions",[])])
    return dict(answer,field_checks=list(evidence.values()))

def materialize(scan,summary,originals,details,responses,*,ask_implicit=True):
    evidence=defaultdict(lambda: dict(decisions=[],questions=[]))
    for allowed,response in responses:
        for key in ("decisions","questions"):
            for item in response.get(key,[]):
                if not isinstance(item,dict):
                    continue
                refs=item.get("fields")
                if not isinstance(refs,list) or len(refs)!=2 or any(
                    not isinstance(fid,str) or fid not in allowed or fid not in originals for fid in refs
                ):
                    continue
                pair=tuple(sorted(refs,key=lambda fid:(originals[fid][0],fid)))
                if originals[pair[0]][0]==originals[pair[1]][0]:
                    continue
                if key=="questions" and (not isinstance(item.get("check"),str) or item["check"] not in CHECKS):
                    continue
                evidence[pair][key].append(item)
    candidates={}
    pairs=[]
    dialogue={}
    for refs,answer in evidence.items():
        cid=f"C{len(pairs)+1}"
        decisions=discovery.parse_decisions({"decisions":[dict(item,candidate=cid) for item in answer["decisions"]]})
        questions=[dict(candidate=cid,check=row["check"]) for row in answer["questions"]]
        if not decisions and not questions:
            continue
        pair,local=discovery.describe_pair(
            scan,cid,*(originals[fid] for fid in refs),summary["target_model"]["id"],details)
        pairs.append(pair)
        candidates[cid]=local
        actions={row["action"] for row in decisions}
        conflict={"reject","fusion"}<=actions
        if conflict:
            decisions=[dict(candidate=cid,action="insufficient",goal="unknown",
                            reason="分批判断有分歧。"+" ".join(row["reason"] for row in decisions)[:800])]
        if ask_implicit and not questions and (conflict or actions & {"insufficient","pending_goal"}):
            questions=[dict(candidate=cid,check=check) for check in ("metadata","time_coverage")]
        dialogue[cid]=dict(decisions=decisions,questions=questions)
    return candidates,dict(summary,catalog_first=False,candidates=pairs),dialogue


def pair_batch(approved,pair):
    scoped=request_batches.scoped(approved,"candidates",[pair])
    # The initial request already included the full field catalogue.
    scoped["sources"]=[]
    return request_batches.Batch(service.preview(scoped),(pair["id"],))


def decide(scan,summary,batches,cache,mock,key,progress,feedback):
    from .fusion_review import review

    def current():
        assert_current(scan)

    originals,details=local_fields(scan,summary)
    reply=dict(mode="模拟；真实 API 未调用" if mock else "DeepSeek API",failures=[],batch_count=len(batches),
               local_checks=[],api_receipts=[],reused_batches=0,decisions=[])
    checker=LocalChecks(scan,{})
    def sample_field(fid):
        return (checker.field_question(originals[fid]) if fid in originals
                else dict(field=fid,status="outside_scope"))
    responses=read_catalogue(batches,cache,mock,key,progress,feedback,current,reply,sample_field=sample_field)
    candidates,approved,dialogue=materialize(scan,summary,originals,details,responses)
    checker.candidates=candidates
    reviews=[]
    gate_input=[]
    for pair in approved["candidates"]:
        cid=pair["id"]
        initial=dialogue[cid]
        initial_quantities={row["physical_quantity"] for row in initial["decisions"] if row.get("physical_quantity")}
        if len(initial_quantities)==1:
            checker.quantities[cid]=next(iter(initial_quantities))
        batch=pair_batch(approved,pair)
        reviews.append(batch)
        try:
            final=request_batches.decision_dialogue(
                batch.request,initial,checker,mock,key,progress,current,receipts=reply["api_receipts"])
        except (ValueError,OSError,TypeError,KeyError,IndexError) as exc:
            reply["failures"].append(dict(batch="追问",ids=[cid],reason=str(exc),stage="local_dialogue"))
            final=initial
        rows=final.get("decisions",[])
        if len(initial_quantities)==1 and not any(row.get("physical_quantity") for row in rows):
            rows=[dict(row,physical_quantity=next(iter(initial_quantities))) for row in rows]
        reply["decisions"].extend(rows)
        reply["local_checks"].extend(final.get("local_checks",[]))
        gate_input.extend(rows)
        if not any(row["action"] in {"reject","direct","fusion"} for row in rows):
            gate_input.append(dict(candidate=cid,action="fusion",goal="evaluate_gain",
                                   physical_quantity=next((row["physical_quantity"] for row in rows if row.get("physical_quantity")),None),
                                   reason="保留为融合试跑候选，评估融合收益"))
    current()
    progress("正在核验 DeepSeek 提出的组合…")
    checked,gated=discovery.check(scan,candidates,dict(decisions=gate_input),approved,station_matches=checker.station_matches)
    review(scan,reviews,checked,reply,mock,key,progress,checker=checker)
    current()
    return reply,checked,gated,candidates,approved

