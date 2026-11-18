"""Build and retry bounded DeepSeek requests for decisions and recommendations."""
from collections import deque
from dataclasses import dataclass, replace
import json

from . import service



@dataclass(frozen=True)
class Batch:
    request: service.Preview
    ids: tuple[str, ...]
    blocked: str = ""


def unique(values):
    seen = set()
    rows = []
    for row in values:
        key = row.get("id")
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def scoped(summary, item_key, items):
    contexts = [item.get("context", item) for item in items]
    fields = {fid for pair in contexts for fid in pair.get("fields", [])}
    source_ids = {fid.partition(".")[0] for fid in fields}
    sources = unique(row for row in summary.get("sources", []) if row["id"] in source_ids)
    project = summary.get("project_context_document_ids", [])
    referenced = set(project)
    for pair in contexts:
        for field in pair.get("pair_details", []):
            referenced.update(field.get("engineering_documents", []))
    for source in sources:
        referenced.update(source.get("engineering_documents", []))
    documents = unique(doc for doc in summary.get("engineering_documents", [])
                       if doc["id"] in referenced or source_ids.intersection(doc.get("source_ids", [])))
    result = dict(summary, **{item_key: items}, engineering_documents=documents)
    if "sources" in summary:
        result["sources"] = sources
    return result


def prepare(summary, *, rejected=None):
    if summary.get("catalog_first") and summary["stage"]=="decision":
        from .catalog_decision import prepare
        return prepare(summary,rejected=rejected)
    item_key = {"decision": "candidates", "recommendation": "tasks"}[summary["stage"]]
    items = summary[item_key]
    if any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in items):
        raise ValueError("候选或任务缺少编号")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("候选或任务编号重复")
    summary = dict(summary)
    for key in ("sources", "engineering_documents", "engineering_inventory"):
        if key in summary:
            summary[key] = unique(summary[key])

    complete = service.preview(summary)
    if rejected is None:
        return [Batch(complete, tuple(ids))]

    def make_batch(group):
        payload = scoped(summary, item_key, group) if group else summary
        draft = service.preview(payload)
        reason = service.retry_capacity_hint(draft,rejected)
        if reason:
            docs = [doc for doc in payload.get("engineering_documents", []) if doc.get("text")]
            docs.sort(key=lambda doc: len(doc["text"]), reverse=True)
            if summary["stage"]=="recommendation" and len(group)==1 and docs:
                # This documentation was already presented during catalogue review.
                # Preserve references, field details and findings within model limits.
                payload=dict(payload,engineering_documents=[
                    {k:v for k,v in doc.items() if k!="text"} for doc in payload["engineering_documents"]],
                    engineering_text_delivery="工程全文已在第一步分批发送，本轮附引用及本地核验结果")
                fallback=service.preview(payload)
                if not service.retry_capacity_hint(fallback,rejected):
                    return Batch(fallback,tuple(item["id"] for item in group))
            if docs:
                sizes = "、".join(f"{doc['id']} ({len(doc['text'].encode('utf8')) / 1024:.0f} KiB)"
                                 for doc in docs[:5])
                reason += "。最大的关联说明 " + sizes
        return Batch(draft, tuple(item["id"] for item in group), reason)

    def pack(group):
        batch = make_batch(group)
        if not batch.blocked or len(group) <= 1:
            return [batch]
        # Split only candidates; each side gets whole documents, once per ID.
        midpoint = len(group) // 2
        return [*pack(group[:midpoint]), *pack(group[midpoint:])]

    batches = pack(items)
    # Rejoin neighboring groups when a different split boundary allows it.
    merged = []
    next_index = 0
    for batch in batches:
        group = items[next_index:next_index + len(batch.ids)]
        next_index += len(batch.ids)
        if merged and not merged[-1][0].blocked and not batch.blocked:
            joined = [*merged[-1][1], *group]
            combined = make_batch(joined)
            if not combined.blocked:
                merged[-1] = combined, joined
                continue
        merged.append((batch, group))
    return [batch for batch, _ in merged]


def enqueue_smaller(pending,batch,attempted):
    retries=prepare(json.loads(batch.request.summary),rejected=batch.request)
    smaller=[replace(retry,blocked="") for retry in retries if (
        len(retry.request.body)<len(batch.request.body) and retry.request.digest not in attempted)]
    pending.extendleft(reversed(smaller))
    return bool(smaller)


def decision_dialogue(draft, reply, checker, mock, key, progress, check_current, *, receipts=None):
    from .decision_checks import CHECKS, MAX_QUESTIONS, MAX_ROUNDS

    approved=json.loads(draft.summary)
    candidates={item["id"]:item for item in approved["candidates"]}
    evidence=[]
    seen=set()
    decisions={}
    unresolved={}
    for turn in range(MAX_ROUNDS+1):
        for cid in {row["candidate"] for row in reply["decisions"]} & candidates.keys():
            decisions[cid]=[row for row in reply["decisions"] if row["candidate"]==cid]
            unresolved.pop(cid,None)
        questions=list(dict.fromkeys((q["candidate"],q["check"] if q["check"] in CHECKS else "unsupported_check")
                                      for q in reply.get("questions",[]) if q["candidate"] in candidates))
        for cid,_ in questions:
            unresolved[cid]="自动核验后仍缺少完整的判断依据。"
        pending=[q for q in questions if q not in seen][:MAX_QUESTIONS-len(seen)]
        if not pending or turn==MAX_ROUNDS:
            break
        progress(f"正在本地核验并自动回复 DeepSeek · 第 {turn+1} 轮")
        for cid,check in pending:
            check_current()
            result=checker(cid,check,candidates[cid])
            seen.add((cid,check))
            evidence.append(dict(candidate=cid,check=check,**result))
        local_evidence=dict(
            rounds_completed=turn+1, final_round=turn+1==MAX_ROUNDS or len(seen)>=MAX_QUESTIONS,
            results=evidence, previous_decisions=[row for values in decisions.values() for row in values],
            awaiting_decision=[cid for cid in candidates if cid not in decisions])
        continuation=service.preview(approved,local_evidence=local_evidence)
        check_current()
        receipt=receipts.append if receipts is not None else None
        try:
            reply=service.send(continuation,continuation.digest,mock,api_key=key,on_receipt=receipt,on_progress=progress)
        except service.RequestCapacityError:
            lean=dict(approved,engineering_documents=[
                {k:v for k,v in doc.items() if k!="text"} for doc in approved.get("engineering_documents",[])])
            if lean["engineering_documents"]==approved.get("engineering_documents",[]):
                raise
            continuation=service.preview(lean,local_evidence=local_evidence)
            reply=service.send(continuation,continuation.digest,mock,api_key=key,on_receipt=receipt,on_progress=progress)
        check_current()

    rows=[]
    for cid in candidates:
        rows.extend(decisions.get(cid) or [dict(candidate=cid,goal="unknown",action="pending_goal",
                        origin="platform",reason=unresolved.get(cid,"DeepSeek 未给出该组的最终判断。"))])
    return dict(reply,decisions=rows,local_checks=evidence)


def send(batches, cache, mock, key, progress, check_current, *, feedback=None, checker=None):
    rows = []
    local_checks = []
    failures = []
    receipts = []
    reused = 0
    stage = batches[0].request.stage
    result_key = "decisions" if stage == "decision" else "recommendations"
    identity = "candidate" if stage == "decision" else "task"

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
        check_current()
        if not batch.blocked:
            progress(f"正在处理第 {index}/{total} 批 · {', '.join(batch.ids)}")
        reply_rows = []
        failure = None
        batch_receipts = []
        cached = False
        try:
            if batch.blocked:
                raise ValueError(batch.blocked)
            draft = batch.request
            if draft.digest in cache:
                reply = cache[draft.digest]
                reused += 1
                cached = True
            else:
                reply = service.send(draft, draft.digest, mock, api_key=key, on_receipt=batch_receipts.append,on_progress=progress)
                if stage=="decision" and checker is not None:
                    reply=decision_dialogue(draft,reply,checker,mock,key,progress,check_current,receipts=batch_receipts)
                accepted = [row for row in reply[result_key] if row[identity] in batch.ids]
                missing = set(batch.ids) - {row[identity] for row in accepted}
                if missing:
                    raise ValueError("回复缺少编号 " + "、".join(sorted(missing)))
                reply = dict(reply, **{result_key: accepted})
                cache[draft.digest] = reply
            reply_rows = reply[result_key]
            rows.extend(reply_rows)
            local_checks.extend(reply.get("local_checks",[]))
        except service.RequestCapacityError as error:
            if enqueue_smaller(pending,batch,attempted):
                progress("DeepSeek 确认请求超限，正在自动分批重试…")
                if feedback is not None:
                    feedback(dict(batch=index,total=index+len(pending),request_batch=batch,rows=[],failure=None,
                                  notice=str(error)+"，将分批重试",api_receipts=batch_receipts))
                receipts.extend(batch_receipts)
                continue
            failure=dict(batch=index,ids=list(batch.ids),reason=str(error),blocked=True)
            failures.append(failure)
        except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
            failure = dict(batch=index, ids=list(batch.ids), reason=str(error), blocked=bool(batch.blocked))
            failures.append(failure)
        receipts.extend(batch_receipts)
        check_current()
        if feedback is not None:
            feedback(dict(batch=index, total=total, request_batch=batch, rows=reply_rows, failure=failure,
                          api_receipts=batch_receipts, local_cache=cached,
                          local_checks=reply.get("local_checks",[]) if failure is None else []))
    check_current()
    return dict(mode="模拟；真实 API 未调用" if mock else "DeepSeek API",
                **{result_key: rows}, failures=failures, batch_count=index, local_checks=local_checks,
                api_receipts=receipts, reused_batches=reused)


def describe_usage(reply):
    if reply.get("mode")!="DeepSeek API" or "api_receipts" not in reply:
        return ""
    receipts=reply["api_receipts"]
    reported=[item["usage"]["total_tokens"] for item in receipts if "total_tokens" in item["usage"]]
    text=f"本次收到 {len(receipts)} 条 DeepSeek 线上响应，复用 {reply.get('reused_batches',0)} 批本地结果"
    if reported:
        text+=f"，服务返回 {sum(reported):,} Tokens"
        if len(reported)<len(receipts):
            text+="（部分响应未返回用量）"
    elif receipts:
        text+="，服务未返回用量"
    return text+"。"

