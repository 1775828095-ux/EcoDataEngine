"""Second opinion on local fusion stops, scoped to the previously approved payload."""
import json

from . import service

HARD_STOPS={"unit_conflict","same_product","scope","disjoint_time"}
STOP_REASONS={
    "unit_conflict":"已确认的物理量单位不兼容",
    "unit_unknown":"缺少已核实的单位",
    "scope":"候选或字段不在已审核范围",
    "different_grid_roles":"边界网格和内部网格角色不同，现有入口不能直接处理",
    "disjoint_time":"本地确认两份资料的时间覆盖没有重叠",
    "same_product":"已核对是同一产品，不能作为两个独立来源",
    "quantity_unknown":"物理含义还未核实为同一量",
    "missing_adapter":"现有融合入口缺少这一输入形态所需的网格、时间或质量适配",
    "different_time_support":"两来源的时间统计语义不同，需明确比较口径",
    "time_support_unknown":"时间统计语义缺少明确声明",
    "verification_pending":"执行条件仍待核验，或原版入口未能完成检查",
}


def review(scan,batches,checked,reply,mock,key,progress):
    from .scan import assert_current

    blocked={row["candidate"]:row for row in checked if row["action"]=="fusion"
             and row["status"]!="通过" and row.get("block_code") not in {None,"scope"}}
    if not blocked:
        return
    reply.setdefault("fusion_reviews",[])
    for batch in batches:
        stops=[blocked[cid] for cid in batch.ids if cid in blocked]
        if not stops:
            continue
        ids={row["candidate"] for row in stops}
        approved=json.loads(batch.request.summary)
        # Include only previously approved context and controlled, locally generated codes.
        from .request_batches import scoped
        context=scoped(approved,"candidates",[row for row in approved["candidates"] if row["id"] in ids])
        context.update(stage="fusion_review",local_stops=[
            dict(candidate=row["candidate"],code=row["block_code"],
                 explanation=STOP_REASONS[row["block_code"]],
                 unit_status=row.get("unit_check",{}).get("status"))
            for row in stops])
        context["local_check_results"]=[row for row in reply.get("local_checks",[]) if row["candidate"] in ids]
        draft=service.preview(context)
        progress(f"DeepSeek 正在复核本地拦截 · {', '.join(sorted(ids))}")
        assert_current(scan)
        try:
            receipts=reply.setdefault("api_receipts",[])
            response=service.send(draft,draft.digest,mock,api_key=key,on_receipt=receipts.append)
            results={row["candidate"]:row for row in response["reviews"] if isinstance(row,dict)
                     and isinstance(row.get("candidate"),str) and row["candidate"] in ids
                     and row.get("verdict") in ("uphold","retain","uncertain")}
        except (ValueError,OSError,KeyError,TypeError,IndexError,AttributeError):
            results={}
        assert_current(scan)
        for row in stops:
            cid=row["candidate"]
            verdict=results.get(cid,{})
            record=dict(candidate=cid,code=row["block_code"],
                        verdict=verdict.get("verdict","unavailable"),
                        reason=str(verdict.get("reason",""))[:600])
            reply["fusion_reviews"].append(record)
            row["fusion_review"]=record
            if record["verdict"]=="retain" and row["block_code"] not in HARD_STOPS:
                row["next_step"]="recommendation_only"
                row["status"]="保留融合建议，执行待核验"
