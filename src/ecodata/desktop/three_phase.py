"""One approved workflow: classify files, decide on the full catalogue, then verify once."""
from collections import deque
import json
import re

from . import catalog_decision, discovery, recommendations, request_batches, service
from .decision_checks import CHECKS, LocalChecks
from .engineering_excerpt import FILE, NUMBER, PRIVATE, associate_documents, redact_private
from .scan import Scanner, assert_current, decode, local_member

KINDS={"engineering","data","mixed","uncertain","archive"}


def excerpt(text):
    text=re.sub(r"[\ud800-\udfff]","[无效字符]",text)
    return NUMBER.sub("[数值]",FILE.sub("[文件]",PRIVATE.sub(redact_private,text)))


def classification_request(scan,approved):
    docs={doc.get("file_id"):doc for doc in approved.get("engineering_documents",[])}
    for source in approved.get("sources",[]):
        if source["id"].startswith("D"):
            index=int(source["id"][1:])-1
            if index<0 or index>=len(scan["rows"]):
                raise ValueError("待发目录含未扫描的来源")
    files=[]
    for index,row in enumerate(scan["rows"],1):
        fid=f"D{index}"
        doc=docs.get(fid,{})
        preview=row.get("classification_preview")
        # For established tabular formats the prefix is only a schema cue.
        # Observation values are transmitted only when explicitly requested.
        structured=row["format"] in {"CSV","TSV","Excel","Parquet","NetCDF","HDF5","GeoTIFF","Shapefile","SWAT+ 输出表"}
        text=doc.get("text") or (excerpt(preview) if preview and not structured else "")
        fields=[dict(name=field.get("original_header",field["name"]),type=field["type"],unit=field["unit"])
                for field in row["fields"][:8]]
        fields=fields or [dict(name=name) for name in row.get("classification_header",[])]
        files.append(dict(id=fid,format=row["format"],suffix=row.get("classification_suffix"),
                          member_bytes=row["bytes"],field_count=len(row["fields"]),field_examples=fields,
                          document_sample=text[:800] or None,data_structure=doc.get("data_structure"),
                          archive_scanned=row.get("nested_members_read",False),
                          structure_hint="文本片段" if text else "已读字段" if fields else "仅目录或二进制结构"))
    return dict(stage="file_classification",target_model=approved.get("target_model"),files=files)


def classify(scan,summary,cache,mock,key,progress,feedback,reply):
    first=classification_request(scan,summary)
    pending=deque([first["files"]])
    kinds={}
    index=0
    while pending:
        group=pending.popleft()
        index+=1
        assert_current(scan)
        progress(f"DeepSeek 正在判断工程说明与数据 · 第 {index} 批")
        draft=service.preview(dict(first,files=group))
        batch=request_batches.Batch(draft,tuple(item["id"] for item in group))
        receipts=[]
        response=[]
        failure=None
        notice=None
        try:
            if draft.digest in cache:
                answer=cache[draft.digest]
                reply["reused_batches"]+=1
            else:
                answer=service.send(draft,draft.digest,mock,api_key=key,on_receipt=receipts.append,on_progress=progress)
                cache[draft.digest]=answer
            approved={item["id"] for item in group}
            for item in answer["classifications"]:
                if not isinstance(item,dict):
                    continue
                fid,kind=item.get("file"),item.get("kind")
                if isinstance(fid,str) and isinstance(kind,str) and fid in approved and kind in KINDS:
                    kinds[fid]=kind
                    response.append(dict(file=fid,kind=kind))
        except service.RequestCapacityError as exc:
            if len(group)>1:
                middle=len(group)//2
                pending.extendleft((group[middle:],group[:middle]))
                notice=str(exc)+"，将分批重试"
            else:
                failure=dict(batch=index,ids=list(batch.ids),reason=str(exc),blocked=True)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            failure=dict(batch=index,ids=list(batch.ids),reason=str(exc),blocked=False)
        if failure:
            reply["failures"].append(failure)
        reply["api_receipts"].extend(receipts)
        feedback(dict(batch=index,request_batch=batch,rows=response,failure=failure,notice=notice,phase=1))
    reply["file_classifications"]=[dict(file=item["id"],kind=kinds.get(item["id"],"uncertain"))
                                   for item in first["files"]]
    return kinds


def complete_catalogue(scan,summary,kinds):
    local_docs=list(scan.get("engineering_docs",()))
    known={name for name,_ in local_docs}
    extras=[]
    for index,row in enumerate(scan["rows"],1):
        fid=f"D{index}"
        if kinds.get(fid) not in {"engineering","mixed"} or row["name"] in known or row["fields"]:
            continue
        if not row.get("classification_preview"):
            continue
        try:
            with local_member(row) as path:
                text=decode(path.read_bytes())
            if "\x00" not in text and "\ufffd" not in text:
                extras.append((row["name"],text))
        except (OSError,RuntimeError,ValueError):
            continue
    if extras:
        generated,_=associate_documents(dict(scan,engineering_docs=[*local_docs,*extras]),
                                        {int(source["id"][1:]):source["id"] for source in summary["sources"]})
    else:
        generated=[]
    extra_ids={f"D{i}" for i,row in enumerate(scan["rows"],1) if row["name"] in {name for name,_ in extras}}
    docs=[dict(doc,ai_file_kind=kinds.get(doc.get("file_id"),"uncertain"))
          for doc in [*summary.get("engineering_documents",[]),
                      *(doc for doc in generated if doc.get("file_id") in extra_ids)]
          if kinds.get(doc.get("file_id")) not in {"data","archive"}]
    sources=[]
    allowed={doc["id"] for doc in docs}
    for source in summary["sources"]:
        sid=source["id"]
        linked=[doc["id"] for doc in docs if sid in doc.get("source_ids",[])]
        sources.append(dict(source,ai_file_kind=kinds.get(sid,"uncertain"),engineering_documents=list(dict.fromkeys(
            [*(d for d in source.get("engineering_documents",[]) if d in allowed),*linked]))))
    consent=dict(summary.get("automatic_local_checks",{}),max_rounds=1)
    consent.pop("max_questions",None)
    return dict(summary,sources=sources,engineering_documents=docs,automatic_local_checks=consent,
                project_context_document_ids=[doc["id"] for doc in docs if doc["scope"]=="project_context"],
                algorithm_registry=recommendations.algorithm_registry())


def questions(scan,responses,catalogue,originals,details):
    checker=LocalChecks(scan,{},documents=catalogue["engineering_documents"])
    candidates,approved,dialogue=catalog_decision.materialize(
        scan,catalogue,originals,details,responses,ask_implicit=False)
    checker.candidates=candidates
    evidence=[]
    wanted=set()
    pending={}
    for allowed,answer in responses:
        for item in answer.get("questions",[]):
            if not isinstance(item,dict):
                continue
            refs=item.get("fields")
            check=item.get("check")
            if (not isinstance(refs,list) or len(refs) not in {1,2} or not isinstance(check,str) or check not in CHECKS
                    or any(not isinstance(fid,str) or fid not in allowed or fid not in originals for fid in refs)):
                continue
            if len(refs)==1 and check not in {"numeric_samples","metadata","field_context"}:
                continue
            question=item.get("question","")
            question=" ".join(question.split())[:240] if isinstance(question,str) else ""
            pending.setdefault((tuple(sorted(refs)),check),question)
    by_pair={tuple(sorted(pair["fields"])):pair for pair in approved["candidates"]}
    for refs,pair in by_pair.items():
        decisions=dialogue[pair["id"]]["decisions"]
        plans=[row["preprocessing_plan"] for row in decisions if row.get("preprocessing_plan")]
        if plans:
            from .statistical_preprocessing import references
            plan=plans[-1]
            if not references(plan)<=originals.keys():
                continue
            checker.preprocessing_plans[refs]=plan
            pending.setdefault((refs,"preprocessing_plan"),"核对统计年份、对象关联和面积字段，试算换算步骤。")
            wanted.update(references(plan))
        if any(row["action"] in {"insufficient","pending_goal"} for row in decisions):
            pending.setdefault((refs,"field_context"),"")
        periods={scan["rows"][originals[fid][0]].get("declared_output_period") for fid in refs}
        if periods=={"average_annual","yearly"} and any(row["action"] not in {"reject","direct"} for row in decisions):
            pending.setdefault((refs,"annual_mean_comparison"),"核对同一对象的平均年度输出是否等于对应年份的逐年值均值。")
    for (refs,check),question in pending.items():
        explanation={"question":question} if question else {}
        if len(refs)==1:
            result=(checker.field_question(originals[refs[0]]) if check=="numeric_samples"
                    else dict(status="checked",basis="local_field_context",metadata=checker.field_context(originals[refs[0]])))
            evidence.append(dict(candidate=refs[0],fields=list(refs),check=check,**explanation,**result))
        else:
            pair=by_pair.get(refs)
            if pair is None:
                continue
            cid=pair["id"]
            quantities={row["physical_quantity"] for row in dialogue[cid]["decisions"] if row.get("physical_quantity")}
            if len(quantities)==1:
                checker.quantities[cid]=next(iter(quantities))
            result=checker(cid,check,pair)
            evidence.append(dict(candidate=cid,fields=list(refs),check=check,**explanation,**result))
        wanted.update(refs)
    sampled={fid for item in evidence if item["check"] in {"numeric_samples","annual_mean_comparison","preprocessing_plan"} for fid in item["fields"]}
    for fid in sorted(wanted-sampled):
        evidence.append(dict(candidate=fid,fields=[fid],check="numeric_samples",**checker.field_question(originals[fid])))
    return checker,evidence,wanted


def field_uses(responses,originals):
    uses={}
    for allowed,answer in responses:
        entries=answer.get("field_uses",[])
        if not isinstance(entries,list):
            continue
        for entry in entries:
            if not isinstance(entry,dict):
                continue
            fid,use,evidence=(entry.get(key) for key in ("field","use","evidence"))
            if not all(isinstance(value,str) and value.strip() for value in (fid,use,evidence)):
                continue
            if fid in allowed and fid in originals:
                uses[fid]=dict(use=" ".join(use.split())[:100],evidence=" ".join(evidence.split())[:300])
    return uses


def run(scan,summary,cache,mock,key,progress,feedback):
    if scan.get('on_demand'):
        from .file_agent import run as inspect_files
        return inspect_files(scan,summary,cache,mock,key,progress,feedback)
    reply=dict(mode="模拟；真实 API 未调用" if mock else "DeepSeek API",failures=[],batch_count=0,
               local_checks=[],api_receipts=[],reused_batches=0,decisions=[])
    display_index=0
    def show(event):
        nonlocal display_index
        display_index+=1
        feedback(dict(event,batch=display_index))
    def current():
        assert_current(scan)

    kinds=classify(scan,summary,cache,mock,key,progress,show,reply)
    if not kinds and reply["failures"]:
        failure=reply["failures"][0]
        raise ValueError("文件分类未完成："+failure["reason"])
    progress("正在准备完整字段目录和工程说明…")
    if scan.get("shallow"):
        first_inventory=[(row["name"],row["ref"],row["bytes"]) for row in scan["rows"]]
        scan,_=Scanner().scan(scan["path"],progress=progress)
        if first_inventory!=[(row["name"],row["ref"],row["bytes"]) for row in scan["rows"]]:
            raise ValueError("浅读与完整扫描的文件清单不一致，请检查归档读取警告后重试")
        summary,_=discovery.discover(scan,summary["target_model"]["id"])
    elif not summary["sources"] and not summary.get("engineering_documents"):
        summary,_=discovery.discover(scan,summary["target_model"]["id"])
    catalogue=complete_catalogue(scan,summary,kinds)
    originals,details=catalog_decision.local_fields(scan,catalogue)
    responses=catalog_decision.read_catalogue(catalog_decision.prepare(catalogue),cache,mock,key,progress,
                                             lambda event:show(dict(event,phase=2)),current,reply)
    reply["field_uses"]=field_uses(responses,originals)
    checker,evidence,wanted=questions(scan,responses,catalogue,originals,details)
    reply["local_checks"]=evidence
    if evidence:
        progress("正在一次回传全部存疑字段的核验结果…")
        prior=[row for _,answer in responses for row in answer.get("decisions",[])
               if isinstance(row,dict) and isinstance(row.get("fields"),list)
               and all(isinstance(fid,str) and fid in originals for fid in row["fields"])
               and set(row["fields"]) & wanted]
        for item in prior:
            wanted.update(item["fields"])
        subset=[dict(source,fields=[field for field in source["fields"] if field["id"] in wanted])
                for source in catalogue["sources"] if any(field["id"] in wanted for field in source["fields"])]
        cited_documents={doc for plan in checker.preprocessing_plans.values() for source in plan["sources"]
                         for doc in source["evidence_documents"]}
        documents=[doc for doc in catalogue["engineering_documents"] if doc["id"] in cited_documents]
        final=dict(catalogue,sources=subset,engineering_documents=documents,engineering_inventory=[],
                   final_evidence=dict(final_round=True,results=evidence,previous_decisions=prior))
        updates=catalog_decision.read_catalogue(catalog_decision.prepare(final),cache,mock,key,progress,
                                               lambda event:show(dict(event,phase=3)),current,reply)
        verified_plans={tuple(sorted(item["fields"])):item["plan"] for item in evidence
                        if item["check"]=="preprocessing_plan" and item["status"]=="checked"}
        for allowed,answer in updates:
            for decision in answer.get("decisions",[]):
                if not isinstance(decision,dict):
                    continue
                refs=decision.get("fields",[])
                if (isinstance(refs,list) and len(refs)==2 and all(isinstance(fid,str) and fid in allowed for fid in refs)
                        and decision.get("action") not in {"reject","direct"} and not decision.get("preprocessing_plan")):
                    if plan:=verified_plans.get(tuple(sorted(refs))):
                        decision["preprocessing_plan"]=plan
        reply["field_uses"].update(field_uses(updates,originals))
        replaced={tuple(sorted(row["fields"])) for allowed,answer in updates for row in answer.get("decisions",[])
                  if isinstance(row,dict) and isinstance(row.get("fields"),list) and len(row["fields"])==2
                  and all(isinstance(fid,str) and fid in allowed for fid in row["fields"])
                  and row.get("action")!="convert_format"
                  and discovery.parse_decisions({"decisions":[dict(row,candidate="C1")]})}
        if replaced:
            responses=[(allowed,dict(answer,questions=[],decisions=[
                row for row in answer.get("decisions",[]) if not (isinstance(row,dict)
                     and isinstance(row.get("fields"),list) and all(isinstance(fid,str) for fid in row["fields"])
                     and tuple(sorted(row["fields"])) in replaced)]))
                       for allowed,answer in responses]+updates
    candidates,approved,dialogue=catalog_decision.materialize(
        scan,catalogue,originals,details,responses,ask_implicit=False)
    id_by_fields={tuple(sorted(item["fields"])):item["id"] for item in approved["candidates"]}
    for item in evidence:
        if len(item["fields"])==2:
            item["candidate"]=id_by_fields.get(tuple(sorted(item["fields"])),item["candidate"])
    gate=[]
    for pair in approved["candidates"]:
        cid=pair["id"]
        pair["possible_actions"]=[action for action in pair["possible_actions"] if action!="convert_format"]
        left,right,actions=candidates[cid]
        candidates[cid]=(left,right,[action for action in actions if action!="convert_format"])
        rows=[row for row in dialogue[cid]["decisions"] if row["action"]!="convert_format"]
        reply["decisions"].extend(rows)
        gate.extend(rows)
        if rows and not any(row["action"] in {"reject","direct","fusion","quality_screen","preprocess"} for row in rows):
            gate.append(dict(candidate=cid,action="fusion",goal="evaluate_gain",
                             physical_quantity=next((row["physical_quantity"] for row in rows if row.get("physical_quantity")),None),
                             reason="条件允许时保留融合试跑"))
    progress("正在核对本地执行入口并整理算法组合…")
    checked,gated=discovery.check(scan,candidates,dict(decisions=gate),approved,station_matches=checker.station_matches)
    from .fusion_review import HARD_STOPS
    for row in checked:
        if row["action"]=="fusion" and row["status"]!="通过" and row.get("block_code") and row["block_code"] not in HARD_STOPS:
            row.update(next_step="recommendation_only",status="保留融合建议，执行待核验")
    reply["batch_count"]=display_index
    return reply,checked,gated,candidates,approved,scan

