"""Scope the original observation-gap dispatcher to the approved pair."""
from dataclasses import dataclass

import pandas as pd

from ecodata.models.task_dispatch import discover, dispatch_preparation, normalise

from .scan import read_selected


@dataclass(frozen=True)
class SelectedUpload:
    name: str
    payload: bytes

    def getvalue(self):
        return self.payload


def selected(scan, task):
    uploads, keys = [], set()
    rows = scan["rows"]
    if sum(rows[i]["bytes"] for i, _ in task["fields"]) > 64*1024**2:
        raise ValueError("观测缺失恢复核验超过 64 MiB，请提供字段子集")
    for index, variable in task["fields"]:
        row = rows[index]
        upload = SelectedUpload(row["name"], read_selected(row, limit=64*1024**2))
        uploads.append(upload)
        keys.add((upload.name, variable))
    return uploads, keys


def verify(scan, task):
    uploads, keys = selected(scan, task)
    candidates, _ = discover(uploads)
    for table, fields in candidates:
        if (table["file"],table["field"]) not in keys:
            continue
        for field in fields:
            if (field["file"],field["field"]) in keys:
                normalise(table,field)
                return
    raise ValueError("原版观测缺失恢复需要带质量标记的实际缺失及匹配的 CF 背景场")


def run(scan, task_id, task, folder, model_id, progress):
    from .local_search import model_request, pareto_comparison
    from .result_summary import ranked_candidates
    from .service import fusion_evidence

    uploads, keys = selected(scan,task)
    folder.mkdir(parents=True)
    original = dispatch_preparation(uploads,model_id,folder,progress,
        task=dict(type="observation_gap_recovery",quantity=task["quantity"]),
        selected_fields=keys,initial_algorithms=list(dict.fromkeys(step for recipe in task["recipes"] for step in recipe)),
        request_builder=lambda request:model_request(request,model_id))
    receipt, = original["manifest"]["preparation"]["tasks"]
    search = receipt["result"]
    product = folder/"task-1"/"recovered_observations.csv"
    frame = pd.read_csv(product)
    rankings = ranked_candidates(folder/"task-1"/"search",search)
    comparison = dict(best=search["best"],selected_graph=search["selected_graph"],
        evidence=fusion_evidence(search),confirmed_initial=search["confirmed_initial"],
        rankings=rankings,pareto_comparison=pareto_comparison(rankings),
        portrait_strategy=search.get("portrait_strategy"),soft_guidance=search.get("soft_guidance"))
    return dict(files=[(product.name,product.read_bytes())],original_report=[dict(task=task_id,contract=receipt["report"])],
        grid_searches={task_id:comparison},search=None,source_formats=["CSV","NetCDF"],target_format="CSV",
        local=dict(artifacts=[dict(name=product.name,dimensions={"record":len(frame)},fields=list(frame.columns))],
            native_ready=False,reason="原观测和 QC 留在原列，估计值和状态另列，便于核对实际补到的位置。"))
