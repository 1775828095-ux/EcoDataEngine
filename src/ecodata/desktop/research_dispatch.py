"""Run the teammate's research services over scoped, declared local inputs."""
import json
import os
from pathlib import Path, PurePosixPath
from time import perf_counter
import zipfile

from .scan import assert_current, read_selected

SCHEMA = "ecodata.research-task.v1"
LABELS = {
    "platform_search": "统一算法与组合寻优",
    "raster_reconstruction": "单源栅格重构",
    "neural_reconstruction": "观测约束融合重构",
    "neural_comparison": "多算法融合重构对照",
    "hourly_search": "旧版小时缺失搜索",
}


def toolkit_request(document):
    """Standalone requests already accepted by the original PlatformSearch CLI."""
    if not isinstance(document,dict) or "schema" in document:
        return False
    inputs, request = document.get("inputs"),document.get("request")
    return bool(isinstance(inputs,list) and inputs and
        all(isinstance(item,dict) and {"path","role"}<=item.keys() for item in inputs) and
        isinstance(request,dict) and {"task","variable","unit","semantic_type"}<=request.keys())


def discover(scan, model=None):
    tasks = []
    for row in scan["rows"]:
        raw = row.get("original_probe_payload")
        if not raw or not row["name"].lower().endswith(".json"):
            continue
        try:
            task = json.loads(raw)
        except (ValueError, UnicodeError):
            continue
        if toolkit_request(task):
            task = dict(task,tool="platform_search")
        elif not isinstance(task,dict) or task.get("schema")!=SCHEMA:
            continue
        if task.get("model_id") not in (None,model):
            continue
        tool = task.get("tool")
        label = LABELS.get(tool,"未知科研工具") if isinstance(tool,str) else "未知科研工具"
        tasks.append(dict(file=row["name"],tool=tool,label=label,request=task))
    return tasks


class Resources:
    def __init__(self, scan, declaration):
        self.parent = PurePosixPath(declaration).parent
        self.source = {}
        self.remaining = 128*1024**2
        for row in scan["rows"]:
            self.source.setdefault(row["name"].replace("\\","/").casefold(),[]).append(row)
        self.cache = {}

    def get(self, value):
        if not isinstance(value,str) or not value or "\\" in value or ":" in value or "\x00" in value:
            raise ValueError("科研资料引用须为相对于任务 JSON 的包内路径")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("科研资料引用越界")
        key = (self.parent/path).as_posix().casefold()
        rows = self.source.get(key,[])
        if len(rows)!=1:
            raise ValueError(f"科研资料缺失或重名：{value}")
        if key not in self.cache:
            row, = rows
            if row["bytes"]>self.remaining:
                raise ValueError("科研任务资料超过 128 MiB 读取预算")
            raw = read_selected(row,limit=self.remaining)
            if len(raw)!=row["bytes"]:
                raise ValueError("科研资料读取不完整，请重新扫描")
            self.remaining -= len(raw)
            self.cache[key] = (path.name, raw)
        return self.cache[key]


def platform_search(task, resources, stage, progress):
    from ecodata.algorithms.platform_search import (
        CURRENT_INPUT_BUDGET, DatasetReference, EngineeringMetric, SearchInput,
        SearchValidation, process_current_input, submit_search,
    )
    from ecodata.recognition import recognise_file

    request = task["request"]
    if task.get("decision_policy")=="current_input":
        request = dict(request,decision_policy="current_input")
    mode = task.get("mode","PortraitMutation")
    if mode not in {"PortraitMutation","Autonomous","AutonomousField","Profile"}:
        raise ValueError("须使用原版已支持的寻优模式")
    if {"fixed_graph","bank","published_policy","portrait_policy_path"} & request.keys():
        raise ValueError("桌面科研任务接收问题及验证声明，固定或外部图配置需由原版工具链发布")
    forbidden = {"path","file","checkpoint","checkpoint_path"}
    def ensure_portable(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if any(part in forbidden for part in key.casefold().split("_")):
                    raise ValueError(f"模型配置中有外部文件参数：{key}")
                ensure_portable(item)
        elif isinstance(value,list):
            for item in value:
                ensure_portable(item)

    ensure_portable(request)
    inputs = []
    for item in task["inputs"]:
        name,raw = resources.get(item["path"])
        metadata = item.get("metadata")
        ensure_portable(metadata)
        inputs.append(SearchInput(name,item["role"],raw,metadata))
    budget = {k:min(v,200 if k!="wall_seconds" else 120) for k,v in task.get("budget",CURRENT_INPUT_BUDGET).items()
              if k in {"objective_limit","proposal_limit","wall_seconds","total_call_limit","probe_call_limit"}
              and type(v) in (int,float) and v>=0}
    def reference(spec):
        name,raw = resources.get(spec["path"])
        return recognise_file(name,raw,record_read_source=False)[0][request["variable"]]

    spec = task.get("evaluation",{})
    validation = task.get("search_validation")
    if request.get("decision_policy")=="current_input":
        if spec or validation or mode!="PortraitMutation":
            raise ValueError("当前数据自动寻优使用原版内部验证，不接收外部选择标签")
        return process_current_input(inputs,request,stage,budget=budget,progress=progress)
    evaluation = None
    if spec.get("kind")=="reference_error":
        evaluation = DatasetReference(reference(spec),request["variable"],role=spec.get("role","validation"))
    elif spec.get("kind")=="declared_task_metric":
        evaluation = EngineeringMetric(request["variable"],spec.get("metric","coverage"))
    elif spec:
        raise ValueError("原版搜索评价方式未识别")
    return submit_search(inputs,request,evaluation,stage,budget=budget,mode=mode,progress=progress,
                         validation=SearchValidation(reference(validation),request["variable"]) if validation else None)


def raster(task, resources, stage):
    from types import SimpleNamespace
    from ecodata.interface.raster_upload import RasterRequest, process_raster_upload

    name, raw = resources.get(task["input"])
    request = RasterRequest(**task["request"])
    mode = task.get("mode","automatic")
    if mode not in {"automatic","automatic_single","benchmark"}:
        raise ValueError("栅格重构模式未识别")
    router = None
    if mode!="benchmark":
        trusted = os.environ.get("ECODATA_TRUSTED_RASTER_ROUTER")
        if not trusted or not Path(trusted).is_file():
            raise ValueError("原版自动栅格重构缺少开发集冻结路由器；可声明 benchmark 运行候选对照")
        import joblib
        router = joblib.load(trusted)
    result = process_raster_upload(SimpleNamespace(name=name,getvalue=lambda:raw),request,router,
                                   mode=mode,execute_all=mode=="benchmark")
    if not result["outputs"]:
        raise ValueError("原版栅格重构没有生成可执行候选")
    for i,(method,ds) in enumerate(result["outputs"].items(),1):
        ds.assign_attrs(reconstruction_recipe=method).to_netcdf(stage/f"raster-{i}.nc",engine="h5netcdf")
    return dict(status="PASS",selected=result["selected"],trace=result["trace"],quality=result["reports"])


def neural(task, resources, stage, *, comparison):
    from dataclasses import fields
    from ecodata.fusion.neural_experiment import run_reconstruction_comparison,save_comparison
    from ecodata.fusion.neural_inputs import read_experiment_grid,read_experiment_stations
    from ecodata.fusion.neural_reconstruction import NeuralReconstructionConfig,run_neural_reconstruction

    config = task.get("config",{})
    options = {f.name for f in fields(NeuralReconstructionConfig) if
               not any(token in f.name for token in ("checkpoint","path","file"))}
    if (set(config)-options or type(config.get("epochs",40)) is not int or not 1 <= config.get("epochs",40) <= 200
            or not 1 <= config.get("hidden_channels",16) <= 128 or not 1 <= config.get("n_layers",3) <= 12):
        raise ValueError("神经重构参数不支持外部模型文件，训练轮数须在 1–200")
    def grid(spec):
        name,raw = resources.get(spec["path"])
        confirmations = {k:spec[k] for k in ("variable","unit","timestamp","categorical") if k in spec}
        return read_experiment_grid(name,raw,**confirmations)

    stations = task["stations"]
    name,raw = resources.get(stations["path"])
    stations = read_experiment_stations(name,raw,value_column=stations.get("value_column","value"),unit=stations.get("unit"))
    kwargs = dict(auxiliaries=[grid(s) for s in task.get("auxiliaries",[])],
                  evaluation_truth=grid(task["reference"]) if task.get("reference") else None,
                  config=NeuralReconstructionConfig(**config))
    if comparison:
        report = run_reconstruction_comparison(grid(task["background"]),stations,**kwargs)
        if not report["fields"]:
            raise ValueError("原版融合对照未生成任何方案")
        save_comparison(report,stage/"comparison")
        return dict(status="PASS" if report["comparison"].status.isin(["PASS","WARN"]).all() else "PARTIAL",
                    comparison=report["comparison"].to_dict("records"))
    result = run_neural_reconstruction(grid(task["background"]),stations,**kwargs)
    result.field.to_netcdf(stage/"reconstructed_field.nc",engine="h5netcdf")
    result.station_predictions.to_csv(stage/"validation_stations.csv",index=False)
    return dict(status="PASS",report=result.report)


def run(scan, tasks, folder, progress=None):
    import xarray as xr
    from ecodata.algorithms.numerical_search import RETIRED_MESSAGE

    assert_current(scan)
    folder.mkdir(parents=True,exist_ok=True)
    records = []
    for i,entry in enumerate(tasks,1):
        stage = folder/f"task-{i}"
        stage.mkdir()
        tool,task = entry["tool"],entry["request"]
        record = dict(task=f"task-{i}",tool=tool,label=entry["label"])
        started = perf_counter()
        if progress:
            progress(f"正在后台调用 · {entry['label']}")
        try:
            data = Resources(scan,entry["file"])
            if tool=="platform_search":
                outcome = platform_search(task,data,stage,progress)
            elif tool=="raster_reconstruction":
                outcome = raster(task,data,stage)
            elif tool in ("neural_reconstruction","neural_comparison"):
                outcome = neural(task,data,stage,comparison=tool=="neural_comparison")
            elif tool=="hourly_search":
                raise ValueError(RETIRED_MESSAGE)
            else:
                raise ValueError("未登记的科研工具")
            outputs = []
            for path in sorted(stage.rglob("*.nc")):
                with xr.open_dataset(path) as ds:
                    if not ds.data_vars:
                        raise ValueError("NetCDF 回读未发现数据变量")
                    outputs.append(path.relative_to(folder).as_posix())
            record.update(status=outcome.get("status","UNKNOWN"),outcome=outcome,readback=outputs)
        except Exception as exc:
            record.update(status="BLOCKED",reason=str(exc))
        record["seconds"] = round(perf_counter()-started,3)
        records.append(record)
    manifest = folder/"research_report.json"
    manifest.write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str),encoding="utf8")
    archive = folder/"research_results.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path!=archive:
                z.write(path,path.relative_to(folder).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip():
            raise ValueError("科研产物 ZIP 回读校验失败")
        members = z.namelist()
    assert_current(scan)
    return dict(original_archive=archive,archive_members=members,source_formats=[],
                manifest=dict(native_ready=False,preparation={}),research_records=records)
