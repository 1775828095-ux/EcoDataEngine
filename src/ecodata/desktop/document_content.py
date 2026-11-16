"""Classify potential engineering documents before constructing outbound text."""
from dataclasses import dataclass
from itertools import islice
import csv
import json
from pathlib import PurePosixPath
import re
from xml.etree import ElementTree


def words(text):
    return frozenset(text.split())


ENGINEERING = words(
    "title description purpose usage role roles model model_id configuration config settings "
    "inputs outputs requirements provenance manifest inventory files documents notes readme source "
    "source_url source_id citation version license dependencies band band_metadata long_name "
    "standard_name unit units scale_factor add_offset nodata parameter parameters grid crs "
    "model_input_candidate selected_input_extraction weather_file_count references policy "
    "quality_flags_preserved grid_mapping_preserved checksum_verified original_bytes_match "
    "not_fused no_fusion_or_calibration_performed "
    "说明 用途 模型 配置 来源 数据来源 文件 清单 备注 处理流程 核验 结论"
)
EXPLANATION = words("title description purpose usage notes readme model model_id role roles 说明 用途 备注 模型")
PAYLOAD = words(
    "data values measurements observations samples records readings timeseries time_series temperature precipitation rainfall "
    "raster pixels matrix series coordinates features rows dataset observations_data 观测 数值 数据 记录"
)
AXES = words(
    "time times date dates datetime timestamp timestamps station stations station_id site site_id "
    "lat latitude lon longitude x y elevation height time_utc 时间 日期 站点 纬度 经度"
)
FIELDS = AXES | words(
    "value values measurement temperature precipitation rainfall humidity pressure wind lai "
    "data pm25 pm2_5 no2 concentration discharge flow air_temperature air_temperature_min "
    "air_temperature_max relative_humidity surface_pressure wind_speed variable band unit units "
    "nrows ncols subset observations records time_series series raster 观测值 温度 降水 流量 浓度"
)
PUBLIC_KEYS = FIELDS | ENGINEERING | PAYLOAD
UNITS = words("K degC Celsius mm mm/day cm m km Pa hPa m/s m3/s ug/m3 mg/L ppm ppb W/m2 degrees_north degrees_east day h s % 1")
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
DATE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[T ].*)?\Z")
TEXT_HEADING = re.compile(r"(?im)^\s*(?:#{1,6}\s+\S|\[.+\]\s*$|&[a-z][\w]*\s*$|"
                          r"(?:用途|说明|模型|输入|配置|参数|usage|description|model|inputs?|configuration)\s*[:：])")
BLOCK = re.compile(r"(?ms)^([ \t]*)(?:\x60{3,}|~{3,})([^\n]*)\n(.*?)^[ \t]*(?:\x60{3,}|~{3,})[ \t]*$")
PLACEHOLDER = "[原始数据已替换，真实值留在本机]"


@dataclass(frozen=True)
class Document:
    mode: str
    reason: str
    text: str | None = None
    structure: dict | None = None
    removed: tuple[dict, ...] = ()


def label(key, index):
    key = str(key).casefold()
    return key if key in PUBLIC_KEYS else f"字段{index}"


def kind(value):
    if value is None:
        return "null"
    if isinstance(value, str) and DATE.fullmatch(value):
        return "datetime"
    return {str:"string", bool:"boolean", int:"integer", float:"number",
            list:"array", dict:"object"}.get(type(value), "unknown")


def shape(value):
    if isinstance(value, list):
        result = dict(type="array", dimensions=[len(value)])
        if value:
            child = shape(value[0])
            if all(type(item) is type(value[0]) for item in islice(value, 32)):
                result["dimensions"] += child.get("dimensions", [])
                result["element_type"] = child.get("element_type",child["type"])
                if "fields" in child:
                    result["fields"] = child["fields"]
            else:
                result["element_type"] = "mixed"
        return result
    if isinstance(value, dict):
        return dict(type="object", fields=[
            dict(name=label(key,index), type=kind(item), **(
                {"unit":item} if str(key).casefold() in {"unit","units"} and isinstance(item,str) and item in UNITS else {}))
            for index,(key,item) in enumerate(islice(value.items(),32),1)])
    return {"type":kind(value)}


def scalar_numeric(value):
    return type(value) in (int,float) or isinstance(value,str) and bool(NUMBER.fullmatch(value.strip()))


def inventory_keys(keys):
    return (bool(keys & {"file","member","original_member","name","archive"})
            and bool(keys & {"bytes","sha256","md5","is_directory","official_relative",
                             "missing","sizes","attrs","global_attrs","syntax_error_count",
                             "missing_sentinel_count","date_gap_count"}))


def engineering_keys(keys):
    return (bool(keys & ENGINEERING) or inventory_keys(keys)
            or {"ref","object","url"} <= keys
            or {"url","bytes"} <= keys and bool(keys & {"sha256","md5","etag","response_headers"})
            or {"rows","stations","date_utc"} <= keys
            or {"shape","fields"} <= keys and any(k.endswith("_preserved") for k in keys))


def records(value,key):
    if not isinstance(value,list) or not value or not isinstance(value[0],dict):
        return False
    first = value[0]
    keys = {key.casefold() for key in first}
    if inventory_keys(keys) or not all(isinstance(row,dict) and row.keys()==first.keys()
                                        for row in islice(value,32)):
        return False
    has_numeric = any(scalar_numeric(item) for row in islice(value,32)
                      for field,item in row.items() if field.casefold() not in AXES)
    has_axis = bool(keys & AXES) or any(isinstance(item,str) and DATE.fullmatch(item) for item in first.values())
    return has_numeric and (has_axis or key in PAYLOAD and not engineering_keys(keys))


def observation_object(value):
    keys={str(key).casefold() for key in value}
    return (bool(keys & AXES) and not engineering_keys(keys-{"unit","units","source","source_id"})
            and any(str(key).casefold() not in AXES and scalar_numeric(item)
                    for key,item in value.items()))


def data_array(value, key):
    if not isinstance(value,list) or not value:
        return False
    sample = list(islice(value,32))
    if records(value,key):
        return True
    if all(isinstance(row,list) and row and all(scalar_numeric(x) for x in islice(row,16)) for row in sample):
        return True
    if key in AXES and all(isinstance(item,str) and DATE.fullmatch(item) for item in sample):
        return True
    if all(scalar_numeric(item) or item is None for item in sample):
        return key in PAYLOAD or key in AXES or len(value)>=8 or not key
    return key in PAYLOAD and all(not isinstance(item,dict) for item in sample)


def structured(value, content):
    removed = []
    has_explanation = False
    has_engineering = False

    def visit(node, path, parent_key=""):
        nonlocal has_engineering, has_explanation
        if data_array(node,parent_key) or (isinstance(node,dict) and parent_key not in {"headers","response_headers","request_headers"}
                                       and observation_object(node)):
            removed.append(dict(path=path, structure=shape(node)))
            return PLACEHOLDER
        if isinstance(node,dict):
            has_engineering |= engineering_keys({str(key).casefold() for key in node})
            output = {}
            for index,(key,item) in enumerate(node.items(),1):
                lower = str(key).casefold()
                if lower in ENGINEERING and item not in (None,"",[],{}):
                    has_engineering = True
                if lower in EXPLANATION and isinstance(item,str) and item.strip():
                    has_explanation = True
                id_hint = label(key,index)
                if lower in PAYLOAD and isinstance(item,str) and len(item)>128:
                    removed.append(dict(path=f"{path}.{id_hint}",structure=dict(type="encoded_or_text_data")))
                    output[key]=PLACEHOLDER
                else:
                    before = len(removed)
                    output[key] = visit(item, f"{path}.{id_hint}", lower)
                    unit = node.get("unit",node.get("units"))
                    if len(removed)>before and isinstance(unit,str) and unit in UNITS:
                        for section in removed[before:]:
                            section["structure"].setdefault("unit",unit)
            return output
        if isinstance(node,list):
            return [visit(item,f"{path}[{index}]",parent_key) for index,item in enumerate(node)]
        return node

    if isinstance(value,dict):
        if ({"nrows","ncols","subset"} <= value.keys()
                and type(value["nrows"]) is int and type(value["ncols"]) is int
                and isinstance(value["subset"],list) and value["subset"]
                and all(isinstance(item,dict) and isinstance(item.get("data"),list)
                        and len(item["data"])==value["nrows"]*value["ncols"] for item in value["subset"])):
            if not any(isinstance(value.get(k),str) and value[k].strip() for k in EXPLANATION):
                return Document("data","栅格记录包含行列维度与原始数值",structure={
                    "type":"raster","dimensions":[value[k] for k in ("nrows","ncols")
                          if type(value[k]) is int and value[k]>0],
                    "slice_count":len(value["subset"]),"fields":shape(value)["fields"]})
        if observation_object(value):
            return Document("data","带时间或站点坐标的单条数值记录",structure=shape(value))
    cleaned = visit(value,"$")
    if removed:
        # Only measured payload is taken out; explanatory siblings remain in place.
        if has_explanation or (has_engineering and isinstance(value,dict) and
                               bool({str(key).casefold() for key in value} &
                                    {"inputs","outputs","config","configuration","manifest","references","weather_file_count"})):
            return Document("mixed","同时包含工程说明与观测或栅格数值",json.dumps(cleaned,ensure_ascii=False,indent=2,default=str),
                            removed=tuple(removed))
        return Document("data","发现重复观测记录或原始数值数组",
                        structure=shape(value),removed=tuple(removed[:32]))
    if has_engineering:
        return Document("full","含模型配置、用途说明或工程清单字段",content)
    return Document("unknown","缺少可确认的工程用途或配置证据",structure=shape(value))


def table_row(line):
    line=line.strip()
    if not line or line.startswith(("#","<!--")):
        return []
    for delimiter in ("\t",",","|",";"):
        if delimiter in line:
            try:
                cells=next(csv.reader([line.strip("|") if delimiter=="|" else line],delimiter=delimiter,strict=True))
            except csv.Error:
                continue
            if len(cells)>=2:
                return [cell.strip() for cell in cells]
    return line.split()


def numeric_row(cells):
    return (len(cells)>=2 and any(scalar_numeric(cell) for cell in cells)
            and (all(scalar_numeric(cell) for cell in cells) or
                 any(DATE.fullmatch(cell) for cell in cells)))


def measurement_header(cells):
    keys = {cell.casefold().replace(" ","_") for cell in cells}
    return bool(keys & AXES and keys & (FIELDS-AXES))


def aligned_row(cells, width):
    return len(cells)==width and any(scalar_numeric(cell) for cell in cells)


def text_document(name, content):
    lines=content.splitlines(keepends=True)
    spans=[]
    total_data=0
    start=0
    while start<len(lines):
        cells=table_row(lines[start])
        header=measurement_header(cells)
        if not header and not numeric_row(cells):
            start+=1
            continue
        data_start=start+1 if header else start
        if header and data_start<len(lines):
            separator=table_row(lines[data_start])
            if len(separator)==len(cells) and all(re.fullmatch(r":?-{3,}:?",cell) for cell in separator):
                data_start+=1
        end=data_start if header else start+1
        while end<len(lines):
            row=table_row(lines[end])
            if not (aligned_row(row,len(cells)) if header else len(row)==len(cells) and numeric_row(row)):
                break
            end+=1
        if end-data_start>=(1 if header else 2):
            spans.append((start,end,dict(path=f"L{start+1}–L{end}",
                         structure=dict(type="table",dimensions=[end-data_start,len(cells)],
                                        fields=[dict(name=label(cell,index),type="unknown")
                                                for index,cell in enumerate(cells,1)] if header else [],
                                        column_types=[kind(cell) if not scalar_numeric(cell) else "number"
                                                      for cell in table_row(lines[data_start])]))))
            total_data+=end-start
        start=end
    if not spans and sum(numeric_row(table_row(line)) for line in lines)>=2:
        return Document("unknown","疑似包含分散的原始数据记录",structure=dict(type="text"))
    suffix=PurePosixPath(name).suffix.casefold()
    settings=sum(bool(re.match(r"^\s*[\w.-]+\s*[=:]\s*\S",line)) for line in lines)
    known=(bool(TEXT_HEADING.search(content)) or settings>=2 or
           (PurePosixPath(name).name.casefold().startswith("readme") or suffix in {".md",".rst"})
           and bool(re.search(r"[A-Za-z\u4e00-\u9fff]{4}",content)))
    if spans:
        if total_data>=sum(bool(line.strip()) for line in lines)-1 or not known:
            return Document("data","发现连续的表格或测量记录",structure={"type":"table"},
                            removed=tuple(item for _,_,item in spans))
        for start,end,_ in reversed(spans):
            lines[start:end]=[PLACEHOLDER+"\n"]
        return Document("mixed","保留说明并替换连续原始数据行","".join(lines),
                        removed=tuple(item for _,_,item in spans))
    if known:
        return Document("full","含说明标题、README 正文或配置段",content)
    return Document("unknown","无法确认文本用途",structure=dict(type="text"))


def classify(name, content):
    suffix=PurePosixPath(name).suffix.casefold()
    if suffix in {".json",".geojson"} or (
        content.lstrip().startswith(("{","["))
        and suffix not in {".ini",".cfg",".conf",".toml",".yaml",".yml"}
    ):
        try:
            return structured(json.loads(content),content)
        except (ValueError,RecursionError,TypeError):
            return Document("unknown","结构化正文无法可靠解析",structure=dict(type="json"))
    if suffix in {".xml",".qgs"} or content.lstrip().startswith("<"):
        try:
            if re.search(r"<!DOCTYPE|<!ENTITY",content,re.I):
                raise ValueError("external or internal entities")
            tree=ElementTree.fromstring(content)
        except (ValueError,ElementTree.ParseError,RecursionError):
            return Document("unknown","XML 无法安全识别",structure=dict(type="xml"))
        tag=tree.tag.rsplit("}",1)[-1].casefold()
        if tag in {"listbucketresult","listallmybucketsresult"}:
            return Document("full","对象存储文件清单",content)
        def tag_name(node):
            return node.tag.rsplit("}",1)[-1].casefold()

        keys={tag_name(child) for child in tree.iter()}
        known=tag in ENGINEERING or tag in {"qgis","project","namelist"} or bool(keys & EXPLANATION)
        removed=[]

        def strip_payload(node,path):
            for index,child in enumerate(list(node),1):
                name=tag_name(child)
                location=f"{path}.{label(name,index)}"
                fields={tag_name(field) for field in child}
                if name in PAYLOAD or fields & AXES and bool(fields & (FIELDS-AXES)):
                    structure=dict(type="xml_data", element_count=len(child))
                    removed.append(dict(path=location,structure=structure))
                    child.clear()
                    child.text=PLACEHOLDER
                else:
                    strip_payload(child,location)

        try:
            strip_payload(tree,"$")
        except RecursionError:
            return Document("unknown","XML 嵌套过深",structure=dict(type="xml"))
        if removed:
            if known:
                return Document("mixed","XML 含工程说明与原始数据段",
                                ElementTree.tostring(tree,encoding="unicode"),removed=tuple(removed))
            return Document("data","XML 含原始数据节点",structure=dict(type="xml"),removed=tuple(removed))
        if keys & AXES:
            return Document("unknown","XML 含坐标节点，需核验用途",structure=dict(type="xml"))
        if known:
            return Document("full","含工程配置或用途节点",content)
        return Document("unknown","无法确认 XML 的工程用途",structure=dict(type="xml"))

    if suffix==".toml":
        import tomllib
        try:
            return structured(tomllib.loads(content),content)
        except (ValueError,RecursionError,TypeError,AttributeError):
            return Document("unknown","TOML 结构未能确认",structure=dict(type="toml"))

    if suffix in {".yaml",".yml"}:
        try:
            import yaml
            if any(isinstance(token,(yaml.AliasToken,yaml.AnchorToken)) for token in yaml.scan(content)):
                raise ValueError("aliases")
            value=yaml.safe_load(content)
            return structured(value,content)
        except (ImportError,ValueError,RecursionError,TypeError,AttributeError):
            return Document("unknown","YAML 结构未能确认",structure=dict(type="yaml"))
        except Exception as error:
            if isinstance(error,yaml.YAMLError):
                return Document("unknown","YAML 无法解析",structure=dict(type="yaml"))
            raise

    if BLOCK.search(content):
        removed=[]
        def replace_block(match):
            language=match[2].strip().casefold()
            body=match[3]
            result=classify(f"block.{language}" if language in {"json","xml","yaml","yml"} else "block.txt",body)
            if result.mode=="full":
                return match[0]
            line=content.count("\n",0,match.start())+1
            removed.append(dict(path=f"L{line}",structure=result.structure or dict(type="embedded_text"),
                                segments=list(result.removed)))
            if result.mode=="mixed":
                return match[0][:match.start(3)-match.start()]+result.text+match[0][match.end(3)-match.start():]
            return PLACEHOLDER
        cleaned=BLOCK.sub(replace_block,content)
        if removed:
            rest=text_document(name,cleaned)
            if rest.mode in {"full","mixed"}:
                return Document("mixed","说明正文含数据或待核验的代码块",rest.text,
                                removed=tuple(removed)+rest.removed)
            return Document("unknown","文档只发现数据或无法确认用途的区块",
                            structure=dict(type="text"),removed=tuple(removed))
    return text_document(name,content)
