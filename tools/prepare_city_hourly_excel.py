"""Read-only streaming extraction for the user's large city-hour workbooks."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path("D:/codex/数据/2021-2024全国空气城市小时数据")
OUT=ROOT/"output/city_pm25_reconstruction_20261019"
NS="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
FIELDS={"code":"entity_id","name":"entity_name","province":"province",
        "longitude":"longitude","latitude":"latitude","pm2_5":"value",
        "quality":"reported_aqi_grade","pubtime":"time"}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def extract(path):
    path=Path(path); file_id=hashlib.sha256(path.name.encode()).hexdigest()[:12]
    folder=OUT/"prepared"/file_id; folder.mkdir(parents=True,exist_ok=True)
    done=folder/"audit.json"
    if done.exists():
        prior=json.loads(done.read_text(encoding="utf-8"))
        assert sha(path)==prior["sha256"]
        return prior
    started=perf_counter(); book=openpyxl.load_workbook(path,read_only=True,data_only=True)
    sheets=[dict(name=s.title,rows=s.max_row,columns=s.max_column,
                 headers=list(next(s.iter_rows(min_row=1,max_row=1,values_only=True)))) for s in book]
    epoch=book.epoch
    assert len(sheets)==1,"Ambiguous worksheets; do not silently combine"
    sheet=book.worksheets[0]; member=sheet._worksheet_path.lstrip("/"); book.close()
    headers=sheets[0]["headers"]; assert set(FIELDS)<=set(headers)
    numeric={"longitude","latitude","value"}; writer=None; buffer=[]; types=Counter(); row_count=0
    def flush():
        nonlocal writer
        frame=pd.DataFrame(buffer,columns=list(FIELDS.values())+["source_row","row_signature"])
        for c in numeric:frame[c]=pd.to_numeric(frame[c],errors="coerce").astype(float)
        serial=pd.to_numeric(frame.time,errors="coerce")
        parsed=pd.to_datetime(serial,unit="D",origin=epoch,errors="coerce").dt.round("s")
        other=serial.isna()&frame.time.notna()
        parsed.loc[other]=pd.to_datetime(frame.loc[other,"time"],errors="coerce")
        frame["time"]=parsed.astype("datetime64[ns]")
        for c in ["entity_id","entity_name","province","reported_aqi_grade"]:frame[c]=frame[c].astype("string")
        table=pa.Table.from_pandas(frame,preserve_index=False)
        if writer is None:writer=pq.ParquetWriter(folder/"raw_selected.parquet",table.schema,compression="zstd")
        writer.write_table(table); buffer.clear()
    with ZipFile(path) as archive:
        strings=[]
        if "xl/sharedStrings.xml" in archive.namelist():
            with archive.open("xl/sharedStrings.xml") as stream:
                for _,e in etree.iterparse(stream,events=("end",),tag=NS+"si"):
                    strings.append("".join(e.itertext()));e.clear()
        for _,e in etree.iterparse(archive.open(member),events=("end",),tag=NS+"row"):
            number=int(e.get("r"))
            if number==1:e.clear();continue
            values=[None]*len(headers)
            for cell in e:
                address=cell.get("r",""); column=0
                for char in address:
                    if char.isdigit():break
                    column=column*26+ord(char)-64
                kind=cell.get("t","n");types[kind]+=1
                v=cell.find(NS+"v")
                value=v.text if v is not None else None
                if kind=="s" and value is not None:value=strings[int(value)]
                elif kind=="inlineStr":value="".join(cell.itertext())
                if 0<column<=len(values):values[column-1]=value
            signature=hashlib.sha256(json.dumps(values,ensure_ascii=False).encode()).hexdigest()
            buffer.append([values[headers.index(c)] for c in FIELDS]+[number,signature]);row_count+=1
            if len(buffer)>=25000:flush()
            e.clear()
            while e.getprevious() is not None:del e.getparent()[0]
        if buffer:flush()
    if writer:writer.close()
    df=pd.read_parquet(folder/"raw_selected.parquet")
    exact=int(df.row_signature.duplicated().sum()); unique=df.drop_duplicates("row_signature").copy()
    key=["entity_id","time"]; conflict=unique.duplicated(key,keep=False)
    reasons=np.select([unique.time.isna(),unique.entity_id.isna(),conflict,unique.value.isna(),
                       ~np.isfinite(unique.value),unique.value.lt(0)],
                      ["INVALID_TIME","MISSING_ENTITY","CONFLICTING_ENTITY_HOUR","MISSING_VALUE","NONFINITE","NEGATIVE"],default="VALID")
    unique["reason"]=reasons
    unique.to_parquet(folder/"observations.parquet",index=False)
    stats=[]
    for entity,g in unique.groupby("entity_id"):
        times=pd.DatetimeIndex(g.time.dropna().unique()).sort_values()
        steps=pd.Series(times[1:]-times[:-1]).value_counts()
        stats.append(dict(entity_id=entity,entity_name=g.entity_name.iloc[0],province=g.province.iloc[0],
            rows=len(g),valid=int(g.reason.eq("VALID").sum()),start=times.min(),end=times.max(),
            missing_rate=float(g.value.isna().mean()),coordinate_pairs=len(g[["longitude","latitude"]].drop_duplicates()),
            longitude_valid=float(g.longitude.between(-180,180).mean()),latitude_valid=float(g.latitude.between(-90,90).mean()),
            interval_counts={str(k):int(v) for k,v in steps.items()}))
    pd.DataFrame(stats).to_csv(folder/"entities.csv",index=False,encoding="utf-8-sig")
    audit=dict(file_id=file_id,path=str(path),filename=path.name,size=path.stat().st_size,sha256=sha(path),
        sheets=sheets,field_mapping=FIELDS,cell_type_counts=dict(types),rows=row_count,
        entities=len(stats),start=str(df.time.min()),end=str(df.time.max()),
        PM25_unit="UNCONFIRMED_NATIVE_VALUE",timezone="UNCONFIRMED_NAIVE_PRESERVED",
        time_window="UNCONFIRMED_HOURLY_LABEL_NOT_INTERVAL_AGGREGATION",
        coordinate_datum="UNCONFIRMED",spatial_entity="CITY_AGGREGATE_NOT_CONFIRMED_MONITOR",
        spatial_status="NOT_APPLICABLE_UNVERIFIED_COORDINATE_DATUM_AND_CITY_SUPPORT",
        temporal_status="ELIGIBLE_NATIVE_UNITS_ONLY" if stats else "BLOCKED",
        exact_duplicate_rows=exact,conflicting_rows=int(conflict.sum()),reason_counts=unique.reason.value_counts().to_dict(),
        PM25_min=float(df.value.min()),PM25_max=float(df.value.max()),PM25_missing_rate=float(df.value.isna().mean()),
        seconds=perf_counter()-started,reader="openpyxl workbook/epoch metadata + streaming lxml values; raw files never written")
    save(done,audit); print("EXTRACTED",file_id,row_count,flush=True);return audit


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    files=sorted(SOURCE.rglob("*.xlsx"));save(OUT/"input_files.json",[str(p) for p in files])
    with ProcessPoolExecutor(max_workers=2) as pool:
        results=[f.result() for f in as_completed([pool.submit(extract,str(p)) for p in files])]
    save(OUT/"dataset_audit.json",results)
    pd.DataFrame([{k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()}
                  for r in results]).to_csv(OUT/"dataset_manifest.csv",index=False,encoding="utf-8-sig")
