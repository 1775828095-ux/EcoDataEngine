"""Local metadata inventory, with temporary files for compressed scientific members."""
import bz2
import csv
import gzip
import io
import lzma
import os
import re
import tarfile
import time
import zipfile
from contextlib import ExitStack, contextmanager, nullcontext
from shutil import copyfileobj
from tempfile import TemporaryDirectory

from ecodata.core.exceptions import EcoDataError
from pathlib import Path, PurePosixPath

PREFIX = 16384
FORMATS = {".csv":"CSV", ".tsv":"TSV", ".xlsx":"Excel", ".parquet":"Parquet",
           ".nc":"NetCDF", ".nc4":"NetCDF", ".netcdf":"NetCDF", ".h5":"HDF5", ".hdf5":"HDF5", ".hdf":"HDF5",
           ".tif":"GeoTIFF", ".tiff":"GeoTIFF", ".json":"JSON", ".geojson":"GeoJSON",
           ".shp":"Shapefile", ".dbf":"DBF", ".rdb":"RDB", ".isd":"NOAA ISD", ".gz":"NOAA ISD"}
SCIENTIFIC = {"NetCDF","Excel","GeoTIFF","Parquet","HDF5","GeoJSON","DBF","RDB","NOAA ISD"}
DOCS = {".md",".txt",".json",".csh",".yaml",".yml",".toml",".ini",".cfg",".nml",".xml",".cio",".sim",".inp",".qgs",".prj",".cli",".rst",".conf",".lst"}
AXES = {"time","time_utc","date","datetime","timestamp","lat","latitude","lon","longitude","x","y"}

def kind(path):
    s = str(path).lower()
    if s.endswith(".zip"): return "zip"
    if s.endswith(".7z"): return "7z"
    if s.endswith((".tar",".tgz",".tar.gz",".tar.bz2",".tar.xz")): return "tar"
    return ""

def safe(name):
    p = PurePosixPath(name)
    return bool(name and not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name)

def inventory(path):
    result = []
    paths = (Path(folder)/name for folder, _, files in os.walk(path, followlinks=False)
             for name in sorted(files)) if path.is_dir() else [path]
    for p in paths:
        if p.is_symlink() or not p.is_file(): continue
        if not p.resolve().is_relative_to(path if path.is_dir() else path.parent): continue
        stat = p.stat()
        result.append((p, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    return sorted(result)

class Meter:
    def __init__(self, stream):
        self.stream, self.count = stream, 0

    def read(self, size=-1):
        data = self.stream.read(size)
        self.count += len(data)
        return data

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

def seven_zip_members(archive, temporary, stack, *, only=None, shallow=False):
    from py7zr.io import Py7zIO, WriterFactory

    class MemberFile(Py7zIO):
        def __init__(self, path):
            self.stream=stack.enter_context(path.open("w+b"))

        def write(self, data):
            return self.stream.write(data)

        def read(self, size=-1):
            return self.stream.read(-1 if size is None else size)

        def seek(self, offset, whence=0):
            return self.stream.seek(offset,whence)

        def flush(self):
            self.stream.flush()

        def size(self):
            return os.fstat(self.stream.fileno()).st_size

        def close(self):
            self.stream.close()

    class Factory(WriterFactory):
        def __init__(self):
            self.paths={}

        def create(self, filename):
            # Never use an archive member name as the destination on disk.
            path=temporary/f"member-{len(self.paths)}"
            self.paths[Path(filename).as_posix()]=path
            return MemberFile(path)

    entries=[entry for entry in archive.list() if not entry.is_directory]
    factory=Factory()
    selected=[]
    for entry in entries:
        filename=PurePosixPath(entry.filename)
        if (safe(entry.filename) and (only is None or entry.filename==only)
                and not getattr(entry,"is_symlink",False)
                and (only is not None or kind(entry.filename) or filename.suffix.lower() in FORMATS.keys() | DOCS
                     or filename.name.casefold().startswith("readme")
                     or filename.name.casefold() in {"namelist.input","print.prt"})):
            selected.append(entry.filename)
    if shallow:
        selected=[name for name in selected if kind(name)]
    if selected:
        archive.extract(path=temporary,targets=selected,factory=factory)
    for entry in entries:
        member=factory.paths.get((temporary/entry.filename).as_posix()) if safe(entry.filename) else None
        yield entry,member


def decode(blob):
    try: return blob.decode("utf-8-sig")
    except UnicodeDecodeError: return blob.decode("gb18030", errors="replace")

def printable_prefix(raw):
    if not raw or b"\x00" in raw:
        return None
    text=decode(raw)
    # A bounded UTF-8 excerpt may finish inside a final multibyte character.
    text=text.rstrip("\ufffd")
    if "\ufffd" in text or any(not char.isprintable() and char not in "\r\n\t" for char in text):
        return None
    return text or None


def spacing(values, temporal=False):
    import numpy as np
    if temporal:
        import pandas as pd
        values = pd.to_datetime(values, errors="coerce", format="mixed", utc=True).dropna().to_numpy(dtype="datetime64[s]").astype("int64")
    a = np.unique(np.asarray(values, dtype=float))
    d = np.diff(a)
    return float(d[0]) if len(d) and np.allclose(d, d[0],rtol=1e-5,atol=1e-9) else None


def swat_result_fields(row,lines):
    from ecodata.interface.swat_results import has_output_header, output_header

    if not has_output_header(lines):
        return False
    row["format"]="SWAT+ 输出表"
    try:
        header,columns,units,_=output_header(lines)
    except ValueError as exc:
        # Keep declarations visible even if the original reader cannot align
        # an invalid or unrecognized unit row.
        header=next(index for index,line in enumerate(lines) if line.split()[:4]==["jday","mon","day","yr"])
        columns=lines[header].split()
        units={}
        row["pending"].append(str(exc))
    for index,name in enumerate(columns):
        row["fields"].append(dict(name=name,unit=units.get(name,""),axis=index<7,column=index,
                                 type="number" if name in units else "unknown"))
    row["swat_output_header"]=header
    cadence=PurePosixPath(row["name"]).stem.rsplit("_",1)[-1].casefold()
    period={"day":"daily","mon":"monthly","yr":"yearly","aa":"average_annual"}.get(cadence)
    if period:
        row["declared_output_period"]=period
        row["time_spacing_basis"]="swat_output_period_suffix"
    if cadence=="day":
        row["declared_time_resolution"]="P1D"
        row["resolution"]["time_seconds"]=86400
        row["time_spacing_basis"]="swat_native_daily_output"
    elif cadence in {"mon","yr"}:
        row["declared_time_resolution"]={"mon":"P1M","yr":"P1Y"}[cadence]
    row["basis"].append("原版 SWAT+ 输出表头和单位行；观测记录留在本机")
    return True

def table_reader(stream, fmt):
    return csv.reader(stream, delimiter="\t" if fmt=="TSV" else ",", strict=True)


def table_prefix(stream, size, fmt):
    blob=bytearray()
    while len(blob)<size:
        chunk=stream.read(min(PREFIX,size-len(blob)))
        if not chunk:
            break
        blob.extend(chunk)
        text=decode(blob)
        source=io.StringIO(text,newline="")
        parser=table_reader(source,fmt)
        try:
            header=next(parser,None)
        except csv.Error:
            if len(blob)==size:
                raise
            continue
        if header is not None and (source.tell()<len(text) or text.endswith(("\n","\r")) or len(blob)==size):
            # A wide table may leave no complete sample row in the same prefix.
            # Read one more bounded window for coordinate and type inference.
            if source.tell()>len(text)//2:
                blob.extend(stream.read(min(len(blob),size-len(blob))))
            break
    return bytes(blob)


def tabular(row, blob, complete):
    decoded=decode(blob)
    reader=table_reader(io.StringIO(decoded,newline=""),row["format"])
    header=next(reader,None)
    if header is None:
        return
    samples=[]
    line_count=len(decoded.splitlines())
    ends_at_line=decoded.endswith(("\n","\r"))
    for _ in range(16):
        try:
            values=next(reader,None)
        except csv.Error:
            if complete:
                raise
            break
        if values is None:
            break
        if not complete and not ends_at_line and reader.line_num>=line_count:
            break
        if len(values)==len(header):
            samples.append(values)
    for i,title in enumerate(header):
        values = [r[i] for r in samples if r[i].strip()]
        m = re.fullmatch(r"(.+?)\s*[\[(]([^\])]+)[\])]", title)
        name, unit = (m[1].strip(),m[2]) if m else (title,"")
        base = name.casefold().strip()
        typ = "unknown"
        if values:
            try:
                for v in values: float(v)
                typ = "integer" if all(re.fullmatch(r"[+-]?\d+",v.strip()) for v in values) else "number"
            except ValueError: typ = "text"
        if base in {"time","date","datetime","timestamp","time_utc"} and typ=="text":
            import pandas as pd
            values = [v.replace("_","T",1) for v in values] if base=="time_utc" else values
            parsed=pd.to_datetime(values,errors="coerce",format="mixed",utc=True).dropna()
            if len(parsed):
                row["time_coverage"]=(parsed.min().timestamp(),parsed.max().timestamp(),
                    "all_table_records" if complete and line_count<=17 and len(parsed)==len(samples) else "sampled_table_records")
            step = spacing(values, True)
            if step is not None:
                typ = "time"
                row["resolution"]["time_seconds"] = step
        field=dict(name=name,original_header=title,column=i,type=typ,unit=unit,axis=base in AXES)
        from .soil_depth import text_support
        if support:=text_support(title):
            field["soil_depth"]=support
        row["fields"].append(field)
    if row["format"]=="CSV" and {"station_id","time_utc","latitude","longitude","levels"} <= set(header):
        from ecodata.models.task_dispatch import OBSERVATION_COLUMNS
        for field in row["fields"]:
            name = field["name"]
            if name in OBSERVATION_COLUMNS and name+"_qc" in header:
                quantity, unit = OBSERVATION_COLUMNS[name]
                field.update(type="number", unit=unit, standard_name=quantity,
                             temporal_semantics="instantaneous", observation_contract=True)
    if row["format"]=="CSV":
        from ecodata.io.noaa_global_hourly import COMPONENTS, is_global_hourly
        if is_global_hourly(header):
            source_columns={entry[0] for entry in COMPONENTS.values()}
            for field in row["fields"]:
                if field["name"] in source_columns:
                    field["noaa_compound"]=True
            for name,(source,_,component) in COMPONENTS.items():
                if source in header and name not in header:
                    row["fields"].append(dict(name=name,type="number",axis=False,unit=component.unit,
                        standard_name=component.quantity,source_column=source,column=header.index(source),
                        noaa_component=True,temporal_semantics="instantaneous",
                        long_name=f"NOAA Global Hourly {source} decoded observation"))
            row["basis"].append("NOAA Global Hourly 复合列已登记数值分量；保留原表头与质控码")
    row["basis"].append(f"表头及最多 {len(samples)} 条记录；间隔为样本估计，站点间距不当作网格分辨率")

def netcdf_support(row, dataset, *, read_values=True):
    from datetime import datetime, timedelta, timezone
    from math import isfinite

    def hhmmss(code):
        code=int(code)
        hours, mmss=divmod(code,10000)
        minutes, seconds=divmod(mmss,100)
        if code<0 or minutes>=60 or seconds>=60:
            raise ValueError("invalid IOAPI clock")
        return hours*3600+minutes*60+seconds

    if not row.get("time_coverage"):
        try:
            first,last=(datetime.fromisoformat(str(getattr(dataset,key)).replace("Z","+00:00"))
                        for key in ("time_coverage_start","time_coverage_end"))
            if first.tzinfo is not None and last.tzinfo is not None and first<last:
                row["time_coverage"]=(first.timestamp(),last.timestamp(),"dataset_declared_interval","gregorian")
        except (AttributeError,ValueError,TypeError,OverflowError):
            pass
    flag=dataset.variables.get("TFLAG")
    if flag is None or tuple(flag.dimensions)!=("TSTEP","VAR","DATE-TIME"):
        if read_values and "Times" in dataset.variables:
            wrf_time_support(row,dataset.variables["Times"])
        return
    try:
        cols,rows=(float(getattr(dataset,key)) for key in ("XCELL","YCELL"))
        if all(isfinite(value) and value>0 for value in (cols,rows)):
            row["resolution"].update(x=dict(step=cols,unit="m"),y=dict(step=rows,unit="m"))
            row["grid_spacing_basis"]="ioapi_xcell_ycell"
    except (AttributeError,ValueError,TypeError):
        pass
    if not read_values:
        return
    try:
        n=len(dataset.dimensions["TSTEP"])
        if n<1 or flag.shape[1]<1 or flag.shape[2]!=2:
            return
        times=[]
        for i in sorted({0,n-1}):
            date,clock=map(int,flag[i,0,:])
            year,day=divmod(date,1000)
            anchor=datetime(year,1,1,tzinfo=timezone.utc)+timedelta(days=day-1)
            if day<1 or anchor.year!=year:
                raise ValueError("invalid IOAPI day")
            times.append((anchor+timedelta(seconds=hhmmss(clock))).timestamp())
        if times[-1]>=times[0]:
            row["time_coverage"]=(times[0],times[-1],"ioapi_tflag_endpoints","gregorian")
        step=hhmmss(dataset.TSTEP)
        if n>1 and step>0 and times[-1]-times[0]==step*(n-1):
            row["resolution"]["time_seconds"]=float(step)
            row["time_spacing_basis"]="ioapi_timestep_and_tflag"
    except (AttributeError,ValueError,TypeError,OverflowError):
        pass


def wrf_time_support(row, variable):
    from datetime import datetime
    from numpy.ma import is_masked
    if variable.ndim!=2 or variable.shape[1]!=19 or not len(variable):
        return
    indices=sorted(set(range(min(8,len(variable)))) | {len(variable)-1})
    times=[]
    try:
        for index in indices:
            value=variable[index,:]
            if is_masked(value):
                return
            times.append(datetime.strptime(value.tobytes().decode("ascii"),"%Y-%m-%d_%H:%M:%S"))
    except (ValueError,UnicodeError,AttributeError,TypeError):
        return
    if any(later<=earlier for earlier,later in zip(times,times[1:])):
        return
    step=(times[1]-times[0]).total_seconds() if len(times)>1 else None
    # WRF Times omit a UTC offset. Preserve clock evidence without publishing a guessed UTC coverage.
    if step and all((value-times[0]).total_seconds()==index*step for value,index in zip(times,indices)):
        row["resolution"]["time_seconds"]=step
        row["time_spacing_basis"]="wrf_times_samples_and_endpoints"


def coordinate_kind(name, attributes):
    standard = str(attributes.get("standard_name","")).casefold()
    axis = str(attributes.get("axis","")).upper()
    unit = str(attributes.get("units","")).strip().casefold()
    if axis=="T" or standard=="time" or " since " in unit or name.casefold() in {"time","valid_time","forecast_time"}:
        return "time"
    if standard=="longitude" or unit in {"degrees_east","degree_east","degrees_e"} or name.casefold() in {"lon","longitude","xlong"}:
        return "lon"
    if standard=="latitude" or unit in {"degrees_north","degree_north","degrees_n"} or name.casefold() in {"lat","latitude","xlat"}:
        return "lat"
    if axis=="X" or standard=="projection_x_coordinate" or name.casefold()=="x":
        return "x"
    if axis=="Y" or standard=="projection_y_coordinate" or name.casefold()=="y":
        return "y"
    return None


def curvilinear_support(row, ds, coordinates):
    import numpy as np
    if row["resolution"].get("x") and row["resolution"].get("y"):
        return
    dx,dy=(getattr(ds,key,None) for key in ("DX","DY"))
    if getattr(ds,"MAP_PROJ",None) in (1,2,3) and all(isinstance(v,(int,float,np.number)) and np.isfinite(v) and v>0 for v in (dx,dy)):
        row["resolution"].update(x=dict(step=float(dx),unit="m"),y=dict(step=float(dy),unit="m"))
        row["grid_spacing_basis"]="wrf_projected_dx_dy"
        return
    lat,lon=(coordinates.get(key) for key in ("lat","lon"))
    if lat is None or lon is None or lat.shape!=lon.shape or lat.dimensions!=lon.dimensions or lat.ndim not in (2,3):
        return
    if min(lat.shape[-2:])<2:
        return
    from pyproj import Geod
    key=(0,slice(0,8),slice(0,8)) if lat.ndim==3 else (slice(0,8),slice(0,8))
    y,x=(np.ma.asarray(v[key],dtype=float).filled(np.nan) for v in (lat,lon))
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(abs(y)>90) or np.any(abs(x)>360):
        return
    geodesic=Geod(ellps="WGS84")
    for axis,a,b in (("x",np.s_[:,:-1],np.s_[:,1:]),("y",np.s_[:-1,:],np.s_[1:,:])):
        distance=geodesic.inv(x[a],y[a],x[b],y[b])[2]
        if np.isfinite(distance).all() and np.min(distance)>0:
            row["resolution"][axis]=dict(step=float(np.median(distance)),unit="m")
    if "x" in row["resolution"] and "y" in row["resolution"]:
        row["grid_spacing_basis"]="sampled_curvilinear_neighbor_distance"


REFERENCE_METADATA=("units","standard_name","long_name","axis","positive","grid_mapping_name","flag_meanings")


def attach_variable_references(row,facts):
    from ecodata.models.scientific_schema import describe_variables, variable_references

    summary={"variables":facts}
    graph=variable_references(summary)
    roles={item["name"]:item["role"] for item in describe_variables(summary,references=graph)}
    identifiers={field["name"]:index for index,field in enumerate(row["fields"],1)}

    def witness(edge):
        details=dict(edge)
        target=edge["target"]
        if target is not None:
            info=facts[target]
            attrs=info["attributes"]
            details.update(field_index=identifiers[target],dimensions=info["dimensions"],
                           unit=attrs.get("units") or None,standard_name=attrs.get("standard_name") or None,
                           long_name=attrs.get("long_name") or None,role=roles[target])
            details.update({key:attrs[key] for key in ("positive","grid_mapping_name","flag_meanings") if attrs.get(key)})
        return details

    for field in row["fields"]:
        field["metadata_role"]=roles[field["name"]]
        relations=[]
        for edge in graph.get(field["name"],[]):
            related=witness(edge)
            if nested:=graph.get(edge["target"]):
                # Preserve coordinate -> bounds and formula term references
                # even if DeepSeek receives a batch with only the data field.
                related["target_references"]=[witness(child) for child in nested]
            relations.append(related)
        if relations:
            field["variable_references"]=relations


def original_fields(row, content):
    from ecodata.io.structures import read_structure
    from ecodata.io.structures import GEOMETRY
    from ecodata.models.scientific_schema import REFERENCE_ATTRIBUTES
    facts={}
    with read_structure(row["name"],content,record_read_source=False) as dataset:
        for name,variable in dataset.variables.items():
            if name==GEOMETRY:
                continue
            keys=("standard_name","physical_variable","long_name","var_desc","temporal_semantics")
            metadata={key:str(variable.attrs[key]) for key in keys if variable.attrs.get(key) is not None}
            attributes={key:str(variable.attrs[key]) for key in (*REFERENCE_ATTRIBUTES,*REFERENCE_METADATA)
                        if variable.attrs.get(key) is not None}
            facts[name]=dict(dimensions=list(variable.dims),attributes=attributes)
            row["fields"].append(dict(name=name,type=str(variable.dtype),unit=str(variable.attrs.get("units","")),
                axis=name in dataset.coords or name.casefold() in AXES,dimensions=list(variable.dims),**metadata))
    attach_variable_references(row,facts)
    row["original_reader"]=True
    row["basis"].append("原版读取器识别的变量与属性")


def shapefile_attribute_rows(rows,docs):
    def key(row):
        path,archive,*_=row["ref"]
        return path if archive else None,PurePosixPath(row["name"]).with_suffix("").as_posix().casefold()

    tables={key(row):row for row in rows if row["format"]=="DBF" and row["fields"]}
    projections={key(row):row for row in rows if PurePosixPath(row["name"]).suffix.casefold()==".prj"}
    texts=dict(docs)
    for row in rows:
        if row["format"]!="Shapefile":
            continue
        if table:=tables.get(key(row)):
            row["attribute_member"]={name:table[name] for name in ("ref","bytes","name")}
            row["fields"]=table["fields"]
            table["fields"]=[]
            row["basis"].append("读取同名 DBF 属性表，几何保持本地")
            row["pending"]=[note for note in row["pending"] if "原版读取器" not in note and "仅登记" not in note]
        projection=projections.get(key(row))
        if projection is None:
            row["coordinate_reference"]=dict(status="companion_missing")
            continue
        text=texts.get(projection["name"])
        if not text:
            row["coordinate_reference"]=dict(status="companion_unread")
            continue
        try:
            from rasterio.crs import CRS
            from rasterio.errors import CRSError
        except ImportError:
            row["coordinate_reference"]=dict(status="reader_unavailable")
            continue
        try:
            crs=CRS.from_wkt(text)
            authority=crs.to_authority()
            row["coordinate_reference"]=dict(status="declared",basis="shapefile_prj",
                authority=":".join(authority) if authority else None,
                kind="geographic" if crs.is_geographic else "projected" if crs.is_projected else "other")
        except (ValueError,CRSError):
            row["coordinate_reference"]=dict(status="companion_invalid")

def scientific(row, path):
    fmt = row["format"]
    if fmt == "NetCDF":
        import netCDF4 as nc
        with nc.Dataset(str(path)) as ds:
            row["dimensions"] = {k:len(v) for k,v in ds.dimensions.items()}
            attrs = {name:ds.getncattr(name) for name in ("TITLE","IOAPI_VERSION","CAMx_NAME",
                "MAP_PROJ","TRUELAT1","TRUELAT2","STAND_LON","DX","DY") if name in ds.ncattrs()}
            if attrs:
                row["original_native_header"] = dict(attributes=attrs,dimensions=row["dimensions"],
                    variables={name:{} for name in ("Times","TFLAG") if name in ds.variables})
            row["metadata"] = {k:str(ds.getncattr(k))[:300] for k in ("title","Conventions","crs","source") if k in ds.ncattrs()}
            row["native_product"]=str(getattr(ds,"product",""))
            grid_attrs=("GDTYP","P_ALP","P_BET","P_GAM","XCENT","YCENT",
                        "XORIG","YORIG","XCELL","YCELL","NCOLS","NROWS")
            if all(key in ds.ncattrs() for key in grid_attrs):
                row["ioapi_grid_signature"]=tuple(float(ds.getncattr(key)) for key in grid_attrs)

            declared=str(getattr(ds,"time_coverage_resolution","")).strip()
            if re.fullmatch(r"P(?=\d|T\d)(?:\d+Y)?(?:\d+M)?(?:\d+W)?(?:\d+D)?(?:T(?=\d)(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?",declared):
                row["declared_time_resolution"]=declared
            from ecodata.models.scientific_schema import REFERENCE_ATTRIBUTES
            from .soil_depth import netcdf_depth_axes, netcdf_support as soil_support
            from .time_metadata import cell_methods
            facts={name:dict(dimensions=list(var.dimensions),attributes={
                key:str(getattr(var,key,"")) for key in (*REFERENCE_ATTRIBUTES,*REFERENCE_METADATA)})
                for name,var in ds.variables.items()}
            bounds={str(getattr(v,key,"")) for v in ds.variables.values() for key in ("bounds","climatology")}
            depth_axes=netcdf_depth_axes(ds)
            depth_cache={}
            coords={}
            coordinate_kinds={name:coordinate_kind(name,{key:getattr(var,key,"") for key in ("axis","standard_name","units")})
                              for name,var in ds.variables.items()}
            time_axes={"time"}
            for name,var in ds.variables.items():
                if coordinate_kinds[name]=="time":
                    time_axes.add(name)
                    if var.ndim==1:
                        time_axes.update(var.dimensions)
            for k,v in ds.variables.items():
                coordinate=coordinate_kinds[k]
                if coordinate and v.ndim in (1,2,3):
                    coords.setdefault(coordinate,v)
                companion=ds.variables.get(k[:-4]) if k.casefold().endswith("_raw") else None
                raw_companion=companion is not None and (
                    getattr(companion,"source_scale_factor",None) is not None or
                    getattr(v,"source_fill_value",None) is not None)
                row["fields"].append(dict(name=k,type=str(v.dtype),unit=str(getattr(v,"units","")).strip(),
                    standard_name=str(getattr(v,"standard_name","")), physical_variable=str(getattr(v,"physical_variable","")), long_name=str(getattr(v,"long_name","")).strip(), var_desc=str(getattr(v,"var_desc","")).strip(), raw_companion=raw_companion, temporal_semantics=str(getattr(v,"temporal_semantics","")), dimensions=list(v.dimensions), axis=bool(coordinate) or k.casefold() in AXES or (k.casefold()=="tflag" and tuple(d.casefold() for d in v.dimensions)==("tstep","var","date-time")) or k in bounds or k.casefold().endswith(("_bounds","_bnds")) or v.dimensions==(k,)))
                if methods:=cell_methods(getattr(v,"cell_methods",""),time_axes):
                    row["fields"][-1]["time_cell_methods"]=methods
                if k not in depth_axes and k not in bounds and not row["fields"][-1]["axis"]:
                    support=soil_support(ds,v,depth_axes,depth_cache)
                    if support:
                        row["fields"][-1]["soil_depth"]=support
                if v.ndim==1 and coordinate=="time" and len(v):
                    try:
                        import numpy as np
                        units=v.units
                        calendar=getattr(v,"calendar","standard")
                        edge=v[[0,len(v)-1]]
                        climatology=not getattr(v,"bounds","") and bool(getattr(v,"climatology",""))
                        bound=ds.variables.get(str(getattr(v,"climatology" if climatology else "bounds","")))
                        basis="axis_endpoints"
                        if bound is not None and bound.ndim==2 and bound.dimensions[0]==v.dimensions[0] and bound.shape[1]==2:
                            edge=bound[[0,len(v)-1],:]
                            basis="climatology_bounds_endpoints" if climatology else "cell_bounds_endpoints"
                        values=np.ma.asarray(edge).compressed()
                        if values.size==edge.size and np.isfinite(values).all():
                            dates=nc.num2date([min(values),max(values)],units,calendar)
                            first,last=nc.date2num(dates,"seconds since 1970-01-01",calendar)
                            calendars={"standard":"gregorian","gregorian":"gregorian","proleptic_gregorian":"proleptic_gregorian"}
                            row["time_coverage"]=(float(first),float(last),basis,calendars.get(calendar,calendar))
                            if len(v)==1 and basis=="cell_bounds_endpoints" and last>first:
                                row["time_cell_seconds"]=float(last-first)
                    except (AttributeError,ValueError,TypeError,OverflowError):
                        pass
                if v.ndim==1 and coordinate and len(v)>=2:
                    if coordinate=="time":
                        u=str(getattr(v,"units","")).lower().split(" since ")[0]
                        factor={"seconds":1,"minutes":60,"hours":3600,"days":86400}.get(u)
                        delta=spacing(v[:8]) if factor else None
                        if delta is not None: row["resolution"]["time_seconds"]=delta*factor
                    else:
                        delta=spacing(v[:8])
                        if delta is not None: row["resolution"][coordinate]=dict(step=delta, unit=str(getattr(v,"units","")))
            netcdf_support(row,ds)
            curvilinear_support(row,ds,coords)
            if "TFLAG" in ds.variables and ds.variables["TFLAG"].dimensions==("TSTEP","VAR","DATE-TIME"):
                dimensions=set(ds.dimensions)
                if "PERIM" in dimensions:
                    row["ioapi_grid_role"]="boundary"
                elif {"ROW","COL"} <= dimensions:
                    row["ioapi_grid_role"]="interior"
            attach_variable_references(row,facts)
            row["basis"].append("NetCDF 维度、变量属性、坐标小样本或 IOAPI 时间标记")
    elif fmt=="GeoTIFF":
        import rasterio
        with rasterio.open(path) as ds:
            row["resolution"]["pixel"]=dict(step=list(ds.res),crs=str(ds.crs))
            row["dimensions"]=dict(width=ds.width,height=ds.height)
            import json
            saved=json.loads(ds.tags().get("ecodata_metadata","{}"))
            for i in range(ds.count):
                name=saved.get("export_variable","value") if ds.count==1 else (ds.descriptions[i] or f"band_{i+1}")
                attrs=saved.get("variables",{}).get(name,{})
                band_tags=ds.tags(i+1)
                semantic={key: attrs.get(key,band_tags.get(key)) for key in
                          ("data_kind","spatial_semantics","flag_values","flag_meanings","flag_masks")
                          if attrs.get(key,band_tags.get(key)) is not None}
                row["fields"].append(dict(name=name,band=i+1,type=ds.dtypes[i],unit=attrs.get("units") or ds.units[i] or "",
                    standard_name=attrs.get("standard_name",""), physical_variable=attrs.get("physical_variable",""),
                    long_name=attrs.get("long_name",""),axis=False,**semantic))
            if ds.count>1:
                row["pending"].append("多波段处理前需核验波段及原版读取器映射")
            row["basis"].append("栅格元数据；未读像元")
    elif fmt=="Excel":
        from openpyxl import load_workbook
        with path.open("rb") as stream:
            workbook=load_workbook(stream,read_only=True,data_only=True)
            try:
                visible=[sheet for sheet in workbook if sheet.sheet_state=="visible" and sheet.title!="_ecodata_metadata"]
                if len(visible)>1:
                    row["pending"].append("已扫描可见工作表表头；多工作表执行前需通过原版规则确认数据区")
                for sheet in visible:
                    buffer=io.StringIO()
                    writer=csv.writer(buffer)
                    start=len(row["fields"])
                    writer.writerows(sheet.iter_rows(max_row=17,values_only=True))
                    tabular(row,buffer.getvalue().encode("utf8"),False)
                    for field in row["fields"][start:]:
                        field["sheet"]=sheet.title
                if "_ecodata_metadata" in workbook.sheetnames:
                    import json
                    from itertools import islice
                    records=list(islice(workbook["_ecodata_metadata"].values, 16))
                    try:
                        metadata=json.loads("".join(str(record[0]) for record in records[1:]))["variables"]
                    except (ValueError, KeyError, TypeError):
                        metadata={}
                    for field in row["fields"]:
                        attrs=metadata.get(field["name"],{})
                        field.update(unit=attrs.get("units",field["unit"]),standard_name=attrs.get("standard_name",""),
                                     physical_variable=attrs.get("physical_variable",""))
            finally:
                workbook.close()
    elif fmt=="Parquet":
        import pyarrow.parquet as pq
        row["fields"]=[dict(name=f.name,type=str(f.type),unit="",axis=f.name.casefold() in AXES)
                       for f in pq.read_schema(path)]
        row["basis"].append("Parquet 页脚 schema；未读数据行")
    elif fmt=="HDF5":
        scan_hdf(row,path)
    elif fmt=="DBF":
        import shapefile
        with path.open("rb") as stream, shapefile.Reader(dbf=stream) as dataset:
            row["fields"]=[dict(name=name,type=("integer" if decimals==0 else "number") if typ in {"N","F"} else "text",
                                 unit="",axis=name.casefold() in AXES)
                           for name,typ,_,decimals in dataset.fields[1:]]
        row["basis"].append("DBF 属性表字段")
    elif fmt in {"GeoJSON","RDB","NOAA ISD"}:
        original_fields(row,path.read_bytes())
    else:
        row["pending"].append("需按所选处理目标用原版读取器核验")

def scan_hdf(row,source):
    import h5py
    from ecodata.models.scientific_schema import REFERENCE_ATTRIBUTES
    facts={}
    with h5py.File(source) as ds:
        def visitor(k,v):
            if isinstance(v,h5py.Dataset):
                def attrs(name):
                    value=v.attrs.get(name,"")
                    return decode(value) if isinstance(value,bytes) else str(value).strip()
                scales=[]
                dimensions=[v.dims[i].label or "" for i in range(v.ndim)]
                for i in range(v.ndim):
                    try:
                        attached=[scale.name for scale in v.dims[i].values()]
                    except (RuntimeError,ValueError):
                        attached=[]
                    scales.extend(attached)
                facts[k]=dict(dimensions=dimensions,dimension_scales=scales,
                              attributes={name:attrs(name) for name in (*REFERENCE_ATTRIBUTES,*REFERENCE_METADATA)})
                row["fields"].append(dict(name=k,type=str(v.dtype),unit=attrs("units"),axis=bool(v.is_scale),
                    standard_name=attrs("standard_name"),physical_variable=attrs("physical_variable"),
                    long_name=attrs("long_name"),var_desc=attrs("var_desc"),
                    temporal_semantics=attrs("temporal_semantics"),shape=list(v.shape)))
        ds.visititems(visitor)
    attach_variable_references(row,facts)
    row["basis"].append("HDF5 数据集属性；未读数据数组")

class Scanner:
    def __init__(self):
        self.cached = None

    def scan(self, location, progress=None, *, shallow=False):
        start = time.perf_counter()
        root = Path(location).resolve(strict=True)
        signature = tuple(inventory(root))
        if self.cached and self.cached[:3]==(root,signature,shallow):
            return self.cached[3], dict(seconds=time.perf_counter()-start, cached=True,
                                       scan_payload_bytes=0, sequential=[], scan_temp_bytes=0)
        rows, docs, warnings, sequential = [], [], [], []
        count = 0
        temporary_bytes = 0

        def item(name,size,ref,reader=None,local=None,open_member=None,containers=()):
            nonlocal count, temporary_bytes
            if not safe(name): return
            p = PurePosixPath(name)
            ext = p.suffix.lower()
            row = dict(name=name, bytes=size, ref=ref, format=FORMATS.get(ext,"其他"),
                       fields=[], resolution={}, basis=[], pending=[], evidence=[], roles=[])
            if containers:
                row["containers"]=containers
            row["classification_suffix"]=ext if re.fullmatch(r"\.[a-z][a-z0-9]{0,12}",ext) else None
            rows.append(row)
            if shallow:
                if reader and not kind(name):
                    raw=reader(min(size,4096))
                    count+=len(raw)
                    row["classification_preview"]=printable_prefix(raw)
                    if ext in {".csv",".tsv"} and row["classification_preview"]:
                        try:
                            columns=next(table_reader(io.StringIO(row["classification_preview"]),row["format"]))
                        except (csv.Error,StopIteration):
                            columns=[]
                        row["classification_header"]=columns[:8]
                return row
            doc = ext in DOCS or p.name.lower().startswith("readme") or p.name.casefold() in {"namelist.input","print.prt"}
            if doc:
                row["engineering_text_status"]="archive_member_not_read" if not reader else "read_error"
                if reader:
                    if ext==".txt" and (open_member or local):
                        with (open_member() if open_member else local.open("rb")) as text:
                            header=[text.readline() for _ in range(21)]
                            prefix=b"".join(header)
                            if swat_result_fields(row,decode(prefix).splitlines()):
                                count+=len(prefix)
                                row.pop("engineering_text_status",None)
                                return row
                            blob=prefix+text.read()
                    else:
                        blob=reader(size)
                    count+=len(blob)
                    if len(blob)==size:
                        content=decode(blob)
                        if "\x00" in content or "\ufffd" in content:
                            row["engineering_text_status"]="binary_or_encoding_unreadable"
                        else:
                            if ext==".json":
                                from json import loads, JSONDecodeError
                                try:
                                    parsed=loads(content)
                                except JSONDecodeError:
                                    parsed=None
                                if (isinstance(parsed,list) and all(isinstance(record,dict) for record in parsed)
                                        or isinstance(parsed,dict) and parsed.get("type")=="FeatureCollection"):
                                    original_fields(row,blob)
                            elif ext==".txt":
                                from ecodata.io.raw_tables import locate_header
                                try:
                                    locate_header(content)
                                except EcoDataError:
                                    pass
                                else:
                                    original_fields(row,blob)
                            is_local_research=False
                            if ext==".json" and size<=256*1024:
                                import json
                                try:
                                    declaration=json.loads(content)
                                    from .research_dispatch import SCHEMA, toolkit_request
                                    is_local_research=(isinstance(declaration,dict) and declaration.get("schema")==SCHEMA
                                                       or toolkit_request(declaration))
                                except ValueError:
                                    pass
                            if ext!=".csh" and not is_local_research:
                                docs.append((name,content))
                            if ext==".cli" or size<=256*1024 and (ext in {".json",".csh",".cio",".inp",".nml"} or
                                    p.name.casefold()=="namelist.input"):
                                row["original_probe_payload"]=blob
                            row["engineering_text_status"]="complete"
                            row["basis"].append("工程说明全文已读取")
            elif reader and ext in {".csv",".tsv"}:
                if open_member or local:
                    with (open_member() if open_member else local.open("rb")) as stream:
                        blob=table_prefix(stream,size,row["format"])
                else:
                    # A materialized member can be read directly.
                    blob=reader(size)
                count+=len(blob)
                if ext==".csv" and (b"LEGEND FOR FIELD NAMES" in blob[:200] or
                        blob.lstrip(b"\xef\xbb\xbf").startswith(b"Latitude:") and b"Daymet" in blob[:2000]):
                    if open_member and not local and ref[1]=="tar":
                        with local_member(row) as source:
                            original_fields(row,source.read_bytes())
                    else:
                        original_fields(row,reader(size))
                else:
                    tabular(row,blob,len(blob)==size)
            elif open_member and not local and row["format"] in SCIENTIFIC:
                # Random access to the member is needed by the scientific
                # readers. Stream to disk in fixed-size buffers, then scan
                # metadata and coordinate samples using the regular reader.
                with TemporaryDirectory(prefix="ecodata-scan-member-") as directory:
                    temporary=Path(directory)/("member"+ext)
                    with open_member() as source, temporary.open("wb") as destination:
                        copyfileobj(source,destination,length=1024*1024)
                    copied=temporary.stat().st_size
                    temporary_bytes+=copied
                    count+=copied
                    if copied!=size:
                        raise ValueError("压缩成员实际大小与清单不一致")
                    scientific(row,temporary)
            elif reader and ext in {".sqlite",".sqlite3",".db"}:
                prefix=reader(16)
                count+=len(prefix)
                if prefix==b"SQLite format 3\x00":
                    row["original_sqlite_header"]=True
            elif local and ext in FORMATS:
                scientific(row,local)
            elif ext in FORMATS:
                row["pending"].append("压缩成员仅登记，待定向核验内部结构")
            if reader and not doc and not row["fields"] and not kind(name) and row["format"]=="其他":
                prefix=decode(reader(min(size,1024)))
                if prefix and all(char in "\r\n\t" or char.isprintable() for char in prefix) and "\ufffd" not in prefix:
                    row["classification_preview"]=prefix
            if row["fields"] and not row["resolution"].get("time_seconds"):
                row["pending"].append("时间间隔未知或不适用")
            return row

        def attempt(name,size,ref,reader=None,local=None,open_member=None,containers=()):
            nonlocal count,temporary_bytes
            row=None
            try:
                row=item(name,size,ref,reader,local,open_member,containers)
                archive_kind=kind(name)
                if not row or not archive_kind or not ref[1]:
                    return
                if local is not None:
                    walk_archive(local,archive_kind,name,ref[0],(*containers,(ref,size)))
                elif open_member is not None:
                    with TemporaryDirectory(prefix="ecodata-nested-") as directory:
                        temporary=Path(directory)/"archive"
                        with open_member() as source, temporary.open("wb") as destination:
                            copyfileobj(source,destination,length=1024*1024)
                        copied=temporary.stat().st_size
                        count+=copied
                        temporary_bytes+=copied
                        if copied!=size:
                            raise ValueError("内层归档实际大小与清单不一致")
                        walk_archive(temporary,archive_kind,name,ref[0],(*containers,(ref,size)))
                else:
                    raise ValueError("内层归档没有可读取的成员入口")
                row["nested_members_read"]=True
            except (EcoDataError,ValueError,OSError,RuntimeError,ImportError,csv.Error,tarfile.TarError,
                    zipfile.BadZipFile,EOFError) as exc:
                warnings.append(f"{name}: {exc}")
                if row is None and rows and rows[-1]["name"]==name:
                    row=rows[-1]
                if row is not None:
                    row["pending"].append("内层归档扫描失败，详见扫描警告" if kind(name)
                                          else "字段或坐标读取未完成，详见扫描警告")

        def walk_archive(path,archive_kind,label,original,containers=()):
            nonlocal temporary_bytes
            namespace=label+"!/" if containers else ""
            if progress:
                progress(f"正在扫描归档 · {PurePosixPath(label).name}")

            def member(name,size,offset,open_member=None,local=None):
                if not safe(name):
                    return

                def prefix(n):
                    with open_member() as stream:
                        return stream.read(n)

                attempt(namespace+name,size,(original,archive_kind,name,offset),
                        prefix if open_member else None,local,open_member,containers)

            if archive_kind=="zip":
                with path.open("rb") as raw:
                    meter=Meter(raw)
                    with zipfile.ZipFile(meter) as archive:
                        for entry in archive.infolist():
                            if not entry.is_dir():
                                member(entry.filename,entry.file_size,entry.header_offset,
                                       lambda entry=entry:archive.open(entry))
                    sequential.append(dict(name=label,raw_read_bytes=meter.count))
            elif archive_kind=="tar":
                with path.open("rb") as raw:
                    physical=Meter(raw)
                    header=physical.read(6)
                    raw.seek(0)
                    op=next((fn for magic,fn in [(b"\x1f\x8b",gzip.GzipFile),(b"BZh",bz2.BZ2File),
                                                (b"\xfd7zXZ",lzma.LZMAFile)] if header.startswith(magic)),None)
                    with (op(fileobj=physical) if op is gzip.GzipFile else op(physical) if op else nullcontext(physical)) as stream:
                        decoded=Meter(stream)
                        try:
                            with tarfile.open(fileobj=decoded,mode="r|") as archive:
                                for entry in archive:
                                    if entry.isfile():
                                        member(entry.name,entry.size,entry.offset_data,
                                               lambda entry=entry:archive.extractfile(entry))
                        finally:
                            sequential.append(dict(name=label,raw_read_bytes=physical.count,decoded_stream_bytes=decoded.count))
            elif archive_kind=="7z":
                import py7zr
                with TemporaryDirectory(prefix="ecodata-scan-7z-") as directory, ExitStack() as stack:
                    try:
                        archive=stack.enter_context(py7zr.SevenZipFile(path))
                        written=0
                        for entry,temporary in seven_zip_members(archive,Path(directory),stack,shallow=shallow):
                            if temporary is not None:
                                written+=temporary.stat().st_size
                            member(entry.filename,entry.uncompressed,None,
                                   (lambda temporary=temporary:temporary.open("rb")) if temporary is not None else None,
                                   temporary)
                        temporary_bytes+=written
                        sequential.append(dict(name=label,archive_bytes=path.stat().st_size,decoded_payload_bytes=written))
                    except py7zr.exceptions.ArchiveError as exc:
                        raise ValueError(str(exc)) from exc

        for path,size,*_ in signature:
            name=path.name if root.is_file() else path.relative_to(root).as_posix()
            archive_kind=kind(path)
            if archive_kind:
                try:
                    walk_archive(path,archive_kind,name,path)
                except (ValueError,OSError,RuntimeError,tarfile.TarError,zipfile.BadZipFile,ImportError,EOFError) as exc:
                    warnings.append(f"{name}: {exc}")
            else:
                def read(n,path=path):
                    with path.open("rb") as stream:
                        return stream.read(n)
                attempt(name,size,(path,"",None,None),read,path)
        cmaq_grid_checks=[]
        if not shallow:
            from .engineering_links import collect
            from .cmaq_grid_audit import audit_scan
            from .time_metadata import attach_swat_print_periods
            if progress:
                progress("正在关联工程说明与数据字段…")
            collect(rows,docs)
            attach_swat_print_periods(rows,docs)
            shapefile_attribute_rows(rows,docs)
            mark_engineering_netcdf(rows,docs)
            cmaq_grid_checks=audit_scan(rows)
        if tuple(inventory(root))!=signature: raise ValueError("扫描期间原始文件已变化，请重新扫描")
        result=dict(path=root,signature=signature,rows=rows,warnings=warnings,engineering_docs=docs,
                    cmaq_grid_checks=cmaq_grid_checks,shallow=shallow)
        self.cached=root,signature,shallow,result
        return result,dict(seconds=time.perf_counter()-start,cached=False,scan_payload_bytes=count,
                           sequential=sequential, scan_temp_bytes=temporary_bytes)

def mark_engineering_netcdf(rows, docs):
    """Record engineering references for NetCDF metadata already read by the scanner."""
    purpose=re.compile(r"用途|输入|驱动|对照|配对|融合|分辨率|网格|单位|观测|验证|"
                       r"comparison|fusion|resolution|model_input|forcing|observation|units",re.I)
    note_suffixes={".md",".rst",".txt"}
    prose=[text for name,text in docs if PurePosixPath(name).suffix.casefold() in note_suffixes]
    referenced_dirs={match.casefold().rstrip("/")+"/" for text in prose if purpose.search(text)
                     for match in re.findall(r"\x60([^\x60\r\n]+/)\x60",text)}
    from .document_content import classify
    descriptive=set()
    structured_purpose=re.compile(r'"(?:purpose|description|model_input_candidate|用途|说明|变量说明)"\s*:',re.I)
    for name,text in docs:
        suffix=PurePosixPath(name).suffix.casefold()
        if not purpose.search(text):
            continue
        if suffix in note_suffixes or suffix==".json" and structured_purpose.search(text):
            document=classify(name,text)
            if document.mode in {"full","mixed"}:
                descriptive.add(name)
    for row in rows:
        if row["format"]!="NetCDF" or not row["fields"]:
            continue
        exact=any(item["document"] in descriptive for item in row["evidence"])
        folder=any(row["name"].casefold().startswith(prefix) for prefix in referenced_dirs)
        if exact or folder:
            row["engineering_probe"]=True


def assert_current(scan):
    if tuple(inventory(scan["path"]))!=scan["signature"]:
        raise ValueError("原文件已变化，旧预览与授权已失效。请重新扫描。")

@contextmanager
def local_member(row):
    """Expose one scanned member as a seekable file for format-specific sampling."""
    path,*reference=row["ref"]
    with ExitStack() as stack:
        for parent,size in row.get("containers",()):
            path=stack.enter_context(materialize_member((path,*parent[1:]),size))
        yield stack.enter_context(materialize_member((path,*reference),row["bytes"]))


@contextmanager
def materialize_member(ref, size):
    path,kind,name,offset=ref
    if not kind:
        yield path
        return
    with TemporaryDirectory(prefix="ecodata-probe-") as directory, ExitStack() as stack:
        root=Path(directory)
        if kind=="7z":
            import py7zr
            archive=stack.enter_context(py7zr.SevenZipFile(path))
            member=next((member for entry,member in seven_zip_members(archive,root,stack,only=name)
                         if entry.filename==name and member is not None),None)
            if member is None or member.stat().st_size!=size:
                raise ValueError("归档成员已变化")
            yield member
            return
        destination=root/("member"+PurePosixPath(name).suffix)
        with archive_stream(path,kind,name,offset) as stream, destination.open("wb") as output:
            copyfileobj(stream,output,length=1024*1024)
        if destination.stat().st_size!=size:
            raise ValueError("归档成员已变化")
        yield destination


@contextmanager
def archive_stream(path,kind,name,offset):
    with ExitStack() as stack:
        if kind=="zip":
            archive=stack.enter_context(zipfile.ZipFile(path))
            item=next((item for item in archive.infolist()
                       if item.header_offset==offset and item.filename==name),None)
            stream=stack.enter_context(archive.open(item)) if item is not None else None
        elif kind=="tar":
            archive=stack.enter_context(tarfile.open(path,"r|*"))
            item=next((item for item in archive if item.isfile() and item.offset_data==offset and item.name==name),None)
            stream=stack.enter_context(archive.extractfile(item)) if item is not None else None
        else:
            raise ValueError("不支持此归档格式")
        if stream is None:
            raise ValueError("归档成员已变化")
        yield stream


@contextmanager
def open_tabular_member(row):
    """Sample archive tables without materializing the potentially large leaf."""
    path,kind,name,offset=row["ref"]
    with ExitStack() as stack:
        for parent,size in row.get("containers",()):
            path=stack.enter_context(materialize_member((path,*parent[1:]),size))
        if kind in {"zip","tar"}:
            yield stack.enter_context(archive_stream(path,kind,name,offset))
        else:
            local=stack.enter_context(materialize_member((path,kind,name,offset),row["bytes"]))
            yield stack.enter_context(local.open("rb"))


def read_selected(row, limit=256*1024**2):
    path,k,name,offset=row["ref"]
    if row["bytes"]>limit: raise ValueError(f"单成员深读超过 {limit//1024**2} MiB；请先准备所需字段或数据块")
    if row.get("containers"):
        with local_member(row) as member, member.open("rb") as stream:
            return stream.read(limit+1)
    if k=="zip":
        with zipfile.ZipFile(path) as z:
            with z.open(next(x for x in z.infolist() if x.header_offset==offset and x.filename==name)) as f:
                return f.read(limit+1)
    if k=="tar":
        with tarfile.open(path,"r|*") as t:
            for member in t:
                if member.offset_data==offset and member.name==name and member.isfile():
                    with t.extractfile(member) as f: return f.read(limit+1)
        raise ValueError("成员已变化")
    if k=="7z":
        from ecodata.models.archive_inputs import normalize_archives
        from ecodata.interface.engineering_upload import LocalUpload
        with TemporaryDirectory(prefix="ecodata-selected-") as d:
            uploads,_=normalize_archives([LocalUpload(path.name,path)],d)
            with zipfile.ZipFile(uploads[0].path) as z:
                with z.open(name) as f: return f.read(limit+1)
    with path.open("rb") as f: return f.read(limit+1)

