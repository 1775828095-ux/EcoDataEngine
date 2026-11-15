"""Check CMAQ static GRIDBDY2D / GRIDCRO2D without rewriting either input."""
from contextlib import contextmanager
from pathlib import PurePosixPath
import re

import numpy as np

from .scan import read_selected

FIELDS = {
    "MSFX2": ("地图投影比例因子的平方", "1"),
    "HT": ("地形高程", "m"),
    "DLUSE": ("主导土地利用类别", "1"),
    "LWMASK": ("陆水掩膜", "1"),
    "PURB": ("陆地面积中的城市比例", "%"),
}
GRID = ("GDTYP","P_ALP","P_BET","P_GAM","XCENT","YCENT",
        "XORIG","YORIG","XCELL","YCELL","NCOLS","NROWS")
LIMIT = 8*1024**2
EARTH_RADIUS = re.compile(r"EARTH RADIUS ASSUMED IN MCIP:\s*([0-9.]+)\s*m")


def identity(row):
    path = PurePosixPath(row["name"])
    match = re.fullmatch(r"grid(bdy|cro)2d(.*)\.(?:nc|nc4)",path.name.casefold())
    if match:
        return str(path.parent).casefold(), match[2], match[1]
    return None


def is_companion(left, right):
    a, b = identity(left), identity(right)
    return bool(a and b and a[:2]==b[:2] and a[2]!=b[2])


@contextmanager
def open_grid(row):
    from netCDF4 import Dataset

    if row["bytes"] > LIMIT:
        raise ValueError("单成员超过 8 MiB 核验预算")
    path, archive, _, _ = row["ref"]
    with (Dataset(str(path)) if not archive else Dataset("grid.nc",memory=read_selected(row,limit=LIMIT))) as data:
        yield data


def grid_metadata(data):
    from ecodata.models.adapters.cmaq_gridded_emissions import _number

    integers = {"GDTYP","NCOLS","NROWS"}
    return {name:_number(data.getncattr(name),name,name in integers) for name in GRID}


def geographic(data, grid, cols, rows):
    from pyproj import CRS, Transformer

    x = grid["XORIG"]+(cols-.5)*grid["XCELL"]
    y = grid["YORIG"]+(rows-.5)*grid["YCELL"]
    if grid["GDTYP"]==1:
        return x,y
    radius = EARTH_RADIUS.search(str(getattr(data,"FILEDESC","")))
    if grid["GDTYP"]!=2 or not radius or grid["XCENT"]!=grid["P_GAM"]:
        raise LookupError("目前可核验经纬度网格或有明确地球半径的 MCIP Lambert 网格")
    crs = CRS.from_proj4(f'+proj=lcc +lat_1={grid["P_ALP"]} +lat_2={grid["P_BET"]} '
                         f'+lat_0={grid["YCENT"]} +lon_0={grid["P_GAM"]} +R={radius[1]} +units=m')
    return Transformer.from_crs(crs,4326,always_xy=True).transform(x,y)


def compare_file(data, dims, grid, cols, rows):
    from ecodata.models.scientific_schema import compare_units

    if not {"LON","LAT"} <= set(data.variables):
        raise LookupError("文件缺少显式 LAT/LON，无法证实索引顺序")
    lon, lat = geographic(data,grid,cols,rows)
    for key, expected in (("LON",lon),("LAT",lat)):
        var = data[key]
        if var.dimensions != dims:
            raise ValueError(f"{key} 维度与 I/O API 网格角色不一致")
        values = values_at_surface(var)
        difference = values-expected
        if key=="LON":
            difference = (difference+180)%360-180
        if values.shape!=expected.shape or not np.isfinite(difference).all() or np.max(np.abs(difference))>1e-4:
            raise ValueError(f"{key} 与声明格心或周界索引顺序不符")

    results = {}
    for name, (label, target_unit) in FIELDS.items():
        if name not in data.variables:
            continue
        var = data[name]
        quantity = label
        unit = str(getattr(var,"units","")).strip()
        report = compare_units(quantity,target_unit,quantity,unit)
        description = str(getattr(var,"var_desc","")).strip()
        issues = []
        if var.dimensions != dims:
            issues.append("维度不匹配")
        if report["status"] not in {"equivalent","convertible"}:
            issues.append("单位缺失或不兼容")
        elif not issues:
            array = values_at_surface(var)*report["scale"]+report["offset"]
            if not np.isfinite(array).all():
                issues.append("含缺值或非有限值")
            finite = array[np.isfinite(array)]
            if name=="LWMASK" and not np.isin(finite,[0,1]).all():
                issues.append("陆水类别须为 0 或 1")
            if name=="DLUSE" and np.any((finite!=np.floor(finite)) | (finite<0)):
                issues.append("土地利用类别须为非负整数编码")
            if name=="PURB" and np.any((finite<0) | (finite>100)):
                issues.append("百分比超出 0–100")
            if name=="MSFX2" and np.any(finite<=0):
                issues.append("投影比例因子平方须为正")
        results[name] = dict(label=label,unit=unit,description=description,issues=issues)
    return results


def values_at_surface(var):
    return np.ma.asarray(var[0,0],dtype=float).filled(np.nan)


def audit_pair(boundary, interior):
    from ecodata.models.adapters.cmaq_boundary import external_perimeter

    with open_grid(boundary) as bdy, open_grid(interior) as cro:
        if (getattr(bdy,"FTYPE",None),getattr(cro,"FTYPE",None)) != (2,1):
            raise ValueError("边界和内部文件的 FTYPE 与声明角色不符")
        if any(len(ds.dimensions[dim])!=1 for ds in (bdy,cro) for dim in ("TSTEP","LAY")):
            raise LookupError("本检查针对单层单时刻的静态网格文件")
        if getattr(bdy,"NTHIK",None)!=1:
            raise LookupError("本检查复用原版 NTHIK=1 外圈索引")
        if str(getattr(bdy,"GDNAM","")).strip()!=str(getattr(cro,"GDNAM","")).strip():
            raise ValueError("两份文件的网格名称 GDNAM 不一致")
        bg, cg = grid_metadata(bdy),grid_metadata(cro)
        if any(not np.isclose(bg[k],cg[k],atol=1e-7,rtol=0) for k in GRID):
            raise ValueError("两份文件的投影、原点、格距或声明行列数不一致")
        if cg["GDTYP"]==2:
            radii = [EARTH_RADIUS.search(str(getattr(ds,"FILEDESC",""))) for ds in (bdy,cro)]
            if not all(radii):
                raise LookupError("至少一份 Lambert 网格缺少 MCIP 地球半径声明")
            if float(radii[0][1])!=float(radii[1][1]):
                raise ValueError("两份文件的 MCIP 地球半径不同")
        nc,nr = cg["NCOLS"],cg["NROWS"]
        if min(nc,nr,cg["XCELL"],cg["YCELL"])<=0 or nc*nr>300000:
            raise LookupError("网格大小超出本地定向核验范围，或声明的行列／格距无效")
        pairs = external_perimeter(nc,nr)
        if (len(cro.dimensions["ROW"]),len(cro.dimensions["COL"]),len(bdy.dimensions["PERIM"])) != (nr,nc,len(pairs)):
            raise ValueError("ROW、COL 或 PERIM 与声明网格大小不符")
        xb,yb = np.asarray(pairs).T
        xc,yc = np.meshgrid(np.arange(1,nc+1),np.arange(1,nr+1))
        bf = compare_file(bdy,("TSTEP","LAY","PERIM"),bg,xb,yb)
        cf = compare_file(cro,("TSTEP","LAY","ROW","COL"),cg,xc,yc)
        fields = []
        for name, (label, _) in FIELDS.items():
            left,right = bf.get(name),cf.get(name)
            issues = []
            if not left or not right:
                issues.append("边界或内部文件缺少此字段")
            else:
                issues.extend("边界："+issue for issue in left["issues"])
                issues.extend("内部："+issue for issue in right["issues"])
                key = lambda value: " ".join(value["description"].casefold().split())
                if not key(left) or key(left)!=key(right):
                    issues.append("字段说明或分类体系需要复核")
            fields.append(dict(field=name,label=label,issues=issues))
        return dict(status="需复核字段" if any(f["issues"] for f in fields) else "通过",
                    perimeter=len(pairs),interior_cells=nr*nc,fields=fields,
                    reason="按各自格心核验边界顺序与内部坐标。外圈与内部代表不同格心。")


def audit_scan(rows):
    groups = {}
    for row in rows:
        token = identity(row)
        if token:
            groups.setdefault(token[:2],{}).setdefault(token[2],[]).append(row)
    checks = []
    remaining = 24*1024**2
    for group in groups.values():
        boundary, interior = group.get("bdy",[]), group.get("cro",[])
        record = dict(files=[r["name"] for r in boundary+interior])
        if len(boundary)!=1 or len(interior)!=1:
            record.update(status="待核验",reason="同目录同网格标识的边界与内部文件无法唯一配对")
        else:
            size = boundary[0]["bytes"]+interior[0]["bytes"]
            if size>remaining:
                record.update(status="待核验",reason="已达 CMAQ 定向核验读取预算")
            else:
                remaining -= size
                try:
                    record.update(audit_pair(boundary[0],interior[0]))
                except (LookupError,KeyError) as exc:
                    record.update(status="待核验",reason=str(exc))
                except (OSError,ValueError,RuntimeError) as exc:
                    record.update(status="未通过",reason=str(exc))
        checks.append(record)
    return checks


def explain(checks):
    for row in checks:
        names = " 与 ".join(PurePosixPath(name).name for name in row["files"])
        yield f"CMAQ 网格与字段核验 · {row['status']}\n{names}\n{row['reason']}"
        if "perimeter" in row:
            yield f"已核对 {row['perimeter']} 个边界格心和 {row['interior_cells']} 个内部格心。"
            for field in row["fields"]:
                yield f"{field['field']}（{field['label']}）· " + ("；".join(field["issues"]) or "字段定义、单位和取值检查通过")
