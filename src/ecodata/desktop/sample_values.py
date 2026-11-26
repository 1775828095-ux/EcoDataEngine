"""Read small numerical samples from fields registered by the local scanner."""
import csv
from io import TextIOWrapper
from itertools import islice, product, zip_longest
from math import ceil, isfinite, nextafter
from secrets import randbelow

from .scan import AXES, local_member, open_tabular_member

SAMPLE_SIZE = 16
PROBE_ROWS = 4096
MAX_SAMPLE_MATERIALIZATION = 8 * 1024**2
IDENTIFIER_COLUMNS = {"id", "station_id", "site_id", "sensor_id", "object_id"}


def numeric(values):
    for value in values:
        try:
            number=float(value)
        except (TypeError,ValueError,OverflowError):
            yield None
        else:
            yield number if isfinite(number) else None


def field_values(values,field):
    """Use the same registered compound codec for advice and local acceptance."""
    if field.get('noaa_component'):
        from ecodata.io.noaa_global_hourly import COMPONENTS,decode
        source,width,component=COMPONENTS[field['name']]
        if field.get('source_column',source)!=source:
            raise ValueError('复合字段与已登记来源列冲突')
        return (decode(value,width,component) for value in values)
    return numeric(values)


def perturb(values):
    result=[]
    for value in values:
        if value is None:
            continue
        shifted=value+(randbelow(999)+1)/1000
        if shifted==value or not isfinite(shifted):
            shifted=nextafter(value,0.0)
        result.append(shifted)
    return result


def numeric_grid(variable):
    import numpy as np
    shape=variable.shape
    if not shape:
        value=variable[()]
        return [None if np.ma.is_masked(value) else value]
    if not all(shape):
        return []
    # Scalar indexing works for both chunked HDF5 and NetCDF and avoids
    # loading the variable, including sparse or very large grids.
    width=max(1,int(PROBE_ROWS**(1/len(shape))))
    axes=[range(0,count,max(1,ceil(count/width))) for count in shape]
    return (None if np.ma.is_masked(value) else value for value in
            (variable[index] for index in islice(product(*axes),PROBE_ROWS)))


def tabular_values(row,field):
    delimiter="\t" if row["format"]=="TSV" else ","
    for encoding in ("utf-8-sig","gb18030"):
        try:
            with open_tabular_member(row) as raw, TextIOWrapper(raw,encoding=encoding,newline="") as stream:
                rows=csv.reader(stream,delimiter=delimiter)
                headers=next(rows,[])
                original=field.get("source_column") or field.get("original_header",field["name"])
                column=field.get("column")
                if column is None:
                    column=headers.index(original)
                if column>=len(headers) or headers[column]!=original:
                    raise ValueError("表头已变化")
                return [row[column] if len(row)>column else None for row in islice(rows,PROBE_ROWS)]
        except UnicodeDecodeError:
            if encoding=="gb18030":
                raise
    return []


def netcdf_window_values(variable):
    """Sample short windows spread through the time axis in one pass."""
    import numpy as np

    if not variable.shape or not all(variable.shape):
        return numeric_grid(variable)
    time_axis = next((axis for axis, name in enumerate(variable.dimensions)
                      if name.casefold() in {"time", "tstep", "date", "datetime"}), None)
    if time_axis is None:
        return islice(numeric_grid(variable), SAMPLE_SIZE * 4)

    count = variable.shape[time_axis]
    window_count = min(4, count)
    width = min(16, count // window_count)
    windows = []
    for index in range(window_count):
        start = index * (count - width) // max(1, window_count - 1)
        selection = [index * (size - 1) // max(1, window_count - 1) for size in variable.shape]
        selection[time_axis] = slice(start, start + width)
        windows.append(np.ma.asarray(variable[tuple(selection)]).ravel())

    # Give each selected window a share of the outbound sample budget.
    return (None if np.ma.is_masked(value) else value
            for group in zip_longest(*windows) for value in group)


def raw_sample(path,row,field,*,outbound=False):
    fmt=row["format"]
    name=field["name"]
    if row.get("original_reader"):
        from ecodata.io.structures import read_structure
        with read_structure(row["name"],path.read_bytes(),record_read_source=False) as dataset:
            values=dataset[name]
            if values.ndim==1:
                values=values.isel({values.dims[0]:slice(0,PROBE_ROWS)})
            return list(numeric(islice(values.values.flat,PROBE_ROWS)))
    if fmt in {"Shapefile","DBF"}:
        import shapefile
        with path.open("rb") as stream, shapefile.Reader(dbf=stream) as dataset:
            return list(numeric(record[name] for record in islice(dataset.iterRecords(fields=[name]),PROBE_ROWS)))
    if fmt=="Excel":
        from openpyxl import load_workbook
        with path.open("rb") as stream:
            book=load_workbook(stream,read_only=True,data_only=True)
            try:
                sheet=book[field["sheet"]] if field.get("sheet") else book.active
                column=field["column"]+1
                values=sheet.iter_rows(min_col=column,max_col=column,min_row=2,max_row=PROBE_ROWS+1,values_only=True)
                return list(numeric(value[0] for value in values))
            finally:
                book.close()
    if fmt=="Parquet":
        from pyarrow.parquet import ParquetFile
        with path.open("rb") as stream:
            parquet=ParquetFile(stream)
            rows=itertools_batches(parquet.iter_batches(columns=[name],batch_size=PROBE_ROWS))
            return list(numeric(rows))
    if fmt=="HDF5":
        import h5py
        with h5py.File(path) as dataset:
            return list(numeric(numeric_grid(dataset[name])))
    if fmt=="NetCDF":
        from netCDF4 import Dataset
        with Dataset(str(path)) as dataset:
            variable = dataset.variables[name]
            values = netcdf_window_values(variable) if outbound else numeric_grid(variable)
            return list(numeric(values))
    if fmt=="GeoTIFF":
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(path) as dataset:
            band=field["band"]
            from itertools import chain
            windows=(Window(max(0,(dataset.width-size)*x//2),max(0,(dataset.height-size)*y//2),
                            min(dataset.width,size),min(dataset.height,size))
                     for size,x,y in ((32,0,0),(32,1,1),(32,2,2)))
            values=chain.from_iterable(dataset.read(band,window=window,masked=True).ravel().tolist()
                                       for window in windows)
            return list(numeric(values))
    return None


def itertools_batches(batches):
    from itertools import chain
    return islice(chain.from_iterable(batch.column(0).to_pylist() for batch in batches),PROBE_ROWS)


def field_sample(row,field,*,outbound=False):
    if outbound:
        name = str(field.get("source_column") or field["name"]).strip().casefold()
        if field.get("axis") or name in AXES or name in IDENTIFIER_COLUMNS:
            return None
        member = row.get("attribute_member", row)
        if any(size > MAX_SAMPLE_MATERIALIZATION for _, size in member.get("containers", ())):
            return None
        archive = member["ref"][1]
        streaming_table = (row["format"] in {"CSV", "TSV"} and not row.get("original_reader")
                           or row.get("swat_output_header") is not None)
        materializes_member = archive and (not streaming_table or archive not in {"zip", "tar"})
        if (row.get("original_reader") or materializes_member) and member["bytes"] > MAX_SAMPLE_MATERIALIZATION:
            return None
    if row.get("swat_output_header") is not None:
        with open_tabular_member(row) as raw, TextIOWrapper(raw,encoding="utf-8-sig",errors="strict") as stream:
            for _ in range(row["swat_output_header"]+2):
                next(stream,None)
            column=field["column"]
            values=[]
            for line in islice(stream,PROBE_ROWS):
                parts=line.split()
                values.append(parts[column].replace("D","E") if len(parts)>column else None)
            return [value if value is not None and abs(value)<1e30 else None for value in numeric(values)]
    if row["format"] in {"CSV","TSV"} and not row.get("original_reader"):
        values=tabular_values(row,field)
        return list(field_values(values,field))
    with local_member(row.get("attribute_member",row)) as path:
        return raw_sample(path,row,field,outbound=outbound)


def export_sample(values):
    if values is None:
        return dict(status="unavailable",basis="sample_format_unavailable")
    sample=perturb(islice((value for value in values if value is not None),SAMPLE_SIZE))
    return dict(status="checked" if sample else "unavailable",
                basis="random_decimal_perturbation" if sample else "no_numeric_samples",
                perturbed_samples=sample)

