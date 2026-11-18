"""Evidence-bearing header and sheet selection; originals are never rewritten."""
import csv
import io
import json
import re

import pandas as pd

from ecodata.core.exceptions import SchemaError
from ecodata.io.record_layout import RecordLayoutError,split_declared_record

DATE = re.compile(r"(?:date|time|采样时间|日期|时间)", re.I)
DATA_DATE = re.compile(r"^\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}")


def select_excel_sheet(book):
    scores = []
    frames = pd.read_excel(book, sheet_name=None, keep_default_na=False, dtype=object)
    for name, frame in frames.items():
        if name == '_ecodata_metadata':
            continue
        date_columns = [str(c) for c in frame if DATE.search(str(c))]
        numeric = sum(pd.to_numeric(frame[c], errors='coerce').notna().sum() for c in frame)
        cells = max(frame.size, 1)
        complete = sum(not str(c).startswith('Unnamed:') for c in frame) / max(len(frame.columns), 1)
        score = 5 * bool(date_columns) + 3 * numeric / cells + complete + min(len(frame), 1000) / 1000
        scores.append(dict(sheet=name, score=score, date_columns=date_columns, numeric_cells=int(numeric), rows=len(frame), header_completeness=complete, explanatory_fraction=1-numeric/cells))
    scores.sort(key=lambda x: x['score'], reverse=True)
    if not scores or (len(scores)>1 and scores[0]['score'] == scores[1]['score']):
        raise SchemaError('Excel 数据 sheet 不明确，请明确选择，不能默认使用第一张表')
    return frames[scores[0]['sheet']], dict(selected_sheet=scores[0]['sheet'], sheet_scores=scores, confidence='high' if scores[0]['date_columns'] and scores[0]['numeric_cells'] else 'needs_confirmation')


def locate_header(text):
    lines = text.splitlines()
    for i, line in enumerate(lines[:100]):
        if not DATE.search(line) or DATA_DATE.search(line):
            continue
        delimiter = '\t' if '\t' in line else ',' if ',' in line else None
        if delimiter:
            columns = next(csv.reader([line], delimiter=delimiter))
            if len(columns)<2:
                continue
        else:
            columns = re.split(r'\s{2,}', line.strip())
        start = next((j for j in range(i+1, min(i+8,len(lines))) if DATA_DATE.search(lines[j])), None)
        if start is None:
            continue
        unit_row=i+1 if start>i+1 and not lines[i+1].strip().startswith('-') else None
        if unit_row is None and i>0 and re.search(r'mg/L|mS/cm|m/d/y',lines[i-1],re.I):unit_row=i-1
        return dict(header_row=i, unit_row=unit_row,
                    data_start_row=start, metadata_rows=list(range(i)), delimiter=delimiter,
                    confidence='high' if delimiter else 'needs_adapter', evidence=lines[max(0,i-1):start+1])
    raise SchemaError('未可靠定位日期/表头/数据起始行，需要用户确认或专用 Adapter')


def read_monitoring_text(content, fixed_layout=None):
    try:
        text = content.decode('utf-8-sig')
        encoding = 'utf-8-sig'
    except UnicodeDecodeError:
        text = content.decode('cp1252')
        encoding = 'cp1252'
    evidence = locate_header(text)
    lines = text.splitlines()
    header, start, sep = evidence['header_row'], evidence['data_start_row'], evidence['delimiter']
    units = {}
    if sep:
        names = [x.strip() for x in next(csv.reader([lines[header]], delimiter=sep))]
        original_names=list(names)
        repeated={x for x in names if names.count(x)>1}
        names=[f'{x}__column_{i+1}' if x in repeated else x for i,x in enumerate(names)]
        evidence['source_field_names']=dict(zip(names,original_names))
        if evidence['unit_row'] is not None:
            units = dict(zip(names, [x.strip() for x in next(csv.reader([lines[evidence['unit_row']]], delimiter=sep))]))
        numeric_columns=[index for index,name in enumerate(names) if units.get(name) and not DATE.search(name)]
        records=[]
        annotations=[]
        for number,line in enumerate(lines[start:],start+1):
            if not DATA_DATE.search(line):
                continue
            cells=next(csv.reader([line],delimiter=sep))
            if len(cells)>len(names):
                if not numeric_columns:
                    raise SchemaError(f'第{number}行有额外列，但缺少字段类型或单位依据，无法识别尾部附注')
                try:
                    cells,extra=split_declared_record(cells,len(names),numeric_columns=numeric_columns)
                except RecordLayoutError as error:
                    raise SchemaError(f'第{number}行无法确认尾部附注：{error.code}') from error
                annotations.append(dict(line=number,extra_tokens=len(extra),basis='trailing_text_annotation'))
            if len(cells)<len(names):
                cells.extend(['']*(len(names)-len(cells)))
            records.append(cells)
        frame=pd.DataFrame(records,columns=names)
        if annotations:
            evidence['trailing_annotations']=annotations
    else:
        if not fixed_layout:
            raise SchemaError('固定宽度表头粘连，需要已核实的仪器 Adapter；不猜测字段')
        pattern = '|'.join(re.escape(x) for x in sorted(fixed_layout['fields'], key=len, reverse=True))
        names = re.findall(pattern, lines[header])
        data = [x for x in lines[start:] if DATA_DATE.search(x)]
        spans=[]
        for line in data[:50]:
            date=re.match(r'^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?',line)
            if date is None:continue
            tokens=list(re.finditer(r'\S+',line[date.end():]))
            if len(tokens)==len(names)-1:
                spans.append([date.end()]+[date.end()+x.end() for x in tokens])
        if not spans:raise SchemaError('固定宽度数值列没有可靠对齐证据')
        ends=[max(x[i] for x in spans) for i in range(len(names))]
        colspecs=list(zip([0]+ends[:-1],ends))
        frame = pd.read_fwf(io.StringIO('\n'.join(data)), header=None, colspecs=colspecs,dtype=str,keep_default_na=False)
        evidence['colspecs']=colspecs
        if len(frame.columns)!=len(names):
            raise SchemaError('固定宽度字段数量与已核实表头不一致，禁止错位读取')
        frame.columns = names
        for row, line in enumerate(data):
            date=re.match(r'^\s*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?',line)
            tokens=line[date.end():].split() if date else []
            parsed=[str(frame.iloc[row,i]).strip() for i in range(1,len(names))]
            if len(tokens)!=len(names)-1 or tokens!=parsed:
                raise SchemaError('固定宽度数据列存在粘连或错位，无法无损确认，禁止继续')
        units = {name:unit for name,unit in fixed_layout['units'].items() if name in names}
        evidence.update(confidence='high', adapter=fixed_layout['id'])
    if not frame.columns.is_unique or frame.empty:
        raise SchemaError('仪器数据为空或字段重复')
    evidence.update(encoding=encoding, source_data_rows=len(frame), units=units,
                    preserved_non_data_lines=[x for j,x in enumerate(lines) if j<start or not DATA_DATE.search(x)])
    return frame, evidence


def attach_table_evidence(dataset, evidence):
    dataset.attrs['reader_evidence'] = json.dumps(evidence, ensure_ascii=False)
    for name, unit in evidence.get('units', {}).items():
        if name in dataset and unit and not DATE.search(unit):
            dataset[name].attrs.update(units=unit, unit_evidence='source unit row / documented instrument layout')
    for name,source in evidence.get('source_field_names',{}).items():
        if name in dataset:dataset[name].attrs['source_field_name']=source
    return dataset
