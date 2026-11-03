"""Read-only WRF bundle routing guard; never convert or substitute products."""
from pathlib import PurePosixPath
import io
import json
import zipfile


def inspect_wrf_bundle(uploads, folder, contract):
    """Recognise engineering inputs before generic tabular/gzip readers run."""
    records = []
    for upload in uploads or []:
        if PurePosixPath(upload.name).suffix.lower() != '.zip':
            continue
        with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                name = PurePosixPath(item.filename.replace('\\', '/'))
                if '..' in name.parts or name.is_absolute():
                    raise ValueError('ZIP 中包含不安全路径：' + item.filename)
                base = name.name.lower()
                role = None
                if base == 'namelist.input':
                    role = 'wrf_run_configuration'
                elif base.startswith(('wrfinput_d', 'wrfbdy_d')):
                    # Confirm native NetCDF content, not filename alone.
                    with archive.open(item) as stream:
                        magic = stream.read(8)
                    if magic.startswith((b'CDF', b'\x89HDF')):
                        role = 'wrf_native_initial' if base.startswith('wrfinput') else 'wrf_native_boundary'
                elif base.endswith(('.tar.gz', '.tgz')):
                    role = 'nested_archive_requires_reader'
                if role:
                    records.append(dict(file=item.filename, size=item.file_size, role=role))
    if not records:
        return
    roles = {item['role'] for item in records}
    available_fields = [field['name'] for field in (contract or {}).get('fields', [])]
    receipt = dict(schema='ecodata.wrf.bundle_inventory.v1', files=records,
                   current_converter_fields=available_fields, original_inputs_modified=False,
                   status='UNSUPPORTED_ENGINEERING_ROUTE', native_model_executed=False)
    path = folder / 'wrf_bundle_inventory.json'
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf8')
    if 'wrf_native_initial' in roles and 'wrf_native_boundary' in roles:
        raise ValueError('已识别 WRF 原生初始场、边界场及配套工程资料。'
                         '当前新版接口只接通气压层气象场转 WPS，尚未接通这类已有 WRF 工程及新资料更新路线；'
                         '这是平台接入能力缺口，不能据此判定用户资料缺失。不会改用旧工程或跳过资料。'
                         '识别记录：' + str(path))
    if 'nested_archive_requires_reader' in roles:
        raise ValueError('发现 TAR/GZIP 嵌套资料包，不能把它当作气象站文本读取。'
                         '当前入口尚未接通其内部资料读取与工程对应；识别记录：' + str(path))
