"""Uploaded-project binding, independent of machine-specific converter settings.

Namelist syntax belongs to f90nml. Model drivers declare file roles, not cases.
Binding verifies file identity and references; it does not certify native physics.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import re
import zipfile


CAMX_INPUTS = {
    'chemistry_parameters', 'photolysis_rates', 'photolyis_rates', 'ozone_column',
    'initial_conditions', 'boundary_conditions', 'top_concentrations',
    'point_sources', 'emiss_grid', 'surface_grid', 'met3d_grid', 'met2d_grid',
    'vdiff_grid', 'cloud_grid', 'srfmod_grid',
}


def safe_name(name):
    name = str(name).replace('\\', '/')
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or ':' in name or '\x00' in name:
        raise ValueError('工程成员路径不安全：' + name)
    return p.as_posix()


class ProjectFiles:
    """Lazy archive/directory resources. No extraction or uploaded code execution."""
    def __init__(self, uploads, stack):
        self.files = {}
        self._folded = set()
        for upload in uploads:
            local = getattr(upload, 'path', None)
            if str(upload.name).lower().endswith('.zip'):
                source = Path(local) if local is not None else io.BytesIO(upload.getvalue())
                archive = stack.enter_context(zipfile.ZipFile(source))
                for item in archive.infolist():
                    if item.is_dir():
                        continue
                    if (item.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError('工程包不接收符号链接：' + item.filename)
                    self.add(item.filename, lambda a=archive, i=item: a.open(i),
                             dict(upload=upload.name, member=item.filename, bytes=item.file_size))
            else:
                if local is not None:
                    path = Path(local)
                    self.add(upload.name, lambda p=path: p.open('rb'),
                             dict(upload=upload.name, bytes=path.stat().st_size))
                else:
                    raw = upload.getvalue()
                    self.add(upload.name, lambda b=raw: io.BytesIO(b),
                             dict(upload=upload.name, bytes=len(raw)))

    def add(self, name, opener, locator):
        name = safe_name(name)
        if name.casefold() in self._folded:
            raise ValueError('工程成员名称重复或大小写冲突：' + name)
        self._folded.add(name.casefold())
        self.files[name] = (opener, locator)

    def resolve(self, control, reference):
        reference = reference.strip().replace('\\', '/')
        parts = PurePosixPath(reference).parts
        if not reference or '\x00' in reference:
            raise ValueError('工程引用越界：' + reference)
        absolute = reference.startswith('/') or bool(re.match(r'^[A-Za-z]:', reference))
        if not absolute:
            # Resolve only against the virtual uploaded root, never the host FS.
            resolved = list(PurePosixPath(control).parent.parts)
            for part in parts:
                if part == '..':
                    if not resolved:
                        raise ValueError('工程引用越界：' + reference)
                    resolved.pop()
                elif part != '.':
                    if ':' in part:
                        raise ValueError('工程引用越界：' + reference)
                    resolved.append(part)
            exact = '/'.join(resolved)
            if exact in self.files:
                return exact, 'control_relative'
            # Relative references are exact: missing does not mean relocate.
            raise ValueError('工程引用缺失：' + reference)
        # Historical machine roots can be relocated only by a unique >=2-part
        # suffix, never by basename or by opening the referenced host path.
        if '..' in parts:
            raise ValueError('工程引用越界：' + reference)
        parts = [p for p in parts if p not in ('/', '') and ':' not in p]
        for length in range(len(parts), 1, -1):
            suffix = '/'.join(parts[-length:])
            hits = [n for n in self.files if n == suffix or n.endswith('/' + suffix)]
            if len(hits) > 1:
                raise ValueError('工程迁移引用歧义：' + reference)
            if hits:
                return hits[0], 'unique_path_suffix:' + str(length)
        raise ValueError('绝对路径无法在上传资料内唯一绑定：' + reference)

    def prefix(self, name, length=65536):
        with self.files[name][0]() as stream:
            return stream.read(length)

    def identity(self, name):
        with self.files[name][0]() as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return dict(self.files[name][1], path=name, sha256=digest)


def _values(value, indexes=()):
    if isinstance(value, list):
        for i, child in enumerate(value):
            yield from _values(child, indexes + (i,))
    else:
        yield indexes, value


def bind_project(uploads, model_id):
    """Return None for ordinary data; fail closed for recognised broken projects."""
    if model_id not in {'camx', 'wrf'}:
        return None  # Other native formats retain their existing drivers.
    import f90nml
    group_name = 'camx_control' if model_id == 'camx' else 'time_control'
    with ExitStack() as stack:
        files = ProjectFiles(uploads, stack)
        controls = []
        for name in sorted(files.files):
            prefix = files.prefix(name, 4096).removeprefix(b'\xef\xbb\xbf')
            prefix = re.sub(rb'(?m)^\s*![^\r\n]*(?:\r?\n|$)', b'', prefix)
            if prefix.lstrip().lower().startswith(('&' + group_name).encode()):
                controls.append(name)
        if not controls:
            return None
        if len(controls) != 1:
            raise ValueError('多个工程控制文件，不能自动猜选：' + '、'.join(controls))
        control = controls[0]
        with files.files[control][0]() as stream:
            nml = f90nml.read(io.StringIO(stream.read().decode('utf-8-sig')))
        group = nml[group_name]
        refs, disabled = [], []
        if model_id == 'camx':
            for required in ('chemistry_parameters', 'photolyis_rates', 'ozone_column', 'boundary_conditions'):
                if required not in group and not (required == 'photolyis_rates' and 'photolysis_rates' in group):
                    raise ValueError('CAMx缺少工程输入声明：' + required)
            if not group.get('restart', False) and not group.get('initial_conditions'):
                raise ValueError('冷启动缺少initial_conditions')
            keys = set(CAMX_INPUTS)
            restart_keys = {'master_grid_restart', 'nested_grid_restart', 'pig_restart'}
            if group.get('restart', False):
                if not group.get('master_grid_restart'):
                    raise ValueError('重启缺少master_grid_restart')
                keys |= restart_keys
            else:
                disabled.extend(dict(parameter=k, reason='restart=false') for k in sorted(restart_keys & set(group)))
            for key in sorted(keys & set(group)):
                for index, value in _values(group[key]):
                    if value is None or isinstance(value, str) and not value.strip():
                        continue
                    if not isinstance(value, str):
                        raise ValueError('工程文件引用必须为字符串：' + key)
                    refs.append((key, index, value))
        else:
            # Reuse the existing WRF contract: primary initial files + boundary.
            domains = nml.get('domains', {})
            count = domains.get('max_dom')
            if not isinstance(count, int) or not 1 <= count <= 99:
                raise ValueError('WRF需要明确合法max_dom')
            if group.get('restart', False):
                raise ValueError('WRF重启文件名依赖时间格式，当前自动绑定尚不支持')
            refs = [('initial_state', (i,), f'wrfinput_d{i:02d}') for i in range(1, count + 1)]
            refs.append(('boundary_state', (), 'wrfbdy_d01'))
        bindings, identities = [], {}
        for key, indexes, reference in refs:
            target, reason = files.resolve(control, reference)
            if target not in identities:
                identities[target] = files.identity(target)
            bindings.append(dict(parameter=key, indexes=list(indexes), original=reference,
                                 target=target, reason=reason, sha256=identities[target]['sha256']))
        return dict(schema='ecodata.project-binding.v1', model_id=model_id,
                    status='BOUND', control=files.identity(control), parameters=nml.todict(),
                    bindings=bindings, resources=list(identities.values()), inactive=disabled,
                    unbound_files=sorted(set(files.files) - set(identities) - {control}),
                    source_modified=False, payload_hashes_verified=True,
                    native_schema_verified=False, native_model_executed=False,
                    parser=dict(name='f90nml', version=f90nml.__version__),
                    scope='显式输入引用绑定与内容指纹；不等于科学一致性、融合或原生运行验收')


@contextmanager
def bound_resource(uploads, binding, parameter, indexes=()):
    """Consumers receive the actual verified upstream bytes, never a case path."""
    rows = [r for r in binding['bindings'] if r['parameter'] == parameter and r['indexes'] == list(indexes)]
    if len(rows) != 1:
        raise ValueError('绑定端口不存在或有歧义：' + parameter)
    row = rows[0]
    with ExitStack() as stack:
        files = ProjectFiles(uploads, stack)
        if files.identity(row['target'])['sha256'] != row['sha256']:
            raise ValueError('绑定后源数据已变化：' + row['target'])
        with files.files[row['target']][0]() as stream:
            yield stream


def prepare_project_binding(uploads, model_id, folder, progress=None):
    if progress:
        progress('解析上传工程引用，流式校验实际依赖文件')
    binding = bind_project(uploads, model_id)
    if binding is None:
        return None
    folder = Path(folder)
    path = folder / 'project_binding.json'
    path.write_text(json.dumps(binding, ensure_ascii=False, indent=2, default=str), encoding='utf8')
    if model_id == 'camx':
        from ecodata.models.native_meteorology import prepare_meteorology
        result = prepare_meteorology(uploads, binding, folder, progress=progress)
        if result is not None:
            return result
    manifest = dict(protocol='ecodata.converter.v1', model_id=model_id,
                    status='ENGINEERING_BOUND', native_ready=False, products=[],
                    preparation=dict(route='uploaded_project_binding_v1', tasks=[],
                                     fusion_decision=dict(status='NOT_EXECUTED'),
                                     binding='project_binding.json'),
                    remaining=['原生格式与科学契约验证', '待处理变量及融合路线绑定（如需要）'],
                    note='工程引用已自动绑定；下载为绑定记录，不含原始大文件，不是可直接运行的模型包')
    handoff = folder / 'handoff.json'
    handoff.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    archive = folder / 'EcoData-project-binding.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for item in (path, handoff, folder / 'archive_preflight.json', folder / 'scientific_inventory.json'):
            if item.is_file():
                z.write(item, item.name)
    return dict(manifest=manifest, archive=str(archive))
