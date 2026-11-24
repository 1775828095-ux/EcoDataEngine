"""One-variable native NetCDF template update with independent raw readback.

Uses netCDF4 for its actual format. No units, packing, coordinate interpretation,
resampling or scientific defaults are inferred here. The caller supplies values
in the native variable's numeric convention and an optional exact change mask.
The original is opened only for reading; a destination must not already exist.
"""
from itertools import product
from pathlib import Path
from contextlib import contextmanager
import hashlib
import math
import shutil
import tempfile
import numpy as np
from netCDF4 import Dataset, Variable


@contextmanager
def _open_native(path, mode):
    """Keep Windows Unicode paths out of netCDF-C, without re-encoding data."""
    path = Path(path)
    if str(path).isascii():
        with Dataset(path, mode) as dataset:
            yield dataset
    else:
        # Stream to an ASCII path rather than materialize the native container
        # in memory. Read-only originals are never copied back or edited.
        with tempfile.TemporaryDirectory(prefix='ecodata_native_') as directory:
            staged = Path(directory) / 'update.nc'
            if not str(staged).isascii():
                raise ValueError('Native writeback needs an ASCII temporary directory; set TMP/TEMP accordingly')
            if shutil.disk_usage(directory).free < path.stat().st_size:
                raise OSError('Insufficient temporary disk space for native I/O')
            shutil.copyfile(path, staged)
            try:
                with Dataset(staged, mode) as dataset:
                    yield dataset
            finally:
                # Preserve partial output for diagnosis on failure, as before.
                if mode != 'r':shutil.copyfile(staged, path)


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _equal(a, b):
    """NetCDF attributes compare exactly, including original NaN attributes."""
    left, right = np.asarray(a), np.asarray(b)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.fields:
        return all(_equal(left[name], right[name]) for name in left.dtype.names)
    if left.dtype.kind == 'O':
        return all(_equal(x, y) for x, y in zip(left.flat, right.flat))
    if left.dtype.kind in 'fc':
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _raw_equal(a, b):
    left, right = np.asarray(a), np.asarray(b)
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.dtype.kind == 'O':
        return _equal(left, right)
    # Preserve non-target raw bits as well as numerical values (NaN payload,
    # signed zero and integer storage included).
    return left.tobytes(order='C') == right.tobytes(order='C')


def _attrs(group):
    return {key: group.getncattr(key) for key in group.ncattrs()}


def _attrs_equal(a, b):
    return set(a) == set(b) and all(_equal(a[key], b[key]) for key in a)


def _groups(group):
    yield group
    for child in group.groups.values():
        yield from _groups(child)


def _variables(root):
    return {group.path.rstrip('/') + '/' + name: value
            for group in _groups(root) for name, value in group.variables.items()}


def _disable_decoding(root):
    for value in _variables(root).values():
        value.set_auto_maskandscale(False)
        value.set_auto_chartostring(False)


def _metadata(root):
    groups = {}
    variables = {}
    for group in _groups(root):
        groups[group.path] = dict(attributes=_attrs(group),
            dimensions={name: (len(dim), bool(dim.isunlimited()))
                        for name, dim in group.dimensions.items()})
    for name, value in _variables(root).items():
        variables[name] = dict(dimensions=value.dimensions, shape=value.shape,
            dtype=str(value.dtype), attributes=_attrs(value),
            filters=value.filters(), chunking=value.chunking(), endian=value.endian())
    return dict(data_model=root.data_model, groups=groups, variables=variables)


def _check_metadata(expected, actual):
    if expected['data_model'] != actual['data_model']:
        raise ValueError('Native NetCDF data model changed')
    if set(expected['groups']) != set(actual['groups']) or set(expected['variables']) != set(actual['variables']):
        raise ValueError('Native NetCDF group/variable inventory changed')
    for name, before in expected['groups'].items():
        after = actual['groups'][name]
        if before['dimensions'] != after['dimensions'] or not _attrs_equal(before['attributes'], after['attributes']):
            raise ValueError('Native group attributes/dimensions changed: ' + name)
    for name, before in expected['variables'].items():
        after = actual['variables'][name]
        if any(before[key] != after[key] for key in ('dimensions', 'shape', 'dtype', 'filters', 'chunking', 'endian')):
            raise ValueError('Native variable structure/encoding changed: ' + name)
        if not _attrs_equal(before['attributes'], after['attributes']):
            raise ValueError('Native variable attributes changed: ' + name)


def _chunks(shape, itemsize, byte_limit=4 * 1024 * 1024):
    if not shape:
        yield ()
        return
    block = list(shape)
    if any(size == 0 for size in block):
        return
    count = max(1, byte_limit // max(1, itemsize))
    for axis in range(len(block)):
        others = math.prod(block) // block[axis]
        block[axis] = min(block[axis], max(1, count // max(1, others)))
        if math.prod(block) <= count:
            break
    for start in product(*(range(0, size, step) for size, step in zip(shape, block))):
        yield tuple(slice(first, min(size, first + step))
                    for first, size, step in zip(start, shape, block))


def _native_cast(values, dtype):
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError('Masked update values are not a complete numeric input')
    original = np.asarray(values)
    dtype = np.dtype(dtype)
    if original.size == 0 or original.dtype.kind not in 'biuf' or dtype.kind not in 'biuf':
        raise ValueError('Native update requires a nonempty real numeric array')
    if not np.isfinite(original).all():
        raise ValueError('Native update contains non-finite values')
    if dtype.kind in 'iu':
        limits = np.iinfo(dtype)
        if np.any(original < limits.min) or np.any(original > limits.max):
            raise ValueError('Native integer dtype overflow')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        converted = original.astype(dtype)
    if not np.isfinite(converted).all():
        raise ValueError('Native dtype conversion overflow/non-finite output')
    if dtype.kind in 'iub' or original.dtype.kind in 'iub':
        with np.errstate(over='ignore', invalid='ignore'):
            back = converted.astype(original.dtype)
        if not np.array_equal(original, back):
            raise ValueError('Native dtype conversion would lose integer precision or truncate values')
    # Normal float64 -> float32 machine-precision rounding is permitted. Large
    # relative loss in the subnormal range or nonzero -> zero underflow is not.
    original_wide = original.astype(np.longdouble)
    converted_wide = converted.astype(np.longdouble)
    delta = np.abs(original_wide - converted_wide)
    nonzero = original_wide != 0
    relative = np.zeros(original.shape, dtype=np.longdouble)
    relative[nonzero] = delta[nonzero] / np.abs(original_wide[nonzero])
    if original.dtype.kind == 'f' and dtype.kind == 'f':
        if np.any(nonzero & (converted == 0)):
            raise ValueError('Native dtype conversion would underflow nonzero values to zero')
        if np.any(relative > np.finfo(dtype).eps):
            raise ValueError('Native dtype conversion error exceeds native machine precision')
    return converted, dict(source_dtype=str(original.dtype), native_dtype=str(dtype),
        max_absolute_cast_error=float(np.max(delta)), max_relative_cast_error=float(np.max(relative)))


def update_variable(template, destination, variable, values, allowed_mask=None):
    """Copy a native template and replace precisely one numeric variable.

    Returns a JSON-serializable checks/report dictionary. Packed or quantized
    target variables are rejected because this API has no encoding contract.
    Float casts permit only ordinary native machine-precision rounding, recorded
    in the report; integer conversions must be exact. Unchanged fields and mask
    exterior must preserve raw storage values. Failed outputs are retained for
    diagnosis and never returned as successful; the template is never written.
    """
    template, destination = Path(template), Path(destination)
    if not template.is_file():
        raise ValueError('Native template does not exist')
    if template.resolve() == destination.resolve():
        raise ValueError('Refusing to write the original native template')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Native destination already exists; refusing overwrite')
    canonical = '/' + str(variable).lstrip('/')
    with _open_native(template, 'r') as source:
        _disable_decoding(source)
        variables = _variables(source)
        if canonical not in variables or not isinstance(variables[canonical], Variable):
            raise ValueError('Native target variable is absent: ' + str(variable))
        target = variables[canonical]
        attributes = _attrs(target)
        for key, neutral in (('scale_factor', 1), ('add_offset', 0)):
            if key in attributes and not np.all(np.asarray(attributes[key]) == neutral):
                raise ValueError('Packed target needs an explicit encoding contract: ' + key)
        if str(attributes.get('_Unsigned', '')).casefold() == 'true':
            raise ValueError('Unsigned reinterpretation needs an explicit encoding contract')
        if 'least_significant_digit' in attributes or any(key.startswith('_Quantize') for key in attributes):
            raise ValueError('Quantized target cannot promise exact native-cast readback')
        if np.shape(values) != target.shape:
            raise ValueError('Update shape differs from native variable shape')
        converted, cast = _native_cast(values, target.dtype)
        if hasattr(target.datatype, 'enum_dict') and not np.isin(converted, list(target.datatype.enum_dict.values())).all():
            raise ValueError('Update contains a value outside the native enumeration')
        mask = None
        if allowed_mask is not None:
            supplied_mask = np.asarray(allowed_mask)
            if supplied_mask.dtype.kind != 'b':
                raise ValueError('Allowed change mask must be boolean')
            try:
                mask = np.broadcast_to(supplied_mask, target.shape)
            except ValueError as exc:
                raise ValueError('Allowed mask cannot broadcast to native shape') from exc
            for block in _chunks(target.shape, np.dtype(target.dtype).itemsize):
                outside = ~mask[block]
                if outside.any() and not _raw_equal(np.asarray(target[block])[outside], converted[block][outside]):
                    raise ValueError('Update changes values outside the allowed mask')
        before_metadata = _metadata(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        copied_hash = hashlib.sha256()
        # Exclusive creation also rejects a destination appearing between the
        # earlier check and copy. No existing output is ever overwritten.
        with template.open('rb') as src, destination.open('xb') as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)
                copied_hash.update(chunk)
        with _open_native(destination, 'r+') as result:
            _disable_decoding(result)
            result_variable = _variables(result)[canonical]
            for block in _chunks(target.shape, np.dtype(target.dtype).itemsize):
                result_variable[block] = converted[block]
        checks = []
        changed = 0
        with _open_native(destination, 'r') as result:
            _disable_decoding(result)
            _check_metadata(before_metadata, _metadata(result))
            for name, original in variables.items():
                actual = _variables(result)[name]
                chunks = 0
                for block in _chunks(original.shape, max(1, np.dtype(original.dtype).itemsize)):
                    left, right = np.asarray(original[block]), np.asarray(actual[block])
                    if name == canonical:
                        if not _raw_equal(right, converted[block]):
                            raise ValueError('Native output differs from the requested native-dtype cast')
                        if mask is not None:
                            outside = ~mask[block]
                            if outside.any() and not _raw_equal(left[outside], right[outside]):
                                raise ValueError('Native output changed outside the allowed mask')
                        changed += int(np.count_nonzero(left != right))
                    elif not _raw_equal(left, right):
                        raise ValueError('Non-target native variable changed: ' + name)
                    chunks += 1
                checks.append(dict(variable=name, chunks_compared=chunks,
                    status='PASS', target=name == canonical,
                    non_target_data_preserved=name != canonical))
    source_hash = _digest(template)
    if source_hash != copied_hash.hexdigest():
        raise ValueError('Native template changed during the update')
    return dict(status='PASS', variable=canonical, shape=list(converted.shape),
        native_data_model=before_metadata['data_model'], changed_values=changed,
        template_sha256=source_hash, destination_sha256=_digest(destination),
        original_template_preserved=True, all_metadata_preserved=True,
        all_non_target_variables_preserved=True, outside_allowed_mask_preserved=True,
        output_equals_native_dtype_cast=True, variable_checks=checks, cast=cast,
        scope='One native variable update and raw readback; no unit/scientific/coordinate interpretation or model execution')
