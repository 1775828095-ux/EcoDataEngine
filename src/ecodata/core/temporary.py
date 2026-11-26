"""Owned desktop scratch space; never sweep inputs or infer ownership by name."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import psutil


MARKER = '.ecodata-owner.json'
SCHEMA = 'ecodata.owned-temporary.v1'


def scratch_root():
    base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / '.cache')
    return base / 'EcoDataEngine' / 'temporary-v1'


def _linked(path):
    return path.is_symlink() or path.is_junction()


def _owner(path):
    if _linked(path) or not path.is_dir():
        raise ValueError('Not an owned directory')
    marker = path / MARKER
    if _linked(marker):
        raise ValueError('Linked ownership marker')
    data = json.loads(marker.read_text(encoding='utf8'))
    if (data['schema'] != SCHEMA or data['directory'] != path.name
            or str(uuid.UUID(data['token'])) != data['token']
            or not isinstance(data['pid'], int) or data['pid'] <= 0):
        raise ValueError('Invalid ownership marker')
    return data


def _remove(path):
    root = scratch_root().absolute()
    path = Path(path).absolute()
    if (path == root or not path.resolve().is_relative_to(root)
            or _linked(root) or _linked(path)):
        raise ValueError('Deletion outside owned scratch root refused')
    # Refuse reparse points anywhere in the tree rather than follow external data.
    for parent, directories, files in os.walk(path, followlinks=False):
        if any(_linked(Path(parent) / name) for name in directories + files):
            raise ValueError('Linked content retained')
    size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
    shutil.rmtree(path)
    return size


def sweep_abandoned():
    """Only directories with valid markers and a proven departed owner qualify."""
    root = scratch_root()
    report = dict(schema=SCHEMA, removed=[], retained=[], failures=[], released_bytes=0)
    if not root.exists():
        return report
    if _linked(root) or root.resolve() != root.absolute():
        report['failures'].append(dict(path=str(root), reason='Unsafe scratch root'))
        return report
    for path in root.iterdir():
        try:
            owner = _owner(path)
            try:
                process = psutil.Process(owner['pid'])
                alive = abs(process.create_time() - owner['created']) < .001
            except psutil.NoSuchProcess:
                alive = False
            if alive:
                report['retained'].append(dict(path=str(path), reason='Owner still running'))
                continue
            size = _remove(path)
            report['removed'].append(dict(path=str(path), bytes=size))
            report['released_bytes'] += size
        except (OSError, ValueError, KeyError, TypeError, psutil.Error) as exc:
            report['failures'].append(dict(path=str(path), reason=str(exc)))
    return report


class TemporaryDirectory:
    """Keep tempfile's lifetime API; mark default scratch for crash recovery."""
    def __init__(self, suffix=None, prefix=None, dir=None):
        self.managed = dir is None
        if self.managed:
            root = scratch_root()
            root.mkdir(parents=True, exist_ok=True)
            if _linked(root) or root.resolve() != root.absolute():
                raise ValueError('Unsafe scratch root')
            dir = root
        self._temporary = tempfile.TemporaryDirectory(suffix=suffix, prefix=prefix, dir=dir)
        self.name = self._temporary.name
        if self.managed:
            data = dict(schema=SCHEMA, directory=Path(self.name).name, token=str(uuid.uuid4()),
                        pid=os.getpid(), created=psutil.Process().create_time())
            (Path(self.name) / MARKER).write_text(json.dumps(data), encoding='utf8')

    def cleanup(self):
        # Match tempfile's retry behavior: a failed deletion remains for startup.
        if self.managed and Path(self.name).exists():
            _owner(Path(self.name))
            _remove(Path(self.name))
        self._temporary.cleanup()

    def __enter__(self):
        return self.name

    def __exit__(self, *args):
        self.cleanup()
