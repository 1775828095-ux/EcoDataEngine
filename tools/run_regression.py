"""Run the complete conversion-platform suite without simulation executables."""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/regression')
    parser.add_argument('tests', nargs='*')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, [ROOT / 'src', ROOT / 'verification_tools', ROOT])),
               PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    for path in (ROOT / 'src', ROOT / 'verification_tools', ROOT):
        sys.path.insert(0, str(path))
    for dependency in ('pytest', 'geographiclib', 'PseudoNetCDF'):
        try: importlib.import_module(dependency)
        except ImportError as exc:
            raise SystemExit(f'Missing verification dependency {dependency}; install requirements-verification.txt: {exc}')
    # HDF5/NetCDF native readers on Windows need an ASCII temporary path.
    temporary = Path(tempfile.gettempdir()) / 'ecodata_regression'
    temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='run_', dir=temporary) as temp:
        env['TEMP'] = env['TMP'] = temp
        command = [sys.executable, '-B', '-X', 'utf8', '-m', 'pytest',
                   *(args.tests or [str(ROOT / 'tests')]), '-q', '-o', 'addopts=',
                   '-p', 'no:cacheprovider', '--basetemp', str(Path(temp) / 'pytest'),
                   '--junitxml', str(output / 'results.xml')]
        (output / 'command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding='utf8')
        with (output / 'pytest.log').open('w', encoding='utf8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print((output / 'pytest.log').read_text(encoding='utf8')[-9000:])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
