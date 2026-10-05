"""Build with the selected Python and publish only an import-verified extension.

Run build.sh as before; PYTHON_BIN selects the interpreter. CXX, ESPPRC_SRC,
and ESP_BUILD_BACKUP retain their previous meanings. Solver workers only load
this build and never compile. Relative manifest paths remain repo-portable.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile


HERE = Path(__file__).resolve().parent


def _atomic_bytes(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    if len(sys.argv) != 1:
        raise SystemExit('build.sh takes no arguments; use PYTHON_BIN/CXX/ESPPRC_SRC environment settings')
    try:
        import pybind11
    except ImportError as exc:
        raise SystemExit(f'{sys.executable} needs pybind11; select the solver environment with PYTHON_BIN') from exc
    source = Path(os.environ.get('ESPPRC_SRC', HERE / 'espprc.cpp')).expanduser()
    if not source.is_absolute():
        source = HERE / source
    source = source.resolve()
    source_bytes = source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not suffix:
        raise SystemExit('This interpreter does not declare a Python extension suffix')
    output_name = 'espprc_cpp' + suffix
    build_dir = HERE / '.native-build' / sys.implementation.cache_tag
    build_dir.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    includes = list(dict.fromkeys((sysconfig.get_path('include'), sysconfig.get_path('platinclude'),
                                  pybind11.get_include())))
    flags = ['-O3', '-Wall', '-shared', '-std=c++17', '-fPIC', '-fvisibility=hidden', '-DNDEBUG', '-march=native']
    flags.extend('-I' + path for path in includes if path)
    if sys.platform == 'darwin':
        flags.extend(['-undefined', 'dynamic_lookup'])
    print(f'[build] Python: {sys.executable} ({sys.implementation.cache_tag})', flush=True)
    print(f'[build] compiling {source} -> {output_name}', flush=True)
    with tempfile.TemporaryDirectory(dir=build_dir, prefix='.building-') as temporary:
        staged = Path(temporary) / output_name
        command = [*compiler, *flags, str(source), '-o', str(staged)]
        subprocess.run(command, check=True)
        if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha:
            raise RuntimeError('C++ source changed during compilation; build was not published')
        print('[build] verifying import...', flush=True)
        subprocess.run([sys.executable, '-c',
            "import importlib.util,sys; "
            "s=importlib.util.spec_from_file_location('espprc_cpp',sys.argv[1]); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "assert callable(m.solve_pctsp); print('  module:',m.__file__)", str(staged)], check=True)
        binary_bytes = staged.read_bytes()
        binary_sha = hashlib.sha256(binary_bytes).hexdigest()
        # A versioned path lets an already running interpreter load a new
        # verified build without dlopen returning the old same-path image.
        version_dir = build_dir / binary_sha
        version_dir.mkdir(exist_ok=True)
        binary = version_dir / output_name
        _atomic_bytes(binary, binary_bytes)
        legacy = HERE / output_name
        if os.environ.get('ESP_BUILD_BACKUP', '0') == '1' and legacy.is_file():
            shutil.copy2(legacy, str(legacy) + '.bak.' + datetime.now().strftime('%H%M%S%f'))
        _atomic_bytes(legacy, binary_bytes)
        manifest = dict(schema='lrp_native_build_v1', binary=str(binary.relative_to(build_dir)),
            binary_sha256=binary_sha, source_sha256=source_sha,
            source=str(source), python_executable=sys.executable,
            python_version=sys.version, python_cache_tag=sys.implementation.cache_tag,
            extension_suffix=suffix, platform=sys.platform, machine=platform.machine(),
            pybind11_version=pybind11.__version__, compiler=compiler, compile_flags=flags,
            built_at=datetime.now(timezone.utc).isoformat())
        manifest_path = build_dir / 'native_build.json'
        _atomic_bytes(manifest_path, (json.dumps(manifest, indent=2) + '\n').encode())
    print(f'[build] manifest: {manifest_path}', flush=True)
    print(f'[build] legacy import: {legacy}', flush=True)
    if hashlib.sha256((HERE / 'espprc.cpp').read_bytes()).hexdigest() != source_sha:
        print('[build] alternate source differs from espprc.cpp; the LRP loader will reject this manifest', flush=True)
    print('[build] OK', flush=True)


if __name__ == '__main__':
    main()
