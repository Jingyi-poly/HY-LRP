"""Source-bound RouteOpt proposal provider. Never compiles during a solve.

The receipt establishes reproducible local build provenance, not an independent
optimization certificate. NG outputs remain proposal-only in the consumer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import ModuleType

SCHEMA = 'routeopt_source_bound_build_v1'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dependencies(path):
    # clang make dependency files escape embedded spaces with a backslash.
    value = Path(path).read_text().replace('\\\n', ' ')
    return {str(Path(p).resolve()) for p in shlex.split(value.split(':', 1)[1])}


def _sources(here):
    packages = here.parents[3] / 'RouteOpt-main/packages'
    pricing = packages / 'application/cvrp/src/pricing'
    return [here / 'pricing_cli.cpp', *sorted((pricing / 'src').glob('*.cpp')),
            packages / 'rank1_cuts/common/src/rank1cuts.cpp',
            *sorted((packages / 'rank1_cuts/chg_rc_getter/src').glob('*.cpp')),
            packages / 'rounded_cap_cuts/chg_rc_getter/src/get_chg_rcc_rc.cpp']


def _mandatory(here):
    return {str((here / name).resolve()) for name in
            ('build.py', 'pricing.py', 'verified_pricing.py', 'pricing_cli.cpp')}


def compile_verified(command, *, deadline=None):
    """Build the unchanged compiler command as source-bound translation units.

    Called only by build.py. A failed/interrupted build leaves no valid receipt.
    Both dependency discovery and compilation consume the same finite deadline.
    """
    here = Path(__file__).resolve().parent
    output = here / 'build'
    receipt = output / 'verified_build.json'
    receipt.unlink(missing_ok=True)
    started = time.monotonic()
    deadline = started + 120. if deadline is None else float(deadline)
    if not math.isfinite(deadline) or deadline <= started:
        raise ValueError('A future finite build deadline is required')
    first = next(i for i, value in enumerate(command) if value.endswith('.cpp'))
    compiler, flags = command[0], command[1:first]
    sources = [Path(p).resolve() for p in command[first:command.index('-o')]]
    if sources != [p.resolve() for p in _sources(here)]:
        raise ValueError('Unexpected RouteOpt translation units')
    executable = output / 'routeopt_pricing'
    if Path(command[-1]).resolve() != executable.resolve():
        raise ValueError('Unexpected RouteOpt output')
    compiler_path = Path(shutil.which(compiler) or compiler).resolve()
    protected = _mandatory(here) | {str(compiler_path)}
    hashes = {p: _sha(p) for p in protected}
    depdir = output / 'verified_dependencies'
    objdir = output / 'verified_objects'
    depdir.mkdir(parents=True, exist_ok=True)
    objdir.mkdir(parents=True, exist_ok=True)
    commands, units, objects = [], [], []

    def call(argv):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('RouteOpt verified build deadline')
        commands.append(list(map(str, argv)))
        subprocess.run(argv, cwd=here, timeout=remaining, check=True)

    for index, source in enumerate(sources):
        dep, obj = depdir / f'{index:02d}.d', objdir / f'{index:02d}.o'
        call([compiler, *flags, '-M', '-MF', str(dep), '-MT', str(obj), str(source)])
        before = _dependencies(dep)
        for path in before:
            digest = _sha(path)
            if path in hashes and hashes[path] != digest:
                raise ValueError('Build input changed: ' + path)
            hashes[path] = digest
        call([compiler, *flags, '-MD', '-MF', str(dep), '-c', str(source), '-o', str(obj)])
        after = _dependencies(dep)
        if after != before or str(source) not in after:
            raise ValueError('Compile dependency closure changed')
        if any(hashes[p] != _sha(p) for p in after):
            raise ValueError('Actual compile input changed')
        units.append(dict(source=str(source), depfile=str(dep.resolve()),
                          depfile_sha256=_sha(dep), object_sha256=_sha(obj)))
        objects.append(str(obj))
    temporary = output / f'routeopt_pricing.pending.{os.getpid()}'
    try:
        call([compiler, *objects, '-o', str(temporary)])
        if any(_sha(p) != digest for p, digest in hashes.items()):
            raise ValueError('A source changed during the build')
        if time.monotonic() >= deadline:
            raise TimeoutError('Build exceeded its deadline')
        os.replace(temporary, executable)
        record = dict(schema=SCHEMA, status='BUILD_PASS', command=command,
                      compiler=str(compiler_path), inputs=hashes, units=units,
                      binary=str(executable.resolve()), binary_sha256=_sha(executable),
                      commands=commands, wall_seconds=time.monotonic()-started)
        pending = receipt.with_suffix('.pending')
        pending.write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
        os.replace(pending, receipt)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def _stamp(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _skip(status):
    return dict(routes=[], min_rc=None, pricing_complete=False, lb_certified=False,
                status=status, seconds=0., verified_provider=False)


class VerifiedPricing:
    """Owned by one bounded CG call; mutations invalidate, never recertify."""
    def __init__(self, price, tracked, deadline, receipt_hash):
        self._price, self._tracked = price, tracked
        self._deadline, self.receipt_sha256 = deadline, receipt_hash

    def __call__(self, *args, **kwargs):
        started = time.monotonic()
        requested = float(kwargs.get('time_limit_s', 1.))
        if not math.isfinite(requested) or requested <= 0:
            raise ValueError('Positive finite pricing time required')
        stop = min(self._deadline, started + requested)
        if started >= stop:
            return _skip('verified_provider_deadline')
        # Includes every source/header, receipt, adapter and binary. Checking
        # ctime as well as mtime also catches rewriting then restoring mtime.
        try:
            for path, stamp in self._tracked.items():
                if time.monotonic() >= stop:
                    return _skip('verified_provider_deadline')
                if _stamp(path) != stamp:
                    return _skip('verified_provider_changed')
        except OSError:
            return _skip('verified_provider_changed')
        kwargs['time_limit_s'] = stop-time.monotonic()
        if kwargs['time_limit_s'] <= 0:
            return _skip('verified_provider_deadline')
        result = self._price(*args, **kwargs)
        try:
            if any(_stamp(path) != stamp for path, stamp in self._tracked.items()):
                return _skip('verified_provider_changed')
        except OSError:
            return _skip('verified_provider_changed')
        if time.monotonic() >= stop:
            # Preserve separately audited route candidates, but a wrapper
            # overrun cannot be reported as complete bounded pricing.
            result.update(pricing_complete=False, lb_certified=False,
                          min_rc=None, status='verified_provider_deadline')
        result['verified_provider'] = True
        result['verified_build_sha256'] = self.receipt_sha256
        result['seconds'] = time.monotonic()-started
        return result


def load_verified_pricing(*, deadline):
    """Return (provider, diagnostics); absent/old/mutated builds fail closed.

    No compile, optimizer, or subprocess is called by this factory. Verification
    belongs to the caller's existing absolute budget; no hidden extra budget.
    """
    started = time.monotonic()
    deadline = float(deadline)
    if not math.isfinite(deadline):
        raise ValueError('A finite absolute deadline is required')
    report = dict(available=False, status='NO_BUDGET')

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('provider verification deadline')

    try:
        check()
        here = Path(__file__).resolve().parent
        receipt = here / 'build/verified_build.json'
        if not receipt.is_file():
            report['status'] = 'VERIFIED_BUILD_REQUIRED'
            return None, report
        record = json.loads(receipt.read_text())
        if record.get('schema') != SCHEMA or record.get('status') != 'BUILD_PASS':
            raise ValueError('Unrecognized verified build receipt')
        inputs, units = record['inputs'], record['units']
        expected = [str(p.resolve()) for p in _sources(here)]
        if not units or [u['source'] for u in units] != expected:
            raise ValueError('Incomplete translation unit list')
        binary = here / 'build/routeopt_pricing'
        if record['binary'] != str(binary.resolve()):
            raise ValueError('Wrong binary path')
        # Dependency files supply a second, compiler-produced closure; deleting
        # an input entry or translation unit cannot silently weaken the check.
        required = _mandatory(here) | {record['compiler']}
        extra = {str(receipt): _sha(receipt), str(binary): record['binary_sha256']}
        for unit in units:
            check()
            dep = Path(unit['depfile'])
            if dep.parent.resolve() != (here / 'build/verified_dependencies').resolve():
                raise ValueError('Dependency manifest outside verified build')
            if _sha(dep) != unit['depfile_sha256']:
                raise ValueError('Dependency manifest changed')
            closure = _dependencies(dep)
            if unit['source'] not in closure:
                raise ValueError('Missing translation unit dependency')
            required.update(closure)
            extra[str(dep)] = unit['depfile_sha256']
        if set(inputs) != required:
            raise ValueError('Incomplete compiler dependency hash map')
        tracked = {}
        for path, digest in {**inputs, **extra}.items():
            check()
            before = _stamp(path)
            if _sha(path) != digest or _stamp(path) != before:
                raise ValueError('Source/binary changed: ' + path)
            tracked[path] = before
        check()
        # Load the exact verified bytes, never a potentially stale .pyc.
        path = here / 'pricing.py'
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != inputs[str(path.resolve())]:
            raise ValueError('Pricing adapter changed while loading')
        module = ModuleType('_verified_routeopt_pricing')
        module.__file__ = str(path)
        exec(compile(source, str(path), 'exec'), module.__dict__)
        check()
        provider = VerifiedPricing(module.price_routes, tracked, deadline, extra[str(receipt)])
        report.update(available=True, status='VERIFIED', inputs=len(inputs),
                      binary_sha256=record['binary_sha256'], receipt_sha256=extra[str(receipt)])
        return provider, report
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        report.update(status='VERIFIED_PROVIDER_UNAVAILABLE', reason=str(error))
        return None, report
    finally:
        report['seconds'] = time.monotonic()-started
