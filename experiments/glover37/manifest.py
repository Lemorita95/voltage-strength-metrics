"""Run manifest written by every stage into its output folder (manifest.json).

A run is of record only if `uncommitted_changes` is false: the code that produced it
is exactly `git_commit`. Results are git-ignored, so the manifest's SHA-256 values are
what identifies them.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from experiments.glover37 import config

MANIFEST = 'manifest.json'


def now():
    return datetime.now(timezone.utc)


def git_state():
    """(commit, uncommitted changes outside results/)."""
    def git(*args):
        return subprocess.run(['git', *args], cwd=config.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        commit = git('rev-parse', 'HEAD')
        dirty = bool(git('status', '--porcelain', '--', '.', ':!results'))
    except (OSError, subprocess.CalledProcessError):
        return None, True
    return commit, dirty


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value.relative_to(config.ROOT)) if value.is_relative_to(config.ROOT) else str(value)
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def config_snapshot():
    """Every upper-case parameter of config.py, JSON-ready."""
    return {name: _plain(value) for name, value in vars(config).items()
            if name.isupper() and not name.startswith('_')}


def files(paths):
    return {_plain(Path(p)): sha256(p) for p in paths if Path(p).is_file()}


def write(out_dir, stage, *, started, inputs=(), outputs=(), extra=None):
    """Write out_dir/manifest.json; returns the manifest dict."""
    commit, dirty = git_state()
    finished = now()
    record = {
        'run_id': f'{started:%Y%m%dT%H%M%SZ}-{(commit or "nogit")[:7]}',
        'stage': stage,
        'git_commit': commit,
        'uncommitted_changes': dirty,
        'of_record': commit is not None and not dirty,
        'started_utc': started.isoformat(),
        'finished_utc': finished.isoformat(),
        'seconds': round((finished - started).total_seconds(), 1),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'pandas': pd.__version__},
        'inputs_sha256': files(inputs),
        'outputs_sha256': files(outputs),
        'config': config_snapshot(),
        **(extra or {}),
    }
    path = Path(out_dir) / MANIFEST
    path.write_text(json.dumps(record, indent=2, default=str), encoding='utf-8')
    return record


def read(directory):
    path = Path(directory) / MANIFEST
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
