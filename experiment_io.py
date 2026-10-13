"""Atomic, portable experiment artifacts (no implicit git/network writes)."""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], 'little')


def atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_text(path, text):
    atomic(path, lambda f: f.write(text.encode('utf-8')))


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_manifest(root):
    root = Path(root)
    entries = [{'file': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
                'sha256': digest(p)} for p in sorted(root.rglob('*'))
               if p.is_file() and p.suffix not in ('.tmp', '.lock')
               and p.name not in ('file_manifest.json', 'status.json')]
    write_json(root / 'file_manifest.json', entries)


def verify_manifest(root):
    for entry in read_json(Path(root) / 'file_manifest.json'):
        path = Path(root) / entry['file']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError(f'Artifact checksum mismatch: {entry["file"]}')


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO(newline='')
    w = csv.DictWriter(buf, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
    write_text(path, '\ufeff' + buf.getvalue())


def save_npz(path, **arrays):
    atomic(path, lambda f: np.savez_compressed(f, **arrays))


def save_checkpoint(path, state):
    atomic(path, lambda f: torch.save(state, f))


def load_checkpoint(path):
    # Only load checkpoints produced by this run, never untrusted external pickles.
    return torch.load(path, map_location='cpu', weights_only=False)


def model_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return {'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


@contextlib.contextmanager
def keep_awake(enabled=True):
    active = False
    if enabled and platform.system() == 'Windows':
        import ctypes
        active = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        if not active:
            warnings.warn('Could not prevent idle sleep. Keep AC power/lid open.')
    try:
        yield
    finally:
        if active:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                   encoding='utf-8', stderr=subprocess.STDOUT).strip()


def publish(root):
    """Stage only this completed run. Never include unrelated staged changes."""
    root = Path(root).resolve()
    repo = Path(git(root, 'rev-parse', '--show-toplevel')).resolve()
    relative = root.relative_to(repo).as_posix()
    if not relative or relative == '.':
        raise ValueError('Output must be a dedicated subdirectory of the repository.')
    if read_json(root / 'status.json')['status'] != 'complete':
        raise ValueError('Only complete, verified runs can be published.')
    verify_manifest(root)
    if git(repo, 'diff', '--cached', '--name-only'):
        raise ValueError('Unrelated staged changes exist; commit/unstage them before --publish.')
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix not in ('.tmp', '.lock')]
    for path in files:
        if path.stat().st_size >= 90 * 1024 * 1024:
            raise ValueError(f'Artifact exceeds safe Git file size: {path.name}')
        ignored = subprocess.run(['git', '-C', str(repo), 'check-ignore', '-q',
                                  path.relative_to(repo).as_posix()], capture_output=True)
        if ignored.returncode == 0:
            raise ValueError(f'Artifact is git-ignored: {path.relative_to(repo)}')
        if ignored.returncode not in (0, 1):
            raise RuntimeError(ignored.stderr.decode(errors='replace'))
    # Add exact artifacts; no credentials, data cache, or other workspace changes.
    for offset in range(0, len(files), 40):
        git(repo, 'add', '--', *[p.relative_to(repo).as_posix() for p in files[offset:offset+40]])
    if git(repo, 'diff', '--cached', '--name-only'):
        git(repo, 'commit', '-m', f'Add verified experiment results: {root.name}')
    branch = git(repo, 'symbolic-ref', '--short', 'HEAD')
    git(repo, 'push', 'origin', f'HEAD:refs/heads/{branch}')
    return git(repo, 'rev-parse', 'HEAD')
