"""Colab Pro+ launcher: run binary study on local SSD, checkpoint to Drive safely.

Design:
- Drive stores immutable SHA256-addressed blobs plus a committed manifest.
- A new manifest is published ONLY after its blobs have been uploaded and checked.
- On reconnect restore the latest completely verifiable manifest (fallback to older).
- No cron, no Drive writes from deep-learning inner loops, no keepalive tricks.
- Backups continue ONLY while the Colab runtime and this foreground cell are running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

CHUNK = 4 * 1024 * 1024
BACKUP_VERSION = 1


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as inp:
        for buf in iter(lambda: inp.read(CHUNK), b""):
            hasher.update(buf)
    return hasher.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial_{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj):
    atomic_write_bytes(path, (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode())


def copy_verified(src: Path, dst: Path, expected: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + f".partial_{os.getpid()}")
    try:
        digest = hashlib.sha256()
        with src.open("rb") as inp, tmp.open("wb") as out:
            for chunk in iter(lambda: inp.read(CHUNK), b""):
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Source changed while copying: {src}")
        os.replace(tmp, dst)
        # Do not publish a manifest if the copy isn't readable with the correct hash.
        if sha256(dst) != expected:
            raise RuntimeError(f"Drive write could not be verified: {dst}")
    finally:
        tmp.unlink(missing_ok=True)


def list_payload_files(sources: dict[str, Path]):
    """Walk only committed files. The Colab variant prunes obsolete step checkpoints."""
    for group, root in sources.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp") or ".partial_" in path.name:
                continue
            if path.name.endswith(".lock") or path.name.startswith(".experiment.lock"):
                continue
            if "__pycache__" in path.parts:
                continue
            yield f"{group}/{path.relative_to(root).as_posix()}", path


class BackupStore:
    def __init__(self, drive: Path, sources: dict[str, Path]):
        self.drive = Path(drive)
        self.sources = {k: Path(p) for k, p in sources.items()}
        self.drive.mkdir(parents=True, exist_ok=True)
        self.hash_cache = {}

    def get_hash(self, path: Path):
        st = path.stat()
        k = str(path)
        cached = self.hash_cache.get(k)
        meta = (st.st_size, st.st_mtime_ns)
        if cached and cached[0] == meta:
            return cached[1], st.st_size
        answer = sha256(path)
        self.hash_cache[k] = (meta, answer)
        return answer, st.st_size

    def blob_path(self, sha):
        return self.drive / "blobs" / sha[:2] / sha

    def snapshot(self):
        entries = []
        uploaded = 0
        uploaded_bytes = 0
        for relative, path in list_payload_files(self.sources):
            digest, size = self.get_hash(path)
            dest = self.blob_path(digest)
            if not dest.exists():
                copy_verified(path, dest, digest)
                uploaded += 1
                uploaded_bytes += size
            entries.append({"path": relative, "sha256": digest, "bytes": size})
        # The snapshot is immutable and the latest pointer is committed last.
        manifest = {"format": BACKUP_VERSION, "created_at": utcnow(),
                    "file_count": len(entries), "entries": entries}
        raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n").encode()
        manifest_hash = hashlib.sha256(raw).hexdigest()
        manifest_name = f"snapshot_{int(time.time())}_{manifest_hash[:10]}.json"
        manifest_path = self.drive / "snapshots" / manifest_name
        atomic_write_bytes(manifest_path, raw)
        atomic_write_json(self.drive / "latest.json", {"manifest": manifest_name,
                          "sha256": manifest_hash, "updated_at": utcnow(),
                          "file_count": len(entries)})
        # Convenience copies: real backups are immutable blobs; these files make
        # research results easy to inspect directly in Google Drive.
        easy_paths = (
            "analysis_brief.md", "progress.json", "status.json", "raw/observations.csv",
            "processed/key_results.csv", "processed/validation_summary.csv",
            "processed/test_summary.csv", "processed/recovery_times.csv",
        )
        results = self.sources.get("results")
        if results:
            candidates = [results / name for name in easy_paths]
            candidates.extend((results / "figures").glob("*.png")
                              if (results / "figures").exists() else [])
            for source in candidates:
                if not source.is_file():
                    continue
                try:
                    rel = source.relative_to(results)
                    digest, _ = self.get_hash(source)
                    preview = self.drive / "view_results" / rel
                    if not preview.exists() or sha256(preview) != digest:
                        copy_verified(source, preview, digest)
                except OSError as exc:
                    print(f"[BACKUP WARN] Preview copy failed: {source.name}: {exc}", flush=True)
        print(f"[BACKUP] {utcnow()} files={len(entries)}, new_blobs={uploaded}, "
              f"uploaded={uploaded_bytes / 1024**2:.1f} MiB", flush=True)
        return manifest_path

    def valid_snapshots(self):
        # Prefer the latest pointer, then previously committed manifests.
        names = []
        pointer = self.drive / "latest.json"
        if pointer.exists():
            try:
                p = json.loads(pointer.read_text(encoding="utf-8"))
                names.append((self.drive / "snapshots" / p["manifest"], p["sha256"]))
            except (ValueError, KeyError):
                pass
        for path in sorted((self.drive / "snapshots").glob("snapshot_*.json"), reverse=True):
            if path not in [x[0] for x in names]:
                names.append((path, None))
        for path, expected in names:
            try:
                if not path.is_file():
                    continue
                if expected and sha256(path) != expected:
                    continue
                obj = json.loads(path.read_text(encoding="utf-8"))
                if obj.get("format") != BACKUP_VERSION:
                    continue
                if obj.get("file_count") != len(obj["entries"]):
                    continue
                unique = set()
                for entry in obj["entries"]:
                    p = Path(entry["path"])
                    if (p.is_absolute() or ".." in p.parts or p.parts[0] not in self.sources
                            or p.as_posix() in unique):
                        raise RuntimeError("Invalid snapshot path")
                    unique.add(p.as_posix())
                    blob = self.blob_path(entry["sha256"])
                    if not blob.is_file() or blob.stat().st_size != entry["bytes"]:
                        raise RuntimeError("Incomplete blob")
                yield path, obj
            except (OSError, ValueError, RuntimeError, KeyError):
                continue

    def restore(self):
        existing_results = self.sources.get("results")
        if existing_results and existing_results.exists() and any(existing_results.iterdir()):
            print("[RESTORE] Local experiment results exist; retaining the in-memory run.", flush=True)
            return False
        # A prior smoke run may have prepared a shared local dataset cache.
        # Restore the results even when the data cache is already populated.
        preexisting_cache = self.sources.get("data")
        keep_cache = bool(preexisting_cache and preexisting_cache.exists()
                          and any(preexisting_cache.iterdir()))
        for manifest_path, manifest in self.valid_snapshots():
            # Restore into staging dirs and promote only when EVERY sha256 matches.
            suffix = f".restore_{os.getpid()}"
            staging = {name: root.with_name(root.name + suffix) for name, root in self.sources.items()}
            for p in staging.values():
                shutil.rmtree(p, ignore_errors=True)
            try:
                for item in manifest["entries"]:
                    path = Path(item["path"])
                    name = path.parts[0]
                    if name == "data" and keep_cache:
                        continue  # existing cache can be reused; missing files download automatically
                    dest = staging[name].joinpath(*path.parts[1:])
                    blob = self.blob_path(item["sha256"])
                    copy_verified(blob, dest, item["sha256"])
                for name, stage in staging.items():
                    if name == "data" and keep_cache:
                        continue
                    root = self.sources[name]
                    if root.exists():
                        if any(root.iterdir()):
                            raise RuntimeError("Workspace unexpectedly populated during restore")
                        root.rmdir()
                    if stage.exists():
                        os.replace(stage, root)
                self._repair_metrics_from_committed_checkpoints()
                print(f"[RESTORE] Recovered {manifest['file_count']} files from {manifest_path.name}", flush=True)
                return True
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"[RESTORE] Skipping incomplete snapshot: {manifest_path.name} ({exc})", flush=True)
                for p in staging.values():
                    shutil.rmtree(p, ignore_errors=True)
        print("[RESTORE] No complete backup found; a new run will be started.", flush=True)
        return False

    def _repair_metrics_from_committed_checkpoints(self):
        """Checkpoint state is authoritative, even if an older metrics.json was copied."""
        try:
            import torch
        except ImportError:
            return
        root = self.sources.get("results")
        if root is None:
            return
        for latest in root.glob("runs/*/seed*/exp*/p*/latest.json"):
            pointer = json.loads(latest.read_text(encoding="utf-8"))
            cp = latest.parent / pointer["file"]
            if sha256(cp) != pointer["sha256"]:
                raise RuntimeError(f"Backup checkpoint mismatch: {cp}")
            # Loads ONLY checkpoint files created by our own experiment runner.
            state = torch.load(cp, map_location="cpu", weights_only=False)
            from experiment_io import write_json, write_csv
            write_json(latest.parent / "metrics.json", state["observations"])
            write_csv(latest.parent / "train_log.csv", state["train_log"],
                      fields=["phase", "epoch", "loss", "accuracy", "num_sources", "global_step"])


def run(args):
    if args.interval_minutes < 1:
        raise ValueError("interval-minutes must be >= 1")
    repo = Path(args.repo).resolve()
    script = repo / "binary_step_experiment_colab.py"
    if not script.exists():
        raise FileNotFoundError(f"Colab experiment script not present: {script}")
    out = Path(args.output).resolve()
    data = Path(args.data_root).resolve()
    drive = Path(args.drive_backup).resolve()
    store = BackupStore(drive, {"results": out, "data": data})
    store.restore()
    # Hard guard: the user can inspect this path in Drive even while the notebook is closed.
    atomic_write_json(drive / "runner_status.json", {"status": "starting", "time": utcnow()})
    extra = list(args.experiment_args)
    if extra and extra[0] == "--":
        extra.pop(0)
    if any(x in ("--output", "--data-root", "--publish") for x in extra):
        raise ValueError("Do not override --output/--data-root or use --publish in extra args")
    cmd = [sys.executable, "-u", str(script), "--output", str(out),
           "--data-root", str(data), *extra]
    print("[RUN]", " ".join(cmd), flush=True)
    logpath = out.parent / "colab_runner.log"  # outside experiment results manifest
    logpath.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    stop = threading.Event()
    lines = [0]
    def forward_output():
        with logpath.open("a", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                lines[0] += 1
                if (lines[0] <= 6 or lines[0] % 40 == 0 or "TEST" in line
                        or "Complete" in line or "Error" in line or "Traceback" in line):
                    print(line, end="", flush=True)
        stop.set()
    reader = threading.Thread(target=forward_output, daemon=True)
    reader.start()
    requested = [False]
    def interrupted(sig, frame):
        print(f"[RUN] Signal {sig}: requesting stop and final backup", flush=True)
        requested[0] = True
        proc.terminate()
        stop.set()
    old_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, interrupted)
    last_snapshot = None
    backup_failed = False
    try:
        next_due = time.monotonic() + args.interval_minutes * 60
        while proc.poll() is None and not requested[0]:
            until = max(0., next_due - time.monotonic())
            stop.wait(timeout=min(until, 5.))
            if time.monotonic() >= next_due:
                try:
                    last_snapshot = store.snapshot()
                    backup_failed = False
                    atomic_write_json(drive / "runner_status.json", {
                        "status": "running", "time": utcnow(),
                        "backup": last_snapshot.name, "log_lines": lines[0]})
                except Exception as exc:
                    backup_failed = True
                    print(f"[BACKUP ERROR] {type(exc).__name__}: {exc}", flush=True)
                next_due = time.monotonic() + args.interval_minutes * 60
        rc = proc.wait()
        reader.join(timeout=10)
        # Always back up at run end or failure, even if interval has not elapsed.
        try:
            last_snapshot = store.snapshot()
            backup_failed = False
        except Exception as exc:
            backup_failed = True
            print(f"[BACKUP ERROR] Final backup failed: {exc}", flush=True)
        if logpath.exists():
            try:
                atomic_write_bytes(drive / "last_runner.log", logpath.read_bytes())
            except OSError as exc:
                print(f"[BACKUP WARN] Could not sync log: {exc}", flush=True)
        atomic_write_json(drive / "runner_status.json", {
            "status": "complete" if rc == 0 and not backup_failed else "stopped_or_failed",
            "exit_code": rc, "backup_ok": not backup_failed,
            "time": utcnow(), "backup": last_snapshot.name if last_snapshot else None})
        return 0 if rc == 0 and not backup_failed else (rc if rc else 2)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--drive-backup", type=Path, required=True)
    parser.add_argument("--interval-minutes", type=float, default=10)
    parser.add_argument("experiment_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
