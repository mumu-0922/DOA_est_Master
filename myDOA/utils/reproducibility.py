"""Helpers for saving reproducible experiment run metadata."""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _git_command(args: Iterable[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Return best-effort git state for the current repository."""
    cwd = Path(repo_root).resolve() if repo_root is not None else Path.cwd()
    try:
        root = Path(_git_command(["rev-parse", "--show-toplevel"], cwd=cwd))
        commit = _git_command(["rev-parse", "HEAD"], cwd=root)
        branch = _git_command(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
        status = _git_command(["status", "--porcelain"], cwd=root)
        return {
            "root": str(root),
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
            "status": status,
        }
    except Exception as exc:
        return {
            "root": str(cwd),
            "commit": None,
            "branch": None,
            "dirty": None,
            "status": None,
            "error": str(exc),
        }


def build_run_metadata(
    args: Any,
    run_type: str,
    argv: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    git_info: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a JSON-serializable metadata record for one experiment run."""
    argv_list = list(sys.argv if argv is None else argv)
    config = vars(args) if hasattr(args, "__dict__") else dict(args)

    return {
        "run_type": run_type,
        "created_at": timestamp or datetime.now().isoformat(timespec="seconds"),
        "command": subprocess.list2cmdline([str(part) for part in argv_list]),
        "argv": [str(part) for part in argv_list],
        "cwd": cwd or str(Path.cwd()),
        "config": _to_jsonable(config),
        "git": _to_jsonable(git_info if git_info is not None else get_git_info()),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "extra": _to_jsonable(extra or {}),
    }


def save_run_metadata(
    save_dir: str | Path,
    args: Any,
    run_type: str,
    argv: Optional[Iterable[str]] = None,
    cwd: Optional[str] = None,
    git_info: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    filename: str = "run_metadata.json",
) -> Path:
    """Write run metadata and a plain command file into an experiment directory."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    metadata = build_run_metadata(
        args=args,
        run_type=run_type,
        argv=argv,
        cwd=cwd,
        git_info=git_info,
        timestamp=timestamp,
        extra=extra,
    )

    metadata_path = save_path / filename
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (save_path / "command.txt").write_text(metadata["command"] + "\n", encoding="utf-8")
    return metadata_path
