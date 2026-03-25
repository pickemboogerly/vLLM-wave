"""Discover locally cached Hugging Face Hub model snapshots (same logic as start_ai_cached_models.sh)."""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class CachedModel:
    label: str
    path: str


def hub_cache_root() -> str:
    if os.environ.get("HF_HUB_CACHE"):
        root = os.environ["HF_HUB_CACHE"]
    elif os.environ.get("HF_HOME"):
        root = os.path.join(os.environ["HF_HOME"], "hub")
    else:
        root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    return os.path.normpath(os.path.expanduser(root.rstrip("/")))


def _decode_models_folder(folder_name: str) -> str:
    if folder_name.startswith("models--"):
        return folder_name[len("models--") :].replace("--", "/")
    return folder_name


def discover_cached_models(cache_dir: str | None = None) -> list[CachedModel]:
    cache = os.path.expanduser(cache_dir or hub_cache_root())
    rows: list[dict] = []

    if os.path.isdir(cache):
        try:
            from huggingface_hub import scan_cache_dir

            info = scan_cache_dir(cache)
            for repo in info.repos:
                if repo.repo_type != "model":
                    continue
                for rev in repo.revisions:
                    p = str(rev.snapshot_path)
                    if os.path.isdir(p):
                        rows.append(
                            {
                                "repo_id": repo.repo_id,
                                "commit_hash": rev.commit_hash or "",
                                "path": p,
                                "last_modified_str": getattr(rev, "last_modified_str", "")
                                or "",
                            }
                        )
        except Exception:
            pass

    if not rows:
        try:
            proc = subprocess.run(
                [
                    "find",
                    cache,
                    "-type",
                    "f",
                    "-path",
                    "*/snapshots/*/config.json",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            seen_paths: set[str] = set()
            for line in proc.stdout.splitlines():
                cfg = line.strip()
                if not cfg:
                    continue
                snap = os.path.dirname(cfg)
                if snap in seen_paths or not os.path.isdir(snap):
                    continue
                seen_paths.add(snap)
                repo_folder = os.path.basename(
                    os.path.dirname(os.path.dirname(snap))
                )
                rid = (
                    _decode_models_folder(repo_folder)
                    if repo_folder.startswith("models--")
                    else repo_folder
                )
                rows.append(
                    {
                        "repo_id": rid,
                        "commit_hash": os.path.basename(snap),
                        "path": snap,
                        "last_modified_str": "",
                    }
                )
        except (subprocess.TimeoutExpired, OSError):
            pass

    seen: set[str] = set()
    uniq: list[dict] = []
    for r in rows:
        p = r["path"]
        if p in seen:
            continue
        seen.add(p)
        uniq.append(r)
    rows = uniq

    rows.sort(
        key=lambda r: (
            r.get("last_modified_str") or "",
            r.get("commit_hash") or "",
        ),
        reverse=True,
    )

    counts = Counter(r["repo_id"] for r in rows)
    out: list[CachedModel] = []
    for r in rows:
        rid = r["repo_id"]
        h = r.get("commit_hash") or ""
        short = h[:8] if h else ""
        if counts[rid] > 1 and short:
            label = f"{rid} ({short})"
        else:
            label = rid
        label = label.replace("\t", " ").replace("\n", " ")
        out.append(CachedModel(label=label, path=r["path"]))
    return out
