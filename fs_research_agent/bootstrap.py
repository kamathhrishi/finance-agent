#!/usr/bin/env python3
"""
S3 corpus bootstrap.

The pattern:
    S3 holds a periodic snapshot of the local corpus as a single tarball
    (`fs_research_agent/corpus/latest.tar.gz`). On Railway, the API and
    the watcher share a persistent volume mounted at `FS_RESEARCH_DATA_ROOT`.
    On a fresh deploy (volume empty), we bootstrap by downloading the
    snapshot from S3 + extracting to the volume. Once warm, the watcher
    keeps the volume up-to-date by polling SEC EDGAR directly.

S3 keys (under the same Railway bucket the platform already uses):
    fs_research_agent/corpus/latest.tar.gz       — current snapshot
    fs_research_agent/corpus/latest.manifest.json — sha256 + counts + ts

Usage:
    # Dev → push a snapshot from your local corpus to S3
    python -m fs_research_agent.bootstrap upload

    # Railway → fetch the snapshot if local volume is empty
    python -m fs_research_agent.bootstrap download
    python -m fs_research_agent.bootstrap bootstrap-if-missing

    # Inspect current local + S3 state
    python -m fs_research_agent.bootstrap check
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("fs_research_agent.bootstrap")

# ──────────────────────────────────────────────────────────────────────────────
# Config — Railway bucket reuses the platform's existing S3 client config
# ──────────────────────────────────────────────────────────────────────────────

S3_KEY_TARBALL = "fs_research_agent/corpus/latest.tar.gz"
S3_KEY_MANIFEST = "fs_research_agent/corpus/latest.manifest.json"

# A volume that has fewer than this many ticker dirs is considered "empty"
# enough that a bootstrap is warranted. 5 picks up a totally-fresh deploy
# while not retriggering on a partially-deleted corpus.
DEFAULT_MIN_TICKERS = 5


# ──────────────────────────────────────────────────────────────────────────────
# S3 client
# ──────────────────────────────────────────────────────────────────────────────


def _make_s3_client():
    """Reuse the same Railway bucket the rest of the platform uses."""
    import boto3
    from botocore.config import Config

    endpoint = os.getenv("RAILWAY_BUCKET_ENDPOINT", "").strip()
    key = os.getenv("RAILWAY_BUCKET_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("RAILWAY_BUCKET_SECRET_KEY", "").strip()
    if not (endpoint and key and secret):
        raise RuntimeError(
            "Railway bucket credentials not set. Need RAILWAY_BUCKET_ENDPOINT, "
            "RAILWAY_BUCKET_ACCESS_KEY_ID, RAILWAY_BUCKET_SECRET_KEY."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _bucket_name() -> str:
    name = os.getenv("RAILWAY_BUCKET_NAME", "").strip()
    if not name:
        raise RuntimeError("RAILWAY_BUCKET_NAME env var is required")
    return name


# ──────────────────────────────────────────────────────────────────────────────
# Local corpus introspection
# ──────────────────────────────────────────────────────────────────────────────


def _data_root() -> Path:
    """Resolve the corpus root the rest of the package is using."""
    from .ingest import DEFAULT_DATA_ROOT
    env = os.getenv("FS_RESEARCH_DATA_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    return DEFAULT_DATA_ROOT


def _count_local(data_root: Path) -> tuple[int, int]:
    """Return (ticker_count, filing_count) on local disk."""
    filings_root = data_root / "filings"
    if not filings_root.is_dir():
        return 0, 0
    tickers = [p for p in filings_root.iterdir() if p.is_dir()]
    filings = sum(1 for _ in filings_root.rglob("metadata.json"))
    return len(tickers), filings


# ──────────────────────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Manifest:
    generated_at: str         # ISO 8601 UTC
    ticker_count: int
    filing_count: int
    tarball_bytes: int
    tarball_sha256: str
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Tar / untar
# ──────────────────────────────────────────────────────────────────────────────


def _tar_corpus(data_root: Path, dest: Path) -> None:
    """Create dest = gzipped tarball of <data_root>'s contents (filings/, README.md, INDEX.md)."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    # Members to include — keep it tight: just filings + the two top-level
    # markdown files. Skip any state files (_seen_accessions.json,
    # _watcher_state.json, _batch_checkpoint.json) — they're per-host state.
    skip_names = {
        "_seen_accessions.json",
        "_watcher_state.json",
        "_batch_checkpoint.json",
        "_last_s3_upload.json",  # per-host upload tracking; never tar in
    }

    def _filter(tarinfo: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        name = Path(tarinfo.name).name
        if name in skip_names:
            return None
        return tarinfo

    with tarfile.open(dest, "w:gz", compresslevel=6) as tar:
        # Include filings/ and any top-level non-state files
        for child in sorted(data_root.iterdir()):
            if child.name in skip_names:
                continue
            tar.add(child, arcname=child.name, filter=_filter)


def _untar_into(tarball: Path, data_root: Path) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        # `data` filter is python 3.12+; fall back if missing
        try:
            tar.extractall(path=data_root, filter="data")
        except TypeError:
            tar.extractall(path=data_root)


# ──────────────────────────────────────────────────────────────────────────────
# Public ops
# ──────────────────────────────────────────────────────────────────────────────


def upload_corpus(data_root: Optional[Path] = None) -> Manifest:
    """Tar+gzip the corpus, upload tarball + manifest to S3."""
    data_root = (data_root or _data_root()).resolve()
    s3 = _make_s3_client()
    bucket = _bucket_name()

    print(f"📁 Source corpus:  {data_root}")
    tickers, filings = _count_local(data_root)
    print(f"   {tickers} tickers / {filings:,} filings")
    if tickers == 0:
        print("⚠ corpus is empty — refusing to upload an empty snapshot")
        raise SystemExit(2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = Path(tmpdir) / "corpus.tar.gz"
        print(f"🗜  Tarballing → {tarball.name} …")
        t0 = time.time()
        _tar_corpus(data_root, tarball)
        size = tarball.stat().st_size
        sha = _sha256_of(tarball)
        print(f"   {size / 1e6:.1f} MB  sha256={sha[:12]}…  ({time.time() - t0:.0f}s)")

        manifest = Manifest(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ticker_count=tickers,
            filing_count=filings,
            tarball_bytes=size,
            tarball_sha256=sha,
        )

        print(f"☁ Uploading to s3://{bucket}/{S3_KEY_TARBALL} …")
        t0 = time.time()
        s3.upload_file(
            Filename=str(tarball),
            Bucket=bucket,
            Key=S3_KEY_TARBALL,
            ExtraArgs={"ContentType": "application/gzip"},
        )
        s3.put_object(
            Bucket=bucket,
            Key=S3_KEY_MANIFEST,
            Body=manifest.to_json().encode("utf-8"),
            ContentType="application/json",
        )
        print(f"✅ Upload done ({time.time() - t0:.0f}s)")
        print(f"   manifest:")
        for k, v in asdict(manifest).items():
            print(f"     {k}: {v}")
    return manifest


def download_and_extract_corpus(data_root: Optional[Path] = None, *, validate: bool = True) -> Manifest:
    """Fetch the latest snapshot from S3 + extract into data_root."""
    data_root = (data_root or _data_root()).resolve()
    s3 = _make_s3_client()
    bucket = _bucket_name()

    print(f"☁ Fetching manifest from s3://{bucket}/{S3_KEY_MANIFEST}")
    try:
        manifest_obj = s3.get_object(Bucket=bucket, Key=S3_KEY_MANIFEST)
    except Exception as e:
        raise RuntimeError(
            f"No manifest at s3://{bucket}/{S3_KEY_MANIFEST}. "
            f"Run `python -m fs_research_agent.bootstrap upload` from a host that has the corpus first. "
            f"({e})"
        )
    manifest_data = json.loads(manifest_obj["Body"].read())
    manifest = Manifest(**manifest_data)
    print(f"   snapshot: {manifest.ticker_count} tickers / {manifest.filing_count:,} filings, "
          f"{manifest.tarball_bytes / 1e6:.1f} MB, generated {manifest.generated_at}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tarball = Path(tmpdir) / "corpus.tar.gz"
        print(f"☁ Downloading tarball → {tarball.name} …")
        t0 = time.time()
        s3.download_file(Bucket=bucket, Key=S3_KEY_TARBALL, Filename=str(tarball))
        print(f"   {tarball.stat().st_size / 1e6:.1f} MB  ({time.time() - t0:.0f}s)")

        if validate:
            sha = _sha256_of(tarball)
            if sha != manifest.tarball_sha256:
                raise RuntimeError(
                    f"sha256 mismatch: expected {manifest.tarball_sha256[:12]}…, "
                    f"got {sha[:12]}…. Refusing to extract a corrupt snapshot."
                )
            print(f"   sha256 ok ({sha[:12]}…)")

        print(f"📦 Extracting → {data_root} …")
        t0 = time.time()
        _untar_into(tarball, data_root)
        tickers_after, filings_after = _count_local(data_root)
        print(f"✅ Extracted in {time.time() - t0:.0f}s — now {tickers_after} tickers / {filings_after:,} filings")
    return manifest


def bootstrap_if_missing(
    data_root: Optional[Path] = None,
    *,
    min_tickers: int = DEFAULT_MIN_TICKERS,
) -> bool:
    """If local corpus is empty/sparse, download from S3. Returns True if a download happened."""
    data_root = (data_root or _data_root()).resolve()
    tickers, filings = _count_local(data_root)
    if tickers >= min_tickers:
        logger.info(f"Local corpus has {tickers} tickers / {filings:,} filings — skipping bootstrap")
        return False
    logger.info(
        f"Local corpus has only {tickers} ticker(s) (< {min_tickers}); "
        f"bootstrapping from S3 into {data_root}"
    )
    download_and_extract_corpus(data_root)
    return True


def check(data_root: Optional[Path] = None) -> None:
    data_root = (data_root or _data_root()).resolve()
    tickers, filings = _count_local(data_root)
    print(f"📁 Local corpus: {data_root}")
    print(f"   {tickers} tickers / {filings:,} filings")
    print()
    print(f"☁ S3:")
    try:
        s3 = _make_s3_client()
        bucket = _bucket_name()
        head = s3.head_object(Bucket=bucket, Key=S3_KEY_TARBALL)
        print(f"   tarball:  s3://{bucket}/{S3_KEY_TARBALL}")
        print(f"             {head['ContentLength'] / 1e6:.1f} MB, last modified {head['LastModified']}")
        m = s3.get_object(Bucket=bucket, Key=S3_KEY_MANIFEST)
        manifest = json.loads(m["Body"].read())
        print(f"   manifest:")
        for k, v in manifest.items():
            print(f"     {k}: {v}")
    except Exception as e:
        print(f"   (no snapshot in S3 — {e})")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def main() -> int:
    # Load .env so RAILWAY_BUCKET_* are visible when running directly
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_up = sub.add_parser("upload", help="Tar+gzip the local corpus + push to S3")
    p_up.add_argument("--data-root", default=None)

    p_dn = sub.add_parser("download", help="Force-download + extract latest snapshot from S3")
    p_dn.add_argument("--data-root", default=None)
    p_dn.add_argument("--no-validate", action="store_true", help="Skip sha256 check (not recommended)")

    p_boot = sub.add_parser("bootstrap-if-missing", help="Download only if local corpus is empty/sparse")
    p_boot.add_argument("--data-root", default=None)
    p_boot.add_argument("--min-tickers", type=int, default=DEFAULT_MIN_TICKERS)

    p_chk = sub.add_parser("check", help="Show local + S3 corpus state")
    p_chk.add_argument("--data-root", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    for noisy in ("botocore", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    data_root = Path(args.data_root) if args.data_root else None

    if args.cmd == "upload":
        upload_corpus(data_root)
    elif args.cmd == "download":
        download_and_extract_corpus(data_root, validate=not args.no_validate)
    elif args.cmd == "bootstrap-if-missing":
        did = bootstrap_if_missing(data_root, min_tickers=args.min_tickers)
        print(f"bootstrap_if_missing: {'downloaded' if did else 'skipped (corpus already populated)'}")
    elif args.cmd == "check":
        check(data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
