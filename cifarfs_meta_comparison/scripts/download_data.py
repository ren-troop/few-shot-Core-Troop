#!/usr/bin/env python
"""Download and prepare CIFAR-FS as class folders.

CIFAR-FS is the CIFAR-100 few-shot benchmark introduced by Bertinetto et al.
(ICLR 2019).  The public mirror used here is the same data used by
``learn2learn``; the three parquet files already contain the 64/16/20 class
split, so no extra split file is needed.

    python scripts/download_data.py
    python scripts/download_data.py --data-root data --force
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

from tqdm import tqdm

HF = "https://huggingface.co/datasets/GATE-engine/cifarfs/resolve/main/data"
FILES = {
    "train": (
        f"{HF}/train-00000-of-00001-b2e0d12edcc41ca1.parquet",
        86_701_506,
    ),
    "val": (
        f"{HF}/validation-00000-of-00001-e04b38bd7284925d.parquet",
        21_595_537,
    ),
    "test": (
        f"{HF}/test-00000-of-00001-6e06f87d92fce065.parquet",
        26_664_899,
    ),
}


def _ssl_context(insecure: bool) -> ssl.SSLContext:
    if insecure:
        return ssl._create_unverified_context()
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download_file(
    url: str,
    destination: Path,
    expected_size: int,
    insecure: bool = False,
    retries: int = 3,
) -> Path:
    """Download with resume support and a final size check."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected_size:
        return destination

    part = destination.with_suffix(destination.suffix + ".part")
    context = _ssl_context(insecure)
    headers = {"User-Agent": "cifarfs-meta-comparison/1.0"}

    for attempt in range(1, retries + 1):
        existing = part.stat().st_size if part.exists() else 0
        if existing > expected_size:
            part.unlink()
            existing = 0
        if existing == expected_size:
            break

        request_headers = dict(headers)
        if existing:
            request_headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=request_headers)
        try:
            with urllib.request.urlopen(
                request, context=context, timeout=120
            ) as response:
                if existing and response.status != 206:
                    existing = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if existing else "wb"
                with part.open(mode) as handle, tqdm(
                    total=expected_size,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=destination.name,
                    dynamic_ncols=True,
                ) as progress:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        progress.update(len(chunk))
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {url}\n{exc}\n"
                    "If this is an SSL certificate error, install certifi "
                    "or retry with --insecure."
                ) from exc
            wait = 2 * attempt
            print(f"  attempt {attempt} failed ({exc}); retrying in {wait}s ...")
            time.sleep(wait)

    if part.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {part}: got {part.stat().st_size}, "
            f"expected {expected_size}"
        )
    os.replace(part, destination)
    return destination


def prepare(raw_dir: Path, data_root: Path, force: bool = False) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "CIFAR-FS preparation needs pyarrow. Install it with:\n"
            "  python -m pip install pyarrow"
        ) from exc

    summary: dict[str, dict[str, int]] = {}
    for split, (url, size) in FILES.items():
        filename = url.rsplit("/", 1)[-1]
        parquet_path = download_file(url, raw_dir / "cifarfs" / filename, size)
        out_split = data_root / "cifar-fs" / split
        if force and out_split.exists():
            import shutil

            shutil.rmtree(out_split)
        out_split.mkdir(parents=True, exist_ok=True)

        table = pq.ParquetFile(parquet_path)
        counters: dict[int, int] = {}
        total = 0
        batches = table.iter_batches(batch_size=512, columns=["image", "label"])
        for batch in tqdm(
            batches,
            total=(table.metadata.num_rows + 511) // 512,
            desc=f"  {split}",
            dynamic_ncols=True,
        ):
            images = batch.column("image").to_pylist()
            labels = batch.column("label").to_pylist()
            for image, label in zip(images, labels):
                label = int(label)
                if image is None or image.get("bytes") is None:
                    raise RuntimeError("Expected embedded image bytes in parquet")
                class_dir = out_split / f"class_{label:03d}"
                class_dir.mkdir(parents=True, exist_ok=True)
                index = counters.get(label, 0)
                counters[label] = index + 1
                (class_dir / f"{index:06d}.png").write_bytes(image["bytes"])
                total += 1
        summary[split] = {
            "classes": len(counters),
            "images": total,
        }
        print(f"  {split}: {len(counters)} classes, {total} images")

    manifest = data_root / "cifar-fs" / "manifest.json"
    manifest.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("== CIFAR-FS ==")
    summary = prepare(raw_dir, data_root, force=args.force)
    print("\n== Summary ==")
    for split in ("train", "val", "test"):
        info = summary[split]
        print(f"  {split:5s}: {info['classes']:3d} classes, {info['images']:6d} images")
    print(f"\nData root: {data_root}")
    print("Next: python scripts/train.py -c configs/cifarfs_maml.yaml")


if __name__ == "__main__":
    main()
