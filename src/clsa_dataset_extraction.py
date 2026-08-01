"""Safe, restartable extraction for encrypted CLSA fundus ZIP releases."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, Callable, Iterator
import zipfile

from clsa_dataset_inventory import (
    IMAGE_EXTENSIONS,
    file_extension,
    fundus_member_identity,
    zip_extra_field_ids,
)


def _safe_archive_destination(
    output_root: Path,
    member_name: str,
) -> Path:
    destination = (output_root / member_name).resolve()
    resolved_root = output_root.resolve()
    if resolved_root != destination and resolved_root not in destination.parents:
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return destination


def extract_fundus_zip_release(
    archive_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    password: str,
    *,
    release_name: str,
    visit: str,
    overwrite: bool = False,
    batch_size: int = 500,
    start_member_index: int = 0,
    max_members: int | None = None,
    progress_every: int = 250,
    progress_callback: Callable[[str], None] | None = print,
) -> list[dict[str, Any]]:
    """Extract a release and return its rows.

    This compatibility wrapper collects all rows in memory. Databricks jobs
    should consume :func:`iter_fundus_zip_release_batches` directly so each
    batch can be checkpointed before the next one is extracted.
    """
    rows: list[dict[str, Any]] = []
    for batch in iter_fundus_zip_release_batches(
        archive_path,
        output_root,
        password,
        release_name=release_name,
        visit=visit,
        overwrite=overwrite,
        batch_size=batch_size,
        start_member_index=start_member_index,
        max_members=max_members,
        progress_every=progress_every,
        progress_callback=progress_callback,
    ):
        rows.extend(batch)
    return rows


def _format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def iter_fundus_zip_release_batches(
    archive_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    password: str,
    *,
    release_name: str,
    visit: str,
    overwrite: bool = False,
    batch_size: int = 500,
    start_member_index: int = 0,
    max_members: int | None = None,
    progress_every: int = 250,
    progress_callback: Callable[[str], None] | None = print,
) -> Iterator[list[dict[str, Any]]]:
    """Yield bounded extraction-manifest batches with progress reporting.

    ``member_index`` is the stable zero-based index among image members in the
    ZIP central directory. Callers can persist the last completed index after
    every yielded batch and resume at the next index after a cluster restart.
    """
    if not password:
        raise ValueError("Archive password is empty.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if start_member_index < 0:
        raise ValueError("start_member_index cannot be negative.")
    if max_members is not None and max_members < 1:
        raise ValueError("max_members must be positive or None.")
    if progress_every < 1:
        raise ValueError("progress_every must be at least 1.")

    destination_root = Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    password_bytes = password.encode("utf-8")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            image_infos = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and file_extension(info.filename) in IMAGE_EXTENSIONS
            ]
            total_images = len(image_infos)
            if start_member_index > total_images:
                raise ValueError(
                    "start_member_index exceeds the number of image members "
                    f"({start_member_index} > {total_images})."
                )
            stop_member_index = total_images
            if max_members is not None:
                stop_member_index = min(
                    total_images,
                    start_member_index + max_members,
                )

            run_total = stop_member_index - start_member_index
            started = time.monotonic()
            status_counts = {
                "extracted": 0,
                "already_present": 0,
                "skipped_unsupported_aes": 0,
                "failed": 0,
            }
            batch: list[dict[str, Any]] = []

            if progress_callback:
                progress_callback(
                    f"[{release_name}] starting at image "
                    f"{start_member_index:,}; processing {run_total:,} of "
                    f"{total_images:,} images in batches of {batch_size:,}."
                )

            for member_index in range(
                start_member_index,
                stop_member_index,
            ):
                info = image_infos[member_index]
                participant_id, eye = fundus_member_identity(info.filename)
                destination = _safe_archive_destination(
                    destination_root,
                    info.filename,
                )
                status = "extracted"
                error = ""
                if 0x9901 in zip_extra_field_ids(info.extra):
                    status = "skipped_unsupported_aes"
                elif (
                    destination.exists()
                    and destination.stat().st_size == info.file_size
                    and not overwrite
                ):
                    status = "already_present"
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with archive.open(info, pwd=password_bytes) as source:
                            with destination.open("wb") as target:
                                while True:
                                    chunk = source.read(8 * 1024 * 1024)
                                    if not chunk:
                                        break
                                    target.write(chunk)
                    except Exception as exc:
                        status = "failed"
                        error = f"{type(exc).__name__}: {str(exc)}"[:500]
                        if destination.exists():
                            destination.unlink()
                status_counts[status] += 1
                batch.append(
                    {
                        "release_name": release_name,
                        "visit": visit,
                        "archive_path": str(archive_path),
                        "member_path": info.filename,
                        "output_path": (
                            str(destination)
                            if status not in {
                                "skipped_unsupported_aes",
                                "failed",
                            }
                            else None
                        ),
                        "participant_id": participant_id,
                        "eye": eye,
                        "uncompressed_bytes": int(info.file_size),
                        "crc32": f"{info.CRC:08x}",
                        "status": status,
                        "error": error,
                        "member_index": member_index,
                    }
                )

                processed = member_index - start_member_index + 1
                should_report = (
                    processed == run_total
                    or processed % progress_every == 0
                )
                if progress_callback and should_report:
                    elapsed = time.monotonic() - started
                    rate = processed / elapsed if elapsed else 0.0
                    progress_callback(
                        f"[{release_name}] {member_index + 1:,}/"
                        f"{total_images:,} ({(member_index + 1) / max(total_images, 1):.1%}) "
                        f"| this run {processed:,}/{run_total:,} "
                        f"| extracted={status_counts['extracted']:,}, "
                        f"existing={status_counts['already_present']:,}, "
                        f"failed={status_counts['failed']:,}, "
                        f"unsupported_aes={status_counts['skipped_unsupported_aes']:,} "
                        f"| {rate:.1f} images/s "
                        f"| elapsed={_format_elapsed(elapsed)}"
                    )

                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

            if progress_callback:
                state = (
                    "complete"
                    if stop_member_index == total_images
                    else "batch limit reached; rerun to resume"
                )
                progress_callback(
                    f"[{release_name}] {state}. Next member index: "
                    f"{stop_member_index:,}."
                )
    finally:
        password_bytes = b""
