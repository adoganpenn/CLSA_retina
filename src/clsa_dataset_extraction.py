"""Safe, restartable extraction for encrypted CLSA fundus ZIP releases."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
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
) -> list[dict[str, Any]]:
    """Extract supported images and log unsupported AES members explicitly."""
    if not password:
        raise ValueError("Archive password is empty.")
    destination_root = Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    password_bytes = password.encode("utf-8")
    rows: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if (
                    info.is_dir()
                    or file_extension(info.filename) not in IMAGE_EXTENSIONS
                ):
                    continue
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
                rows.append(
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
                    }
                )
    finally:
        password_bytes = b""
    return rows
