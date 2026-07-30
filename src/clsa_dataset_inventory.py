"""Metadata-only inventory and README generation for the CLSA release Volume.

ZIP functions read central directories and CSV headers only. They do not
extract participant rows, image pixels, passwords, or genotype payloads.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence
import zipfile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dcm"}
TABULAR_EXTENSIONS = {".csv", ".csv.gz", ".tsv", ".txt", ".txt.gz", ".xlsx", ".xls"}
COMPOUND_EXTENSIONS = (".bgen.bgi", ".csv.gz", ".txt.gz")


def file_extension(name: str) -> str:
    """Return a normalized extension, preserving known compound extensions."""
    lower_name = name.casefold()
    for extension in COMPOUND_EXTENSIONS:
        if lower_name.endswith(extension):
            return extension
    return PurePosixPath(lower_name).suffix


def zip_extra_field_ids(extra: bytes) -> tuple[int, ...]:
    """Parse ZIP extra-field IDs without interpreting their payloads."""
    field_ids: list[int] = []
    cursor = 0
    while cursor + 4 <= len(extra):
        field_id = int.from_bytes(extra[cursor : cursor + 2], "little")
        size = int.from_bytes(extra[cursor + 2 : cursor + 4], "little")
        cursor += 4
        if cursor + size > len(extra):
            break
        field_ids.append(field_id)
        cursor += size
    return tuple(field_ids)


def classify_archive_member(member_path: str) -> str:
    """Classify an archive member from its filename metadata."""
    extension = file_extension(member_path)
    lower_name = PurePosixPath(member_path).name.casefold()
    if extension in IMAGE_EXTENSIONS:
        return "fundus_image"
    if extension in {".xlsx", ".xls"} and "dictionar" in lower_name:
        return "data_dictionary"
    if extension in TABULAR_EXTENSIONS:
        return "tabular_data"
    if extension in {".bgen", ".bgen.bgi", ".bed", ".bim", ".fam", ".sample"}:
        return "genetics"
    if extension in {".md", ".pdf", ".doc", ".docx"}:
        return "documentation"
    if extension in {".zip", ".7z"}:
        return "nested_archive"
    return "other"


def fundus_member_identity(member_path: str) -> tuple[str | None, str | None]:
    """Return participant ID and normalized eye for the observed CLSA layout."""
    match = re.search(
        r"(?:^|/)(?P<participant_id>\d{7})/"
        r"retinal_(?P<eye>left|right)\.jpe?g$",
        member_path,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, None
    eye = "L" if match.group("eye").casefold() == "left" else "R"
    return match.group("participant_id"), eye


def inspect_zip_archive(
    archive_path: str | os.PathLike[str],
    *,
    release_name: str | None = None,
    role: str | None = None,
    visit: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inventory a ZIP central directory without extracting its members."""
    archive_path = str(archive_path)
    release_name = release_name or Path(archive_path).stem
    extension_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    participant_eyes: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            extension = file_extension(info.filename)
            member_class = classify_archive_member(info.filename)
            participant_id, eye = fundus_member_identity(info.filename)
            if participant_id and eye:
                participant_eyes.setdefault(participant_id, set()).add(eye)
            extension_key = extension or "<none>"
            extension_counts[extension_key] = extension_counts.get(extension_key, 0) + 1
            class_counts[member_class] = class_counts.get(member_class, 0) + 1
            rows.append(
                {
                    "release_name": release_name,
                    "role": role or "",
                    "visit": visit or "",
                    "archive_path": archive_path,
                    "member_path": info.filename,
                    "member_name": PurePosixPath(info.filename).name,
                    "extension": extension,
                    "member_class": member_class,
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                    "crc32": f"{info.CRC:08x}",
                    "encrypted": bool(info.flag_bits & 0x1),
                    "possible_aes": 0x9901 in zip_extra_field_ids(info.extra),
                    "participant_id": participant_id,
                    "eye": eye,
                }
            )

    total_uncompressed = sum(row["uncompressed_bytes"] for row in rows)
    total_compressed = sum(row["compressed_bytes"] for row in rows)
    summary = {
        "release_name": release_name,
        "role": role or "",
        "visit": visit or "",
        "archive_path": archive_path,
        "archive_bytes": int(Path(archive_path).stat().st_size),
        "file_count": len(rows),
        "uncompressed_bytes": int(total_uncompressed),
        "compressed_member_bytes": int(total_compressed),
        "compression_ratio": (
            float(total_uncompressed / total_compressed) if total_compressed else None
        ),
        "image_count": int(class_counts.get("fundus_image", 0)),
        "tabular_count": int(class_counts.get("tabular_data", 0)),
        "dictionary_count": int(class_counts.get("data_dictionary", 0)),
        "encrypted_count": int(sum(row["encrypted"] for row in rows)),
        "possible_aes_count": int(sum(row["possible_aes"] for row in rows)),
        "participant_count": len(participant_eyes),
        "complete_eye_pair_count": sum(
            {"L", "R"}.issubset(eyes) for eyes in participant_eyes.values()
        ),
        "left_eye_count": int(sum(row["eye"] == "L" for row in rows)),
        "right_eye_count": int(sum(row["eye"] == "R" for row in rows)),
        "extension_counts_json": json.dumps(extension_counts, sort_keys=True),
        "class_counts_json": json.dumps(class_counts, sort_keys=True),
    }
    return rows, summary


def inspect_zip_releases(
    releases: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inspect all configured releases."""
    members: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for release in releases:
        release_members, summary = inspect_zip_archive(
            release["path"],
            release_name=str(release["name"]),
            role=str(release.get("role") or ""),
            visit=str(release.get("visit") or ""),
        )
        members.extend(release_members)
        summaries.append(summary)
    return members, summaries


def inspect_zip_csv_headers(
    releases: Sequence[Mapping[str, Any]],
    *,
    sample_bytes: int = 256 * 1024,
) -> list[dict[str, Any]]:
    """Read questionnaire CSV headers from ZIP streams without participant rows."""
    rows: list[dict[str, Any]] = []
    for release in releases:
        if str(release.get("role") or "") != "questionnaire":
            continue
        with zipfile.ZipFile(release["path"]) as archive:
            for info in archive.infolist():
                if info.is_dir() or file_extension(info.filename) != ".csv":
                    continue
                columns: list[str] = []
                delimiter = ""
                error = ""
                try:
                    with archive.open(info) as stream:
                        sample = stream.read(sample_bytes)
                    text = sample.decode("utf-8-sig", errors="replace")
                    try:
                        delimiter = csv.Sniffer().sniff(
                            text, delimiters=",\t;|"
                        ).delimiter
                    except csv.Error:
                        delimiter = ","
                    columns = [
                        column.strip()
                        for column in next(
                            csv.reader(io.StringIO(text), delimiter=delimiter)
                        )
                    ]
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc)}"[:500]
                rows.append(
                    {
                        "release_name": str(release["name"]),
                        "visit": str(release.get("visit") or ""),
                        "archive_path": str(release["path"]),
                        "member_path": info.filename,
                        "delimiter": delimiter,
                        "column_count": len(columns),
                        "columns_json": json.dumps(columns),
                        "inspection_error": error,
                    }
                )
    return rows


def profile_dictionary_workbook(
    workbook_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Profile workbook sheets and headers without exposing row values."""
    import openpyxl

    workbook = openpyxl.load_workbook(
        workbook_path, read_only=True, data_only=True
    )
    sheets: list[dict[str, Any]] = []
    try:
        for sheet in workbook.worksheets:
            iterator = sheet.iter_rows(values_only=True)
            first_row = next(iterator, ())
            headers = [str(value) if value is not None else "" for value in first_row]
            table_index = next(
                (
                    index
                    for index, value in enumerate(headers)
                    if value.casefold() == "table"
                ),
                None,
            )
            table_names: set[str] = set()
            data_rows = 0
            for row in iterator:
                if not any(value is not None for value in row):
                    continue
                data_rows += 1
                if (
                    table_index is not None
                    and table_index < len(row)
                    and row[table_index] is not None
                ):
                    table_names.add(str(row[table_index]))
            sheets.append(
                {
                    "sheet_name": sheet.title,
                    "data_rows": data_rows,
                    "column_count": len(headers),
                    "headers": headers,
                    "table_count": len(table_names),
                    "tables": sorted(table_names),
                }
            )
    finally:
        workbook.close()
    return {
        "path": str(workbook_path),
        "sheet_count": len(sheets),
        "sheets": sheets,
    }


def format_bytes(value: int | float | None) -> str:
    """Format bytes for Markdown tables."""
    if value is None:
        return "—"
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(amount) < 1024 or unit == units[-1]:
            return f"{amount:,.1f} {unit}"
        amount /= 1024
    return f"{amount:,.1f} TiB"


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def build_dataset_readme(
    *,
    volume_root: str,
    output_root: str,
    file_inventory: Sequence[Mapping[str, Any]],
    releases: Sequence[Mapping[str, Any]],
    zip_members: Sequence[Mapping[str, Any]],
    zip_summaries: Sequence[Mapping[str, Any]],
    csv_schemas: Sequence[Mapping[str, Any]],
    dictionary_profile: Mapping[str, Any],
    genetics_readiness: Mapping[str, Any],
    genomics_readme_text: str = "",
    generated_utc: str | None = None,
) -> str:
    """Generate a comprehensive metadata-only README."""
    generated_utc = generated_utc or datetime.now(timezone.utc).isoformat()
    names = {str(row.get("name")) for row in file_inventory}
    summary_by_release = {
        str(row["release_name"]): row for row in zip_summaries
    }
    total_source_bytes = sum(int(row.get("bytes") or 0) for row in file_inventory)
    imaging = [
        row for row in zip_summaries if row.get("role") == "fundus_imaging"
    ]
    questionnaire_members = [
        row
        for row in zip_members
        if row.get("role") == "questionnaire"
        and row.get("member_class")
        in {"tabular_data", "data_dictionary", "documentation"}
    ]
    genomics_rows = sorted(
        (
            row
            for row in file_inventory
            if str(row.get("relative_path") or "").startswith("Genomics3_clsa/")
        ),
        key=lambda row: str(row.get("relative_path") or ""),
    )

    lines = [
        "# CLSA retinal-aging dataset inventory",
        "",
        f"Generated: `{generated_utc}`",
        "",
        "This README is generated from file metadata, ZIP central directories, "
        "CSV headers, the Follow-up 2 dictionary workbook, and the genetics "
        "staging README. It does not extract participant rows, image pixels, "
        "passwords, API tokens, or genotype payloads.",
        "",
        "## Governed locations",
        "",
        f"- Raw/source root: `{volume_root}`",
        f"- Derived/output root: `{output_root}`",
        "- Restricted inputs and derived participant-level data must never be committed to Git.",
        "",
        "## At a glance",
        "",
        f"- Inventoried source files: **{len(file_inventory):,}**",
        f"- Inventoried source size: **{format_bytes(total_source_bytes)}**",
        f"- ZIP releases: **{len(zip_summaries):,}**",
        f"- Fundus images visible inside ZIPs: "
        f"**{sum(int(row['image_count']) for row in imaging):,}**",
        f"- Imaging participants across release-specific counts: "
        f"**{sum(int(row['participant_count']) for row in imaging):,}**",
        "",
        "Participant counts are release-specific and must not be summed to "
        "estimate unique longitudinal participants.",
        "",
        "## Configured release catalog",
        "",
        "| Release | Role | Visit | Source path | Present | Size | Members |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for release in releases:
        name = str(release["name"])
        summary = summary_by_release.get(name)
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(name),
                    _cell(release.get("role")),
                    _cell(release.get("visit")),
                    f"`{release['path']}`",
                    "yes" if Path(str(release["path"])).exists() else "**NO**",
                    format_bytes(summary.get("archive_bytes") if summary else None),
                    f"{int(summary['file_count']):,}" if summary else "—",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## ZIP archive inventory",
            "",
            "| Release | Role | Visit | Files | Uncompressed | Images | Tables | "
            "Encrypted | Possible AES | Participants | Complete eye pairs |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in zip_summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(summary["release_name"]),
                    _cell(summary.get("role")),
                    _cell(summary.get("visit")),
                    f"{int(summary['file_count']):,}",
                    format_bytes(summary["uncompressed_bytes"]),
                    f"{int(summary['image_count']):,}",
                    f"{int(summary['tabular_count']) + int(summary['dictionary_count']):,}",
                    f"{int(summary['encrypted_count']):,}",
                    f"{int(summary['possible_aes_count']):,}",
                    f"{int(summary['participant_count']):,}",
                    f"{int(summary['complete_eye_pair_count']):,}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### Fundus image releases",
            "",
            "Observed layout: "
            "`<release>/<participant_id>/retinal_<left|right>.jpeg`. "
            "Participant IDs remain only in the governed member-inventory Delta "
            "table and are intentionally omitted from this README.",
            "",
            "| Release | Visit | Images | Left | Right | Participants | Complete pairs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in imaging:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(summary["release_name"]),
                    _cell(summary.get("visit")),
                    f"{int(summary['image_count']):,}",
                    f"{int(summary['left_eye_count']):,}",
                    f"{int(summary['right_eye_count']):,}",
                    f"{int(summary['participant_count']):,}",
                    f"{int(summary['complete_eye_pair_count']):,}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "ZIP inspection does not need the archive password. Extraction must "
            "use an approved secret or temporary widget and preserve BL/F1 visit.",
            "",
            "### Questionnaire archive members",
            "",
            "| Release | Visit | Member | Type | Uncompressed |",
            "|---|---|---|---|---:|",
        ]
    )
    for member in questionnaire_members:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(member["release_name"]),
                    _cell(member.get("visit")),
                    f"`{_cell(member['member_path'])}`",
                    _cell(member["member_class"]),
                    format_bytes(member["uncompressed_bytes"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "### Questionnaire CSV schemas",
            "",
            "Only headers are inspected; participant values are not copied here.",
            "",
        ]
    )
    if csv_schemas:
        for schema in csv_schemas:
            columns = json.loads(schema["columns_json"])
            lines.extend(
                [
                    f"<details><summary>{_cell(schema['release_name'])}: "
                    f"{_cell(schema['member_path'])} "
                    f"({int(schema['column_count']):,} columns)</summary>",
                    "",
                    (
                        "`" + "`, `".join(_cell(column) for column in columns) + "`"
                        if columns
                        else f"Inspection error: `{schema.get('inspection_error')}`"
                    ),
                    "",
                    "</details>",
                    "",
                ]
            )
    else:
        lines.extend(["No CSV headers were available.", ""])

    lines.extend(
        [
            "## Follow-up 2 dictionary workbook",
            "",
            f"- Path: `{dictionary_profile['path']}`",
            f"- Worksheets: **{int(dictionary_profile['sheet_count']):,}**",
            "",
            "| Worksheet | Data rows | Columns | Referenced tables | Headers |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for sheet in dictionary_profile["sheets"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(sheet["sheet_name"]),
                    f"{int(sheet['data_rows']):,}",
                    f"{int(sheet['column_count']):,}",
                    f"{int(sheet['table_count']):,}",
                    ", ".join(f"`{_cell(item)}`" for item in sheet["headers"]),
                ]
            )
            + " |"
        )

    direct = genetics_readiness["direct_genotypes"]
    lines.extend(
        [
            "",
            "## Genetics release readiness",
            "",
            f"- PLINK BED present: **{bool(direct['bed'])}**",
            f"- PLINK BIM present: **{bool(direct['bim'])}**",
            f"- PLINK FAM present: **{bool(direct['fam'])}**",
            f"- Direct genotype dataset ready: **{bool(genetics_readiness['direct_ready'])}**",
            f"- Missing BGEN chromosomes: `{genetics_readiness['missing_bgen_chromosomes']}`",
            f"- Missing BGI chromosomes: `{genetics_readiness['missing_bgi_chromosomes']}`",
            f"- Imputed genotype dataset ready: **{bool(genetics_readiness['imputed_ready'])}**",
            f"- Missing required metadata: `{genetics_readiness['missing_metadata']}`",
            "",
            "A `.bgen.bgi` index is not a genotype dataset without its matching "
            "`.bgen`. BIM and FAM are also unusable without BED.",
            "",
            "### BGEN/BGI chromosome pairing",
            "",
            "| Chromosome | BGEN | BGI | Pair ready |",
            "|---:|---:|---:|---:|",
        ]
    )
    for chromosome in range(1, 24):
        bgen = f"clsa_imp_{chromosome}_v3.bgen" in names
        bgi = f"clsa_imp_{chromosome}_v3.bgen.bgi" in names
        lines.append(
            f"| {chromosome} | {'yes' if bgen else 'no'} | "
            f"{'yes' if bgi else 'no'} | {'yes' if bgen and bgi else 'no'} |"
        )

    lines.extend(
        [
            "",
            "### Current genomics files in the Volume",
            "",
            "| Relative path | Type | Size |",
            "|---|---|---:|",
        ]
    )
    for row in genomics_rows:
        lines.append(
            f"| `{_cell(row.get('relative_path'))}` | "
            f"{_cell(row.get('extension'))} | {format_bytes(row.get('bytes'))} |"
        )

    lines.extend(
        [
            "",
            "## Generated machine-readable outputs",
            "",
            f"- `{output_root}/file_inventory`",
            f"- `{output_root}/release_catalog`",
            f"- `{output_root}/zip_archive_summary`",
            f"- `{output_root}/zip_member_inventory`",
            f"- `{output_root}/questionnaire_csv_schemas`",
            f"- `{output_root}/dictionary_variables`",
            f"- `{output_root}/dictionary_categories`",
            f"- `{output_root}/dictionary_missing_codes`",
            f"- `{output_root}/genetics_sample_qc` when available",
            f"- `{output_root}/genetics_hla` when available",
            "",
            "## Recommended execution order",
            "",
            "1. Generate this metadata inventory with extraction disabled.",
            "2. Review release counts and questionnaire member names.",
            "3. Extract questionnaire releases into release-specific directories.",
            "4. Extract BL and F1 fundus archives separately, preserving visit.",
            "5. Build visit-aware image and questionnaire manifests.",
            "6. Run technical image QC and RETFound embeddings.",
            "7. Link genetics only through the authorized participant-to-"
            "`ADM_GWAS_COM` crosswalk.",
            "8. Record code commit, configuration, checkpoint hash, and table "
            "versions for each analysis freeze.",
            "",
            "## Governance and interpretation cautions",
            "",
            "- Passwords and API tokens must never appear in README, Delta, output, or Git.",
            "- ZIP member inventories are restricted because paths can contain participant IDs.",
            "- BL and F1 images are different visits; do not attach one age record to both.",
            "- Technical image quality is not clinical gradability.",
            "- Direct genotypes are GRCh37; imputed data are GRCh38.",
            "- BGEN indexes cannot reconstruct excluded genotype payloads.",
        ]
    )

    if genomics_readme_text.strip():
        lines.extend(
            [
                "",
                "## Appendix: genomics staging README",
                "",
                "The following source note is included for provenance:",
                "",
            ]
        )
        lines.extend(
            f"> {line}" if line else ">"
            for line in genomics_readme_text.strip().splitlines()
        )
    return "\n".join(lines).rstrip() + "\n"
