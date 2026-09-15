# Replace only the last epigenetics notebook cell with this entire block.
# ruff: noqa: F821 - repo_root and figure_root already exist in the completed notebook
import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image

renderer_root = repo_root / "Age_Glaucoma" / "Algorithm Fairness"
if str(renderer_root) not in sys.path:
    sys.path.insert(0, str(renderer_root))

clsa_figure_renderer = importlib.import_module("clsa_figure_renderer")

clsa_figure_renderer = importlib.reload(clsa_figure_renderer)
# Works with either the original renderer or the newly updated renderer.
# The original save routine remains responsible for styles, metadata, DPI,
# LZW compression, and missing-glyph checks; only its target becomes local.
_encode_locally = getattr(
    clsa_figure_renderer, "_encode_local_file", clsa_figure_renderer._save
)


def _volume_safe_save(fig, path, profile, bbox=None):
    path = Path(path)
    preferred = Path("/local_disk0/tmp")
    local_root = (
        preferred if preferred.is_dir() and os.access(preferred, os.W_OK) else None
    )
    with tempfile.TemporaryDirectory(
        prefix="clsa-figure-export-", dir=local_root
    ) as staging:
        local_path = Path(staging) / path.name
        _encode_locally(fig, local_path, profile, bbox)
        if path.suffix.lower() in {".tif", ".tiff"}:
            with Image.open(local_path) as encoded:
                encoded.load()
                if encoded.tag_v2.get(259) != 5:
                    raise AssertionError(f"TIFF is not LZW compressed: {path}")
                dpi = encoded.info.get("dpi", ())
                if len(dpi) != 2 or any(
                    abs(float(value) - profile.dpi) > 0.1 for value in dpi
                ):
                    raise AssertionError(
                        f"TIFF DPI does not match {profile.dpi}: {path}"
                    )
        path.parent.mkdir(parents=True, exist_ok=True)
        with local_path.open("rb") as source, path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        if path.stat().st_size != local_path.stat().st_size:
            raise OSError(f"Published figure size mismatch: {path}")


clsa_figure_renderer._save = _volume_safe_save
renderer_acceptance = clsa_figure_renderer.render_all(figure_root)
display(renderer_acceptance)
