"""Verify local encoding and forward-only publication to mounted volumes."""

import hashlib
import importlib
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RENDERER_ROOT = (
    ROOT
    if (ROOT / "clsa_figure_renderer.py").exists()
    else ROOT / "Age_Glaucoma/Algorithm Fairness"
)
sys.path.insert(0, str(RENDERER_ROOT))
renderer = importlib.import_module("clsa_figure_renderer")


class ForwardOnlyWriter:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        return self.stream.write(data)

    def seek(self, *args):
        raise OSError("Mounted volume does not support random writes")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


class FigureVolumeExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.style = mpl.rc_context(renderer._style(renderer.PROFILES["publication"]))
        self.style.__enter__()
        self.addCleanup(self.style.__exit__, None, None, None)
        self.fig, self.ax = plt.subplots(figsize=(1.5, 1.1), layout="constrained")
        self.ax.plot([0, 1], [0, 1])
        self.ax.set_xlabel("Age")
        self.addCleanup(plt.close, self.fig)

    def test_lzw_tiff_600_dpi_on_forward_only_destination(self):
        destination = self.root / "volume/publication/figure_01.tif"
        path_open = Path.open

        def forward_only(path, mode="r", *args, **kwargs):
            stream = path_open(path, mode, *args, **kwargs)
            return (
                ForwardOnlyWriter(stream)
                if path == destination and mode == "wb"
                else stream
            )

        with (
            patch.object(Path, "open", forward_only),
            patch.object(
                renderer, "_encode_local_file", wraps=renderer._encode_local_file
            ) as encoder,
        ):
            renderer._save(self.fig, destination, renderer.PROFILES["publication"])
        encoded_path = encoder.call_args.args[1]
        self.assertNotEqual(encoded_path, destination)
        self.assertEqual(encoded_path.name, destination.name)
        self.assertFalse(encoded_path.exists())
        with Image.open(destination) as image:
            image.load()
            self.assertEqual(image.tag_v2[259], 5)
            self.assertEqual(
                tuple(round(value) for value in image.info["dpi"]), (600, 600)
            )

    def test_encode_failure_preserves_existing_destination(self):
        destination = self.root / "figure_01.tif"
        destination.write_bytes(b"previous-valid-output")
        with patch.object(
            renderer, "_encode_local_file", side_effect=OSError("encoder failed")
        ):
            with self.assertRaisesRegex(OSError, "encoder failed"):
                renderer._save(self.fig, destination, renderer.PROFILES["publication"])
        self.assertEqual(destination.read_bytes(), b"previous-valid-output")

    def test_png_bytes_are_unchanged_and_rerender_is_deterministic(self):
        profile = renderer.PROFILES["slide"]
        direct = self.root / "direct/figure_01.png"
        published = self.root / "volume/figure_01.png"
        self.fig.canvas.draw()
        self.fig.set_layout_engine("none")
        renderer._encode_local_file(self.fig, direct, profile)
        renderer._save(self.fig, published, profile)
        first_hash = hashlib.sha256(published.read_bytes()).hexdigest()
        self.assertEqual(direct.read_bytes(), published.read_bytes())
        renderer._save(self.fig, published, profile)
        self.assertEqual(first_hash, hashlib.sha256(published.read_bytes()).hexdigest())

    def test_vector_pdf_remains_selectable(self):
        from pypdf import PdfReader

        destination = self.root / "volume/figure_01.pdf"
        renderer._save(self.fig, destination, renderer.PROFILES["publication"])
        self.assertIn("Age", PdfReader(destination).pages[0].extract_text())

    def _exercise_last_cell(self, updated_module):
        destination = self.root / "volume/figure_01.tif"
        module = SimpleNamespace(_save=renderer._encode_local_file)
        if updated_module:
            module._encode_local_file = renderer._encode_local_file
        module.render_all = lambda outdir: module._save(
            self.fig, destination, renderer.PROFILES["publication"]
        )
        source = (RENDERER_ROOT / "epigenetics_last_cell_volume_safe.py").read_text()
        namespace = {
            "repo_root": self.root,
            "figure_root": destination.parent,
            "display": lambda result: None,
        }
        with (
            patch.object(importlib, "import_module", return_value=module),
            patch.object(importlib, "reload", return_value=module),
            patch.object(sys, "path", []),
        ):
            exec(compile(source, "last-cell", "exec"), namespace)
        with Image.open(destination) as image:
            image.load()
            self.assertEqual(image.tag_v2[259], 5)

    def test_replacement_last_cell_works_with_original_renderer(self):
        self._exercise_last_cell(updated_module=False)

    def test_replacement_last_cell_works_with_updated_renderer(self):
        self._exercise_last_cell(updated_module=True)


if __name__ == "__main__":
    unittest.main()
