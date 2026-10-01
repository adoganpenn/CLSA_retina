"""Configuration must run without imports surviving a previous kernel state."""

import ast
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class MortalityNotebookConfigurationTests(unittest.TestCase):
    def execute_configuration_in_empty_namespace(self, source):
        tree = ast.parse(source)
        with tempfile.TemporaryDirectory() as directory:
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "volume_root"
                    for target in node.targets
                ):
                    node.value = ast.Constant(value=directory)
            ast.fix_missing_locations(tree)
            namespace = {}
            exec(compile(tree, "configuration-cell", "exec"), namespace)
            self.assertIs(namespace["Path"], Path)
            self.assertTrue(namespace["figure_root"].is_dir())
            self.assertTrue(namespace["table_root"].is_dir())

    def test_python_notebook_config_after_kernel_restart(self):
        source = (
            ROOT / "notebooks/04_mortality_prediction_retfound_epigenetic.py"
        ).read_text()
        cell = next(
            chunk
            for chunk in source.split("# COMMAND ----------")
            if "# Reproducible configuration." in chunk
        )
        self.execute_configuration_in_empty_namespace(cell)

    def test_ipython_notebook_config_after_kernel_restart(self):
        notebook = json.loads(
            (
                ROOT
                / "notebooks/04_mortality_prediction_retfound_epigenetic_updated.ipynb"
            ).read_text()
        )
        cell = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "# Reproducible configuration." in "".join(cell["source"])
        )
        self.execute_configuration_in_empty_namespace(cell)


if __name__ == "__main__":
    unittest.main()
