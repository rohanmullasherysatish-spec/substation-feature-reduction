"""Use SHAP 0.53.0's interventional tree code with float64 split thresholds.

The two-line upstream correction is https://github.com/shap/shap/pull/5122.
Its MIT-licensed source is kept locally so no archived project or environment
modification is required. Compile once per process with the current Python/NumPy
headers; no source hashes, receipts or binary-cache identity framework is used.
"""

from contextlib import contextmanager
from functools import lru_cache
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import numpy as np

from src import settings
from src.checks import require


@lru_cache(maxsize=1)
def native_backend():
    import shap

    require(shap.__version__ == "0.53.0", "Review the local SHAP correction before changing SHAP version.")
    source = Path(__file__).parent / "vendor/shap_0_53_0"
    compiler = shutil.which("clang++") or shutil.which("c++")
    require(compiler is not None, "A C++ compiler is required for the local SHAP precision correction.")
    folder = settings.PROJECT_ROOT / ".cache/shap_precision"
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / ("_cext" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary = binary.with_name(binary.name + ".tmp")
    arguments = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC"]
    if sys.platform == "darwin":
        arguments += ["-undefined", "dynamic_lookup"]
    arguments += ["-I" + sysconfig.get_paths()["include"], "-I" + np.get_include(),
                  "-I" + str(source), str(source / "_cext.cc"), "-o", str(temporary)]
    result = subprocess.run(arguments, capture_output=True, text=True)
    require(result.returncode == 0, "SHAP precision compilation failed: " + result.stderr[-3000:])
    temporary.replace(binary)
    spec = importlib.util.spec_from_file_location("_local_shap_precision._cext", binary)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiler_version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    return module, {"compiler": compiler_version, "python_abi": sysconfig.get_config_var("SOABI"),
                    "numpy": np.__version__, "shap": shap.__version__,
                    "correction": "https://github.com/shap/shap/pull/5122"}


@contextmanager
def precision_backend():
    """Limit replacement to this explanation call; never modify installed SHAP files."""
    import shap.explainers._tree as tree

    original = tree._cext
    tree._cext, identity = native_backend()
    try:
        yield identity
    finally:
        tree._cext = original
