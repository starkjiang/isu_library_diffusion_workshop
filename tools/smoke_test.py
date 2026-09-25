#!/usr/bin/env python3
"""Execute every SOLUTION notebook end-to-end on tiny synthetic data (no downloads, CPU is fine).

    python tools/smoke_test.py            # all notebooks, about two minutes on a laptop
    python tools/smoke_test.py 02 04      # only notebooks whose name contains 02 or 04

It checks that the code runs and that every ✅ check passes. It does not check image quality:
for that, run the real notebooks once on a Colab T4 before the workshop.
"""
import os
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ["WORKSHOP_SMOKE"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

targets = sorted((ROOT / "solutions").glob("*_solution.ipynb")) + [ROOT / "assessment" / "quiz.ipynb"]
wanted = sys.argv[1:]
failed = []
for path in targets:
    if wanted and not any(w in path.name for w in wanted):
        continue
    t0 = time.time()
    nb = nbformat.read(path, as_version=4)
    try:
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(path.parent)}}).execute()
        print(f"PASS  {path.name:45s} {time.time() - t0:5.0f}s")
    except Exception as err:  # noqa: BLE001
        failed.append(path.name)
        print(f"FAIL  {path.name}\n{str(err)[-1500:]}")
sys.exit(1 if failed else 0)
