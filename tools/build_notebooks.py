#!/usr/bin/env python3
"""Build the student and solution notebooks from notebook_src/*.nbsrc.

    python tools/build_notebooks.py

EDIT THESE TWO LINES before pushing to GitHub, then re-run the script:
"""
GITHUB_REPO = "YOUR-ORG/diffusion-workshop"   # e.g. "isu-trac/diffusion-workshop"
BRANCH = "main"

# ----------------------------------------------------------------------------
# Source format (.nbsrc) - plain text, one notebook per file:
#
#   #%% md            starts a markdown cell
#   #%% code          starts a code cell
#   #%% setup         inserts the standard "run me first" cell
#   {{BADGE}}         (in markdown) becomes the "Open in Colab" badge
#
# Inside code cells:
#   ### BEGIN SOLUTION ... ### END SOLUTION   kept in solutions/, removed for students
#   ### STUDENT: <code>                       shown to students only (the blank to fill in)
# ----------------------------------------------------------------------------
import pathlib
import re
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "notebook_src"

# source file -> output folder for the student version
TARGETS = {
    "00_pytorch_refresher": "notebooks",
    "01_unet_denoising": "notebooks",
    "02_diffusion_ddpm": "notebooks",
    "03_optimizations": "notebooks",
    "04_classifier_free_guidance": "notebooks",
    "05_clip_text_to_image": "notebooks",
    "quiz": "assessment",
    "coding_assessment": "assessment",
}

SETUP = '''# --- Workshop setup: run this cell first ------------------------------------
import os, sys

REPO_URL = "https://github.com/{repo}.git"
if os.path.isdir("../diffusion_workshop"):            # running inside a local clone
    sys.path.insert(0, os.path.abspath(".."))
else:                                                 # running on Google Colab
    if not os.path.isdir("diffusion-workshop"):
        !git clone -q {{REPO_URL}} diffusion-workshop
    sys.path.insert(0, os.path.abspath("diffusion-workshop"))
    !pip -q install einops

import torch
import diffusion_workshop as dw
from diffusion_workshop import pick

device = dw.get_device()
dw.seed_everything(0)
print("device:", device, "| torch", torch.__version__)
if device.type != "cuda":
    print("No GPU found. On Colab: Runtime > Change runtime type > T4 GPU, then re-run this cell.")
'''


def badge(out_dir, name):
    url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{BRANCH}/{out_dir}/{name}.ipynb"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def split_cells(text):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        m = re.match(r"^#%%\s*(md|code|setup)\s*$", line)
        if m:
            if kind:
                cells.append((kind, buf))
            kind, buf = m.group(1), []
        elif kind:
            buf.append(line)
    if kind:
        cells.append((kind, buf))
    return cells


def render_code(lines, solution):
    out, in_sol = [], False
    for line in lines:
        s = line.strip()
        if s == "### BEGIN SOLUTION":
            in_sol = True
        elif s == "### END SOLUTION":
            in_sol = False
        elif s.startswith("### STUDENT:"):
            if not solution:
                indent = line[: len(line) - len(line.lstrip())]
                out.append(indent + s[len("### STUDENT:"):].lstrip(" ").rstrip())
        elif in_sol and not solution:
            continue
        else:
            out.append(line)
    return "\n".join(out).strip("\n")


def build(name, out_dir, solution):
    cells = []
    for kind, lines in split_cells((SRC / f"{name}.nbsrc").read_text()):
        if kind == "md":
            text = "\n".join(lines).strip("\n").replace("{{BADGE}}", badge(out_dir, name))
            cells.append(new_markdown_cell(text))
        elif kind == "setup":
            cells.append(new_code_cell(SETUP.format(repo=GITHUB_REPO)))
        else:
            cells.append(new_code_cell(render_code(lines, solution)))
    nb = new_notebook(cells=cells)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
    })
    for i, c in enumerate(nb.cells):          # stable ids -> clean git diffs
        c["id"] = f"cell-{i:03d}"
    folder = ROOT / ("solutions" if solution else out_dir)
    folder.mkdir(exist_ok=True)
    suffix = "_solution" if solution else ""
    path = folder / f"{name}{suffix}.ipynb"
    nbformat.write(nb, path)
    return path


def main():
    for name, out_dir in TARGETS.items():
        if name == "quiz":                     # the quiz has no separate solution notebook
            print("wrote", build(name, out_dir, solution=False).relative_to(ROOT))
            continue
        for solution in (False, True):
            print("wrote", build(name, out_dir, solution).relative_to(ROOT))
    if GITHUB_REPO.startswith("YOUR-ORG"):
        print("\nNOTE: set GITHUB_REPO at the top of tools/build_notebooks.py and rebuild,\n"
              "      otherwise the Colab badges and the `git clone` in the setup cell point nowhere.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
