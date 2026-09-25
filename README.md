# Generative AI with Diffusion Models: a two-day hands-on workshop

Two 4-hour sessions that take an audience with **no generative-AI background** from "an image is a tensor" to a small **text-to-image** model. The topics follow the outline of NVIDIA's *Generative AI with Diffusion Models* course: U-Nets, diffusion, optimizations, classifier-free guidance and CLIP. All code here is written for this workshop and runs on a free Google Colab T4 GPU.

| Day | Block | Slides | Hands-on (Colab) |
|---|---|---|---|
| 1 | Generative AI primer | Intro | `00_pytorch_refresher` (warm-up / pre-work) |
| 1 | From U-Net to diffusion | Module 1 | `01_unet_denoising` |
| 1 | Diffusion models | Module 2 | `02_diffusion_ddpm` |
| 1 | Optimizations | Module 3 | `03_optimizations` |
| 2 | Classifier-free diffusion guidance | Module 4 | `04_classifier_free_guidance` |
| 2 | CLIP and text-to-image | Module 5 | `05_clip_text_to_image` |
| 2 | Assessment | Wrap-up | `assessment/quiz`, `assessment/coding_assessment` |

## Repository layout

```
diffusion_workshop/   helper package: data, plotting, reference U-Net, DDPM, CLIP wrapper
notebooks/            student notebooks (contain FIXME blanks and ✅ check cells)
solutions/            the same notebooks, filled in            <- instructors only
assessment/           quiz.ipynb, quiz.md (printable), coding_assessment.ipynb
instructor/           INSTRUCTOR_GUIDE.md, RUBRIC.md, quiz_answer_key.md   <- instructors only
notebook_src/         plain-text sources the notebooks are generated from
tools/                build_notebooks.py, make_quiz.py, smoke_test.py
tests/                unit tests for the helper package
```

## Instructor quick start

1. **Create the GitHub repository** and set its name at the top of `tools/build_notebooks.py`
   (`GITHUB_REPO = "your-org/diffusion-workshop"`).
2. **Rebuild the notebooks** so the Colab badges and the `git clone` line point at your repository:
   ```bash
   pip install -r requirements.txt
   python tools/make_quiz.py
   python tools/build_notebooks.py
   ```
3. **Check the install** (runs every solution notebook on tiny synthetic data, no downloads, about 2 minutes on CPU):
   ```bash
   pytest -q
   python tools/smoke_test.py
   ```
4. **Push**, then open a notebook through its "Open in Colab" badge and do one full dry run on a T4. See `instructor/INSTRUCTOR_GUIDE.md` for what to look for and for timing.

### Before you publish: keep the answers private

`solutions/`, `instructor/` and `tools/make_quiz.py` contain answers. The student notebooks clone this repository, so anything you push is visible to students. Either keep those three paths in a private instructor repository (or branch) until the workshop is over, or add them to `.gitignore` in the public copy. A ready-made list is in `.gitignore.public`.

## Editing the notebooks

Do not edit the `.ipynb` files directly: edit `notebook_src/<name>.nbsrc` and run `python tools/build_notebooks.py`. The format is plain text:

```
#%% md                      a markdown cell
#%% code                    a code cell
#%% setup                   the standard "run me first" cell
### BEGIN SOLUTION          lines kept only in solutions/
### END SOLUTION
### STUDENT: x = FIXME      line shown only to students
```

To change the quiz, edit the question bank in `tools/make_quiz.py`, then run it followed by the builder.

## Student quick start

Open a notebook with its Colab badge, choose *Runtime > Change runtime type > T4 GPU*, and run the first cell. Replace every `FIXME`, then run the ✅ check cell underneath. Nothing needs to be installed locally.

## Datasets and models

* FashionMNIST and MNIST (via `torchvision`), resized to 16×16 and 28×28.
* TensorFlow flower photos (3,670 images, about 220 MB), with CIFAR-10 as a fallback, for the CLIP lab.
* OpenAI CLIP ViT-B/32 through Hugging Face `transformers` (pre-installed on Colab; about 600 MB on first use).

## References

* Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
* Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
* Ho, Salimans. *Classifier-Free Diffusion Guidance.* 2022.
* Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
* Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
* Wu, He. *Group Normalization.* ECCV 2018. · Hendrycks, Gimpel. *Gaussian Error Linear Units.* 2016.
* NVIDIA Deep Learning Institute, *Generative AI with Diffusion Models* (topic outline).
