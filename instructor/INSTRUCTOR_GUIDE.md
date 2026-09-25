# Instructor guide

## Before the workshop (do these once, about two hours)

1. Set `GITHUB_REPO` in `tools/build_notebooks.py`, rebuild, run `pytest -q` and `python tools/smoke_test.py`, push.
2. **Dry-run every solution notebook on a Colab T4.** The smoke test only proves the code runs; it says nothing about speed or image quality. Note the wall-clock time of each training cell and adjust `EPOCHS` so that no cell runs longer than about 6 minutes in the room. The defaults (3 / 5 / 5 / 6 epochs for Labs 1–4, 80 for Lab 5, 5 for the assessment) are starting points that have **not** been timed on a T4.
3. **Lab 5 checkpoint (strongly recommended).** Run `05_clip_text_to_image_solution.ipynb` to the end, download `clip_unet.pt`, attach it to a GitHub Release, and set `CHECKPOINT = "<release URL>"` in `notebook_src/05_clip_text_to_image.nbsrc`. Students then load the weights in seconds and spend the lab on prompts. Keep `CHECKPOINT = None` only if you have the time budget you measured in step 2.
4. **Assessment threshold.** In your dry run of `coding_assessment_solution.ipynb`, confirm the generated digits score ≥ 90% with `W = 2.0`. If they fall short, raise `EPOCHS` (or lower the threshold in the notebook source and in `RUBRIC.md`).
5. Check that the room's network can reach `github.com`, `colab.research.google.com`, `huggingface.co` and `storage.googleapis.com`. If the flower download is blocked, switch Lab 5 to `DATASET = "cifar10"`.
6. Send participants `notebooks/00_pytorch_refresher.ipynb` as optional pre-work, and ask them to confirm they can get a T4 runtime with their Google account.
7. Add your own images to the slide deck: every dashed box names the figure that belongs there.

## Timing (4 hours per day, breaks included)

### Day 1
| Time | Block | Material |
|---|---|---|
| 0:00–0:35 | Welcome, what generative AI is, families of generative models, the diffusion idea in one picture | Slides: Intro |
| 0:35–0:50 | Colab setup + Lab 0 highlights (tensors, noise, training loop) | `00_pytorch_refresher` |
| 0:50–1:15 | Module 1: convolutions, down/up-sampling, U-Net, skip connections | Slides: Module 1 |
| 1:15–1:50 | **Lab 1** | `01_unet_denoising` |
| 1:50–2:00 | Break | |
| 2:00–2:30 | Module 2: forward process, schedule, closed form, noise-prediction loss, reverse process | Slides: Module 2 |
| 2:30–3:10 | **Lab 2** | `02_diffusion_ddpm` |
| 3:10–3:15 | Short break | |
| 3:15–3:30 | Module 3: GroupNorm, GELU, rearrange pooling, sinusoidal embeddings | Slides: Module 3 |
| 3:30–3:55 | **Lab 3** (start training first, explain while it runs) | `03_optimizations` |
| 3:55–4:00 | Day 1 recap | |

### Day 2
| Time | Block | Material |
|---|---|---|
| 0:00–0:15 | Recap of Day 1, questions | Slides: Day 2 opener |
| 0:15–0:40 | Module 4: conditioning, Bernoulli mask, guidance weight | Slides: Module 4 |
| 0:40–1:20 | **Lab 4** | `04_classifier_free_guidance` |
| 1:20–1:30 | Break | |
| 1:30–1:55 | Module 5: CLIP, contrastive training, the image-embedding/text-embedding swap | Slides: Module 5 |
| 1:55–2:40 | **Lab 5** | `05_clip_text_to_image` |
| 2:40–2:55 | Beyond the workshop: latent diffusion, faster samplers, responsible use | Slides: Outlook |
| 2:55–3:00 | Short break | |
| 3:00–3:20 | **Quiz** | `assessment/quiz` |
| 3:20–3:55 | **Coding assessment** | `assessment/coding_assessment` |
| 3:55–4:00 | Wrap-up, resources, feedback form | |

## Running the labs

* **Pattern for every lab:** 2 minutes of framing, students work, stop 5 minutes before the end and walk through the solution notebook on the projector.
* **Start long training cells early** and talk over them. In Labs 3–5 the training cell can run while you explain the next section.
* **Nobody gets stranded.** From Lab 3 onwards the notebooks import the reference implementation from `diffusion_workshop/`, so a participant who did not finish an earlier lab can still do the next one.
* **Fast finishers:** every notebook ends with "If you have time" experiments. The ablation in Lab 3 and the `DROP_PROB` experiments in Lab 4 lead to the best discussions.
* **Colab hiccups:** "No GPU available" means the free quota is used up; pair the participant with a neighbour. A `NameError: FIXME` means a blank was not filled in. If a runtime disconnects, re-run the setup cell first.

## Points worth making out loud

* Lab 1's gray blobs are the motivation for everything that follows; do not skip the failure.
* The forward process has **no learnable parameters**. All learning happens in the noise predictor.
* The check in Lab 2 TODO 5 (reverse step recovers x₀ exactly when given the true noise) is a good moment to explain *why* predicting noise is enough.
* Guidance weight: `w = -1` unconditional, `w = 0` conditional, `w > 0` extrapolates. The grid in Lab 4 makes the diversity-fidelity trade-off visible.
* Lab 5 images are 32×32 blobs of colour. Set expectations: the point is that the blobs follow *text the model never trained on*. Then connect to Stable Diffusion on the outlook slides.

## Known limits

* Image quality is bounded by tiny images, a small U-Net and a few minutes of training. That is a deliberate trade for a workshop.
* CLIP image and text embeddings do not overlap perfectly (the "modality gap"); stretch exercise 4 in Lab 5 explores one remedy.
* The quiz hashes stop casual peeking only. Treat the quiz as formative unless you proctor it.
