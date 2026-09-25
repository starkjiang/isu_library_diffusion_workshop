#!/usr/bin/env python3
"""Single source of truth for the end-of-workshop quiz.

    python tools/make_quiz.py && python tools/build_notebooks.py

writes
    assessment/quiz.md                  printable student version
    instructor/quiz_answer_key.md       answers + one-line explanations
    notebook_src/quiz.nbsrc             self-grading Colab notebook (answers stored only as salted hashes)

INSTRUCTORS: this file contains the answers. Keep it out of any repository students can see.
"""
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SALT = "dw-quiz-v1"          # change this if you edit the questions and want fresh hashes

# (module, question, [a, b, c, d], correct letter, explanation)
QUESTIONS = [
    ("Foundations", "What does a *generative* model learn to do?",
     ["Assign each input to one of a fixed set of labels",
      "Produce new samples that resemble its training data",
      "Compress files without any loss of information",
      "Retrieve the training example closest to a query"],
     "b", "Generative models learn the data distribution well enough to draw new samples from it; classifiers (a) are discriminative."),

    ("Foundations", "In PyTorch, a batch of 64 RGB images of 32×32 pixels has the shape…",
     ["(64, 32, 32, 3)", "(3, 64, 32, 32)", "(64, 3, 32, 32)", "(32, 32, 3, 64)"],
     "c", "PyTorch uses (batch, channels, height, width)."),

    ("U-Net", "What is the job of the skip connections in a U-Net?",
     ["They hand fine spatial detail from the encoder directly to the decoder stage of the same size",
      "They skip training on easy images to save time",
      "They reduce the number of parameters by sharing weights",
      "They remove the need for an activation function"],
     "a", "Down-sampling loses detail; concatenating encoder features gives the decoder that detail back."),

    ("U-Net", "In Lab 1 the denoising U-Net produced gray blobs when given pure noise. The main reason is that…",
     ["the U-Net had too many parameters",
      "the GPU ran out of memory",
      "FashionMNIST images are too small to generate",
      "it was trained on one moderate noise level only, and an MSE-trained model answers an ambiguous input with an average"],
     "d", "Pure noise was outside its training distribution, and the MSE-optimal guess under uncertainty is the mean image."),

    ("Diffusion", "What happens at each step of the *forward* diffusion process?",
     ["The U-Net removes a little noise",
      "A small, scheduled amount of Gaussian noise is added to the image",
      "The image is down-sampled by a factor of two",
      "The image is encoded by CLIP"],
     "b", "The forward process needs no network: it only adds noise according to the beta schedule."),

    ("Diffusion", "ᾱ_t (\"alpha bar\") is…",
     ["the learning rate at step t",
      "the sum of all betas up to step t",
      "the cumulative product of α_1 … α_t, which tells how much of the original image survives after t steps",
      "the U-Net's prediction at step t"],
     "c", "a_bar = cumprod(1 - beta). Its square root is the weight on x_0 in the closed-form forward jump."),

    ("Diffusion", "Why is the closed form  x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε  so useful for training?",
     ["We can create a training example for any random timestep in one line, without looping over t steps",
      "It removes the need for a noise schedule",
      "It makes the U-Net smaller",
      "It guarantees that the loss reaches zero"],
     "a", "Each batch draws random timesteps and jumps straight to them, which keeps training cheap."),

    ("Diffusion", "As t approaches T, the noised image x_t looks like…",
     ["the original image", "a blurred version of the image", "an all-black image", "a sample of standard Gaussian noise"],
     "d", "With a suitable schedule ᾱ_T ≈ 0, so x_T ≈ ε ~ N(0, I). That is why sampling can start from torch.randn."),

    ("Diffusion", "In our DDPM, what is the U-Net trained to output?",
     ["The class label of the image", "The noise that was added to the image", "The timestep t", "The next, noisier image"],
     "b", "The loss is the MSE between the true noise ε and the predicted noise ε̂(x_t, t)."),

    ("Diffusion", "Why does the U-Net receive the timestep t as an input?",
     ["To know how many epochs have passed",
      "To decide which class to draw",
      "Because one network must handle every noise level, and it needs to know which level it is looking at",
      "Because PyTorch requires two inputs"],
     "c", "The right amount of noise to remove is very different at t = 5 and at t = 295."),

    ("Diffusion", "In the reverse step a little fresh noise is added, except at t = 0. Why the exception?",
     ["Step 0 produces the final image, which should be clean",
      "The random number generator runs out",
      "β_0 is negative",
      "It makes training faster"],
     "a", "Fresh noise keeps sampling stochastic along the way; adding it to the final output would only leave noise in the result."),

    ("Optimizations", "How does Group Normalization differ from Batch Normalization?",
     ["It normalizes across the whole dataset",
      "It has no learnable parameters",
      "It only works on grayscale images",
      "It computes statistics within each sample, over groups of channels, so it does not depend on the rest of the batch"],
     "d", "Batch statistics are unreliable when a batch mixes very different noise levels; GroupNorm avoids them."),

    ("Optimizations", "Compared with ReLU, GELU…",
     ["is exactly zero for all negative inputs",
      "is smooth and lets a small signal (and gradient) through for negative inputs",
      "is a linear function",
      "can only be used in the last layer"],
     "b", "This avoids the 'dying ReLU' problem and tends to train more smoothly."),

    ("Optimizations", "Rearrange pooling turns (B, C, H, W) into…",
     ["(B, C, H/2, W/2) by keeping the largest value of each 2×2 patch",
      "(B, C/4, 2H, 2W)",
      "(B, 4C, H/2, W/2) by moving each 2×2 patch into the channel axis, so no pixel is discarded",
      "(B, C, H, W) unchanged"],
     "c", "A following convolution then learns how to mix the four values, instead of max-pool's fixed rule."),

    ("Optimizations", "Why use sinusoidal position embeddings for the timestep instead of the single number t/T?",
     ["A vector of sines and cosines at many frequencies gives every timestep a distinct, easy-to-read fingerprint",
      "They make sampling need fewer steps",
      "They remove the need for skip connections",
      "They are required by GroupNorm"],
     "a", "Neighbouring values of t/T are nearly identical; the multi-frequency code separates them clearly."),

    ("Guidance", "Training with a Bernoulli context mask and drop probability 0.1 means that…",
     ["10% of the pixels are set to zero",
      "10% of the weights are frozen",
      "10% of the timesteps are skipped",
      "for roughly 10% of the samples the context is zeroed, so the same network also learns unconditional denoising"],
     "d", "Classifier-free guidance needs both a conditional and an unconditional prediction from one model."),

    ("Guidance", "With  ε̂ = (1 + w) · ε_cond − w · ε_uncond,  which w gives plain conditional sampling with no extra guidance?",
     ["w = −1", "w = 0", "w = 1", "w = 2"],
     "b", "w = 0 leaves ε_cond alone; w = −1 gives the purely unconditional prediction."),

    ("Guidance", "What is the typical effect of a very large guidance weight w?",
     ["More varied samples that ignore the prompt",
      "Faster sampling",
      "Samples match the condition strongly but lose diversity and can look over-saturated",
      "The model stops needing a timestep"],
     "c", "Guidance trades diversity for fidelity; too much of it distorts the images."),

    ("CLIP", "How was CLIP trained?",
     ["With a contrastive objective on image–caption pairs: matching pairs are pulled together, mismatched pairs pushed apart",
      "By denoising images step by step",
      "By classifying ImageNet into 1,000 fixed labels",
      "By predicting the next word of a caption"],
     "a", "The result is an image encoder and a text encoder that share one embedding space."),

    ("CLIP", "In Lab 5 we trained on CLIP *image* embeddings but sampled with CLIP *text* embeddings. Why does that work?",
     ["The U-Net contains a hidden language model",
      "The flower photos came with captions",
      "Text embeddings are converted to one-hot labels first",
      "CLIP places an image and a description of it close together in the same vector space, so a text vector is a usable stand-in for an image vector"],
     "d", "The diffusion model only ever sees a 512-d vector; CLIP makes captions and pictures land near each other."),
]


def h(i, letter):
    return hashlib.sha256(f"{SALT}:q{i}:{letter}".encode()).hexdigest()[:12]


def student_md():
    out = ["# Quiz · Generative AI with Diffusion Models", "",
           "20 questions · 20 minutes · one correct answer each · closed book", "",
           "Name: ______________________", ""]
    for i, (module, q, opts, _, _) in enumerate(QUESTIONS, 1):
        out += [f"**{i}. {q}**  <sub>({module})</sub>", ""]
        out += [f"- ({l}) {o}" for l, o in zip("abcd", opts)]
        out.append("")
    return "\n".join(out)


def key_md():
    out = ["# Quiz answer key (instructors only)", "",
           "Suggested pass mark: 14 / 20 (70%).", "",
           "| # | Module | Answer | Why |", "|---|---|---|---|"]
    for i, (module, _, _, ans, why) in enumerate(QUESTIONS, 1):
        out.append(f"| {i} | {module} | **{ans}** | {why} |")
    out += ["", "## Verifying a completion code", "",
            "The quiz notebook prints `name | score | code`. To verify it:", "",
            "```python", "import hashlib",
            f'hashlib.sha256(f"{SALT}:{{name}}:{{score}}".encode()).hexdigest()[:8]', "```", ""]
    return "\n".join(out)


def nbsrc():
    out = ["#%% md", "# Quiz · Generative AI with Diffusion Models", "", "{{BADGE}}", "",
           "**20 questions · 20 minutes · one correct answer each.**", "",
           "For every question choose a letter in the drop-down on the right (in Colab) or type it between the quotes, e.g. `q1 = \"b\"`. "
           "Run the **last cell** to see your score. No GPU needed.", "",
           "#%% code", 'name = ""  #@param {type:"string"}', ""]
    for i, (module, q, opts, _, _) in enumerate(QUESTIONS, 1):
        out += ["#%% md", f"### {i}. {q}", f"*{module}*", ""]
        out += [f"- **({l})** {o}" for l, o in zip("abcd", opts)]
        out += ["", "#%% code", f'q{i} = ""  #@param ["", "a", "b", "c", "d"]', ""]
    key = ", ".join(f'"q{i}": "{h(i, ans)}"' for i, (_, _, _, ans, _) in enumerate(QUESTIONS, 1))
    n = len(QUESTIONS)
    out += ["#%% md", "## Your score", "", "#%% code",
            "import hashlib",
            f'_SALT, _KEY = "{SALT}", {{{key}}}',
            f"_answers = {{f\"q{{i}}\": str(globals().get(f\"q{{i}}\", \"\")).strip().lower() for i in range(1, {n + 1})}}",
            '_right = {k: hashlib.sha256(f"{_SALT}:{k}:{v}".encode()).hexdigest()[:12] == _KEY[k] for k, v in _answers.items()}',
            "_blank = [k for k, v in _answers.items() if v == \"\"]",
            "score = sum(_right.values())",
            f'print(f"Score: {{score}} / {n}  ->", "PASS" if score >= {round(0.7 * n)} else "below the pass mark of {round(0.7 * n)}")',
            'if _blank: print("Unanswered:", ", ".join(_blank))',
            'print("Review these:", ", ".join(k for k, ok in _right.items() if not ok and k not in _blank) or "none")',
            '_code = hashlib.sha256(f"{_SALT}:{name}:{score}".encode()).hexdigest()[:8]',
            'print(f"\\nSend this line to your instructor:  {name or \'<your name>\'} | {score} | {_code}")', ""]
    return "\n".join(out)


if __name__ == "__main__":
    (ROOT / "assessment" / "quiz.md").write_text(student_md())
    (ROOT / "instructor" / "quiz_answer_key.md").write_text(key_md())
    (ROOT / "notebook_src" / "quiz.nbsrc").write_text(nbsrc())
    counts = {l: sum(a == l for *_, a, _ in QUESTIONS) for l in "abcd"}
    print(f"wrote quiz with {len(QUESTIONS)} questions; answer distribution {counts}")
