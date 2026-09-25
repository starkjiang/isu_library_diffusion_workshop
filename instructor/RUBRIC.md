# Assessment rubric

Both assessments run in the last hour of Day 2. Suggested weighting for a certificate: quiz 40%, coding 60%, pass at 70% overall.

## Quiz (20 points, 20 minutes, closed book)
One point per question. Pass mark 14/20. Students submit the line printed by the last cell of `assessment/quiz.ipynb` (`name | score | code`); `quiz_answer_key.md` shows how to verify the code. A printable version is in `assessment/quiz.md`.

## Coding assessment (100 points, 35 minutes, open book)

| Part | Points | How it is graded |
|---|---|---|
| TODO 1 noise schedule | 10 | ✅ check 1 passes |
| TODO 2 forward diffusion `q` | 10 | ✅ check 2 passes |
| TODO 3 one-hot context + Bernoulli mask | 10 | ✅ check 3 passes |
| TODO 4 noise-prediction loss | 10 | ✅ check 4 passes |
| TODO 5 reverse step + guidance | 10 | ✅ check 5 passes |
| Generated digits | 40 | grader classifier agrees with the requested digit on ≥ 90% of 100 images |
| Written answers | 10 | instructor, see below |

Partial credit for the generation part (suggested): 80–89% → 30 points, 60–79% → 20 points, below 60% → 0.

### Written answers (3 + 3 + 4 points)
1. *Why predict the noise?* Full marks for any of: the target has the same simple distribution (unit Gaussian) at every timestep, which makes the regression well-scaled and stable; it is equivalent to predicting x₀ up to a known rescaling but empirically trains better (Ho et al. 2020); the reverse-step formula uses ε̂ directly.
2. *DROP_PROB = 0?* The network never learns an unconditional prediction, so ε_drop is meaningless (the model has never seen an all-zero context) and guidance with w > 0 pushes in an arbitrary direction. Conditional sampling (w = 0) still works.
3. *Raising W?* Samples match the requested digit more reliably and look crisper/bolder; the cost is lower diversity and, for very large W, saturated or distorted strokes.

Submission: students share their executed notebook (File > Save a copy in Drive, then share the link) or download the `.ipynb`.
