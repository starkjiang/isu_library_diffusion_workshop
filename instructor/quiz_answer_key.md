# Quiz answer key (instructors only)

Suggested pass mark: 14 / 20 (70%).

| # | Module | Answer | Why |
|---|---|---|---|
| 1 | Foundations | **b** | Generative models learn the data distribution well enough to draw new samples from it; classifiers (a) are discriminative. |
| 2 | Foundations | **c** | PyTorch uses (batch, channels, height, width). |
| 3 | U-Net | **a** | Down-sampling loses detail; concatenating encoder features gives the decoder that detail back. |
| 4 | U-Net | **d** | Pure noise was outside its training distribution, and the MSE-optimal guess under uncertainty is the mean image. |
| 5 | Diffusion | **b** | The forward process needs no network: it only adds noise according to the beta schedule. |
| 6 | Diffusion | **c** | a_bar = cumprod(1 - beta). Its square root is the weight on x_0 in the closed-form forward jump. |
| 7 | Diffusion | **a** | Each batch draws random timesteps and jumps straight to them, which keeps training cheap. |
| 8 | Diffusion | **d** | With a suitable schedule ᾱ_T ≈ 0, so x_T ≈ ε ~ N(0, I). That is why sampling can start from torch.randn. |
| 9 | Diffusion | **b** | The loss is the MSE between the true noise ε and the predicted noise ε̂(x_t, t). |
| 10 | Diffusion | **c** | The right amount of noise to remove is very different at t = 5 and at t = 295. |
| 11 | Diffusion | **a** | Fresh noise keeps sampling stochastic along the way; adding it to the final output would only leave noise in the result. |
| 12 | Optimizations | **d** | Batch statistics are unreliable when a batch mixes very different noise levels; GroupNorm avoids them. |
| 13 | Optimizations | **b** | This avoids the 'dying ReLU' problem and tends to train more smoothly. |
| 14 | Optimizations | **c** | A following convolution then learns how to mix the four values, instead of max-pool's fixed rule. |
| 15 | Optimizations | **a** | Neighbouring values of t/T are nearly identical; the multi-frequency code separates them clearly. |
| 16 | Guidance | **d** | Classifier-free guidance needs both a conditional and an unconditional prediction from one model. |
| 17 | Guidance | **b** | w = 0 leaves ε_cond alone; w = −1 gives the purely unconditional prediction. |
| 18 | Guidance | **c** | Guidance trades diversity for fidelity; too much of it distorts the images. |
| 19 | CLIP | **a** | The result is an image encoder and a text encoder that share one embedding space. |
| 20 | CLIP | **d** | The diffusion model only ever sees a 512-d vector; CLIP makes captions and pictures land near each other. |

## Verifying a completion code

The quiz notebook prints `name | score | code`. To verify it:

```python
import hashlib
hashlib.sha256(f"dw-quiz-v1:{name}:{score}".encode()).hexdigest()[:8]
```
