# Quiz · Generative AI with Diffusion Models

20 questions · 20 minutes · one correct answer each · closed book

Name: ______________________

**1. What does a *generative* model learn to do?**  <sub>(Foundations)</sub>

- (a) Assign each input to one of a fixed set of labels
- (b) Produce new samples that resemble its training data
- (c) Compress files without any loss of information
- (d) Retrieve the training example closest to a query

**2. In PyTorch, a batch of 64 RGB images of 32×32 pixels has the shape…**  <sub>(Foundations)</sub>

- (a) (64, 32, 32, 3)
- (b) (3, 64, 32, 32)
- (c) (64, 3, 32, 32)
- (d) (32, 32, 3, 64)

**3. What is the job of the skip connections in a U-Net?**  <sub>(U-Net)</sub>

- (a) They hand fine spatial detail from the encoder directly to the decoder stage of the same size
- (b) They skip training on easy images to save time
- (c) They reduce the number of parameters by sharing weights
- (d) They remove the need for an activation function

**4. In Lab 1 the denoising U-Net produced gray blobs when given pure noise. The main reason is that…**  <sub>(U-Net)</sub>

- (a) the U-Net had too many parameters
- (b) the GPU ran out of memory
- (c) FashionMNIST images are too small to generate
- (d) it was trained on one moderate noise level only, and an MSE-trained model answers an ambiguous input with an average

**5. What happens at each step of the *forward* diffusion process?**  <sub>(Diffusion)</sub>

- (a) The U-Net removes a little noise
- (b) A small, scheduled amount of Gaussian noise is added to the image
- (c) The image is down-sampled by a factor of two
- (d) The image is encoded by CLIP

**6. ᾱ_t ("alpha bar") is…**  <sub>(Diffusion)</sub>

- (a) the learning rate at step t
- (b) the sum of all betas up to step t
- (c) the cumulative product of α_1 … α_t, which tells how much of the original image survives after t steps
- (d) the U-Net's prediction at step t

**7. Why is the closed form  x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε  so useful for training?**  <sub>(Diffusion)</sub>

- (a) We can create a training example for any random timestep in one line, without looping over t steps
- (b) It removes the need for a noise schedule
- (c) It makes the U-Net smaller
- (d) It guarantees that the loss reaches zero

**8. As t approaches T, the noised image x_t looks like…**  <sub>(Diffusion)</sub>

- (a) the original image
- (b) a blurred version of the image
- (c) an all-black image
- (d) a sample of standard Gaussian noise

**9. In our DDPM, what is the U-Net trained to output?**  <sub>(Diffusion)</sub>

- (a) The class label of the image
- (b) The noise that was added to the image
- (c) The timestep t
- (d) The next, noisier image

**10. Why does the U-Net receive the timestep t as an input?**  <sub>(Diffusion)</sub>

- (a) To know how many epochs have passed
- (b) To decide which class to draw
- (c) Because one network must handle every noise level, and it needs to know which level it is looking at
- (d) Because PyTorch requires two inputs

**11. In the reverse step a little fresh noise is added, except at t = 0. Why the exception?**  <sub>(Diffusion)</sub>

- (a) Step 0 produces the final image, which should be clean
- (b) The random number generator runs out
- (c) β_0 is negative
- (d) It makes training faster

**12. How does Group Normalization differ from Batch Normalization?**  <sub>(Optimizations)</sub>

- (a) It normalizes across the whole dataset
- (b) It has no learnable parameters
- (c) It only works on grayscale images
- (d) It computes statistics within each sample, over groups of channels, so it does not depend on the rest of the batch

**13. Compared with ReLU, GELU…**  <sub>(Optimizations)</sub>

- (a) is exactly zero for all negative inputs
- (b) is smooth and lets a small signal (and gradient) through for negative inputs
- (c) is a linear function
- (d) can only be used in the last layer

**14. Rearrange pooling turns (B, C, H, W) into…**  <sub>(Optimizations)</sub>

- (a) (B, C, H/2, W/2) by keeping the largest value of each 2×2 patch
- (b) (B, C/4, 2H, 2W)
- (c) (B, 4C, H/2, W/2) by moving each 2×2 patch into the channel axis, so no pixel is discarded
- (d) (B, C, H, W) unchanged

**15. Why use sinusoidal position embeddings for the timestep instead of the single number t/T?**  <sub>(Optimizations)</sub>

- (a) A vector of sines and cosines at many frequencies gives every timestep a distinct, easy-to-read fingerprint
- (b) They make sampling need fewer steps
- (c) They remove the need for skip connections
- (d) They are required by GroupNorm

**16. Training with a Bernoulli context mask and drop probability 0.1 means that…**  <sub>(Guidance)</sub>

- (a) 10% of the pixels are set to zero
- (b) 10% of the weights are frozen
- (c) 10% of the timesteps are skipped
- (d) for roughly 10% of the samples the context is zeroed, so the same network also learns unconditional denoising

**17. With  ε̂ = (1 + w) · ε_cond − w · ε_uncond,  which w gives plain conditional sampling with no extra guidance?**  <sub>(Guidance)</sub>

- (a) w = −1
- (b) w = 0
- (c) w = 1
- (d) w = 2

**18. What is the typical effect of a very large guidance weight w?**  <sub>(Guidance)</sub>

- (a) More varied samples that ignore the prompt
- (b) Faster sampling
- (c) Samples match the condition strongly but lose diversity and can look over-saturated
- (d) The model stops needing a timestep

**19. How was CLIP trained?**  <sub>(CLIP)</sub>

- (a) With a contrastive objective on image–caption pairs: matching pairs are pulled together, mismatched pairs pushed apart
- (b) By denoising images step by step
- (c) By classifying ImageNet into 1,000 fixed labels
- (d) By predicting the next word of a caption

**20. In Lab 5 we trained on CLIP *image* embeddings but sampled with CLIP *text* embeddings. Why does that work?**  <sub>(CLIP)</sub>

- (a) The U-Net contains a hidden language model
- (b) The flower photos came with captions
- (c) Text embeddings are converted to one-hot labels first
- (d) CLIP places an image and a description of it close together in the same vector space, so a text vector is a usable stand-in for an image vector
