"""Small plotting helpers. Tensors are expected in [-1, 1], shape (B, C, H, W)."""
import math

import matplotlib.pyplot as plt
import torch


def to_image(t):
    """(C, H, W) tensor in [-1, 1] -> (H, W, C) or (H, W) numpy array in [0, 1]."""
    t = ((t.detach().float().cpu().clamp(-1, 1) + 1) / 2).permute(1, 2, 0)
    return t.squeeze(-1).numpy()


def show_images(batch, titles=None, ncols=8, scale=1.4, suptitle=None):
    """Show a batch of images in a grid."""
    batch = batch.detach().cpu()
    n = len(batch)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * scale, nrows * scale * (1.25 if titles is not None else 1)))
    axes = [axes] if nrows * ncols == 1 else list(axes.flat)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(to_image(batch[i]), cmap="gray", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def show_rows(rows, row_labels, ncols=8, scale=1.3):
    """Show several batches as labelled rows (e.g. clean / noisy / denoised)."""
    fig, axes = plt.subplots(len(rows), ncols, figsize=(ncols * scale, len(rows) * scale), squeeze=False)
    for r, (batch, label) in enumerate(zip(rows, row_labels)):
        for c in range(ncols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(batch):
                ax.imshow(to_image(batch[c]), cmap="gray", vmin=0, vmax=1)
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_losses(losses, title="Training loss", window=50):
    losses = torch.tensor(losses, dtype=torch.float)
    plt.figure(figsize=(6, 3))
    plt.plot(losses, alpha=0.3, label="per step")
    if len(losses) > window:
        smooth = losses.unfold(0, window, 1).mean(1)
        plt.plot(range(window - 1, len(losses)), smooth, label=f"{window}-step mean")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.show()


def animate(frames, interval=80, scale=3.0):
    """frames: list of (C, H, W) tensors -> HTML animation for notebooks."""
    from IPython.display import HTML
    from matplotlib import animation

    fig, ax = plt.subplots(figsize=(scale, scale))
    ax.axis("off")
    ims = [[ax.imshow(to_image(f), cmap="gray", vmin=0, vmax=1, animated=True)] for f in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=1500)
    plt.close(fig)
    return HTML(ani.to_jshtml())
