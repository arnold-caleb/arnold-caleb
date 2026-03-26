"""
Train an RBM on MNIST in JAX — designed for Google Colab.

Run in Colab:
    !pip install jax jaxlib tensorflow-datasets matplotlib
    # Then run this script or paste cells into notebook.

What's happening during training:
    Each CD-k step runs k iterations of Gibbs sampling. This is the
    "distribution evolution during training" — the Markov chain is
    exploring the model's current distribution to estimate the gradient.

    Compare with diffusion: there, training just samples (timestep, noise)
    pairs — no iterative chain. The iterative part only happens at inference.
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from rbm import (
    RBMParams, init_rbm, cd_k, pcd_step,
    init_train_state, update_params,
    visible_to_hidden_prob, hidden_to_visible_prob,
    generate_samples, compute_free_energy, sample_binary,
)


# --- Data loading ---

def load_mnist_binary(threshold: float = 0.5):
    """Load MNIST and binarize (RBMs work with binary visible units).

    Binarization: pixel > threshold -> 1, else 0.
    This is standard for RBM experiments on MNIST.
    """
    try:
        import tensorflow_datasets as tfds
        ds = tfds.load("mnist", split="train", as_supervised=True, batch_size=-1)
        images, labels = tfds.as_numpy(ds)
        images = images.reshape(-1, 784).astype(np.float32) / 255.0

        ds_test = tfds.load("mnist", split="test", as_supervised=True, batch_size=-1)
        test_images, test_labels = tfds.as_numpy(ds_test)
        test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0
    except ImportError:
        # Fallback: use keras
        from tensorflow.keras.datasets import mnist
        (images, labels), (test_images, test_labels) = mnist.load_data()
        images = images.reshape(-1, 784).astype(np.float32) / 255.0
        test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0

    # Binarize
    train = (images > threshold).astype(np.float32)
    test = (test_images > threshold).astype(np.float32)
    return train, test, labels, test_labels


def make_batches(data: np.ndarray, batch_size: int, key: jax.Array):
    """Shuffle and split into batches."""
    n = data.shape[0]
    perm = random.permutation(key, n)
    data = data[perm]
    n_batches = n // batch_size
    batches = data[:n_batches * batch_size].reshape(n_batches, batch_size, -1)
    return batches


# --- Training loop ---

def train_rbm(
    n_visible: int = 784,
    n_hidden: int = 500,
    batch_size: int = 64,
    n_epochs: int = 50,
    cd_steps: int = 1,
    lr: float = 0.01,
    momentum_init: float = 0.5,
    momentum_final: float = 0.9,
    momentum_switch_epoch: int = 5,
    weight_decay: float = 0.0001,
    use_pcd: bool = False,
    seed: int = 42,
):
    """Train an RBM on MNIST.

    Hyperparameters follow Hinton's "A Practical Guide to Training RBMs" (2010):
    - 500 hidden units
    - CD-1 (one Gibbs step)
    - lr=0.01
    - momentum: 0.5 initially, 0.9 after epoch 5
    - weight decay: 0.0001
    - mini-batch size: ~64
    """
    print("Loading MNIST...")
    train_data, test_data, train_labels, test_labels = load_mnist_binary()
    print(f"Train: {train_data.shape}, Test: {test_data.shape}")

    key = random.PRNGKey(seed)
    key, init_key = random.split(key)

    # Initialize
    params = init_rbm(init_key, n_visible, n_hidden)

    # Initialize visible bias to log(p_i / (1 - p_i)) where p_i = mean activation
    # This helps the model start near the data distribution
    p = train_data.mean(axis=0).clip(0.001, 0.999)
    params = params._replace(vbias=jnp.log(p / (1 - p)))

    state = init_train_state(params)

    # For PCD: maintain persistent chains
    if use_pcd:
        key, chain_key = random.split(key)
        persistent_chains = random.bernoulli(chain_key, 0.5, (batch_size, n_visible)).astype(jnp.float32)

    history = {"recon_error": [], "free_energy": []}

    print(f"\nTraining RBM: {n_visible} visible, {n_hidden} hidden")
    print(f"Method: {'PCD' if use_pcd else 'CD'}-{cd_steps}, lr={lr}, batch_size={batch_size}")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        key, batch_key = random.split(key)
        batches = make_batches(train_data, batch_size, batch_key)

        # Momentum schedule
        momentum = momentum_final if epoch >= momentum_switch_epoch else momentum_init

        epoch_recon = 0.0
        epoch_fe = 0.0
        n_batches = batches.shape[0]

        for i in range(n_batches):
            v_batch = jnp.array(batches[i])
            key, cd_key = random.split(key)

            if use_pcd:
                grads, persistent_chains, metrics = pcd_step(
                    state.params, v_batch, persistent_chains, cd_key, k=cd_steps
                )
            else:
                grads, metrics = cd_k(state.params, v_batch, cd_key, k=cd_steps)

            state = update_params(state, grads, lr=lr, momentum=momentum, weight_decay=weight_decay)

            epoch_recon += metrics["recon_error"]
            if "free_energy" in metrics:
                epoch_fe += metrics["free_energy"]

        avg_recon = epoch_recon / n_batches
        avg_fe = epoch_fe / n_batches
        history["recon_error"].append(float(avg_recon))
        history["free_energy"].append(float(avg_fe))

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"Epoch {epoch:3d} | Recon Error: {avg_recon:.4f} | "
                  f"Free Energy: {avg_fe:.2f} | Momentum: {momentum}")

    print(f"\nTraining complete. Final recon error: {history['recon_error'][-1]:.4f}")
    return state.params, history, (train_data, test_data, train_labels, test_labels)


# --- Visualization ---

def plot_filters(params: RBMParams, n_filters: int = 100, cols: int = 10):
    """Visualize learned weight filters (receptive fields).

    Each column of W is a filter — what pattern in visible space
    activates a particular hidden unit. Good RBMs learn edge-like,
    stroke-like features reminiscent of Gabor filters.
    """
    rows = n_filters // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i in range(n_filters):
        ax = axes[i // cols, i % cols]
        w = np.array(params.W[:, i].reshape(28, 28))
        ax.imshow(w, cmap="RdBu_r", vmin=-np.abs(w).max(), vmax=np.abs(w).max())
        ax.axis("off")
    plt.suptitle("Learned RBM Filters (W columns)", fontsize=14)
    plt.tight_layout()
    plt.savefig("rbm_filters.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rbm_filters.png")


def plot_reconstructions(params: RBMParams, data: np.ndarray, n_examples: int = 10):
    """Show original vs reconstructed images (one Gibbs step).

    Good reconstructions = the model can encode and decode well.
    """
    v = jnp.array(data[:n_examples])
    h_prob = visible_to_hidden_prob(params, v)
    v_recon = hidden_to_visible_prob(params, h_prob)

    fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 1.5, 3))
    for i in range(n_examples):
        axes[0, i].imshow(np.array(v[i]).reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(np.array(v_recon[i]).reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstruction", fontsize=10)
    plt.suptitle("RBM Reconstructions (v → h → v')", fontsize=14)
    plt.tight_layout()
    plt.savefig("rbm_reconstructions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rbm_reconstructions.png")


def plot_samples(params: RBMParams, key: jax.Array, n_samples: int = 100,
                 n_gibbs: int = 1000):
    """Generate samples by running long Gibbs chains.

    This shows what the model has learned as a generative model.
    Samples should look like plausible digits.
    """
    print(f"Generating {n_samples} samples with {n_gibbs} Gibbs steps...")
    samples = generate_samples(params, key, n_samples, n_gibbs_steps=n_gibbs)

    cols = 10
    rows = n_samples // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    for i in range(n_samples):
        ax = axes[i // cols, i % cols]
        ax.imshow(np.array(samples[i]).reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.suptitle(f"RBM Samples ({n_gibbs} Gibbs steps)", fontsize=14)
    plt.tight_layout()
    plt.savefig("rbm_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rbm_samples.png")


def plot_training_curves(history: dict):
    """Plot reconstruction error and free energy over training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["recon_error"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Error (MSE)")
    ax1.set_title("Reconstruction Error")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["free_energy"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Free Energy")
    ax2.set_title("Average Free Energy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rbm_training.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: rbm_training.png")


# --- Main ---

if __name__ == "__main__":
    # Train
    params, history, (train_data, test_data, train_labels, test_labels) = train_rbm(
        n_hidden=500,
        n_epochs=50,
        cd_steps=1,
        lr=0.01,
        batch_size=64,
    )

    # Visualize
    plot_training_curves(history)
    plot_filters(params, n_filters=100)
    plot_reconstructions(params, test_data)
    plot_samples(params, random.PRNGKey(0), n_samples=100, n_gibbs=2000)
