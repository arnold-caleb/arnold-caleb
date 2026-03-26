"""
Restricted Boltzmann Machine (RBM) in JAX.

An RBM is an energy-based model with visible units v and hidden units h.
Energy: E(v, h) = -v^T W h - b^T v - c^T h
The joint distribution is: p(v, h) = exp(-E(v,h)) / Z

Training uses Contrastive Divergence (CD-k): approximate the intractable
negative phase gradient with k steps of Gibbs sampling starting from data.

The key insight vs diffusion models: here MCMC (distribution evolution) is
*inside* the training loop as part of the gradient estimate. In diffusion
models, the iterative process only happens at inference.
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from typing import NamedTuple


class RBMParams(NamedTuple):
    """RBM parameters."""
    W: jnp.ndarray   # (n_visible, n_hidden) weight matrix
    vbias: jnp.ndarray  # (n_visible,) visible bias
    hbias: jnp.ndarray  # (n_hidden,) hidden bias


def init_rbm(key: jax.Array, n_visible: int, n_hidden: int, scale: float = 0.01) -> RBMParams:
    """Initialize RBM parameters.

    Small random weights, zero biases. The visible bias can optionally
    be initialized to log(p_i / (1 - p_i)) where p_i is the mean activation
    of visible unit i in the training data (done separately).
    """
    k1, k2 = random.split(key)
    W = scale * random.normal(k1, (n_visible, n_hidden))
    vbias = jnp.zeros(n_visible)
    hbias = jnp.zeros(n_hidden)
    return RBMParams(W=W, vbias=vbias, hbias=hbias)


def visible_to_hidden_prob(params: RBMParams, v: jnp.ndarray) -> jnp.ndarray:
    """p(h=1|v) = sigmoid(v @ W + c).

    This is the "recognition" or "inference" direction.
    Because RBMs have no intra-layer connections, all hidden units
    are conditionally independent given the visible units.
    """
    return jax.nn.sigmoid(v @ params.W + params.hbias)


def hidden_to_visible_prob(params: RBMParams, h: jnp.ndarray) -> jnp.ndarray:
    """p(v=1|h) = sigmoid(h @ W^T + b).

    This is the "generative" direction.
    Similarly, all visible units are conditionally independent given hidden.
    """
    return jax.nn.sigmoid(h @ params.W.T + params.vbias)


def sample_binary(key: jax.Array, probs: jnp.ndarray) -> jnp.ndarray:
    """Sample binary units from Bernoulli distribution."""
    return (random.uniform(key, probs.shape) < probs).astype(jnp.float32)


def gibbs_step(params: RBMParams, v: jnp.ndarray, key: jax.Array):
    """One full Gibbs sampling step: v -> h -> v'.

    Returns:
        v_new: reconstructed visible sample
        h_prob: hidden probabilities (used for gradient)
        h_sample: hidden binary sample
        v_prob: visible probabilities (reconstruction)
    """
    k1, k2 = random.split(key)

    h_prob = visible_to_hidden_prob(params, v)
    h_sample = sample_binary(k1, h_prob)

    v_prob = hidden_to_visible_prob(params, h_sample)
    v_sample = sample_binary(k2, v_prob)

    return v_sample, h_prob, h_sample, v_prob


def cd_k(params: RBMParams, v_data: jnp.ndarray, key: jax.Array, k: int = 1):
    """Contrastive Divergence with k Gibbs steps.

    This is where the "distribution evolution during training" happens.
    We run k steps of the Markov chain to approximate the model's
    equilibrium distribution (the negative phase).

    CD-1 (k=1) works surprisingly well in practice — Hinton (2002) showed
    that even one step gives a good enough gradient direction, even though
    the chain hasn't mixed.

    Returns:
        grads: RBMParams with gradient estimates
        metrics: dict with reconstruction error etc.
    """
    batch_size = v_data.shape[0]

    # Positive phase: clamp visible to data, compute hidden
    h_prob_pos = visible_to_hidden_prob(params, v_data)

    # Negative phase: run k Gibbs steps (this IS distribution evolution)
    v_neg = v_data
    for i in range(k):
        key, step_key = random.split(key)
        # Use jax.lax.cond or just loop — for small k, unrolled loop is fine
        keys = random.split(step_key, batch_size)
        v_neg, h_prob_neg, h_sample, v_prob_neg = jax.vmap(
            lambda v, rng: gibbs_step(params, v, rng)
        )(v_neg, keys)

    # For the last step, use probabilities instead of samples for visible
    # (reduces sampling noise in the gradient — Hinton's practical guide)
    h_prob_neg = visible_to_hidden_prob(params, v_prob_neg)

    # Compute gradients: <v_data h_data^T> - <v_model h_model^T>
    # These are the "positive" and "negative" statistics
    dW = (v_data.T @ h_prob_pos - v_prob_neg.T @ h_prob_neg) / batch_size
    dvbias = (v_data - v_prob_neg).mean(axis=0)
    dhbias = (h_prob_pos - h_prob_neg).mean(axis=0)

    grads = RBMParams(W=dW, vbias=dvbias, hbias=dhbias)

    # Reconstruction error (not the actual objective, but useful for monitoring)
    recon_error = jnp.mean((v_data - v_prob_neg) ** 2)

    # Free energy for monitoring (proportional to log-likelihood)
    free_energy = compute_free_energy(params, v_data).mean()

    return grads, {"recon_error": recon_error, "free_energy": free_energy}


def compute_free_energy(params: RBMParams, v: jnp.ndarray) -> jnp.ndarray:
    """Free energy F(v) = -b^T v - sum_j log(1 + exp(W_j^T v + c_j)).

    Marginalizing out hidden units analytically.
    F(v) = -log sum_h exp(-E(v,h)) (up to constant).
    Lower free energy = higher probability under the model.
    """
    vbias_term = v @ params.vbias
    hidden_input = v @ params.W + params.hbias
    hidden_term = jnp.sum(jax.nn.softplus(hidden_input), axis=-1)
    return -vbias_term - hidden_term


# --- Persistent Contrastive Divergence (PCD) ---
# PCD maintains persistent Markov chains across updates.
# Instead of starting from data each time, chains continue from
# where they left off. This gives a better approximation of the
# model distribution (negative phase) since chains have more time to mix.

def pcd_step(params: RBMParams, v_data: jnp.ndarray, persistent_chains: jnp.ndarray,
             key: jax.Array, k: int = 1):
    """Persistent Contrastive Divergence.

    Like CD-k but starts Gibbs chains from persistent_chains
    (the state from the previous parameter update) rather than from data.
    This gives a better gradient estimate.
    """
    batch_size = v_data.shape[0]

    # Positive phase
    h_prob_pos = visible_to_hidden_prob(params, v_data)

    # Negative phase: continue from persistent chains
    v_neg = persistent_chains
    for i in range(k):
        key, step_key = random.split(key)
        keys = random.split(step_key, persistent_chains.shape[0])
        v_neg, h_prob_neg, h_sample, v_prob_neg = jax.vmap(
            lambda v, rng: gibbs_step(params, v, rng)
        )(v_neg, keys)

    h_prob_neg = visible_to_hidden_prob(params, v_neg)

    # Gradients
    dW = (v_data.T @ h_prob_pos - v_neg[:batch_size].T @ h_prob_neg[:batch_size]) / batch_size
    dvbias = (v_data - v_neg[:batch_size]).mean(axis=0)
    dhbias = (h_prob_pos - h_prob_neg[:batch_size]).mean(axis=0)

    grads = RBMParams(W=dW, vbias=dvbias, hbias=dhbias)
    recon_error = jnp.mean((v_data - hidden_to_visible_prob(params, sample_binary(key, visible_to_hidden_prob(params, v_data)))) ** 2)

    return grads, v_neg, {"recon_error": recon_error}


# --- Training utilities ---

class TrainState(NamedTuple):
    """Training state with momentum."""
    params: RBMParams
    velocity: RBMParams  # momentum buffer
    step: int


def init_train_state(params: RBMParams) -> TrainState:
    """Initialize training state with zero velocity."""
    zero_vel = RBMParams(
        W=jnp.zeros_like(params.W),
        vbias=jnp.zeros_like(params.vbias),
        hbias=jnp.zeros_like(params.hbias),
    )
    return TrainState(params=params, velocity=zero_vel, step=0)


def update_params(state: TrainState, grads: RBMParams,
                  lr: float = 0.01, momentum: float = 0.9,
                  weight_decay: float = 0.0001) -> TrainState:
    """SGD with momentum and weight decay.

    Following Hinton's practical guide:
    - Start with momentum=0.5, increase to 0.9 after initial epochs
    - Weight decay ~0.0001 for regularization
    - lr ~0.01 for CD-1 on MNIST
    """
    new_vel = RBMParams(
        W=momentum * state.velocity.W + lr * (grads.W - weight_decay * state.params.W),
        vbias=momentum * state.velocity.vbias + lr * grads.vbias,
        hbias=momentum * state.velocity.hbias + lr * grads.hbias,
    )
    new_params = RBMParams(
        W=state.params.W + new_vel.W,
        vbias=state.params.vbias + new_vel.vbias,
        hbias=state.params.hbias + new_vel.hbias,
    )
    return TrainState(params=new_params, velocity=new_vel, step=state.step + 1)


# --- Sampling / Generation ---

def generate_samples(params: RBMParams, key: jax.Array, n_samples: int,
                     n_gibbs_steps: int = 1000, n_visible: int = 784) -> jnp.ndarray:
    """Generate samples by running a long Gibbs chain from random init.

    This is pure MCMC — we're sampling from the model's learned distribution.
    Need many steps for the chain to mix (reach equilibrium).
    """
    key, init_key = random.split(key)
    v = random.bernoulli(init_key, 0.5, (n_samples, n_visible)).astype(jnp.float32)

    def gibbs_body(carry, _):
        v, key = carry
        key, step_key = random.split(key)
        keys = random.split(step_key, n_samples)
        v_new, _, _, v_prob = jax.vmap(
            lambda v, rng: gibbs_step(params, v, rng)
        )(v, keys)
        return (v_new, key), v_prob

    (v_final, _), _ = jax.lax.scan(gibbs_body, (v, key), None, length=n_gibbs_steps)
    # Return probabilities from last step (smoother than binary samples)
    h_prob = visible_to_hidden_prob(params, v_final)
    v_prob = hidden_to_visible_prob(params, h_prob)
    return v_prob
