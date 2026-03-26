"""
Deep Belief Network (DBN) in JAX.

A DBN is a stack of RBMs trained greedily layer by layer (Hinton et al., 2006).

Architecture:
    visible -> RBM_1 -> RBM_2 -> ... -> RBM_L

Training procedure (greedy layer-wise pretraining):
    1. Train RBM_1 on the raw data
    2. Use RBM_1's hidden activations as "data" for RBM_2
    3. Continue stacking...

Why this works (the key theoretical insight):
    After training RBM_1, its hidden representation h_1 captures structure
    in the data. Training RBM_2 on h_1 models the prior p(h_1) better than
    the single-layer RBM does. Hinton proved that each additional layer
    improves a variational lower bound on the log-likelihood.

    This is fundamentally different from diffusion models:
    - DBN: each layer models a DIFFERENT level of abstraction
    - Diffusion: each step operates at the SAME level but different noise scales

After pretraining, the DBN can be:
    1. Used as a generative model (top-down sampling through layers)
    2. Fine-tuned as a discriminative model (add softmax layer, backprop)
    3. Used to initialize a deep autoencoder
"""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from typing import List, NamedTuple

from rbm import (
    RBMParams, init_rbm, cd_k, init_train_state, update_params,
    visible_to_hidden_prob, hidden_to_visible_prob,
    sample_binary, compute_free_energy,
)


class DBNParams(NamedTuple):
    """DBN parameters = list of RBM parameters."""
    rbm_layers: List[RBMParams]


def init_dbn(key: jax.Array, layer_sizes: List[int], scale: float = 0.01) -> DBNParams:
    """Initialize a DBN with given layer sizes.

    layer_sizes: [n_visible, n_hidden_1, n_hidden_2, ..., n_hidden_L]
    """
    rbm_layers = []
    for i in range(len(layer_sizes) - 1):
        key, init_key = random.split(key)
        rbm = init_rbm(init_key, layer_sizes[i], layer_sizes[i + 1], scale=scale)
        rbm_layers.append(rbm)
    return DBNParams(rbm_layers=rbm_layers)


def propagate_up(dbn_params: DBNParams, v: jnp.ndarray, to_layer: int = -1,
                 use_probs: bool = True) -> jnp.ndarray:
    """Propagate data upward through the DBN.

    Computes hidden representations at each layer.
    use_probs=True: pass probabilities (deterministic, good for features)
    use_probs=False: pass binary samples (stochastic, needed for training next RBM)
    """
    if to_layer == -1:
        to_layer = len(dbn_params.rbm_layers)

    h = v
    for i in range(min(to_layer, len(dbn_params.rbm_layers))):
        h = visible_to_hidden_prob(dbn_params.rbm_layers[i], h)
    return h


def propagate_down(dbn_params: DBNParams, h: jnp.ndarray, from_layer: int = -1) -> jnp.ndarray:
    """Propagate hidden state downward through the DBN (generation).

    Goes from top layer down to visible units.
    """
    if from_layer == -1:
        from_layer = len(dbn_params.rbm_layers) - 1

    v = h
    for i in range(from_layer, -1, -1):
        v = hidden_to_visible_prob(dbn_params.rbm_layers[i], v)
    return v


def greedy_layerwise_pretrain(
    data: np.ndarray,
    layer_sizes: List[int],
    n_epochs_per_layer: int = 30,
    cd_steps: int = 1,
    batch_size: int = 64,
    lr: float = 0.01,
    momentum_init: float = 0.5,
    momentum_final: float = 0.9,
    momentum_switch_epoch: int = 5,
    weight_decay: float = 0.0001,
    seed: int = 42,
) -> DBNParams:
    """Greedy layer-wise pretraining of a DBN.

    This is the core DBN training algorithm:
    1. Train layer 1 RBM on raw data
    2. Transform data through layer 1 to get hidden representations
    3. Train layer 2 RBM on those representations
    4. Repeat...

    At each layer, CD-k runs Gibbs chains — distribution evolution
    happening inside the training of each layer.
    """
    key = random.PRNGKey(seed)
    rbm_layers = []
    current_data = data

    print(f"DBN Architecture: {' -> '.join(map(str, layer_sizes))}")
    print(f"Training {len(layer_sizes) - 1} RBM layers")
    print(f"{'='*60}")

    for layer_idx in range(len(layer_sizes) - 1):
        n_vis = layer_sizes[layer_idx]
        n_hid = layer_sizes[layer_idx + 1]

        print(f"\n--- Layer {layer_idx + 1}: {n_vis} -> {n_hid} ---")

        key, init_key = random.split(key)
        rbm_params = init_rbm(init_key, n_vis, n_hid)

        # Initialize visible bias from data statistics (first layer)
        if layer_idx == 0:
            p = current_data.mean(axis=0).clip(0.001, 0.999)
            rbm_params = rbm_params._replace(vbias=jnp.log(p / (1 - p)))

        state = init_train_state(rbm_params)

        for epoch in range(n_epochs_per_layer):
            key, batch_key = random.split(key)
            n = current_data.shape[0]
            perm = random.permutation(batch_key, n)
            shuffled = current_data[perm]
            n_batches = n // batch_size

            momentum = momentum_final if epoch >= momentum_switch_epoch else momentum_init
            epoch_recon = 0.0

            for i in range(n_batches):
                v_batch = jnp.array(shuffled[i * batch_size:(i + 1) * batch_size])
                key, cd_key = random.split(key)

                grads, metrics = cd_k(state.params, v_batch, cd_key, k=cd_steps)
                state = update_params(state, grads, lr=lr, momentum=momentum,
                                      weight_decay=weight_decay)
                epoch_recon += metrics["recon_error"]

            avg_recon = epoch_recon / n_batches
            if epoch % 10 == 0 or epoch == n_epochs_per_layer - 1:
                print(f"  Epoch {epoch:3d} | Recon Error: {avg_recon:.4f}")

        rbm_layers.append(state.params)

        # Transform data through this layer for the next one
        # Use probabilities (deterministic) as the representation
        print(f"  Transforming data through layer {layer_idx + 1}...")
        # Process in chunks to avoid memory issues
        chunk_size = 5000
        transformed_chunks = []
        for start in range(0, current_data.shape[0], chunk_size):
            chunk = jnp.array(current_data[start:start + chunk_size])
            h = visible_to_hidden_prob(state.params, chunk)
            transformed_chunks.append(np.array(h))
        current_data = np.concatenate(transformed_chunks, axis=0)
        print(f"  New data shape: {current_data.shape}")

    print(f"\n{'='*60}")
    print("DBN pretraining complete!")
    return DBNParams(rbm_layers=rbm_layers)


def dbn_generate(dbn_params: DBNParams, key: jax.Array, n_samples: int = 100,
                 n_gibbs: int = 1000) -> jnp.ndarray:
    """Generate samples from the DBN.

    1. Run Gibbs sampling in the top-level RBM to get top-layer samples
    2. Propagate down through all layers deterministically

    The top RBM is the "associative memory" — it models the prior over
    the highest-level features. Lower layers are just deterministic
    decoders (using the generative weights).
    """
    top_rbm = dbn_params.rbm_layers[-1]
    n_top_hidden = top_rbm.W.shape[1]

    # Initialize top layer randomly
    key, init_key = random.split(key)
    h_top = random.bernoulli(init_key, 0.5, (n_samples, n_top_hidden)).astype(jnp.float32)

    # Gibbs sampling in top RBM
    # Start from the "visible" side of top RBM = hidden of layer below
    v_top = hidden_to_visible_prob(top_rbm, h_top)

    def gibbs_body(carry, _):
        v, key = carry
        key, k1, k2 = random.split(key, 3)
        h_prob = visible_to_hidden_prob(top_rbm, v)
        h_sample = sample_binary(k1, h_prob)
        v_prob = hidden_to_visible_prob(top_rbm, h_sample)
        v_sample = sample_binary(k2, v_prob)
        return (v_sample, key), None

    (v_top_sampled, _), _ = jax.lax.scan(gibbs_body, (v_top, key), None, length=n_gibbs)

    # Get probabilities for smoother output
    h_prob = visible_to_hidden_prob(top_rbm, v_top_sampled)
    v_top_final = hidden_to_visible_prob(top_rbm, h_prob)

    # Propagate down through remaining layers
    v = v_top_final
    for i in range(len(dbn_params.rbm_layers) - 2, -1, -1):
        v = hidden_to_visible_prob(dbn_params.rbm_layers[i], v)

    return v


# --- Fine-tuning as classifier (optional) ---

class ClassifierParams(NamedTuple):
    """DBN + softmax classifier."""
    dbn: DBNParams
    W_clf: jnp.ndarray  # (n_top_hidden, n_classes)
    b_clf: jnp.ndarray  # (n_classes,)


def init_classifier(dbn_params: DBNParams, n_classes: int, key: jax.Array) -> ClassifierParams:
    """Add a softmax classification layer on top of pretrained DBN."""
    n_top = dbn_params.rbm_layers[-1].W.shape[1]
    W_clf = 0.01 * random.normal(key, (n_top, n_classes))
    b_clf = jnp.zeros(n_classes)
    return ClassifierParams(dbn=dbn_params, W_clf=W_clf, b_clf=b_clf)


def classify(clf_params: ClassifierParams, v: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: propagate through DBN then softmax."""
    h = propagate_up(clf_params.dbn, v)
    logits = h @ clf_params.W_clf + clf_params.b_clf
    return jax.nn.softmax(logits, axis=-1)


def cross_entropy_loss(clf_params: ClassifierParams, v: jnp.ndarray,
                       labels: jnp.ndarray) -> jnp.ndarray:
    """Cross-entropy loss for fine-tuning."""
    probs = classify(clf_params, v)
    one_hot = jax.nn.one_hot(labels, probs.shape[-1])
    return -jnp.mean(jnp.sum(one_hot * jnp.log(probs + 1e-8), axis=-1))


@jax.jit
def finetune_step(clf_params: ClassifierParams, v: jnp.ndarray,
                  labels: jnp.ndarray, lr: float = 0.001):
    """One step of supervised fine-tuning with backprop.

    After unsupervised pretraining, we fine-tune the entire network
    with labeled data. This was the original motivation for DBNs —
    pretraining provides a good initialization for deep networks,
    avoiding the vanishing gradient problem.
    """
    loss, grads = jax.value_and_grad(cross_entropy_loss)(clf_params, v, labels)

    # Update all parameters
    new_rbm_layers = []
    for rbm, g_rbm in zip(clf_params.dbn.rbm_layers, grads.dbn.rbm_layers):
        new_rbm = RBMParams(
            W=rbm.W - lr * g_rbm.W,
            vbias=rbm.vbias - lr * g_rbm.vbias,
            hbias=rbm.hbias - lr * g_rbm.hbias,
        )
        new_rbm_layers.append(new_rbm)

    new_params = ClassifierParams(
        dbn=DBNParams(rbm_layers=new_rbm_layers),
        W_clf=clf_params.W_clf - lr * grads.W_clf,
        b_clf=clf_params.b_clf - lr * grads.b_clf,
    )
    return new_params, loss


def finetune_classifier(
    dbn_params: DBNParams,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int = 10,
    n_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 0,
):
    """Fine-tune the pretrained DBN as a classifier."""
    key = random.PRNGKey(seed)
    key, init_key = random.split(key)

    clf_params = init_classifier(dbn_params, n_classes, init_key)

    print(f"\nFine-tuning classifier for {n_epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        key, perm_key = random.split(key)
        perm = random.permutation(perm_key, train_data.shape[0])
        n_batches = train_data.shape[0] // batch_size
        epoch_loss = 0.0

        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            v = jnp.array(train_data[idx])
            y = jnp.array(train_labels[idx])
            clf_params, loss = finetune_step(clf_params, v, y, lr=lr)
            epoch_loss += loss

        # Evaluate on test set
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            test_probs = classify(clf_params, jnp.array(test_data))
            test_preds = jnp.argmax(test_probs, axis=-1)
            accuracy = jnp.mean(test_preds == jnp.array(test_labels))
            print(f"Epoch {epoch:3d} | Loss: {epoch_loss / n_batches:.4f} | "
                  f"Test Accuracy: {accuracy:.4f}")

    return clf_params
