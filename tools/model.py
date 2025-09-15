import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tensorflow as tf
from typing import Dict
from sklearn import metrics
from tqdm import tqdm
from flax.training.train_state import TrainState

# model
class Model(nn.Module):
    """Simple MLP for protein function prediction."""
    
    num_targets: int
    dim: int = 256

    @nn.compact
    def __call__(self, x):
        """Apply MLP layers to input features."""
        x = nn.Sequential([
            nn.Dense(self.dim * 2),
            jax.nn.gelu,
            nn.Dense(self.dim),
            jax.nn.gelu,
            nn.Dense(self.num_targets),
        ])(x)
        return x

    def create_train_state(self, rng: jax.Array, dummy_input, tx) -> TrainState:
        """Initialize model parameters and return a training state."""
        variables = self.init(rng, dummy_input)
        return TrainState.create(
            apply_fn=self.apply, params=variables["params"], tx=tx
        )

# training utilities
def convert_to_tfds(
    df: pd.DataFrame,
    embeddings_prefix: str = "ME:",
    target_prefix: str = "GO:",
    is_training: bool = False,
    shuffle_buffer: int = 50,
) -> tf.data.Dataset:
    """Convert embedding DataFrame into a TensorFlow dataset."""
    dataset = tf.data.Dataset.from_tensor_slices({
        "embedding": df.filter(regex=f"^{embeddings_prefix}").to_numpy(),
        "target": df.filter(regex=f"^{target_prefix}").to_numpy(),
    })
    if is_training:
        dataset = dataset.shuffle(shuffle_buffer).repeat()
    return dataset

def compute_metrics(
    targets: np.ndarray, probs: np.ndarray, thresh=0.5
) -> Dict[str, float]:
    """Compute accuracy, recall, precision, auPRC, and auROC."""
    if np.sum(targets) == 0:
        return {
            m: 0.0 for m in ["accuracy", "recall", "precision", "auprc", "auroc"]
        }
    
    # Convert to numpy arrays if needed
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    
    # Compute predictions
    predictions = (probs >= thresh).astype(int)
    
    # Compute metrics
    accuracy = metrics.accuracy_score(targets, predictions)
    recall = metrics.recall_score(targets, predictions, zero_division=0.0)
    precision = metrics.precision_score(targets, predictions, zero_division=0.0)
    auprc = metrics.average_precision_score(targets, probs)
    auroc = metrics.roc_auc_score(targets, probs)
    
    return {
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "auprc": float(auprc),
        "auroc": float(auroc),
    }

@jax.jit
def train_step(state, batch):
    """Run a single training step and update model parameters."""
    
    def calculate_loss(params):
        """Compute sigmoid cross-entropy loss from logits."""
        logits = state.apply_fn({"params": params}, x=batch["embedding"])
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
        return loss

    grad_fn = jax.value_and_grad(calculate_loss, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def eval_step(state, batch) -> Dict[str, float]:
    """Run evaluation step and return mean metrics over targets."""
    logits = state.apply_fn({"params": state.params}, x=batch["embedding"])
    loss = optax.sigmoid_binary_cross_entropy(logits, batch["target"]).mean()
    
    # Calculate per-target metrics
    probs = jax.nn.sigmoid(logits)
    target_metrics = []
    for target, prob in zip(batch["target"], probs):
        target_metrics.append(compute_metrics(target, prob))
    
    metrics_dict = {
        "loss": float(loss),
        **pd.DataFrame(target_metrics).mean(axis=0).to_dict(),
    }
    return metrics_dict

def train(
    state: TrainState,
    dataset_splits: Dict[str, tf.data.Dataset],
    batch_size: int,
    num_steps: int = 300,
    eval_every: int = 30,
):
    """Train model using batched TF datasets and track performance metrics."""
    # Create containers to handle calculated during training and evaluation.
    train_metrics, valid_metrics = [], []

    # Create batched dataset to pluck batches from for each step.
    train_batches = (
        dataset_splits["train"]
        .batch(batch_size, drop_remainder=True)
        .as_numpy_iterator()
    )

    steps = tqdm(range(num_steps))  # Steps with progress bar.
    for step in steps:
        steps.set_description(f"Step {step + 1}")

        # Get batch of training data, convert into a JAX array, and train.
        state, loss = train_step(state, next(train_batches))
        train_metrics.append({"step": step, "loss": float(loss)})

        if step % eval_every == 0:
            # For all the evaluation batches, calculate metrics.
            eval_metrics = []
            for eval_batch in (
                dataset_splits["valid"].batch(batch_size=batch_size).as_numpy_iterator()
            ):
                eval_metrics.append(eval_step(state, eval_batch))
            valid_metrics.append(
                {"step": step, **pd.DataFrame(eval_metrics).mean(axis=0).to_dict()}
            )

    return state, {"train": train_metrics, "valid": valid_metrics}