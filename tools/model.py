import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import tensorflow as tf
import pickle
from typing import Dict, List, Tuple
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

# model saving/loading utilities
def save_model(state: TrainState, targets: List[str], model_path: str, 
               training_config: Dict = None, final_metrics: Dict = None) -> None:
    """Save trained model to disk."""
    model_data = {
        'params': state.params,
        'num_targets': len(targets),
        'targets': targets,
        'model_config': {
            'dim': 256,
            'num_targets': len(targets)
        },
        'training_config': training_config or {},
        'final_metrics': final_metrics or {}
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to: {model_path}")

def load_model(model_path: str) -> Tuple[Model, TrainState, Dict]:
    """Load a trained model from disk."""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # create model with saved config
    model = Model(num_targets=model_data['num_targets'])
    
    # create dummy state to initialize
    dummy_input = jnp.zeros((1, 640))  # ESM2-150M embedding size
    rng = jax.random.PRNGKey(42)
    state = model.create_train_state(rng, dummy_input, optax.adam(0.001))
    
    # replace with saved parameters
    state = state.replace(params=model_data['params'])
    
    return model, state, model_data

def predict_protein_functions(
    protein_sequences: List[str], 
    model_path: str,
    model_name: str = "facebook/esm2_t30_150M_UR50D"
) -> pd.DataFrame:
    """Predict protein functions for new sequences using a saved model."""
    from .data import get_mean_embeddings
    from transformers import AutoTokenizer, EsmModel
    from .utils import get_device
    
    # load trained model
    model, state, model_data = load_model(model_path)
    targets = model_data['targets']
    
    # load ESM2 for embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    esm_model = EsmModel.from_pretrained(model_name)
    device = get_device()
    
    # generate embeddings
    embeddings = get_mean_embeddings(protein_sequences, tokenizer, esm_model, device)
    
    # make predictions
    logits = state.apply_fn({"params": state.params}, x=embeddings)
    probabilities = jax.nn.sigmoid(logits)
    
    # create results DataFrame
    results = pd.DataFrame(probabilities, columns=targets)
    results['sequence'] = protein_sequences
    results['sequence_length'] = [len(seq) for seq in protein_sequences]
    
    return results

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
    
    # convert to numpy arrays if needed
    targets = np.asarray(targets)
    probs = np.asarray(probs)
    
    # compute predictions
    predictions = (probs >= thresh).astype(int)
    
    # compute metrics
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
    
    # calculate per-target metrics
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
    # create containers to handle calculated during training and evaluation.
    train_metrics, valid_metrics = [], []

    # create batched dataset to pluck batches from for each step.
    train_batches = (
        dataset_splits["train"]
        .batch(batch_size, drop_remainder=True)
        .as_numpy_iterator()
    )

    steps = tqdm(range(num_steps))  # steps with progress bar.
    for step in steps:
        steps.set_description(f"Step {step + 1}")

        # get batch of training data, convert into a JAX array, and train.
        state, loss = train_step(state, next(train_batches))
        train_metrics.append({"step": step, "loss": float(loss)})

        if step % eval_every == 0:
            # for all the evaluation batches, calculate metrics.
            eval_metrics = []
            for eval_batch in (
                dataset_splits["valid"].batch(batch_size=batch_size).as_numpy_iterator()
            ):
                eval_metrics.append(eval_step(state, eval_batch))
            valid_metrics.append(
                {"step": step, **pd.DataFrame(eval_metrics).mean(axis=0).to_dict()}
            )

    return state, {"train": train_metrics, "valid": valid_metrics}