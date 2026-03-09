import json
import logging
import random
import sys
from collections import OrderedDict, deque, namedtuple
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import NDArrays, Scalar, ndarrays_to_parameters
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

import typer

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-8s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("frl_ids")

###############################################################################
# Typer application
###############################################################################
app = typer.Typer(
    name="frl-ids",
    help="Federated Reinforcement Learning IDS — DDQN + Flower FedAvg.",
    add_completion=False,
)

###############################################################################
# Type aliases & named tuples
###############################################################################
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)

###############################################################################
# Device selection
###############################################################################


def get_device() -> torch.device:
    """Return the best available compute device (CUDA > CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu")
        log.info("Using CPU")
    return dev


###############################################################################
# Q-Network
###############################################################################


class QNetwork(nn.Module):
    """
    Fully-connected Q-Network mapping a feature vector to Q-values.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        num_actions: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


###############################################################################
# Replay buffer helpers
###############################################################################


def create_replay_buffer(capacity: int) -> deque:
    """Create a fixed-capacity FIFO replay buffer."""
    return deque(maxlen=capacity)


def push_transition(
    buf: deque,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
) -> None:
    """Append a single SARS'D transition to the replay buffer."""
    buf.append(Transition(state, action, reward, next_state, done))


def sample_transitions(buf: deque, batch_size: int) -> List[Transition]:
    """Uniformly sample a mini-batch of transitions from the buffer."""
    return random.sample(buf, min(batch_size, len(buf)))


###############################################################################
# IDS classification environment
###############################################################################


def create_env(features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Wrap dataset arrays into a dict-based RL environment.

    The environment streams samples sequentially; the agent classifies each
    sample and receives +1 (correct) or -1 (incorrect) as reward.
    """
    return {
        "features": features,
        "labels": labels,
        "num_samples": len(labels),
        "current_idx": 0,
        "indices": np.arange(len(labels)),
    }


def env_reset(env: Dict[str, Any]) -> np.ndarray:
    """Shuffle samples and return the first observation (feature vector)."""
    np.random.shuffle(env["indices"])
    env["current_idx"] = 0
    return env["features"][env["indices"][0]].copy()


def env_step(
    env: Dict[str, Any], action: int
) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """
    Execute one classification step.

    Args:
        env: Environment state dictionary.
        action: Predicted class — 0 (benign) or 1 (malicious).

    Returns:
        (next_state, reward, done, info) tuple.
    """
    idx = env["indices"][env["current_idx"]]
    true_label = int(env["labels"][idx])

    # Symmetric reward: +1 correct, -1 incorrect
    correct = action == true_label
    reward = 1.0 if correct else -1.0
    info = {"true_label": true_label, "predicted": action, "correct": correct}

    env["current_idx"] += 1
    done = env["current_idx"] >= env["num_samples"]

    if done:
        # Terminal: return a zero-vector as dummy next state
        next_state = np.zeros_like(env["features"][0])
    else:
        next_state = env["features"][env["indices"][env["current_idx"]]].copy()

    return next_state, reward, done, info


###############################################################################
# DDQN agent functions
###############################################################################


def select_action(
    online_net: QNetwork,
    state: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    """
    Epsilon-greedy action selection.

    With probability *epsilon* a random action is chosen (exploration);
    otherwise the action with the highest Q-value is selected (exploitation).
    """
    if random.random() < epsilon:
        return random.randint(0, 1)

    with torch.no_grad():
        tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        online_net.eval()
        q_vals = online_net(tensor)
        online_net.train()
        return int(q_vals.argmax(dim=1).item())


def compute_ddqn_loss(
    online_net: QNetwork,
    target_net: QNetwork,
    batch: List[Transition],
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the Double DQN loss (Smooth-L1) for a transition batch.

    DDQN decouples action *selection* (online net) from action *evaluation*
    (target net) to mitigate Q-value overestimation.
    """
    states = torch.as_tensor(
        np.array([t.state for t in batch]), dtype=torch.float32
    ).to(device)
    actions = (
        torch.tensor([t.action for t in batch], dtype=torch.int64)
        .unsqueeze(1)
        .to(device)
    )
    rewards = (
        torch.tensor([t.reward for t in batch], dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )
    next_states = torch.as_tensor(
        np.array([t.next_state for t in batch]), dtype=torch.float32
    ).to(device)
    dones = (
        torch.tensor([float(t.done) for t in batch], dtype=torch.float32)
        .unsqueeze(1)
        .to(device)
    )

    # Q-values for actually taken actions
    current_q = online_net(states).gather(1, actions)

    with torch.no_grad():
        # Action selection via online network; evaluation via target network
        best_actions = online_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, best_actions)
        target_q = rewards + gamma * next_q * (1.0 - dones)

    return F.smooth_l1_loss(current_q, target_q)


def update_target_network(online_net: QNetwork, target_net: QNetwork) -> None:
    """Hard-copy online network parameters to the target network."""
    target_net.load_state_dict(online_net.state_dict())


def get_epsilon(step: int, start: float, end: float, decay: float) -> float:
    """Exponential Epsilon-decay schedule."""
    return end + (start - end) * np.exp(-step / decay)


###############################################################################
# DDQN training loop
###############################################################################


def train_ddqn(
    online_net: QNetwork,
    target_net: QNetwork,
    env: Dict[str, Any],
    buf: deque,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    num_episodes: int,
    batch_size: int,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    target_update_freq: int,
    global_step: int = 0,
    max_steps_per_episode: Optional[int] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Train the DDQN agent for *num_episodes* passes over the environment.

    Returns:
        (metrics_dict, updated_global_step).
    """
    online_net.train()
    total_reward = 0.0
    total_loss = 0.0
    loss_count = 0
    all_true: List[int] = []
    all_pred: List[int] = []

    for ep in range(num_episodes):
        state = env_reset(env)
        ep_reward = 0.0
        ep_steps = 0

        while True:
            epsilon = get_epsilon(global_step, eps_start, eps_end, eps_decay)
            action = select_action(online_net, state, epsilon, device)
            next_state, reward, done, info = env_step(env, action)

            push_transition(buf, state, action, reward, next_state, done)
            all_true.append(info["true_label"])
            all_pred.append(info["predicted"])
            ep_reward += reward
            ep_steps += 1
            global_step += 1

            # Gradient step on a sampled mini-batch
            if len(buf) >= batch_size:
                batch = sample_transitions(buf, batch_size)
                loss = compute_ddqn_loss(online_net, target_net, batch, gamma, device)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                loss_count += 1

            # Periodic hard update of the target network
            if global_step % target_update_freq == 0:
                update_target_network(online_net, target_net)

            state = next_state
            if done:
                break
            if max_steps_per_episode is not None and ep_steps >= max_steps_per_episode:
                break

        total_reward += ep_reward
        log.info(
            "Episode %d/%d | reward=%.0f | Epsilon=%.4f | steps=%d",
            ep + 1,
            num_episodes,
            ep_reward,
            epsilon,
            ep_steps,
        )

    metrics = compute_metrics(np.array(all_true), np.array(all_pred))
    metrics["avg_reward"] = total_reward / max(num_episodes, 1)
    metrics["avg_loss"] = total_loss / max(loss_count, 1)
    metrics["total_steps"] = float(global_step)
    return metrics, global_step


###############################################################################
# Evaluation & metrics
###############################################################################


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return balanced-accuracy, macro-precision, macro-recall, macro-F1."""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_agent(
    net: QNetwork,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    """Greedy (Epsilon=0) evaluation of the Q-network on a full dataset."""
    net.eval()
    preds: List[int] = []
    eval_batch = 4096

    with torch.no_grad():
        for start in range(0, len(features), eval_batch):
            end = min(start + eval_batch, len(features))
            x = torch.as_tensor(features[start:end], dtype=torch.float32).to(device)
            q = net(x)
            preds.extend(q.argmax(dim=1).cpu().tolist())

    return compute_metrics(labels, np.array(preds))


###############################################################################
# NSL-KDD column definitions
###############################################################################

# 41 features defined by the KDD Cup 99 / NSL-KDD schema
NSLKDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Columns that need one-hot encoding
NSLKDD_CATEGORICAL: List[str] = ["protocol_type", "service", "flag"]


###############################################################################
# Data loading & preprocessing  (NSL-KDD)
###############################################################################


def load_dataset(
    file_path: Path,
    label_column: str = "label",
    one_hot_categories: Optional[List[List[str]]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load an NSL-KDD file and return (features, binary_labels).

    NSL-KDD files (KDDTrain+.txt / KDDTest+.txt) are headerless,
    comma-separated, with 41 feature columns + label + difficulty-level
    (the last column is dropped).

    Processing steps:
        1. Apply canonical column names.
        2. Drop the trailing *difficulty* column.
        3. One-hot encode categorical columns (protocol_type, service, flag).
           If *one_hot_categories* is provided the encoder is aligned to those
           exact category lists so train/test frames share the same columns.
        4. Replace ±inf / NaN with 0.
        5. Binarise labels: Epsilon"normal" -> 0, any attack type -> 1.

    Args:
        file_path: Path to the NSL-KDD .txt file.
        label_column: Internal name assigned to the label column.
        one_hot_categories: If supplied, a list of three sorted lists
            Epsilon[protocol_types, services, flags]Epsilon used to align one-hot
            columns (ensures train & test have identical feature sets).

    Returns:
        (features_dataframe, binary_labels_series)

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If the file has an unexpected number of columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    log.info("Loading %s …", file_path)

    # NSL-KDD columns: 41 features + label + difficulty
    col_names = NSLKDD_COLUMNS + [label_column, "difficulty"]
    df = pd.read_csv(file_path, header=None, names=col_names)

    if df.shape[1] != len(col_names):
        raise ValueError(
            f"Expected {len(col_names)} columns, got {df.shape[1]}. "
            f"Check that '{file_path}' is a valid NSL-KDD file."
        )

    # Drop the difficulty-level column (not a feature)
    df = df.drop(columns=["difficulty"])

    # Separate labels
    labels = df[label_column].copy()
    features = df.drop(columns=[label_column])

    # One-hot encode categorical columns
    if one_hot_categories is not None:
        # Use pre-defined category lists to guarantee aligned columns
        for col, cats in zip(NSLKDD_CATEGORICAL, one_hot_categories):
            features[col] = pd.Categorical(features[col], categories=cats)
        features = pd.get_dummies(
            features, columns=NSLKDD_CATEGORICAL, dtype=np.float64
        )
    else:
        features = pd.get_dummies(
            features, columns=NSLKDD_CATEGORICAL, dtype=np.float64
        )

    # Sanitise: replace ±inf with NaN, then fill with 0
    features = features.replace([np.inf, -np.inf], np.nan)
    nan_count = int(features.isna().sum().sum())
    if nan_count:
        log.warning("Replacing %d NaN/inf values with 0", nan_count)
        features = features.fillna(0.0)

    # Binary-label mapping: "normal" -> 0, everything else -> 1
    if labels.dtype == object:
        labels = (labels.str.strip().str.lower() != "normal").astype(np.int64)
    else:
        labels = (labels != 0).astype(np.int64)

    log.info(
        "  -> %d samples, %d features | normal=%d  attack=%d",
        len(labels),
        features.shape[1],
        int((labels == 0).sum()),
        int((labels == 1).sum()),
    )
    return features, labels


def get_one_hot_categories(features_df: pd.DataFrame) -> List[List[str]]:
    """
    Extract sorted unique values for each NSL-KDD categorical column.

    Must be called *before* one-hot encoding so that the raw string values
    are still present in the DataFrame.
    """
    return [sorted(features_df[c].unique().tolist()) for c in NSLKDD_CATEGORICAL]


def _extract_categories_from_file(file_path: Path) -> List[List[str]]:
    """Read an NSL-KDD file just to extract categorical-column categories."""
    col_names = NSLKDD_COLUMNS + ["label", "difficulty"]
    df = pd.read_csv(
        file_path, header=None, names=col_names, usecols=NSLKDD_CATEGORICAL
    )
    return [sorted(df[c].unique().tolist()) for c in NSLKDD_CATEGORICAL]


def merge_categories(*cats_list: List[List[str]]) -> List[List[str]]:
    """Merge multiple category lists so train & test share the same one-hot columns."""
    n_cols = len(NSLKDD_CATEGORICAL)
    merged: List[List[str]] = [[] for _ in range(n_cols)]
    for cats in cats_list:
        for i in range(n_cols):
            merged[i] = sorted(set(merged[i]) | set(cats[i]))
    return merged


def split_train_val(
    features: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified split of arrays into train / validation subsets.

    Returns:
        (train_features, train_labels, val_features, val_labels)
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split = int(len(indices) * (1.0 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    log.info(
        "Train/Val split -> train=%d  val=%d  (val_ratio=%.2f)",
        len(train_idx),
        len(val_idx),
        val_ratio,
    )
    return features[train_idx], labels[train_idx], features[val_idx], labels[val_idx]


def fit_scaler(features: pd.DataFrame) -> StandardScaler:
    """Fit a StandardScaler on the provided feature DataFrame."""
    scaler = StandardScaler()
    scaler.fit(features.values)
    return scaler


def apply_scaler(features: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform a feature DataFrame with an already-fitted scaler."""
    return scaler.transform(features.values).astype(np.float32)


def partition_data(
    features: np.ndarray,
    labels: np.ndarray,
    partition_id: int,
    num_partitions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IID partition: split data into roughly equal shards (deterministic).

    Raises:
        ValueError: If *partition_id* is out of range.
    """
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id must be in [0, {num_partitions - 1}], got {partition_id}"
        )

    indices = np.arange(len(labels))
    rng = np.random.RandomState(42)  # fixed seed for reproducible splits
    rng.shuffle(indices)
    shards = np.array_split(indices, num_partitions)
    sel = shards[partition_id]

    log.info(
        "Partition %d/%d -> %d samples (%.1f%%)",
        partition_id,
        num_partitions,
        len(sel),
        100.0 * len(sel) / len(labels),
    )
    return features[sel], labels[sel]


###############################################################################
# Model I/O
###############################################################################


def get_net_parameters(net: QNetwork) -> NDArrays:
    """Extract network parameters as a list of NumPy arrays."""
    return [v.cpu().numpy() for v in net.state_dict().values()]


def set_net_parameters(net: QNetwork, parameters: NDArrays) -> None:
    """Load a list of NumPy arrays into the network."""
    state_dict = OrderedDict(
        (k, torch.from_numpy(v)) for k, v in zip(net.state_dict().keys(), parameters)
    )
    net.load_state_dict(state_dict, strict=True)


def save_model(
    net: QNetwork, path: Path, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Persist Q-network weights (state_dict only) and optional JSON metadata."""
    torch.save(net.state_dict(), path)
    log.info("Model weights saved -> %s", path)

    if metadata:
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, default=str)
        log.info("Metadata saved -> %s", meta_path)


def load_model(net: QNetwork, path: Path, device: torch.device) -> None:
    """
    Load Q-network weights from a file.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    state_dict = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(state_dict)
    log.info("Model loaded ← %s", path)


###############################################################################
# Flower client  (class required by the NumPyClient interface)
###############################################################################


class IDSClient(fl.client.NumPyClient):
    """
    Flower client wrapping a local DDQN-based IDS agent.

    Each client owns a data partition, a replay buffer, and a pair of
    online / target Q-networks whose weights are synchronised with the
    global model via FedAvg every round.
    """

    def __init__(
        self,
        online_net: QNetwork,
        target_net: QNetwork,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        val_features: np.ndarray,
        val_labels: np.ndarray,
        device: torch.device,
        lr: float,
        num_episodes: int,
        batch_size: int,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: float,
        buffer_capacity: int,
        target_update_freq: int,
        max_steps_per_episode: Optional[int],
    ) -> None:
        self.online_net = online_net
        self.target_net = target_net
        self.device = device
        self.optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)

        self.train_env = create_env(train_features, train_labels)
        self.val_features = val_features
        self.val_labels = val_labels

        self.buf = create_replay_buffer(buffer_capacity)
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.max_steps_per_episode = max_steps_per_episode
        self.global_step = 0

    # --- Flower interface ------------------------------------------------- #

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return current online-network weights."""
        return get_net_parameters(self.online_net)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set online-network weights and sync the target network."""
        set_net_parameters(self.online_net, parameters)
        update_target_network(self.online_net, self.target_net)
        self.target_net.eval()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Local DDQN training round."""
        self.set_parameters(parameters)

        num_ep = int(config.get("num_episodes", self.num_episodes))
        rnd = int(config.get("current_round", 0))
        log.info("─── Round %d: local training (%d episodes) ───", rnd, num_ep)

        metrics, self.global_step = train_ddqn(
            self.online_net,
            self.target_net,
            self.train_env,
            self.buf,
            self.optimizer,
            self.device,
            num_episodes=num_ep,
            batch_size=self.batch_size,
            gamma=self.gamma,
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            eps_decay=self.eps_decay,
            target_update_freq=self.target_update_freq,
            global_step=self.global_step,
            max_steps_per_episode=self.max_steps_per_episode,
        )

        log.info(
            "Round %d results -> bal_acc=%.4f  f1_macro=%.4f  reward=%.1f",
            rnd,
            metrics["balanced_accuracy"],
            metrics["f1_macro"],
            metrics["avg_reward"],
        )
        return (
            get_net_parameters(self.online_net),
            self.train_env["num_samples"],
            {k: float(v) for k, v in metrics.items()},
        )

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on the local validation split."""
        self.set_parameters(parameters)
        m = evaluate_agent(
            self.online_net, self.val_features, self.val_labels, self.device
        )
        log.info(
            "Validation -> bal_acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
            m["balanced_accuracy"],
            m["precision"],
            m["recall"],
            m["f1_macro"],
        )
        # Flower expects (loss, num_samples, metrics); use 1−bal_acc as proxy loss
        return (
            1.0 - m["balanced_accuracy"],
            len(self.val_labels),
            {k: float(v) for k, v in m.items()},
        )


###############################################################################
# Flower server helpers
###############################################################################


def _fit_config_factory(local_episodes: int):
    """Return a Flower on_fit_config_fn that passes round metadata."""

    def fn(server_round: int) -> Dict[str, Scalar]:
        return {"current_round": server_round, "num_episodes": local_episodes}

    return fn


def _weighted_average(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]],
) -> Dict[str, Scalar]:
    """Weighted average of per-client evaluation metrics."""
    total = sum(n for n, _ in metrics_list)
    if total == 0:
        return {}
    keys = metrics_list[0][1].keys()
    return {k: sum(n * m[k] for n, m in metrics_list) / total for k in keys}


def _make_evaluate_fn(
    input_dim: int,
    hidden_dims: List[int],
    dropout: float,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    device: torch.device,
    save_path: Optional[Path],
):
    """
    Build a server-side centralized evaluation function.

    Evaluates the aggregated global model on the test set after each round
    and saves the checkpoint with the best macro-F1 score.
    """
    best_f1: List[float] = [-1.0]  # mutable slot for the closure

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        net = QNetwork(input_dim, hidden_dims, dropout=dropout).to(device)
        set_net_parameters(net, parameters)
        m = evaluate_agent(net, test_features, test_labels, device)

        log.info(
            "Server round %d -> TEST  bal_acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f",
            server_round,
            m["balanced_accuracy"],
            m["precision"],
            m["recall"],
            m["f1_macro"],
        )

        # Persist the best model so far
        if save_path and m["f1_macro"] > best_f1[0]:
            best_f1[0] = m["f1_macro"]
            save_model(
                net,
                save_path,
                metadata={
                    "round": server_round,
                    "metrics": m,
                    "input_dim": input_dim,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                },
            )

        return 1.0 - m["balanced_accuracy"], {k: float(v) for k, v in m.items()}

    return evaluate_fn


###############################################################################
# Parameter validators (called at the start of each CLI command)
###############################################################################


def _check_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise typer.BadParameter(f"{name} must be a positive integer, got {value}")


def _check_positive_float(value: float, name: str) -> None:
    if value <= 0.0:
        raise typer.BadParameter(f"{name} must be positive, got {value}")


def _check_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise typer.BadParameter(f"{name} must be in [0, 1], got {value}")


def _check_unit_interval(value: float, name: str) -> None:
    """Validate that *value* is in [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise typer.BadParameter(f"{name} must be in [0, 1], got {value}")


def _parse_hidden_dims(raw: str) -> List[int]:
    """Parse comma-separated hidden dimensions, e.g. '256,128,64'."""
    try:
        dims = [int(d.strip()) for d in raw.split(",")]
        if not dims or any(d <= 0 for d in dims):
            raise ValueError
        return dims
    except ValueError:
        raise typer.BadParameter(
            f"hidden-dims must be comma-separated positive ints, got '{raw}'"
        )


###############################################################################
# CLI: server
###############################################################################


@app.command()
def server(
    server_address: Annotated[
        str, typer.Option(help="gRPC listen address (host:port).")
    ] = "0.0.0.0:8080",
    num_rounds: Annotated[int, typer.Option(help="Federated training rounds.")] = 10,
    min_fit_clients: Annotated[
        int, typer.Option(help="Min clients required for a training round.")
    ] = 2,
    min_evaluate_clients: Annotated[
        int, typer.Option(help="Min clients required for an evaluation round.")
    ] = 2,
    min_available_clients: Annotated[
        int,
        typer.Option(help="Min clients that must be connected before training starts."),
    ] = 2,
    local_episodes: Annotated[
        int, typer.Option(help="DDQN episodes each client trains per round.")
    ] = 3,
    fraction_fit: Annotated[
        float, typer.Option(help="Fraction of connected clients sampled for training.")
    ] = 1.0,
    fraction_evaluate: Annotated[
        float,
        typer.Option(help="Fraction of connected clients sampled for evaluation."),
    ] = 1.0,
    data_dir: Annotated[
        Optional[Path],
        typer.Option(help="Data directory for server-side test evaluation."),
    ] = None,
    label_column: Annotated[
        str, typer.Option(help="Name of the label column in the CSV files.")
    ] = "label",
    hidden_dims: Annotated[
        str, typer.Option(help="Q-network hidden layers (comma-separated).")
    ] = "256,128,64",
    dropout: Annotated[float, typer.Option(help="Q-network dropout rate.")] = 0.2,
    model_save_path: Annotated[
        Path, typer.Option(help="Where to save the best global model.")
    ] = Path("frl_ids_model.pth"),
) -> None:
    """Launch the Flower federated-learning server (FedAvg strategy)."""

    # Validate
    _check_positive_int(num_rounds, "num-rounds")
    _check_positive_int(min_fit_clients, "min-fit-clients")
    _check_positive_int(min_evaluate_clients, "min-evaluate-clients")
    _check_positive_int(min_available_clients, "min-available-clients")
    _check_positive_int(local_episodes, "local-episodes")
    _check_unit_interval(fraction_fit, "fraction-fit")
    _check_unit_interval(fraction_evaluate, "fraction-evaluate")
    _check_probability(dropout, "dropout")
    h_dims = _parse_hidden_dims(hidden_dims)

    device = get_device()

    # Optional server-side data for centralized evaluation
    evaluate_fn = None
    initial_params: Optional[fl.common.Parameters] = None

    if data_dir is not None:
        dp = Path(data_dir)
        if not dp.is_dir():
            log.error("Data directory does not exist: %s", dp)
            raise typer.Exit(code=1)
        try:
            # Build unified one-hot categories from both files
            train_path = dp / "KDDTrain+.txt"
            test_path = dp / "KDDTest+.txt"
            cats = merge_categories(
                _extract_categories_from_file(train_path),
                _extract_categories_from_file(test_path),
            )

            train_feat, _ = load_dataset(
                train_path, label_column, one_hot_categories=cats
            )
            test_feat, test_lbl = load_dataset(
                test_path, label_column, one_hot_categories=cats
            )

            scaler = fit_scaler(train_feat)
            test_scaled = apply_scaler(test_feat, scaler)
            test_labels_np = test_lbl.values
            input_dim = test_scaled.shape[1]
        except (FileNotFoundError, ValueError) as exc:
            log.error("Server data loading failed: %s", exc)
            raise typer.Exit(code=1)

        # Create centralized evaluate function
        evaluate_fn = _make_evaluate_fn(
            input_dim,
            h_dims,
            dropout,
            test_scaled,
            test_labels_np,
            device,
            model_save_path,
        )

        # Provide initial parameters so clients start from the same point
        init_net = QNetwork(input_dim, h_dims, dropout=dropout).to(device)
        initial_params = ndarrays_to_parameters(get_net_parameters(init_net))
    else:
        log.warning(
            "No --data-dir supplied; server-side evaluation & model saving disabled."
        )

    # FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=_fit_config_factory(local_episodes),
        evaluate_metrics_aggregation_fn=_weighted_average,
        initial_parameters=initial_params,
    )

    log.info(
        "Starting Flower server @ %s  |  rounds=%d  min_clients=%d",
        server_address,
        num_rounds,
        min_available_clients,
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    log.info("Federated training finished.")


###############################################################################
# CLI: client
###############################################################################


@app.command()
def client(
    server_address: Annotated[
        str, typer.Option(help="Flower server address (host:port).")
    ] = "127.0.0.1:8080",
    data_dir: Annotated[
        Path, typer.Option(help="Directory with KDDTrain+.txt / KDDTest+.txt.")
    ] = Path("data"),
    partition_id: Annotated[
        int, typer.Option(help="This client's partition index (0-based).")
    ] = 0,
    num_partitions: Annotated[
        int, typer.Option(help="Total number of data partitions across all clients.")
    ] = 2,
    label_column: Annotated[
        str, typer.Option(help="Name of the label column.")
    ] = "label",
    hidden_dims: Annotated[
        str, typer.Option(help="Q-network hidden layers (comma-separated).")
    ] = "256,128,64",
    dropout: Annotated[float, typer.Option(help="Q-network dropout rate.")] = 0.2,
    learning_rate: Annotated[
        float, typer.Option(help="Adam optimiser learning rate.")
    ] = 1e-3,
    num_episodes: Annotated[
        int, typer.Option(help="DDQN episodes per federated training round.")
    ] = 3,
    batch_size: Annotated[
        int, typer.Option(help="Replay-buffer sampling batch size.")
    ] = 256,
    gamma: Annotated[
        float, typer.Option(help="DDQN discount factor (low for classification tasks).")
    ] = 0.1,
    eps_start: Annotated[
        float, typer.Option(help="Initial Epsilon for Epsilon-greedy exploration.")
    ] = 1.0,
    eps_end: Annotated[
        float, typer.Option(help="Minimum Epsilon (final exploration rate).")
    ] = 0.01,
    eps_decay: Annotated[
        float, typer.Option(help="Exponential Epsilon-decay rate (in steps).")
    ] = 10000.0,
    buffer_capacity: Annotated[
        int, typer.Option(help="Maximum replay-buffer size.")
    ] = 100_000,
    target_update_freq: Annotated[
        int, typer.Option(help="Steps between target-network hard updates.")
    ] = 1000,
    max_steps_per_episode: Annotated[
        Optional[int],
        typer.Option(help="Cap on steps per episode (None -> full dataset)."),
    ] = None,
    val_ratio: Annotated[
        float,
        typer.Option(help="Fraction of training data reserved for local validation."),
    ] = 0.2,
    seed: Annotated[int, typer.Option(help="Random seed for reproducibility.")] = 42,
) -> None:
    """Launch a Flower client running a DDQN-based IDS agent."""

    # Validate
    _check_positive_int(num_partitions, "num-partitions")
    _check_positive_int(num_episodes, "num-episodes")
    _check_positive_int(batch_size, "batch-size")
    _check_positive_int(buffer_capacity, "buffer-capacity")
    _check_positive_int(target_update_freq, "target-update-freq")
    _check_positive_float(learning_rate, "learning-rate")
    _check_positive_float(eps_decay, "eps-decay")
    _check_probability(eps_start, "eps-start")
    _check_probability(eps_end, "eps-end")
    _check_probability(gamma, "gamma")
    _check_probability(dropout, "dropout")
    if partition_id < 0 or partition_id >= num_partitions:
        raise typer.BadParameter(
            f"partition-id must be in [0, {num_partitions - 1}], got {partition_id}"
        )
    if max_steps_per_episode is not None:
        _check_positive_int(max_steps_per_episode, "max-steps-per-episode")
    _check_unit_interval(val_ratio, "val-ratio")
    h_dims = _parse_hidden_dims(hidden_dims)

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = get_device()

    # Load data
    dp = Path(data_dir)
    train_path = dp / "KDDTrain+.txt"
    test_path = dp / "KDDTest+.txt"
    try:
        # Build unified one-hot categories from both files
        cats = merge_categories(
            _extract_categories_from_file(train_path),
            _extract_categories_from_file(test_path),
        )
        train_feat, train_lbl = load_dataset(
            train_path, label_column, one_hot_categories=cats
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("Data loading failed: %s", exc)
        raise typer.Exit(code=1)

    # Fit scaler on full training set so every client normalises consistently
    scaler = fit_scaler(train_feat)
    train_scaled = apply_scaler(train_feat, scaler)
    train_labels_np = train_lbl.values

    # NSL-KDD has no separate validation file -> split from training data
    train_scaled, train_labels_np, val_scaled, val_labels_np = split_train_val(
        train_scaled,
        train_labels_np,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Partition training data for this client
    part_features, part_labels = partition_data(
        train_scaled, train_labels_np, partition_id, num_partitions
    )

    input_dim = part_features.shape[1]
    log.info(
        "Input dim=%d | Hidden dims=%s | Partition samples=%d",
        input_dim,
        h_dims,
        len(part_labels),
    )

    # Build DDQN networks
    online_net = QNetwork(input_dim, h_dims, dropout=dropout).to(device)
    target_net = QNetwork(input_dim, h_dims, dropout=dropout).to(device)
    update_target_network(online_net, target_net)
    target_net.eval()

    # Flower client
    fl_client = IDSClient(
        online_net=online_net,
        target_net=target_net,
        train_features=part_features,
        train_labels=part_labels,
        val_features=val_scaled,
        val_labels=val_labels_np,
        device=device,
        lr=learning_rate,
        num_episodes=num_episodes,
        batch_size=batch_size,
        gamma=gamma,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay=eps_decay,
        buffer_capacity=buffer_capacity,
        target_update_freq=target_update_freq,
        max_steps_per_episode=max_steps_per_episode,
    )

    log.info(
        "Connecting to %s  (partition %d/%d) …",
        server_address,
        partition_id,
        num_partitions,
    )
    fl.client.start_client(
        server_address=server_address,
        client=fl_client.to_client(),
    )
    log.info("Client %d finished.", partition_id)


###############################################################################
# CLI: evaluate
###############################################################################


@app.command()
def evaluate(
    model_path: Annotated[
        Path, typer.Option(help="Path to the saved model (.pth).")
    ] = Path("frl_ids_model.pth"),
    data_dir: Annotated[
        Path,
        typer.Option(
            help="Directory with KDDTrain+.txt (for scaler) and KDDTest+.txt."
        ),
    ] = Path("data"),
    label_column: Annotated[
        str, typer.Option(help="Name of the label column.")
    ] = "label",
    hidden_dims: Annotated[
        str, typer.Option(help="Q-network hidden layers (must match training config).")
    ] = "256,128,64",
    dropout: Annotated[
        float, typer.Option(help="Q-network dropout (must match training config).")
    ] = 0.2,
) -> None:
    """Evaluate a saved FRL-IDS model on the test set and print metrics."""

    h_dims = _parse_hidden_dims(hidden_dims)
    _check_probability(dropout, "dropout")
    dp = Path(data_dir)
    device = get_device()

    # If a metadata sidecar exists, try to read hidden_dims / dropout from it
    meta_path = model_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            saved_dims = meta.get("hidden_dims")
            saved_drop = meta.get("dropout")
            if saved_dims is not None:
                h_dims = [int(d) for d in saved_dims]
                log.info("Using hidden_dims from metadata: %s", h_dims)
            if saved_drop is not None:
                dropout = float(saved_drop)
                log.info("Using dropout from metadata: %s", dropout)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            log.warning("Could not parse metadata file: %s", exc)

    # Load data
    try:
        train_path = dp / "KDDTrain+.txt"
        test_path = dp / "KDDTest+.txt"
        cats = merge_categories(
            _extract_categories_from_file(train_path),
            _extract_categories_from_file(test_path),
        )
        train_feat, _ = load_dataset(train_path, label_column, one_hot_categories=cats)
        test_feat, test_lbl = load_dataset(
            test_path, label_column, one_hot_categories=cats
        )
    except (FileNotFoundError, ValueError) as exc:
        log.error("Data loading failed: %s", exc)
        raise typer.Exit(code=1)

    scaler = fit_scaler(train_feat)
    test_scaled = apply_scaler(test_feat, scaler)
    test_labels_np = test_lbl.values
    input_dim = test_scaled.shape[1]

    # Load model & evaluate
    net = QNetwork(input_dim, h_dims, dropout=dropout).to(device)
    try:
        load_model(net, model_path, device)
    except FileNotFoundError as exc:
        log.error(str(exc))
        raise typer.Exit(code=1)

    m = evaluate_agent(net, test_scaled, test_labels_np, device)

    # Display results
    divider = "=" * 62
    typer.echo(f"\n{divider}")
    typer.echo("  FRL-IDS — Test-Set Evaluation Results")
    typer.echo(divider)
    typer.echo(f"  Model path       : {model_path}")
    typer.echo(f"  Test samples     : {len(test_labels_np)}")
    typer.echo(f"  Hidden dims      : {h_dims}")
    typer.echo(f"  Dropout          : {dropout}")
    typer.echo(f"  ")
    typer.echo(f"  Balanced Accuracy : {m['balanced_accuracy']:.4f}")
    typer.echo(f"  Precision (macro) : {m['precision']:.4f}")
    typer.echo(f"  Recall    (macro) : {m['recall']:.4f}")
    typer.echo(f"  F1-Score  (macro) : {m['f1_macro']:.4f}")
    typer.echo(f"{divider}\n")


###############################################################################
# Entry point
###############################################################################

if __name__ == "__main__":
    app()
