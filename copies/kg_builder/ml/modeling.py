"""Model training and evaluation."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""

    auc: float
    f1: float
    n_train: int
    n_val: int


def build_day_embedding_frame(
    df_day_nodes: pd.DataFrame,
    embedding_dict: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build Day embedding frame from nodes and embeddings.

    df_day_nodes must have: ['node_id', 'day', 'direction', 'return_next_day']

    Args:
        df_day_nodes: DataFrame with day node info
        embedding_dict: Dict mapping node_id to embedding vector

    Returns:
        DataFrame with embeddings for day nodes
    """
    logger.info(
        "Building Day embedding frame from %d day nodes and %d embeddings",
        len(df_day_nodes),
        len(embedding_dict),
    )

    rows = []
    missing = 0

    for _, row in df_day_nodes.iterrows():
        node_id = row["node_id"]
        if node_id not in embedding_dict:
            missing += 1
            continue

        emb = embedding_dict[node_id]
        rows.append(
            {
                "node_id": node_id,
                "day": row["day"],
                "direction": int(row["direction"]),
                "return_next_day": float(row["return_next_day"]),
                "embedding": emb,
            }
        )

    df = pd.DataFrame(rows)

    logger.info(
        "Created Day embedding frame: %d rows (%d missing embeddings)",
        len(df),
        missing,
    )
    return df


def temporal_train_val_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple:
    """Temporal split by day: first N_train days for train, remaining for validation.

    Args:
        df: DataFrame with 'day' column
        train_fraction: Fraction of days for training (default 0.7)

    Returns:
        (train_df, val_df)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot perform temporal train/val split.")

    logger.info("Performing temporal train/val split")

    df_sorted = df.sort_values("day").reset_index(drop=True)
    unique_days = sorted(df_sorted["day"].unique().tolist())
    n_days = len(unique_days)

    logger.debug("Found %d unique days in dataset", n_days)

    n_train_days = max(1, int(train_fraction * n_days))
    train_days = set(unique_days[:n_train_days])

    train_df = df_sorted[df_sorted["day"].isin(train_days)].copy()
    val_df = df_sorted[~df_sorted["day"].isin(train_days)].copy()

    logger.info(
        "Temporal split: %d train days (%d rows), %d validation days (%d rows)",
        n_train_days,
        len(train_df),
        n_days - n_train_days,
        len(val_df),
    )

    return train_df, val_df


def train_classifier_on_embeddings(
    df_embed: pd.DataFrame,
    train_fraction: float = 0.7,
) -> ModelMetrics:
    """Train XGBClassifier on HOPE embeddings to predict direction (up/down).

    Args:
        df_embed: DataFrame with embeddings and labels
        train_fraction: Fraction of data for training (default 0.7)

    Returns:
        ModelMetrics with AUC, F1, n_train, n_val
    """
    logger.info("Training classifier on %d embedding rows", len(df_embed))

    if df_embed.empty:
        logger.warning("Empty embedding frame, returning zero metrics")
        return ModelMetrics(auc=0.0, f1=0.0, n_train=0, n_val=0)

    train_df, val_df = temporal_train_val_split(df_embed, train_fraction)

    if len(train_df) == 0 or len(val_df) == 0:
        logger.warning("Insufficient data for train/val split")
        return ModelMetrics(auc=0.0, f1=0.0, n_train=len(train_df), n_val=len(val_df))

    # Expand embeddings to numerical arrays
    logger.debug("Expanding embeddings to matrix form")
    X_train = np.vstack(train_df["embedding"].to_list())
    y_train = train_df["direction"].values

    X_val = np.vstack(val_df["embedding"].to_list())
    y_val = val_df["direction"].values

    logger.info(
        "Training XGBClassifier: train size = %d, validation size = %d",
        len(train_df),
        len(val_df),
    )

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=4,
        verbosity=0,
    )

    model.fit(X_train, y_train)
    logger.debug("Model training completed")

    # Predictions
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, y_proba)
    else:
        auc = float("nan")
    f1 = f1_score(y_val, y_pred)

    logger.info("Validation metrics — AUC: %.4f, F1 Score: %.4f", auc, f1)

    return ModelMetrics(
        auc=auc,
        f1=f1,
        n_train=len(train_df),
        n_val=len(val_df),
    )


def train_classifier(
    embeddings_df: pd.DataFrame,
    labels: Dict[str, int],
) -> Tuple[XGBClassifier, ModelMetrics]:
    """Train an XGBoost classifier (legacy interface).

    Args:
        embeddings_df: DataFrame with embeddings and day labels
        labels: Dict mapping day to label (0 or 1)

    Returns:
        Trained model and metrics
    """
    logger.info("Training classifier on %d samples", len(embeddings_df))

    # Prepare data
    embeddings_df = embeddings_df.copy()
    embeddings_df["label"] = embeddings_df["date"].map(labels)
    embeddings_df = embeddings_df.dropna(subset=["label"])

    # Extract features and labels
    X = embeddings_df[[col for col in embeddings_df.columns if col not in ["date", "label"]]].values
    y = embeddings_df["label"].values

    # Temporal train/val split (70% train, 30% val)
    split_idx = int(len(X) * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info("Temporal split: %d train, %d validation", len(X_train), len(X_val))

    # Train model
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, model.predict(X_val))

    logger.info("Validation metrics — AUC: %.4f, F1 Score: %.4f", auc, f1)

    return model, ModelMetrics(
        auc=auc,
        f1=f1,
        n_train=len(X_train),
        n_val=len(X_val),
    )
