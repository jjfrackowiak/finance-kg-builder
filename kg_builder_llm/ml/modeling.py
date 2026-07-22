"""Model training and evaluation."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model evaluation metrics."""

    auc: float
    f1: float
    max_hops_train: int
    max_hops_val: int
    precision: float = 0.0
    recall: float = 0.0
    brier_score: float = 0.0
    n_train_days: int = 0
    n_val_days: int = 0
    n_nodes_total: int = 0
    n_rels_total: int = 0
    feature_importances: Optional[List[float]] = None

    @classmethod
    def empty(cls) -> "ModelMetrics":
        """Zero-valued metrics used as the default for unevaluated candidates."""
        return cls(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)


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
    days: list,
    train_ratio: float = 0.7,
) -> Tuple[list, list]:
    """Temporal train/val split by day (returns indices).

    Args:
        days: List of day dates (strings in YYYY-MM-DD format)
        train_ratio: Fraction for training (default 0.7)

    Returns:
        (train_indices, val_indices) - lists of indices
    """
    sorted_days = sorted(set(days))
    n_train = max(1, int(len(sorted_days) * train_ratio))
    train_days_set = set(sorted_days[:n_train])

    train_idx = [i for i, d in enumerate(days) if d in train_days_set]
    val_idx = [i for i, d in enumerate(days) if d not in train_days_set]

    logger.info(
        "Temporal split on %d unique days: train=%d indices, val=%d indices",
        len(sorted_days),
        len(train_idx),
        len(val_idx),
    )

    return train_idx, val_idx



def train_classifier_on_embeddings(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame = None,
    train_fraction: float = 0.7,
) -> ModelMetrics:
    """Train XGBClassifier on embeddings/features to predict direction (up/down).

    Supports two modes:
    1. Single DataFrame mode (legacy): Receives one DataFrame, performs temporal split
    2. Split DataFrames mode (new): Receives separate train/val DataFrames with features

    Args:
        df_train: Training DataFrame with 'embedding' or 'features' column + 'direction' label
                  OR full dataset for legacy mode
        df_val: Validation DataFrame (if provided, uses split mode)
        train_fraction: Fraction for training in legacy mode (default 0.7)

    Returns:
        ModelMetrics with AUC, F1, n_train, n_val
    """
    logger.info("Training classifier on embedding/feature data")

    # Check if in split mode (new) or legacy mode
    if df_val is not None:
        # Split mode: df_train and df_val are already prepared
        logger.debug("Using split DataFrames mode: train=%d, val=%d", len(df_train), len(df_val))
        train_df = df_train
        val_df = df_val
    else:
        # Legacy mode: perform temporal split on single DataFrame
        if df_train.empty:
            logger.warning("Empty embedding frame, returning zero metrics")
            return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

        logger.debug("Using legacy temporal split mode")
        train_idx, val_idx = temporal_train_val_split(df_train["day"].tolist(), train_fraction)
        train_df = df_train.iloc[train_idx]
        val_df = df_train.iloc[val_idx]

    if len(train_df) == 0 or len(val_df) == 0:
        logger.warning("Insufficient data for train/val split")
        return ModelMetrics(auc=0.0, f1=0.0, max_hops_train=0, max_hops_val=0)

    # Extract features and labels
    # Handle both 'embedding' (legacy) and 'features' columns
    if "embedding" in train_df.columns:
        logger.debug("Extracting embeddings from 'embedding' column")
        X_train = np.vstack(train_df["embedding"].to_list())
        X_val = np.vstack(val_df["embedding"].to_list())
    elif "features" in train_df.columns:
        logger.debug("Extracting feature vectors from 'features' column")
        X_train = np.vstack(train_df["features"].to_list())
        X_val = np.vstack(val_df["features"].to_list())
    else:
        # Assume all columns except 'direction' and 'day' are features
        logger.debug("Using all numeric columns as features")
        feature_cols = [c for c in train_df.columns if c not in ["direction", "day", "node_id"]]
        X_train = train_df[feature_cols].values
        X_val = val_df[feature_cols].values

    y_train = train_df["direction"].values
    y_val = val_df["direction"].values

    logger.info(
        "Training XGBClassifier: train size = %d (features %s), validation size = %d",
        len(train_df),
        X_train.shape,
        len(val_df),
    )
    logger.info(
        "Label distribution — Train: %d positive (%.1f%%), Val: %d positive (%.1f%%)",
        y_train.sum(),
        100 * y_train.mean(),
        y_val.sum(),
        100 * y_val.mean(),
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

    logger.info(
        "Predictions — Mean proba: %.4f, Predicted positive: %d/%d (%.1f%%)",
        y_proba.mean(),
        y_pred.sum(),
        len(y_pred),
        100 * y_pred.mean(),
    )

    # Metrics
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, y_proba)
        brier = brier_score_loss(y_val, y_proba)
    else:
        auc = float("nan")
        brier = float("nan")
    f1 = f1_score(y_val, y_pred, zero_division=0.0)
    prec = precision_score(y_val, y_pred, zero_division=0.0)
    rec = recall_score(y_val, y_pred, zero_division=0.0)

    n_train_days = train_df["day"].nunique() if "day" in train_df.columns else len(train_df)
    n_val_days = val_df["day"].nunique() if "day" in val_df.columns else len(val_df)

    logger.info(
        "Validation metrics — AUC: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f, Brier: %.4f",
        auc, f1, prec, rec, brier,
    )

    return ModelMetrics(
        auc=auc,
        f1=f1,
        precision=prec,
        recall=rec,
        brier_score=brier,
        max_hops_train=len(train_df),
        max_hops_val=len(val_df),
        n_train_days=n_train_days,
        n_val_days=n_val_days,
        feature_importances=model.feature_importances_.tolist(),
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
        max_hops_train=len(X_train),
        max_hops_val=len(X_val),
    )
