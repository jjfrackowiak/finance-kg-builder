# file: finance_kg_experiment/modeling.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

from config import ExperimentConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    auc: float
    f1: float
    n_train: int
    n_val: int


def build_day_embedding_frame(
    df_day_nodes: pd.DataFrame,
    embedding_dict: Dict[int, np.ndarray],
) -> pd.DataFrame:
    """
    df_day_nodes must have: ['node_id', 'day', 'direction', 'return_next_day']
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
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split by day: first N_train days for train, remaining for validation.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty. Cannot perform temporal train/val split.")

    logger.info("Performing temporal train/val split")
    
    df_sorted = df.sort_values("day").reset_index(drop=True)
    unique_days = sorted(df_sorted["day"].unique().tolist())
    n_days = len(unique_days)

    logger.debug("Found %d unique days in dataset", n_days)

    n_train_days = max(1, int(config.train_fraction * n_days))
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
    config: ExperimentConfig,
) -> ModelMetrics:
    """
    Train XGBClassifier on HOPE embeddings to predict direction (up/down).
    """
    logger.info("Training classifier on %d embedding rows", len(df_embed))

    train_df, val_df = temporal_train_val_split(df_embed, config)

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
        random_state=config.random_state,
        n_jobs=4,
    )

    model.fit(X_train, y_train)
    logger.debug("Model training completed")

    # Predictions
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_val, y_proba) if len(np.unique(y_val)) > 1 else float("nan")
    f1 = f1_score(y_val, y_pred)

    logger.info("Validation metrics — AUC: %.4f, F1 Score: %.4f", auc, f1)

    return ModelMetrics(
        auc=auc,
        f1=f1,
        n_train=len(train_df),
        n_val=len(val_df),
    )
