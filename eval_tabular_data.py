"""
eval_tabular_data.py

Метрики качества табличной синтетики:

1) Distribution fidelity (по числовым и категориальным фичам):
   - Wasserstein distance по числовым колонкам
   - KS-statistics per feature
   - Jensen–Shannon divergence по категориальным колонкам

2) Dependence structure:
   - L2 (Фробениус) разность корреляционных матриц (TabDDPM-style)
   - Средняя абсолютная ошибка по off-diagonal корреляциям (Tabsyn-style)

3) Discriminator-based (ЗАКОММЕНТИРОВАНО НА БУДУЩЕЕ):
   - RandomForest-классификатор real vs synthetic
   - accuracy, ROC-AUC и метрика indistinguishability (1 при acc=0.5)

4) Privacy (ЗАКОММЕНТИРОВАНО НА БУДУЩЕЕ):
   - Distance to Closest Record (DCR)
   - Distance-based membership inference (train vs val real)

План использования:

    from eval_tabular_data import FidelityMetrics, evaluate_fidelity_privacy

    fp = evaluate_fidelity_privacy(
        X_real_train=X_val_w,
        X_syn_train=syn_val_w,
    )

    # fp.fidelity_uni["wasserstein"]["mean"]
    # fp.fidelity_uni["ks"]["mean_ks"]
    # fp.fidelity_dep["corr_frobenius"]
    # fp.fidelity_dep["corr_mae_offdiag"]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Sequence
import pickle
from dsb_tabular import sliced_wasserstein, mmd_rbf

import numpy as np
import pandas as pd
from scipy import stats

# Закомментировано на будущее (discriminator / privacy)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import NearestNeighbors


# =========================
#  Univariate / Distribution fidelity
# =========================

def wasserstein_per_feature(
    X_real: np.ndarray,
    X_syn: np.ndarray,
) -> Dict[str, Any]:
    """
    Wasserstein distance по каждому числовому признаку.

    Args:
        X_real: (n_real, d)
        X_syn:  (n_syn,  d)

    Returns:
        {
          "per_feature": [w_0, ..., w_{d-1}],
          "mean": float,
          "max": float,
        }
    """
    if X_real.shape[1] != X_syn.shape[1]:
        raise ValueError("X_real and X_syn must have the same number of features")

    d = X_real.shape[1]
    vals: List[float] = []

    for j in range(d):
        w = stats.wasserstein_distance(X_real[:, j], X_syn[:, j])
        vals.append(float(w))

    return {
        "per_feature": vals,
        "mean": float(np.mean(vals)) if vals else 0.0,
        "max": float(np.max(vals)) if vals else 0.0,
    }


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen–Shannon divergence между двумя дискретными распределениями p и q.
    p, q — неотрицательные массивы, суммирование не обязательно = 1.
    Возвращает JS в битах (base=2).
    """
    p = p.astype(float)
    q = q.astype(float)

    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        # тривиальные случаи — полная разность
        return 1.0

    p = p / (p_sum + eps)
    q = q / (q_sum + eps)
    m = 0.5 * (p + q)

    kl_pm = stats.entropy(p, m, base=2)
    kl_qm = stats.entropy(q, m, base=2)
    return float(0.5 * (kl_pm + kl_qm))


def js_per_categorical(
    col_real: np.ndarray,
    col_syn: np.ndarray,
) -> float:
    """
    JS-divergence для одной категориальной колонки.
    col_* — одномерные массивы с категориальными значениями (строки/инты).
    """
    vals = np.unique(np.concatenate([col_real, col_syn]))
    real_counts = np.array([(col_real == v).sum() for v in vals], dtype=float)
    syn_counts = np.array([(col_syn == v).sum() for v in vals], dtype=float)
    return js_divergence(real_counts, syn_counts)


def js_per_categorical_features(
    df_real: pd.DataFrame,
    df_syn: pd.DataFrame,
    cat_cols: Sequence[str],
) -> Dict[str, Any]:
    """
    JS-divergence по набору категориальных колонок.

    Args:
        df_real: DataFrame с real-данными (до one-hot)
        df_syn:  DataFrame с synthetic-данными (та же структура)
        cat_cols: список имён категориальных колонок

    Returns:
        {
          "per_feature": [{"col": name, "js": value}, ...],
          "mean": float | None,
          "max": float | None,
        }
    """
    vals: List[Dict[str, Any]] = []
    for col in cat_cols:
        if col not in df_real.columns or col not in df_syn.columns:
            continue
        js = js_per_categorical(
            df_real[col].to_numpy(),
            df_syn[col].to_numpy(),
        )
        vals.append({"col": col, "js": float(js)})

    if not vals:
        return {"per_feature": [], "mean": None, "max": None}

    js_list = [v["js"] for v in vals]
    return {
        "per_feature": vals,
        "mean": float(np.mean(js_list)),
        "max": float(np.max(js_list)),
    }


def ks_per_feature(
    X_real: np.ndarray,
    X_syn: np.ndarray,
) -> Dict[str, Any]:
    """
    KS-statistics per feature (Kolmogorov–Smirnov).

    Args:
        X_real: (n_real, d)
        X_syn:  (n_syn,  d)

    Returns:
        {
          "per_feature": [{"feature": j, "ks": ..., "p": ...}, ...],
          "mean_ks": float,
          "max_ks": float,
        }
    """
    if X_real.shape[1] != X_syn.shape[1]:
        raise ValueError("X_real and X_syn must have the same number of features")

    d = X_real.shape[1]
    stats_list: List[Dict[str, Any]] = []

    for j in range(d):
        stat, p = stats.ks_2samp(X_real[:, j], X_syn[:, j])
        stats_list.append(
            {"feature": int(j), "ks": float(stat), "p": float(p)}
        )

    mean_ks = float(np.mean([x["ks"] for x in stats_list])) if stats_list else 0.0
    max_ks = float(np.max([x["ks"] for x in stats_list])) if stats_list else 0.0

    return {
        "per_feature": stats_list,
        "mean_ks": mean_ks,
        "max_ks": max_ks,
    }


# =========================
#  Dependence structure
# =========================

def correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Корреляционная матрица (Pearson) для признаков.
    X: (n, d), возвращает (d, d).
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    return np.corrcoef(X, rowvar=False)


def dependence_fidelity(
    X_real: np.ndarray,
    X_syn: np.ndarray,
) -> Dict[str, float]:
    """
    Метрики по структуре зависимостей:

    - corr_frobenius: Фробениусова норма разности корреляционных матриц
    - corr_mae_offdiag: средняя абсолютная ошибка по off-diagonal элементам
    """
    C_real = correlation_matrix(X_real)
    C_syn = correlation_matrix(X_syn)

    if C_real.shape != C_syn.shape:
        raise ValueError("Correlation matrices must have the same shape")

    diff = C_real - C_syn
    frob = float(np.linalg.norm(diff, ord="fro"))

    mask_offdiag = ~np.eye(diff.shape[0], dtype=bool)
    mae_offdiag = float(np.mean(np.abs(diff[mask_offdiag])))

    return {
        "corr_frobenius": frob,
        "corr_mae_offdiag": mae_offdiag,
    }


# =========================
#  High-level helper
# =========================

@dataclass
class FidelityMetrics:
    """
    Обёртка для удобного хранения блоков метрик распределения.
    """

    fidelity_uni: Dict[str, Any]
    fidelity_dep: Dict[str, float]
    # discriminator: Dict[str, float]  # reserved for future
    # privacy: Dict[str, Any]          # reserved for future


def evaluate_fidelity_privacy(
    X_real_train: np.ndarray,
    X_syn_train: np.ndarray,
    X_real_val: Optional[np.ndarray] = None,  # пока не используем, оставлено на будущее
    df_real_cat: Optional[pd.DataFrame] = None,
    df_syn_cat: Optional[pd.DataFrame] = None,
    cat_cols: Optional[Sequence[str]] = None,
    random_state: int = 0,  # пока не используем (для будущего RF / privacy)
) -> FidelityMetrics:
    """
    Комплексно считает метрики распределения (univariate + dependence)
    по одному датасету и одному генератору.

    Обязательные аргументы:
        X_real_train: real train (n_real, d)
        X_syn_train:  synthetic train (n_syn, d)

    Необязательные:
        df_real_cat, df_syn_cat, cat_cols: если нужен JS по категориальным фичам.
    """

    # --- Univariate fidelity (numeric) ---
    wd = wasserstein_per_feature(X_real_train, X_syn_train)
    ks = ks_per_feature(X_real_train, X_syn_train)

    uni_block: Dict[str, Any] = {
        "wasserstein": wd,
        "ks": ks,
    }

    # --- Univariate fidelity (categorical, опционально) ---
    if (
        df_real_cat is not None
        and df_syn_cat is not None
        and cat_cols is not None
        and len(cat_cols) > 0
    ):
        js = js_per_categorical_features(df_real_cat, df_syn_cat, cat_cols)
        uni_block["js_categorical"] = js
    else:
        uni_block["js_categorical"] = {
            "per_feature": [],
            "mean": None,
            "max": None,
        }

    # --- Dependence structure ---
    dep_block = dependence_fidelity(X_real_train, X_syn_train)

    return FidelityMetrics(
        fidelity_uni=uni_block,
        fidelity_dep=dep_block,
    )


# =========================
#  Geometry + агрегирование по датасетам/моделям
# =========================

def _subsample_for_metrics(
    X_real: np.ndarray,
    X_syn: np.ndarray,
    max_eval_samples: int = 2500,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Подвыборка real/syn до одинакового размера <= max_eval_samples.
    """
    X_real = np.asarray(X_real, dtype=np.float32)
    X_syn = np.asarray(X_syn, dtype=np.float32)

    n_real = X_real.shape[0]
    n_syn = X_syn.shape[0]
    n = min(n_real, n_syn, max_eval_samples)

    rng = np.random.RandomState(random_state)

    if n_real > n:
        idx_r = rng.choice(n_real, size=n, replace=False)
        X_real_sub = X_real[idx_r]
    else:
        X_real_sub = X_real

    if n_syn > n:
        idx_s = rng.choice(n_syn, size=n, replace=False)
        X_syn_sub = X_syn[idx_s]
    else:
        X_syn_sub = X_syn

    return X_real_sub, X_syn_sub


def evaluate_models_on_datasets(
    real_data: Dict[str, Dict[str, np.ndarray]],
    synthetic_data: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    n_proj: int = 256,
    sigma: float = 1.0,
    max_eval_samples: int = 2500,
    random_state: int = 0,
    pkl_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Высокоуровневая обёртка: считает SWD/MMD + FidelityMetrics
    для всех моделей и датасетов.

    real_data:
      ds_name -> {
        "train_w": np.ndarray,
        "val_w":   np.ndarray | None,
      }

    synthetic_data:
      model_name -> {
        ds_name -> {
          "train_w": np.ndarray,
          "val_w":   np.ndarray | None,
        }
      }
    """
    metrics_all: Dict[str, Dict[str, Any]] = {}

    for model_name, model_dict in synthetic_data.items():
        model_metrics: Dict[str, Any] = {}

        for ds_name, syn_dict in model_dict.items():
            if ds_name not in real_data:
                # нет real-данных для этого датасета — пропускаем
                continue

            real_dict = real_data[ds_name]

            # Берём val если есть, иначе train
            X_real = real_dict.get("val_w")
            if X_real is None:
                X_real = real_dict["train_w"]

            X_syn = syn_dict.get("val_w")
            if X_syn is None:
                X_syn = syn_dict["train_w"]

            # Подвыборка до max_eval_samples
            X_real_eval, X_syn_eval = _subsample_for_metrics(
                X_real, X_syn, max_eval_samples=max_eval_samples, random_state=random_state
            )

            # Геометрия: SWD + MMD
            swd = sliced_wasserstein(
                X_real_eval,
                X_syn_eval,
                n_proj=n_proj,
            )
            mmd = mmd_rbf(
                X_real_eval,
                X_syn_eval,
                sigma=sigma,
            )

            # Распределение + зависимости (наш FidelityMetrics)
            fp = evaluate_fidelity_privacy(
                X_real_train=X_real_eval,
                X_syn_train=X_syn_eval,
                random_state=random_state,
            )

            model_metrics[ds_name] = {
                "n_real": int(X_real_eval.shape[0]),
                "n_syn": int(X_syn_eval.shape[0]),
                "dim": int(X_real_eval.shape[1]),
                "swd": float(swd),
                "mmd": float(mmd),
                "fidelity_uni": fp.fidelity_uni,
                "fidelity_dep": fp.fidelity_dep,
            }

        metrics_all[model_name] = model_metrics

    if pkl_path is not None:
        with open(pkl_path, "wb") as f:
            pickle.dump(metrics_all, f)

    return metrics_all
