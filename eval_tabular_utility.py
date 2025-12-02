"""
eval_tabular_utility.py

Utility-метрики для табличной синтетики:
  - TSTR (Train on Synthetic, Test on Real) для регрессии

Ожидается, что у тебя есть pkl-ы с такой структурой:

# для каждой модели отдельно
synthetic_data_dsb.pkl:     {"DSB": {ds_name: {"train_w": ..., "val_w": ...}, ...}}
synthetic_data_ctgan.pkl:   {"CTGAN": {ds_name: {...}, ...}}
synthetic_data_tabddpm.pkl: {"TabDDPM": {ds_name: {...}, ...}}

real_data_dsb.pkl (или любой один общий real_data): 
  {ds_name: {"train_w": ..., "val_w": ...}, ...}

Если real_data у всех моделей одинаковый — достаточно одного pkl.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import pickle
from typing import Dict, Any, List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from catboost import CatBoostRegressor


# --------- 1. Пути к файлам (ПОДРЕДАКТИРУЙ ПОД СЕБЯ) ---------

REAL_DATA_PATH = Path(r"data/real_data_dsb.pkl")


import pickle
from typing import Dict, Any, List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------
# Настройки
# -------------------------------------------------

REAL_PKL_PATH = Path(r"data/real_data_dsb.pkl")

SYN_PKL_PATHS = {
    "DSB": Path(r"data/synthetic_data_dsb (1).pkl"),
    "CTGAN": Path(r"data/synthetic_data_ctgan.pkl"),
    "TabDDPM": Path(r"data/synthetic_data_tabddpm.pkl"),
}

# Какие датасеты смотреть
TARGET_DATASETS = [
    "california_housing",
    "king_county_housing",
    "diabetes",
    "online_news_popularity",
    "adult_numeric",
]

# Сколько объектов использовать для train (и real, и synthetic)
# Если None — используем все
MAX_TRAIN_SAMPLES = 2500
RANDOM_STATE = 42

# По какой колонке строим задачу (target = столбец target_col, признаки = остальные)
# Можно задать маппинг по датасетам, дефолт -1 (последний столбец)


TARGET_COLS = {
    "california_housing": -1,
    "king_county_housing": 1,
    "diabetes": -1,
    "online_news_popularity": -1,
    "adult_numeric": 2,
    # остальные либо добавишь сюда, либо их лучше пока скипать
}


# -------------------------------------------------
# Вспомогательные функции
# -------------------------------------------------

def build_Xy_from_matrix(X: np.ndarray, target_col: int):
    """Разбить матрицу X на (X_features, y) по target_col."""
    X = np.asarray(X, dtype=np.float32)
    d = X.shape[1]

    if target_col < 0:
        target_col = d + target_col

    if not (0 <= target_col < d):
        raise ValueError(f"Bad target_col={target_col} for d={d}")

    y = X[:, target_col].copy()
    X_feat = np.delete(X, target_col, axis=1)
    return X_feat, y


def subsample_Xy(X: np.ndarray, y: np.ndarray, max_samples: int | None, rng: np.random.Generator):
    """Случайно подсэмплировать до max_samples объектов (если max_samples=None — вернуть как есть)."""
    n = X.shape[0]
    if max_samples is None or n <= max_samples:
        return X, y
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx], y[idx]


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """RMSE и R2."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


# -------------------------------------------------
# Основная функция TSTR-оценки
# -------------------------------------------------

def evaluate_tstr_for_dataset(
    ds_name: str,
    real_data_ds: Dict[str, np.ndarray],
    syn_data_all_models: Dict[str, Dict[str, np.ndarray]],
    estimators: Dict[str, Any],
    max_train_samples: int | None = MAX_TRAIN_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Для одного датасета:
    - REAL_TRTR baseline
    - TSTR для каждого генератора: train(synthetic), test(real)
    """

    rng = np.random.default_rng(random_state)

    # 1. real train/test
    X_real_train_full = real_data_ds["train_w"]
    X_real_test_full = real_data_ds["val_w"]

    target_col = TARGET_COLS[ds_name]

    Xr_tr, yr_tr = build_Xy_from_matrix(X_real_train_full, target_col)
    Xr_te, yr_te = build_Xy_from_matrix(X_real_test_full, target_col)

    # 2. Опционально ограничиваем размер train для real
    Xr_tr_sub, yr_tr_sub = subsample_Xy(Xr_tr, yr_tr, max_train_samples, rng)

    # 3. Подготовим синтетику для всех моделей
    syn_Xy: Dict[str, Dict[str, np.ndarray]] = {}
    for model_name, syn_ds_dict in syn_data_all_models.items():
        syn_ds = syn_ds_dict[ds_name]  # {"train_w": ..., "val_w": ...}
        X_syn_tr_full = syn_ds["train_w"]
        Xs_tr, ys_tr = build_Xy_from_matrix(X_syn_tr_full, target_col)
        Xs_tr_sub, ys_tr_sub = subsample_Xy(Xs_tr, ys_tr, max_train_samples, rng)
        syn_Xy[model_name] = {"X_tr": Xs_tr_sub, "y_tr": ys_tr_sub}

    results: Dict[str, Any] = {
        "dataset": ds_name,
        "target_col": target_col,
        "n_real_train_full": int(Xr_tr.shape[0]),
        "n_real_train_used": int(Xr_tr_sub.shape[0]),
        "n_real_test": int(Xr_te.shape[0]),
        "per_estimator": {},
    }

    # 4. Для каждого регрессора считаем baseline и TSTR
    for est_name, est_ctor in estimators.items():
        # baseline REAL_TRTR
        est_real = est_ctor()
        est_real.fit(Xr_tr_sub, yr_tr_sub)
        y_pred_real = est_real.predict(Xr_te)
        metrics_real = evaluate_regression(yr_te, y_pred_real)

        per_model_metrics: Dict[str, Dict[str, float]] = {
            "REAL_TRTR": metrics_real
        }

        # TSTR по каждой модели синтетики
        for model_name, xy_syn in syn_Xy.items():
            est_syn = est_ctor()
            est_syn.fit(xy_syn["X_tr"], xy_syn["y_tr"])
            y_pred_syn = est_syn.predict(Xr_te)
            metrics_syn = evaluate_regression(yr_te, y_pred_syn)

            # можно сразу добавить дельты относительно baseline
            metrics_syn_with_delta = {
                **metrics_syn,
                "delta_rmse": metrics_syn["rmse"] - metrics_real["rmse"],
                "delta_r2": metrics_syn["r2"] - metrics_real["r2"],
            }

            per_model_metrics[f"{model_name}_TSTR"] = metrics_syn_with_delta

        results["per_estimator"][est_name] = per_model_metrics

    return results


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

GEN_ORDER = ["REAL", "DSB", "CTGAN", "TabDDPM"]

def plot_grouped_utility(df: pd.DataFrame, out_dir: Path = Path("utility_plots_grouped")):
    """
    На один датасет:
      - ось X: estimator (LinReg, Ridge, RF, CatBoost)
      - цветом: generator (REAL, DSB, CTGAN, TabDDPM)
      - отдельный график для RMSE и R2
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["rmse", "r2"]

    for ds in sorted(df["dataset"].unique()):
        df_ds = df[df["dataset"] == ds].copy()

        for metric in metrics:
            # сводим: estimator x generator -> значение метрики
            pivot = df_ds.pivot_table(
                index="estimator",
                columns="generator",
                values=metric,
                aggfunc="mean",
            )

            # гарантируем фиксированный порядок колонок
            pivot = pivot.reindex(columns=GEN_ORDER)

            estimators = list(pivot.index)
            if len(estimators) == 0:
                continue

            x = np.arange(len(estimators))
            n_gen = len(GEN_ORDER)
            width = 0.8 / n_gen

            plt.figure(figsize=(10, 4))

            for j, gen in enumerate(GEN_ORDER):
                if gen not in pivot.columns:
                    continue
                vals = pivot[gen].values
                # если вообще нет значений — пропускаем
                if np.all(np.isnan(vals)):
                    continue

                offset = (j - (n_gen - 1) / 2) * width
                plt.bar(
                    x + offset,
                    vals,
                    width,
                    label=gen,
                )

            plt.xticks(x, estimators)
            plt.ylabel(metric.upper())
            plt.title(f"{ds} — {metric.upper()} (REAL vs DSB/CTGAN/TabDDPM)")
            plt.legend(title="train source")
            plt.tight_layout()
            plt.ylim(bottom=0)

            plt.savefig(out_dir / f"{ds}_{metric}_grouped.png", dpi=150)
            plt.close()


def flatten_results_to_df(all_results: Dict[str, Any]) -> pd.DataFrame:
    """
    all_results[ds_name]["per_estimator"][est_name] = {
        "REAL_TRTR": {"rmse":..., "r2":...},
        "DSB_TSTR":  {"rmse":..., "r2":..., "delta_rmse":..., "delta_r2":...},
        ...
    }

    -> DataFrame со строками:
        dataset, estimator, train_source, generator, rmse, r2, delta_rmse, delta_r2
    """
    rows: List[Dict[str, Any]] = []

    for ds_name, ds_res in all_results.items():
        per_est = ds_res["per_estimator"]
        for est_name, metrics_block in per_est.items():
            # базовый REAL_TRTR
            base = metrics_block["REAL_TRTR"]
            rows.append(
                dict(
                    dataset=ds_name,
                    estimator=est_name,
                    train_source="REAL_TRTR",
                    generator="REAL",
                    rmse=base["rmse"],
                    r2=base["r2"],
                    delta_rmse=0.0,
                    delta_r2=0.0,
                )
            )

            # все TSTR
            for key, m in metrics_block.items():
                if key == "REAL_TRTR":
                    continue
                gen_name = key.replace("_TSTR", "")  # "DSB_TSTR" -> "DSB"

                rows.append(
                    dict(
                        dataset=ds_name,
                        estimator=est_name,
                        train_source="TSTR",
                        generator=gen_name,
                        rmse=m["rmse"],
                        r2=m["r2"],
                        delta_rmse=m.get("delta_rmse", np.nan),
                        delta_r2=m.get("delta_r2", np.nan),
                    )
                )

    df = pd.DataFrame(rows)
    return df

import matplotlib.pyplot as plt

def plot_metrics_per_dataset(
    df: pd.DataFrame,
    out_dir: Path = Path("utility_plots"),
):
    """
    Для каждого датасета рисуем barplot'ы:
      - RMSE по (estimator, generator)
      - R2 по (estimator, generator)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in df["dataset"].unique():
        df_ds = df[df["dataset"] == ds].copy()

        # Упорядочим генераторов, чтобы REAL всегда был первым
        gen_order = ["REAL", "DSB", "CTGAN", "TabDDPM"]
        df_ds["generator"] = pd.Categorical(df_ds["generator"], gen_order)

        for metric in ["rmse", "r2"]:
            # сводим по (estimator, generator)
            pivot = (
                df_ds
                .groupby(["estimator", "generator"], observed=True)[metric]
                .mean()
                .reset_index()
                .sort_values(["estimator", "generator"])
            )

            x_labels = [
                f"{est}\n{gen}" for est, gen in zip(pivot["estimator"], pivot["generator"])
            ]
            x = np.arange(len(pivot))

            plt.figure(figsize=(10, 4))
            plt.bar(x, pivot[metric].values)
            plt.xticks(x, x_labels, rotation=45, ha="right")
            plt.ylabel(metric.upper())
            plt.title(f"{ds} — {metric.upper()} (REAL vs TSTR)")
            plt.ylim(bottom=0)

            plt.tight_layout()
            plt.savefig(out_dir / f"{ds}_{metric}.png", dpi=150)
            plt.close()

# -------------------------------------------------
# Обёртка: пройти по всем датасетам и моделям
# -------------------------------------------------

def main():
    # 1. грузим real
    with open(REAL_PKL_PATH, "rb") as f:
        real_data = pickle.load(f)

    # 2. грузим синтетику по трём моделям
    syn_all_models: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for model_name, path in SYN_PKL_PATHS.items():
        with open(path, "rb") as f:
            syn_dict = pickle.load(f)  # {"DSB": {...}} или {"CTGAN": {...}}
        # сверху ключ = имя модели
        if model_name in syn_dict:
            syn_all_models[model_name] = syn_dict[model_name]
        else:
            # на всякий случай, если в pkl сразу {ds_name: {...}}
            syn_all_models[model_name] = syn_dict

    # 3. какие регрессоры использовать
    estimators = {
        # "LinReg": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(
            alpha=0.1,
            solver='auto',
            random_state=RANDOM_STATE,
            max_iter=1000),
        "RF": lambda: RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=RANDOM_STATE,
            n_jobs=-1,),
        "CatBoost": lambda: CatBoostRegressor(
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            border_count=254,
            verbose=False,
            random_seed=RANDOM_STATE,
            allow_writing_files=False, )
    }

    all_results: Dict[str, Any] = {}

    for ds_name in TARGET_DATASETS:
        if ds_name not in real_data:
            print(f"[WARN] dataset '{ds_name}' not found in real_data, skip")
            continue

        # соберём по этому датасету синтетику всех моделей
        syn_data_for_ds: Dict[str, Dict[str, np.ndarray]] = {}
        for model_name, syn_dict in syn_all_models.items():
            if ds_name not in syn_dict:
                print(f"[WARN] dataset '{ds_name}' not found in synthetic_data[{model_name}], skip this model")
                continue
            syn_data_for_ds[model_name] = syn_dict

        if not syn_data_for_ds:
            print(f"[WARN] no synthetic models for dataset '{ds_name}', skip")
            continue

        print(f"\n===== Utility (TSTR) for dataset '{ds_name}' =====")
        res_ds = evaluate_tstr_for_dataset(
            ds_name=ds_name,
            real_data_ds=real_data[ds_name],
            syn_data_all_models=syn_data_for_ds,
            estimators=estimators,
            max_train_samples=MAX_TRAIN_SAMPLES,
            random_state=RANDOM_STATE,
        )
        all_results[ds_name] = res_ds

        # Красиво печатаем
        print(f"  target_col: {res_ds['target_col']}")
        print(f"  n_real_train_full: {res_ds['n_real_train_full']}, "
              f"used: {res_ds['n_real_train_used']}, "
              f"n_real_test: {res_ds['n_real_test']}")

        for est_name, metrics_block in res_ds["per_estimator"].items():
            print(f"\n  Estimator: {est_name}")
            base = metrics_block["REAL_TRTR"]
            print(f"    REAL_TRTR:     RMSE={base['rmse']:.4f}, R2={base['r2']:.4f}")
            for key, m in metrics_block.items():
                if key == "REAL_TRTR":
                    continue
                print(
                    f"    {key:12s}: RMSE={m['rmse']:.4f} "
                    f"(Δ={m['delta_rmse']:+.4f}), "
                    f"R2={m['r2']:.4f} (Δ={m['delta_r2']:+.4f})"
                )

    # Если нужно — сохранить результаты в pkl/json
    with open("utility_tstr_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("\nSaved utility_tstr_results.pkl")

        # --------- агрегируем в табличку и строим графики ---------
    df_summary = flatten_results_to_df(all_results)
    df_summary.to_csv("utility_tstr_summary.csv", index=False)
    print("\n=== Summary table (first rows) ===")
    print(df_summary.head(20))

    plot_grouped_utility(df_summary)
    print("Saved summary CSV and per-dataset plots in ./utility_plots/")



if __name__ == "__main__":
    main()
