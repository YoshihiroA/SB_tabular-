"""
run_dsb_and_prepare.py

Бенчмарк DSBTabularBridge на наборе числовых датасетов
из datasets_numeric.pkl и подготовка данных для eval_tabular_generators.py.

Источник данных:
  - /kaggle/input/dsb-nss-datasets/datasets_numeric.pkl

Форматы:

1) all_results: dict[dataset_name] -> {
      "name": str,
      "bridge": DSBTabularBridge,
      "metrics": {
          "train": { "swd", "mmd", "ks" },
          "val":   { "swd", "mmd", "ks" },
      },
      "syn": {
          "train_w": np.ndarray,
          "val_w":   np.ndarray | None,
      },
   }

2) real_data_dsb.pkl:

   real_data: {
       dataset_name: {
           "train_w": np.ndarray [n_train, D],
           "val_w":   np.ndarray [n_val, D] или None,
       },
       ...
   }

3) synthetic_data_dsb.pkl:

   synthetic_data: {
       "DSB": {
           dataset_name: {
               "train_w": np.ndarray [n_train_syn, D],
               "val_w":   np.ndarray [n_val_syn, D] или None,
           },
           ...
       }
   }
"""


from typing import Dict, Any, Optional, Tuple

from dsb_tabular import DSBTabularBridge, set_seed


import numpy as np
import pandas as pd
import pickle
from dsb_tabular import set_seed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# from dsb_tabular import DSBTabularBridge, set_seed

# ---- путь к pkl с датасетами ----
DATASETS_NUMERIC_PATH = "E:\projects\Diffusion_Sredinger_Bridge\datasets_numeric.pkl"

# None = использовать все датасеты из pkl
TARGET_DATASETS: Optional[list[str]] = None


# ================== helpers: загрузка и препроцесс ================== #

def load_datasets_numeric_from_pkl(
    path: str = DATASETS_NUMERIC_PATH,
) -> Dict[str, np.ndarray]:
    """
    Загружает словарь {name -> X} из pkl и приводит X к np.ndarray float32.
    """
    with open(path, "rb") as f:
        raw = pickle.load(f)

    data: Dict[str, np.ndarray] = {}
    for name, X in raw.items():
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float32)
        data[name] = X

    return data


def prepare_numpy_X(
    X: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    use_pca: bool = False,
    pca_whiten: bool = True,
    pca_components=None,
):
    """
    Для чисто числовых X (np.ndarray):
      - train/val split
      - StandardScaler
      - опционально PCA(whiten=True)
    Возвращает:
      X_train_w, X_val_w, scaler, pca
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float32)
    X_train, X_val = train_test_split(
        X, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train_z = scaler.transform(X_train)
    X_val_z = scaler.transform(X_val)

    if use_pca:
        pca = PCA(
            whiten=pca_whiten,
            n_components=pca_components,
            svd_solver="full",
        ).fit(X_train_z)
        X_train_w = pca.transform(X_train_z).astype(np.float32)
        X_val_w = pca.transform(X_val_z).astype(np.float32)
    else:
        pca = None
        X_train_w = X_train_z.astype(np.float32)
        X_val_w = X_val_z.astype(np.float32)

    return X_train_w, X_val_w, scaler, pca


def run_dsb_on_matrix(
    name: str,
    X_train_w: np.ndarray,
    X_val_w: np.ndarray,
    dsb_kwargs: dict | None = None,
    train_kwargs: dict | None = None,
    eval_kwargs: dict | None = None,
):
    """
    Унифицированный запуск DSBTabularBridge на уже подготовленной матрице.
    Возвращает словарь с метриками и сэмплами.
    """
    dsb_kwargs = dsb_kwargs or {}
    train_kwargs = train_kwargs or {}
    eval_kwargs = eval_kwargs or {}

    bridge = DSBTabularBridge(
        X_train_w=X_train_w,
        X_val_w=X_val_w,
        **dsb_kwargs,
    )

    bridge.train(**train_kwargs)

    eval_res = bridge.evaluate(**eval_kwargs)

    metrics = {
        "train": {
            "swd": eval_res["swd_train"],
            "mmd": eval_res["mmd_train"],
            "ks": eval_res["ks_train"],
        },
        "val": {
            "swd": eval_res.get("swd_val"),
            "mmd": eval_res.get("mmd_val"),
            "ks": eval_res.get("ks_val"),
        },
    }

    syn = {
        "train_w": eval_res["syn_train_w"],
        "val_w": eval_res.get("syn_val_w"),
    }

    return {
        "name": name,
        "bridge": bridge,
        "metrics": metrics,
        "syn": syn,
    }


# ================== 1) Бенчмарк DSB на pkl-датасетах ================== #

def run_dsb_benchmarks(
    target_datasets: Optional[list[str]] = None,
    use_pca: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Запускает DSB на датасетах из datasets_numeric.pkl и возвращает all_results.

    Если target_datasets is None — берём все ключи из pkl.
    """
    set_seed(0)

    # грузим все X из pkl
    raw = load_datasets_numeric_from_pkl(DATASETS_NUMERIC_PATH)

    if target_datasets is None:
        ds_items = raw.items()
    else:
        ds_items = [(name, raw[name]) for name in target_datasets if name in raw]

    # Общие гиперпараметры DSB / train / eval (пока те же)
    DSB_CFG = dict(
        T=1.0,
        N=24,
        alpha_ou=0.3255,
        lr=4.56e-4,
    )
    
    TRAIN_CFG = dict(
        ipf_iters=3,
        steps_B=800,
        steps_F=800,
        batch=2048,
        pretrain_steps=500,
        pretrain_batch=1559,
        print_swd=True,
    )

    EVAL_CFG = dict(
        n_proj=512,
        sigma=None,
        steps_per_edge=3,
        max_eval_samples=2500,
    )

    all_results: Dict[str, Dict[str, Any]] = {}

    for ds_name, X in ds_items:
        print(f"\n=== DSB on dataset '{ds_name}' ===")

        X_train_w, X_val_w, _, _ = prepare_numpy_X(
            X,
            test_size=test_size,
            random_state=random_state,
            use_pca=use_pca,
        )

        all_results[ds_name] = run_dsb_on_matrix(
            ds_name,
            X_train_w,
            X_val_w,
            dsb_kwargs=DSB_CFG,
            train_kwargs=TRAIN_CFG,
            eval_kwargs=EVAL_CFG,
        )

    # Краткий summary по SWD/MMD
    for name, res in all_results.items():
        tr = res["metrics"]["train"]
        va = res["metrics"]["val"]
        swd_val = va["swd"] if va["swd"] is not None else float("nan")
        mmd_val = va["mmd"] if va["mmd"] is not None else float("nan")
        print(
            f"{name:28s} | "
            f"train SWD={tr['swd']:.5f}, MMD={tr['mmd']:.5f} | "
            f"val SWD={swd_val:.5f}, MMD={mmd_val:.5f}"
        )

    return all_results


# ================== 2) real_data / synthetic_data ================== #

def build_real_and_synthetic_from_dsb(
    all_results: Dict[str, Dict[str, Any]]
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
]:
    """
    Преобразует all_results от DSB-бенчмарка в два словаря:
      real_data и synthetic_data (для модели "DSB").
    """
    real_data: Dict[str, Dict[str, np.ndarray]] = {}
    synthetic_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"DSB": {}}

    for ds_name, res in all_results.items():
        bridge: DSBTabularBridge = res["bridge"]
        syn = res["syn"]

        X_train_w = bridge.X_train_w.astype(np.float32)
        X_val_w = (
            bridge.X_val_w.astype(np.float32)
            if bridge.X_val_w is not None
            else None
        )

        real_data[ds_name] = {
            "train_w": X_train_w,
            "val_w": X_val_w,
        }

        syn_train_w = syn["train_w"].astype(np.float32)
        syn_val_raw = syn.get("val_w", None)
        syn_val_w = syn_val_raw.astype(np.float32) if syn_val_raw is not None else None

        synthetic_data["DSB"][ds_name] = {
            "train_w": syn_train_w,
            "val_w": syn_val_w,
        }

    return real_data, synthetic_data


def main():
    # 1) Запускаем DSB на всех датасетах из pkl (или подмножестве, если задать TARGET_DATASETS)
    all_results = run_dsb_benchmarks(
        target_datasets=TARGET_DATASETS,
        use_pca=False,
        test_size=0.2,
        random_state=42,
    )

    # 2) Строим real_data и synthetic_data для eval_tabular_generators
    real_data, synthetic_data = build_real_and_synthetic_from_dsb(all_results)

    # 3) Сохраняем в pkl
    with open("real_data_dsb.pkl", "wb") as f:
        pickle.dump(real_data, f)

    with open("synthetic_data_dsb.pkl", "wb") as f:
        pickle.dump(synthetic_data, f)

    print("\nSaved:")
    print("  real_data_dsb.pkl")
    print("  synthetic_data_dsb.pkl")


if __name__ == "__main__":
    main()
