"""
run_ctgan_and_prepare.py

Запускает CTGAN на числовых датасетах из datasets_numeric.pkl
и готовит данные для eval_tabular_generators.py.

Ожидаем, что в pkl лежит словарь:
  name -> X (np.ndarray или DataFrame), все фичи — числовые.

На выходе создаёт:
  - real_data_ctgan.pkl
  - synthetic_data_ctgan.pkl

Форматы:

real_data_ctgan: {
    dataset_name: {
        "train_w": np.ndarray [n_train, D],
        "val_w":   np.ndarray [n_val, D],
    },
    ...
}

synthetic_data_ctgan: {
    "CTGAN": {
        dataset_name: {
            "train_w": np.ndarray [n_train_syn, D],
            "val_w":   np.ndarray [n_val_syn, D],
        },
        ...
    }
}
"""

from typing import Dict, Any

import numpy as np
import pandas as pd
import pickle

from ctgan import CTGAN

from dsb_tabular import set_seed
from run_dsb_and_prepare import prepare_numpy_X


# путь к pkl с матрицами датасетов
DATASETS_NUMERIC_PATH = "datasets_numeric.pkl"


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


def run_ctgan_on_matrix(
    name: str,
    X_train_w: np.ndarray,
    X_val_w: np.ndarray,
    ctgan_kwargs: dict | None = None,
) -> Dict[str, Any]:
    """
    Унифицированный запуск CTGAN на матрице (whitened / нормированной).
    Предполагаем, что ВСЕ признаки — непрерывные.
    Возвращает:
      {
        "name": name,
        "model": CTGAN,
        "syn": {
            "train_w": np.ndarray,
            "val_w":   np.ndarray,
        },
      }
    """
    ctgan_kwargs = ctgan_kwargs or {}

    X_train_w = np.asarray(X_train_w, dtype=np.float32)
    X_val_w = np.asarray(X_val_w, dtype=np.float32)

    n_train, d = X_train_w.shape
    n_val = X_val_w.shape[0]

    cols = [f"f{j}" for j in range(d)]
    df_train = pd.DataFrame(X_train_w, columns=cols)

    # Все признаки считаем непрерывными → discrete_columns = []
    # model = CTGAN(**ctgan_kwargs)
    CTGAN_CFG = dict(
        # архитектура
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),

        # оптимизация
        batch_size=256,
        epochs=250,                 # между 169 и 314
        generator_lr=1.7e-4,        # среднее между 1.2e-4 и 2.2e-4
        discriminator_lr=4.3e-4,    # среднее между 3.6e-4 и 5.0e-4

        # прочее
        pac=2,
        verbose=False,
    )

    model = CTGAN(
        embedding_dim=CTGAN_CFG["embedding_dim"],
        generator_dim=CTGAN_CFG["generator_dim"],
        discriminator_dim=CTGAN_CFG["discriminator_dim"],
        generator_lr=CTGAN_CFG["generator_lr"],
        discriminator_lr=CTGAN_CFG["discriminator_lr"],
        batch_size=CTGAN_CFG["batch_size"],
        epochs=CTGAN_CFG["epochs"],
        pac=CTGAN_CFG["pac"],
        verbose=CTGAN_CFG["verbose"],
    )

    model.fit(df_train, discrete_columns=[])

    syn_train_df = model.sample(n_train)
    syn_val_df = model.sample(n_val)

    syn_train_w = syn_train_df.to_numpy(dtype=np.float32)
    syn_val_w = syn_val_df.to_numpy(dtype=np.float32)

    return {
        "name": name,
        "model": model,
        "syn": {
            "train_w": syn_train_w,
            "val_w": syn_val_w,
        },
    }


def run_ctgan_benchmarks(
    target_datasets: list[str] | None = None,
) -> tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
]:
    """
    Прогоняет CTGAN по датасетам из datasets_numeric.pkl и строит:

      real_data: {
          ds_name: { "train_w": ..., "val_w": ... },
          ...
      }

      synthetic_data: {
          "CTGAN": {
              ds_name: { "train_w": ..., "val_w": ... },
              ...
          }
      }

    Если target_datasets is not None — берём только эти имена.
    Возвращает (real_data, synthetic_data).
    """
    set_seed(0)

    # базовые гиперы CTGAN (при необходимости подправишь)
    CTGAN_CFG = dict(
        # архитектура
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),

        # оптимизация
        batch_size=256,
        epochs=250,                 # между 169 и 314
        generator_lr=1.7e-4,        # среднее между 1.2e-4 и 2.2e-4
        discriminator_lr=4.3e-4,    # среднее между 3.6e-4 и 5.0e-4

        # прочее
        pac=2,
        verbose=False,
    )


    # загружаем все матрицы
    raw = load_datasets_numeric_from_pkl(DATASETS_NUMERIC_PATH)

    if target_datasets is not None:
        raw = {k: v for k, v in raw.items() if k in target_datasets}

    real_data: Dict[str, Dict[str, np.ndarray]] = {}
    synthetic_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"CTGAN": {}}

    for name, X in raw.items():
        print(f"\n=== CTGAN on dataset: {name} ===")
        # тот же препроцесс, что и для DSB: train/val + scaler + PCA/whiten
        X_train_w, X_val_w, _, _ = prepare_numpy_X(
            X,
            test_size=0.2,
            random_state=42,
            use_pca=False,
        )

        res = run_ctgan_on_matrix(
            name=name,
            X_train_w=X_train_w,
            X_val_w=X_val_w,
            ctgan_kwargs=CTGAN_CFG,
        )

        real_data[name] = {
            "train_w": X_train_w,
            "val_w": X_val_w,
        }
        synthetic_data["CTGAN"][name] = {
            "train_w": res["syn"]["train_w"],
            "val_w": res["syn"]["val_w"],
        }

    return real_data, synthetic_data


def main():
    # если нужно ограничить список датасетов:
    # target = ["california_housing", "king_county_housing"]
    # real_data, synthetic_data = run_ctgan_benchmarks(target_datasets=target)
    real_data, synthetic_data = run_ctgan_benchmarks(target_datasets=None)

    with open("real_data_ctgan.pkl", "wb") as f:
        pickle.dump(real_data, f)

    with open("synthetic_data_ctgan.pkl", "wb") as f:
        pickle.dump(synthetic_data, f)

    print("\nSaved:")
    print("  real_data_ctgan.pkl")
    print("  synthetic_data_ctgan.pkl")


if __name__ == "__main__":
    main()
