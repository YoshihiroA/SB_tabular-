from typing import Dict, Tuple

import pickle
import numpy as np
import pandas as pd
import torch
from synthcity.plugins import Plugins

from dsb_tabular import set_seed
from run_dsb_and_prepare import prepare_numpy_X

DATASETS_NUMERIC_PATH = "datasets_numeric.pkl"

# None → использовать все датасеты из pkl
TARGET_DATASETS: list[str] | None = None


def load_datasets_numeric_from_pkl(
    path: str = DATASETS_NUMERIC_PATH,
) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        raw = pickle.load(f)

    data: Dict[str, np.ndarray] = {}
    for name, X in raw.items():
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float32)
        data[name] = X
    return data


def train_tabddpm_on_matrix(
    X_train_w: np.ndarray,
    n_iter: int = 241,
    num_timesteps: int = 195,
    model_type: str = "mlp",
    validation_size: float = 0.2,
    sampling_patience: int = 270,
    device: torch.device | str | None = None,
    **extra_kwargs,
):
    X_train_w = np.asarray(X_train_w, dtype=np.float32)
    n_train, d = X_train_w.shape
    cols = [f"f{j}" for j in range(d)]
    df_train = pd.DataFrame(X_train_w, columns=cols)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    tabddpm = Plugins().get(
        "ddpm",
        n_iter=n_iter,
        num_timesteps=num_timesteps,
        model_type=model_type,
        validation_size=validation_size,
        sampling_patience=sampling_patience,
        device=device,
        **extra_kwargs,
    )

    tabddpm.fit(df_train)

    def sample_fn(n: int) -> np.ndarray:
        loader = tabddpm.generate(count=n)
        syn_df = loader.dataframe()
        return syn_df.to_numpy(dtype=np.float32)

    return tabddpm, sample_fn


def run_tabddpm_benchmarks(
    target_datasets: list[str] | None = None,
    use_pca: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
]:
    set_seed(0)

    raw = load_datasets_numeric_from_pkl(DATASETS_NUMERIC_PATH)

    if target_datasets is None:
        ds_items = raw.items()              # ← все датасеты
    else:
        ds_items = [(name, raw[name]) for name in target_datasets if name in raw]

    TABDDPM_CFG = dict(
        n_iter=241,
        num_timesteps=195,
        model_type="mlp",
        validation_size=0.2,
        sampling_patience=270,
        device="cuda",
    )

    real_data: Dict[str, Dict[str, np.ndarray]] = {}
    synthetic_data: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"TabDDPM": {}}

    for ds_name, X in ds_items:
        print(f"\n=== TabDDPM on dataset '{ds_name}' ===")

        X_train_w, X_val_w, _, _ = prepare_numpy_X(
            X,
            test_size=test_size,
            random_state=random_state,
            use_pca=use_pca,
        )

        _, sample_fn = train_tabddpm_on_matrix(
            X_train_w,
            **TABDDPM_CFG,
        )

        syn_train = sample_fn(X_train_w.shape[0])
        syn_val = sample_fn(X_val_w.shape[0])

        real_data[ds_name] = {
            "train_w": X_train_w,
            "val_w": X_val_w,
        }
        synthetic_data["TabDDPM"][ds_name] = {
            "train_w": syn_train,
            "val_w": syn_val,
        }

    return real_data, synthetic_data


def main():
    real_data, synthetic_data = run_tabddpm_benchmarks(
        target_datasets=TARGET_DATASETS,  # None → все
        use_pca=False,
        test_size=0.2,
        random_state=42,
    )

    with open("real_data_tabddpm.pkl", "wb") as f:
        pickle.dump(real_data, f)

    with open("synthetic_data_tabddpm.pkl", "wb") as f:
        pickle.dump(synthetic_data, f)

    print("\nSaved:")
    print("  real_data_tabddpm.pkl")
    print("  synthetic_data_tabddpm.pkl")


if __name__ == "__main__":
    main()
