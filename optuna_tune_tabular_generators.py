"""
optuna_tune_tabular_generators.py

Тюнинг генераторов табличных данных с помощью Optuna.

Поддерживаемые модели:
  - "DSB"      : DSBTabularBridge (dsb_tabular.py)
  - "CTGAN"    : CTGAN (библиотека `ctgan`)
  - "TabDDPM"  : ddpm-плагин из synthcity (TabDDPM)

API:

1) tune_model_with_optuna(...)
   — одна стадия Optuna, общие гиперы на набор датасетов.

2) tune_model_per_dataset(...)
   — отдельная стадия Optuna для КАЖДОГО датасета,
     возвращает по датасету свои best_params и метрики (fidelity_loss + SWD/MMD/...)
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional

import pickle

import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.INFO)

from ctgan import CTGAN
from synthcity.plugins import Plugins
import torch

from dsb_tabular import (
    DSBTabularBridge,
    set_seed,
    sliced_wasserstein,
    mmd_rbf,
)
from run_dsb_and_prepare import prepare_numpy_X
from eval_tabular_data import FidelityMetrics, evaluate_fidelity_privacy


# ===================== DATA ===================== #

def load_datasets_numeric_from_pkl(
    path: str = "datasets_numeric.pkl",
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


def prepare_all_datasets(
    use_pca: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    pkl_path: str = "datasets_numeric.pkl",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Готовит датасеты из datasets_numeric.pkl в виде (X_train_w, X_val_w).

    Ожидается, что в pkl лежит словарь:
        name -> X (np.ndarray или DataFrame) с числовыми фичами.

    Возвращает:
        name -> (X_train_w, X_val_w)
    """
    raw = load_datasets_numeric_from_pkl(pkl_path)
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name, X in raw.items():
        X_tr, X_val, _, _ = prepare_numpy_X(
            X,
            test_size=test_size,
            random_state=random_state,
            use_pca=use_pca,
        )
        datasets[name] = (X_tr, X_val)

    return datasets


# ===================== АГРЕГИРОВАННАЯ МЕТРИКА ===================== #

def _loss_from_metric(x: float, scale: float) -> float:
    """
    Переводит любую неотрицательную метрику в [0,1):

    x = 0       -> 0.0 (идеально)
    x ~ scale   -> ~0.5
    x >> scale  -> -> 1.0

    Главное: монотонно, без жесткой чувствительности к единицам измерения.
    """
    x = float(max(x, 0.0))
    s = float(scale) + 1e-8
    score = 1.0 / (1.0 + x / s)   # [~0,1], больше = лучше
    loss  = 1.0 - score           # [0,~1], больше = хуже
    return loss


def aggregate_fidelity_for_optuna(
    fp: FidelityMetrics,
    swd: float,
    mmd: float,
) -> float:
    """
    Итоговый лосс для Optuna: меньше = лучше.

    Используем:
      - wd_mean, ks_mean  (univariate)
      - corr_frobenius, corr_mae_offdiag (dependence)
      - swd, mmd          (геометрия)
    Все метрики сначала нормализуются в [0,1], потом агрегируются.
    """

    wd_mean    = float(fp.fidelity_uni["wasserstein"]["mean"])
    ks_mean    = float(fp.fidelity_uni["ks"]["mean_ks"])
    corr_frob  = float(fp.fidelity_dep["corr_frobenius"])
    corr_mae   = float(fp.fidelity_dep["corr_mae_offdiag"])
    swd        = float(swd)
    mmd        = float(mmd)

    # --- Нормализация в [0,1] (scale — "типично плохой" уровень, подстроить при желании) ---
    wd_loss    = _loss_from_metric(wd_mean,   scale=0.5)
    ks_loss    = _loss_from_metric(ks_mean,   scale=0.5)
    corr_mae_l = _loss_from_metric(corr_mae,  scale=0.3)
    corr_frob_l= _loss_from_metric(corr_frob, scale=10.0)
    swd_loss   = _loss_from_metric(swd,       scale=5.0)
    mmd_loss   = _loss_from_metric(mmd,       scale=1.0)
    # print('wd_loss', wd_loss)
    # print('ks_loss', ks_loss)
    # print('corr_mae_l', corr_mae_l)
    print('swd_loss', swd_loss)
    print('mmd_loss', mmd_loss)

    # блоки: univariate, dependence, geometry
    uni_block  = 0.3 * (wd_loss + ks_loss)
    dep_block  = 0.4 * (corr_mae_l + corr_frob_l)
    geom_block = swd_loss + mmd_loss

    # print('uni_block', uni_block)
    # print('dep_block', dep_block)
    # print('geom_block', geom_block)

    # можно просто сложить (3 блока) или усреднить — разницы для Optuna почти нет
    # total_loss = uni_block + dep_block + geom_block
    total_loss = geom_block

    return float(total_loss)



# ===================== DSB ===================== #

def _objective_dsb(
    trial: optuna.Trial,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> float:
    """
    Optuna objective для DSBTabularBridge на МНОЖЕСТВЕ датасетов.
    Возвращает средний aggregated fidelity loss по датасетам.
    """
    set_seed(0)

    # гиперпараметры DSB
    T = 1.0
    N = trial.suggest_int("N", 20, 100, step=10)
    alpha_ou = trial.suggest_float("alpha_ou", 0.1, 2.0, log=True)
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    hidden = trial.suggest_categorical("hidden", [256, 512, 1024])
    time_features = trial.suggest_categorical("time_features", [8, 16, 32])

    ipf_iters = trial.suggest_int("ipf_iters", 3, 6)
    steps_B = trial.suggest_int("steps_B", 300, 1000)
    steps_F = trial.suggest_int("steps_F", 300, 1000)
    batch = trial.suggest_int("batch", 512, 4096)
    scores_per_dataset: Dict[str, float] = {}

    for ds_name, (X_train_w, X_val_w) in datasets.items():
        bridge = DSBTabularBridge(
            X_train_w=X_train_w,
            X_val_w=X_val_w,
            T=T,
            N=N,
            alpha_ou=alpha_ou,
            lr=lr,
            hidden=hidden,
            time_features=time_features,
        )

        bridge.train(
            ipf_iters=ipf_iters,
            steps_B=steps_B,
            steps_F=steps_F,
            batch=batch,
            pretrain_steps=100,
            pretrain_batch=512,
            print_swd=False,
        )

        syn_val_w = bridge.sample_from_bridge(num=X_val_w.shape[0], steps_per_edge=2)

        swd = sliced_wasserstein(X_val_w, syn_val_w, n_proj=256)
        mmd = mmd_rbf(X_val_w, syn_val_w, sigma=1.0)

        fp = evaluate_fidelity_privacy(
            X_real_train=X_val_w,
            X_syn_train=syn_val_w,
        )

        loss_ds = aggregate_fidelity_for_optuna(fp, swd=swd, mmd=mmd)
        scores_per_dataset[ds_name] = loss_ds

    for ds_name, score in scores_per_dataset.items():
        trial.set_user_attr(f"{ds_name}_loss", float(score))

    return float(np.mean(list(scores_per_dataset.values())))


def _make_objective_dsb_single(
    X_train_w: np.ndarray,
    X_val_w: np.ndarray,
):
    """
    Объектив для DSB на ОДНОМ датасете (используется в tune_model_per_dataset).
    """

    def objective(trial: optuna.Trial) -> float:
        set_seed(0)

        T = 1.0
        N = trial.suggest_int("N", 8, 32, step=8)
        alpha_ou = trial.suggest_float("alpha_ou", 0.1, 2.0, log=True)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        hidden = trial.suggest_categorical("hidden", [ 256, 512, 1024])
        time_features = trial.suggest_categorical("time_features", [8, 16, 32])

        ipf_iters = trial.suggest_int("ipf_iters", 3,6)
        steps_B = trial.suggest_int("steps_B", 300, 1000)
        steps_F = trial.suggest_int("steps_F", 300, 1000)
        batch = trial.suggest_int("batch", 512, 4096)

        # alpha_ou = trial.suggest_float("alpha_ou", 0.1, 2.0, log=True)
        # lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        # hidden = trial.suggest_categorical("hidden", [256, 512, 1024])
        # time_features = trial.suggest_categorical("time_features", [8, 16, 32])

        # ipf_iters = trial.suggest_int("ipf_iters", 3, 6)
        # steps_B = trial.suggest_int("steps_B", 300, 1000)
        # steps_F = trial.suggest_int("steps_F", 300, 1000)
        # batch = trial.suggest_int("batch", 512, 4096)
        bridge = DSBTabularBridge(
            X_train_w=X_train_w,
            X_val_w=X_val_w,
            T=T,
            N=N,
            alpha_ou=alpha_ou,
            lr=lr,
            hidden=hidden,
            time_features=time_features,
        )

        bridge.train(
            ipf_iters=ipf_iters,
            steps_B=steps_B,
            steps_F=steps_F,
            batch=batch,
            pretrain_steps=500,
            pretrain_batch=512,
            print_swd=False,
        )

        syn_val_w = bridge.sample_from_bridge(num=X_val_w.shape[0], steps_per_edge=2)

        swd = sliced_wasserstein(X_val_w, syn_val_w, n_proj=256)
        mmd = mmd_rbf(X_val_w, syn_val_w, sigma=1.0)

        fp = evaluate_fidelity_privacy(
            X_real_train=X_val_w,
            X_syn_train=syn_val_w,
        )

        loss = aggregate_fidelity_for_optuna(fp, swd=swd, mmd=mmd)

        trial.set_user_attr("swd_val", float(swd))
        trial.set_user_attr("mmd_val", float(mmd))
        trial.set_user_attr("wd_mean", float(fp.fidelity_uni["wasserstein"]["mean"]))
        trial.set_user_attr("ks_mean", float(fp.fidelity_uni["ks"]["mean_ks"]))
        trial.set_user_attr("corr_frobenius", float(fp.fidelity_dep["corr_frobenius"]))
        trial.set_user_attr("corr_mae_offdiag", float(fp.fidelity_dep["corr_mae_offdiag"]))

        return float(loss)

    return objective


# ===================== CTGAN ===================== #

def _train_ctgan_on_matrix(
    X_train_w: np.ndarray,
    epochs: int,
    batch_size: int,
    embedding_dim: int = 128,
    generator_dim: Tuple[int, int] = (256, 256),
    discriminator_dim: Tuple[int, int] = (256, 256),
    generator_lr: float = 2e-4,
    discriminator_lr: float = 2e-4,
    pac: int = 1,
) -> CTGAN:
    """
    Обучение CTGAN на матрице признаков (все фичи предполагаем числовыми).
    """
    X_train_w = np.asarray(X_train_w, dtype=np.float32)
    n_train, d = X_train_w.shape
    cols = [f"f{j}" for j in range(d)]
    df_train = pd.DataFrame(X_train_w, columns=cols)

    model = CTGAN(
        embedding_dim=embedding_dim,
        generator_dim=generator_dim,
        discriminator_dim=discriminator_dim,
        generator_lr=generator_lr,
        discriminator_lr=discriminator_lr,
        batch_size=batch_size,
        epochs=epochs,
        pac=pac,
        verbose=False,
        cuda=True,  # GPU
    )

    # ВСЕ фичи считаем непрерывными → discrete_columns=[]
    model.fit(df_train, discrete_columns=[])

    return model


def _objective_ctgan(
    trial: optuna.Trial,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> float:
    set_seed(0)

    # базовые hyper’ы
    epochs = trial.suggest_int("epochs", 50, 400)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    # архитектура
    embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
    gen_width = trial.suggest_categorical("gen_width", [128, 256, 512])
    disc_width = trial.suggest_categorical("disc_width", [128, 256, 512])
    generator_dim = (gen_width, gen_width)
    discriminator_dim = (disc_width, disc_width)

    pac = trial.suggest_categorical("pac", [1, 2, 4, 8])

    generator_lr = trial.suggest_float("generator_lr", 1e-4, 5e-4, log=True)
    discriminator_lr = trial.suggest_float("discriminator_lr", 1e-4, 5e-4, log=True)

    scores_per_dataset: Dict[str, float] = {}

    for ds_name, (X_train_w, X_val_w) in datasets.items():
        model = _train_ctgan_on_matrix(
            X_train_w=X_train_w,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            pac=pac,
        )

        n_val = X_val_w.shape[0]
        syn_val_df = model.sample(n_val)
        syn_val_w = syn_val_df.to_numpy(dtype=np.float32)

        swd = sliced_wasserstein(X_val_w, syn_val_w, n_proj=256)
        mmd = mmd_rbf(X_val_w, syn_val_w, sigma=1.0)

        fp = evaluate_fidelity_privacy(
            X_real_train=X_val_w,
            X_syn_train=syn_val_w,
        )

        loss_ds = aggregate_fidelity_for_optuna(fp, swd=swd, mmd=mmd)
        scores_per_dataset[ds_name] = loss_ds

    for ds_name, score in scores_per_dataset.items():
        trial.set_user_attr(f"{ds_name}_loss", float(score))

    return float(np.mean(list(scores_per_dataset.values())))


def _make_objective_ctgan_single(
    X_train_w: np.ndarray,
    X_val_w: np.ndarray,
):
    def objective(trial: optuna.Trial) -> float:
        set_seed(0)

        epochs = trial.suggest_int("epochs", 50, 400)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        gen_width = trial.suggest_categorical("gen_width", [128, 256, 512])
        disc_width = trial.suggest_categorical("disc_width", [128, 256, 512])
        generator_dim = (gen_width, gen_width)
        discriminator_dim = (disc_width, disc_width)

        pac = trial.suggest_categorical("pac", [1, 2, 4, 8])

        generator_lr = trial.suggest_float("generator_lr", 1e-4, 5e-4, log=True)
        discriminator_lr = trial.suggest_float("discriminator_lr", 1e-4, 5e-4, log=True)

        model = _train_ctgan_on_matrix(
            X_train_w=X_train_w,
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            pac=pac,
        )

        n_val = X_val_w.shape[0]
        syn_val_df = model.sample(n_val)
        syn_val_w = syn_val_df.to_numpy(dtype=np.float32)

        swd = sliced_wasserstein(X_val_w, syn_val_w, n_proj=256)
        mmd = mmd_rbf(X_val_w, syn_val_w, sigma=1.0)

        fp = evaluate_fidelity_privacy(
            X_real_train=X_val_w,
            X_syn_train=syn_val_w,
        )

        loss = aggregate_fidelity_for_optuna(fp, swd=swd, mmd=mmd)

        trial.set_user_attr("swd_val", float(swd))
        trial.set_user_attr("mmd_val", float(mmd))
        trial.set_user_attr("wd_mean", float(fp.fidelity_uni["wasserstein"]["mean"]))
        trial.set_user_attr("ks_mean", float(fp.fidelity_uni["ks"]["mean_ks"]))
        trial.set_user_attr("corr_frobenius", float(fp.fidelity_dep["corr_frobenius"]))
        trial.set_user_attr("corr_mae_offdiag", float(fp.fidelity_dep["corr_mae_offdiag"]))

        return float(loss)

    return objective


# ===================== TabDDPM (SynthCity ddpm) ===================== #

def _train_tabddpm_on_matrix(
    X_train_w: np.ndarray,
    n_iter: int,
    num_timesteps: int,
    model_type: str = "mlp",
    validation_size: float = 0.2,
    sampling_patience: int = 200,
    device: torch.device | str | None = None,
    **extra_kwargs,
):
    """
    Обучение TabDDPM (SynthCity) на числовой матрице признаков.
    Возвращает (tabddpm, sample_fn), где sample_fn(n) -> np.ndarray.
    """
    # 1) матрица -> DataFrame
    X_train_w = np.asarray(X_train_w, dtype=np.float32)
    n_train, d = X_train_w.shape
    cols = [f"f{j}" for j in range(d)]
    df_train = pd.DataFrame(X_train_w, columns=cols)

    # 2) устройство
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # 3) получаем плагин ddpm
    tabddpm = Plugins().get(
        "ddpm",
        n_iter=n_iter,
        num_timesteps=num_timesteps,
        model_type=model_type,          # "mlp" стабильно
        validation_size=validation_size,
        sampling_patience=sampling_patience,
        device=device,                  # torch.device
        **extra_kwargs,
    )

    # 4) fit
    tabddpm.fit(df_train)

    # 5) функция семплирования
    def sample_fn(n: int) -> np.ndarray:
        loader = tabddpm.generate(count=n)
        syn_df = loader.dataframe()
        return syn_df.to_numpy(dtype=np.float32)

    return tabddpm, sample_fn


def _objective_tabddpm(
    trial: optuna.Trial,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> float:
    set_seed(0)

    n_iter = trial.suggest_int("n_iter", 50, 300)
    num_timesteps = trial.suggest_int("num_timesteps", 100, 600)
    sampling_patience = trial.suggest_int("sampling_patience", 100, 500)

    # фиксируем архитектуру
    model_type = "mlp"

    scores: Dict[str, float] = {}

    for ds_name, (X_train_w, X_val_w) in datasets.items():
        tabddpm, sample_fn = _train_tabddpm_on_matrix(
            X_train_w=X_train_w,
            n_iter=n_iter,
            num_timesteps=num_timesteps,
            model_type=model_type,
            validation_size=0.2,
            sampling_patience=sampling_patience,
            device="cuda",
        )
        syn_val_w = sample_fn(X_val_w.shape[0])

        swd = sliced_wasserstein(X_val_w, syn_val_w, n_proj=256)
        mmd = mmd_rbf(X_val_w, syn_val_w, sigma=1.0)

        fp = evaluate_fidelity_privacy(
            X_real_train=X_val_w,
            X_syn_train=syn_val_w,
        )

        loss_ds = aggregate_fidelity_for_optuna(fp, swd=swd, mmd=mmd)
        scores[ds_name] = loss_ds

    for k, v in scores.items():
        trial.set_user_attr(f"{k}_loss", float(v))

    return float(np.mean(list(scores.values())))


def _make_objective_tabddpm_single(
    X_train_w: np.ndarray,
    X_val_w: np.ndarray,
):
    def objective(trial: optuna.Trial) -> float:
        set_seed(0)

        n_iter = trial.suggest_int("n_iter", 50, 300)
        num_timesteps = trial.suggest_int("num_timesteps", 100, 600)
        sampling_patience = trial.suggest_int("sampling_patience", 100, 500)
        model_type = "mlp"

        tabddpm, sample_fn = _train_tabddpm_on_matrix(
            X_train_w=X_train_w,
            n_iter=n_iter,
            num_timesteps=num_timesteps,
            model_type=model_type,
            validation_size=0.2,
            sampling_patience=sampling_patience,
            device="cuda",
        )

        syn_val_w = sample_fn(X_val_w.shape[0])

        swd = sliced_wasserstein(X_val_w, syn_val_w, n_proj=256)
        mmd = mmd_rbf(X_val_w, syn_val_w, sigma=1.0)

        fp = evaluate_fidelity_privacy(
            X_real_train=X_val_w,
            X_syn_train=syn_val_w,
        )

        loss = aggregate_fidelity_for_optuna(fp, swd=swd, mmd=mmd)

        trial.set_user_attr("swd_val", float(swd))
        trial.set_user_attr("mmd_val", float(mmd))
        trial.set_user_attr("wd_mean", float(fp.fidelity_uni["wasserstein"]["mean"]))
        trial.set_user_attr("ks_mean", float(fp.fidelity_uni["ks"]["mean_ks"]))
        trial.set_user_attr("corr_frobenius", float(fp.fidelity_dep["corr_frobenius"]))
        trial.set_user_attr("corr_mae_offdiag", float(fp.fidelity_dep["corr_mae_offdiag"]))

        return float(loss)

    return objective


# ===================== PUBLIC API ===================== #

def tune_model_with_optuna(
    model_name: str,
    n_trials: int = 20,
    target_datasets: Optional[List[str]] = None,
    use_pca: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    direction: str = "minimize",
) -> optuna.Study:
    """
    Тюнинг модели на НАБОРЕ датасетов: одна Optuna-стадия, общие гиперпараметры.

    model_name: "DSB", "CTGAN" или "TabDDPM".
    """
    all_datasets = prepare_all_datasets(
        use_pca=use_pca,
        test_size=test_size,
        random_state=random_state,
    )

    if target_datasets is not None:
        datasets = {
            name: all_datasets[name]
            for name in target_datasets
            if name in all_datasets
        }
        if not datasets:
            raise ValueError(
                f"No datasets from target_datasets={target_datasets} "
                f"found in available {list(all_datasets.keys())}"
            )
    else:
        datasets = all_datasets

    model_name_up = model_name.upper()
    if model_name_up == "DSB":
        objective_fn = lambda trial: _objective_dsb(trial, datasets)
    elif model_name_up == "CTGAN":
        objective_fn = lambda trial: _objective_ctgan(trial, datasets)
    elif model_name_up == "TABDDPM":
        objective_fn = lambda trial: _objective_tabddpm(trial, datasets)
    else:
        raise ValueError(
            f"Unknown model_name='{model_name}'. Expected 'DSB', 'CTGAN', 'TabDDPM'."
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=(storage is not None),
    )

    study.optimize(objective_fn, n_trials=n_trials)
    return study


def tune_model_per_dataset(
    model_name: str,
    n_trials: int = 20,
    target_datasets: Optional[List[str]] = None,
    use_pca: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    storage: Optional[str] = None,
    direction: str = "minimize",
    save_path: Optional[str] = None,
):
    """
    Тюнинг модели ОТДЕЛЬНО для каждого датасета.

    Для каждого датасета создаётся своя Optuna-стадия, подбираются
    лучшие гиперпараметры и считаются финальные метрики.

    Возвращает:
        results: dict вида
            {
              ds_name: {
                "study": study,
                "best_params": {...},
                "best_value": float,   # минимальный fidelity_loss
                "metrics": {
                    "fidelity_loss": float,
                    "swd_val": float,
                    "mmd_val": float,
                    "wd_mean": float,
                    "ks_mean": float,
                    "corr_frobenius": float,
                    "corr_mae_offdiag": float,
                },
              },
              ...
            }
    """
    all_datasets = prepare_all_datasets(
        use_pca=use_pca,
        test_size=test_size,
        random_state=random_state,
    )

    if target_datasets is not None:
        datasets = {
            name: all_datasets[name]
            for name in target_datasets
            if name in all_datasets
        }
        if not datasets:
            raise ValueError(
                f"No datasets from target_datasets={target_datasets} "
                f"found in available {list(all_datasets.keys())}"
            )
    else:
        datasets = all_datasets

    model_name_up = model_name.upper()
    results = {}

    for ds_name, (X_train_w, X_val_w) in datasets.items():
        print(f"\n=== Tuning {model_name_up} on dataset '{ds_name}' ===")

        if model_name_up == "DSB":
            objective = _make_objective_dsb_single(X_train_w, X_val_w)
        elif model_name_up == "CTGAN":
            objective = _make_objective_ctgan_single(X_train_w, X_val_w)
        elif model_name_up == "TABDDPM":
            objective = _make_objective_tabddpm_single(X_train_w, X_val_w)
        else:
            raise ValueError(
                f"Unknown model_name='{model_name}'. Expected 'DSB', 'CTGAN', 'TabDDPM'."
            )

        study_name = f"{model_name_up}_{ds_name}"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=(storage is not None),
        )

        def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
            print(
                f"[{model_name}][{ds_name}] "
                f"trial {trial.number} finished, value={trial.value:.6f}, state={trial.state}"
            )

        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[trial_callback],
        )

        best_trial = study.best_trial
        best_score = float(best_trial.value)
        swd_val = float(best_trial.user_attrs.get("swd_val", float("nan")))
        mmd_val = float(best_trial.user_attrs.get("mmd_val", float("nan")))
        wd_mean = float(best_trial.user_attrs.get("wd_mean", float("nan")))
        ks_mean = float(best_trial.user_attrs.get("ks_mean", float("nan")))
        corr_frob = float(best_trial.user_attrs.get("corr_frobenius", float("nan")))
        corr_mae = float(best_trial.user_attrs.get("corr_mae_offdiag", float("nan")))

        results[ds_name] = {
            "study": study,
            "best_params": best_trial.params,
            "best_value": best_score,  # общий loss = fidelity_loss
            "metrics": {
                "fidelity_loss": best_score,
                "swd_val": swd_val,
                "mmd_val": mmd_val,
                "wd_mean": wd_mean,
                "ks_mean": ks_mean,
                "corr_frobenius": corr_frob,
                "corr_mae_offdiag": corr_mae,
            },
        }

        print(
            f"[{model_name_up} | {ds_name}] "
            f"loss={best_score:.5f}, swd={swd_val:.5f}, mmd={mmd_val:.5f}, "
            f"wd_mean={wd_mean:.5f}, ks_mean={ks_mean:.5f}, "
            f"corr_frob={corr_frob:.5f}, corr_mae={corr_mae:.5f}"
        )

    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved per-dataset tuning results to {save_path}")

    return results
