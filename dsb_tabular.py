"""
dsb_tabular.py

DSB (Diffusion / Schrödinger Bridge) для табличных данных.

Использование (в другом файле):
    from dsb_tabular import DSBTabularBridge, set_seed

    set_seed(0)
    bridge = DSBTabularBridge(X_train_w, X_val_w, T=1.0, N=16, alpha_ou=1.0, lr=2e-3)
    bridge.train(ipf_iters=5, steps_B=800, steps_F=800, batch=2048)
    res = bridge.evaluate()
"""

import math
import random
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from scipy import stats


# ----------------- базовая настройка -----------------

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 0) -> None:
    """Фиксация сидов (при желании)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ----------------- Временная эмбеддинга + MeanMap -----------------

class FourierTime(nn.Module):
    def __init__(self, features: int = 16, max_freq: float = 20.0):
        super().__init__()
        self.register_buffer("freq", torch.linspace(1.0, max_freq, features))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        t = t[:, None]  # (B, 1)
        pe = 2 * math.pi * t * self.freq[None, :]  # (B, F)
        return torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1)  # (B, 2F)


class MeanMap(nn.Module):
    def __init__(self, xdim: int, hidden: int = 256, time_features: int = 16):
        super().__init__()
        self.temb = FourierTime(time_features)
        self.net = nn.Sequential(
            nn.Linear(xdim + 2 * time_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, xdim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, D), t: (B,)
        h = torch.cat([x, self.temb(t)], dim=-1)
        return self.net(h)


# ----------------- Основной класс DSB -----------------

class DSBTabularBridge:
    """
    Обёртка над DSB для табличных данных.

    Ожидает, что X_train_w / X_val_w уже предобработаны (например, StandardScaler+PCA).
    """

    def __init__(
        self,
        X_train_w: np.ndarray,
        X_val_w: Optional[np.ndarray] = None,
        T: float = 1.0,
        N: int = 16,
        alpha_ou: float = 1.0,
        lr: float = 2e-3,
        device: Optional[torch.device] = None,
        hidden: int = 256,
        time_features: int = 16,
    ):
        """
        Args:
            X_train_w: (n_train, D) — уже подготовленные признаки (whitened / нормированные и т.п.)
            X_val_w:   (n_val, D) или None
            T:         финальное время
            N:         число временных шагов
            alpha_ou:  параметр OU-процесса для предобучения F
            lr:        learning rate для AdamW
            hidden:    ширина MLP
            time_features: число частот в FourierTime
        """
        X_train_w = np.asarray(X_train_w, dtype=np.float32)
        self.X_train_w = X_train_w
        self.X_val_w = (
            np.asarray(X_val_w, dtype=np.float32) if X_val_w is not None else None
        )

        self.T = float(T)
        self.N = int(N)
        self.alpha_ou = float(alpha_ou)

        self.device = device if device is not None else DEFAULT_DEVICE

        self.D = self.X_train_w.shape[1]
        self.gamma = torch.full((self.N,), self.T / self.N, device=self.device)

        # Сети F и B
        self.F_net = MeanMap(
            self.D, hidden=hidden, time_features=time_features
        ).to(self.device)
        self.B_net = MeanMap(
            self.D, hidden=hidden, time_features=time_features
        ).to(self.device)

        # Оптимизаторы
        self.opt_F = torch.optim.AdamW(self.F_net.parameters(), lr=lr)
        self.opt_B = torch.optim.AdamW(self.B_net.parameters(), lr=lr)

    # ---------- вспомогательные выборки ----------

    def sample_data_p0_train(self, n: int, jitter: float = 0.0) -> np.ndarray:
        idx = np.random.randint(0, self.X_train_w.shape[0], size=n)
        x = self.X_train_w[idx]
        if jitter > 0:
            x = x + jitter * np.random.randn(*x.shape)
        return x.astype(np.float32)

    def sample_prior_pN(self, n: int, std: float = 1.0) -> np.ndarray:
        return (std * np.random.randn(n, self.D)).astype(np.float32)

    # ---------- предобучение F (OU) ----------

    def pretrain_F_ou(self, steps: int = 300, batch: int = 1024) -> None:
        """
        Предобучение F_net как шага OU-процесса.
        """
        for _ in range(steps):
            x = np.vstack(
                [
                    self.sample_data_p0_train(batch // 2, jitter=0.01),
                    self.sample_prior_pN(batch // 2),
                ]
            )
            x = torch.tensor(x, device=self.device)
            k = torch.randint(0, self.N, (x.size(0),), device=self.device)
            t_k = (k.float() + 0.5) / self.N
            dt = self.gamma[k]
            ou_mean = x + dt[:, None] * (-self.alpha_ou * x)
            pred = self.F_net(x, t_k)
            loss = F.mse_loss(pred, ou_mean)
            self.opt_F.zero_grad(set_to_none=True)
            loss.backward()
            self.opt_F.step()

    # ---------- симуляции вперёд/назад ----------

    @torch.no_grad()
    def simulate_forward_pairs(self, batch: int):
        Xk = torch.tensor(
            self.sample_data_p0_train(batch, jitter=0.01), device=self.device
        )
        pairs = []
        for k in range(self.N):
            t = torch.full((batch,), (k + 0.5) / self.N, device=self.device)
            mean = self.F_net(Xk, t)
            noise = torch.randn_like(Xk) * math.sqrt(2.0 * self.gamma[k].item())
            Xk1 = mean + noise
            pairs.append((k, Xk.detach(), Xk1.detach()))
            Xk = Xk1
        return pairs

    @torch.no_grad()
    def simulate_backward_pairs(self, batch: int):
        Xk = torch.tensor(self.sample_prior_pN(batch), device=self.device)
        seq = [Xk]
        for k in range(self.N - 1, -1, -1):
            t = torch.full((batch,), (k + 0.5) / self.N, device=self.device)
            mean = self.B_net(Xk, t)
            noise = torch.randn_like(Xk) * math.sqrt(2.0 * self.gamma[k].item())
            Xk_1 = mean + noise
            seq.append(Xk_1)
            Xk = Xk_1
        seq = seq[::-1]  # теперь seq[0] соответствует времени 0
        return [(k, seq[k].detach(), seq[k + 1].detach()) for k in range(self.N)]

    # ---------- лоссы для IPF ----------

    def loss_backward_on_forward_pairs(self, pairs) -> torch.Tensor:
        total = 0.0
        for (k, Xk, Xk1) in pairs:
            t_k1 = torch.full(
                (Xk1.size(0),),
                ((k + 1) + 0.5) / self.N if (k + 1) < self.N else (self.N - 0.5) / self.N,
                device=self.device,
            )
            t_k = torch.full(
                (Xk.size(0),), (k + 0.5) / self.N, device=self.device
            )
            F_Xk = self.F_net(Xk, t_k)
            F_Xk1k = self.F_net(Xk1, t_k)
            target = Xk1 + (F_Xk - F_Xk1k)
            pred = self.B_net(Xk1, t_k1)
            total += F.mse_loss(pred, target)
        return total / len(pairs)

    def loss_forward_on_backward_pairs(self, pairs) -> torch.Tensor:
        total = 0.0
        for (k, Xk, Xk1) in pairs:
            t_k = torch.full(
                (Xk.size(0),), (k + 0.5) / self.N, device=self.device
            )
            t_k1 = torch.full(
                (Xk1.size(0),),
                ((k + 1) + 0.5) / self.N if (k + 1) < self.N else (self.N - 0.5) / self.N,
                device=self.device,
            )
            B_Xk1 = self.B_net(Xk1, t_k1)
            B_Xk1k = self.B_net(Xk, t_k1)
            target = Xk + (B_Xk1 - B_Xk1k)
            pred = self.F_net(Xk, t_k)
            total += F.mse_loss(pred, target)
        return total / len(pairs)

    # ---------- сэмплирование из моста ----------

    @torch.no_grad()
    def sample_from_bridge(self, num: int, steps_per_edge: int = 2) -> np.ndarray:
        """
        Сэмплируем num точек из p_0 через обратный (backward) процесс.
        """
        Xk = torch.tensor(self.sample_prior_pN(num), device=self.device)
        for k in range(self.N - 1, -1, -1):
            t = torch.full((num,), (k + 0.5) / self.N, device=self.device)
            mean = self.B_net(Xk, t)
            noise = torch.randn_like(Xk) * math.sqrt(
                2.0 * self.gamma[k].item() / steps_per_edge
            )
            Xk = mean + noise
        return Xk.detach().cpu().numpy().astype(np.float32)

    # ---------- обучение ----------

    def train(
        self,
        ipf_iters: int = 5,
        steps_B: int = 800,
        steps_F: int = 800,
        batch: int = 2048,
        pretrain_steps: int = 300,
        pretrain_batch: int = 1024,
        print_swd: bool = True,
    ) -> None:
        """
        Полный цикл IPF-обучения.
        """
        self.pretrain_F_ou(steps=pretrain_steps, batch=pretrain_batch)

        for n in range(ipf_iters):
            # train B
            for _ in trange(steps_B, desc=f"IPF {n+1}/{ipf_iters} train B"):
                pairs = self.simulate_forward_pairs(batch)
                loss_b = self.loss_backward_on_forward_pairs(pairs)
                self.opt_B.zero_grad(set_to_none=True)
                loss_b.backward()
                nn.utils.clip_grad_norm_(self.B_net.parameters(), 1.0)
                self.opt_B.step()

            # train F
            for _ in trange(steps_F, desc=f"IPF {n+1}/{ipf_iters} train F"):
                pairs = self.simulate_backward_pairs(batch)
                loss_f = self.loss_forward_on_backward_pairs(pairs)
                self.opt_F.zero_grad(set_to_none=True)
                loss_f.backward()
                nn.utils.clip_grad_norm_(self.F_net.parameters(), 1.0)
                self.opt_F.step()

            if print_swd:
                with torch.no_grad():
                    xgen_tr = self.sample_from_bridge(
                        num=self.X_train_w.shape[0], steps_per_edge=2
                    )
                    swd_tr = sliced_wasserstein(self.X_train_w, xgen_tr)
                    print(f"[IPF {n+1}] train SWD={swd_tr:.5f}")

    # ---------- оценка ----------

    def evaluate(
        self,
        n_proj: int = 512,
        sigma: float = 1.0,
        steps_per_edge: int = 3,
        max_eval_samples: int = 5000,
    ) -> Dict[str, Any]:
        """
        Сэмплирует синтетику и считает SWD, MMD, KS по train (+ val, если есть).

        max_eval_samples — максимум точек для расчёта метрик (subsample),
        чтобы не взрывать память в MMD (O(n^2) ядро).
        """
        # ---- train ----
        n_train = self.X_train_w.shape[0]
        syn_train_w_full = self.sample_from_bridge(
            num=n_train, steps_per_edge=steps_per_edge
        )

        X_train_full = self.X_train_w

        if (max_eval_samples is not None) and (n_train > max_eval_samples):
            idx = np.random.choice(n_train, size=max_eval_samples, replace=False)
            X_train_eval = X_train_full[idx]
            syn_train_eval = syn_train_w_full[idx]
        else:
            X_train_eval = X_train_full
            syn_train_eval = syn_train_w_full

        swd_train = sliced_wasserstein(X_train_eval, syn_train_eval, n_proj=n_proj)
        mmd_train = mmd_rbf(X_train_eval, syn_train_eval, sigma=sigma)
        ks_train = ks_per_feature(X_train_eval, syn_train_eval)

        results: Dict[str, Any] = {
            "syn_train_w": syn_train_w_full,
            "swd_train": swd_train,
            "mmd_train": mmd_train,
            "ks_train": ks_train,
        }

        # ---- val ----
        if self.X_val_w is not None:
            n_val = self.X_val_w.shape[0]
            syn_val_w_full = self.sample_from_bridge(
                num=n_val, steps_per_edge=steps_per_edge
            )
            X_val_full = self.X_val_w

            if (max_eval_samples is not None) and (n_val > max_eval_samples):
                idx = np.random.choice(n_val, size=max_eval_samples, replace=False)
                X_val_eval = X_val_full[idx]
                syn_val_eval = syn_val_w_full[idx]
            else:
                X_val_eval = X_val_full
                syn_val_eval = syn_val_w_full

            swd_val = sliced_wasserstein(X_val_eval, syn_val_eval, n_proj=n_proj)
            mmd_val = mmd_rbf(X_val_eval, syn_val_eval, sigma=sigma)
            ks_val = ks_per_feature(X_val_eval, syn_val_eval)

            results.update(
                {
                    "syn_val_w": syn_val_w_full,
                    "swd_val": swd_val,
                    "mmd_val": mmd_val,
                    "ks_val": ks_val,
                }
            )
        else:
            results.update(
                {
                    "syn_val_w": None,
                    "swd_val": None,
                    "mmd_val": None,
                    "ks_val": None,
                }
            )

        return results




# ----------------- Метрики (как отдельные функции) -----------------

def sliced_wasserstein(
    X: np.ndarray, Y: np.ndarray, n_proj: int = 256
) -> float:
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    Xc, Yc = X_t - X_t.mean(0), Y_t - Y_t.mean(0)
    thetas = torch.randn(n_proj, X_t.shape[1])
    thetas = thetas / thetas.norm(dim=1, keepdim=True)

    sw2 = 0.0
    for theta in thetas:
        x1 = Xc @ theta
        y1 = Yc @ theta
        x1, _ = torch.sort(x1)
        y1, _ = torch.sort(y1)
        sw2 += F.mse_loss(x1, y1, reduction="mean")
    return float(sw2 / n_proj)

import numpy as np
from scipy.spatial.distance import cdist

def _rbf_kernel(d2, sigma):
    # d2 — попарные квадраты расстояний; sigma — ширина ядра
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 1.0
    return np.exp(-d2 / (2.0 * (sigma ** 2)))

def mmd_rbf(X: np.ndarray,
            Y: np.ndarray,
            sigma: float | None = None,
            sigmas: list[float] | None = None,
            max_samples: int =2500,
            random_state: int = 0) -> float:
    """
    Biased MMD^2 с RBF-ядром.
    - если sigma is None → используем median heuristic
    - если sigmas задан → MK-MMD (усредняем по нескольким сигмам)
    """

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    sigma=None
    if max_samples is not None:
        if X.shape[0] > max_samples:
            X = X[rng.choice(X.shape[0], max_samples, replace=False)]
        if Y.shape[0] > max_samples:
            Y = Y[rng.choice(Y.shape[0], max_samples, replace=False)]

    Z = np.vstack([X, Y])
    # попарные расстояния (не квадраты)
    D = cdist(Z, Z, metric="euclidean")
    # квадраты расстояний
    D2 = D ** 2

    # median heuristic по real∪syn, игнорируя нули на диагонали
    if sigma is None and not sigmas:
        med = np.median(D[D > 0])
        sigma = float(med) if np.isfinite(med) and med > 0 else 1.0

    def _mmd_with_sigma(s):
        # блоки попарных расстояний
        n, m = X.shape[0], Y.shape[0]
        K = _rbf_kernel(D2, s)

        Kxx = K[:n, :n]
        Kyy = K[n:, n:]
        Kxy = K[:n, n:]

        # biased MMD^2
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
        return float(max(mmd2, 0.0))

    if sigmas:
        vals = [_mmd_with_sigma(s) for s in sigmas]
        return float(np.mean(vals))

    return _mmd_with_sigma(sigma)



def ks_per_feature(X_real: np.ndarray, X_syn: np.ndarray) -> List[Tuple[int, float, float]]:
    out: List[Tuple[int, float, float]] = []
    for j in range(X_real.shape[1]):
        stat, p = stats.ks_2samp(X_real[:, j], X_syn[:, j])
        out.append((j, float(stat), float(p)))
    return out
