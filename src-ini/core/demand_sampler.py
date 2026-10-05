"""把 HFVRP 名义需求 d̄_j 包成 stochastic demand: 返回 (a, n, w, volume) 4 张表.

三种分布 (env / 配置项 demand_dist):
    lognormal (默认) — log(D_j) ~ N(μ, σ²), σ² = ln(1+cv²), μ = ln(d̄_j) - σ²/2
        正性保证, 取均值=d̄_j, 重尾贴合配送负载.
    normal               — D_j ~ N(d̄_j, (cv·d̄_j)²) | D_j > 0 (拒绝重采样)
    uniform              — D_j ~ Uniform[(1-cv)·d̄_j, (1+cv)·d̄_j], 截断到 ≥ 0

active 标记: Bernoulli(active_prob) 独立采样 (默认 1.0 = 客户每个场景都需要服务).
若 active=0, 则 (n, w, volume) 同步置 0.

`n` 用 round → int，并保证 active 客户至少为 1；`volume` 与 `w` 保留 float。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _sample_positive_normal(
    rng: np.random.Generator,
    mean: np.ndarray,
    sd: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """从逐元素正截断 normal 抽样，不用 clip/epsilon 制造零体积。"""
    volume = rng.normal(loc=mean, scale=sd, size=size)
    nonpositive = volume <= 0.0
    while np.any(nonpositive):
        volume[nonpositive] = rng.normal(
            loc=np.broadcast_to(mean, size)[nonpositive],
            scale=np.broadcast_to(sd, size)[nonpositive],
        )
        nonpositive = volume <= 0.0
    return volume


def sample_stochastic_demand(
    demand_nominal: np.ndarray,
    num_scenarios: int,
    dist: str = "lognormal",
    cv: float = 0.3,
    active_prob: float = 1.0,
    seed: Optional[int] = None,
) -> dict:
    """返回 dict: {'a','n','w','volume'} 各 shape (num_scenarios, num_customers).

    `cv` = 标准差 / 均值 (coefficient of variation), 控制采样幅度.
    `seed` = 与 VRPConfig.scenario_seed 联动, 保证复现.
    """
    rng = np.random.default_rng(seed)
    d_bar = np.asarray(demand_nominal, dtype=np.float64)
    if d_bar.ndim != 1:
        raise ValueError(f"demand_nominal 应为 1-D, 实际 shape={d_bar.shape}")
    if not np.isfinite(d_bar).all():
        raise ValueError("demand_nominal 必须全部为有限值")
    if np.any(d_bar <= 0.0):
        j = int(np.argwhere(d_bar <= 0.0)[0, 0])
        raise ValueError(
            f"demand_nominal[{j}]={d_bar[j]:.12g} 必须 > 0；"
            "active 客户不允许零体积"
        )
    nc = int(d_bar.size)
    S = int(num_scenarios)
    if S < 1:
        raise ValueError(f"num_scenarios 必须 ≥ 1, 收到 {S}")
    if cv < 0:
        raise ValueError(f"cv 必须 ≥ 0, 收到 {cv}")
    if not 0.0 <= active_prob <= 1.0:
        raise ValueError(f"active_prob 必须位于 [0,1], 收到 {active_prob}")

    if cv == 0.0:
        volume = np.tile(d_bar, (S, 1))
    else:
        dist_lc = dist.strip().lower()
        if dist_lc == "lognormal":
            sigma2 = np.log1p(cv ** 2)
            sigma = np.sqrt(sigma2)
            mu = np.where(d_bar > 0, np.log(np.maximum(d_bar, 1e-12)) - 0.5 * sigma2, 0.0)
            mu_mat = np.tile(mu, (S, 1))
            eps = rng.standard_normal(size=(S, nc))
            volume = np.where(d_bar > 0, np.exp(mu_mat + sigma * eps), 0.0)
        elif dist_lc == "normal":
            sd = cv * d_bar
            volume = _sample_positive_normal(rng, d_bar, sd, (S, nc))
        elif dist_lc == "uniform":
            low = np.clip((1.0 - cv) * d_bar, 0.0, None)
            high = (1.0 + cv) * d_bar
            volume = rng.uniform(low=low, high=high, size=(S, nc))
        else:
            raise ValueError(
                f"未知 demand_dist={dist!r}; 支持: lognormal/normal/uniform"
            )

    if active_prob >= 1.0:
        a = np.ones((S, nc), dtype=int)
    else:
        a = (rng.random(size=(S, nc)) < float(active_prob)).astype(int)
        volume = volume * a

    n = np.maximum(1, np.round(volume).astype(int)) * a
    w = volume.copy()

    return {"a": a, "n": n, "w": w.astype(np.float64), "volume": volume.astype(np.float64)}
