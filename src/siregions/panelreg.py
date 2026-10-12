"""Панельная регрессия внутри групп с кластерными ошибками.

Зачем свой код, а не linearmodels: в репозитории нужен ровно within-оценщик с
одно- и двусторонней кластеризацией, и важнее иметь его читаемым и покрытым
тестами, чем тянуть зависимость. Поправки на степени свободы те же, что в Stata
areg/reghdfe: G/(G-1) * (N-1)/(N-K), где K включает поглощённые эффекты.

Двусторонняя кластеризация — Cameron–Gelbach–Miller: V = V1 + V2 - V12.
Если V оказывается не положительно определённой (бывает при малом числе
кластеров), отрицательные собственные значения зануляются, и об этом
сообщается в поле psd.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


def demean(df: pd.DataFrame, cols: list[str], absorb: list[str],
           tol: float = 1e-10, max_iter: int = 200) -> tuple[pd.DataFrame, int]:
    """Вычитает групповые средние по каждому измерению absorb (альтернирующие проекции).

    Для одного измерения это один проход, для двух и более — итерации до сходимости.
    Возвращает таблицу остатков и число поглощённых параметров (сумма уровней минус
    число измерений, потому что константа общая).
    """
    X = df[cols].astype(float).copy()
    if not absorb:
        return X - X.mean(), 0
    keys = [df[a] for a in absorb]
    for _ in range(max_iter if len(absorb) > 1 else 1):
        prev = X.to_numpy().copy()
        for k in keys:
            X = X - X.groupby(k).transform("mean")
        if len(absorb) == 1 or np.nanmax(np.abs(X.to_numpy() - prev)) < tol:
            break
    k_absorbed = sum(df[a].nunique() for a in absorb) - (len(absorb) - 1)
    return X, k_absorbed


@dataclass
class Fit:
    params: pd.Series
    se: pd.Series
    tstat: pd.Series
    pvalue: pd.Series
    ci_low: pd.Series
    ci_high: pd.Series
    nobs: int
    vcov: pd.DataFrame = field(default_factory=pd.DataFrame)
    dof_t: int = 0
    n_clusters: dict[str, int] = field(default_factory=dict)
    r2_within: float = float("nan")
    psd: bool = True
    note: str = ""

    def wald(self, terms: list[str]) -> dict:
        """Совместный тест H0: все перечисленные коэффициенты равны нулю (хи-квадрат и F)."""
        b = self.params[terms].to_numpy()
        V = self.vcov.loc[terms, terms].to_numpy()
        stat = float(b @ np.linalg.pinv(V) @ b)
        q = len(terms)
        return {"chi2": stat, "df": q, "p_chi2": float(stats.chi2.sf(stat, q)),
                "F": stat / q, "p_F": float(stats.f.sf(stat / q, q, max(self.dof_t, 1)))}

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "coef": self.params, "se": self.se, "t": self.tstat, "p": self.pvalue,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
        })

    def line(self, term: str, digits: int = 4) -> str:
        t = self.table().loc[term]
        return (f"{term}: {t.coef:.{digits}f} (ст.ош. {t.se:.{digits}f}, "
                f"p = {t.p:.3g}, ди [{t.ci_low:.{digits}f}; {t.ci_high:.{digits}f}], N = {self.nobs})")


def fit(df: pd.DataFrame, y: str, x: list[str], absorb: list[str] | None = None,
        cluster: list[str] | None = None, add_const: bool | None = None) -> Fit:
    """МНК с поглощёнными эффектами и кластерными ошибками.

    absorb  — колонки групп, средние по которым вычитаются (регион, месяц);
    cluster — колонки кластеров (одна или две);
    add_const — по умолчанию константа добавляется, только если ничего не поглощается.
    """
    absorb = list(absorb or [])
    cluster = list(cluster or [])
    cols = [y] + list(x)
    use = df.dropna(subset=cols + absorb + cluster).copy()
    if add_const is None:
        add_const = not absorb
    Z, k_abs = demean(use, cols, absorb)
    yy = Z[y].to_numpy()
    names = list(x)
    XX = Z[list(x)].to_numpy()
    if add_const:
        XX = np.column_stack([np.ones(len(use)), XX])
        names = ["const"] + names
    n, k = XX.shape
    xtx_inv = np.linalg.pinv(XX.T @ XX)
    beta = xtx_inv @ (XX.T @ yy)
    resid = yy - XX @ beta
    k_total = k + k_abs
    dof = max(n - k_total, 1)

    def meat(key: pd.Series) -> np.ndarray:
        g = key.to_numpy()
        order = np.argsort(g, kind="stable")
        S = np.zeros((k, k))
        for _, idx in pd.Series(np.arange(n)[order]).groupby(pd.Series(g[order]).values):
            ii = idx.to_numpy()
            s = XX[ii].T @ resid[ii]
            S += np.outer(s, s)
        return S

    n_clusters: dict[str, int] = {}
    if not cluster:
        V = xtx_inv * (resid @ resid) / dof
        dof_t = dof
    else:
        parts = []
        for c in cluster:
            G = use[c].nunique()
            n_clusters[c] = int(G)
            adj = G / max(G - 1, 1) * (n - 1) / dof
            parts.append(adj * xtx_inv @ meat(use[c]) @ xtx_inv)
        if len(cluster) == 1:
            V = parts[0]
        else:
            joint = use[cluster[0]].astype(str) + "|" + use[cluster[1]].astype(str)
            G12 = joint.nunique()
            adj = G12 / max(G12 - 1, 1) * (n - 1) / dof
            V = parts[0] + parts[1] - adj * xtx_inv @ meat(joint) @ xtx_inv
        dof_t = max(min(n_clusters.values()) - 1, 1)
    w, Q = np.linalg.eigh(V)
    psd = bool((w >= -1e-12).all())
    if not psd:
        V = Q @ np.diag(np.clip(w, 0, None)) @ Q.T
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = beta / se
    crit = stats.t.ppf(0.975, dof_t)
    idx = pd.Index(names)
    tss = float(((yy - yy.mean()) ** 2).sum())
    return Fit(
        params=pd.Series(beta, idx), se=pd.Series(se, idx),
        tstat=pd.Series(tstat, idx),
        pvalue=pd.Series(2 * stats.t.sf(np.abs(tstat), dof_t), idx),
        ci_low=pd.Series(beta - crit * se, idx), ci_high=pd.Series(beta + crit * se, idx),
        nobs=int(n), vcov=pd.DataFrame(V, index=idx, columns=idx), dof_t=int(dof_t),
        n_clusters=n_clusters,
        r2_within=float(1 - (resid @ resid) / tss) if tss > 0 else float("nan"),
        psd=psd,
        note=f"поглощено: {', '.join(absorb) or 'нет'}; кластеры: {', '.join(cluster) or 'нет'}",
    )


def nw_ols(y: pd.Series, X: pd.DataFrame, lags: int | None = None) -> Fit:
    """Временной ряд: МНК с ошибками Ньюи–Уэста (лаг по правилу 4*(T/100)^(2/9))."""
    d = pd.concat([y.rename("__y"), X], axis=1).dropna()
    yy = d["__y"].to_numpy()
    names = ["const"] + list(X.columns)
    XX = np.column_stack([np.ones(len(d)), d[list(X.columns)].to_numpy()])
    n, k = XX.shape
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    xtx_inv = np.linalg.pinv(XX.T @ XX)
    beta = xtx_inv @ (XX.T @ yy)
    u = yy - XX @ beta
    S = (XX * u[:, None]).T @ (XX * u[:, None])
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = (XX[l:] * u[l:, None]).T @ (XX[:-l] * u[:-l, None])
        S += w * (G + G.T)
    dof = max(n - k, 1)
    V = xtx_inv @ S @ xtx_inv * n / dof
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    crit = stats.t.ppf(0.975, dof)
    idx = pd.Index(names)
    tss = float(((yy - yy.mean()) ** 2).sum())
    return Fit(
        params=pd.Series(beta, idx), se=pd.Series(se, idx), tstat=pd.Series(t, idx),
        pvalue=pd.Series(2 * stats.t.sf(np.abs(t), dof), idx),
        ci_low=pd.Series(beta - crit * se, idx), ci_high=pd.Series(beta + crit * se, idx),
        nobs=int(n), vcov=pd.DataFrame(V, index=idx, columns=idx), dof_t=int(dof),
        r2_within=float(1 - (u @ u) / tss) if tss > 0 else float("nan"),
        note=f"Ньюи–Уэст, лаг {lags}",
    )


def mde(se: float, power: float = 0.8, alpha: float = 0.05) -> float:
    """Минимально обнаружимый эффект при заданной мощности (двусторонний тест)."""
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)
