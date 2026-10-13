"""Разложение решений по ключевой ставке на неожиданную и ожидаемую часть.

Зачем. Гипотеза постоянного дохода (Холл, 1978) утверждает, что потребление меняется
только от новостей. Дата заседания известна заранее, и бо́льшая часть решения обычно уже
в ожиданиях, поэтому реагировать должна не вся величина изменения, а только её сюрприз.
В исследовании rate.py реакция меряется на изменение целиком — здесь измерение
исправляется.

Мера сюрприза. Классическая схема высокочастотной идентификации (Kuttner 2001;
Gürkaynak, Sack, Swanson 2005): сюрприз — это сдвиг короткого конца рыночной кривой
в день решения, а не разница между решением и опросом аналитиков. Котировки обновляются
непрерывно и стоят участникам денег, в отличие от ответа в анкете.

Источник. Кривая бескупонной доходности ОФЗ Банка России (КБД), дневная, срок 0,25 года:
cbr.ru/hd_base/zcyc_params/. Решение объявляется в 13:30 МСК, торги закрываются позже,
поэтому дневное значение КБД за день решения новость уже содержит, а за предыдущий
рабочий день — ещё нет. Отсюда базовое окно (−1, 0). Окно (−1, +1) — проверка.

Ограничение, которое надо держать в голове. Доходность на сроке 0,25 года — это средняя
ожидаемая ставка на квартал плюс премия за срок. Разовое изменение уровня ставки, которое
рынок считает устойчивым, переносится в неё почти один к одному, но не ровно: для
решений, которые рынок считает временными, сюрприз по этой мере занижен. Премия за срок
за один день меняется мало, поэтому в разности она в основном сокращается.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import panelreg as pr
from .rate import controls  # noqa: F401  (переэкспорт для удобства)

TENOR = "y0_25"


def load_curve(path: str | Path) -> pd.DataFrame:
    """Дневная КБД ОФЗ: индекс — дата, столбцы — срок до погашения в годах."""
    c = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return c.astype(float)


def decompose(decisions: pd.DataFrame, curve: pd.DataFrame, tenor: str = TENOR,
              post: int = 0, max_gap: int = 7) -> pd.DataFrame:
    """На каждое решение: сюрприз (сдвиг кривой) и ожидаемая часть (остаток изменения).

    «До» — последний торговый день строго раньше дня решения. «После» — день решения
    (post = 0) или следующий торговый день (post = 1). Если котировок в окне нет или
    разрыв больше max_gap дней (остановка торгов в феврале–марте 2022-го), строка
    помечается в столбце «нет_котировок» и в оценку не входит.
    """
    idx = curve.index
    rows = []
    for _, r in decisions.iterrows():
        d = r["decision_date"]
        before = idx[idx < d]
        after = idx[idx >= d]
        y_pre = y_post = np.nan
        if len(before) and len(after) > post:
            t0, t1 = before[-1], after[post]
            if (d - t0).days <= max_gap and (t1 - d).days <= max_gap:
                y_pre, y_post = float(curve.loc[t0, tenor]), float(curve.loc[t1, tenor])
        sur = y_post - y_pre
        rows.append({"decision_date": d, "kind": r["kind"], "change": float(r["change"]),
                     "y_pre": y_pre, "y_post": y_post, "сюрприз": sur,
                     "ожидаемое": float(r["change"]) - sur,
                     "нет_котировок": not np.isfinite(sur)})
    out = pd.DataFrame(rows)
    return out


def align_parts(parts: pd.DataFrame, weeks: pd.DatetimeIndex, keep=None) -> pd.DataFrame:
    """Сюрприз и ожидаемая часть, отнесённые к той же неделе, что и в rate.align."""
    out = pd.DataFrame(0.0, index=weeks, columns=["сюрприз", "ожидаемое"])
    for _, r in parts.iterrows():
        if keep is not None and not keep(r):
            continue
        if r["нет_котировок"]:
            continue
        wk = weeks[weeks >= r["decision_date"]]
        if len(wk):
            out.loc[wk[0], "сюрприз"] += r["сюрприз"]
            out.loc[wk[0], "ожидаемое"] += r["ожидаемое"]
    return out


def _design(series: pd.Series, shocks: pd.DataFrame, ctrl: pd.DataFrame, lags: int) -> pd.DataFrame:
    g = series.diff()
    extra = [g.shift(l).rename(f"g{l}") for l in range(1, lags + 1)]
    for c in shocks.columns:
        extra += [shocks[c].shift(l).rename(f"{c}{l}") for l in range(1, lags + 1)]
    return pd.concat([shocks, ctrl] + extra, axis=1)


def local_projection(series: pd.Series, shocks: pd.DataFrame, ctrl: pd.DataFrame,
                     hmax: int = 4, lags: int = 2) -> pd.DataFrame:
    """Отклик на сюрприз и на ожидаемую часть — в одной регрессии, в % на п.п."""
    rows = []
    for h in range(hmax + 1):
        yh = series.shift(-h) - series.shift(1)
        f = pr.nw_ols(yh, _design(series, shocks, ctrl, lags), lags=h + 1)
        row = {"h": h, "n": int(f.nobs)}
        for c in shocks.columns:
            row[f"коэф_{c}"] = float(f.params[c] * 100)
            row[f"ст_ош_{c}"] = float(f.se[c] * 100)
            row[f"p_{c}"] = float(f.pvalue[c])
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_p(series: pd.Series, shocks: pd.DataFrame, ctrl: pd.DataFrame, h: int,
                  term: str = "сюрприз", n_perm: int = 500, lags: int = 2,
                  seed: int = 0) -> dict:
    """Те же пары (сюрприз, ожидаемое), случайно разложенные по неделям."""
    rng = np.random.default_rng(seed)
    mask = (shocks != 0).any(axis=1).to_numpy()
    vals = shocks[mask].to_numpy()
    weeks = shocks.index
    yh = series.shift(-h) - series.shift(1)
    obs = float(pr.nw_ols(yh, _design(series, shocks, ctrl, lags), lags=h + 1).params[term] * 100)
    cnt = 0
    for _ in range(n_perm):
        fake = pd.DataFrame(0.0, index=weeks, columns=shocks.columns)
        pos = rng.choice(len(weeks), size=len(vals), replace=False)
        fake.iloc[pos] = vals
        c = float(pr.nw_ols(yh, _design(series, fake, ctrl, lags), lags=h + 1).params[term] * 100)
        cnt += abs(c) >= abs(obs)
    return {"h": h, "term": term, "coef": obs, "p_perm": cnt / n_perm,
            "n_perm": n_perm, "n_events": len(vals)}


def jackknife(series: pd.Series, parts: pd.DataFrame, weeks: pd.DatetimeIndex,
              ctrl: pd.DataFrame, keep=None, term: str = "сюрприз", h: int = 0,
              lags: int = 2) -> pd.DataFrame:
    """Исключение по одному заседанию: насколько коэффициент держится на отдельном событии."""
    used = parts[~parts["нет_котировок"]]
    if keep is not None:
        used = used[used.apply(keep, axis=1)]
    rows = []
    for d in used["decision_date"]:
        sub = parts[parts["decision_date"] != d]
        sh = align_parts(sub, weeks, keep)
        f = pr.nw_ols(series.shift(-h) - series.shift(1), _design(series, sh, ctrl, lags),
                      lags=h + 1)
        rows.append({"без": d.date().isoformat(), "коэф": float(f.params[term] * 100)})
    return pd.DataFrame(rows)
