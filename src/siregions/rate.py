"""Кто первым реагирует на ставку: недельные траты по возрастным группам вокруг решений ЦБ.

Данные: СберИндекс, «Потребительская активность по категориям товаров в разрезе возрастов»,
недельный индекс по четырём возрастным группам и по всем возрастам, только Россия целиком.

Две конструкции, на которых всё держится.

1. Относительная мера. Вопрос «кто первым» — про разницу между группами, а не про уровень
   трат. Поэтому основная переменная — отклонение логарифма активности группы от логарифма
   активности всех возрастов. Общая часть (праздники, погода, выплаты, сам макрошок) уходит,
   остаётся различие в реакции. Это тот же панельный приём, что и в остальных работах.

2. Локальные проекции. Для каждого горизонта h оценивается отдельная регрессия
   y(t+h) − y(t−1) на величину изменения ставки в неделе t, с недельными сезонными эффектами,
   позицией недели в месяце (зарплатные циклы), трендом и лагами. Ошибки Ньюи–Уэста.

Почему к p-значениям регрессии добавлена перестановочная проверка: при перекрывающихся окнах
локальные проекции дают ложные срабатывания на дальних горизонтах — плацебо-сдвиги это
показывают прямо. Перестановочный тест раскладывает те же самые изменения ставки по случайным
неделям и считает, как часто случайная расстановка даёт коэффициент не меньше наблюдаемого.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import data as data_mod
from . import panelreg as pr

ALL = "Все возрасты"


def load_decisions(path: str | Path) -> pd.DataFrame:
    """Реестр решений ЦБ: дата решения, дата вступления в силу, новая ставка, изменение.

    Даты сверены с сайтом ЦБ (календарь заседаний) и с таблицей «Гаранта» по датам
    вступления в силу. Решения о сохранении ставки входят в реестр с нулевым изменением:
    на оценку они не влияют, но показывают, что ни одно заседание не пропущено.
    """
    d = pd.read_csv(path)
    d["decision_date"] = pd.to_datetime(d["decision_date"])
    d["effective_date"] = pd.to_datetime(d["effective_date"])
    return d.sort_values("decision_date").reset_index(drop=True)


def build(cfg: dict) -> dict:
    """Недельная панель логарифмов активности и ряд изменений ставки, выровненный по неделям."""
    long = data_mod.read(cfg["datasets"]["activity"])
    w = long.pivot_table(index="period", columns="age", values="value").sort_index()
    if cfg.get("start"):
        w = w[w.index >= pd.Timestamp(cfg["start"])]
    y = np.log(w)
    dec = load_decisions(cfg["events"])
    return {"index": w, "log": y, "decisions": dec, "weeks": w.index}


def align(decisions: pd.DataFrame, weeks: pd.DatetimeIndex, keep=None) -> pd.Series:
    """Изменение ставки, отнесённое к первой неделе, которая кончается не раньше дня решения.

    Недели в выгрузке помечены воскресеньем. Решение принимается в пятницу, поэтому новость
    попадает в ту же неделю — это и есть горизонт h = 0.
    """
    ev = pd.Series(0.0, index=weeks)
    for _, r in decisions.iterrows():
        if keep is not None and not keep(r):
            continue
        wk = weeks[weeks >= r["decision_date"]]
        if len(wk):
            ev.loc[wk[0]] += float(r["change"])
    return ev


def controls(weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Сезонность недели года, позиция недели в месяце и линейный тренд."""
    woy = pd.get_dummies(pd.Series(weeks.isocalendar().week.values, index=weeks),
                         prefix="w", drop_first=True).astype(float)
    pos = pd.get_dummies(pd.Series(np.minimum((weeks.day - 1) // 7, 3), index=weeks),
                         prefix="pos", drop_first=True).astype(float)
    trend = pd.Series(np.arange(len(weeks)) / 52.0, index=weeks, name="trend")
    return pd.concat([woy, pos, trend], axis=1)


def _design(series: pd.Series, shock: pd.Series, ctrl: pd.DataFrame, lags: int) -> pd.DataFrame:
    g = series.diff()
    extra = [g.shift(l).rename(f"g{l}") for l in range(1, lags + 1)]
    extra += [shock.shift(l).rename(f"s{l}") for l in range(1, lags + 1)]
    return pd.concat([shock.rename("shock"), ctrl] + extra, axis=1)


def local_projection(series: pd.Series, shock: pd.Series, ctrl: pd.DataFrame,
                     hmax: int = 8, lags: int = 2) -> pd.DataFrame:
    """Отклик на горизонтах 0..hmax в процентах на процентный пункт изменения ставки."""
    rows = []
    for h in range(hmax + 1):
        yh = series.shift(-h) - series.shift(1)
        f = pr.nw_ols(yh, _design(series, shock, ctrl, lags), lags=h + 1)
        rows.append({"h": h, "coef": float(f.params["shock"] * 100),
                     "se": float(f.se["shock"] * 100), "p": float(f.pvalue["shock"]),
                     "ci_low": float(f.ci_low["shock"] * 100),
                     "ci_high": float(f.ci_high["shock"] * 100), "n": int(f.nobs)})
    return pd.DataFrame(rows)


def permutation_p(series: pd.Series, shock: pd.Series, ctrl: pd.DataFrame, h: int,
                  n_perm: int = 500, lags: int = 2, seed: int = 0) -> dict:
    """Те же изменения ставки, случайные недели: доля расстановок с |коэффициентом| не меньше."""
    rng = np.random.default_rng(seed)
    sizes = shock[shock != 0].to_numpy()
    obs = float(pr.nw_ols(series.shift(-h) - series.shift(1),
                          _design(series, shock, ctrl, lags), lags=h + 1).params["shock"] * 100)
    weeks = shock.index
    cnt = 0
    for _ in range(n_perm):
        fake = pd.Series(0.0, index=weeks)
        fake.iloc[rng.choice(len(weeks), size=len(sizes), replace=False)] = sizes
        c = float(pr.nw_ols(series.shift(-h) - series.shift(1),
                            _design(series, fake, ctrl, lags), lags=h + 1).params["shock"] * 100)
        cnt += abs(c) >= abs(obs)
    return {"h": h, "coef": obs, "p_perm": cnt / n_perm, "n_perm": n_perm, "n_events": len(sizes)}


def series_for(y: pd.DataFrame, group: str, mode: str) -> pd.Series:
    """mode = relative — отклонение от всех возрастов, absolute — сам логарифм активности."""
    return y[group] - y[ALL] if mode == "relative" else y[group]


KEEPS = {
    "все решения": lambda r: r["change"] != 0,
    "плановые": lambda r: r["change"] != 0 and r["kind"] == "плановое",
    "плановые повышения": lambda r: r["change"] > 0 and r["kind"] == "плановое",
    "плановые снижения": lambda r: r["change"] < 0 and r["kind"] == "плановое",
    "плановые, без 2022": lambda r: r["change"] != 0 and r["kind"] == "плановое"
                                    and r["decision_date"].year != 2022,
}


def irf_table(b: dict, groups: list[str], mode: str = "relative", variant: str = "все решения",
              hmax: int = 8, lags: int = 2) -> pd.DataFrame:
    ctrl = controls(b["weeks"])
    ev = align(b["decisions"], b["weeks"], KEEPS[variant])
    out = []
    for g in groups:
        t = local_projection(series_for(b["log"], g, mode), ev, ctrl, hmax, lags)
        out.append(t.assign(группа=g, вариант=variant, мера=mode))
    return pd.concat(out, ignore_index=True)


def robustness(b: dict, groups: list[str], horizons=(0, 1), mode: str = "relative",
               n_perm: int = 500, lags: int = 2, seed: int = 0,
               placebo_shifts=(-6, 4)) -> pd.DataFrame:
    """Сводка: по вариантам решений и по группам — коэффициент, p Ньюи–Уэста, p перестановок.

    Плацебо-сдвиги сдвигают весь набор событий на заданное число недель: содержательной
    реакции там быть не может, и если она есть, значит машинка ловит не то.
    """
    ctrl = controls(b["weeks"])
    rows = []
    for vname, keep in KEEPS.items():
        ev = align(b["decisions"], b["weeks"], keep)
        for g in groups:
            s = series_for(b["log"], g, mode)
            lp = local_projection(s, ev, ctrl, max(horizons), lags).set_index("h")
            for h in horizons:
                perm = permutation_p(s, ev, ctrl, h, n_perm, lags, seed)
                rows.append({"вариант": vname, "группа": g, "h": h,
                             "коэф": lp.loc[h, "coef"], "ст_ош": lp.loc[h, "se"],
                             "p_nw": lp.loc[h, "p"], "p_perm": perm["p_perm"],
                             "событий": perm["n_events"]})
    ev_all = align(b["decisions"], b["weeks"], KEEPS["все решения"])
    for sh in placebo_shifts:
        fake = ev_all.shift(sh).fillna(0.0)
        for g in groups:
            s = series_for(b["log"], g, mode)
            lp = local_projection(s, fake, ctrl, max(horizons), lags).set_index("h")
            for h in horizons:
                rows.append({"вариант": f"плацебо {sh:+d} нед", "группа": g, "h": h,
                             "коэф": lp.loc[h, "coef"], "ст_ош": lp.loc[h, "se"],
                             "p_nw": lp.loc[h, "p"], "p_perm": np.nan,
                             "событий": int((fake != 0).sum())})
    return pd.DataFrame(rows)


def jackknife(b: dict, group: str, variant: str = "плановые повышения", h: int = 0,
              mode: str = "relative", lags: int = 2) -> pd.DataFrame:
    """Исключение по одному событию: не держится ли результат на одном-двух заседаниях."""
    ctrl = controls(b["weeks"])
    dec = b["decisions"][b["decisions"].apply(KEEPS[variant], axis=1)]
    s = series_for(b["log"], group, mode)
    rows = []
    for d in dec["decision_date"]:
        keep = lambda r, d=d: KEEPS[variant](r) and r["decision_date"] != d
        ev = align(b["decisions"], b["weeks"], keep)
        f = pr.nw_ols(s.shift(-h) - s.shift(1), _design(s, ev, ctrl, lags), lags=h + 1)
        rows.append({"исключено": str(d.date()), "коэф": float(f.params["shock"] * 100),
                     "p": float(f.pvalue["shock"])})
    return pd.DataFrame(rows).sort_values("коэф")


def event_window(b: dict, groups: list[str], variant: str = "плановые повышения",
                 pre: int = 4, post: int = 8, mode: str = "relative") -> pd.DataFrame:
    """Среднее окно вокруг событий без регрессии: простая картинка для проверки глазами.

    Каждое событие взвешивается величиной изменения ставки, отклонения считаются от недели
    перед решением. Сезонность здесь не убирается — это иллюстрация, а не оценка.
    """
    weeks = b["weeks"]
    dec = b["decisions"][b["decisions"].apply(KEEPS[variant], axis=1)]
    rows = []
    for g in groups:
        s = series_for(b["log"], g, mode)
        acc = {k: [] for k in range(-pre, post + 1)}
        for _, r in dec.iterrows():
            wk = weeks[weeks >= r["decision_date"]]
            if not len(wk):
                continue
            i = weeks.get_loc(wk[0])
            if i - pre - 1 < 0 or i + post >= len(weeks):
                continue
            base = s.iloc[i - 1]
            for k in range(-pre, post + 1):
                acc[k].append((s.iloc[i + k] - base) / float(r["change"]) * 100)
        for k, v in acc.items():
            if v:
                rows.append({"группа": g, "k": k, "среднее": float(np.mean(v)),
                             "медиана": float(np.median(v)), "n": len(v)})
    return pd.DataFrame(rows)
