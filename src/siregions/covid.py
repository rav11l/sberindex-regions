"""Что детектор видит и чего не видит: общий шок, идиосинкратический и эхо.

Панель внутреннего туризма СберИндекса (83 региона, 01.2019–04.2022, прирост год к году в %)
даёт то, чего не было в «Шоке по расписанию»: реальный шок известной даты и огромной величины.

Разложение простое. Общая часть — медиана по регионам в каждом месяце. Идиосинкратическая —
отклонение региона от этой медианы. Панельный детектор по построению работает со второй частью,
поэтому общий шок он не видит: это цена приёма, а не ошибка.

Эхо — отдельная беда рядов «год к году»: провал 2020 года возвращается в 2021 году как всплеск
на той же дате. Детектор, не знающий об этом, принимает эхо за новое событие.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .changepoint.detectors import REGISTRY


def decompose(wide: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Общая часть (медиана по рядам) и отклонения от неё."""
    common = wide.median(axis=0)
    return common, wide.sub(common, axis=1)


def run_detectors(d: pd.DataFrame, eval_cols, specs: dict) -> dict[str, pd.DataFrame]:
    """Матрицы тревог по детекторам: строки — ряды, столбцы — месяцы окна оценки."""
    out = {}
    for name, spec in specs.items():
        fn = REGISTRY[spec.get("fn", name)]
        params = {k: v for k, v in spec.items() if k not in {"fn", "enabled", "use_news"}}
        out[name] = fn(d, list(eval_cols), **params)
    return out


def alarms_by_month(alarms: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Доля рядов с тревогой в каждом месяце, по детекторам."""
    return pd.DataFrame({k: v.mean(axis=0) for k, v in alarms.items()})


def echo_months(common: pd.Series, eval_cols, k: float = 3.0, lag: int = 12) -> pd.Series:
    """Месяцы, в которых ряд «год к году» механически отражает прошлогоднее событие.

    Признак считается по общей части: если год назад изменение общей части превышало k робастных
    сигм, то в этом месяце и ряд страны, и ряды регионов несут эхо базы, а не новое событие.
    """
    center = common.median()
    mad = float((common - center).abs().median() * 1.4826)
    # на почти постоянном ряде MAD вырождается в ноль: тогда берём обычное отклонение
    scale = mad if mad > 1e-9 else float(common.std(ddof=0)) + 1e-9
    flag = {}
    for col in eval_cols:
        base = col - pd.DateOffset(months=lag)
        flag[col] = bool(base in common.index and abs(common[base] - center) > k * scale)
    return pd.Series(flag)


def echo_filter(alarms: pd.DataFrame, common: pd.Series, k: float = 3.0, lag: int = 12) -> pd.DataFrame:
    """Гасит тревоги в месяцах эха. Применяется к детектору общей части.

    К панельному детектору фильтр не нужен: отклонение от медианы того же месяца уже
    нормирует эхо, потому что оно общее для всех рядов.
    """
    flag = echo_months(common, alarms.columns, k, lag)
    out = alarms.copy()
    for col in alarms.columns:
        if flag.get(col, False):
            out[col] = False
    return out


def shock_size(common: pd.Series, dev: pd.DataFrame, when: str, before: str) -> dict:
    """Величина общего шока против величины разброса между рядами: виден ли он панели."""
    w, b = pd.Timestamp(when), pd.Timestamp(before)
    return {
        "month": str(w.date()),
        "common_pp": float(common[w]),
        "common_change_pp": float(common[w] - common[b]),
        "dispersion_pp": float(dev[w].std()),
        "dispersion_before_pp": float(dev[b].std()),
        "dispersion_change_pp": float(dev[w].std() - dev[b].std()),
    }
