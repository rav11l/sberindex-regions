"""Короткие проверки узлов, на которых держатся выводы."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from siregions import covid
from siregions.changepoint.detectors import REGISTRY


def _panel(n=40, months=36, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=months, freq="MS")
    base = pd.Series(rng.normal(0, 3, months), index=idx)          # общая динамика
    rows = {f"r{i}": base + rng.normal(0, 2, months) for i in range(n)}
    return pd.DataFrame(rows).T


def test_decompose_removes_common_part():
    w = _panel()
    common, dev = covid.decompose(w)
    assert len(common) == w.shape[1]
    # у отклонений медиана по рядам равна нулю в каждом месяце
    assert np.allclose(dev.median(axis=0).to_numpy(), 0, atol=1e-9)


def test_common_shock_invisible_to_panel_detector():
    """Ключевое утверждение исследования: общий шок панельный детектор не видит."""
    w = _panel()
    w.iloc[:, 20:] -= 50                                            # общий обвал, одинаковый для всех
    _, dev = covid.decompose(w)
    cols = list(dev.columns[12:])
    alarms = REGISTRY["cross_section"](dev, cols, k=3.0)
    shock_col = dev.columns[20]
    assert alarms[shock_col].mean() < 0.05
    # тот же обвал на собственной истории ряда виден
    own = REGISTRY["zscore"](w, cols, k=3.0)
    assert own[shock_col].mean() > 0.5


def test_echo_months_flag_year_after_shock():
    idx = pd.date_range("2019-01-01", periods=40, freq="MS")
    common = pd.Series(np.r_[np.zeros(15), np.full(3, -60.0), np.zeros(22)], index=idx)
    flags = covid.echo_months(common, idx[20:], k=2.5, lag=12)
    assert flags[idx[27]] and not flags[idx[35]]


def test_echo_filter_only_touches_flagged_months():
    idx = pd.date_range("2019-01-01", periods=40, freq="MS")
    common = pd.Series(np.r_[np.zeros(15), np.full(3, -60.0), np.zeros(22)], index=idx)
    cols = list(idx[20:])
    alarms = pd.DataFrame(True, index=["Россия"], columns=cols)
    kept = covid.echo_filter(alarms, common, k=2.5, lag=12)
    assert not kept[idx[27]].iloc[0] and kept[idx[35]].iloc[0]


@pytest.mark.parametrize("name,params", [("cross_section", {"k": 3.0}), ("zscore", {"k": 3.0})])
def test_detectors_do_not_look_ahead(name, params):
    """Тревога в месяце t не должна зависеть от того, что произойдёт позже."""
    w = _panel(seed=3)
    _, dev = covid.decompose(w)
    cols = list(dev.columns[12:24])
    a1 = REGISTRY[name](dev, cols, **params)
    a2 = REGISTRY[name](dev.iloc[:, :24], cols, **params)
    assert a1.equals(a2)
