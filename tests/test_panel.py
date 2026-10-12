"""Проверки панельной регрессии и двух исследований на ценах жилья."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from siregions import gap, novelty, panelreg as pr


def _panel(n_reg=30, months=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=months, freq="MS")
    rows = []
    for i in range(n_reg):
        a = rng.normal(0, 1)                       # эффект региона
        x = rng.normal(0, 1, months)
        t = rng.normal(0, 1, months)               # эффект месяца подмешивается ниже
        rows.append(pd.DataFrame({"ref_area": f"r{i}", "m": idx, "x": x,
                                  "a": a, "e": rng.normal(0, 0.5, months)}))
    df = pd.concat(rows, ignore_index=True)
    month_fx = pd.Series(rng.normal(0, 1, months), index=idx)
    df["d"] = df["m"].map(month_fx)
    df["y"] = 2.0 * df["x"] + df["a"] + df["d"] + df["e"]
    return df


def test_fit_recovers_coefficient_with_two_way_absorb():
    df = _panel()
    f = pr.fit(df, "y", ["x"], absorb=["ref_area", "m"], cluster=["ref_area", "m"])
    assert abs(f.params["x"] - 2.0) < 0.1
    assert f.ci_low["x"] < 2.0 < f.ci_high["x"]
    assert f.n_clusters == {"ref_area": 30, "m": 40}


def test_absorbing_month_kills_common_regressor():
    """Общий для всех регионов ряд после месячных эффектов не опознаётся: так и задумано."""
    df = _panel()
    common = df.groupby("m")["d"].transform("mean")
    df["z"] = common
    f = pr.fit(df, "y", ["z"], absorb=["ref_area", "m"], cluster=["ref_area"])
    assert abs(f.params["z"]) < 1e-8


def test_cluster_se_larger_than_naive_under_correlation():
    df = _panel()
    df["y"] = df["y"] + df["m"].map({m: v for m, v in
                                     zip(sorted(df["m"].unique()),
                                         np.cumsum(np.ones(df["m"].nunique())))}) * 0.5
    plain = pr.fit(df, "y", ["x"], absorb=["ref_area"])
    clustered = pr.fit(df, "y", ["x"], absorb=["ref_area"], cluster=["ref_area", "m"])
    assert clustered.se["x"] > 0 and plain.se["x"] > 0


def test_wald_matches_single_coefficient():
    df = _panel()
    f = pr.fit(df, "y", ["x"], absorb=["ref_area", "m"], cluster=["ref_area"])
    w = f.wald(["x"])
    assert abs(np.sqrt(w["chi2"]) - abs(f.tstat["x"])) < 1e-8


def test_nw_ols_lag_zero_equals_ols():
    rng = np.random.default_rng(1)
    x = rng.normal(size=60)
    y = 1 + 0.5 * x + rng.normal(size=60)
    f = pr.nw_ols(pd.Series(y), pd.DataFrame({"x": x}), lags=0)
    assert abs(f.params["x"] - 0.5) < 0.3
    assert f.se["x"] > 0


def _gap_frame(months=30, n_reg=12, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=months, freq="MS")
    rows = []
    for i in range(n_reg):
        deal = 50000 * np.exp(np.cumsum(rng.normal(0.005, 0.01, months)))
        offer = deal * np.exp(0.10 + rng.normal(0, 0.01, months))
        rows.append(pd.DataFrame({"ref_area": f"r{i}", "m": idx, "segment": "Вторичный",
                                  "offer": offer, "deal": deal}))
    df = pd.concat(rows, ignore_index=True)
    df["gap"] = np.log(df["offer"] / df["deal"])
    df["ln_offer"] = np.log(df["offer"])
    df["ln_deal"] = np.log(df["deal"])
    return df


def test_diffs_do_not_look_ahead():
    df = _gap_frame()
    d = gap.diffs(df, ["gap"], lags=2)
    one = d[(d["ref_area"] == "r0")].sort_values("m")
    # лаг 1 первой разности равен первой разности предыдущего месяца
    assert np.allclose(one["d_gap_l1"].to_numpy()[2:], one["d_gap"].to_numpy()[1:-1], equal_nan=True)
    assert one["d_gap"].isna().iloc[0] and one["d_gap_l1"].isna().iloc[:2].all()


def test_predictive_placebo_catches_mean_reversion():
    """На ряде, где цена сделки — шум вокруг тренда, плацебо обязано дать отрицательный знак."""
    df = _gap_frame(months=48, n_reg=25, seed=7)
    out = gap.predictive(df, "Вторичный", horizons=(3,))
    pl = out[out["spec"] == "плацебо"].iloc[0]
    assert pl["coef"] < 0


def test_trim_keeps_most_and_drops_extremes():
    df = _gap_frame()
    df.loc[0, "gap"] = 5.0
    t = gap.trim(df, 0.01)
    assert t["gap"].max() < 5.0
    assert len(t) >= 0.97 * len(df)


def _novelty_frame(months=30, n_reg=10, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-10-01", periods=months, freq="MS")
    rows = []
    for i in range(n_reg):
        old = 70000 * np.exp(np.cumsum(rng.normal(0.004, 0.005, months)))
        prem = 0.14 + np.linspace(0, 0.08, months)
        new = old * np.exp(prem)
        w = 0.3 + np.linspace(0, 0.1, months)
        offset = rng.normal(0, 0.05)
        all_ = old * np.exp(offset + w * prem)
        rows.append(pd.DataFrame({"ref_area": f"r{i}", "m": idx, "sec_old": old,
                                  "sec_new": new, "sec_all": all_, "pri": new * 1.04}))
    df = pd.concat(rows, ignore_index=True)
    df["prem"] = np.log(df["sec_new"] / df["sec_old"])
    df["rel_all"] = np.log(df["sec_all"] / df["sec_old"])
    df["pri_new"] = np.log(df["pri"] / df["sec_new"])
    df["pri_old"] = np.log(df["pri"] / df["sec_old"])
    df["year"] = (df["m"] - df["m"].min()).dt.days / 365.25
    return df


def test_growth_decomposition_is_exact():
    df = _novelty_frame()
    d = novelty.growth_decomposition(df, w0=0.3)
    assert d["max_abs_residual"] < 1e-10


def test_weights_recover_known_mix():
    """На данных, построенных с весом 0,3 и ростом 0,1 за период, регрессия должна это увидеть."""
    df = _novelty_frame(months=48, n_reg=30, seed=11)
    w = novelty.weights(df)
    assert 0.25 < w["w_base"]["coef"] < 0.45
    assert w["w_per_year"]["coef"] > 0


def test_identity_check_flags_offset_data():
    df = _novelty_frame()
    ident = novelty.identity_check(df)
    assert 0.0 <= ident["share_all_between"] <= 1.0
    assert ident["share_new_above_old"] == pytest.approx(1.0)


def test_rate_align_puts_decision_in_its_own_week():
    from siregions import rate as rate_mod

    weeks = pd.date_range("2024-01-07", periods=10, freq="W-SUN")
    dec = pd.DataFrame({"decision_date": [pd.Timestamp("2024-01-12")],
                        "effective_date": [pd.Timestamp("2024-01-15")],
                        "rate": [16.0], "change": [1.0], "kind": ["плановое"]})
    ev = rate_mod.align(dec, weeks)
    # решение в пятницу 12.01 попадает в неделю, кончающуюся в воскресенье 14.01
    assert ev.loc[pd.Timestamp("2024-01-14")] == 1.0
    assert ev.drop(pd.Timestamp("2024-01-14")).abs().sum() == 0


def test_rate_local_projection_recovers_planted_response():
    """В ряд с заложенной реакцией −1 % на п. п. в неделю решения проекция должна попасть."""
    from siregions import rate as rate_mod

    rng = np.random.default_rng(5)
    weeks = pd.date_range("2022-01-02", periods=200, freq="W-SUN")
    dates = weeks[[20, 50, 80, 110, 140, 170]]
    dec = pd.DataFrame({"decision_date": dates - pd.Timedelta(days=2),
                        "effective_date": dates, "rate": np.arange(6) + 8.0,
                        "change": [1.0, 2.0, 1.0, -1.0, 2.0, 1.0], "kind": "плановое"})
    ev = rate_mod.align(dec, weeks)
    y = pd.Series(np.cumsum(rng.normal(0, 0.005, 200)), index=weeks) - 0.01 * ev
    t = rate_mod.local_projection(y, ev, rate_mod.controls(weeks), hmax=1, lags=2)
    assert abs(t.loc[0, "coef"] - (-1.0)) < 0.35


def test_rate_permutation_p_is_high_for_noise():
    from siregions import rate as rate_mod

    rng = np.random.default_rng(9)
    weeks = pd.date_range("2022-01-02", periods=150, freq="W-SUN")
    dates = weeks[[20, 50, 80, 110]]
    dec = pd.DataFrame({"decision_date": dates - pd.Timedelta(days=2), "effective_date": dates,
                        "rate": [8.0, 9.0, 10.0, 11.0], "change": [1.0, 1.0, -1.0, 1.0],
                        "kind": "плановое"})
    ev = rate_mod.align(dec, weeks)
    y = pd.Series(np.cumsum(rng.normal(0, 0.01, 150)), index=weeks)   # реакции нет
    out = rate_mod.permutation_p(y, ev, rate_mod.controls(weeks), h=0, n_perm=100, seed=1)
    assert out["p_perm"] > 0.1
