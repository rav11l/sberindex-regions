"""Сверка чисел из docs/ с результатами прогона.

Тесты пропускаются, если прогона нет: выгрузки и outputs/ в репозиторий не кладутся.
Смысл в том, чтобы текст нельзя было незаметно рассинхронизировать с кодом: после любого
пересчёта `pytest` покажет, какие числа в документах устарели.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

OUT = Path("outputs")


def _load(name: str) -> dict:
    p = OUT / name / "summary.json"
    if not p.exists():
        pytest.skip(f"нет прогона {name}")
    return json.loads(p.read_text(encoding="utf-8"))


def test_gap_levels_and_trend_match_doc():
    s = _load("gap")
    assert round(s["gap_median"]["Вторичный"] * 100, 1) == 10.3
    assert round(s["gap_median"]["Первичный"] * 100, 1) == 3.4
    assert round(s["trend_pp_per_year"]["Вторичный"]["coef"] * 100, 2) == -1.35
    assert s["trend_pp_per_year"]["Первичный"]["p"] > 0.5          # тренда нет
    assert round(s["primary_as_control"]["trend"]["coef"] * 100, 2) == -1.31


def test_gap_rate_subsamples_match_doc():
    s = _load("gap")
    tab = {r["start"]: r for r in s["rate_subsamples"]}
    assert round(tab["2022-01-01"]["instant"] * 100, 2) == 0.18
    assert tab["2023-01-01"]["instant_p"] > 0.1                    # мгновенный эффект без 2022 исчезает
    assert round(tab["2023-01-01"]["cum"] * 100, 2) == 0.23
    assert tab["2023-01-01"]["cum_p"] < 0.01


def test_gap_placebo_is_negative_at_every_horizon():
    s = _load("gap")
    pred = pd.DataFrame(s["predictive"])
    pl = pred[pred["spec"] == "плацебо"]
    assert (pl["coef"] < 0).all() and (pl["p"] < 0.01).all()
    lag6 = pred[(pred["spec"] == "лаг 6") & (pred["h"] == 6)].iloc[0]
    assert lag6["p"] > 0.1                                        # сигнал затухает


def test_gap_decomposition_rejects_restriction():
    s = _load("gap")
    dec = pd.DataFrame(s["decompose_forward"])
    assert (dec["p_restriction"] < 0.01).all()
    assert (dec["coef_deal"] < 0).all()


def test_novelty_premium_structure_matches_doc():
    s = _load("novelty")
    prem = {r["y"]: r for r in s["premium_by_year"]}
    assert round(prem[2026]["prem_new_old"] * 100) == 22
    assert s["premium_trend"]["pri_new"]["p"] > 0.05               # первичка против молодой вторички не расходится
    assert s["premium_trend"]["prem"]["p"] < 0.001


def test_novelty_decomposition_is_exact_and_matches_doc():
    s = _load("novelty")
    for w0, d in s["decomposition"].items():
        assert d["max_abs_residual"] < 1e-10
        assert round(d["median"]["d_all"] * 100, 1) == 31.9
        assert round(d["median"]["d_old"] * 100, 1) == 27.6


def test_novelty_gap_composition_matches_doc():
    s = _load("novelty")
    rows = {r["мера"]: r for r in s["gap_composition"]}
    assert round(rows["цены сделок (исходный набор)"]["тренд_пп_в_год"], 2) == -1.20
    assert round(rows["этот набор, старше 5 лет"]["тренд_пп_в_год"], 2) == -0.31
    assert rows["этот набор, старше 5 лет"]["p"] > 0.1
    assert round(rows["сдвиг состава"]["тренд_пп_в_год"], 2) == 1.63


def test_deals_metrics_match_doc():
    p = OUT / "deals" / "metrics.csv"
    if not p.exists():
        pytest.skip("нет прогона deals")
    m = pd.read_csv(p).set_index("model")["MAE"].round(0)
    assert m["ets_relative"] == 2181
    assert m["factor_only"] == 2538
    assert m["prophet"] == 3709
    assert m["prophet_relative"] == 3830
    s = json.loads((OUT / "deals" / "summary.json").read_text(encoding="utf-8"))
    assert s["n_series"] == 155 and s["pairs"] == 10230


def test_rate_column_numbers_match_run():
    s = _load("rate")
    assert s["decisions"]["с изменением"] == 33
    assert s["decisions"]["плановых повышений"] == 15 and s["decisions"]["плановых снижений"] == 15
    rob = pd.DataFrame(s["robustness"])
    young = rob[(rob["группа"] == "15 - 24 лет") & (rob["h"] == 0)].set_index("вариант")
    assert round(young.loc["плановые повышения", "коэф"], 2) == -1.31
    assert young.loc["плановые повышения", "p_perm"] < 0.06
    assert young.loc["плановые, без 2022", "p_perm"] < 0.05
    # на снижениях ничего ни у кого
    cuts = rob[(rob["вариант"] == "плановые снижения") & (rob["h"] == 0)]
    assert (cuts["p_perm"] > 0.15).all()
    # плацебо в неделю фиктивного решения молчит
    pl = rob[rob["вариант"].str.startswith("плацебо") & (rob["h"] == 0)]
    assert (pl["p_nw"] > 0.05).all()
    # оценка по 65+ не переживает исключения по одному событию — в выводы не идёт
    jk = s["jackknife"]["65+ лет | плановые повышения"]
    assert jk["p_max"] > 0.2


def test_surprise_matches_doc():
    """Числа из docs/surprise.md: разложение, оценки и то, что проверка не пройдена."""
    s = _load("surprise")
    assert s["решений"] == 34 and s["с изменением"] == 33
    assert s["без котировок"] == 1                     # 28.02.2022: торги остановлены
    head = {r["вариант"]: r for r in s["оценки_h0"]}
    hikes = head["плановые повышения"]
    assert round(hikes["коэф_сюрприз"], 2) == -2.40
    assert round(hikes["ст_ош_сюрприз"], 2) == 1.31
    assert round(hikes["коэф_ожидаемое"], 2) == -0.74
    assert round(s["mde"], 1) == 3.7                   # порог различимости выше оценки
    perm = {(r["вариант"], r["term"]): r["p_perm"] for r in s["перестановки"]}
    assert perm[("плановые повышения", "сюрприз")] > 0.1      # проверка не пройдена
    assert min(v for (_, t), v in perm.items() if t == "сюрприз") > 0.2
    chk = {(r["срок"], r["окно"]): r for r in s["проверки_меры"]}
    assert all(-3.3 < r["коэф_сюрприз"] < -2.0 for r in chk.values())
    assert round(chk[("y0_25", 0)]["sd_сюрприз"], 2) == 0.38
    assert round(chk[("y0_25", 0)]["корреляция"], 2) == 0.70
