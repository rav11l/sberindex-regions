"""Перенос панельного приёма на длинную региональную панель.

В «Шоке по расписанию» панель была из 24 месяцев: формальные тесты на шести целевых месяцах
ничего не доказывали. Цены сделок с жильём дают 117 месяцев по 86 регионам и двум сегментам
рынка. Здесь тот же набор моделей и тот же приём — прогнозировать отклонение ряда от общего
фактора панели, — но с числом точек прогноза, на котором тесты имеют смысл.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import analysis, backtest, data as data_mod

log = logging.getLogger(__name__)


def build_panel(cfg) -> pd.DataFrame:
    long = data_mod.read(cfg["dataset"])
    f = cfg["forecast"]
    wide = long.pivot_table(index=["ref_area", f["segment_col"]], columns="period", values="value")
    wide.index = [f"{a}|{b}" for a, b in wide.index]
    # даты в выгрузке стоят на конец месяца; приводим к первому числу, иначе прошлый год
    # не находится из-за разной длины февраля
    wide.columns = pd.PeriodIndex(wide.columns, freq="M").to_timestamp()
    wide = wide.sort_index(axis=1)
    if f.get("start"):
        wide = wide.loc[:, wide.columns >= pd.Timestamp(f["start"])]
    full = wide[wide.notna().all(axis=1)]
    log.info("панель: %d рядов из %d с полной историей, %d месяцев", len(full), len(wide), wide.shape[1])
    return full


def cmd_forecast(cfg, args):
    from .cli import out_dir

    status = data_mod.verify()
    if any(v != "ok" for v in status.values()):
        log.warning("хеши выгрузок: %s", {k: v for k, v in status.items() if v != "ok"})
    wide = build_panel(cfg)
    o = out_dir(cfg)
    pred, st = backtest.backtest(wide, cfg)
    pred.to_parquet(o / "forecast_predictions.parquet")
    st.to_csv(o / "forecast_status.csv", index=False)

    parts = [pred]
    have = set(pred["model"])
    for c in cfg.get("analysis", {}).get("ensembles", []):
        if set(c["members"]) <= have:
            parts.append(analysis.combine(pred, c["members"], c["name"], c.get("how", "mean")))
        else:
            log.warning("ансамбль %s пропущен: нет %s", c["name"], sorted(set(c["members"]) - have))
    full = pd.concat(parts, ignore_index=True)

    m = backtest.metrics(full).sort_values("MAE")
    m.to_csv(o / "metrics.csv", index=False)
    backtest.metrics(full, by=("model", "h")).to_csv(o / "metrics_by_h.csv", index=False)
    base = cfg.get("analysis", {}).get("dm_base", "prophet")
    if base in set(full["model"]):
        pd.DataFrame([backtest.dm_test(full, a, base) for a in m["model"] if a != base]).to_csv(
            o / "dm.csv", index=False)
        pd.DataFrame([backtest.paired_summary(full, a, base) for a in m["model"] if a != base]).to_csv(
            o / "paired.csv", index=False)
    meta = {"n_series": int(len(wide)), "months": int(wide.shape[1]),
            "period": [str(wide.columns.min().date()), str(wide.columns.max().date())],
            "origins": cfg["forecast"]["origins"], "horizons": cfg["forecast"]["horizons"],
            "pairs": int(m["n"].max())}
    (o / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(m.round(1).to_string(index=False))
    if cfg.get("figures", True):
        from . import figures
        figures.deals_figures(o, m, backtest.metrics(full, by=("model", "h")))
    return m
