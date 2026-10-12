"""Точка входа: python -m siregions.cli <команда> --config configs/<файл>.yaml

Команды:
  covid     проверка детекторов на ковидном шоке: общий, идиосинкратический, эхо
  forecast  скользящий бэктест прогнозных моделей на длинной региональной панели
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import covid as covid_mod
from . import data as data_mod

log = logging.getLogger("siregions")


def load_cfg(path: str, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def out_dir(cfg) -> Path:
    p = Path(cfg["output_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def cmd_covid(cfg, args):
    c = cfg["covid"]
    status = data_mod.verify()
    if any(v != "ok" for v in status.values()):
        log.warning("хеши выгрузок: %s", {k: v for k, v in status.items() if v != "ok"})
    long = data_mod.read(cfg["dataset"])
    wide = data_mod.panel(long)
    common, dev = covid_mod.decompose(wide)
    eval_cols = [x for x in wide.columns if x >= pd.Timestamp(c["eval_start"])]
    o = out_dir(cfg)

    # слой 1: детекторы на общей части страны
    one = pd.DataFrame([common.values], columns=common.index, index=["Россия"])
    layer1 = covid_mod.run_detectors(one, eval_cols, c["detectors"])
    rows = []
    for name, A in layer1.items():
        if name == "cross_section":
            continue                       # панельному детектору один ряд не с чем сравнивать
        F = covid_mod.echo_filter(A, common, c["echo_k"], c["echo_lag"])
        fire = lambda M: [str(x.date())[:7] for x in M.columns[M.iloc[0].to_numpy()]]
        rows.append({"detector": name, "alarms": " ".join(fire(A)), "after_echo_filter": " ".join(fire(F))})
    layer1_tab = pd.DataFrame(rows)
    layer1_tab.to_csv(o / "layer1_common_alarms.csv", index=False)

    # слой 2: детекторы на отклонениях регионов
    layer2 = covid_mod.run_detectors(dev, eval_cols, c["detectors"])
    by_month = covid_mod.alarms_by_month(layer2)
    by_month.to_csv(o / "layer2_alarms_by_month.csv")

    calm = [x for x in eval_cols if any(pd.Timestamp(a) <= x <= pd.Timestamp(b) for a, b in c["calm_windows"])]
    echo = [x for x in eval_cols if pd.Timestamp(c["echo_window"][0]) <= x <= pd.Timestamp(c["echo_window"][1])]
    shock = pd.Timestamp(c["shock_month"])
    comp = pd.DataFrame([{
        "detector": k,
        "shock_month_share": float(A[shock].mean()),
        "calm_share": float(A[calm].to_numpy().mean()),
        "echo_share": float(A[echo].to_numpy().mean()),
    } for k, A in layer2.items()]).sort_values("echo_share")
    comp.to_csv(o / "layer2_summary.csv", index=False)

    summary = covid_mod.shock_size(common, dev, c["shock_month"], c["before_month"])
    summary["n_series"] = int(len(wide))
    summary["period"] = [str(wide.columns.min().date()), str(wide.columns.max().date())]
    summary["echo_months"] = [str(x.date())[:7] for x, v in
                              covid_mod.echo_months(common, eval_cols, c["echo_k"], c["echo_lag"]).items() if v]
    (o / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(layer1_tab.to_string(index=False))
    print(comp.round(3).to_string(index=False))
    if c.get("figures", True):
        from . import figures
        figures.covid_figures(o, common, dev, by_month, c)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="siregions")
    ap.add_argument("command", choices=["covid", "forecast"])
    ap.add_argument("--config", default="configs/covid.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(args.config, args.set)
    np.random.seed(cfg.get("seed", 0))
    if args.command == "covid":
        cmd_covid(cfg, args)
    else:
        from .forecast import cmd_forecast
        cmd_forecast(cfg, args)


if __name__ == "__main__":
    main()
