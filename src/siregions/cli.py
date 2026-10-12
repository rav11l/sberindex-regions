"""Точка входа: python -m siregions.cli <команда> --config configs/<файл>.yaml

Команды:
  covid     проверка детекторов на ковидном шоке: общий, идиосинкратический, эхо
  forecast  скользящий бэктест прогнозных моделей на длинной региональной панели
  novelty   премия за новизну жилья, состав сделок и его вклад в рост цен
  gap       разрыв «объявление минус сделка»: уровень, динамика, опережение, прогнозная ценность
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
    print(rate_meas.round(3).to_string(index=False))
    if c.get("figures", True):
        from . import figures
        figures.covid_figures(o, common, dev, by_month, c)
    return summary


def cmd_gap(cfg, args):
    from . import gap as gap_mod
    from . import panelreg as pr

    c = cfg["gap"]
    seg, rate_col = c["segment"], c.get("rate_col", "rate")
    status = data_mod.verify()
    if any(v != "ok" for v in status.values()):
        log.warning("хеши выгрузок: %s", {k: v for k, v in status.items() if v != "ok"})
    df = gap_mod.build(cfg)
    o = out_dir(cfg)

    cov = gap_mod.coverage(df)
    cov.to_csv(o / "coverage.csv", index=False)
    desc = gap_mod.describe(df)
    desc.to_csv(o / "describe.csv", index=False)
    common = gap_mod.common_series(df)
    common.to_csv(o / "common_gap.csv")
    reg = gap_mod.by_region(df, seg)
    reg.to_csv(o / "by_region.csv")
    src = gap_mod.source_check(df)
    src.to_csv(o / "source_check.csv", index=False)

    tr = {s: gap_mod.trend(df, s) for s in common.columns}
    ctrl = gap_mod.primary_as_control(df)
    rl = gap_mod.rate_link_ts(df, seg, rate_col)
    rp = gap_mod.rate_link_panel(df, seg, rate_col)
    dl = gap_mod.rate_dlag(df, seg, rate_col, c.get("lags", 3))
    het = gap_mod.heterogeneity(df, seg, rate_col)
    subs = gap_mod.rate_subsamples(df, c.get("rate_subsample_starts",
                                             ["2022-01-01", "2022-07-01", "2022-10-01", "2023-01-01"]),
                                   seg, rate_col, c.get("lags", 3))
    subs.to_csv(o / "rate_subsamples.csv", index=False)
    ll = gap_mod.lead_lag(df, seg, c.get("lags", 3))
    pred = gap_mod.predictive(df, seg, c.get("horizons", (3, 6, 12)))
    pred.to_csv(o / "predictive.csv", index=False)
    dec = gap_mod.decompose_fwd(df, seg, c.get("horizons", (3, 6, 12)))
    dec.to_csv(o / "decompose_forward.csv", index=False)
    # устойчивость к отсечению крайних наблюдений разрыва
    q = c.get("trim_q", 0.01)
    tdf = gap_mod.trim(df, q)
    robust = {
        "trim_q": q, "n_dropped": int(len(df) - len(tdf)),
        "trend_segment": {k: float(v) for k, v in
                          gap_mod.trend(tdf, seg).table().loc["year"].items()},
        "rate_panel": {k: float(v) for k, v in
                       gap_mod.rate_link_panel(tdf, seg, rate_col).table().loc["d_" + rate_col].items()},
        "predictive_lag": gap_mod.predictive(tdf, seg, c.get("horizons", (3, 6, 12)))
                                 .query("spec == 'лаг'").round(4).to_dict("records"),
    }

    oos_runs = []
    for h in c.get("oos_horizons", [6, 12]):
        for mode in ["absolute", "relative"]:
            for gcol in ["gap", "gap_l1"]:
                for fe in [True, False]:
                    oos_runs.append(gap_mod.oos(df, seg, h, c.get("oos_min_train", 24),
                                                c.get("n_boot", 2000), cfg.get("seed", 0),
                                                mode=mode, gap_col=gcol, region_fe=fe))
    oos_tab = pd.DataFrame(oos_runs)
    oos_tab.to_csv(o / "oos.csv", index=False)

    tables = {
        "trend": pd.concat({s: f.table() for s, f in tr.items()}),
        "rate_levels_with_trend": rl["levels_with_trend"].table(),
        "rate_differences": rl["differences"].table(),
        "rate_panel": rp.table(),
        "rate_dlag": dl["fit"].table(),
        "heterogeneity": het.table(),
        "lead_lag_offer": ll["ln_offer"]["fit"].table(),
        "lead_lag_deal": ll["ln_deal"]["fit"].table(),
    }
    for name, t in tables.items():
        t.to_csv(o / f"reg_{name}.csv")

    summary = {
        "segment": seg,
        "rate_col": rate_col,
        "coverage": cov.astype(str).to_dict("records"),
        "gap_median": {s: float(df[df["segment"] == s]["gap"].median()) for s in common.columns},
        "gap_first_last": {s: [float(common[s].dropna().iloc[:3].mean()),
                               float(common[s].dropna().iloc[-3:].mean())] for s in common.columns},
        "trend_pp_per_year": {s: {"coef": float(f.params["year"]), "se": float(f.se["year"]),
                                  "p": float(f.pvalue["year"])} for s, f in tr.items()},
        "primary_as_control": {"n_regions": ctrl["n_regions"], "level_median": ctrl["level_median"],
                               "trend": {"coef": float(ctrl["trend"].params["year"]),
                                         "se": float(ctrl["trend"].se["year"]),
                                         "p": float(ctrl["trend"].pvalue["year"])}},
        "rate_diff_ts": {"coef": float(rl["differences"].params["d_rate"]),
                         "se": float(rl["differences"].se["d_rate"]),
                         "p": float(rl["differences"].pvalue["d_rate"]),
                         "xcorr_by_lag": rl["xcorr_diff"]},
        "rate_panel": {"coef": float(rp.params["d_" + rate_col]), "se": float(rp.se["d_" + rate_col]),
                       "p": float(rp.pvalue["d_" + rate_col]),
                       "mde": float(pr.mde(rp.se["d_" + rate_col])),
                       "clusters": rp.n_clusters},
        "rate_dlag": {"cum": dl["cum"], "cum_se": dl["cum_se"], "cum_p": dl["cum_p"],
                      "cum_ci": dl["cum_ci"], "joint_p": dl["joint_p"],
                      "coefs": dl["fit"].params[dl["terms"]].round(5).to_dict()},
        "rate_subsamples": subs.round(5).to_dict("records"),
        "heterogeneity": {k: {"coef": float(het.params[k]), "se": float(het.se[k]),
                              "p": float(het.pvalue[k])} for k in ["rate_x_price", "rate_x_supply"]},
        "lead_lag": {"sd_monthly_growth": ll["sd_monthly_growth"],
                     **{y: {"sum_other_lags": v["sum_other"], "sum_other_std": v["sum_other_std"],
                            "r2_gain_other": v["r2_gain_other"],
                            "sum_own_lags": v["sum_own"],
                            "coefs_other": v["coefs_other"], "coefs_own": v["coefs_own"],
                            "wald_other_p": v["wald_other"]["p_F"],
                            "wald_own_p": v["wald_own"]["p_F"], "n": v["fit"].nobs}
                        for y, v in ll.items() if y != "sd_monthly_growth"}},
        "predictive": pred.to_dict("records"),
        "decompose_forward": dec.to_dict("records"),
        "oos": oos_runs,
        "robust_trimmed": robust,
        "source_check": src.to_dict("records"),
    }
    (o / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                    encoding="utf-8")
    print(cov.to_string(index=False))
    print(desc.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"coverage", "predictive", "source_check"}},
                     ensure_ascii=False, indent=2, default=str))
    print(pred.round(4).to_string(index=False))
    print(dec.round(4).to_string(index=False))
    print(oos_tab.round(5).to_string(index=False))
    print(subs.round(4).to_string(index=False))
    if c.get("figures", True):
        from . import figures
        rate = df.drop_duplicates("m").set_index("m")[rate_col].sort_index()
        wide_gap = df[df["segment"] == seg].pivot_table(index="m", columns="ref_area", values="gap")
        figures.gap_figures(o, common, rate, wide_gap,
                            pred[(pred["segment"] == seg) & pred["spec"].isin(["лаг", "плацебо"])])
    return summary


def cmd_novelty(cfg, args):
    from . import gap as gap_mod
    from . import novelty as nov

    c = cfg["novelty"]
    status = data_mod.verify()
    if any(v != "ok" for v in status.values()):
        log.warning("хеши выгрузок: %s", {k: v for k, v in status.items() if v != "ok"})
    df = nov.build(cfg)
    o = out_dir(cfg)

    ident = nov.identity_check(df)
    prem = nov.premium(df)
    prem.to_csv(o / "premium_by_year.csv", index=False)
    ptr = nov.premium_trend(df)
    wts = nov.weights(df)

    decomp = {}
    for w0 in c.get("w0_sensitivity", [c["w0"]]):
        d = nov.growth_decomposition(df, w0, c.get("base_months", 3), c.get("last_months", 3))
        if w0 == c["w0"]:
            d["by_region"].to_csv(o / "decomposition_by_region.csv", index=False)
        decomp[str(w0)] = {k: v for k, v in d.items() if k != "by_region"}

    gap_df = gap_mod.build(cfg)
    comp = nov.gap_composition(df, gap_df)
    comp.to_csv(o / "gap_composition.csv", index=False)
    rate_meas = nov.gap_rate_by_measure(df, gap_df)
    rate_meas.to_csv(o / "gap_rate_by_measure.csv", index=False)

    summary = {"coverage": {"n": ident["n"], "regions": ident["n_regions"],
                            "start": str(df["m"].min().date()), "end": str(df["m"].max().date())},
               "identity_check": ident, "premium_by_year": prem.round(4).to_dict("records"),
               "premium_trend": ptr, "weights": wts, "decomposition": decomp,
               "gap_composition": comp.round(4).to_dict("records"),
               "gap_rate_by_measure": rate_meas.round(4).to_dict("records")}
    (o / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                    encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "premium_by_year"},
                     ensure_ascii=False, indent=2, default=str))
    print(prem.round(3).to_string(index=False))
    print(comp.round(3).to_string(index=False))
    print(rate_meas.round(3).to_string(index=False))
    if c.get("figures", True):
        from . import figures
        wp = nov.weight_path(df, c["w0"], c.get("base_months", 3))
        figures.novelty_figures(o, df, wp, comp)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="siregions")
    ap.add_argument("command", choices=["covid", "forecast", "gap", "novelty"])
    ap.add_argument("--config", default="configs/covid.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_cfg(args.config, args.set)
    np.random.seed(cfg.get("seed", 0))
    if args.command == "covid":
        cmd_covid(cfg, args)
    elif args.command == "gap":
        cmd_gap(cfg, args)
    elif args.command == "novelty":
        cmd_novelty(cfg, args)
    else:
        from .forecast import cmd_forecast
        cmd_forecast(cfg, args)


if __name__ == "__main__":
    main()
