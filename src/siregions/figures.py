"""Рисунки. Читают только то, что уже посчитано."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, ACCENT, GRID = "#1f2328", "#6e7781", "#0b6e4f", "#d0d7de"
SERIES = ["#0b6e4f", "#1f6feb", "#bc4c00", "#8250df", "#57606a", "#a40e26"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)


def covid_figures(o: Path, common: pd.Series, dev: pd.DataFrame, by_month: pd.DataFrame, cfg: dict):
    fig_dir = Path(o) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Общая часть и разброс регионов вокруг неё
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = common.index
    ax.fill_between(x, common + dev.quantile(0.1), common + dev.quantile(0.9), color=ACCENT, alpha=0.15, lw=0,
                    label="10–90 % регионов")
    ax.plot(x, common, color=ACCENT, lw=2, label="медианный регион")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axvline(pd.Timestamp(cfg["shock_month"]), color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("прирост числа туристов, % г/г", color=MUTED, fontsize=8)
    ax.set_title("Внутренний туризм: общий шок виден, разброс регионов почти не меняется",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_common_and_spread.png", dpi=160)
    plt.close(fig)

    # 2. Доля рядов с тревогой по месяцам
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, col in enumerate(by_month.columns):
        lw = 2.2 if col == "cross_section" else 1.2
        ax.plot(by_month.index, by_month[col] * 100, lw=lw, color=SERIES[i % len(SERIES)], label=col)
    ax.axvspan(pd.Timestamp(cfg["echo_window"][0]), pd.Timestamp(cfg["echo_window"][1]),
               color="#8A5A12", alpha=0.10, lw=0)
    ax.axvline(pd.Timestamp(cfg["shock_month"]), color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("доля регионов с тревогой, %", color=MUTED, fontsize=8)
    ax.set_title("Пунктир — ковидный шок, полоса — эхо базы годом позже", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=3)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_alarms_by_month.png", dpi=160)
    plt.close(fig)

    # 3. Разброс отклонений
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(dev.columns, dev.std(axis=0), color=SERIES[2], lw=1.8)
    ax.axvline(pd.Timestamp(cfg["shock_month"]), color=MUTED, lw=0.8, ls="--")
    ax.set_ylabel("ст. откл. отклонений, п. п.", color=MUTED, fontsize=8)
    ax.set_title("Разброс между регионами: ковид его не поднял, эхо подняло в десять раз",
                 loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_dispersion.png", dpi=160)
    plt.close(fig)


def deals_figures(o: Path, metrics: pd.DataFrame, by_h: pd.DataFrame):
    fig_dir = Path(o) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    m = metrics.sort_values("MAE")
    m = m[m["model"] != "snaive"]                     # сезонный наивный сжал бы шкалу втрое
    fig, ax = plt.subplots(figsize=(7, 3.2))
    y = np.arange(len(m))
    colors = [SERIES[0] if k == m.iloc[0]["model"] else ("#bc4c00" if k == "prophet" else "#C9C3B8")
              for k in m["model"]]
    ax.barh(y, m["MAE"], color=colors, height=0.6)
    ax.set_yticks(y, m["model"], fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(m["MAE"]):
        ax.text(v + 60, i, f"{v:,.0f}".replace(",", " "), va="center", fontsize=8, color=INK)
    ax.set_xlabel("MAE, руб. за квадратный метр", color=MUTED, fontsize=8)
    ax.set_title("Цены сделок: 157 рядов, 22 точки прогноза, 66 целевых месяцев",
                 loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_mae.png", dpi=160)
    plt.close(fig)
