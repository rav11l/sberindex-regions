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


def gap_figures(o: Path, common: pd.DataFrame, rate: pd.Series, wide_gap: pd.DataFrame,
                pred: pd.DataFrame):
    """wide_gap — таблица месяц × регион разрыва в основном сегменте."""
    fig_dir = Path(o) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Общий разрыв по сегментам и ключевая ставка
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, col in enumerate(common.columns):
        ax.plot(common.index, common[col] * 100, lw=2 if col == "Вторичный" else 1.4,
                color=SERIES[i], label=col + " рынок")
    ax.set_ylabel("разрыв, % от цены сделки", color=MUTED, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(rate.index, rate, color=MUTED, lw=1.0, ls="--", label="ключевая ставка")
    ax2.set_ylabel("ключевая ставка, %", color=MUTED, fontsize=8)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines[["top"]].set_visible(False)
    ax.set_title("Запрос в объявлении сходится с ценой сделки: вторичка потеряла половину разрыва",
                 loc="left", fontsize=10, color=INK)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper center", ncol=3)
    ax.set_ylim(top=float(common.max().max() * 100) * 1.62)
    # легенда стоит сверху, поэтому обеим осям нужен запас: линия ставки живёт на правой
    ax2.set_ylim(top=float(rate.max()) * 1.30)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_common_gap.png", dpi=160)
    plt.close(fig)

    # 2. Распределение по регионам: сдвиг вниз без сужения
    med = wide_gap.median(axis=1) * 100
    p25, p75 = wide_gap.quantile(0.25, axis=1) * 100, wide_gap.quantile(0.75, axis=1) * 100
    neg = (wide_gap < 0).mean(axis=1) * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.fill_between(med.index, p25, p75, color=SERIES[0], alpha=0.15, lw=0, label="25–75 % регионов")
    ax.plot(med.index, med, color=SERIES[0], lw=2, label="медианный регион")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_ylabel("разрыв, % от цены сделки", color=MUTED, fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(neg.index, neg, color=SERIES[2], lw=1.2, ls=":", label="регионов с отрицательным разрывом")
    ax2.set_ylabel("доля регионов, %", color=MUTED, fontsize=8)
    ax2.tick_params(colors=MUTED, labelsize=8)
    ax2.spines[["top"]].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left", ncol=2)
    ax.set_title("Распределение сдвинулось вниз, но не сжалось: межквартильный размах тот же",
                 loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_regions.png", dpi=160)
    plt.close(fig)

    # 3. Прогнозная регрессия против плацебо
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    specs = [s for s in ["лаг", "плацебо"] if s in set(pred["spec"])]
    hs = sorted(pred["h"].unique())
    for j, spec in enumerate(specs):
        sub = pred[pred["spec"] == spec].set_index("h").loc[hs]
        x = np.arange(len(hs)) + (j - 0.5) * 0.18
        ax.errorbar(x, sub["coef"], yerr=[sub["coef"] - sub["ci_low"], sub["ci_high"] - sub["coef"]],
                    fmt="o", color=SERIES[j], ecolor=MUTED, elinewidth=1, capsize=3,
                    label="разрыв месяцем раньше → будущий прирост" if spec == "лаг"
                          else "тот же разрыв → ПРОШЛЫЙ прирост (плацебо)")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(np.arange(len(hs)), [f"h = {h}" for h in hs], fontsize=9)
    ax.set_ylabel("коэффициент разрыва", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Плацебо того же порядка: разрыв во многом ловит возврат цены сделки",
                 loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_predictive.png", dpi=160)
    plt.close(fig)


def novelty_figures(o: Path, df: pd.DataFrame, wp: pd.DataFrame, comp: pd.DataFrame):
    fig_dir = Path(o) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Три этажа рынка: первичка, молодая вторичка, старая вторичка
    med = df.groupby("m")[["prem", "pri_new", "pri_old"]].median() * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(med.index, med["pri_old"], color=SERIES[0], lw=2, label="первичка против жилья старше 5 лет")
    ax.plot(med.index, med["prem"], color=SERIES[1], lw=1.6, label="молодая вторичка против старой")
    ax.plot(med.index, med["pri_new"], color=SERIES[2], lw=1.6, label="первичка против молодой вторички")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel("превышение цены, %", color=MUTED, fontsize=8)
    ax.set_title("Разрыв первички и вторички — это почти целиком возраст жилья",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_premium.png", dpi=160)
    plt.close(fig)

    # 2. Сдвиг состава сделок
    w = wp.groupby("m")["w"].median() * 100
    q1 = wp.groupby("m")["w"].quantile(0.25) * 100
    q3 = wp.groupby("m")["w"].quantile(0.75) * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.fill_between(w.index, q1, q3, color=SERIES[0], alpha=0.15, lw=0, label="25–75 % регионов")
    ax.plot(w.index, w, color=SERIES[0], lw=2, label="медианный регион")
    ax.set_ylabel("доля жилья младше 5 лет в сделках, %", color=MUTED, fontsize=8)
    ax.set_title("Состав сделок молодеет: уровень задан предположением, наклон — нет",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_weight.png", dpi=160)
    plt.close(fig)

    # 3. Чем измеряем цену сделки — то и получаем про разрыв
    c = comp[comp["мера"] != "сдвиг состава"]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    y = np.arange(len(c))
    ax.errorbar(c["тренд_пп_в_год"], y,
                xerr=[c["тренд_пп_в_год"] - c["ди_низ"], c["ди_верх"] - c["тренд_пп_в_год"]],
                fmt="o", color=SERIES[0], ecolor=MUTED, elinewidth=1, capsize=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, c["мера"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("тренд разрыва, п. п. в год", color=MUTED, fontsize=8)
    ax.set_title("Схождение запроса и сделки держится на составе сделок",
                 loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_gap_composition.png", dpi=160)
    plt.close(fig)


def rate_figures(o: Path, irf: pd.DataFrame, rob: pd.DataFrame, win: pd.DataFrame):
    fig_dir = Path(o) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    short = {"15 - 24 лет": "15–24", "25 - 34 лет": "25–34", "35 - 64 лет": "35–64", "65+ лет": "65+"}

    # 1. Отклик по возрастам на плановые повышения, относительная мера
    sub = irf[(irf["мера"] == "relative") & (irf["вариант"] == "плановые повышения")]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for i, (g, s) in enumerate(sub.groupby("группа")):
        s = s.sort_values("h")
        ax.plot(s["h"], s["coef"], lw=2 if g.startswith(("15", "65")) else 1.2,
                color=SERIES[i], label=short.get(g, g))
        if g.startswith("15"):
            ax.fill_between(s["h"], s["ci_low"], s["ci_high"], color=SERIES[i], alpha=0.12, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    # за пределами двух недель плацебо-сдвиги дают такие же «значимые» оценки,
    # поэтому зона отмечена явно
    ax.axvspan(2.5, float(sub["h"].max()), color=MUTED, alpha=0.10, lw=0)
    ax.text(2.7, ax.get_ylim()[1] * 0.92, "дальше плацебо срабатывает так же",
            fontsize=7.5, color=MUTED)
    ax.set_xlabel("недель после решения", color=MUTED, fontsize=8)
    ax.set_ylabel("% на п. п. повышения, к средней активности", color=MUTED, fontsize=8)
    ax.set_title("Разница видна в неделю решения: 15–24 уходят вниз, остальные стоят",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_irf_hikes.png", dpi=160)
    plt.close(fig)

    # 2. Повышения против снижений, неделя решения
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    groups = sorted(irf["группа"].unique())
    x = np.arange(len(groups))
    for j, v in enumerate(["плановые повышения", "плановые снижения"]):
        s = irf[(irf["мера"] == "relative") & (irf["вариант"] == v) & (irf["h"] == 0)]
        s = s.set_index("группа").loc[groups]
        # приглушаем столбцы, где оценка не отличается от нуля: иначе картинка утверждает
        # больше, чем выдерживают проверки
        solid = (s["coef"].abs() > 2 * s["se"]).to_numpy()
        ax.bar(x + (j - 0.5) * 0.36, s["coef"], width=0.34, color=SERIES[j],
               alpha=1.0, label="повышения" if j == 0 else "снижения")
        for xi, val, ok in zip(x + (j - 0.5) * 0.36, s["coef"], solid):
            if not ok:
                ax.bar([xi], [val], width=0.34, color="white", alpha=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x, [short.get(g, g) for g in groups], fontsize=9)
    ax.set_ylabel("% на п. п., неделя решения", color=MUTED, fontsize=8)
    ax.set_title("На снижение не реагирует никто; бледным — незначимые оценки",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_hikes_vs_cuts.png", dpi=160)
    plt.close(fig)

    # 3. Проверка устойчивости: коэффициент в неделю решения по вариантам
    s = rob[(rob["h"] == 0)]
    order = [v for v in ["все решения", "плановые", "плановые повышения", "плановые снижения",
                         "плановые, без 2022"] if v in set(s["вариант"])]
    order += [v for v in s["вариант"].unique() if v.startswith("плацебо")]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    width = 0.2
    for i, g in enumerate(sorted(s["группа"].unique())):
        sub = s[s["группа"] == g].set_index("вариант").reindex(order)
        ax.bar(np.arange(len(order)) + (i - 1.5) * width, sub["коэф"], width=width,
               color=SERIES[i], label=short.get(g, g))
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(np.arange(len(order)), order, fontsize=7.5, rotation=12)
    ax.set_ylabel("% на п. п., неделя решения", color=MUTED, fontsize=8)
    ax.set_title("Что остаётся от результата при смене набора событий и на плацебо",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, ncol=4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "03_robustness.png", dpi=160)
    plt.close(fig)


def surprise_figures(o: Path, parts: pd.DataFrame, est: pd.DataFrame, mde: float):
    """Разложение решений на сюрприз и ожидаемую часть: сколько в решении новости
    и что из этого следует для оценки."""
    fig_dir = Path(o) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    p = parts[~parts["нет_котировок"]]

    # 1. Сколько в решении было новостью
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    lim = float(np.ceil(max(p["change"].abs().max(), p["сюрприз"].abs().max())))
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls="--")
    hike = p["change"] > 0
    ax.scatter(p.loc[hike, "change"], p.loc[hike, "сюрприз"], s=26, color=SERIES[0],
               label="повышения", zorder=3)
    ax.scatter(p.loc[~hike, "change"], p.loc[~hike, "сюрприз"], s=26, color=SERIES[1],
               label="снижения и сохранения", zorder=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.text(-lim * 0.95, lim * 0.82, "пунктир — решение целиком оказалось новостью",
            fontsize=7.5, color=MUTED)
    ax.set_xlabel("изменение ключевой ставки, п. п.", color=MUTED, fontsize=8)
    ax.set_ylabel("сюрприз: сдвиг кривой ОФЗ, п. п.", color=MUTED, fontsize=8)
    ax.set_title("Почти все решения рынок ждал: точки лежат у нуля, а не на диагонали",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_surprise_vs_change.png", dpi=160)
    plt.close(fig)

    # 2. Отклик на сюрприз и на ожидаемую часть с интервалами и порогом различимости
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(est))
    for j, (c, name) in enumerate([("сюрприз", "сюрприз"), ("ожидаемое", "ожидаемая часть")]):
        off = (j - 0.5) * 0.3
        ax.errorbar(x + off, est[f"коэф_{c}"], yerr=1.645 * est[f"ст_ош_{c}"], fmt="o",
                    ms=5, lw=1.4, capsize=3, color=SERIES[j], label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axhline(-mde, color=MUTED, lw=0.8, ls=":")
    ax.text(-0.42, -mde, "порог различимости", fontsize=7.5, color=MUTED,
            ha="left", va="bottom")
    ax.set_xticks(x, est["вариант"], fontsize=7.5, rotation=10)
    ax.set_ylabel("% на п. п., неделя решения", color=MUTED, fontsize=8)
    ax.set_title("Знаки в пользу Холла, но перестановочный тест не держит ни одну оценку",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_surprise_vs_expected.png", dpi=160)
    plt.close(fig)
