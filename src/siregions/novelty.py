"""Новизна жилья: премия за возраст, состав сделок и его вклад в рост цен.

Данные: «Цены на жилую недвижимость» СберИндекса. На вторичном рынке есть три ряда —
«Всего», «Построенные менее 5 лет назад», «Построенные более 5 лет назад», — на первичном
только «Всего».

Первое, что проверяется и чего не получается: «Всего» НЕ является средневзвешенным двух
возрастных групп. В большинстве регион-месяцев оно лежит ниже обеих (в Татарстане на 19 %
ниже группы «старше 5 лет»), а наивно восстановленная доля новых выходит за [0; 1] более
чем в половине случаев. Значит, у рядов разный охват, и уровень доли по ним не восстановить.

Что восстановить можно — изменение доли. Если охват региона постоянен во времени, то
    log(Всего / старше 5 лет) = c_i + w_t · log(менее 5 / старше 5),
где c_i — постоянный сдвиг охвата, а w_t — вес молодого жилья в сделках. Отсюда берутся
и средний вес (регрессия с эффектами региона), и его сдвиг относительно базы. Уровень w
опознаётся только при предположении о c_i, поэтому он всюду идёт с проверкой
чувствительности, а выводы строятся на изменениях.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as data_mod
from . import panelreg as pr

COLS = {
    ("Вторичный рынок", "Всего"): "sec_all",
    ("Вторичный рынок", "Построенные более 5 лет назад"): "sec_old",
    ("Вторичный рынок", "Построенные менее 5 лет назад"): "sec_new",
    ("Первичный рынок", "Всего"): "pri",
}


def build(cfg: dict) -> pd.DataFrame:
    """Панель регион × месяц: четыре цены, премия за новизну, относительный уровень «Всего»."""
    rp = data_mod.read(cfg["datasets"]["prices"])
    rp["m"] = pd.PeriodIndex(rp["period"], freq="M").to_timestamp()
    rp = rp[~rp["ref_area"].isin(cfg.get("exclude_areas", ["Россия"]))]
    w = rp.pivot_table(index=["ref_area", "m"],
                       columns=["real_estate_type", "real_estate_novelty"], values="value")
    w.columns = [COLS.get(tuple(c), "_".join(c)) for c in w.columns]
    df = w.dropna(subset=["sec_all", "sec_old", "sec_new"]).reset_index()
    df["prem"] = np.log(df["sec_new"] / df["sec_old"])          # премия за возраст внутри вторички
    df["rel_all"] = np.log(df["sec_all"] / df["sec_old"])       # где «Всего» относительно старого жилья
    df["pri_new"] = np.log(df["pri"] / df["sec_new"])
    df["pri_old"] = np.log(df["pri"] / df["sec_old"])
    df["year"] = (df["m"] - df["m"].min()).dt.days / 365.25
    return df.sort_values(["ref_area", "m"]).reset_index(drop=True)


def identity_check(df: pd.DataFrame) -> dict:
    """Проверка «Всего = смесь двух групп». Нужна, чтобы честно отказаться от уровней."""
    naive = (df["sec_all"] - df["sec_old"]) / (df["sec_new"] - df["sec_old"])
    inside = ((df["sec_all"] >= df[["sec_old", "sec_new"]].min(axis=1)) &
              (df["sec_all"] <= df[["sec_old", "sec_new"]].max(axis=1)))
    off = df.groupby("ref_area")["rel_all"].median()
    return {"n": int(len(df)), "n_regions": int(df["ref_area"].nunique()),
            "share_all_between": float(inside.mean()),
            "share_naive_weight_outside_01": float(((naive < 0) | (naive > 1)).mean()),
            "offset_median": float(off.median()), "offset_p10": float(off.quantile(0.1)),
            "offset_p90": float(off.quantile(0.9)),
            "share_new_above_old": float((df["sec_new"] > df["sec_old"]).mean())}


def premium(df: pd.DataFrame) -> pd.DataFrame:
    """Премия за новизну и место первичного рынка между новой и старой вторичкой, по годам."""
    g = df.assign(y=df["m"].dt.year).groupby("y")
    out = g.agg(prem_new_old=("prem", "median"), prem_pri_new=("pri_new", "median"),
                prem_pri_old=("pri_old", "median"), n=("prem", "size"))
    out["spread_regions_p10_p90"] = g["prem"].quantile(0.9).values - g["prem"].quantile(0.1).values
    return out.reset_index()


def premium_trend(df: pd.DataFrame) -> dict:
    """Тренды премий внутри региона, кластеры по региону и месяцу."""
    out = {}
    for col in ["prem", "pri_new", "pri_old"]:
        f = pr.fit(df, col, ["year"], absorb=["ref_area"], cluster=["ref_area", "m"])
        out[col] = {k: float(v) for k, v in f.table().loc["year"].items()}
    return out


def weights(df: pd.DataFrame) -> dict:
    """Средний вес молодого жилья в сделках и его тренд (регрессия rel_all на премию)."""
    d = df.copy()
    d["prem_year"] = d["prem"] * d["year"]
    flat = pr.fit(d, "rel_all", ["prem"], absorb=["ref_area"], cluster=["ref_area", "m"])
    trend = pr.fit(d, "rel_all", ["prem", "prem_year"], absorb=["ref_area"], cluster=["ref_area", "m"])
    return {"w_mean": {k: float(v) for k, v in flat.table().loc["prem"].items()},
            "w_base": {k: float(v) for k, v in trend.table().loc["prem"].items()},
            "w_per_year": {k: float(v) for k, v in trend.table().loc["prem_year"].items()}}


def weight_path(df: pd.DataFrame, w0: float, base_months: int = 3) -> pd.DataFrame:
    """Траектория веса по региону при заданном весе в базовом периоде.

    w_t выводится из тождества w_t·prem_t = w_0·prem_0 + [rel_all_t − rel_all_0]: постоянный
    сдвиг охвата c_i сокращается. Уровень зависит от w_0, изменения — нет.
    """
    out = []
    for area, s in df.groupby("ref_area"):
        s = s.sort_values("m")
        base = s.head(base_months)
        x0, p0 = base["rel_all"].mean(), base["prem"].mean()
        s = s.assign(w=(w0 * p0 + (s["rel_all"] - x0)) / s["prem"])
        out.append(s)
    return pd.concat(out, ignore_index=True)


def growth_decomposition(df: pd.DataFrame, w0: float, base_months: int = 3,
                         last_months: int = 3) -> dict:
    """Разложение роста средней цены вторички на цену старого жилья, премию и состав.

    log P_всего = c_i + log P_старое + w·prem, отсюда точное разложение Беннета:
        Δlog P_всего = Δlog P_старое + w̄·Δprem + prem̄·Δw.
    Третье слагаемое и есть вклад новизны: цены групп не менялись бы, а средняя цена росла
    бы просто потому, что покупают более молодое жильё.
    """
    wp = df.assign(ln_old=np.log(df["sec_old"]), ln_all=np.log(df["sec_all"]))
    rows = []
    for area, s in wp.groupby("ref_area"):
        s = s.sort_values("m")
        b, e = s.head(base_months), s.tail(last_months)
        # вес считается из усреднённых по окну величин, а не усреднением помесячных весов:
        # среднее произведения не равно произведению средних, и тождество бы поплыло
        d_old = e["ln_old"].mean() - b["ln_old"].mean()
        d_all = e["ln_all"].mean() - b["ln_all"].mean()
        prem_b, prem_e = b["prem"].mean(), e["prem"].mean()
        w_b = w0
        w_e = (w0 * prem_b + (e["rel_all"].mean() - b["rel_all"].mean())) / prem_e
        prem_eff = (w_b + w_e) / 2 * (prem_e - prem_b)
        comp_eff = (prem_b + prem_e) / 2 * (w_e - w_b)
        rows.append({"ref_area": area, "d_all": d_all, "d_old": d_old,
                     "premium_effect": prem_eff, "composition_effect": comp_eff,
                     "residual": d_all - (d_old + prem_eff + comp_eff),
                     "w_base": w_b, "w_last": w_e})
    tab = pd.DataFrame(rows)
    med = tab.median(numeric_only=True)
    return {"w0": w0, "n_regions": int(len(tab)), "by_region": tab,
            "median": {k: float(v) for k, v in med.items()},
            "share_composition_in_growth": float(med["composition_effect"] / med["d_all"]),
            "share_premium_in_growth": float(med["premium_effect"] / med["d_all"]),
            "max_abs_residual": float(tab["residual"].abs().max())}


def gap_rate_by_measure(df: pd.DataFrame, gap_df: pd.DataFrame, segment: str = "Вторичный",
                       rate_col: str = "rate", lags: int = 3) -> pd.DataFrame:
    """Цикл против состава: накопленная реакция разрыва на ставку на трёх мерах цены сделки.

    Сдвиг состава — медленный тренд, и если реакция на ставку держится на однородной по
    возрасту группе так же, как на общей, значит это разные явления и цикл составом не объясняется.
    """
    from . import gap as gap_mod

    s = gap_df[gap_df["segment"] == segment].merge(
        df[["ref_area", "m", "sec_all", "sec_old"]], on=["ref_area", "m"], how="inner")
    s["gap_all2"] = np.log(s["offer"] / s["sec_all"])
    s["gap_old"] = np.log(s["offer"] / s["sec_old"])
    rows = []
    for col, label in [("gap", "цены сделок (исходный набор)"), ("gap_all2", "этот набор, всего"),
                       ("gap_old", "этот набор, старше 5 лет")]:
        z = s.copy()
        z["gap"] = z[col]
        dl = gap_mod.rate_dlag(z, segment, rate_col, lags)
        rows.append({"мера": label, "накопленная_реакция_пп": dl["cum"] * 100,
                     "ст_ош": dl["cum_se"] * 100, "p": dl["cum_p"],
                     "ди_низ": dl["cum_ci"][0] * 100, "ди_верх": dl["cum_ci"][1] * 100,
                     "n": int(dl["fit"].nobs)})
    return pd.DataFrame(rows)


def gap_composition(df: pd.DataFrame, gap_df: pd.DataFrame, segment: str = "Вторичный") -> pd.DataFrame:
    """Сколько из схождения «объявление минус сделка» объясняется составом сделок.

    Разрыв пересчитывается на трёх мерах цены сделки: исходной («Цены сделок»), той же
    величине из этого набора («Всего») и однородной группе «старше 5 лет». Если тренд
    держится только на двух первых, значит схождение — это сдвиг состава сделок к молодому
    жилью, а не сближение запроса с ценой.
    """
    s = gap_df[gap_df["segment"] == segment].merge(
        df[["ref_area", "m", "sec_all", "sec_old", "sec_new"]], on=["ref_area", "m"], how="inner")
    s["gap_all2"] = np.log(s["offer"] / s["sec_all"])
    s["gap_old"] = np.log(s["offer"] / s["sec_old"])
    s["drift"] = np.log(s["sec_all"] / s["sec_old"])
    s["year"] = (s["m"] - s["m"].min()).dt.days / 365.25
    rows = []
    for col, label in [("gap", "цены сделок (исходный набор)"), ("gap_all2", "этот набор, всего"),
                       ("gap_old", "этот набор, старше 5 лет"), ("drift", "сдвиг состава")]:
        f = pr.fit(s, col, ["year"], absorb=["ref_area"], cluster=["ref_area", "m"])
        t = f.table().loc["year"]
        rows.append({"мера": label, "уровень": float(s[col].median()),
                     "тренд_пп_в_год": float(t.coef * 100), "ст_ош": float(t.se * 100),
                     "p": float(t.p), "ди_низ": float(t.ci_low * 100), "ди_верх": float(t.ci_high * 100),
                     "n": f.nobs, "регионов": f.n_clusters.get("ref_area"),
                     "месяцев": f.n_clusters.get("m")})
    return pd.DataFrame(rows)
