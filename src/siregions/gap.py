"""Разрыв «объявление минус сделка»: цены Домклик против цен сделок СберИндекса.

Что сравнивается. Домклик публикует средние цены предложения в объявлениях
(руб./м², по типу рынка), СберИндекс — средние цены состоявшихся сделок
(руб./м², первичный и вторичный рынок). Разрыв определяется как
    gap = log(цена объявления) − log(цена сделки),
то есть примерно относительная скидка, с которой объект уходит от запроса.

Чего разрыв НЕ измеряет. Это не торг по конкретному объекту: совокупности
разные. В объявлениях — всё выставленное, в сделках — только проданное, причём
сделки СберИндекса опираются на его собственный поток, а не на весь рынок.
Поэтому уровень разрыва включает и состав (метраж, район, класс), и охват
источников. Отсюда три решения:
  * уровень разрыва нигде не интерпретируется как размер торга;
  * первичный рынок используется как контроль: у застройщика прайс и есть цена
    сделки, значит разрыв там — почти чистая мера расхождения источников;
  * все выводы строятся на изменениях внутри региона, а не на уровнях.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as data_mod
from . import panelreg as pr

SEGMENTS = {
    "Первичный": {"offer": "Первичный рынок", "deal": "Первичный"},
    "Вторичный": {"offer": "Вторичный рынок", "deal": "Вторичный"},
}


def _month(df: pd.DataFrame) -> pd.DataFrame:
    """Разные наборы помечают месяц то первым, то последним числом — приводим к первому."""
    out = df.copy()
    out["m"] = pd.PeriodIndex(out["period"], freq="M").to_timestamp()
    return out


def build(cfg: dict) -> pd.DataFrame:
    """Панель регион × месяц × сегмент: цены объявлений, цены сделок, предложение, ставка."""
    d = cfg["datasets"]
    off = _month(data_mod.read(d["offers"]))
    deals = _month(data_mod.read(d["deals"]))
    alt = _month(data_mod.read(d["deals_alt"]))
    sup = _month(data_mod.read(d["supply"]))
    rate = _month(data_mod.read(d["rate"]))

    excl = cfg.get("exclude_areas", ["Россия"])
    for frame in (off, deals, alt, sup):
        frame.drop(frame.index[frame["ref_area"].isin(excl)], inplace=True)

    rows = []
    for seg, keys in SEGMENTS.items():
        o = off[off["price_type"] == keys["offer"]].set_index(["ref_area", "m"])["value"]
        dl = deals[deals["realty"] == keys["deal"]].set_index(["ref_area", "m"])["value"]
        s = sup[sup["offer_type"] == keys["offer"]].set_index(["ref_area", "m"])["value"]
        a = alt[(alt["real_estate_type"] == keys["offer"]) & (alt["real_estate_novelty"] == "Всего")]
        a = a.set_index(["ref_area", "m"])["value"]
        t = pd.concat([o.rename("offer"), dl.rename("deal"), s.rename("supply"),
                       a.rename("deal_alt")], axis=1)
        t["segment"] = seg
        rows.append(t.reset_index())
    df = pd.concat(rows, ignore_index=True)

    r = rate.pivot_table(index="m", columns="key_rate_categories", values="value")
    df = df.merge(r.rename(columns={"Ключевая ставка": "rate",
                                    "Ключевая ставка в реальном выражении": "rate_real",
                                    "Базовая инфляция по трем месяцам": "core_infl"}),
                  left_on="m", right_index=True, how="left")

    df = df.dropna(subset=["offer", "deal"])
    if cfg.get("start"):
        df = df[df["m"] >= pd.Timestamp(cfg["start"])]
    df["gap"] = np.log(df["offer"] / df["deal"])
    df["gap_alt"] = np.log(df["offer"] / df["deal_alt"])
    df["ln_offer"] = np.log(df["offer"])
    df["ln_deal"] = np.log(df["deal"])
    df["ln_supply"] = np.log(df["supply"])
    df["t"] = (df["m"].dt.year - df["m"].dt.year.min()) + (df["m"].dt.month - 1) / 12
    return df.sort_values(["segment", "ref_area", "m"]).reset_index(drop=True)


def diffs(df: pd.DataFrame, cols: list[str], lags: int = 0) -> pd.DataFrame:
    """Первые разности и лаги внутри ряда регион × сегмент (без заглядывания вперёд)."""
    out = df.sort_values(["segment", "ref_area", "m"]).copy()
    g = out.groupby(["segment", "ref_area"], sort=False)
    for c in cols:
        out["d_" + c] = g[c].diff()
        for l in range(1, lags + 1):
            out[f"d_{c}_l{l}"] = g["d_" + c].shift(l)
            out[f"{c}_l{l}"] = g[c].shift(l)
    return out


def trim(df: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """Отсекает крайние q и 1 − q наблюдений разрыва внутри сегмента.

    Нужно потому, что в нескольких регионах разрыв неправдоподобен (в Чеченской Республике
    он доходит до 0,77: поток сделок там мал, и состав объявлений с составом сделок почти
    не связан). Выводы, которые от такого отсечения меняются, не годятся.
    """
    keep = []
    for seg, sub in df.groupby("segment"):
        lo, hi = sub["gap"].quantile([q, 1 - q])
        keep.append(sub[(sub["gap"] >= lo) & (sub["gap"] <= hi)])
    return pd.concat(keep).sort_values(["segment", "ref_area", "m"]).reset_index(drop=True)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("segment")
              .agg(n=("gap", "size"), regions=("ref_area", "nunique"), months=("m", "nunique"),
                   start=("m", "min"), end=("m", "max"))
              .reset_index())


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Уровень и разброс разрыва по сегментам и годам."""
    g = df.assign(year=df["m"].dt.year).groupby(["segment", "year"])["gap"]
    out = g.agg(median="median", mean="mean", sd="std",
                p10=lambda x: x.quantile(0.10), p90=lambda x: x.quantile(0.90), n="size")
    return out.reset_index()


def common_series(df: pd.DataFrame, col: str = "gap") -> pd.DataFrame:
    """Общая часть разрыва: медиана по регионам в каждом месяце, по сегментам."""
    return df.pivot_table(index="m", columns="segment", values=col, aggfunc="median").sort_index()


def by_region(df: pd.DataFrame, segment: str = "Вторичный") -> pd.DataFrame:
    """Средний разрыв по региону и его изменение между первым и последним годом."""
    s = df[df["segment"] == segment].copy()
    s["year"] = s["m"].dt.year
    first, last = s["year"].min(), s["year"].max()
    tab = s.pivot_table(index="ref_area", columns="year", values="gap", aggfunc="median")
    out = pd.DataFrame({
        "gap_mean": s.groupby("ref_area")["gap"].mean(),
        f"gap_{first}": tab.get(first), f"gap_{last}": tab.get(last),
    })
    out["change"] = out[f"gap_{last}"] - out[f"gap_{first}"]
    return out.sort_values("gap_mean", ascending=False)


def trend(df: pd.DataFrame, segment: str, col: str = "gap") -> pr.Fit:
    """Тренд общей части разрыва: процентных пунктов в год, ошибки Ньюи–Уэста."""
    c = common_series(df, col)[segment].dropna()
    X = pd.DataFrame({"year": (c.index - c.index.min()).days / 365.25}, index=c.index)
    return pr.nw_ols(c, X)


def rate_link_ts(df: pd.DataFrame, segment: str = "Вторичный", rate_col: str = "rate") -> dict:
    """Связь общего разрыва со ставкой: уровни с трендом и первые разности.

    Уровни без тренда бессмысленны: и разрыв, и ставка за 2022–2026 имеют свои тренды,
    корреляция уровней их и ловит. Поэтому приводится регрессия с трендом и регрессия
    в разностях; выводы делаются по разностям.
    """
    c = common_series(df, "gap")[segment].dropna()
    r = df.drop_duplicates("m").set_index("m")[rate_col].reindex(c.index)
    lev = pr.nw_ols(c, pd.DataFrame({"year": (c.index - c.index.min()).days / 365.25,
                                     rate_col: r}))
    dif = pr.nw_ols(c.diff().dropna(), pd.DataFrame({"d_rate": r.diff()}).loc[c.index[1:]])
    xcorr = {int(l): float(c.diff().corr(r.diff().shift(l))) for l in range(0, 7)}
    return {"levels_with_trend": lev, "differences": dif, "xcorr_diff": xcorr}


def rate_link_panel(df: pd.DataFrame, segment: str = "Вторичный", rate_col: str = "rate") -> pr.Fit:
    """То же в панели: Δgap по региону на Δставки, эффекты региона, кластеры регион и месяц."""
    s = diffs(df[df["segment"] == segment], ["gap", rate_col, "ln_supply"])
    return pr.fit(s, "d_gap", ["d_" + rate_col], absorb=["ref_area"], cluster=["ref_area", "m"])


def rate_dlag(df: pd.DataFrame, segment: str = "Вторичный", rate_col: str = "rate",
              lags: int = 3) -> dict:
    """Распределённые лаги ставки: мгновенная реакция против накопленной.

    Одной мгновенной оценки мало: кросс-корреляция приростов меняет знак на третьем
    месяце, то есть часть расширения разрыва потом отыгрывается. Сумма коэффициентов
    по лагам 0..L и есть накопленный эффект; тест Вальда проверяет, отличается ли он
    от нуля.
    """
    s = diffs(df[df["segment"] == segment], ["gap", rate_col], lags=lags)
    terms = ["d_" + rate_col] + [f"d_{rate_col}_l{l}" for l in range(1, lags + 1)]
    f = pr.fit(s, "d_gap", terms, absorb=["ref_area"], cluster=["ref_area", "m"])
    w = np.ones(len(terms))
    V = f.vcov.loc[terms, terms].to_numpy()
    cum = float(f.params[terms].sum())
    se = float(np.sqrt(max(w @ V @ w, 0)))
    from scipy import stats as st
    return {"fit": f, "terms": terms, "cum": cum, "cum_se": se,
            "cum_p": float(2 * st.norm.sf(abs(cum / se))) if se > 0 else float("nan"),
            "cum_ci": [cum - 1.96 * se, cum + 1.96 * se],
            "joint_p": f.wald(terms)["p_F"]}


def rate_subsamples(df: pd.DataFrame, starts: list[str], segment: str = "Вторичный",
                    rate_col: str = "rate", lags: int = 3) -> pd.DataFrame:
    """Та же реакция на ставку на выборках, начинающихся позже.

    Март 2022 года двигал ставку на 10,4 п. п. за месяц — больше, чем всё остальное за
    четыре года вместе (максимум 2,7 п. п.). Если результат держится на одном этом месяце,
    он должен рассыпаться при отбрасывании 2022 года.
    """
    rows = []
    for start in starts:
        sub = df[df["m"] >= pd.Timestamp(start)]
        s = diffs(sub[sub["segment"] == segment], ["gap", rate_col])
        f = pr.fit(s, "d_gap", ["d_" + rate_col], absorb=["ref_area"], cluster=["ref_area", "m"])
        dl = rate_dlag(sub, segment, rate_col, lags)
        rows.append({"start": start, "months": int(s["m"].nunique()),
                     "instant": float(f.params["d_" + rate_col]), "instant_p": float(f.pvalue["d_" + rate_col]),
                     "cum": dl["cum"], "cum_se": dl["cum_se"], "cum_p": dl["cum_p"],
                     "cum_ci_low": dl["cum_ci"][0], "cum_ci_high": dl["cum_ci"][1]})
    return pd.DataFrame(rows)


def heterogeneity(df: pd.DataFrame, segment: str = "Вторичный", rate_col: str = "rate") -> pr.Fit:
    """Кто сильнее реагирует на ставку: Δставки × характеристика региона, с месячными эффектами.

    Месячные эффекты поглощают саму ставку (она одна на всю страну), поэтому здесь
    опознаётся только различие между регионами, зато опознаётся честно.
    Характеристики — уровень цены сделки и размер предложения в первом году выборки,
    то есть предопределённые по отношению к последующим изменениям.
    """
    s = df[df["segment"] == segment].copy()
    base_year = s["m"].dt.year.min()
    base = (s[s["m"].dt.year == base_year].groupby("ref_area")
              .agg(x_price=("ln_deal", "mean"), x_supply=("ln_supply", "mean")))
    base = (base - base.mean()) / base.std()
    s = s.merge(base, left_on="ref_area", right_index=True, how="inner")
    s = diffs(s, ["gap", rate_col])
    s["rate_x_price"] = s["d_" + rate_col] * s["x_price"]
    s["rate_x_supply"] = s["d_" + rate_col] * s["x_supply"]
    return pr.fit(s, "d_gap", ["rate_x_price", "rate_x_supply"],
                  absorb=["ref_area", "m"], cluster=["ref_area", "m"])


def lead_lag(df: pd.DataFrame, segment: str = "Вторичный", lags: int = 3) -> dict:
    """Кто подстраивается первым: объявление или сделка.

    Две регрессии в разностях с эффектами региона и месяца: прирост цены объявления
    на свои лаги и лаги прироста цены сделки — и наоборот. Месячные эффекты убирают
    общестрановую динамику, поэтому тест говорит об опережении внутри региона.
    Совместный тест лагов — Вальда на кластерной матрице (кластеры: регион, месяц).
    """
    s = diffs(df[df["segment"] == segment], ["ln_offer", "ln_deal"], lags=lags)
    sd = {c: float(s["d_" + c].std()) for c in ["ln_offer", "ln_deal"]}
    out = {"sd_monthly_growth": sd}
    for y, x in [("ln_offer", "ln_deal"), ("ln_deal", "ln_offer")]:
        terms_own = [f"d_{y}_l{l}" for l in range(1, lags + 1)]
        terms_oth = [f"d_{x}_l{l}" for l in range(1, lags + 1)]
        f = pr.fit(s, "d_" + y, terms_own + terms_oth,
                   absorb=["ref_area", "m"], cluster=["ref_area", "m"])
        # прирост объяснённой доли от лагов другого ряда: сравнение коэффициентов между
        # двумя уравнениями зависит от масштаба, а прирост R² — нет
        f_own = pr.fit(s.dropna(subset=terms_own + terms_oth), "d_" + y, terms_own,
                       absorb=["ref_area", "m"], cluster=["ref_area"])
        out[y] = {"fit": f, "r2_gain_other": float(f.r2_within - f_own.r2_within),
                  "wald_other": f.wald(terms_oth), "wald_own": f.wald(terms_own),
                  "sum_other": float(f.params[terms_oth].sum()),
                  "sum_own": float(f.params[terms_own].sum()),
                  # коэффициенты несравнимы напрямую: у цен сделок разброс больше, чем у
                  # цен объявлений, поэтому приводим к единицам ст. откл. обеих переменных
                  "sum_other_std": float(f.params[terms_oth].sum() * sd[x] / sd[y]),
                  "coefs_other": f.params[terms_oth].round(4).to_dict(),
                  "coefs_own": f.params[terms_own].round(4).to_dict()}
    return out


def predictive(df: pd.DataFrame, segment: str = "Вторичный", horizons=(3, 6, 12)) -> pd.DataFrame:
    """Несёт ли разрыв информацию о будущих ценах сделок.

    Цель: log P_deal(t+h) − log P_deal(t). Регрессоры: разрыв и прирост цены сделки за
    последние 12 месяцев. Эффекты региона и месяца: общий фактор цен, который в прежних
    прогонах объяснял почти всё, уходит в месячные эффекты, остаётся идиосинкразия региона.
    Ошибки кластеризованы по региону и по месяцу — перекрытие горизонтов этим учитывается.

    Главная опасность здесь механическая. Разрыв в месяце t содержит −log P_deal(t), и та
    же величина стоит в цели с обратным знаком. Шум в цене сделки месяца t (а он есть:
    средняя цена сделок зависит от состава проданного) сам по себе даёт положительный
    коэффициент — классическое возвращение к среднему, а не предсказание. Поэтому считаются
    три варианта:
      spec = "одновременный" — разрыв месяца t, подвержен этому смещению;
      spec = "лаг"           — разрыв месяца t − 1: шум цены сделки t попадает только в цель,
                               шум цены сделки t − 1 — только в регрессор, и при некоррелированном
                               по времени шуме механической связи не остаётся;
      spec = "плацебо"       — на том же разрыве месяца t объясняется ПРОШЛЫЙ прирост за h
                               месяцев. Содержательного предсказания тут быть не может,
                               а механика даёт отрицательный коэффициент: так видно её масштаб.
    """
    rows = []
    for seg in ([segment] if isinstance(segment, str) else segment):
        s = df[df["segment"] == seg].sort_values(["ref_area", "m"]).copy()
        g = s.groupby("ref_area", sort=False)
        s["g12"] = s["ln_deal"] - g["ln_deal"].shift(12)
        for l in (1, 3, 6):
            s[f"gap_l{l}"] = g["gap"].shift(l)
            s[f"g12_l{l}"] = g["g12"].shift(l)
        for h in horizons:
            s["y_fwd"] = g["ln_deal"].shift(-h) - s["ln_deal"]
            s["y_past"] = s["ln_deal"] - g["ln_deal"].shift(h)
            # в варианте с лагом и контроль берётся лагированным, иначе он сам тянет
            # за собой цену сделки месяца t; в плацебо контроль не нужен — при h = 12
            # он совпал бы с целью тождественно
            for spec, y, xs in [("одновременный", "y_fwd", ["gap", "g12"]),
                                ("лаг", "y_fwd", ["gap_l1", "g12_l1"]),
                                ("лаг 3", "y_fwd", ["gap_l3", "g12_l3"]),
                                ("лаг 6", "y_fwd", ["gap_l6", "g12_l6"]),
                                ("плацебо", "y_past", ["gap"])]:
                x = xs[0]
                f = pr.fit(s, y, xs, absorb=["ref_area", "m"], cluster=["ref_area", "m"])
                t = f.table().loc[x]
                rows.append({"segment": seg, "spec": spec, "h": h, "coef": t.coef, "se": t.se,
                             "p": t.p, "ci_low": t.ci_low, "ci_high": t.ci_high,
                             "mde": pr.mde(t.se), "n": f.nobs,
                             "n_regions": f.n_clusters.get("ref_area"),
                             "n_months": f.n_clusters.get("m")})
    return pd.DataFrame(rows)


def decompose_fwd(df: pd.DataFrame, segment: str = "Вторичный", horizons=(3, 6, 12)) -> pd.DataFrame:
    """Чей это сигнал: объявления или сделки.

    Разрыв — разность двух логарифмов, и регрессия на него накладывает ограничение
    «коэффициенты при цене объявления и цене сделки равны по модулю и противоположны».
    Здесь то же уравнение оценивается без этого ограничения. Если информацию несёт
    запрос продавца, значимым должен быть коэффициент при цене объявления. Если же всё
    дело в возвращении к среднему шумной цены сделки, останется только отрицательный
    коэффициент при ней, а ограничение будет отвергнуто.
    """
    rows = []
    s = df[df["segment"] == segment].sort_values(["ref_area", "m"]).copy()
    g = s.groupby("ref_area", sort=False)
    for h in horizons:
        s["y"] = g["ln_deal"].shift(-h) - s["ln_deal"]
        f = pr.fit(s, "y", ["ln_offer", "ln_deal"], absorb=["ref_area", "m"],
                   cluster=["ref_area", "m"])
        b = f.params
        V = f.vcov
        w = np.array([1.0, 1.0])
        se_sum = float(np.sqrt(max(w @ V.loc[["ln_offer", "ln_deal"], ["ln_offer", "ln_deal"]].to_numpy() @ w, 0)))
        from scipy import stats as st
        ssum = float(b["ln_offer"] + b["ln_deal"])
        rows.append({"h": h, "coef_offer": float(b["ln_offer"]), "se_offer": float(f.se["ln_offer"]),
                     "p_offer": float(f.pvalue["ln_offer"]),
                     "coef_deal": float(b["ln_deal"]), "se_deal": float(f.se["ln_deal"]),
                     "p_deal": float(f.pvalue["ln_deal"]),
                     "sum": ssum, "se_sum": se_sum,
                     "p_restriction": float(2 * st.norm.sf(abs(ssum / se_sum))) if se_sum > 0 else np.nan,
                     "n": f.nobs})
    return pd.DataFrame(rows)


def oos(df: pd.DataFrame, segment: str = "Вторичный", h: int = 6,
        min_train: int = 24, n_boot: int = 2000, seed: int = 0,
        mode: str = "absolute", gap_col: str = "gap", region_fe: bool = True) -> dict:
    """Вневыборочная проверка прогнозной ценности разрыва.

    На каждой точке T коэффициенты оцениваются только на наблюдениях, где и регрессоры,
    и цель уже известны (то есть t + h ≤ T), затем прогноз строится для t = T.
    Сравниваются две модели одного вида: с разрывом и без него; обе с константой и
    прошлым годовым приростом. Разность средних абсолютных ошибок проверяется блочным
    бутстрапом по целевым месяцам.

    mode = "absolute" — прогнозируется прирост как он есть. Ошибку тогда определяет
        общестрановая динамика, которую в реальном времени не знает никто, и
        идиосинкразия региона в ней тонет.
    mode = "relative" — и прогноз, и факт берутся как отклонение от среднего по стране
        в целевом месяце. Это уже не настоящий прогноз (среднее целевого месяца взято из
        будущего), а ответ на вопрос «если общий фактор известен, помогает ли разрыв»:
        ровно тот приём, который в прошлых прогонах давал почти весь выигрыш.
    """
    s = df[df["segment"] == segment].sort_values(["ref_area", "m"]).copy()
    g = s.groupby("ref_area", sort=False)
    s["g12"] = s["ln_deal"] - g["ln_deal"].shift(12)
    s["gap_l1"] = g["gap"].shift(1)
    s["y"] = g["ln_deal"].shift(-h) - s["ln_deal"]
    base = None
    months = sorted(s["m"].unique())
    recs = []
    for origin in months[min_train:]:
        tr = s[(s["m"] <= origin - pd.DateOffset(months=h)) & s["y"].notna()].dropna(subset=[gap_col, "g12"])
        te = s[(s["m"] == origin) & s["y"].notna()].dropna(subset=[gap_col, "g12"])
        if len(tr) < 200 or te.empty:
            continue
        for name, xs in [("с разрывом", [gap_col, "g12"]), ("без разрыва", ["g12"])]:
            ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
            Xtr = [tr[c].to_numpy() for c in xs]
            Xte = [te[c].to_numpy() for c in xs]
            if region_fe:
                # эффекты региона оцениваются только на обучающей части: это известно в точке T
                mu = {c: tr.groupby("ref_area")[c].mean() for c in ["y"] + xs}
                base = te["ref_area"].map(mu["y"])
                if base.isna().any():
                    continue
                ytr = ytr - tr["ref_area"].map(mu["y"]).to_numpy()
                yte_dm = yte - base.to_numpy()
                Xtr = [tr[c].to_numpy() - tr["ref_area"].map(mu[c]).to_numpy() for c in xs]
                Xte = [te[c].to_numpy() - te["ref_area"].map(mu[c]).to_numpy() for c in xs]
            X = np.column_stack([np.ones(len(tr))] + Xtr)
            beta = np.linalg.lstsq(X, ytr, rcond=None)[0]
            Xt = np.column_stack([np.ones(len(te))] + Xte)
            yhat = Xt @ beta + (base.to_numpy() if region_fe else 0.0)
            recs.append(pd.DataFrame({"model": name, "origin": origin, "ref_area": te["ref_area"].to_numpy(),
                                      "target": origin + pd.DateOffset(months=h),
                                      "y": yte, "yhat": yhat}))
    pred = pd.concat(recs, ignore_index=True)
    if mode == "relative":
        k = ["model", "target"]
        pred["y"] = pred["y"] - pred.groupby(k)["y"].transform("mean")
        pred["yhat"] = pred["yhat"] - pred.groupby(k)["yhat"].transform("mean")
    pred["ae"] = (pred["y"] - pred["yhat"]).abs()
    w = pred.pivot_table(index=["origin", "ref_area", "target"], columns="model", values="ae")
    d = (w["с разрывом"] - w["без разрыва"]).dropna()
    tgt = d.index.get_level_values("target").to_numpy()
    rng = np.random.default_rng(seed)
    uniq = pd.unique(tgt)
    blocks = [d.to_numpy()[tgt == m] for m in uniq]
    sums = np.array([b.sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"h": h, "mode": mode, "gap_col": gap_col, "region_fe": region_fe,
            "n_pairs": int(len(d)), "n_origins": int(w.index.get_level_values("origin").nunique()),
            "mae_with": float(w["с разрывом"].mean()), "mae_without": float(w["без разрыва"].mean()),
            "mean_diff": float(d.mean()),
            "rel_gain": float(-d.mean() / w["без разрыва"].mean()),
            "boot_ci_low": float(lo), "boot_ci_high": float(hi),
            "share_better": float((d < 0).mean())}


def source_check(df: pd.DataFrame) -> pd.DataFrame:
    """Сверка двух наборов цен сделок СберИндекса и устойчивость разрыва к выбору набора."""
    s = df.dropna(subset=["deal", "deal_alt"]).copy()
    s["ratio"] = s["deal"] / s["deal_alt"]
    out = (s.groupby("segment")
             .agg(n=("ratio", "size"), ratio_median=("ratio", "median"), ratio_sd=("ratio", "std"),
                  gap_median=("gap", "median"), gap_alt_median=("gap_alt", "median"),
                  corr_month=("gap", lambda x: np.nan))
             .reset_index())
    # корреляция изменений разрыва, посчитанных по двум наборам
    corr = []
    for seg, sub in s.groupby("segment"):
        z = diffs(sub, ["gap", "gap_alt"])
        corr.append(z["d_gap"].corr(z["d_gap_alt"]))
    out["corr_d_gap"] = corr
    return out.drop(columns=["corr_month"])


def primary_as_control(df: pd.DataFrame) -> dict:
    """Вторичный минус первичный в регионах, где есть оба сегмента.

    Если бы снижение разрыва объяснялось дрейфом охвата источников, оно было бы видно и
    в первичном сегменте, где торга почти нет. Разность трендов отделяет одно от другого
    ценой предположения, что дрейф охвата одинаков в двух сегментах.
    """
    w = df.pivot_table(index=["ref_area", "m"], columns="segment", values="gap").dropna()
    d = (w["Вторичный"] - w["Первичный"]).rename("gap_diff").reset_index()
    c = d.groupby("m")["gap_diff"].median()
    X = pd.DataFrame({"year": (c.index - c.index.min()).days / 365.25}, index=c.index)
    return {"n_regions": int(d["ref_area"].nunique()), "n": int(len(d)),
            "level_median": float(d["gap_diff"].median()), "trend": pr.nw_ols(c, X)}
