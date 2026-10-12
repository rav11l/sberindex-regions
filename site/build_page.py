"""Сборка страницы серии для digitaleconomylab.ru (блок T123 в Tilda).

Страница двуязычная: русский и английский варианты лежат рядом, лишний скрыт CSS
(`[data-en]{display:none}` и наоборот). Разметка и классы — те же, что на странице
«Шока по расписанию», чтобы серия выглядела одинаково.

Запуск: python site/build_page.py  → site/sberindex-regions_T123.html
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import charts as ch

HERE = pathlib.Path(__file__).parent
OUT = HERE / "sberindex-regions_T123.html"
GITHUB = "https://github.com/rav11l/sberindex-regions"
DOI = None          # проставляется после депозита на Zenodo


def two(tag: str, ru: str, en: str, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<{tag}{c} data-ru>{ru}</{tag}><{tag}{c} data-en>{en}</{tag}>"


def figure(svg: str, cap_ru: str, cap_en: str, note_ru: str = "", note_en: str = "") -> str:
    out = ["<figure>", two("figcaption", cap_ru, cap_en),
           '<div class="svgwrap">', svg, "</div>"]
    if note_ru:
        out.append(two("p", note_ru, note_en, cls="note"))
    out.append("</figure>")
    return "".join(out)


def section(num: str, h_ru: str, h_en: str, body: str) -> str:
    return (f'<section><div class="sec-head"><span class="sec-num">{num}</span>'
            f'<div>{two("h2", h_ru, h_en)}</div></div>{body}</section>')


# ───────────────────────────── рисунки ─────────────────────────────

fig_covid = ch.hbars(
    [("панельный", "cross-section", [0.08, 0.05]),
     ("CUSUM", "CUSUM", [0.22, 0.10]),
     ("BOCPD", "BOCPD", [0.41, 0.05]),
     ("BinSeg", "BinSeg", [0.62, 0.11]),
     ("z-score", "z-score", [0.63, 0.09]),
     ("PELT", "PELT", [0.63, 0.11])],
    series=[ch.S2, ch.REST], fmt="{:.2f}",
    legend=[("месяцы эха", "echo months"), ("спокойные месяцы", "calm months")])

fig_deals = ch.hbars(
    [("ets_relative", "ets_relative", [2181]), ("factor_only", "factor_only", [2538]),
     ("ens3_mean", "ens3_mean", [2569]), ("ens_ets_snaive", "ens_ets_snaive", [2712]),
     ("snaive_growth", "snaive_growth", [3672]), ("prophet", "prophet", [3709]),
     ("prophet_relative", "prophet_relative", [3830])],
    series=[ch.ACCENT], fmt="{:.0f}", left=150)

GAP = json.loads((HERE / "data_gap.json").read_text(encoding="utf-8"))
fig_gap = ch.lines_two_panel(
    GAP["labels"],
    [{"v": GAP["sec"], "c": ch.ACCENT, "ru": "вторичка", "en": "resale"},
     {"v": GAP["pri"], "c": ch.S1, "ru": "первичка", "en": "new-build"}],
    [{"v": GAP["rate"], "c": ch.MUTED, "ru": "ставка", "en": "key rate"}],
    top_ru="разрыв, % от цены сделки", top_en="gap, % of the transaction price",
    bot_ru="ключевая ставка, %", bot_en="key rate, %")

fig_novelty = ch.stacked_bar(
    [("возраст жилья", "age of the housing", 18.9, ch.ACCENT),
     ("собственно первичка", "the new-build market itself", 4.1, ch.S2)],
    "Разрыв первички и жилья старше пяти лет: 24,0",
    "New-build versus housing older than five years: 24.0")

fig_rate = ch.vbars_grouped(
    [("15–24", "15–24", [-1.31, -0.22]), ("25–34", "25–34", [-0.19, 0.29]),
     ("35–64", "35–64", [0.04, -0.04]), ("65+", "65+", [1.74, -0.30])],
    [ch.GREEN, ch.S1],
    [("повышения ставки", "rate hikes"), ("снижения ставки", "rate cuts")],
    faded={(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)})

# ───────────────────────────── текст ─────────────────────────────

HEAD = f'''<div class="wrap">
  <div class="backtop"><a class="backlink" href="https://digitaleconomylab.ru/"><span data-ru>← Digital Economy Lab · все материалы</span><span data-en>← Digital Economy Lab · all materials</span></a></div>
  <div class="top">
    <span class="kicker" data-ru>Digital Economy Lab · Серия исследований</span>
    <span class="kicker" data-en>Digital Economy Lab · Research series</span>
    <div class="seg" role="group" aria-label="Language">
      <button type="button" id="sr-ru" aria-pressed="true">RU</button>
      <button type="button" id="sr-en" aria-pressed="false">EN</button>
    </div>
  </div>
  <div class="head">
    <h1 data-ru>Региональные панели СберИндекса</h1>
    <h1 data-en>SberIndex regional panels</h1>
    <p class="sub" data-ru>Пять исследований об одном приёме: ряд как отклонение от общего фактора панели</p>
    <p class="sub" data-en>Five studies on one trick: a series as its deviation from the panel's common factor</p>
    <p class="byline" data-ru><b>Равиль Ахтямов</b>, кандидат экономических наук<br><span class="m">Digital Economy Lab, Казань · ORCID 0000-0002-0783-8573</span></p>
    <p class="byline" data-en><b>Ravil Akhtyamov</b>, Candidate of Economic Sciences<br><span class="m">Digital Economy Lab, Kazan · ORCID 0000-0002-0783-8573</span></p>
    <div class="stamp">
      <span class="status" data-ru>прогон · 12.10.2026</span>
      <span class="status" data-en>run · 12.10.2026</span>
      <span data-ru>Данные: <b>СберИндекс</b>, Домклик, Банк России</span>
      <span data-en>Data: <b>SberIndex</b>, Domclick, Bank of Russia</span>
      <span data-ru>Код: <a href="{GITHUB}">GitHub</a> · MIT</span>
      <span data-en>Code: <a href="{GITHUB}">GitHub</a> · MIT</span>
    </div>
  </div>'''

SUMMARY = '''<div class="summary">
  <span class="lbl" data-ru>Коротко</span>
  <span class="lbl" data-en>In brief</span>
  <p data-ru>Региональная статистика устроена так, что отдельный ряд почти ничего о себе не говорит: истории мало, шума много, а всё заметное в нём — общее для страны. Приём, на котором держатся все пять работ, простой: смотреть не на ряд, а на его отклонение от общего фактора панели. Тогда праздники, инфляция и сам макрошок сокращаются, и остаётся то, чем регион или группа отличается от остальных.</p>
  <p data-en>Regional statistics are arranged so that a single series says almost nothing about itself: the history is short, the noise is large, and everything visible in it is common to the whole country. The trick behind all five studies is simple: look not at the series but at its deviation from the panel's common factor. Holidays, inflation and the macro shock itself then cancel out, and what remains is how a region or a group differs from the rest.</p>
  <ol>
    <li data-ru><b>Приём решает больше, чем модель.</b> На 117 месяцах цен сделок с жильём контрольная модель без единого обученного параметра обыгрывает Prophet на 32 %, и это уже не вопрос мощности теста: p < 0,0001.</li>
    <li data-en><b>The trick matters more than the model.</b> On 117 months of housing transaction prices, a control model with no fitted parameters beats Prophet by 32 %, and this is no longer a question of statistical power: p &lt; 0.0001.</li>
    <li data-ru><b>Тот же приём делает детектор слепым — и в этом его польза.</b> Панельный детектор не видит общего шока, потому что шок общий; по той же причине он не принимает за событие эхо прошлогодней базы, на которое попадаются две трети рядов у остальных методов.</li>
    <li data-en><b>The same trick makes a detector blind — and that is its value.</b> The panel detector does not see a common shock, because the shock is common; for the same reason it does not mistake last year's base effect for an event, which trips two thirds of the series for the other methods.</li>
    <li data-ru><b>Он же снимает результаты, которые выглядели настоящими.</b> Схождение цен объявлений и цен сделок оказалось сдвигом состава сделок, а предсказательная сила разрыва — возвратом к среднему шумной цены. Обе находки сняты в этой же серии и теми же данными.</li>
    <li data-en><b>And it also removes results that looked real.</b> The convergence of asking prices and transaction prices turned out to be a shift in the composition of transactions, and the predictive power of the gap turned out to be mean reversion of a noisy price. Both findings are retracted within this same series, on the same data.</li>
  </ol>
  <div class="kpis">
    <div class="kpi"><div class="v">−32 %</div>
      <div class="t" data-ru>ошибка контрольной модели без обученных параметров против Prophet, p &lt; 0,0001</div>
      <div class="t" data-en>error of a control model with no fitted parameters versus Prophet, p &lt; 0.0001</div></div>
    <div class="kpi"><div class="v">0,08 / 0,62</div>
      <div class="t" data-ru>доля ложных тревог в месяцы эха: панельный детектор против методов на собственной истории ряда</div>
      <div class="t" data-en>share of false alarms in echo months: the panel detector versus methods using a series' own history</div></div>
    <div class="kpi"><div class="v">−1,3 %</div>
      <div class="t" data-ru>траты группы 15–24 в неделю решения ЦБ, на процентный пункт повышения ставки</div>
      <div class="t" data-en>spending of the 15–24 group in the week of a rate decision, per percentage point of the hike</div></div>
  </div>
</div>'''

S1_BODY = (
    two("p", """В «Шоке по расписанию» детекторы сдвигов проверялись на искусственных ступеньках и на паводках весны 2024 года, которые по величине оказались сравнимы с обычным шумом ряда. Вопрос «работает ли метод на настоящем шоке» остался без ответа. Ковид даёт то, чего не хватало: событие известной даты, одновременное для всей страны и огромное по величине. Медианный регион в апреле 2020 года показывает −50 % год к году против +18 % в феврале — обвал на 68 процентных пунктов.""",
        """In «Shocks on Schedule» the break detectors were tested on synthetic steps and on the spring 2024 floods, which turned out to be comparable in size to the ordinary noise of the series. The question «does the method work on a real shock» stayed unanswered. Covid provides what was missing: an event with a known date, simultaneous across the country and enormous in size. In April 2020 the median region shows −50 % year on year against +18 % in February — a 68 percentage point collapse.""") +
    two("p", """Панельный детектор в этот месяц не даёт ни одной тревоги. Это не сбой: он сравнивает регион с другими регионами того же месяца, а упали все одинаково — разброс между регионами 16,7 п. п. против 15,9 п. п. в феврале. Общий шок для него по построению не событие. Зато через год тот же ряд даёт всплеск — не потому, что что-то случилось, а потому, что база год назад была провальной, — и на это эхо попадаются почти все остальные методы.""",
        """In that month the panel detector raises no alarm at all. This is not a failure: it compares a region with other regions in the same month, and all of them fell alike — the spread between regions is 16.7 pp against 15.9 pp in February. A common shock is by construction not an event for it. A year later the same series spikes — not because anything happened, but because the base a year earlier was collapsed — and almost all the other methods are caught by that echo.""") +
    figure(fig_covid,
           "Доля рядов с тревогой: месяцы эха против спокойных месяцев",
           "Share of series raising an alarm: echo months versus calm months",
           "Эхо — апрель–июнь 2021 года, ровно год спустя после обвала. Чем выше столбец, тем чаще детектор принимает за событие арифметику прошлогодней базы.",
           "The echo is April–June 2021, exactly a year after the collapse. The taller the bar, the more often the detector mistakes last year's base arithmetic for an event.") +
    two("p", """Полезный вывод не в том, что один детектор лучше другого, а в том, что слепота к общему и устойчивость к эху — это одно и то же свойство. Выбирая метод, выбирают не чувствительность, а то, что считать событием.""",
        """The useful conclusion is not that one detector is better than another, but that blindness to the common component and robustness to the echo are the same property. Choosing a method means choosing not sensitivity but what counts as an event.""")
)

S2_BODY = (
    two("p", """У муниципальной панели было 24 месяца и шесть целевых месяцев: после поправки на перекрытие горизонтов тест Диболда–Мариано давал p = 0,18, и выводы приходилось строить на доле выигранных рядов и бутстрапе. Цены сделок с жильём дают 117 месяцев по 85 регионам и двум сегментам рынка: 22 точки прогноза, 66 целевых месяцев, 10 230 пар «ряд — точка — горизонт». Мощности наконец достаточно.""",
        """The municipal panel had 24 months and six target months: after correcting for overlapping horizons the Diebold–Mariano test gave p = 0.18, and the conclusions had to rest on the share of series won and on a bootstrap. Housing transaction prices give 117 months across 85 regions and two market segments: 22 forecast origins, 66 target months, 10,230 series–origin–horizon pairs. The power is finally sufficient.""") +
    figure(fig_deals,
           "Средняя абсолютная ошибка, рублей за квадратный метр",
           "Mean absolute error, roubles per square metre",
           "Сезонный наивный прогноз (12 047 руб.) не показан: он сжал бы шкалу втрое.",
           "The seasonal naive forecast (12,047 roubles) is not shown: it would compress the scale threefold.") +
    two("p", """Контрольная модель <b>factor_only</b> не обучает ни одного параметра: она берёт отклонение региона от общей динамики, замораживает его и двигает вместе с этой динамикой. Этого хватает, чтобы обыграть Prophet на 32 % (p &lt; 0,0001, лучше в 94 % рядов). Надстройка над приёмом добавляет немного и уже с трудом: ETS на отклонении даёт ещё −357 руб. (p = 0,0007).""",
        """The control model <b>factor_only</b> fits no parameters at all: it takes a region's deviation from the common path, freezes it and moves it along with that path. That is enough to beat Prophet by 32 % (p &lt; 0.0001, better in 94 % of series). Building on top of the trick adds little and with effort: ETS on the deviation gives a further −357 roubles (p = 0.0007).""") +
    two("p", """А вот перенос приёма на сам Prophet здесь не сработал: <b>prophet_relative</b> на 121 руб. хуже обычного Prophet. На муниципальных расходах было наоборот. Объяснение в природе ряда: цены жилья сильно трендовые, Prophet оценивает тренд сам, и вычитание общего фактора ломает ему именно ту часть, которую он умеет. Переносится приём, а не рецепт целиком.""",
        """Transferring the trick to Prophet itself did not work here: <b>prophet_relative</b> is 121 roubles worse than plain Prophet. On municipal spending it was the other way round. The explanation lies in the nature of the series: housing prices are strongly trending, Prophet estimates the trend itself, and subtracting the common factor breaks exactly the part it is good at. What transfers is the trick, not the whole recipe.""")
)

S3_BODY = (
    two("p", """Домклик публикует цены предложения в объявлениях, СберИндекс — цены состоявшихся сделок. Разрыв между ними — примерно то, насколько запрос продавца выше того, за что жильё уходит: 10,3 % на вторичном рынке и 3,4 % на первичном, где торговаться не с кем. За четыре года разрыв на вторичке сжался на 1,35 п. п. в год, на первичке тренда нет вовсе.""",
        """Domclick publishes asking prices from listings, SberIndex publishes prices of completed transactions. The gap between them is roughly how far above the eventual price the seller asks: 10.3 % on the resale market and 3.4 % on the new-build market, where there is nobody to bargain with. Over four years the resale gap narrowed by 1.35 pp a year; on the new-build market there is no trend at all.""") +
    figure(fig_gap,
           "Разрыв между ценой объявления и ценой сделки; ниже — ключевая ставка",
           "The gap between the asking price and the transaction price; below, the key rate",
           "Две величины — две панели: общая ось времени, но ни одной попытки свести разные шкалы на одно поле.",
           "Two quantities, two panels: a shared time axis, and no attempt to force different scales onto one field.") +
    two("p", """Устойчивый результат здесь один: ставка разводит объявление и сделку. Накопленная за квартал реакция — около 0,2 п. п. разрыва на процентный пункт ставки, и без 2022 года она становится крупнее, а не меньше. Механика знакома любому, кто продавал квартиру: цену в объявлении не снижают первой, а покупатель с подорожавшей ипотекой исчезает. Сделки встают раньше, чем падают цены.""",
        """One result here is robust: the rate drives the asking price and the transaction price apart. The cumulative response over a quarter is about 0.2 pp of the gap per percentage point of the rate, and excluding 2022 it becomes larger, not smaller. The mechanics are familiar to anyone who has sold a flat: the asking price is not the first to fall, while the buyer facing a dearer mortgage disappears. Transactions stall before prices do.""") +
    '<div class="box warn">' +
    two("span", "Что не подтвердилось", "What did not hold up", cls="lbl") +
    two("p", """Разрыв выглядел опережающим индикатором цен: регион с завышенным запросом за полгода прибавлял к цене сделки около 2 %. Проверки это сняли. Тот же разрыв «объясняет» и прошлый прирост, с обратным знаком и того же порядка; по лагам сигнал затухает к шестому месяцу; снятие ограничения на коэффициенты показывает, что работает цена сделки со своим возвратом к среднему, а не запрос продавца; вневыборочного выигрыша нет.""",
        """The gap looked like a leading indicator of prices: a region with an inflated asking price added about 2 % to its transaction price over six months. The checks removed this. The same gap also «explains» past growth, with the opposite sign and of the same magnitude; across lags the signal decays by the sixth month; relaxing the restriction on the coefficients shows that what works is the transaction price reverting to its own level, not the seller's asking price; and there is no out-of-sample gain.""") +
    "</div>" +
    figure(ch.vbars_grouped(
        [("3 месяца", "3 months", [0.13, -0.11]), ("6 месяцев", "6 months", [0.20, -0.26]),
         ("12 месяцев", "12 months", [0.21, -0.40])],
        [ch.ACCENT, ch.S2],
        [("разрыв → будущий прирост", "gap → future growth"),
         ("тот же разрыв → прошлый прирост", "the same gap → past growth")],
        fmt="{:+.2f}"),
        "Плацебо-тест: предсказание будущего против «предсказания» прошлого",
        "Placebo test: predicting the future versus «predicting» the past",
        "Содержательного предсказания прошлого быть не может. Если оно того же порядка, что и предсказание будущего, работает механика, а не информация.",
        "There can be no meaningful prediction of the past. If it is of the same magnitude as the prediction of the future, what works is mechanics, not information.")
)

S4_BODY = (
    two("p", """На вторичном рынке СберИндекс публикует цены отдельно для жилья младше и старше пяти лет. Это позволяет разобрать знаменитый разрыв между первичкой и вторичкой — 24 логарифмических пункта у медианного региона. Оказывается, он почти весь состоит из возраста жилья: квартира в доме, сданном четыре года назад, стоит лишь на 4 % дешевле строящейся.""",
        """For the resale market SberIndex publishes prices separately for housing younger and older than five years. This makes it possible to take apart the famous gap between new-build and resale — 24 log points for the median region. It turns out to consist almost entirely of the age of the housing: a flat in a building completed four years ago costs only 4 % less than one still under construction.""") +
    figure(fig_novelty,
           "Из чего состоит разрыв первички и вторички, логарифмические пункты",
           "What the new-build versus resale gap is made of, log points",
           "Возрастная премия выросла с 13,8 пункта в 2022 году до 24,4 в 2025-м и составляет 21,7 сейчас. Превышение первички над молодой вторичкой за четыре года значимо не изменилось.",
           "The age premium grew from 13.8 points in 2022 to 24.4 in 2025 and stands at 21.7 now. The margin of new-build over young resale has not changed significantly over four years.") +
    two("p", """Отсюда же — поправка к предыдущей работе. Если цену сделки мерить не «в среднем», а на однородной группе «жильё старше пяти лет», схождение запроса и сделки исчезает: тренд −0,31 п. п. в год против −1,20 и p = 0,35. Индекс «в среднем» растёт относительно индекса фиксированного возраста на 1,63 п. п. в год — потому что в сделках стало больше молодого жилья. Цены не сближались; менялось то, что покупают.""",
        """Hence a correction to the previous study. If the transaction price is measured not «on average» but on the homogeneous group «housing older than five years», the convergence of asking and transaction prices disappears: a trend of −0.31 pp a year against −1.20, with p = 0.35. The «average» index grows relative to the fixed-age index by 1.63 pp a year — because transactions have shifted towards younger housing. Prices were not converging; what people buy was changing.""") +
    two("p", """Попутно — предупреждение для всех, кто берёт эти данные: ряд «Всего» не является смесью двух возрастных срезов. Он лежит между ними лишь в 45 % регион-месяцев, а в Татарстане идёт на 19 % ниже группы «старше пяти лет». Складывать уровни трёх рядов нельзя; восстанавливаются только изменения.""",
        """A warning in passing for anyone using these data: the «Total» series is not a mixture of the two age slices. It lies between them in only 45 % of region-months, and in Tatarstan it runs 19 % below the «older than five years» group. The levels of the three series cannot be combined; only changes can be recovered.""")
)

S5_BODY = (
    two("p", """Последний вопрос серии — не про жильё. СберИндекс публикует недельный индекс потребительской активности по четырём возрастным группам: 300 недель, 33 изменения ключевой ставки внутри них. Кто замечает решение ЦБ в ту неделю, когда оно объявлено?""",
        """The last question in the series is not about housing. SberIndex publishes a weekly consumer activity index for four age groups: 300 weeks, with 33 key rate changes inside them. Who notices the central bank's decision in the week it is announced?""") +
    figure(fig_rate,
           "Траты по возрастным группам в неделю решения, % на процентный пункт",
           "Spending by age group in the week of the decision, % per percentage point",
           "Бледным — оценки, неотличимые от нуля. Оценка для группы 65+ не пережила исключения по одному заседанию и в выводы не входит.",
           "Faded bars are estimates indistinguishable from zero. The estimate for the 65+ group did not survive leaving out one meeting at a time and is not part of the conclusions.") +
    two("p", """Первыми отвечают самые молодые: на плановых повышениях траты группы 15–24 падают в ту же неделю на 1,3 % на каждый процентный пункт повышения относительно общей динамики. У групп 25–34 и 35–64 не происходит ничего. Это странно ровно до того момента, пока не вспомнишь, у кого есть кредиты: у двадцатилетних долгов почти нет, а платёж по уже взятому не меняется в день заседания. Значит, работает не канал долга, а канал ожиданий.""",
        """The youngest respond first: on scheduled hikes the spending of the 15–24 group falls in the very same week by 1.3 % per percentage point of the hike, relative to the common path. For the 25–34 and 35–64 groups nothing happens. This is odd only until one remembers who holds the debt: people in their twenties have almost none, and the payment on a loan already taken does not change on the day of the meeting. So what works is not the debt channel but the expectations channel.""") +
    two("p", """Реакция живёт две недели: дальше плацебо-сдвиги дают «значимые» оценки не хуже настоящих событий, и всё, что сказано про третий месяц, было бы артефактом метода. На снижениях ставки не реагирует никто — но все снижения выборки пришлись на 2025–2026 годы, а повышения на 2021–2024-й, так что асимметрию поведения от смены периода эти данные не отделяют.""",
        """The response lives for two weeks: beyond that, placebo shifts produce «significant» estimates just as readily as real events, and anything said about the third month would be an artefact of the method. Nobody reacts to rate cuts — but all the cuts in the sample fall in 2025–2026 and all the hikes in 2021–2024, so these data cannot separate an asymmetry of behaviour from a change of period.""")
)

S6_BODY = (
    two("p", """В каждой работе результат проходит один и тот же набор проверок, и часть выводов их не переживает. Это записано там же, где сам результат, — не в сноске и не задним числом.""",
        """In every study the result goes through the same set of checks, and some conclusions do not survive them. That is recorded next to the result itself — not in a footnote and not after the fact.""") +
    '<ol class="num">' +
    two("li", "<b>Плацебо.</b> Выдуманные даты событий, сдвинутые окна, объяснение прошлого вместо будущего. Если метод «находит» эффект там, где его быть не может, дальние горизонты в выводы не идут.",
        "<b>Placebo.</b> Invented event dates, shifted windows, explaining the past instead of the future. If the method «finds» an effect where none can exist, distant horizons do not enter the conclusions.") +
    two("li", "<b>Перестановочные тесты.</b> При перекрывающихся окнах обычные p-значения врут. Те же изменения ставки раскладываются по случайным неделям пятьсот раз, и смотрят, как часто случайность даёт столько же.",
        "<b>Permutation tests.</b> With overlapping windows the usual p-values lie. The same rate changes are laid out over random weeks five hundred times, and one looks at how often chance produces as much.") +
    two("li", "<b>Исключение по одному событию и по годам.</b> Результат, который держится на одном заседании или на одном 2022 годе, — не результат.",
        "<b>Leaving out one event, and one year, at a time.</b> A result that rests on a single meeting, or on 2022 alone, is not a result.") +
    two("li", "<b>Вневыборочная проверка.</b> Там, где утверждается предсказание, коэффициенты оцениваются только на прошлом, и выигрыш считается на будущем.",
        "<b>Out-of-sample testing.</b> Where prediction is claimed, coefficients are estimated on the past alone and the gain is measured on the future.") +
    two("li", "<b>Сверка текста с прогоном.</b> Отдельный тест сравнивает числа в документах с результатами расчёта, так что разойтись незаметно они не могут.",
        "<b>Checking the text against the run.</b> A separate test compares the numbers in the documents with the results of the computation, so they cannot drift apart unnoticed.") +
    "</ol>"
)

LIMITS = ('<div class="limit">' +
    two("span", "Границы", "Limitations", cls="lbl") +
    '<ol class="num">' +
    two("li", "Выгрузки СберИндекса — агрегаты по картам одного банка, пусть и крупнейшего. Это не потребление и не весь рынок жилья.",
        "SberIndex data are aggregates of card transactions of one bank, albeit the largest. They are neither consumption nor the whole housing market.") +
    two("li", "Цены объявлений и цены сделок приходят из разных источников. Состав сделок проверен по возрасту жилья; состав объявлений не проверяется ничем — возрастного среза по ним нет.",
        "Asking prices and transaction prices come from different sources. The composition of transactions is checked by the age of the housing; the composition of listings is checked by nothing — there is no age slice for them.") +
    two("li", "Возрастная активность доступна только по стране целиком, поэтому отдельно по регионам о реакции на ставку сказать нечего.",
        "Age-specific activity is available only for the country as a whole, so nothing can be said about the regional response to the rate.") +
    two("li", "Решение о ставке не случайно: оно принимается в ответ на инфляцию и прочее, что одновременно влияет на траты. Относительная мера убирает общую часть, но не делает эксперимент экспериментом.",
        "A rate decision is not random: it responds to inflation and other things that affect spending at the same time. The relative measure removes the common part but does not turn the exercise into an experiment.") +
    two("li", "Четыре с половиной года — одна фаза цикла ставки. Разделить «ставку» и «всё, что менялось вместе с ней», эти данные не позволяют.",
        "Four and a half years are one phase of the rate cycle. These data cannot separate «the rate» from «everything that changed alongside it».") +
    "</ol></div>")

REPRO = (
    two("p", f"""Код, конфигурации и тексты всех пяти работ открыты: <a href="{GITHUB}">GitHub</a>, лицензия MIT. Сырые выгрузки в репозиторий не кладутся — их скачивает скрипт, а версия фиксируется контрольной суммой sha256, так что прогон на других данных это заметит.""",
        f"""The code, configurations and texts of all five studies are open: <a href="{GITHUB}">GitHub</a>, MIT licence. The raw data files are not stored in the repository — a script downloads them and the version is pinned by an sha256 checksum, so a run on different data will notice.""") +
    two("p", """Каждая работа запускается одной командой вида <code>python -m siregions.cli &lt;команда&gt; --config configs/&lt;файл&gt;.yaml</code>; все параметры лежат в YAML и переопределяются из командной строки.""",
        """Each study runs with a single command of the form <code>python -m siregions.cli &lt;command&gt; --config configs/&lt;file&gt;.yaml</code>; all parameters live in YAML and can be overridden from the command line.""")
)

FOOT = '''<div class="foot">
  <p data-ru><b>Digital Economy Lab · Равиль Ахтямов, к. э. н. · ORCID 0000-0002-0783-8573</b><br>Данные: СберИндекс (sberindex.ru), Домклик, Банк России. Код — MIT, текст страницы — CC BY 4.0.</p>
  <p data-en><b>Digital Economy Lab · Ravil Akhtyamov, PhD in Economics · ORCID 0000-0002-0783-8573</b><br>Data: SberIndex (sberindex.ru), Domclick, Bank of Russia. Code — MIT, page text — CC BY 4.0.</p>
</div>
<div class="backfoot"><a class="backlink" href="https://digitaleconomylab.ru/"><span data-ru>← Вернуться на главную</span><span data-en>← Back to the main page</span></a></div>
</div>'''

SCRIPT = '''<script>
(function(){
  var root=document.getElementById('dgl-sr');
  if(!root) return;
  var ru=document.getElementById('sr-ru'), en=document.getElementById('sr-en');
  function set(l){ root.setAttribute('data-lang',l); ru.setAttribute('aria-pressed',String(l==='ru')); en.setAttribute('aria-pressed',String(l==='en')); }
  ru.addEventListener('click',function(){set('ru');});
  en.addEventListener('click',function(){set('en');});
  function fromHost(){ var l=(document.documentElement.getAttribute('lang')||'').toLowerCase(); set(l.indexOf('en')===0?'en':'ru'); }
  fromHost();
  try{ new MutationObserver(fromHost).observe(document.documentElement,{attributes:true,attributeFilter:['lang']}); }catch(e){}
})();
</script>'''

EXTRA_CSS = """<style>
#dgl-sr figure svg{width:100%;height:auto;display:block;background:#fff;border:1px solid var(--line);padding:10px 8px}
/* на узком экране подписи внутри SVG становятся нечитаемыми, поэтому рисунок
   прокручивается вбок, а не сжимается */
@media (max-width:640px){
  #dgl-sr figure .svgwrap{overflow-x:auto;-webkit-overflow-scrolling:touch}
  #dgl-sr figure .svgwrap svg{min-width:560px}
}
#dgl-sr figure svg text{font-family:var(--sans)}
#dgl-sr code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:14px;background:var(--accent-soft);padding:1px 5px;border-radius:2px}
</style>"""


def build() -> str:
    base = (HERE / "base_css.html").read_text(encoding="utf-8")
    body = (HEAD + SUMMARY +
            section("01", "Ковид, эхо и панель", "Covid, the echo and the panel", S1_BODY) +
            section("02", "Длинная панель: когда тестов наконец хватает",
                    "A long panel: when the tests finally have power", S2_BODY) +
            section("03", "Разрыв «объявление минус сделка»",
                    "The asking-price minus transaction-price gap", S3_BODY) +
            section("04", "Новизна жилья", "The age of the housing", S4_BODY) +
            section("05", "Кто первым реагирует на ставку",
                    "Who reacts to the key rate first", S5_BODY) +
            section("06", "Что здесь считается проверкой",
                    "What counts as a check here", S6_BODY) +
            section("07", "Границы", "Limitations", LIMITS) +
            section("08", "Воспроизведение", "Reproduction", REPRO) +
            FOOT)
    return base + EXTRA_CSS + '<div id="dgl-sr" data-lang="ru">' + body + "</div>" + SCRIPT


if __name__ == "__main__":
    html = build()
    OUT.write_text(html, encoding="utf-8")
    payload = json.dumps([{"name": "code", "value": html}], ensure_ascii=False)
    print(f"символов: {len(html)}  ·  JSON для Tilda: {len(payload)} (потолок 120 000)")
