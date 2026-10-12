"""Статические inline-SVG для страницы серии на digitaleconomylab.ru.

Почему статические, а не рисуемые скриптом, как на странице «Шока по расписанию»: данных
здесь мало (шесть детекторов, семь моделей, четыре возрастные группы), а блок Tilda ограничен
120 000 символов JSON — готовый SVG дешевле, чем данные плюс рисовальщик.

Двуязычие — теми же атрибутами data-ru/data-en, что и остальной текст: CSS прячет ненужный
вариант, в том числе у элементов внутри SVG.
"""
from __future__ import annotations

ACCENT, S1, S2, S3 = "#1D3C5C", "#3A6EA5", "#C9821F", "#7B5EA7"
GREEN, REST, INK, MUTED, LINE = "#2C5F44", "#C9C3B8", "#15181C", "#5F6873", "#E2DED6"


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def t(x, y, text_ru, text_en, size=11.5, fill=MUTED, anchor="start", weight=None):
    w = f' font-weight="{weight}"' if weight else ""
    out = (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
           f'text-anchor="{anchor}"{w} data-ru>{_esc(text_ru)}</text>')
    if text_en is not None:
        out += (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                f'text-anchor="{anchor}"{w} data-en>{_esc(text_en)}</text>')
    return out


def _svg(inner: str, w: int, h: int) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" role="img" preserveAspectRatio="xMidYMid meet">'
            f'{inner}</svg>')


def hbars(rows, w=700, bar=20, gap=13, left=168, right=58, top=8, fmt="{:.2f}",
          series=None, legend=None):
    """Горизонтальные столбцы. rows: (подпись ru, подпись en, [значения]), один или два ряда."""
    series = series or [ACCENT]
    n_ser = len(series)
    h = top + len(rows) * (bar * n_ser + gap) + (34 if legend else 10)
    mx = max(max(r[2]) for r in rows) or 1
    scale = (w - left - right) / mx
    out = []
    y = top
    if legend:
        for i, (lab_ru, lab_en) in enumerate(legend):
            x = left + i * 190
            out.append(f'<rect x="{x}" y="{y}" width="10" height="10" fill="{series[i]}" rx="2"/>')
            out.append(t(x + 15, y + 9, lab_ru, lab_en, 11.5, MUTED))
        y += 26
    for lab_ru, lab_en, vals in rows:
        y_row = y
        for i, v in enumerate(vals):
            bw = max(v * scale, 0.5)
            out.append(f'<rect x="{left}" y="{y_row}" width="{bw:.1f}" height="{bar}" '
                       f'fill="{series[i]}" rx="2"/>')
            out.append(t(left + bw + 7, y_row + bar * 0.78, fmt.format(v).replace(".", ","),
                         fmt.format(v), 11.5, INK))
            y_row += bar
        out.append(t(left - 10, y + (bar * n_ser) * 0.62, lab_ru, lab_en, 12, INK, anchor="end"))
        y += bar * n_ser + gap
    return _svg("".join(out), w, int(h))


def vbars_grouped(groups, series_colors, legend, w=700, h=250, fmt="{:+.1f}",
                  ylab_ru="", ylab_en="", faded=None):
    """Сгруппированные вертикальные столбцы с нулевой линией. groups: (ru, en, [значения])."""
    faded = faded or set()
    left, right, top, bottom = 46, 16, 34, 46
    vals = [v for _, _, vs in groups for v in vs]
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = (hi - lo) * 0.18 or 1
    lo, hi = lo - pad, hi + pad
    sy = lambda v: top + (hi - v) / (hi - lo) * (h - top - bottom)
    gw = (w - left - right) / len(groups)
    bw = min(gw / (len(series_colors) + 1.2), 54)
    out = [f'<line x1="{left}" x2="{w - right}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" '
           f'stroke="{MUTED}" stroke-width="1"/>']
    for i, (lab_ru, lab_en) in enumerate(legend):
        x = left + i * 180
        out.append(f'<rect x="{x}" y="6" width="10" height="10" fill="{series_colors[i]}" rx="2"/>')
        out.append(t(x + 15, 15, lab_ru, lab_en, 11.5, MUTED))
    for gi, (lab_ru, lab_en, vs) in enumerate(groups):
        cx = left + gw * (gi + 0.5)
        for si, v in enumerate(vs):
            x = cx + (si - (len(vs) - 1) / 2) * (bw + 3) - bw / 2
            y0, y1 = sy(0), sy(v)
            dim = (gi, si) in faded
            op = ' opacity="0.32"' if dim else ''
            out.append(f'<rect x="{x:.1f}" y="{min(y0, y1):.1f}" width="{bw:.1f}" '
                       f'height="{abs(y1 - y0):.1f}" fill="{series_colors[si]}" rx="2"{op}/>')
            ty = min(y0, y1) - 6 if v > 0 else max(y0, y1) + 15
            out.append(t(x + bw / 2, ty, fmt.format(v).replace(".", ","), fmt.format(v),
                         11.5, MUTED if dim else INK, anchor="middle",
                         weight=None if dim else "600"))
        out.append(t(cx, h - 16, lab_ru, lab_en, 13, INK, anchor="middle"))
    return _svg("".join(out), w, h)


def lines_two_panel(x_labels, top_series, bottom_series, w=700, h=300,
                    top_ru="", top_en="", bot_ru="", bot_en=""):
    """Две панели с общей осью времени: сверху несколько рядов, снизу один (ставка).

    Две оси на одном поле не рисуются сознательно: разные величины — разные панели.
    """
    left, right = 44, 96
    h_top, h_bot, gap = int(h * 0.56), int(h * 0.26), 30
    n = len(x_labels)
    sx = lambda i: left + i * (w - left - right) / (n - 1)

    def panel(series, y0, hh, ru, en):
        vals = [v for s in series for v in s["v"]]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.15 or 1
        lo, hi = lo - pad, hi + pad
        sy = lambda v: y0 + (hi - v) / (hi - lo) * hh
        out = [f'<line x1="{left}" x2="{w - right}" y1="{y0 + hh:.1f}" y2="{y0 + hh:.1f}" '
               f'stroke="{LINE}" stroke-width="1"/>']
        for g in (lo + (hi - lo) * 0.18, lo + (hi - lo) * 0.82):
            out.append(f'<line x1="{left}" x2="{w - right}" y1="{sy(g):.1f}" y2="{sy(g):.1f}" '
                       f'stroke="{LINE}" stroke-width="1" stroke-dasharray="2 4"/>')
            lab = f'{g:.0f}' if abs(g) >= 10 else f'{g:.1f}'
            out.append(t(left - 6, sy(g) + 4, lab.replace('.', ','), lab, 11, MUTED, anchor='end'))
        for s in series:
            d = "M" + "L".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(s["v"]))
            out.append(f'<path d="{d}" fill="none" stroke="{s["c"]}" stroke-width="2" '
                       f'stroke-linejoin="round"/>')
            out.append(t(w - right + 7, sy(s["v"][-1]) + 4, s["ru"], s["en"], 11.5, s["c"]))
        out.append(t(left, y0 - 8, ru, en, 11.5, MUTED))
        return "".join(out)

    out = [panel(top_series, 18, h_top, top_ru, top_en),
           panel(bottom_series, 18 + h_top + gap, h_bot, bot_ru, bot_en)]
    for i, lab in enumerate(x_labels):
        if lab:
            out.append(t(sx(i), h - 6, lab, lab, 11.5, MUTED, anchor="middle"))
    return _svg("".join(out), w, h)


def stacked_bar(parts, total_label_ru, total_label_en, w=700, h=96):
    """Одна полоса, разбитая на части: 24,0 = 18,9 + 4,1."""
    left, right, y, bar = 16, 16, 34, 30
    total = sum(p[2] for p in parts)
    scale = (w - left - right) / total
    out, x = [], left
    for ru, en, v, col in parts:
        bw = v * scale
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(bw - 2, 1):.1f}" height="{bar}" '
                   f'fill="{col}" rx="2"/>')
        out.append(t(x + bw / 2 - 1, y + bar + 17, f"{ru} — {v:.1f}".replace(".", ","),
                     f"{en} — {v:.1f}", 11.5, MUTED, anchor="middle"))
        x += bw
    out.append(t(left, y - 10, total_label_ru, total_label_en, 12, INK, weight="600"))
    return _svg("".join(out), w, h)
