"""Загрузка выгрузок СберИндекса из data/raw со сверкой контрольных сумм.

Все наборы скачаны с sberindex.ru 12.10.2026 как csv.zip. Хеши и имена файлов внутри архивов
зафиксированы в data/raw/manifest.json: прогон на другой версии данных это заметит.
Сырые файлы в репозиторий не кладутся, их скачивает scripts/download.ps1.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
MANIFEST = RAW / "manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(raw_dir: Path = RAW, manifest: Path | None = None) -> dict[str, str]:
    man = json.loads(Path(manifest or MANIFEST).read_text(encoding="utf-8"))
    out = {}
    for name, meta in man.items():
        p = Path(raw_dir) / name
        out[name] = "missing" if not p.exists() else ("ok" if sha256(p) == meta["sha256"] else "mismatch")
    return out


def read(dataset: str, raw_dir: Path = RAW) -> pd.DataFrame:
    """Длинная таблица набора: колонки как в выгрузке, period уже datetime."""
    p = Path(raw_dir) / f"si_{dataset}.csv.zip"
    with zipfile.ZipFile(p) as z:
        name = z.namelist()[0]
        df = pd.read_csv(io.BytesIO(z.read(name)), sep=";", quotechar='"')
    df["period"] = pd.to_datetime(df["period"])
    df.attrs["source_file"] = name
    return df


def panel(df: pd.DataFrame, index: str = "ref_area", value: str = "value",
          where: dict[str, str] | None = None) -> pd.DataFrame:
    """Широкая таблица: строки — ряды, столбцы — периоды."""
    sub = df
    for k, v in (where or {}).items():
        sub = sub[sub[k] == v]
    w = sub.pivot_table(index=index, columns="period", values=value)
    return w.sort_index(axis=1)
