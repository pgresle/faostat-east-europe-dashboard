from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

CATALOG_URL = "https://bulks-faostat.fao.org/production/datasets_E.xml"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTFILE = PROCESSED_DIR / "faostat_eastern_europe_long.parquet"

# UN M49 Eastern Europe geoscheme; upravte podle potřeby.
EASTERN_EUROPE = [
    "Belarus",
    "Bulgaria",
    "Czechia",
    "Hungary",
    "Poland",
    "Republic of Moldova",
    "Romania",
    "Russian Federation",
    "Slovakia",
    "Ukraine",
]

DATASETS = {
    "QCL": "Production: Crops and livestock products",
    "RL": "Land, Inputs and Sustainability: Land Use",
    "OA": "Population and Employment: Annual population",
}

# Fallbacks are used only if the XML catalogue changes or is temporarily unavailable.
FALLBACK_URLS = {
    "QCL": [
        "https://bulks-faostat.fao.org/production/Production_Crops_Livestock_E_All_Data_(Normalized).zip",
        "https://fenixservices.fao.org/faostat/static/bulkdownloads/Production_Crops_Livestock_E_All_Data_(Normalized).zip",
        "https://fenixservices.fao.org/faostat/static/bulkdownloads/crop_production_E_All_Data_(Normalized).zip",
    ],
    "RL": [
        "https://bulks-faostat.fao.org/production/Land_Use_E_All_Data_(Normalized).zip",
        "https://bulks-faostat.fao.org/production/Inputs_LandUse_E_All_Data_(Normalized).zip",
        "https://fenixservices.fao.org/faostat/static/bulkdownloads/Inputs_LandUse_E_All_Data_(Normalized).zip",
    ],
    "OA": [
        "https://bulks-faostat.fao.org/production/Annual_Population_E_All_Data_(Normalized).zip",
        "https://fenixservices.fao.org/faostat/static/bulkdownloads/Annual_Population_E_All_Data_(Normalized).zip",
    ],
}


def _first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Nenalezen žádný z očekávaných sloupců: {list(candidates)}")


def find_bulk_url(dataset_code: str) -> str:
    """Find All Data Normalized ZIP URL from the FAOSTAT bulk catalogue."""
    try:
        r = requests.get(CATALOG_URL, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        candidates: list[str] = []

        for ds in root.iter():
            texts = {child.tag.split("}")[-1]: (child.text or "") for child in list(ds)}
            if texts.get("DatasetCode", "").upper() == dataset_code.upper():
                for txt in texts.values():
                    if isinstance(txt, str) and txt.startswith("http") and txt.endswith(".zip"):
                        candidates.append(txt)

        # Robust scoring: prefer English all-data normalized files.
        def score(url: str) -> int:
            u = url.lower()
            return sum([
                "normalized" in u,
                "all_data" in u or "all data" in u,
                "_e_" in u,
                dataset_code.lower() in u,
            ])

        if candidates:
            return sorted(candidates, key=score, reverse=True)[0]
    except Exception as exc:
        print(f"Catalogue lookup failed for {dataset_code}: {exc}")

    # Last resort: try known historical URL patterns.
    for url in FALLBACK_URLS.get(dataset_code, []):
        try:
            head = requests.head(url, timeout=20, allow_redirects=True)
            if head.status_code < 400:
                return url
        except Exception:
            pass
    raise RuntimeError(f"Nelze najít bulk ZIP URL pro dataset {dataset_code}")


def download_zip(dataset_code: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = find_bulk_url(dataset_code)
    safe_name = re.sub(r"[^A-Za-z0-9_.() -]+", "_", url.split("/")[-1])
    target = RAW_DIR / safe_name
    if target.exists() and target.stat().st_size > 0:
        print(f"Using cached {target}")
        return target
    print(f"Downloading {dataset_code}: {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return target


def read_faostat_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        # The main data file is usually the largest CSV in the archive.
        main_csv = max(csvs, key=lambda n: zf.getinfo(n).file_size)
        with zf.open(main_csv) as f:
            return pd.read_csv(f, encoding="latin1", low_memory=False)


def normalize_columns(df: pd.DataFrame, domain_code: str) -> pd.DataFrame:
    area_col = _first_existing_col(df, ["Area", "Area Name"])
    area_code_col = _first_existing_col(df, ["Area Code", "Area Code (M49)", "Area Code (FAO)"])
    item_col = _first_existing_col(df, ["Item", "Item Name"])
    element_col = _first_existing_col(df, ["Element", "Element Name"])
    year_col = _first_existing_col(df, ["Year"])
    unit_col = _first_existing_col(df, ["Unit"])
    value_col = _first_existing_col(df, ["Value"])

    out = df[[area_code_col, area_col, item_col, element_col, year_col, unit_col, value_col]].copy()
    out.columns = ["area_code", "area", "item", "element", "year", "unit", "value"]
    out["domain_code"] = domain_code
    out["domain"] = DATASETS[domain_code]
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[out["area"].isin(EASTERN_EUROPE)]
    out = out[(out["year"] >= 1961) & out["value"].notna()]
    return out


def classify_metric(domain_code: str, element: str, unit: str) -> str | None:
    e = str(element).lower()
    u = str(unit).lower()
    if domain_code == "RL" and ("area" in e or u in {"ha", "1000 ha", "thousand ha"}):
        return "Land use"
    if domain_code == "QCL" and "production" in e and ("tonne" in u or u in {"t", "tonnes"}):
        return "Agricultural production"
    if domain_code == "QCL" and "area harvested" in e:
        return "Harvested area"
    return None


def load_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    # Population domain usually contains item Population-Est. & Proj.; element Total Population - Both sexes.
    p = pop_df.copy()
    p = p[p["value"].notna()]
    # Prefer total population, both sexes, when present.
    mask = p["element"].str.contains("total population|population", case=False, na=False)
    if mask.any():
        p = p[mask]
    # If multiple records per country-year, keep the largest value as total population proxy.
    p = p.sort_values("value").groupby(["area", "year"], as_index=False).tail(1)
    p = p[["area", "year", "value", "unit"]].rename(columns={"value": "population_raw", "unit": "population_unit"})
    p["population_persons"] = p["population_raw"]
    p.loc[p["population_unit"].str.contains("1000|thousand", case=False, na=False), "population_persons"] *= 1000
    return p[["area", "year", "population_persons"]]


def prepare() -> pd.DataFrame:
    frames = {}
    for code in DATASETS:
        zip_path = download_zip(code)
        raw = read_faostat_zip(zip_path)
        frames[code] = normalize_columns(raw, code)

    pop = load_population(frames["OA"])
    data = pd.concat([frames["QCL"], frames["RL"]], ignore_index=True)
    data["metric_group"] = [classify_metric(d, e, u) for d, e, u in zip(data["domain_code"], data["element"], data["unit"])]
    data = data[data["metric_group"].notna()].copy()
    data = data.merge(pop, on=["area", "year"], how="left")
    data["value_per_capita"] = data["value"] / data["population_persons"]

    # A compact key for Streamlit selectors.
    data["series"] = data["item"] + " — " + data["element"] + " [" + data["unit"] + "]"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUTFILE, index=False)
    data.to_csv(PROCESSED_DIR / "faostat_eastern_europe_long.csv.gz", index=False)
    print(f"Saved {len(data):,} rows to {OUTFILE}")
    return data


if __name__ == "__main__":
    prepare()
