from __future__ import annotations

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

# New filename version so Streamlit does not keep using the old file without livestock data.
OUTFILE = PROCESSED_DIR / "faostat_eastern_europe_long_v3.parquet"

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
    try:
        response = requests.get(CATALOG_URL, timeout=60)
        response.raise_for_status()

        root = ET.fromstring(response.content)
        candidates: list[str] = []

        for ds in root.iter():
            texts = {
                child.tag.split("}")[-1]: (child.text or "")
                for child in list(ds)
            }

            if texts.get("DatasetCode", "").upper() == dataset_code.upper():
                for txt in texts.values():
                    if isinstance(txt, str) and txt.startswith("http") and txt.endswith(".zip"):
                        candidates.append(txt)

        def score(url: str) -> int:
            u = url.lower()
            return sum(
                [
                    "normalized" in u,
                    "all_data" in u or "all data" in u,
                    "_e_" in u,
                    dataset_code.lower() in u,
                ]
            )

        if candidates:
            return sorted(candidates, key=score, reverse=True)[0]

    except Exception as exc:
        print(f"Catalogue lookup failed for {dataset_code}: {exc}")

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

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()

        with open(target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return target


def read_faostat_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        csvs = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        main_csv = max(csvs, key=lambda name: zf.getinfo(name).file_size)

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

    out = df[
        [
            area_code_col,
            area_col,
            item_col,
            element_col,
            year_col,
            unit_col,
            value_col,
        ]
    ].copy()

    out.columns = [
        "area_code",
        "area",
        "item",
        "element",
        "year",
        "unit",
        "value",
    ]

    out["domain_code"] = domain_code
    out["domain"] = DATASETS[domain_code]

    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out[out["area"].isin(EASTERN_EUROPE)]
    out = out[(out["year"] >= 1961) & out["value"].notna()]

    return out


def classify_metric(domain_code: str, element: str, unit: str, item: str) -> str | None:
    e = str(element).lower()
    u = str(unit).lower()
    i = str(item).lower()

    if domain_code == "RL":
        if "area" in e or u in {"ha", "1000 ha", "thousand ha"}:
            return "Land use"
        return None

    if domain_code != "QCL":
        return None

    # Crops and crop production.
    if "area harvested" in e:
        return "Harvested area"

    if "production" in e:
        if any(token in u for token in ["t", "tonne", "tonnes"]):
            if any(
                token in i
                for token in [
                    "milk",
                    "meat",
                    "egg",
                    "honey",
                    "wool",
                    "hide",
                    "skin",
                    "silk",
                ]
            ):
                return "Livestock products"

            return "Agricultural production"

        if any(token in u for token in ["no", "1000 no", "head", "1000 head"]):
            if any(token in i for token in ["egg", "hides", "skins"]):
                return "Livestock products"

    # Livestock stocks.
    if any(
        token in e
        for token in [
            "stocks",
            "producing animals",
            "animals live",
            "laying",
            "milking",
        ]
    ):
        if any(token in u for token in ["head", "1000 head", "no", "1000 no"]):
            return "Livestock stocks"

    if "slaughtered animals" in e or "producing or slaughtered animals" in e:
        if any(token in u for token in ["head", "1000 head", "no", "1000 no"]):
            return "Slaughtered animals"

    return None


def load_population(pop_df: pd.DataFrame) -> pd.DataFrame:
    p = pop_df.copy()
    p = p[p["value"].notna()]

    mask = p["element"].str.contains("total population|population", case=False, na=False)

    if mask.any():
        p = p[mask]

    p = p.sort_values("value").groupby(["area", "year"], as_index=False).tail(1)

    p = p[["area", "year", "value", "unit"]].rename(
        columns={
            "value": "population_raw",
            "unit": "population_unit",
        }
    )

    p["population_persons"] = p["population_raw"]

    p.loc[
        p["population_unit"].str.contains("1000|thousand", case=False, na=False),
        "population_persons",
    ] *= 1000

    return p[["area", "year", "population_persons"]]


def prepare(force: bool = False) -> pd.DataFrame:
    if OUTFILE.exists() and not force:
        existing = pd.read_parquet(OUTFILE)

        required = {
            "Agricultural production",
            "Harvested area",
            "Land use",
            "Livestock products",
            "Livestock stocks",
        }

        if required.issubset(set(existing["metric_group"].dropna().unique())):
            return existing

        print("Existing processed file is incomplete. Rebuilding data.")

    frames = {}

    for code in DATASETS:
        zip_path = download_zip(code)
        raw = read_faostat_zip(zip_path)
        frames[code] = normalize_columns(raw, code)

    pop = load_population(frames["OA"])

    data = pd.concat(
        [
            frames["QCL"],
            frames["RL"],
        ],
        ignore_index=True,
    )

    data["metric_group"] = [
        classify_metric(domain_code, element, unit, item)
        for domain_code, element, unit, item in zip(
            data["domain_code"],
            data["element"],
            data["unit"],
            data["item"],
        )
    ]

    data = data[data["metric_group"].notna()].copy()

    data = data.merge(pop, on=["area", "year"], how="left")
    data["value_per_capita"] = data["value"] / data["population_persons"]

    data["series"] = (
        data["item"].astype(str)
        + " — "
        + data["element"].astype(str)
        + " ["
        + data["unit"].astype(str)
        + "]"
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data.to_parquet(OUTFILE, index=False)
    data.to_csv(PROCESSED_DIR / "faostat_eastern_europe_long_v3.csv.gz", index=False)

    print(f"Saved {len(data):,} rows to {OUTFILE}")

    return data


if __name__ == "__main__":
    prepare(force=True)
