from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.download_prepare_faostat import OUTFILE, prepare


st.set_page_config(page_title="FAOSTAT Eastern Europe", layout="wide")

st.title("FAOSTAT: zemědělská produkce, hospodářská zvířata a land use ve východní Evropě")
st.caption(
    "Časové řady od roku 1961 podle dostupnosti v FAOSTAT. "
    "U některých států začínají samostatné řady až po rozpadu dřívějších státních celků."
)


REQUIRED_GROUPS = {
    "Agricultural production",
    "Harvested area",
    "Land use",
    "Livestock products",
    "Livestock stocks",
}


@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def load_data(force: bool = False) -> pd.DataFrame:
    path = Path(OUTFILE)

    if force or not path.exists():
        return prepare(force=True)

    data = pd.read_parquet(path)

    existing_groups = set(data["metric_group"].dropna().unique())

    if not REQUIRED_GROUPS.issubset(existing_groups):
        return prepare(force=True)

    return data


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def clean_item_name(item: str) -> str:
    if pd.isna(item):
        return ""

    name = str(item)

    replacements = {
        "Raw milk of cattle": "Cow milk",
        "Raw milk of buffalo": "Buffalo milk",
        "Raw milk of sheep": "Sheep milk",
        "Raw milk of goats": "Goat milk",
        "Meat of cattle with the bone, fresh or chilled": "Beef",
        "Meat of pig with the bone, fresh or chilled": "Pig meat",
        "Meat of chickens, fresh or chilled": "Chicken meat",
        "Meat of sheep, fresh or chilled": "Sheep meat",
        "Meat of goat, fresh or chilled": "Goat meat",
        "Hen eggs in shell, fresh": "Hen eggs",
        "Eggs from other birds in shell, fresh, n.e.c.": "Other eggs",
        "Natural honey": "Honey",
        "Cattle": "Cattle",
        "Pigs": "Pigs",
        "Sheep": "Sheep",
        "Goats": "Goats",
        "Chickens": "Chickens",
        "Buffalo": "Buffalo",
        "Horses": "Horses",
        "Asses": "Asses",
        "Mules and hinnies": "Mules and hinnies",
        "Ducks": "Ducks",
        "Geese": "Geese",
        "Turkeys": "Turkeys",
    }

    return replacements.get(name, name)


def add_display_series(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    out["item_display"] = out["item"].apply(clean_item_name)
    out["display_series"] = out["item_display"]

    duplicates = (
        out.groupby("item_display")["element"]
        .nunique()
        .reset_index(name="n_elements")
    )

    ambiguous_items = set(
        duplicates.loc[duplicates["n_elements"] > 1, "item_display"]
    )

    mask = out["item_display"].isin(ambiguous_items)

    out.loc[mask, "display_series"] = (
        out.loc[mask, "item_display"].astype(str)
        + " — "
        + out.loc[mask, "element"].astype(str)
    )

    return out


def make_safe_filename(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def to_tonnes(value: pd.Series, unit: pd.Series) -> pd.Series:
    out = value.astype(float).copy()
    u = unit.astype(str).str.lower()

    out.loc[u.str.contains("1000", na=False)] *= 1000

    return out


def to_heads(value: pd.Series, unit: pd.Series) -> pd.Series:
    out = value.astype(float).copy()
    u = unit.astype(str).str.lower()

    out.loc[u.str.contains("1000", na=False)] *= 1000

    return out


def to_hectares(value: pd.Series, unit: pd.Series) -> pd.Series:
    out = value.astype(float).copy()
    u = unit.astype(str).str.lower()

    out.loc[u.str.contains("1000", na=False)] *= 1000

    return out


def prepare_map_indicators(data: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """
    Build cumulative choropleth indicators per agricultural land area.

    Indicators:
    - cereals production per agricultural land area [t/ha]
    - pulses production per agricultural land area [t/ha]
    - pigs per agricultural land area [heads/ha]
    - cattle per agricultural land area [heads/ha]
    """

    d = data[data["area"].isin(countries)].copy()

    # Agricultural land denominator.
    land = d[
        (d["metric_group"] == "Land use")
        & (d["item"].astype(str).str.contains("Agricultural land", case=False, na=False))
    ].copy()

    land = land[
        land["element"].astype(str).str.contains("area", case=False, na=False)
    ].copy()

    land["agri_land_ha"] = to_hectares(land["value"], land["unit"])

    land = (
        land.groupby(["area", "year"], as_index=False)["agri_land_ha"]
        .sum()
    )

    indicators = []

    # Cereals production.
    cereals = d[
        (d["metric_group"] == "Agricultural production")
        & (
            d["item"].astype(str).str.contains("Cereals", case=False, na=False)
            | d["item"].astype(str).str.contains("Wheat|Barley|Rye|Oats|Maize|Rice|Sorghum|Millet", case=False, na=False)
        )
        & d["element"].astype(str).str.contains("Production", case=False, na=False)
    ].copy()

    cereals["numerator"] = to_tonnes(cereals["value"], cereals["unit"])

    cereals = (
        cereals.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    cereals["indicator"] = "Cereals production"
    cereals["unit"] = "t/ha agricultural land"
    indicators.append(cereals)

    # Pulses production.
    pulses = d[
        (d["metric_group"] == "Agricultural production")
        & (
            d["item"].astype(str).str.contains("Pulses", case=False, na=False)
            | d["item"].astype(str).str.contains("Beans|Peas|Lentils|Chick peas|Cow peas|Vetches|Lupins", case=False, na=False)
        )
        & d["element"].astype(str).str.contains("Production", case=False, na=False)
    ].copy()

    pulses["numerator"] = to_tonnes(pulses["value"], pulses["unit"])

    pulses = (
        pulses.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    pulses["indicator"] = "Pulses production"
    pulses["unit"] = "t/ha agricultural land"
    indicators.append(pulses)

    # Pigs.
    pigs = d[
        (d["metric_group"] == "Livestock stocks")
        & (d["item"].astype(str).str.fullmatch("Pigs", case=False, na=False))
    ].copy()

    pigs["numerator"] = to_heads(pigs["value"], pigs["unit"])

    pigs = (
        pigs.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    pigs["indicator"] = "Pigs"
    pigs["unit"] = "heads/ha agricultural land"
    indicators.append(pigs)

    # Cattle / cows.
    cattle = d[
        (d["metric_group"] == "Livestock stocks")
        & (
            d["item"].astype(str).str.fullmatch("Cattle", case=False, na=False)
            | d["item"].astype(str).str.fullmatch("Cows", case=False, na=False)
        )
    ].copy()

    cattle["numerator"] = to_heads(cattle["value"], cattle["unit"])

    cattle = (
        cattle.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    cattle["indicator"] = "Cattle"
    cattle["unit"] = "heads/ha agricultural land"
    indicators.append(cattle)

    numerators = pd.concat(indicators, ignore_index=True)

    out = numerators.merge(land, on=["area", "year"], how="left")

    out = out[
        out["agri_land_ha"].notna()
        & (out["agri_land_ha"] > 0)
    ].copy()

    out["value_per_agri_land"] = out["numerator"] / out["agri_land_ha"]

    return out


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

with st.sidebar:
    refresh_data = st.button("Obnovit / znovu stáhnout FAOSTAT data")

with st.spinner("Načítám data. Při prvním spuštění se stáhnou bulk soubory z FAOSTAT."):
    df = load_data(force=refresh_data)


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Nastavení grafů")

    metric_group = st.selectbox(
        "Datová oblast",
        sorted(df["metric_group"].dropna().unique()),
        index=0,
    )

    available_units = sorted(
        df.loc[df["metric_group"] == metric_group, "unit"]
        .dropna()
        .unique()
    )

    selected_unit = st.selectbox(
        "Jednotka FAOSTAT",
        available_units,
        index=0,
    )

    norm = st.radio(
        "Vyjádření",
        ["Absolutně", "Na osobu"],
        horizontal=True,
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())

    years = st.slider(
        "Roky",
        year_min,
        year_max,
        (1961, year_max),
    )

    countries = st.multiselect(
        "Země",
        sorted(df["area"].dropna().unique()),
        default=sorted(df["area"].dropna().unique()),
    )

    top_n = st.slider(
        "Kolik největších položek zobrazit v každé zemi",
        3,
        20,
        8,
    )

    chart_type = st.radio(
        "Typ grafu",
        ["Plošný kumulativní", "Čárový"],
        horizontal=False,
    )

    n_cols = st.slider(
        "Počet grafů v řádku",
        1,
        4,
        3,
    )

    chart_height = st.slider(
        "Výška grafu",
        220,
        500,
        280,
    )


# -------------------------------------------------------------------
# Filter data for country charts
# -------------------------------------------------------------------

base = df[
    (df["metric_group"] == metric_group)
    & (df["unit"] == selected_unit)
    & (df["area"].isin(countries))
].copy()

base = base[
    (base["year"] >= years[0])
    & (base["year"] <= years[1])
].copy()

base = add_display_series(base)

value_col = "value" if norm == "Absolutně" else "value_per_capita"

if norm == "Absolutně":
    y_label = f"hodnota [{selected_unit}]"
else:
    y_label = f"hodnota na osobu [{selected_unit}/osobu]"


st.subheader(f"{metric_group} — {norm.lower()} — jednotka: {selected_unit}")

if base.empty:
    st.warning("Pro zadané filtry nejsou dostupná data.")
    st.stop()


# -------------------------------------------------------------------
# Summary panel
# -------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Země", base["area"].nunique())
c2.metric("Roky", f"{int(base['year'].min())}–{int(base['year'].max())}")
c3.metric("Řádků dat", f"{len(base):,}".replace(",", " "))
c4.metric("Jednotka", selected_unit)


selected_countries = [
    country for country in countries
    if not base[base["area"] == country].empty
]


# -------------------------------------------------------------------
# Shared legend preparation
# -------------------------------------------------------------------

visible_series = set()
other_needed = False

for country in selected_countries:
    d_country = base[base["area"] == country].copy()

    if d_country.empty:
        continue

    latest_year = d_country.groupby("display_series")["year"].max().rename("latest_year")
    latest = d_country.merge(latest_year, on="display_series")
    latest = latest[latest["year"] == latest["latest_year"]]

    country_top_series = (
        latest.groupby("display_series", as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(top_n)["display_series"]
        .tolist()
    )

    visible_series.update(country_top_series)

    if d_country["display_series"].nunique() > len(country_top_series):
        other_needed = True


series_order = (
    base[base["display_series"].isin(visible_series)]
    .groupby("display_series", as_index=False)[value_col]
    .sum()
    .sort_values(value_col, ascending=False)["display_series"]
    .tolist()
)

if other_needed and "Other" not in series_order:
    series_order.append("Other")


palette = [
    "#2E7D32",
    "#66BB6A",
    "#A5D6A7",
    "#FBC02D",
    "#F9A825",
    "#8D6E63",
    "#A1887F",
    "#D84315",
    "#E57373",
    "#1565C0",
    "#64B5F6",
    "#00695C",
    "#9E9D24",
    "#C0CA33",
    "#5D4037",
    "#BCAAA4",
    "#AD1457",
    "#283593",
    "#4FC3F7",
    "#78909C",
]

color_map = {
    series: palette[idx % len(palette)]
    for idx, series in enumerate(series_order)
}


# -------------------------------------------------------------------
# Sticky shared legend
# -------------------------------------------------------------------

legend_html = """
<style>
.sticky-legend {
    position: sticky;
    top: 0;
    z-index: 999;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.sticky-legend-title {
    font-weight: 700;
    margin-bottom: 6px;
}
.sticky-legend-items {
    display: flex;
    flex-wrap: wrap;
    gap: 7px 13px;
    font-size: 12px;
    line-height: 1.25;
}
.sticky-legend-item {
    white-space: nowrap;
}
.legend-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}
.unit-note {
    font-size: 12px;
    color: #6b7280;
    margin-top: 6px;
}
</style>

<div class="sticky-legend">
  <div class="sticky-legend-title">Společná legenda pro grafy podle zemí</div>
  <div class="sticky-legend-items">
"""

for item in series_order:
    color = color_map.get(item, "#999999")
    legend_html += (
        f'<div class="sticky-legend-item">'
        f'<span class="legend-dot" style="background:{color};"></span>{item}'
        f'</div>'
    )

legend_html += f"""
  </div>
  <div class="unit-note">Jednotka grafů: {selected_unit}</div>
</div>
"""


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------

tab_charts, tab_map = st.tabs(
    [
        "Grafy podle zemí",
        "Kartogramy: produkce a zvířata na zemědělskou půdu",
    ]
)


# -------------------------------------------------------------------
# Tab 1: Country chart grid
# -------------------------------------------------------------------

with tab_charts:
    st.markdown(legend_html, unsafe_allow_html=True)

    for i in range(0, len(selected_countries), n_cols):
        cols = st.columns(n_cols)

        for j, country in enumerate(selected_countries[i:i + n_cols]):
            d = base[base["area"] == country].copy()

            if d.empty:
                continue

            latest_year = d.groupby("display_series")["year"].max().rename("latest_year")
            latest = d.merge(latest_year, on="display_series")
            latest = latest[latest["year"] == latest["latest_year"]]

            top_series = (
                latest.groupby("display_series", as_index=False)[value_col]
                .sum()
                .sort_values(value_col, ascending=False)
                .head(top_n)["display_series"]
                .tolist()
            )

            d["series_plot"] = d["display_series"].where(
                d["display_series"].isin(top_series),
                "Other",
            )

            plot = (
                d.groupby(["year", "series_plot"], as_index=False)[value_col]
                .sum()
            )

            plot["series_plot"] = pd.Categorical(
                plot["series_plot"],
                categories=series_order,
                ordered=True,
            )

            plot = plot.sort_values(["series_plot", "year"])

            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"#### {country}")

                    if chart_type == "Plošný kumulativní":
                        fig = px.area(
                            plot,
                            x="year",
                            y=value_col,
                            color="series_plot",
                            color_discrete_map=color_map,
                            category_orders={"series_plot": series_order},
                            labels={
                                "year": "rok",
                                value_col: y_label,
                                "series_plot": "položka",
                            },
                        )
                    else:
                        fig = px.line(
                            plot,
                            x="year",
                            y=value_col,
                            color="series_plot",
                            color_discrete_map=color_map,
                            category_orders={"series_plot": series_order},
                            labels={
                                "year": "rok",
                                value_col: y_label,
                                "series_plot": "položka",
                            },
                        )

                    fig.update_layout(
                        height=chart_height,
                        showlegend=False,
                        margin=dict(l=5, r=5, t=25, b=5),
                        font=dict(size=10),
                    )

                    fig.update_xaxes(
                        title_text="rok",
                        title_font=dict(size=10),
                        tickfont=dict(size=9),
                    )

                    fig.update_yaxes(
                        title_text=y_label,
                        title_font=dict(size=10),
                        tickfont=dict(size=9),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Data"):
                        st.dataframe(
                            plot,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "CSV",
                            data=plot.to_csv(index=False).encode("utf-8"),
                            file_name=(
                                f"faostat_"
                                f"{make_safe_filename(country)}_"
                                f"{make_safe_filename(metric_group)}_"
                                f"{make_safe_filename(selected_unit)}.csv"
                            ),
                            mime="text/csv",
                        )


# -------------------------------------------------------------------
# Tab 2: cumulative choropleth indicators per agricultural land
# -------------------------------------------------------------------

with tab_map:
    st.markdown("### Kartogramy: kumulativní hodnoty na plochu zemědělské půdy")

    st.caption(
        "Kartogramy počítají agregované indikátory pro vybraný rok. "
        "Cereals a pulses jsou vyjádřeny jako tuny produkce na hektar zemědělské půdy. "
        "Pigs a cattle jsou vyjádřeny jako počet kusů na hektar zemědělské půdy."
    )

    map_indicators = prepare_map_indicators(df, countries)

    map_indicators = map_indicators[
        (map_indicators["year"] >= years[0])
        & (map_indicators["year"] <= years[1])
    ].copy()

    if map_indicators.empty:
        st.warning("Pro zadané země a roky nejsou dostupná data pro kartogramy.")
        st.stop()

    map_c1, map_c2 = st.columns([1, 1])

    with map_c1:
        map_indicator = st.selectbox(
            "Indikátor",
            [
                "Cereals production",
                "Pulses production",
                "Pigs",
                "Cattle",
            ],
            index=0,
        )

    available_map_years = sorted(
        map_indicators.loc[
            map_indicators["indicator"] == map_indicator,
            "year",
        ].dropna().unique()
    )

    with map_c2:
        map_year = st.selectbox(
            "Rok",
            available_map_years,
            index=len(available_map_years) - 1,
        )

    map_data = map_indicators[
        (map_indicators["indicator"] == map_indicator)
        & (map_indicators["year"] == map_year)
    ].copy()

    iso3_map = {
        "Albania": "ALB",
        "Belarus": "BLR",
        "Bosnia and Herzegovina": "BIH",
        "Bulgaria": "BGR",
        "Croatia": "HRV",
        "Czechia": "CZE",
        "Czech Republic": "CZE",
        "Estonia": "EST",
        "Hungary": "HUN",
        "Latvia": "LVA",
        "Lithuania": "LTU",
        "Montenegro": "MNE",
        "North Macedonia": "MKD",
        "Poland": "POL",
        "Republic of Moldova": "MDA",
        "Moldova": "MDA",
        "Romania": "ROU",
        "Russian Federation": "RUS",
        "Russia": "RUS",
        "Serbia": "SRB",
        "Slovakia": "SVK",
        "Slovenia": "SVN",
        "Ukraine": "UKR",
    }

    map_data["iso3"] = map_data["area"].map(iso3_map)

    missing_iso = (
        map_data[map_data["iso3"].isna()]["area"]
        .dropna()
        .unique()
        .tolist()
    )

    map_data = map_data.dropna(subset=["iso3"])

    if map_data.empty:
        st.warning("Pro vybraný indikátor a rok nejsou dostupná mapová data.")
    else:
        map_unit = map_data["unit"].iloc[0]

        fig_map = px.choropleth(
            map_data,
            locations="iso3",
            color="value_per_agri_land",
            hover_name="area",
            hover_data={
                "iso3": False,
                "numerator": ":,.0f",
                "agri_land_ha": ":,.0f",
                "value_per_agri_land": ":,.4f",
            },
            color_continuous_scale=[
                "#FFF7BC",
                "#FEC44F",
                "#FE9929",
                "#D95F0E",
                "#993404",
            ],
            labels={
                "value_per_agri_land": map_unit,
                "numerator": "čitatel",
                "agri_land_ha": "zemědělská půda [ha]",
            },
            title=f"{map_indicator}, {map_year} — {map_unit}",
            scope="europe",
        )

        fig_map.update_geos(
            showcountries=True,
            showcoastlines=True,
            showland=True,
            fitbounds="locations",
        )

        fig_map.update_layout(
            height=650,
            margin=dict(l=0, r=0, t=55, b=0),
            coloraxis_colorbar=dict(
                title=map_unit,
            ),
        )

        st.plotly_chart(fig_map, use_container_width=True)

        with st.expander("Data použitá pro kartogram"):
            st.dataframe(
                map_data.sort_values("value_per_agri_land", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Stáhnout mapová data jako CSV",
                data=map_data.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"faostat_map_"
                    f"{make_safe_filename(map_indicator)}_"
                    f"{map_year}.csv"
                ),
                mime="text/csv",
            )

        if missing_iso:
            st.info(
                "Tyto země nebyly zobrazeny, protože pro ně chybí ISO3 mapování: "
                + ", ".join(missing_iso)
            )

    if force or not path.exists():
        return prepare(force=True)

    data = pd.read_parquet(path)

    existing_groups = set(data["metric_group"].dropna().unique())

    if not REQUIRED_GROUPS.issubset(existing_groups):
        return prepare(force=True)

    return data


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def clean_item_name(item: str) -> str:
    if pd.isna(item):
        return ""

    name = str(item)

    replacements = {
        "Raw milk of cattle": "Cow milk",
        "Raw milk of buffalo": "Buffalo milk",
        "Raw milk of sheep": "Sheep milk",
        "Raw milk of goats": "Goat milk",
        "Meat of cattle with the bone, fresh or chilled": "Beef",
        "Meat of pig with the bone, fresh or chilled": "Pig meat",
        "Meat of chickens, fresh or chilled": "Chicken meat",
        "Meat of sheep, fresh or chilled": "Sheep meat",
        "Meat of goat, fresh or chilled": "Goat meat",
        "Hen eggs in shell, fresh": "Hen eggs",
        "Eggs from other birds in shell, fresh, n.e.c.": "Other eggs",
        "Natural honey": "Honey",
        "Cattle": "Cattle",
        "Pigs": "Pigs",
        "Sheep": "Sheep",
        "Goats": "Goats",
        "Chickens": "Chickens",
        "Buffalo": "Buffalo",
        "Horses": "Horses",
        "Asses": "Asses",
        "Mules and hinnies": "Mules and hinnies",
        "Ducks": "Ducks",
        "Geese": "Geese",
        "Turkeys": "Turkeys",
    }

    return replacements.get(name, name)


def add_display_series(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()

    out["item_display"] = out["item"].apply(clean_item_name)
    out["display_series"] = out["item_display"]

    duplicates = (
        out.groupby("item_display")["element"]
        .nunique()
        .reset_index(name="n_elements")
    )

    ambiguous_items = set(
        duplicates.loc[duplicates["n_elements"] > 1, "item_display"]
    )

    mask = out["item_display"].isin(ambiguous_items)

    out.loc[mask, "display_series"] = (
        out.loc[mask, "item_display"].astype(str)
        + " — "
        + out.loc[mask, "element"].astype(str)
    )

    return out


def make_safe_filename(text: str) -> str:
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def to_tonnes(value: pd.Series, unit: pd.Series) -> pd.Series:
    out = value.astype(float).copy()
    u = unit.astype(str).str.lower()

    out.loc[u.str.contains("1000", na=False)] *= 1000

    return out


def to_heads(value: pd.Series, unit: pd.Series) -> pd.Series:
    out = value.astype(float).copy()
    u = unit.astype(str).str.lower()

    out.loc[u.str.contains("1000", na=False)] *= 1000

    return out


def to_hectares(value: pd.Series, unit: pd.Series) -> pd.Series:
    out = value.astype(float).copy()
    u = unit.astype(str).str.lower()

    out.loc[u.str.contains("1000", na=False)] *= 1000

    return out


def prepare_map_indicators(data: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    """
    Build cumulative choropleth indicators per agricultural land area.

    Indicators:
    - cereals production per agricultural land area [t/ha]
    - pulses production per agricultural land area [t/ha]
    - pigs per agricultural land area [heads/ha]
    - cattle per agricultural land area [heads/ha]
    """

    d = data[data["area"].isin(countries)].copy()

    # Agricultural land denominator.
    land = d[
        (d["metric_group"] == "Land use")
        & (d["item"].astype(str).str.contains("Agricultural land", case=False, na=False))
    ].copy()

    land = land[
        land["element"].astype(str).str.contains("area", case=False, na=False)
    ].copy()

    land["agri_land_ha"] = to_hectares(land["value"], land["unit"])

    land = (
        land.groupby(["area", "year"], as_index=False)["agri_land_ha"]
        .sum()
    )

    indicators = []

    # Cereals production.
    cereals = d[
        (d["metric_group"] == "Agricultural production")
        & (
            d["item"].astype(str).str.contains("Cereals", case=False, na=False)
            | d["item"].astype(str).str.contains("Wheat|Barley|Rye|Oats|Maize|Rice|Sorghum|Millet", case=False, na=False)
        )
        & d["element"].astype(str).str.contains("Production", case=False, na=False)
    ].copy()

    cereals["numerator"] = to_tonnes(cereals["value"], cereals["unit"])

    cereals = (
        cereals.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    cereals["indicator"] = "Cereals production"
    cereals["unit"] = "t/ha agricultural land"
    indicators.append(cereals)

    # Pulses production.
    pulses = d[
        (d["metric_group"] == "Agricultural production")
        & (
            d["item"].astype(str).str.contains("Pulses", case=False, na=False)
            | d["item"].astype(str).str.contains("Beans|Peas|Lentils|Chick peas|Cow peas|Vetches|Lupins", case=False, na=False)
        )
        & d["element"].astype(str).str.contains("Production", case=False, na=False)
    ].copy()

    pulses["numerator"] = to_tonnes(pulses["value"], pulses["unit"])

    pulses = (
        pulses.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    pulses["indicator"] = "Pulses production"
    pulses["unit"] = "t/ha agricultural land"
    indicators.append(pulses)

    # Pigs.
    pigs = d[
        (d["metric_group"] == "Livestock stocks")
        & (d["item"].astype(str).str.fullmatch("Pigs", case=False, na=False))
    ].copy()

    pigs["numerator"] = to_heads(pigs["value"], pigs["unit"])

    pigs = (
        pigs.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    pigs["indicator"] = "Pigs"
    pigs["unit"] = "heads/ha agricultural land"
    indicators.append(pigs)

    # Cattle / cows.
    cattle = d[
        (d["metric_group"] == "Livestock stocks")
        & (
            d["item"].astype(str).str.fullmatch("Cattle", case=False, na=False)
            | d["item"].astype(str).str.fullmatch("Cows", case=False, na=False)
        )
    ].copy()

    cattle["numerator"] = to_heads(cattle["value"], cattle["unit"])

    cattle = (
        cattle.groupby(["area", "year"], as_index=False)["numerator"]
        .sum()
    )

    cattle["indicator"] = "Cattle"
    cattle["unit"] = "heads/ha agricultural land"
    indicators.append(cattle)

    numerators = pd.concat(indicators, ignore_index=True)

    out = numerators.merge(land, on=["area", "year"], how="left")

    out = out[
        out["agri_land_ha"].notna()
        & (out["agri_land_ha"] > 0)
    ].copy()

    out["value_per_agri_land"] = out["numerator"] / out["agri_land_ha"]

    return out


# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------

with st.sidebar:
    refresh_data = st.button("Obnovit / znovu stáhnout FAOSTAT data")

with st.spinner("Načítám data. Při prvním spuštění se stáhnou bulk soubory z FAOSTAT."):
    df = load_data(force=refresh_data)


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Nastavení grafů")

    metric_group = st.selectbox(
        "Datová oblast",
        sorted(df["metric_group"].dropna().unique()),
        index=0,
    )

    available_units = sorted(
        df.loc[df["metric_group"] == metric_group, "unit"]
        .dropna()
        .unique()
    )

    selected_unit = st.selectbox(
        "Jednotka FAOSTAT",
        available_units,
        index=0,
    )

    norm = st.radio(
        "Vyjádření",
        ["Absolutně", "Na osobu"],
        horizontal=True,
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())

    years = st.slider(
        "Roky",
        year_min,
        year_max,
        (1961, year_max),
    )

    countries = st.multiselect(
        "Země",
        sorted(df["area"].dropna().unique()),
        default=sorted(df["area"].dropna().unique()),
    )

    top_n = st.slider(
        "Kolik největších položek zobrazit v každé zemi",
        3,
        20,
        8,
    )

    chart_type = st.radio(
        "Typ grafu",
        ["Plošný kumulativní", "Čárový"],
        horizontal=False,
    )

    n_cols = st.slider(
        "Počet grafů v řádku",
        1,
        4,
        3,
    )

    chart_height = st.slider(
        "Výška grafu",
        220,
        500,
        280,
    )


# -------------------------------------------------------------------
# Filter data for country charts
# -------------------------------------------------------------------

base = df[
    (df["metric_group"] == metric_group)
    & (df["unit"] == selected_unit)
    & (df["area"].isin(countries))
].copy()

base = base[
    (base["year"] >= years[0])
    & (base["year"] <= years[1])
].copy()

base = add_display_series(base)

value_col = "value" if norm == "Absolutně" else "value_per_capita"

if norm == "Absolutně":
    y_label = f"hodnota [{selected_unit}]"
else:
    y_label = f"hodnota na osobu [{selected_unit}/osobu]"


st.subheader(f"{metric_group} — {norm.lower()} — jednotka: {selected_unit}")

if base.empty:
    st.warning("Pro zadané filtry nejsou dostupná data.")
    st.stop()


# -------------------------------------------------------------------
# Summary panel
# -------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)

c1.metric("Země", base["area"].nunique())
c2.metric("Roky", f"{int(base['year'].min())}–{int(base['year'].max())}")
c3.metric("Řádků dat", f"{len(base):,}".replace(",", " "))
c4.metric("Jednotka", selected_unit)


selected_countries = [
    country for country in countries
    if not base[base["area"] == country].empty
]


# -------------------------------------------------------------------
# Shared legend preparation
# -------------------------------------------------------------------

visible_series = set()
other_needed = False

for country in selected_countries:
    d_country = base[base["area"] == country].copy()

    if d_country.empty:
        continue

    latest_year = d_country.groupby("display_series")["year"].max().rename("latest_year")
    latest = d_country.merge(latest_year, on="display_series")
    latest = latest[latest["year"] == latest["latest_year"]]

    country_top_series = (
        latest.groupby("display_series", as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(top_n)["display_series"]
        .tolist()
    )

    visible_series.update(country_top_series)

    if d_country["display_series"].nunique() > len(country_top_series):
        other_needed = True


series_order = (
    base[base["display_series"].isin(visible_series)]
    .groupby("display_series", as_index=False)[value_col]
    .sum()
    .sort_values(value_col, ascending=False)["display_series"]
    .tolist()
)

if other_needed and "Other" not in series_order:
    series_order.append("Other")


palette = [
    "#2E7D32",
    "#66BB6A",
    "#A5D6A7",
    "#FBC02D",
    "#F9A825",
    "#8D6E63",
    "#A1887F",
    "#D84315",
    "#E57373",
    "#1565C0",
    "#64B5F6",
    "#00695C",
    "#9E9D24",
    "#C0CA33",
    "#5D4037",
    "#BCAAA4",
    "#AD1457",
    "#283593",
    "#4FC3F7",
    "#78909C",
]

color_map = {
    series: palette[idx % len(palette)]
    for idx, series in enumerate(series_order)
}


# -------------------------------------------------------------------
# Sticky shared legend
# -------------------------------------------------------------------

legend_html = """
<style>
.sticky-legend {
    position: sticky;
    top: 0;
    z-index: 999;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.sticky-legend-title {
    font-weight: 700;
    margin-bottom: 6px;
}
.sticky-legend-items {
    display: flex;
    flex-wrap: wrap;
    gap: 7px 13px;
    font-size: 12px;
    line-height: 1.25;
}
.sticky-legend-item {
    white-space: nowrap;
}
.legend-dot {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}
.unit-note {
    font-size: 12px;
    color: #6b7280;
    margin-top: 6px;
}
</style>

<div class="sticky-legend">
  <div class="sticky-legend-title">Společná legenda pro grafy podle zemí</div>
  <div class="sticky-legend-items">
"""

for item in series_order:
    color = color_map.get(item, "#999999")
    legend_html += (
        f'<div class="sticky-legend-item">'
        f'<span class="legend-dot" style="background:{color};"></span>{item}'
        f'</div>'
    )

legend_html += f"""
  </div>
  <div class="unit-note">Jednotka grafů: {selected_unit}</div>
</div>
"""


# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------

tab_charts, tab_map = st.tabs(
    [
        "Grafy podle zemí",
        "Kartogramy: produkce a zvířata na zemědělskou půdu",
    ]
)


# -------------------------------------------------------------------
# Tab 1: Country chart grid
# -------------------------------------------------------------------

with tab_charts:
    st.markdown(legend_html, unsafe_allow_html=True)

    for i in range(0, len(selected_countries), n_cols):
        cols = st.columns(n_cols)

        for j, country in enumerate(selected_countries[i:i + n_cols]):
            d = base[base["area"] == country].copy()

            if d.empty:
                continue

            latest_year = d.groupby("display_series")["year"].max().rename("latest_year")
            latest = d.merge(latest_year, on="display_series")
            latest = latest[latest["year"] == latest["latest_year"]]

            top_series = (
                latest.groupby("display_series", as_index=False)[value_col]
                .sum()
                .sort_values(value_col, ascending=False)
                .head(top_n)["display_series"]
                .tolist()
            )

            d["series_plot"] = d["display_series"].where(
                d["display_series"].isin(top_series),
                "Other",
            )

            plot = (
                d.groupby(["year", "series_plot"], as_index=False)[value_col]
                .sum()
            )

            plot["series_plot"] = pd.Categorical(
                plot["series_plot"],
                categories=series_order,
                ordered=True,
            )

            plot = plot.sort_values(["series_plot", "year"])

            with cols[j]:
                with st.container(border=True):
                    st.markdown(f"#### {country}")

                    if chart_type == "Plošný kumulativní":
                        fig = px.area(
                            plot,
                            x="year",
                            y=value_col,
                            color="series_plot",
                            color_discrete_map=color_map,
                            category_orders={"series_plot": series_order},
                            labels={
                                "year": "rok",
                                value_col: y_label,
                                "series_plot": "položka",
                            },
                        )
                    else:
                        fig = px.line(
                            plot,
                            x="year",
                            y=value_col,
                            color="series_plot",
                            color_discrete_map=color_map,
                            category_orders={"series_plot": series_order},
                            labels={
                                "year": "rok",
                                value_col: y_label,
                                "series_plot": "položka",
                            },
                        )

                    fig.update_layout(
                        height=chart_height,
                        showlegend=False,
                        margin=dict(l=5, r=5, t=25, b=5),
                        font=dict(size=10),
                    )

                    fig.update_xaxes(
                        title_text="rok",
                        title_font=dict(size=10),
                        tickfont=dict(size=9),
                    )

                    fig.update_yaxes(
                        title_text=y_label,
                        title_font=dict(size=10),
                        tickfont=dict(size=9),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    with st.expander("Data"):
                        st.dataframe(
                            plot,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "CSV",
                            data=plot.to_csv(index=False).encode("utf-8"),
                            file_name=(
                                f"faostat_"
                                f"{make_safe_filename(country)}_"
                                f"{make_safe_filename(metric_group)}_"
                                f"{make_safe_filename(selected_unit)}.csv"
                            ),
                            mime="text/csv",
                        )


# -------------------------------------------------------------------
# Tab 2: cumulative choropleth indicators per agricultural land
# -------------------------------------------------------------------

with tab_map:
    st.markdown("### Kartogramy: kumulativní hodnoty na plochu zemědělské půdy")

    st.caption(
        "Kartogramy počítají agregované indikátory pro vybraný rok. "
        "Cereals a pulses jsou vyjádřeny jako tuny produkce na hektar zemědělské půdy. "
        "Pigs a cattle jsou vyjádřeny jako počet kusů na hektar zemědělské půdy."
    )

    map_indicators = prepare_map_indicators(df, countries)

    map_indicators = map_indicators[
        (map_indicators["year"] >= years[0])
        & (map_indicators["year"] <= years[1])
    ].copy()

    if map_indicators.empty:
        st.warning("Pro zadané země a roky nejsou dostupná data pro kartogramy.")
        st.stop()

    map_c1, map_c2 = st.columns([1, 1])

    with map_c1:
        map_indicator = st.selectbox(
            "Indikátor",
            [
                "Cereals production",
                "Pulses production",
                "Pigs",
                "Cattle",
            ],
            index=0,
        )

    available_map_years = sorted(
        map_indicators.loc[
            map_indicators["indicator"] == map_indicator,
            "year",
        ].dropna().unique()
    )

    with map_c2:
        map_year = st.selectbox(
            "Rok",
            available_map_years,
            index=len(available_map_years) - 1,
        )

    map_data = map_indicators[
        (map_indicators["indicator"] == map_indicator)
        & (map_indicators["year"] == map_year)
    ].copy()

    iso3_map = {
        "Albania": "ALB",
        "Belarus": "BLR",
        "Bosnia and Herzegovina": "BIH",
        "Bulgaria": "BGR",
        "Croatia": "HRV",
        "Czechia": "CZE",
        "Czech Republic": "CZE",
        "Estonia": "EST",
        "Hungary": "HUN",
        "Latvia": "LVA",
        "Lithuania": "LTU",
        "Montenegro": "MNE",
        "North Macedonia": "MKD",
        "Poland": "POL",
        "Republic of Moldova": "MDA",
        "Moldova": "MDA",
        "Romania": "ROU",
        "Russian Federation": "RUS",
        "Russia": "RUS",
        "Serbia": "SRB",
        "Slovakia": "SVK",
        "Slovenia": "SVN",
        "Ukraine": "UKR",
    }

    map_data["iso3"] = map_data["area"].map(iso3_map)

    missing_iso = (
        map_data[map_data["iso3"].isna()]["area"]
        .dropna()
        .unique()
        .tolist()
    )

    map_data = map_data.dropna(subset=["iso3"])

    if map_data.empty:
        st.warning("Pro vybraný indikátor a rok nejsou dostupná mapová data.")
    else:
        map_unit = map_data["unit"].iloc[0]

        fig_map = px.choropleth(
            map_data,
            locations="iso3",
            color="value_per_agri_land",
            hover_name="area",
            hover_data={
                "iso3": False,
                "numerator": ":,.0f",
                "agri_land_ha": ":,.0f",
                "value_per_agri_land": ":,.4f",
            },
            color_continuous_scale=[
                "#FFF7BC",
                "#FEC44F",
                "#FE9929",
                "#D95F0E",
                "#993404",
            ],
            labels={
                "value_per_agri_land": map_unit,
                "numerator": "čitatel",
                "agri_land_ha": "zemědělská půda [ha]",
            },
            title=f"{map_indicator}, {map_year} — {map_unit}",
            scope="europe",
        )

        fig_map.update_geos(
            showcountries=True,
            showcoastlines=True,
            showland=True,
            fitbounds="locations",
        )

        fig_map.update_layout(
            height=650,
            margin=dict(l=0, r=0, t=55, b=0),
            coloraxis_colorbar=dict(
                title=map_unit,
            ),
        )

        st.plotly_chart(fig_map, use_container_width=True)

        with st.expander("Data použitá pro kartogram"):
            st.dataframe(
                map_data.sort_values("value_per_agri_land", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Stáhnout mapová data jako CSV",
                data=map_data.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"faostat_map_"
                    f"{make_safe_filename(map_indicator)}_"
                    f"{map_year}.csv"
                ),
                mime="text/csv",
            )

        if missing_iso:
            st.info(
                "Tyto země nebyly zobrazeny, protože pro ně chybí ISO3 mapování: "
                + ", ".join(missing_iso)
            )
