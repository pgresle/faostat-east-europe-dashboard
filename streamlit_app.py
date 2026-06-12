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


@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def load_data(force: bool = False) -> pd.DataFrame:
    path = Path(OUTFILE)

    if force or not path.exists():
        return prepare(force=True)

    return pd.read_parquet(path)


with st.sidebar:
    refresh_data = st.button("Obnovit / znovu stáhnout FAOSTAT data")


with st.spinner("Načítám data. Při prvním spuštění se stáhnou bulk soubory z FAOSTAT."):
    df = load_data(force=refresh_data)


# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Nastavení")

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
# Filter data
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

for country in selected_countries:
    d_country = base[base["area"] == country].copy()

    if d_country.empty:
        continue

    latest_year = d_country.groupby("series")["year"].max().rename("latest_year")
    latest = d_country.merge(latest_year, on="series")
    latest = latest[latest["year"] == latest["latest_year"]]

    country_top_series = (
        latest.groupby("series", as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(top_n)["series"]
        .tolist()
    )

    visible_series.update(country_top_series)


series_order = (
    base[base["series"].isin(visible_series)]
    .groupby("series", as_index=False)[value_col]
    .sum()
    .sort_values(value_col, ascending=False)["series"]
    .tolist()
)

if "Other" not in series_order:
    series_order.append("Other")


# Calm land-use / agriculture palette: greens, yellows, browns, reds, blues.
palette = [
    "#2E7D32",  # dark green
    "#66BB6A",  # medium green
    "#A5D6A7",  # light green
    "#FBC02D",  # yellow
    "#F9A825",  # amber
    "#8D6E63",  # brown
    "#A1887F",  # light brown
    "#D84315",  # burnt red
    "#E57373",  # soft red
    "#1565C0",  # blue
    "#64B5F6",  # light blue
    "#00695C",  # teal green
    "#9E9D24",  # olive
    "#C0CA33",  # yellow green
    "#5D4037",  # dark brown
    "#BCAAA4",  # pale brown
    "#AD1457",  # muted pink/red
    "#283593",  # indigo blue
    "#4FC3F7",  # sky blue
    "#78909C",  # blue grey
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

tab_charts, tab_map = st.tabs(["Grafy podle zemí", "Kartogram Evropy"])


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

            latest_year = d.groupby("series")["year"].max().rename("latest_year")
            latest = d.merge(latest_year, on="series")
            latest = latest[latest["year"] == latest["latest_year"]]

            top_series = (
                latest.groupby("series", as_index=False)[value_col]
                .sum()
                .sort_values(value_col, ascending=False)
                .head(top_n)["series"]
                .tolist()
            )

            d["series_plot"] = d["series"].where(
                d["series"].isin(top_series),
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
                                f"{country.replace(' ', '_')}_"
                                f"{metric_group.replace(' ', '_')}_"
                                f"{selected_unit.replace(' ', '_')}.csv"
                            ),
                            mime="text/csv",
                        )


# -------------------------------------------------------------------
# Tab 2: Choropleth map of Europe
# -------------------------------------------------------------------

with tab_map:
    st.markdown("### Kartogram Evropy")

    st.caption(
        "Kartogram zobrazuje jednu vybranou položku pro vybraný rok. "
        "Data jsou filtrována podle stejné datové oblasti, jednotky a vyjádření jako grafy."
    )

    map_c1, map_c2 = st.columns([1, 2])

    available_map_years = sorted(base["year"].dropna().unique())

    with map_c1:
        map_year = st.selectbox(
            "Rok pro mapu",
            available_map_years,
            index=len(available_map_years) - 1,
        )

    available_map_series = sorted(base["series"].dropna().unique())

    with map_c2:
        map_series = st.selectbox(
            "Položka pro kartogram",
            available_map_series,
            index=0,
        )

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

    map_data = base[
        (base["year"] == map_year)
        & (base["series"] == map_series)
    ].copy()

    map_data = (
        map_data.groupby(["area"], as_index=False)[value_col]
        .sum()
    )

    map_data["iso3"] = map_data["area"].map(iso3_map)

    missing_iso = (
        map_data[map_data["iso3"].isna()]["area"]
        .dropna()
        .unique()
        .tolist()
    )

    map_data = map_data.dropna(subset=["iso3"])

    if map_data.empty:
        st.warning("Pro vybranou položku a rok nejsou dostupná mapová data.")
    else:
        fig_map = px.choropleth(
            map_data,
            locations="iso3",
            color=value_col,
            hover_name="area",
            hover_data={
                "iso3": False,
                value_col: ":,.2f",
            },
            color_continuous_scale=[
                "#FFF7BC",
                "#FEC44F",
                "#FE9929",
                "#D95F0E",
                "#993404",
            ],
            labels={
                value_col: y_label,
            },
            title=f"{map_series}, {map_year} — {y_label}",
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
                title=y_label,
            ),
        )

        st.plotly_chart(fig_map, use_container_width=True)

        with st.expander("Data použitá pro mapu"):
            st.dataframe(
                map_data.sort_values(value_col, ascending=False),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Stáhnout mapová data jako CSV",
                data=map_data.to_csv(index=False).encode("utf-8"),
                file_name=(
                    f"faostat_map_"
                    f"{metric_group.replace(' ', '_')}_"
                    f"{selected_unit.replace(' ', '_')}_"
                    f"{map_year}.csv"
                ),
                mime="text/csv",
            )

        if missing_iso:
            st.info(
                "Tyto země nebyly zobrazeny, protože pro ně chybí ISO3 mapování: "
                + ", ".join(missing_iso)
            )
