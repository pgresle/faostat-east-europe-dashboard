from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.download_prepare_faostat import OUTFILE, prepare


st.set_page_config(page_title="FAOSTAT Eastern Europe", layout="wide")

st.title("FAOSTAT: zemědělská produkce a land use ve východní Evropě")
st.caption(
    "Časové řady od roku 1961 podle dostupnosti v FAOSTAT. "
    "U některých států začínají samostatné řady až po rozpadu dřívějších státních celků."
)


@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def load_data() -> pd.DataFrame:
    path = Path(OUTFILE)
    if not path.exists():
        return prepare()
    return pd.read_parquet(path)


with st.spinner("Načítám data. Při prvním spuštění se stáhnou bulk soubory z FAOSTAT."):
    df = load_data()


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
# First, identify all series that will appear in at least one country chart.
# This ensures that the shared legend corresponds to the actual plotted series.

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

# "Other" is used in country charts for all non-top categories.
if "Other" not in series_order:
    series_order.append("Other")


# Shared color palette for both the sticky legend and all charts.
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
  <div class="sticky-legend-title">Společná legenda</div>
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

st.markdown(legend_html, unsafe_allow_html=True)


# -------------------------------------------------------------------
# Country chart grid
# -------------------------------------------------------------------

for i in range(0, len(selected_countries), n_cols):
    cols = st.columns(n_cols)

    for j, country in enumerate(selected_countries[i:i + n_cols]):
        d = base[base["area"] == country].copy()

        if d.empty:
            continue

        # Choose top series by latest available value and aggregate the remaining series as Other.
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

        # Make sure Plotly uses the same category order as the shared legend.
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
