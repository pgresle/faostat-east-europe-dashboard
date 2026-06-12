from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.download_prepare_faostat import OUTFILE, prepare

st.set_page_config(page_title="FAOSTAT Eastern Europe", layout="wide")

st.title("FAOSTAT: zemědělská produkce a land use ve východní Evropě")
st.caption("Časové řady od roku 1961 podle dostupnosti v FAOSTAT. U některých států začínají samostatné řady až po rozpadu dřívějších státních celků.")

@st.cache_data(show_spinner=True, ttl=24 * 60 * 60)
def load_data() -> pd.DataFrame:
    path = Path(OUTFILE)
    if not path.exists():
        return prepare()
    return pd.read_parquet(path)

with st.spinner("Načítám data. Při prvním spuštění se stáhnou bulk soubory z FAOSTAT."):
    df = load_data()

with st.sidebar:
    st.header("Nastavení")
    metric_group = st.selectbox(
        "Datová oblast",
        sorted(df["metric_group"].dropna().unique()),
        index=0,
    )
    norm = st.radio("Vyjádření", ["Absolutně", "Na osobu"], horizontal=True)
    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    years = st.slider("Roky", year_min, year_max, (1961, year_max))
    countries = st.multiselect(
        "Země",
        sorted(df["area"].dropna().unique()),
        default=sorted(df["area"].dropna().unique()),
    )
    top_n = st.slider("Kolik největších položek zobrazit v každé zemi", 3, 20, 8)
    chart_type = st.radio("Typ grafu", ["Plošný kumulativní", "Čárový"], horizontal=False)

base = df[(df["metric_group"] == metric_group) & (df["area"].isin(countries))]
base = base[(base["year"] >= years[0]) & (base["year"] <= years[1])]

value_col = "value" if norm == "Absolutně" else "value_per_capita"
y_label = "hodnota" if norm == "Absolutně" else "hodnota na osobu"

st.subheader(f"{metric_group} — {norm.lower()}")

if base.empty:
    st.warning("Pro zadané filtry nejsou dostupná data.")
    st.stop()

# Summary panel
c1, c2, c3 = st.columns(3)
c1.metric("Země", base["area"].nunique())
c2.metric("Roky", f"{int(base['year'].min())}–{int(base['year'].max())}")
c3.metric("Řádků dat", f"{len(base):,}".replace(",", " "))

with st.sidebar:
    n_cols = st.slider("Počet grafů v řádku", 1, 4, 3)
    chart_height = st.slider("Výška grafu", 220, 500, 280)

selected_countries = [c for c in countries if not base[base["area"] == c].empty]
# Sticky shared legend
legend_series = (
    base.groupby("series", as_index=False)[value_col]
    .sum()
    .sort_values(value_col, ascending=False)
    .head(top_n)["series"]
    .tolist()
)

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
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.sticky-legend-title {
    font-weight: 700;
    margin-bottom: 6px;
}
.sticky-legend-items {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    font-size: 13px;
}
.sticky-legend-item {
    white-space: nowrap;
}
</style>
<div class="sticky-legend">
  <div class="sticky-legend-title">Legenda</div>
  <div class="sticky-legend-items">
"""

for item in legend_series:
    legend_html += f'<div class="sticky-legend-item">● {item}</div>'

legend_html += """
  </div>
</div>
"""

st.markdown(legend_html, unsafe_allow_html=True)
for i in range(0, len(selected_countries), n_cols):
    cols = st.columns(n_cols)

    for j, country in enumerate(selected_countries[i:i + n_cols]):
        d = base[base["area"] == country].copy()
        if d.empty:
            continue

        # Choose top series by latest available value and aggregate remaining as Other.
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

        d["series_plot"] = d["series"].where(d["series"].isin(top_series), "Other")
        plot = d.groupby(["year", "series_plot"], as_index=False)[value_col].sum()

        with cols[j]:
            with st.container(border=True):
                st.markdown(f"#### {country}")

                if chart_type == "Plošný kumulativní":
                    fig = px.area(
                        plot,
                        x="year",
                        y=value_col,
                        color="series_plot",
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

                fig.update_xaxes(title_font=dict(size=10), tickfont=dict(size=9))
                fig.update_yaxes(title_font=dict(size=10), tickfont=dict(size=9))

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Data"):
                    st.dataframe(plot, use_container_width=True, hide_index=True)
                    st.download_button(
                        "CSV",
                        data=plot.to_csv(index=False).encode("utf-8"),
                        file_name=f"faostat_{country.replace(' ', '_')}_{metric_group.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
