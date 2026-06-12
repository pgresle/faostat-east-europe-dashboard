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

for country in countries:
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

    with st.container(border=True):
        st.markdown(f"### {country}")
        if chart_type == "Plošný kumulativní":
            fig = px.area(plot, x="year", y=value_col, color="series_plot", labels={"year": "rok", value_col: y_label, "series_plot": "položka"})
        else:
            fig = px.line(plot, x="year", y=value_col, color="series_plot", labels={"year": "rok", value_col: y_label, "series_plot": "položka"})
        fig.update_layout(height=420, legend_title_text="", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Data ke stažení"):
            st.dataframe(plot, use_container_width=True, hide_index=True)
            st.download_button(
                "Stáhnout CSV pro tuto zemi",
                data=plot.to_csv(index=False).encode("utf-8"),
                file_name=f"faostat_{country.replace(' ', '_')}_{metric_group.replace(' ', '_')}.csv",
                mime="text/csv",
            )
