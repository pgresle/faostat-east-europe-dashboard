# FAOSTAT Eastern Europe dashboard

Streamlit dashboard pro časové řady FAOSTAT: zemědělská produkce, harvested area a land use pro země východní Evropy.

## Co projekt dělá

- stahuje FAOSTAT bulk data:
  - `QCL` — Production: Crops and livestock products,
  - `RL` — Land, Inputs and Sustainability: Land Use,
  - `OA` — Population and Employment: Annual population,
- filtruje země podle UN M49 Eastern Europe:
  - Belarus, Bulgaria, Czechia, Hungary, Poland, Republic of Moldova, Romania, Russian Federation, Slovakia, Ukraine,
- připraví dlouhou tabulku `data/processed/faostat_eastern_europe_long.parquet`,
- dopočítá hodnoty na osobu přes FAOSTAT populaci,
- vykreslí pro každou zemi samostatný graf v dashboardu,
- umožní přepínat absolutní / per capita a plošný kumulativní / čárový graf.

## Lokální spuštění

```bash
pip install -r requirements.txt
python scripts/download_prepare_faostat.py
streamlit run streamlit_app.py
```

Při prvním spuštění dashboardu se data stáhnou automaticky, i když předtím nespustíte přípravný skript.

## Poznámka k časovým řadám od roku 1961

FAOSTAT má řady od roku 1961, ale ne vždy pro dnešní stát jako samostatnou jednotku. Pro státy vzniklé po rozpadu USSR, Czechoslovakia apod. začínají samostatné řady obvykle až od 90. let. Projekt nedělá historickou rekonstrukci ani rozpad bývalých federací; ukazuje dostupná data podle FAOSTAT.

## Nasazení na Streamlit Community Cloud

1. Nahrajte složku do GitHub repozitáře.
2. Na https://streamlit.io/cloud vytvořte novou aplikaci.
3. Jako main file zadejte `streamlit_app.py`.
4. Streamlit si nainstaluje balíčky z `requirements.txt`.

## Úprava regionu

Seznam zemí je v souboru `scripts/download_prepare_faostat.py` v proměnné `EASTERN_EUROPE`.
