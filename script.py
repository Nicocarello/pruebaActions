import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from apify_client import ApifyClient

# === Config desde entorno (poner en GitHub Secrets) ===
APIFY_TOKEN = os.getenv("APIFY_TOKEN")  # <-- definir en GitHub Secrets
ACTOR_ID = os.getenv("ACTOR_ID", "apidojo/twitter-scraper-lite")
SEARCH_TERMS = os.getenv("SEARCH_TERMS", "mercado libre")

if not APIFY_TOKEN:
    raise RuntimeError("Falta APIFY_TOKEN (definilo en GitHub Secrets).")

apify_client = ApifyClient(APIFY_TOKEN)

# Ventana de tiempo: última hora (UTC)
now_utc = datetime.now(timezone.utc)
one_hour_ago = now_utc - timedelta(hours=1)

# El actor de ejemplo usa fechas tipo YYYY-MM-DD; si tu actor acepta datetime, podés pasar ISO
run_input = {
    "start": one_hour_ago.strftime("%Y-%m-%d"),
    "searchTerms": [s.strip() for s in SEARCH_TERMS.split(",") if s.strip()],
    "sort": "Top",
    "maxItems": 1000000
}

print(f"Ejecutando Actor: {ACTOR_ID}...")
run = apify_client.actor(ACTOR_ID).call(run_input=run_input)

print(f"Actor {ACTOR_ID} ejecutado. ID: {run['id']}, estado: {run['status']}")

# Obtener items del dataset de la run
dataset_items = apify_client.run(run["id"]).dataset().list_items().items

if not dataset_items:
    print("No se encontraron resultados en el dataset.")
    # Igualmente generamos un CSV vacío con timestamp para trazar ejecuciones
    df = pd.DataFrame()
else:
    df = pd.DataFrame(dataset_items)
    # Enriquecemos columnas si existen
    if "author" in df.columns:
        df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
        df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

    desired_columns = [
        "text","createdAt","author/userName","author/followers","url",
        "likeCount","replyCount","retweetCount","quoteCount","bookmarkCount",
        "viewCount","source"
    ]
    # Filtramos solo las columnas que existan
    existing = [c for c in desired_columns if c in df.columns]
    if existing:
        df = df[existing]

    # Parseo de fecha si está la columna
    if "createdAt" in df.columns:
        date_format = "%a %b %d %H:%M:%S %z %Y"
        df["createdAt"] = pd.to_datetime(df["createdAt"], format=date_format, errors="coerce")


# Eliminamos duplicados
df.drop_duplicates(subset = 'url')

# Guardamos CSV con timestamp UTC
Path("output").mkdir(exist_ok=True)
ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
csv_path = Path("output") / f"twitter_scrape_{ts}.csv"
df.to_csv(csv_path, index=False)
print(f"CSV guardado en: {csv_path}")








