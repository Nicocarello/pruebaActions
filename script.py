import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd
from apify_client import ApifyClient
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
df.drop_duplicates(subset = 'url',inplace = True)

# Guardamos CSV con timestamp UTC
Path("output").mkdir(exist_ok=True)
ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
csv_path = Path("output") / f"twitter_scrape_{ts}.csv"
df.to_csv(csv_path, index=False)
print(f"CSV guardado en: {csv_path}")


# <----------------- ENVIO MAIL ------------------->


EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")  # puede ser "a@b.com,c@d.com"

def fmt(n):
    # separador de miles: punto (estilo ES)
    try:
        return f"{int(n):,}".replace(",", ".")
    except Exception:
        try:
            return f"{float(n):,.0f}".replace(",", ".")
        except Exception:
            return str(n)

def coerce_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        else:
            df[c] = 0
    return df

def truncate_text(s, length=140):
    if not isinstance(s, str):
        return s
    s = s.strip()
    return (s[:length] + "…") if len(s) > length else s

def df_to_html_table(df, cols):
    if not df.empty:
        # solo columnas existentes
        cols = [c for c in cols if c in df.columns]
        # orden final
        out = df[cols].copy()
        # escape automático; index=False y sin bordes HTML por CSS
        return out.to_html(index=False, escape=True, border=0)
    return "<p>No hay registros.</p>"

# === Preparar datos del día (UTC) ===
counts_cols = ["likeCount","quoteCount","retweetCount","replyCount","bookmarkCount"]
numeric_cols = ["viewCount"] + counts_cols

work_df = df.copy()

# asegurar tipos numéricos
work_df = coerce_numeric(work_df, numeric_cols)

# createdAt a datetime ya lo trabajaste arriba; si existe, filtramos por el día UTC
if "createdAt" in work_df.columns and pd.api.types.is_datetime64_any_dtype(work_df["createdAt"]):
    day_mask = work_df["createdAt"].dt.date == now_utc.date()
    day_df = work_df[day_mask].copy()
else:
    # si no hay createdAt, usamos todo el DF
    day_df = work_df.copy()

# métricas
total_tweets = len(day_df)
total_views = int(day_df["viewCount"].sum()) if "viewCount" in day_df.columns else 0
total_interactions = int(day_df[counts_cols].sum().sum()) if not day_df.empty else 0
df['interacciones'] = df['likeCount'] + df['replyCount'] + df['retweetCount'] + df['bookmarkCount'] + df['quoteCount']

# columnas simpáticas para mostrar
show_cols = [
    "author/userName","author/followers","text","url","viewCount","interacciones"
]
# textos cortos
if "text" in day_df.columns:
    day_df["text"] = day_df["text"].fillna("")
    day_df["text_short"] = day_df["text"].apply(lambda s: truncate_text(s, 180))
    # mostramos text_short en lugar de text
    show_cols = [c for c in show_cols if c != "text"]
    show_cols.insert(9, "text_short")  # cerca de métricas

# Top 10 por viewCount
if "viewCount" in day_df.columns:
    top_views = day_df.sort_values("viewCount", ascending=False).head(10)
else:
    top_views = day_df.head(0)

# Top 10 por followers del autor
if "author/followers" in day_df.columns:
    # coerce followers a num
    top_followers = day_df.copy()
    top_followers["author/followers"] = pd.to_numeric(top_followers["author/followers"], errors="coerce").fillna(0).astype(int)
    top_followers = top_followers.sort_values("author/followers", ascending=False).head(10)
else:
    top_followers = day_df.head(0)

# Tablas HTML
top_views_html = df_to_html_table(top_views, show_cols)
top_followers_html = df_to_html_table(top_followers, show_cols)

# Nota si no hay datos del día
empty_note = ""
if total_tweets == 0:
    empty_note = "<p><em>No se encontraron tweets para el día (UTC) seleccionado.</em></p>"

# === HTML del correo ===
html_body = f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Resumen Twitter Scraper</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:0; padding:0; background:#f6f7f9; }}
  .container {{ max-width: 980px; margin: 24px auto; background:#ffffff; border-radius: 10px; overflow:hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
  .header {{ background:#111827; color:#fff; padding:16px 20px; }}
  .header h1 {{ margin:0; font-size:18px; }}
  .sub {{ color:#9ca3af; font-size:12px; margin-top:4px; }}
  .content {{ padding: 20px; }}
  h2 {{ margin: 16px 0 8px; font-size: 16px; color:#111827; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:12px; margin: 12px 0 20px; }}
  .card {{ flex:1 1 220px; background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:14px; }}
  .metric {{ font-size: 22px; font-weight:700; color:#111827; }}
  .label {{ font-size: 12px; color:#6b7280; }}
  table {{ width:100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid #e5e7eb; vertical-align: top; }}
  th {{ font-weight:600; background:#f3f4f6; }}
  a {{ color:#2563eb; text-decoration:none; }}
  .footer {{ color:#6b7280; font-size:12px; padding: 16px 20px 20px; }}
  .muted {{ color:#6b7280; }}
</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Resumen diario Twitter Scraper</h1>
      <div class="sub">Fecha (UTC): {now_utc.strftime("%Y-%m-%d")} &middot; Generado a las {now_utc.strftime("%H:%M:%S")} UTC</div>
    </div>
    <div class="content">
      <h2>1) Totales del día</h2>
      <div class="cards">
        <div class="card">
          <div class="metric">{fmt(total_tweets)}</div>
          <div class="label">Tweets recolectados</div>
        </div>
        <div class="card">
          <div class="metric">{fmt(total_views)}</div>
          <div class="label">Vistas (viewCount)</div>
        </div>
        <div class="card">
          <div class="metric">{fmt(total_interactions)}</div>
          <div class="label">Interacciones</div>
        </div>
      </div>

      <h2>3) Top 10 por viewCount</h2>
      {top_views_html}

      <h2>4) Top 10 por followers del autor</h2>
      {top_followers_html}

      {empty_note}
    </div>
    <div class="footer">
      <div class="muted">Fuente: {ACTOR_ID}. Archivo CSV: {csv_path.name}</div>
    </div>
  </div>
</body>
</html>
"""

# === Envío de correo ===
should_send = all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, EMAIL_RECIPIENT])
if should_send:
    recipients = [email.strip() for email in EMAIL_RECIPIENT.split(",") if email.strip()]

    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"Resumen Twitter Scraper - {now_utc.strftime('%Y-%m-%d')} (UTC)"
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            # send_message toma los destinatarios de los headers
            server.send_message(msg)
            print(f"Correo enviado a: {', '.join(recipients)}")
    except Exception as e:
        print(f"Error enviando correo: {e}")
else:
    print("No se envió correo (faltan variables EMAIL_*).")

