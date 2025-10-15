# script.py
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from apify_client import ApifyClient
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from email.mime.image import MIMEImage
from zoneinfo import ZoneInfo
# import google.generativeai as genai
import re
import json
import math
import time
from google import genai


# === Config desde entorno (poner en GitHub Secrets) ===
APIFY_TOKEN = os.getenv("APIFY_TOKEN") # <-- definir en GitHub Secrets
# Aceptamos ambos nombres por compatibilidad
ACTOR_ID = os.getenv("ACTOR_ID") or os.getenv("APIFY_ACTOR_ID") or "apidojo/twitter-scraper-lite"
SEARCH_TERMS = os.getenv("SEARCH_TERMS_MEPAGO", "mercado pago, mercadopago")
tz_ar = ZoneInfo("America/Argentina/Buenos_Aires")

# Configuración del cliente (usa GEMINI_API_KEY desde el entorno)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        print("GenAI client inicializado.")
    except Exception as e:
        print(f"Advertencia: no se pudo inicializar genai.Client: {e}")
else:
    print("Advertencia: GEMINI_API_KEY no definida. La extracción de temas con IA no funcionará.")

# Modelo por defecto/fallback (lo cambiás si querés)
PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-pro",
    "gemini-flash-latest"
]

if not APIFY_TOKEN:
    raise RuntimeError("Falta APIFY_TOKEN (definilo en GitHub Secrets).")

apify_client = ApifyClient(APIFY_TOKEN)

# === Ventana de tiempo: TODO EL DÍA UTC actual (00:00 -> ahora) ===
now_utc = datetime.now(timezone.utc)
now_ar = now_utc.astimezone(tz_ar)

start_date_utc = now_ar.strftime("%Y-%m-%d") # día actual en UTC

run_input = {
    "start": start_date_utc, # día actual UTC
    "searchTerms": [s.strip() for s in SEARCH_TERMS.split(",") if s.strip()],
    "sort": "Top",
    "maxItems": 100000 # razonable
}

print(f"Ejecutando Actor: {ACTOR_ID}...")
print(f"Términos de búsqueda: {run_input['searchTerms']}")
run = apify_client.actor(ACTOR_ID).call(run_input=run_input)
print(f"Actor {ACTOR_ID} ejecutado. ID: {run['id']}, estado: {run['status']}")

# === Obtener items del dataset de la run (forma recomendada) ===
dataset_id = run.get("defaultDatasetId")
if not dataset_id:
    raise RuntimeError("No se encontró defaultDatasetId en la run de Apify.")

items_resp = apify_client.dataset(dataset_id).list_items()
dataset_items = items_resp.items or []

if not dataset_items:
    print("No se encontraron resultados en el dataset.")
    df = pd.DataFrame()
else:
    df = pd.DataFrame(dataset_items)
    # Enriquecer columnas si existe la estructura 'author'
    if "author" in df.columns:
        df["author/followers"] = df["author"].apply(lambda x: x.get("followers") if isinstance(x, dict) else None)
        df["author/userName"] = df["author"].apply(lambda x: x.get("userName") if isinstance(x, dict) else None)

    desired_columns = [
        "text","createdAt","author/userName","author/followers","url",
        "likeCount","replyCount","retweetCount","quoteCount","bookmarkCount",
        "viewCount","source"
    ]
    existing = [c for c in desired_columns if c in df.columns]
    if existing:
        df = df[existing]

    # Parseo de fecha si está la columna
    if "createdAt" in df.columns:
        date_format = "%a %b %d %H:%M:%S %z %Y"
        df["createdAt"] = pd.to_datetime(df["createdAt"], format=date_format, errors="coerce")

# Eliminar duplicados por URL si existe la columna
if "url" in df.columns:
    df.drop_duplicates(subset="url", inplace=True)

# Guardar CSV con timestamp UTC
Path("output").mkdir(exist_ok=True)
ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
csv_path = Path("output") / f"twitter_scrape_{ts}.csv"
df.to_csv(csv_path, index=False)
print(f"CSV guardado en: {csv_path}")

# <----------------- ENVÍO MAIL ------------------->

EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT_MEPAGO") # puede ser "a@b.com,c@d.com"

def fmt(n):
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

def truncate_text(s, length=180):
    if not isinstance(s, str):
        return s
    s = s.strip()
    return (s[:length] + "…") if len(s) > length else s

def df_to_html_table(df, cols):
    if not df.empty:
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()
        # permitir tags HTML (para el <a href="...">Tweet</a>)
        return out.to_html(index=False, escape=False, border=0)
    return "<p>No hay registros.</p>"

def list_available_models():
    """Intenta listar modelos disponibles (devuelve lista de nombres)."""
    if client is None:
        return []
    try:
        resp = client.models.list(page_size=200)  # iterador / pageable
        names = []
        for m in resp:
            # cada 'm' típicamente tiene .name o .modelId
            nm = getattr(m, "name", None) or getattr(m, "model", None) or getattr(m, "id", None)
            if nm:
                names.append(nm)
        return names
    except Exception as e:
        print(f"[IA] Error listando modelos: {e}")
        return []

def choose_model():
    """Escoge un modelo preferido que esté disponible; si no, devuelve None."""
    available = list_available_models()
    if not available:
        return None
    # comparo por substring (case-insensitive)
    available_low = [a.lower() for a in available]
    for pref in PREFERRED_MODELS:
        if pref.lower() in " ".join(available_low):
            # devolver la primera coincidencia exacta si existe
            for a in available:
                if pref.lower() in a.lower():
                    return a
    # si no hay coincidencias con preferidos, devolver el primer 'gemini' que vea
    for a in available:
        if "gemini" in a.lower():
            return a
    return available[0] if available else None

def extraer_temas_generales_con_ia(df, contexto, num_temas=3, sample_size=200):
    """
    Versión que usa google-genai Client.models.generate_content y maneja 404 / list-models fallback.
    """
    if client is None:
        return "El modelo de IA no está disponible. No se pudo realizar el análisis."

    tweets = df["text"].astype(str).dropna().tolist()
    if not tweets:
        return "No hay tweets suficientes para extraer temas generales."
    tweets = tweets[:sample_size]

    # agrupar en chunks por chars (heurística para tokens)
    max_chars = 3500
    chunks = []
    cur = ""
    for t in tweets:
        if not cur:
            cur = t
        elif len(cur) + 1 + len(t) > max_chars:
            chunks.append(cur)
            cur = t
        else:
            cur += "\n" + t
    if cur:
        chunks.append(cur)

    # escoger modelo inicial (puede ser el preferido o quemado)
    model_to_try = None
    # si configuraste algo explícito, probalo primero
    if os.getenv("GEMINI_MODEL"):
        model_to_try = os.getenv("GEMINI_MODEL")
    else:
        model_to_try = choose_model() or PREFERRED_MODELS[0]

    topic_counts = {}
    topic_examples = {}
    debug_responses = []

    for i, chunk in enumerate(chunks):
        prompt = f"""CONTEXTO: {contexto}

Analiza la siguiente colección de tweets y extrae hasta {num_temas} temas principales.
Devuelve SOLO un JSON: una lista de objetos con keys: name, explanation, example, score (opcional).
Tweets (chunk {i+1}/{len(chunks)}):
{chunk}
"""
        resp_text = None
        last_exc = None
        for attempt in range(2):
            try:
                # llamada con el client del SDK moderno
                resp = client.models.generate_content(model=model_to_try, contents=prompt, config={"temperature":0.2, "max_output_tokens":800})
                # la respuesta suele exponer .text
                resp_text = getattr(resp, "text", None) or getattr(resp, "output", None) or str(resp)
                break
            except Exception as e:
                last_exc = e
                # si es 404 de modelo, intentar obtener lista y cambiar modelo
                msg = str(e).lower()
                print(f"[IA] intento {attempt+1} fallo en chunk {i+1}: {e}")
                if "not found" in msg or "404" in msg or "is not found" in msg:
                    print("[IA] Modelo no encontrado o no compatible. Intentando listar modelos disponibles y elegir fallback...")
                    fallback = choose_model()
                    if fallback and fallback != model_to_try:
                        print(f"[IA] Cambiando modelo a: {fallback}")
                        model_to_try = fallback
                    else:
                        print("[IA] No se encontró modelo fallback.")
                time.sleep(1)

        if not resp_text:
            print(f"[IA] No hubo respuesta para chunk {i+1}. último error: {last_exc}")
            continue

        debug_responses.append(resp_text)

        # extraer JSON si lo devuelve
        m = re.search(r'(\[.*\]|\{.*\})', resp_text, flags=re.S)
        parsed = None
        if m:
            candidate = m.group(1)
            try:
                parsed = json.loads(candidate)
            except Exception:
                cleaned = candidate.replace("“", '"').replace("”", '"')
                cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
                try:
                    parsed = json.loads(cleaned)
                except Exception:
                    parsed = None

        # heurística si no hay JSON
        if parsed and isinstance(parsed, list):
            for it in parsed[:num_temas]:
                name = (it.get("name") or it.get("topic") or "").strip()
                if not name:
                    continue
                score = int(it.get("score", 0) or 0) if isinstance(it.get("score", 0), (int, float, str)) else 0
                topic_counts[name] = topic_counts.get(name, 0) + (score if score>0 else 1)
                if it.get("example") and name not in topic_examples:
                    topic_examples[name] = it.get("example")
        else:
            # intentar parseo simple de líneas "1. Tema - ..."
            lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
            for ln in lines:
                mnum = re.match(r'^\d+\.\s*(.+)', ln)
                if mnum:
                    name = mnum.group(1).split("-")[0].strip()[:80]
                    topic_counts[name] = topic_counts.get(name, 0) + 1

    if not topic_counts:
        print("[IA] No se encontraron temas. Respuestas crudas (primeros 300 chars):")
        for r in debug_responses:
            print(r[:300])
        return "No se pudieron extraer temas generales."

    sorted_topics = sorted(topic_counts.items(), key=lambda kv: kv[1], reverse=True)[:num_temas]
    out_lines = []
    for idx, (name, score) in enumerate(sorted_topics, start=1):
        example = topic_examples.get(name, "")
        if not example:
            matched = next((t for t in tweets if name.lower().split()[0] in t.lower()), None)
            if matched:
                example = matched
        out_lines.append(f"{idx}. {name}\n—\nEjemplo: \"{example or '—'}\"")
    return "\n\n".join(out_lines)


# === Preparación de datos del día (UTC) ===
counts_cols = ["likeCount","quoteCount","retweetCount","replyCount","bookmarkCount"]
numeric_cols = ["viewCount"] + counts_cols

work_df = df.copy()
work_df = coerce_numeric(work_df, numeric_cols)

# Filtrar al día UTC actual (coincide con 'start' que pasamos)
if "createdAt" in work_df.columns and pd.api.types.is_datetime64_any_dtype(work_df["createdAt"]):
    day_mask = work_df["createdAt"].dt.tz_convert(tz_ar).dt.date == now_ar.date()
    day_df = work_df[day_mask].copy()
else:
    day_df = work_df.copy()

# --- Distribución por hora (UTC) como imagen (PNG) embebida por CID ---
hourly_html = "<p>No hay datos para distribución por hora.</p>"
hourly_img_bytes = None

if "createdAt" in day_df.columns and not day_df.empty and pd.api.types.is_datetime64_any_dtype(day_df["createdAt"]):
    day_df["hour"] = day_df["createdAt"].dt.tz_convert(tz_ar).dt.hour
    hourly_counts = day_df.groupby("hour").size().reindex(range(24), fill_value=0)

    # gráfico simple con matplotlib
    fig, ax = plt.subplots(figsize=(8, 3))
    hourly_counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Hora")
    ax.set_ylabel("Tweets")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    hourly_img_bytes = buf.read()

    # En el HTML referenciamos la imagen por CID
    hourly_html = '<img src="cid:hourly_dist" alt="Distribución por hora" style="max-width:100%;">'

# --- NUEVO: Llamada a la función de IA ---
temas_generales = "No se pudo realizar el análisis de temas."
if not day_df.empty:
    contexto_ia = f"Análisis de tweets sobre {SEARCH_TERMS} del día {now_ar.strftime('%Y-%m-%d')}."
    temas_generales = extraer_temas_generales_con_ia(day_df, contexto_ia)

# Crear 'interacciones'
day_df["interacciones"] = (
    day_df.get("likeCount", 0)
    + day_df.get("replyCount", 0)
    + day_df.get("retweetCount", 0)
    + day_df.get("bookmarkCount", 0)
    + day_df.get("quoteCount", 0)
)

# Texto corto
if "text" in day_df.columns:
    day_df["text"] = day_df["text"].fillna("")
    day_df["text"] = day_df["text"].apply(lambda s: truncate_text(s, 180))

# --- Renombrar columnas para mostrar en el mail ---
rename_map = {
    "author/userName": "Usuario",
    "author/followers": "Seguidores",
    "url": "url",
    "viewCount": "Impresiones",
    "interacciones": "interacciones"
}
day_df = day_df.rename(columns=rename_map)


# Hacer que la URL sea clickeable en las tablas
if "url" in day_df.columns:
    day_df["url"] = day_df["url"].apply(lambda u: f'<a href="{u}">Tweet</a>' if isinstance(u, str) else u)

# Columnas a mostrar en tablas
show_cols = ["Usuario","Seguidores","text","url","Impresiones","interacciones"]

# Top 10 por impresiones
if "Impresiones" in day_df.columns:
    top_views = day_df.sort_values("Impresiones", ascending=False).head(10)
else:
    top_views = day_df.head(0)

# Top 10 por seguidores (1 tweet por usuario: el de más "Impresiones"; excluye 'grok')
need_cols = {"Usuario", "Seguidores"}
if need_cols.issubset(day_df.columns):
    tf = day_df.copy()

    # tipos numéricos
    tf["Seguidores"] = pd.to_numeric(tf["Seguidores"], errors="coerce").fillna(0).astype(int)
    if "Impresiones" in tf.columns:
        tf["Impresiones"] = pd.to_numeric(tf["Impresiones"], errors="coerce").fillna(0).astype(int)
    else:
        tf["Impresiones"] = 0

    # excluir 'grok'
    if "Usuario" in tf.columns:
        tf = tf[tf["Usuario"].str.lower() != "grok"]

    if not tf.empty:
        # elegir, para cada usuario, el tweet con mayor "Impresiones"
        best_idx = tf.groupby("Usuario")["Impresiones"].idxmax()
        top_followers = tf.loc[best_idx].copy()

        # ordenar por seguidores y tomar top 10
        top_followers = top_followers.sort_values("Seguidores", ascending=False).head(10)
    else:
        top_followers = tf.head(0)
else:
    top_followers = day_df.head(0)


# Tablas HTML
top_views_html = df_to_html_table(top_views, show_cols)
top_followers_html = df_to_html_table(top_followers, show_cols)

# Métricas totales (sobre el día)
total_tweets = len(day_df)
total_views = int(day_df["Impresiones"].sum()) if "Impresiones" in day_df.columns else 0
total_interactions = int(day_df["interacciones"].sum()) if "interacciones" in day_df.columns else 0

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
    .summary-box {{ background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; padding:16px; margin-bottom:20px; white-space: pre-wrap; }}
</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Tweets más destacados del día</h1>
            <div class="sub">Fecha : {now_ar.strftime("%Y-%m-%d")} &middot; Generado a las {now_ar.strftime("%H:%M:%S")}</div>
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
                    <div class="label">Impresiones</div>
                </div>
                <div class="card">
                    <div class="metric">{fmt(total_interactions)}</div>
                    <div class="label">Interacciones</div>
                </div>
            </div>

            <h2>2) Tweets más vistos</h2>
            {top_views_html}

            <h2>3) Usuarios con más seguidores</h2>
            {top_followers_html}
            <h2>4) Temas más mencionados (Análisis con IA)</h2>
            <div class="summary-box">
                <pre>{temas_generales}</pre>
            </div>
            <h2>5) Evolución tweets</h2>
{hourly_html}
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

    # Estructura MIME correcta para HTML + imágenes embebidas
    msg = MIMEMultipart("related")
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = f"Resumen Twitter Scraper - {now_utc.strftime('%Y-%m-%d')}"

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html_body, "html"))
    msg.attach(alt)

    # Adjuntar la imagen del gráfico si la generamos
    if hourly_img_bytes:
        img = MIMEImage(hourly_img_bytes, _subtype="png")
        img.add_header("Content-ID", "<hourly_dist>") # <-- Debe coincidir con el src="cid:hourly_dist"
        img.add_header("Content-Disposition", "inline", filename="hourly_dist.png")
        msg.attach(img)

    try:
        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
            print(f"Correo enviado a: {', '.join(recipients)}")
    except Exception as e:
        print(f"Error enviando correo: {e}")
else:
    print("No se envió correo (faltan variables EMAIL_*).")
