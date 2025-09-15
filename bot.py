import os
import logging
import asyncio
import threading
import requests
from flask import Flask, request
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from deep_translator import GoogleTranslator
from langdetect import detect

# ─────────────────────────────────────────────
# Environment variables
# ─────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
DEFAULT_THRESHOLD = float(os.environ.get("TALKSHIELD_THRESHOLD", "0.50"))

if not TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN not set")
if not HF_API_TOKEN:
    raise RuntimeError("❌ HF_API_TOKEN not set")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TalkShield")

# ─────────────────────────────────────────────
# Hugging Face API setup
# ─────────────────────────────────────────────
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
ENG_MODEL_ID = "Sekhinah/Talk_Shield_English"
TWI_MODEL_ID = "Sekhinah/Talk_Shield"

def classify_english(text: str):
    payload = {"inputs": text}
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{ENG_MODEL_ID}",
        headers=HEADERS,
        json=payload,
        timeout=30
    )
    if resp.status_code != 200:
        return {"error": resp.text}
    return resp.json()

def classify_twi(text: str):
    payload = {"inputs": text}
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{TWI_MODEL_ID}",
        headers=HEADERS,
        json=payload,
        timeout=30
    )
    if resp.status_code != 200:
        return {"error": resp.text}
    return resp.json()

# ─────────────────────────────────────────────
# Language detection
# ─────────────────────────────────────────────
def detect_lang(text: str) -> str:
    try:
        lg = (GoogleTranslator().detect(text) or "").lower()
        if lg in ("ak","twi","akan","tw"): return "ak"
        if lg == "en": return "en"
    except: pass
    try:
        lg2 = (detect(text) or "").lower()
        if lg2 in ("ak","twi","akan","tw"): return "ak"
        if lg2 == "en": return "en"
    except: pass
    return "en"

# ─────────────────────────────────────────────
# Telegram handlers
# ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me a message and I’ll analyze it with TalkShield.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    lang = detect_lang(text)

    if lang == "ak":
        result = classify_twi(text)
        await update.message.reply_text(f"📊 TalkShield Report\nLang: TWI\n{result}")
    else:
        result = classify_english(text)
        await update.message.reply_text(f"📊 TalkShield Report\nLang: EN\n{result}")

# ─────────────────────────────────────────────
# Flask + Telegram Application
# ─────────────────────────────────────────────
flask_app = Flask(__name__)
application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

telegram_loop = asyncio.new_event_loop()
def run_telegram():
    asyncio.set_event_loop(telegram_loop)
    telegram_loop.run_until_complete(application.initialize())
    telegram_loop.run_until_complete(application.start())
    telegram_loop.run_forever()

threading.Thread(target=run_telegram, daemon=True).start()

@flask_app.route("/")
def index():
    return "✅ TalkShield bot service is alive", 200

@flask_app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True)
    if not data:
        return "no data", 400
    update = Update.de_json(data, application.bot)
    asyncio.run_coroutine_threadsafe(application.process_update(update), telegram_loop)
    return "ok", 200

# Gunicorn entrypoint
app = flask_app
