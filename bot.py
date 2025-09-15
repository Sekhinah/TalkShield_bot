import os
import logging
import torch
import asyncio
import threading
from flask import Flask, request
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from langdetect import detect

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DEFAULT_THRESHOLD = float(os.environ.get("TALKSHIELD_THRESHOLD", "0.50"))

if not TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN not set")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TalkShield")

# ─────────────────────────────────────────────
# Hugging Face models
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Hugging Face models
# ─────────────────────────────────────────────
ENG_MODEL_ID = "Sekhinah/Talk_Shield_English"
TWI_MODEL_ID = "Sekhinah/Talk_Shield"

ENG_LABELS = ["toxicity","severe_toxicity","obscene","threat","insult","identity_attack","sexual_explicit"]
TWI_LABELS = ["Negative","Neutral","Positive"]

_eng_tok, _eng_mdl = None, None
_twi_tok, _twi_mdl = None, None

def load_english():
    global _eng_tok, _eng_mdl
    if _eng_tok is None or _eng_mdl is None:
        log.info("📥 Loading English TalkShield model...")
        _eng_tok = AutoTokenizer.from_pretrained(
            ENG_MODEL_ID,
            use_fast=False
        )
        _eng_mdl = AutoModelForSequenceClassification.from_pretrained(
            ENG_MODEL_ID
        ).eval()
        log.info("✅ English model ready")

def load_twi():
    global _twi_tok, _twi_mdl
    if _twi_tok is None or _twi_mdl is None:
        log.info("📥 Loading Twi TalkShield model...")
        _twi_tok = AutoTokenizer.from_pretrained(
            TWI_MODEL_ID,
            use_fast=False
        )
        _twi_mdl = AutoModelForSequenceClassification.from_pretrained(
            TWI_MODEL_ID
        ).eval()
        log.info("✅ Twi model ready")

# ─────────────────────────────────────────────
# Preload models at startup
# ─────────────────────────────────────────────
load_english()
load_twi()

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

def classify_english(text: str):
    load_english()
    inputs = _eng_tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = _eng_mdl(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().tolist()
    return {ENG_LABELS[i]: float(probs[i]) for i in range(len(ENG_LABELS))}

def classify_twi(text: str):
    load_twi()
    inputs = _twi_tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = _twi_mdl(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        idx = int(torch.argmax(logits, dim=-1))
    return {TWI_LABELS[i]: float(probs[i]) for i in range(len(TWI_LABELS))}, TWI_LABELS[idx]

# ─────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me a message and I’ll analyze it with TalkShield.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    lang = detect_lang(text)

    if lang == "ak":
        probs, pred = classify_twi(text)
        report = f"📊 TalkShield Report\nLang: TWI\nPrediction: {pred}\nProbs: {probs}"
    else:
        probs = classify_english(text)
        harmful = [lbl for lbl, v in probs.items() if v >= DEFAULT_THRESHOLD]
        if not harmful:
            harmful = ["non_toxic"]
        report = f"📊 TalkShield Report\nLang: EN\nLabels ≥ {DEFAULT_THRESHOLD}: {harmful}\nProbs: {probs}"

    await update.message.reply_text(report)

# ─────────────────────────────────────────────
# Flask + Telegram Application in background
# ─────────────────────────────────────────────
flask_app = Flask(__name__)
application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

# Dedicated event loop for Telegram
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
