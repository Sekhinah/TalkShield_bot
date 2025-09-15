import os
import logging
import torch
import threading, asyncio
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
# Environment variables
# ─────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")
DEFAULT_THRESHOLD = float(os.environ.get("TALKSHIELD_THRESHOLD", "0.50"))

if not TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN not set in environment variables")

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TalkShield")

# ─────────────────────────────────────────────
# Hugging Face models
# ─────────────────────────────────────────────
ENG_MODEL_ID = "Sekhinah/Talk_Shield_English"   # 7-label toxicity
TWI_MODEL_ID = "Sekhinah/Talk_Shield"           # 3-class sentiment

ENG_LABELS = ["toxicity","severe_toxicity","obscene","threat","insult","identity_attack","sexual_explicit"]
TWI_LABELS = ["Negative","Neutral","Positive"]

_eng_tok, _eng_mdl = None, None
_twi_tok, _twi_mdl = None, None

def load_english():
    global _eng_tok, _eng_mdl
    if _eng_tok is None or _eng_mdl is None:
        _eng_tok = AutoTokenizer.from_pretrained(ENG_MODEL_ID)
        _eng_mdl = AutoModelForSequenceClassification.from_pretrained(ENG_MODEL_ID).eval()

def load_twi():
    global _twi_tok, _twi_mdl
    if _twi_tok is None or _twi_mdl is None:
        _twi_tok = AutoTokenizer.from_pretrained(TWI_MODEL_ID)
        _twi_mdl = AutoModelForSequenceClassification.from_pretrained(TWI_MODEL_ID).eval()

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
# Classification
# ─────────────────────────────────────────────
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
# Telegram Handlers
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
# Flask app (Gunicorn entrypoint)
# ─────────────────────────────────────────────
flask_app = Flask(__name__)

application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

@flask_app.route("/")
def index():
    return "✅ TalkShield bot service is alive"

@flask_app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True)
    update = Update.de_json(data, application.bot)
    application.update_queue.put_nowait(update)
    return "ok"

# ─────────────────────────────────────────────
# Background dispatcher loop (critical fix)
# ─────────────────────────────────────────────
def run_async_loop():
    async def runner():
        await application.initialize()
        await application.start()
        await asyncio.Event().wait()  # keep loop alive

    asyncio.run(runner())

threading.Thread(target=run_async_loop, daemon=True).start()

# Gunicorn entrypoint
app = flask_app
