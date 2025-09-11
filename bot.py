import os
import sys
import logging
import asyncio
import threading

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder, Application, CommandHandler,
    MessageHandler, ContextTypes, filters
)

# Hugging Face / NLP imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from langdetect import detect

# ─────────────────────────────────────────────
# Environment variables (Render Dashboard → Settings → Environment)
# ─────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")
DEFAULT_THRESHOLD = float(os.environ.get("TALSHIELD_THRESHOLD", "0.50"))

if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment variables")

# ─────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TalkShield")

# ─────────────────────────────────────────────
# Models (Hugging Face Hub, public repos)
# ─────────────────────────────────────────────
ENG_MODEL_ID = "Sekhinah/Talk_Shield_English"   # 7-label toxicity
TWI_MODEL_ID = "Sekhinah/Talk_Shield"           # 3-class sentiment

ENG_LABELS = ["toxicity","severe_toxicity","obscene","threat","insult","identity_hate","non_toxic"]
TWI_LABELS = ["Negative","Neutral","Positive"]
HARMFUL_ENG = ENG_LABELS[:-1]  # exclude non_toxic

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
# Language detection (Google → langdetect)
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
# Telegram handlers
# ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Send me a message and I’ll analyze it.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    lang = detect_lang(text)

    if lang == "ak":
        probs, pred = classify_twi(text)
        report = f"📊 TalkShield Report\nLang: TWI\n{probs}\nPred: {pred}"
    else:
        probs = classify_english(text)
        harmful = [lbl for lbl in HARMFUL_ENG if probs.get(lbl,0.0) >= DEFAULT_THRESHOLD] or ["non_toxic"]
        report = f"📊 TalkShield Report\nLang: EN\n{probs}\nLabels ≥ {DEFAULT_THRESHOLD}: {harmful}"

    if update.effective_chat.type == "private":
        await update.message.reply_text(report)
    else:
        # Group mode: auto delete harmful
        delete_flag = any(probs.get(lbl,0.0) >= DEFAULT_THRESHOLD for lbl in HARMFUL_ENG) if lang=="en" else (probs.get("Negative",0.0) >= DEFAULT_THRESHOLD)
        if delete_flag:
            try:
                await update.message.delete()
            except Exception:
                pass
            if ADMIN_CHAT_ID:
                try:
                    await context.bot.send_message(chat_id=int(ADMIN_CHAT_ID), text=f"🧹 Deleted message: {text}")
                except Exception:
                    pass

# ─────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────
async def main_async():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # IMPORTANT for background-threaded run under Gunicorn:
    # disable signal handling in PTB because signals are only allowed in main thread
    await app.run_polling(stop_signals=None)

def main():
    if "ipykernel" in sys.modules:  # running in Jupyter
        return asyncio.create_task(main_async())
    else:
        asyncio.run(main_async())

# ─────────────────────────────────────────────
# Flask app definition (for Render/Gunicorn)
# ─────────────────────────────────────────────
from flask import Flask
flask_app = Flask(__name__)   # This is the WSGI app Gunicorn will serve
app = flask_app               # Expose as 'app' for Gunicorn (gunicorn bot:app)

@flask_app.route("/")
def index():
    return "✅ TalkShield bot service is alive"

# Start the Telegram bot ONCE in a background thread when module imports
# (each Gunicorn worker imports this once)
_bot_started = False
_bot_lock = threading.Lock()

def _start_bot_once():
    global _bot_started
    with _bot_lock:
        if not _bot_started:
            t = threading.Thread(target=lambda: asyncio.run(main_async()), daemon=True)
            t.start()
            _bot_started = True
            log.info("TalkShield Telegram bot started in background thread.")

_start_bot_once()

# ─────────────────────────────────────────────
# Local run (for development)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
