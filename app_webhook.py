import os
import re
import logging
import asyncio
from flask import Flask, request

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator
from langdetect import detect

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ──────────────────────────────────────────────
# Config & Logging
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("TalkShield")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment variables")

ENG_MODEL_ID = "Sekhinah/Talk_Shield_English"  # 7-label toxicity
TWI_MODEL_ID = "Sekhinah/Talk_Shield"          # 3-class sentiment

ENG_LABELS = [
    "toxicity", "severe_toxicity", "obscene",
    "threat", "insult", "identity_hate", "non_toxic"
]
HARMFUL_ENG = ENG_LABELS[:-1]  # exclude non_toxic
TWI_LABELS = ["Negative", "Neutral", "Positive"]

DEFAULT_THRESHOLD = 0.5

# ──────────────────────────────────────────────
# Lazy model loading
# ──────────────────────────────────────────────
_eng_tok, _eng_mdl = None, None
_twi_tok, _twi_mdl = None, None

def load_english():
    global _eng_tok, _eng_mdl
    if _eng_tok is None:
        log.info("Loading English model…")
        _eng_tok = AutoTokenizer.from_pretrained(ENG_MODEL_ID)
        _eng_mdl = AutoModelForSequenceClassification.from_pretrained(ENG_MODEL_ID).eval()

def load_twi():
    global _twi_tok, _twi_mdl
    if _twi_tok is None:
        log.info("Loading Twi model…")
        _twi_tok = AutoTokenizer.from_pretrained(TWI_MODEL_ID)
        _twi_mdl = AutoModelForSequenceClassification.from_pretrained(TWI_MODEL_ID).eval()

# ──────────────────────────────────────────────
# Language detection
# ──────────────────────────────────────────────
WORD_RE = re.compile(r"[A-Za-z0-9'’]+")
TWI_HINTS = {"agyimi", "ankasa", "koraa", "paa", "hw3", "hwe", "s3", "se", "biaa", "dabi", "medaase"}

def detect_lang(text: str) -> str:
    try:
        lg = (GoogleTranslator().detect(text) or "").lower()
        if lg in ("ak", "tw", "akan", "twi"):
            return "ak"
        if lg == "en":
            return "en"
    except Exception:
        pass
    low = text.lower()
    if "3" in low: return "ak"
    toks = set(m.group(0).lower() for m in WORD_RE.finditer(low))
    if toks & TWI_HINTS: return "ak"
    try:
        lg2 = (detect(text) or "").lower()
        if lg2 in ("ak", "tw", "akan", "twi"): return "ak"
        if lg2 == "en": return "en"
    except Exception:
        pass
    return "en"

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def classify_english(text: str):
    load_english()
    inputs = _eng_tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        probs = torch.sigmoid(_eng_mdl(**inputs).logits).squeeze().tolist()
    return {ENG_LABELS[i]: float(probs[i]) for i in range(len(ENG_LABELS))}

def classify_twi(text: str):
    load_twi()
    inputs = _twi_tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = _twi_mdl(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        idx = int(torch.argmax(logits, dim=-1))
    return {TWI_LABELS[i]: float(probs[i]) for i in range(len(TWI_LABELS))}, TWI_LABELS[idx]

def route_and_classify(text: str, threshold=DEFAULT_THRESHOLD):
    lang = detect_lang(text)
    if lang == "ak":
        probs, pred = classify_twi(text)
        delete_flag = probs.get("Negative", 0.0) >= threshold
        report = f"📊 TalkShield\nLang: Twi\n{probs}\nPred: {pred}"
        return {"lang": lang, "probs": probs, "pred": pred, "delete": delete_flag, "report": report}
    else:
        probs = classify_english(text)
        harmful = [lbl for lbl in HARMFUL_ENG if probs.get(lbl, 0.0) >= threshold]
        delete_flag = bool(harmful)
        report = f"📊 TalkShield\nLang: English\n{probs}\nFlags ≥{threshold}: {harmful or ['non_toxic']}"
        return {"lang": lang, "probs": probs, "pred": "", "delete": delete_flag, "report": report}

# ──────────────────────────────────────────────
# Telegram Handlers
# ──────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello 👋 TalkShield bot is active!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    result = await asyncio.to_thread(route_and_classify, text, DEFAULT_THRESHOLD)

    if update.effective_chat.type == "private":
        await update.message.reply_text(result["report"])
    else:
        if result["delete"]:
            try:
                await update.message.delete()
                log.info(f"Deleted toxic message: {text}")
            except Exception as e:
                log.error(f"Failed to delete: {e}")

# ──────────────────────────────────────────────
# Flask + Webhook
# ──────────────────────────────────────────────
app = Flask(__name__)
telegram_app = Application.builder().token(TOKEN).build()

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    telegram_app.update_queue.put_nowait(update)
    return "ok", 200

@app.route("/")
def index():
    return "TalkShield Bot Running ✅", 200
