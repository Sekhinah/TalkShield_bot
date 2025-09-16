import os
import logging
import asyncio
import threading
import requests
import time
from flask import Flask, request
from typing import Dict, Any

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
SPACE_URL = os.environ.get("SPACE_URL")  # e.g. https://<username>-talkshield-api.hf.space
DEFAULT_THRESHOLD = float(os.environ.get("TALKSHIELD_THRESHOLD", "0.50"))

if not TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN not set")
if not SPACE_URL:
    raise RuntimeError("❌ SPACE_URL not set (e.g. https://<user>-talkshield-api.hf.space)")

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TalkShield")
# ──────────────────────────────
# Logging deleted messages
# ──────────────────────────────
LOG_FILE = "deleted_logs.jsonl"

def log_deleted_message(chat_id, user_id, text, labels):
    """Append deleted message info to a log file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "chat_id": chat_id,
        "user_id": user_id,
        "text": text,
        "labels": labels,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info("📝 Deleted message logged: %s", entry)


# ─────────────────────────────────────────────
# Helpers: API calls to your Space
# ─────────────────────────────────────────────

def call_space(path, payload, attempts=3):
    url = f"{SPACE_URL.rstrip('/')}/{path.lstrip('/')}"
    for i in range(attempts):
        try:
            resp = requests.post(url, json=payload, timeout=45)
            if resp.status_code == 200:
                return resp.json()
            # If Space is still warming up
            if resp.status_code in (503, 502):
                if i < attempts - 1:
                    wait = 2 ** i
                    log.warning("Space cold start, retrying in %s sec...", wait)
                    time.sleep(wait)
                    continue
            return {"error": f"Space error {resp.status_code}: {resp.text}"}
        except Exception as e:
            if i < attempts - 1:
                wait = 2 ** i
                log.warning("Request failed (%s). Retrying in %s sec...", e, wait)
                time.sleep(wait)
                continue
            return {"error": str(e)}


def classify_english(text: str) -> Dict[str, Any]:
    return call_space("/english", {"text": text})

def classify_twi(text: str) -> Dict[str, Any]:
    return call_space("/twi", {"text": text})

# ─────────────────────────────────────────────
# Pretty formatting
# ─────────────────────────────────────────────
def format_english(scores: Dict[str, float]) -> str:
    if "error" in scores:
        return f"❌ Inference error: {scores['error']}"

    # Expecting shape: {"toxicity": 0.12, "severe_toxicity": 0.01, ...}
    harmful = [k for k, v in scores.items() if v >= DEFAULT_THRESHOLD]
    lines = [f"• {k}: {scores[k]:.2f}" for k in sorted(scores.keys())]
    harm_line = "None (non_toxic)" if not harmful else ", ".join(harmful)
    return (
        f"Labels ≥ {DEFAULT_THRESHOLD:.2f}: {harm_line}\n"
        + "\n".join(lines)
    )

def format_twi(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"❌ Inference error: {result['error']}"

    # Expecting shape: {"scores": {"Negative": 0.9, ...}, "prediction": "Negative"}
    pred = result.get("prediction", "?")
    scores = result.get("scores", {})
    lines = [f"• {k}: {scores.get(k, 0):.2f}" for k in ["Negative", "Neutral", "Positive"]]
    return f"Prediction: {pred}\n" + "\n".join(lines)

# ─────────────────────────────────────────────
# Telegram Handlers
# ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🛡️ TalkShield is live!\n"
        "• Send an English message for toxicity scores\n"
        "• Send a Twi message for sentiment\n"
        f"• Harmful threshold: {DEFAULT_THRESHOLD:.2f}\n"
    )

def is_twi_like(text: str) -> bool:
    """
    Quick heuristic for Twi/Akan (offline-friendly).
    If you later prefer, replace with langid/fasttext, or send both and route by confidence.
    """
    text_l = text.lower()
    # common Twi tokens; adjust as you like
    hints = ["ɛ", "ɔ", "wo", "w'","me", "ɛyɛ", "nsɛm", "waa", "agyimi", "dam", "pɔ"]
    return any(h in text_l for h in hints)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (update.message.text or "").strip()
    if not text:
        return

    chat = update.effective_chat
    is_group = chat.type in ("group", "supergroup")

    # ─────────────── TWI HANDLING ───────────────
    if is_twi_like(text):
        result = classify_twi(text)
        pretty = format_twi(result)

        if is_group:
            # In groups: only act if prediction is Negative
            pred = result.get("prediction", "")
            if pred == "Negative":
                try:
                    await context.bot.delete_message(chat.id, update.message.message_id)
                    await context.bot.send_message(
                        chat.id,
                        "🚨 A message was removed for toxicity (Twi: Negative sentiment)"
                    )
                    # ✅ Log deletion
                    log_deleted_message(
                        chat.id,
                        update.effective_user.id,
                        text,
                        ["Negative"]
                    )
                except Exception as e:
                    log.warning("Failed to delete Twi message: %s", e)
            # If not Negative → safe → do nothing in group
        else:
            # Private chat → always reply
            await update.message.reply_text(f"📊 TalkShield Report\nLang: TWI\n{pretty}")

    # ─────────────── ENGLISH HANDLING ───────────────
    else:
        result = classify_english(text)
        pretty = format_english(result)

        # collect harmful labels above threshold
        harmful_labels = [
            k for k, v in result.items() if isinstance(v, float) and v >= DEFAULT_THRESHOLD
        ]

        if is_group:
            if harmful_labels:
                try:
                    await context.bot.delete_message(chat.id, update.message.message_id)
                    await context.bot.send_message(
                        chat.id,
                        f"🚨 A message was removed for toxicity: {', '.join(harmful_labels)}"
                    )
                    # ✅ Log deletion
                    log_deleted_message(
                        chat.id,
                        update.effective_user.id,
                        text,
                        harmful_labels
                    )
                except Exception as e:
                    log.warning("Failed to delete EN message: %s", e)
            # If safe → do nothing in group
        else:
            # Private chat → always reply
            await update.message.reply_text(f"📊 TalkShield Report\nLang: EN\n{pretty}")


from telegram import InputFile

async def getlogs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send the deleted_logs.jsonl file to the bot owner."""
    user_id = update.effective_user.id

    if user_id != BOT_OWNER_ID:
        await update.message.reply_text("⛔ You are not authorized to view logs.")
        return

    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("📭 No logs yet.")
        return

    try:
        with open(LOG_FILE, "rb") as f:
            await update.message.reply_document(
                InputFile(f, filename="deleted_logs.jsonl"),
                caption="📝 Deleted messages log"
            )
    except Exception as e:
        log.error("Failed to send logs: %s", e)
        await update.message.reply_text("⚠️ Failed to send log file.")


# ─────────────────────────────────────────────
# Flask + PTB Application (webhook)
# ─────────────────────────────────────────────
flask_app = Flask(__name__)
application = ApplicationBuilder().token(TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
application.add_handler(CommandHandler("getlogs", getlogs))


# Dedicated event loop for PTB
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
