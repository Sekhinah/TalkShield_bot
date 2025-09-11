import os
import logging
import nest_asyncio
import asyncio
from flask import Flask, request
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# Patch event loop (for async inside Flask)
nest_asyncio.apply()

# Logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TalkShield")

# Flask app (for webhook)
app = Flask(__name__)

# Bot setup
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("❌ TELEGRAM_BOT_TOKEN not set")

telegram_app = Application.builder().token(TOKEN).build()

# Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✅ Bot is alive on Render!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"You said: {update.message.text}")

telegram_app.add_handler(CommandHandler("start", start))
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

# Flask route for Telegram webhook
@app.route(f"/webhook/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), telegram_app.bot)
    asyncio.get_event_loop().create_task(telegram_app.process_update(update))
    return "ok", 200

# Health check
@app.route("/", methods=["GET"])
def home():
    return "TalkShield bot running on Render!", 200
