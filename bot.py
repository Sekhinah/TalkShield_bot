import os
import re
import sys
import logging
import asyncio
from typing import Dict, List, Tuple, Optional

# Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional Jupyter loop patch
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from deep_translator import GoogleTranslator
from langdetect import detect

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder, Application, CommandHandler, MessageHandler, ContextTypes, filters
)
print('we are here1')
# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO
)
log = logging.getLogger("TalkShield")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
ENG_MODEL_ID = "Sekhinah/Talk_Shield_English"  # 7-label toxicity (multi-label)
TWI_MODEL_ID = "Sekhinah/Talk_Shield"          # 3-class sentiment

ENG_LABELS = [
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "non_toxic",
]
HARMFUL_ENG = ENG_LABELS[:-1]  # exclude non_toxic

TWI_LABELS = ["Negative", "Neutral", "Positive"]

DEFAULT_THRESHOLD = float(os.getenv("TALSHIELD_THRESHOLD", "0.50"))
ADMIN_CHAT_ID = os.getenv("ADMIN_CHAT_ID", "").strip() or None  # optional moderation log

# Twi typed-text hints (no diacritics)
WORD_RE = re.compile(r"[A-Za-z0-9'’]+")
TWI_HINT_STRINGS = {
    "agyimi", "wagyimi", "w'agyimi",
    "ankasa", "koraa", "paa", "hw3", "hwe", "s3", "se",
    "biaa", "dabi", "medaase", "p3", "y3", "woara", "nti",
}

# ──────────────────────────────────────────────────────────────────────────────
# Lazy models (PUBLIC HF: no token needed)
# ──────────────────────────────────────────────────────────────────────────────
_eng_tok: Optional[AutoTokenizer] = None
_eng_mdl: Optional[AutoModelForSequenceClassification] = None
_twi_tok: Optional[AutoTokenizer] = None
_twi_mdl: Optional[AutoModelForSequenceClassification] = None


def load_english():
    global _eng_tok, _eng_mdl
    if _eng_tok is None or _eng_mdl is None:
        log.info("Loading English model from Hub…")
        _eng_tok = AutoTokenizer.from_pretrained(ENG_MODEL_ID)
        _eng_mdl = AutoModelForSequenceClassification.from_pretrained(ENG_MODEL_ID).eval()
        log.info("English model ready.")


def load_twi():
    global _twi_tok, _twi_mdl
    if _twi_tok is None or _twi_mdl is None:
        log.info("Loading Twi model from Hub…")
        _twi_tok = AutoTokenizer.from_pretrained(TWI_MODEL_ID)
        _twi_mdl = AutoModelForSequenceClassification.from_pretrained(TWI_MODEL_ID).eval()
        log.info("Twi model ready.")

# ──────────────────────────────────────────────────────────────────────────────
# Language detection
# ──────────────────────────────────────────────────────────────────────────────
def detect_lang_google_first(text: str) -> str:
    """Prefer Google translator detection; fallback to heuristics/langdetect; normalize to 'en' or 'ak'."""
    try:
        lg = (GoogleTranslator().detect(text) or "").lower()
        if lg in ("ak", "tw", "akan", "twi"):
            return "ak"
        if lg == "en":
            return "en"
    except Exception:
        pass

    low = text.lower()
    if "3" in low:  # hw3, b3y3
        return "ak"
    toks = set(m.group(0).lower() for m in WORD_RE.finditer(low))
    if toks & TWI_HINT_STRINGS:
        return "ak"

    try:
        lg2 = (detect(text) or "").lower()
        if lg2 in ("ak", "tw", "akan", "twi"):
            return "ak"
        if lg2 == "en":
            return "en"
    except Exception:
        pass

    return "en"

# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────
def classify_english(text: str) -> Dict[str, float]:
    """Multi-label toxicity with sigmoid. Returns dict[label -> prob]."""
    load_english()
    inputs = _eng_tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = _eng_mdl(**inputs).logits  # [1, 7]
        probs = torch.sigmoid(logits).squeeze().tolist()
    if not isinstance(probs, list):
        probs = [float(probs)]
    return {ENG_LABELS[i]: float(probs[i]) for i in range(len(ENG_LABELS))}


def classify_twi(text: str) -> Tuple[Dict[str, float], str]:
    """3-class sentiment with softmax. Returns (dict[label -> prob], predicted_label)."""
    load_twi()
    inputs = _twi_tok(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = _twi_mdl(**inputs).logits  # [1, 3]
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        idx = int(torch.argmax(logits, dim=-1))
    prob_dict = {TWI_LABELS[i]: float(probs[i]) for i in range(len(TWI_LABELS))}
    return prob_dict, TWI_LABELS[idx]


def format_report_en(text: str, prob_dict: Dict[str, float], threshold: float) -> str:
    prob_lines = "\n".join(f"- {k}: {v:.3f}" for k, v in prob_dict.items())
    picked = [lbl for lbl in HARMFUL_ENG if prob_dict.get(lbl, 0.0) >= threshold]
    if not picked:
        picked = ["non_toxic"]
    return (
        "📊 TalkShield Report\n"
        f"Message: {text}\n\n"
        "Language: EN\n\n"
        f"{prob_lines}\n\n"
        f"Labels ≥ {threshold:.2f}: {', '.join(picked)}"
    )


def format_report_twi(text: str, prob_dict: Dict[str, float], pred: str) -> str:
    prob_lines = "\n".join(f"- {k}: {v:.3f}" for k, v in prob_dict.items())
    return (
        "📊 TalkShield Report\n"
        f"Message: {text}\n\n"
        "Language: TWI (AK)\n\n"
        f"{prob_lines}\n\n"
        f"Predicted: {pred}"
    )

print('we are here2')
def route_and_classify_structured(
    text: str, user_lang_pref: str, threshold: float
) -> Dict:
    """
    Returns structured dict:
    {
      'lang': 'en'|'ak',
      'model_id': str,
      'probs': {label: prob},
      'picked': [...],        # EN harmful labels over threshold (or ['non_toxic'])
      'pred': 'Negative|Neutral|Positive'  # Twi only
      'report': 'string to print'
    }
    """
    if user_lang_pref in ("ak", "en"):
        lang = user_lang_pref
    else:
        lang = detect_lang_google_first(text)

    if lang == "ak":
        probs, pred = classify_twi(text)
        report = format_report_twi(text, probs, pred)
        return {"lang": "ak", "model_id": TWI_MODEL_ID, "probs": probs, "picked": [], "pred": pred, "report": report}

    probs = classify_english(text)
    picked = [lbl for lbl in HARMFUL_ENG if probs.get(lbl, 0.0) >= threshold] or ["non_toxic"]
    report = format_report_en(text, probs, threshold)
    return {"lang": "en", "model_id": ENG_MODEL_ID, "probs": probs, "picked": picked, "pred": "", "report": report}

# ──────────────────────────────────────────────────────────────────────────────
# Telegram handlers
# ──────────────────────────────────────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thr = context.chat_data.get("threshold", context.user_data.get("threshold", DEFAULT_THRESHOLD))
    lang_pref = context.user_data.get("lang_pref", "auto")
    await update.message.reply_text(
        "Hi! I’m Talk_Shield.\n"
        "• English → multi-label toxicity\n"
        "• Twi (Akan) → sentiment (Negative/Neutral/Positive)\n\n"
        f"Commands:\n"
        f"/threshold 0.6 — set toxicity threshold for this chat (now {thr:.2f})\n"
        f"/lang ak|en|auto — force language for *your* messages (now {lang_pref.upper()})\n"
        "/status — show current settings\n"
        "/help — show help"
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thr = context.chat_data.get("threshold", context.user_data.get("threshold", DEFAULT_THRESHOLD))
    lang_pref = context.user_data.get("lang_pref", "auto")
    await update.message.reply_text(
        "*How it works*\n"
        "1) Detect language (Google → Twi-SMS hints → langdetect)\n"
        "2) Route to Twi sentiment or English toxicity models\n"
        "3) Return probabilities; in groups delete if any score ≥ threshold\n\n"
        f"Threshold: *{thr:.2f}*\n"
        f"Language mode: *{lang_pref.upper()}*",
        parse_mode=ParseMode.MARKDOWN
    )


async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thr_chat = context.chat_data.get("threshold")
    thr_user = context.user_data.get("threshold")
    thr = thr_chat if thr_chat is not None else (thr_user if thr_user is not None else DEFAULT_THRESHOLD)
    lang_pref = context.user_data.get("lang_pref", "auto")
    await update.message.reply_text(
        f"Status\n"
        f"- Chat type: {update.effective_chat.type}\n"
        f"- Threshold (this chat): {thr:.2f}\n"
        f"- Your language preference: {lang_pref.upper()}\n"
        f"- ENG model: {ENG_MODEL_ID}\n"
        f"- TWI model: {TWI_MODEL_ID}"
    )


async def threshold_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    is_group = update.effective_chat.type in ("group", "supergroup")
    current = context.chat_data.get("threshold", context.user_data.get("threshold", DEFAULT_THRESHOLD))
    if not context.args:
        await update.message.reply_text(f"Current threshold: {current:.2f}\nUsage: /threshold 0.5")
        return
    try:
        val = float(context.args[0])
        if 0.10 <= val <= 0.90:
            if is_group:
                context.chat_data["threshold"] = val
            else:
                context.user_data["threshold"] = val
            await update.message.reply_text(f"✅ Threshold set to {val:.2f} ({'group' if is_group else 'you only'})")
        else:
            await update.message.reply_text("❌ Choose a value between 0.10 and 0.90.")
    except ValueError:
        await update.message.reply_text("❌ That didn’t look like a number. Example: /threshold 0.5")


async def lang_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    arg = (context.args[0].lower() if context.args else "auto")
    if arg not in ("ak", "en", "auto"):
        await update.message.reply_text("Use: /lang ak | /lang en | /lang auto")
        return
    context.user_data["lang_pref"] = arg
    await update.message.reply_text(f"✅ Language mode set to: {arg.upper()} (for your messages)")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user and update.effective_user.is_bot:
        return

    text = (update.message.text or "").strip()
    if not text:
        return

    if update.effective_chat.type in ("group", "supergroup"):
        thr = context.chat_data.get("threshold", DEFAULT_THRESHOLD)
    else:
        thr = context.user_data.get("threshold", DEFAULT_THRESHOLD)

    lang_pref = context.user_data.get("lang_pref", "auto")

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    try:
        result = await asyncio.to_thread(route_and_classify_structured, text, lang_pref, thr)
    except Exception as e:
        log.exception("Inference error")
        await update.message.reply_text(f"⚠️ Error while classifying: {e}")
        return

    chat_type = update.effective_chat.type

    # DMs → reply with full report
    if chat_type == "private":
        await update.message.reply_text(result["report"])
        return

    # Groups → moderation
    probs = result["probs"]
    delete_flag = False

    if result["lang"] == "en":
        delete_flag = any(probs.get(lbl, 0.0) >= thr for lbl in HARMFUL_ENG)
    else:  # 'ak'
        delete_flag = probs.get("Negative", 0.0) >= thr

    if delete_flag:
        try:
            await update.message.delete()
            log.info(f"Deleted message in group (thr={thr:.2f}): {text}")
            if ADMIN_CHAT_ID:
                await context.bot.send_message(
                    chat_id=int(ADMIN_CHAT_ID),
                    text=(
                        "🧹 *TalkShield Moderation*\n"
                        f"- Chat: {update.effective_chat.title or update.effective_chat.id}\n"
                        f"- User: {update.effective_user.full_name} (@{update.effective_user.username})\n"
                        f"- Lang: {result['lang'].upper()}\n"
                        f"- Threshold: {thr:.2f}\n"
                        f"- Scores: { {k: round(v,3) for k,v in probs.items()} }\n"
                        f"- Text: {text}"
                    ),
                    parse_mode=ParseMode.MARKDOWN
                )
        except Exception as e:
            log.error(f"Failed to delete message: {e}")


async def handle_nontext(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat.type == "private":
        await update.message.reply_text("Please send text. Media/files aren’t supported yet.")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
async def main_async():
    # Read from env (or .env via python-dotenv at the top of the file)
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN environment variable.")

    app: Application = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("threshold", threshold_cmd))
    app.add_handler(CommandHandler("lang", lang_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(~filters.TEXT & ~filters.COMMAND, handle_nontext))

    log.info("Talk_Shield bot starting…")
    await app.run_polling()


def main():
    # Terminal vs Jupyter
    if "ipykernel" in sys.modules:
        return asyncio.create_task(main_async())
    else:
        asyncio.run(main_async())


if __name__ == "__main__":
    main()

