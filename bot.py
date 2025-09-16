from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import requests

SPACE_URL = "https://Sekhinah-talkshield-api.hf.space"
DEFAULT_THRESHOLD = 0.5

app = Flask(__name__)

# ──────────────────────────────
# Helpers
# ──────────────────────────────
def call_space(path: str, payload: dict):
    try:
        r = requests.post(f"{SPACE_URL.rstrip('/')}/{path.lstrip('/')}",
                          json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "raw": getattr(r, "text", "")}

def is_twi_like(text: str) -> bool:
    text_l = text.lower()
    hints = ["ɛ", "ɔ", "wo", "w'", "me", "ɛyɛ", "nsɛm", "waa", "agyimi", "dam", "pɔ"]
    return any(h in text_l for h in hints)

def format_english(scores: dict) -> str:
    if "error" in scores:
        return f"❌ Inference error: {scores['error']}"
    harmful = [k for k, v in scores.items() if isinstance(v, float) and v >= DEFAULT_THRESHOLD]
    harm_line = "None (non_toxic)" if not harmful else ", ".join(harmful)
    lines = [f"• {k}: {scores[k]:.2f}" for k in sorted(scores.keys()) if isinstance(scores[k], float)]
    return f"Labels ≥ {DEFAULT_THRESHOLD:.2f}: {harm_line}\n" + "\n".join(lines)

def format_twi(result: dict) -> str:
    if "error" in result:
        return f"❌ Inference error: {result['error']}"
    pred = result.get("prediction", "?")
    scores = result.get("scores", {})
    lines = [f"• {k}: {scores.get(k, 0):.2f}" for k in ["Negative", "Neutral", "Positive"]]
    return f"Prediction: {pred}\n" + "\n".join(lines)

# ──────────────────────────────
# WhatsApp Webhook
# ──────────────────────────────
@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    msg = request.form.get("Body", "")
    reply = MessagingResponse()

    if not msg.strip():
        reply.message("⚠️ Empty message received.")
        return str(reply)

    # Decide Twi or English
    if is_twi_like(msg):
        result = call_space("/twi", {"text": msg})
        pretty = format_twi(result)
        reply.message(f"📊 TalkShield Report\nLang: TWI\n{pretty}")
    else:
        result = call_space("/english", {"text": msg})
        pretty = format_english(result)
        reply.message(f"📊 TalkShield Report\nLang: EN\n{pretty}")

    return str(reply)

# Optional root route for browser check
@app.route("/")
def index():
    return "✅ WhatsApp TalkShield service is alive", 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)
