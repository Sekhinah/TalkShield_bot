from flask import Flask, request

flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return "✅ Service alive", 200

@flask_app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json(force=True, silent=True)
    print("📩 Got update:", data)   # Will show in Render logs
    return "ok", 200

# Gunicorn entrypoint
app = flask_app
