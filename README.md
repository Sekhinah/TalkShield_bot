 TalkShield Bot

## Overview
TalkShield is a moderation bot that helps keep conversations safe by detecting:
- English → Toxicity (threats, insults, obscene, etc.)
- Twi (Akan) → Sentiment (Positive, Neutral, Negative)

It integrates with **Telegram**, deletes toxic group messages, alerts members, and logs violations.

---

## Features
  Group moderation → Deletes toxic messages & sends alerts  
  Private analysis → Reports scores in private chat  
  Logging → Stores deleted messages in `deleted_logs.jsonl`  
  Queuing → Limits concurrent Hugging Face API calls  
  Owner tools → `/getlogs` to download moderation logs  

---

 Setup
 Clone repo 
   ```bash
   git clone https://github.com/Sekhinah/TalkShield_bot.git
   cd TalkShield_bot

Install dependencies
pip install -r requirements.txt

Set environment variables
TELEGRAM_BOT_TOKEN = <your-telegram-bot-token>
SPACE_URL = https://Sekhinah/talkshield-api.hf.space
TALKSHIELD_THRESHOLD = 0.50
HF_MAX_CONCURRENCY = 2
BOT_OWNER_ID = <your-telegram-user-id>

Run locally
python bot.py

 Deployment
Render → host the Telegram bot
Hugging Face Spaces → serve the ML models

 Models
Sekhinah/Talk_Shield_English
Sekhinah/Talk_Shield








