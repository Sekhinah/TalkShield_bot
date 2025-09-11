# Use Python 3.10 (stable with our libs)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for some wheels (if needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Render exposes $PORT (we’ll bind gunicorn to it)
ENV PORT=10000

# Start Gunicorn serving our Flask WSGI app named "app" in bot.py
CMD ["gunicorn", "-k", "gthread", "-w", "1", "--threads", "8", "--timeout", "120", "bot:app", "--bind", "0.0.0.0:10000"]
