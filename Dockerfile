# Use Python 3.10 (safer than 3.13 for now)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

# Expose port for Render
EXPOSE 10000

# Start Gunicorn with Flask app inside bot.py
CMD ["gunicorn", "-k", "gthread", "-w", "1", "--threads", "8", "--timeout", "120", "bot:app"]
