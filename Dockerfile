# Use official Python base
FROM python:3.10-slim

# Workdir inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 10000

# Start using gunicorn
CMD ["gunicorn", "-k", "gthread", "-w", "1", "--threads", "8", "--timeout", "120", "app_webhook:app"]
