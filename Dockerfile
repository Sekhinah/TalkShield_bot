# Use official Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip & dependencies
RUN pip install --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install -r requirements.txt

# Expose default port (Render requirement)
EXPOSE 10000

# Start command (same as Procfile)
CMD ["gunicorn", "-k", "gthread", "-w", "1", "--threads", "8", "--timeout", "120", "app_webhook:app"]
