# Use lightweight Python image
FROM python:3.10-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port 10000 for Render
EXPOSE 10000

# Start app with Gunicorn (Render looks for this port)
CMD ["gunicorn", "-b", "0.0.0.0:10000", "bot:app"]
