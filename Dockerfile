# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to cache the pip install step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY . .

# Set environment variables so Python can find your 'src' folder
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Change this to whatever script actually starts your app
CMD ["python", "scripts/run_dashboard.py"]
