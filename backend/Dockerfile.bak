# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000

# Set the working directory
WORKDIR /app

# Install system dependencies
# These are less likely to change, so we put them first
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libreoffice \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && ldconfig

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN adduser --disabled-password --gecos "" appuser

# Copy the application code
# This layer changes frequently, so we put it last
COPY . .

# Change ownership of the application files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Make port 5000 available
EXPOSE $PORT

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]