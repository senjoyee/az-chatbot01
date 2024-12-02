# Use an official Python runtime as a parent image
FROM python:3.11-bookworm-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PostgreSQL client & other dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
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
    && ldconfig

# Copy wait-for-postgres.sh into the container, fix line endings, and make it executable
COPY ./wait-for-postgres.sh /wait-for-postgres.sh
# Fix Windows line endings (CRLF -> LF)
RUN sed -i 's/\r$//' /wait-for-postgres.sh && \
    chmod +x /wait-for-postgres.sh

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the script to run on container start
ENTRYPOINT ["/wait-for-postgres.sh"]

# Define the command to run the app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]