# # ---- Stage 1: The Builder ----
# # This stage installs dependencies, including any that need to be compiled.
# FROM python:3.12-alpine AS builder

# # Install build-time system dependencies
# RUN apk add --no-cache build-base

# WORKDIR /app

# # Copy only the requirements file first to leverage Docker's layer caching.
# # This step will only be re-run if requirements.txt changes.
# COPY requirements.txt .

# # Install python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # ---- Stage 2: The Final Image ----
# # This stage creates the clean, final image for production.
# FROM python:3.12-alpine

# # Install runtime system dependencies. ffmpeg is needed by the bot to process audio files.
# RUN apk add --no-cache ffmpeg

# WORKDIR /app

# # Copy the installed packages from the 'builder' stage
# COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# # Copy the application code from the build context (your git repo)
# COPY app.py .
# # COPY .env.example . # It's good practice to include an example env file

# # Set the command to run your bot
# CMD ["python", "app.py"]

# -------------------------------------------------
# Stage 1 – Builder (Debian‑based)
# -------------------------------------------------
# Use the official Python image built on Debian slim.
FROM python:3.12-slim-bookworm AS builder

# Install build‑time system packages.
# - build-essential: compiler, make, libc-dev, etc.
# - gcc, libffi-dev, libssl-dev: common headers needed by many wheels.
#   (Add any additional libs your requirements need, e.g. libpq-dev, libxml2-dev…)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev && \
    # Clean up apt caches to keep the layer small
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage Docker cache: copy only the lockfile/requirements first.
COPY requirements.txt .

# Install Python dependencies into the global site‑packages directory.
# --no-cache-dir avoids keeping the pip download cache.
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------
# Stage 2 – Runtime (Debian‑based)
# -------------------------------------------------
FROM python:3.12-slim-bookworm

# Install only the **runtime** system packages.
# ffmpeg is required for audio handling.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled Python packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy the application code.
COPY app.py .
# COPY .env.example .   # Uncomment if you want to ship an example env file.

# Run the bot.
CMD ["python", "app.py"]