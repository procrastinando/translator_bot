# -------------------------------------------------
# Stage 1 – Builder
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
# Stage 2 – Runtime
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

# Run the bot.
CMD ["python", "app.py"]