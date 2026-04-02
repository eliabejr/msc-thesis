# ── Stage 1: dependency builder ──────────────────────────────────────────────
# Uses a slim Debian image to compile all Python packages once,
# then copies only the installed files into the final image (smaller result).
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries needed to compile numpy / scipy / cvxpy / osqp
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        pkg-config \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix so we can copy it cleanly
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="MSc Thesis – Shu et al. 2024 replication"
LABEL description="JM-XGB Dynamic Asset Allocation framework"

# Runtime system libraries (only what's needed to *run*, not compile)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash researcher
USER researcher
WORKDIR /home/researcher/app

# Copy project source
COPY --chown=researcher:researcher . .

# Pre-create directories that will be used at runtime
RUN mkdir -p data/raw results

# Expose Jupyter port
EXPOSE 8888

# Default: launch Jupyter Lab (can be overridden with `docker run ... python pipeline.py`)
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''", \
     "--notebook-dir=/home/researcher/app/notebooks"]
