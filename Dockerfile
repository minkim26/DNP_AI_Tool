# Multi-stage Dockerfile for Solder Joint Classifier
# Stage 1: Build stage with all dependencies
FROM python:3.9-slim AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgtk-3-dev \
    libglib2.0-dev \
    libgdk-pixbuf2.0-dev \
    libpangocairo-1.0-0 \
    libatk1.0-dev \
    libgl1-mesa-dev \
    libxrender-dev \
    pkg-config \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python packages to a specific directory with optimizations
RUN pip install --no-cache-dir --user --no-compile \
    torch==2.0.1 \
    torchvision==0.15.2 \
    opencv-python==4.8.1.78 \
    Pillow==10.0.0 \
    numpy==1.24.3 \
    scikit-learn==1.3.0 \
    scipy==1.11.1 \
    matplotlib==3.7.2 \
    tqdm==4.66.1 \
    joblib==1.3.2 \
    pyyaml \
    sv_ttk \
    && find /root/.local -name "*.pyc" -delete \
    && find /root/.local -name "__pycache__" -type d -exec rm -rf {} + || true

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.9-slim AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:0 \
    PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/root/.local/lib/python3.9/site-packages:$PYTHONPATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install only runtime dependencies in a single optimized layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    x11-apps \
    libgtk-3-0 \
    libglib2.0-0 \
    libgdk-pixbuf2.0-0 \
    libxss1 \
    libxtst6 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Set working directory and create structure in one layer
WORKDIR /app
RUN mkdir -p models input output/crack output/nocrack output/unknown

# Copy application files
COPY inference_gui.py config.yaml ./
COPY dnp_ai_model_pytorch.pth ./models/

# Create startup script inline to avoid extra layer
RUN printf '#!/bin/bash\nexport DISPLAY=${DISPLAY:-:0}\ncd /app\npython inference_gui.py\n' > start_gui.sh && \
    chmod +x start_gui.sh

# Default command
CMD ["./start_gui.sh"]