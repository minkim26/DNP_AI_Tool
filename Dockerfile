# Multi-stage Dockerfile for Solder Joint Classifier
# Stage 1: Build stage with all dependencies
FROM python:3.9-slim AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies and system packages needed for compilation
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

# Install Python packages to a specific directory
RUN pip install --no-cache-dir --user \
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
    sv_ttk

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.9-slim AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y \
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
    && apt-get clean

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Make Python packages available system-wide
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/root/.local/lib/python3.9/site-packages:$PYTHONPATH

# Set working directory
WORKDIR /app

# Copy application files
COPY inference_gui.py /app/
COPY dnp_ai_model_pytorch.pth /app/
COPY config.yaml /app/

# Create necessary directories and move model file
RUN mkdir -p /app/models /app/input /app/output/crack /app/output/nocrack /app/output/unknown \
    && mv /app/dnp_ai_model_pytorch.pth /app/models/

# Create startup script
RUN echo '#!/bin/bash\n\
export DISPLAY=${DISPLAY:-:0}\n\
cd /app\n\
python inference_gui.py\n\
' > /app/start_gui.sh && chmod +x /app/start_gui.sh

# Default command
CMD ["/app/start_gui.sh"]