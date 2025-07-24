# Dockerfile that works with your actual project structure
FROM python:3.9-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:0

# Install system dependencies
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
    libgtk-3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
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

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy your actual files
COPY inference_gui.py /app/
COPY dnp_ai_model_pytorch.pth /app/
COPY config.yaml /app/

# Create necessary directories
RUN mkdir -p /app/models /app/input /app/output /app/output/crack /app/output/nocrack /app/output/unknown

# Move model file to models directory
RUN mv /app/dnp_ai_model_pytorch.pth /app/models/

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create startup script
RUN echo '#!/bin/bash\n\
export DISPLAY=${DISPLAY:-:0}\n\
cd /app\n\
python inference_gui.py\n\
' > /app/start_gui.sh && chmod +x /app/start_gui.sh

# Default command
CMD ["/app/start_gui.sh"]