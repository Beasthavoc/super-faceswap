# Use NVIDIA CUDA base image for GPU acceleration
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    git \
    wget \
    curl \
    unzip \
    cmake \
    build-essential \
    libopencv-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Download and install specific model dependencies
RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install git+https://github.com/facebookresearch/detectron2.git

# Create directories for models and assets
RUN mkdir -p /app/models/face_detection \
    /app/models/face_alignment \
    /app/models/identity_preservation \
    /app/models/inference_engine \
    /app/models/frame_interpolation \
    /app/assets/samples \
    /app/assets/temp

# Copy application code
COPY . .

# Set permissions
RUN chmod +x /app/face_detection/scripts/*.py
RUN chmod +x /app/face_alignment/scripts/*.py
RUN chmod +x /app/identity_preservation/scripts/*.py
RUN chmod +x /app/inference_engine/scripts/*.py
RUN chmod +x /app/frame_interpolation/scripts/*.py

# Download pre-trained models (this will be done at runtime to avoid large image size)
RUN mkdir -p /app/models/downloads

# Expose port for Gradio web interface
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860 || exit 1

# Create non-root user for security
RUN useradd -m -u 1001 faceswap
RUN chown -R faceswap:faceswap /app
USER faceswap

# Default command
CMD ["python", "main.py", "--mode", "gradio", "--host", "0.0.0.0", "--port", "7860"]