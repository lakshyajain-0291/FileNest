# Base image with PyTorch + CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory inside container
WORKDIR /usr/src/app/image_classifier

# Copy contents from the image_classifier directory on host into container
COPY ./tasks/image_classifier/ .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Optional: Prevent Ray memory monitor errors in Docker
ENV RAY_memory_monitor_refresh_ms=0

# Default command
CMD ["python", "main.py"]
