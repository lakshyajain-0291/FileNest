# Use Python 3.11 slim base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency list
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt


# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy rest of the project files
COPY . /app

# Set PYTHONPATH so src/ is discoverable
ENV PYTHONPATH=/app

# Set default command
CMD ["python", "main.py"]
#CMD ["python", "src/raytune.py"]


