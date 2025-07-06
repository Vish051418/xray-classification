# Use official PyTorch image as base
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create directories for MLflow and data
RUN mkdir -p /app/mlruns /app/data/processed /app/models

# Expose ports
EXPOSE 5000 8000

# Command to run the training
CMD ["python", "-m", "src.train"]