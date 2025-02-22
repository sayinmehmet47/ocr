# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx libgomp1 libgeos-dev libopenblas-dev && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a dummy image for model initialization
RUN mkdir -p /root/.paddleocr/whl && python -c "import numpy as np; import cv2; img = np.zeros((100,100,3), np.uint8); cv2.imwrite('dummy.jpg', img); from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=True, lang='german', show_log=False); ocr.ocr('dummy.jpg', cls=True)"

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
