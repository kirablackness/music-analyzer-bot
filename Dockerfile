FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1-dev \
    libsamplerate0-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==2.4.3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir numpy==2.4.3 && \
    pip install --no-cache-dir -r requirements.txt

COPY bot.py .

CMD ["python", "bot.py"]
