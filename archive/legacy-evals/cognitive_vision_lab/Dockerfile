# Cognitive Vision Lab v2.0
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching
COPY cognitive_vision_lab/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The lab reads checkpoints/ + results from the repo, so copy the whole tree.
COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "cognitive_vision_lab/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
