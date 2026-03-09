# Simple Dockerfile for Verifixia backend
# This image installs Python, dependencies, and copies pretrained model

FROM python:3.12-slim

# avoid pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# copy requirements
COPY requirements.txt ./

# install base deps (pytorch example: CPU-only)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# copy source code
COPY . /app

# ensure uploads and models directories exist
RUN mkdir -p uploads models

# if you want to bake pretrained models into image, place them under models/ before building
# alternatively, set MODEL_URL & SKLEARN_URL as env vars and the server will download on startup

EXPOSE 3001

CMD ["python", "app.py"]
