# syntax=docker/dockerfile:1

# Stage 1: Astro frontend build
FROM node:20-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
RUN npm run build
# Astro outputs to ../smileornot/static (configured in astro.config.mjs);
# from /frontend that's /smileornot/static.

# Stage 2: Python runtime
FROM python:3.11-slim
RUN useradd -m -u 1000 user
WORKDIR /app

# CPU-only torch wheels — saves ~1.5 GB vs the default cuda-bundled torch.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user pyproject.toml ./
COPY --chown=user smileornot ./smileornot
COPY --chown=user weights ./weights
RUN pip install --no-cache-dir -e .

# Pull the prebuilt static directory from the Node stage.
COPY --from=frontend --chown=user /smileornot/static ./smileornot/static

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/').status==200 else 1)" || exit 1

CMD ["uvicorn", "smileornot.app:app", "--host", "0.0.0.0", "--port", "7860"]
