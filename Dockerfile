# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

COPY app/ ./app/
COPY tests/ ./tests/
COPY verify/ ./verify/

# Image build manifest, validated by the one-shot `verify` service.
ARG APP_VERSION=1.0.0
ENV APP_VERSION=$APP_VERSION
RUN printf '{"name": "nifti-sampler", "version": "%s", "python": ">=3.11"}\n' \
        "$APP_VERSION" > /app/image-manifest.json \
    && python -m compileall -q app tests verify

USER nobody
EXPOSE 8000

CMD ["python", "-m", "app.server"]
