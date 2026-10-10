FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json vite.config.ts ./
COPY frontend ./frontend
RUN npm run build

FROM python:3.11-slim-bookworm AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 EYEWEAR_DATA_DIR=/data EYEWEAR_START_WORKER=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home eyewear && mkdir /data && chown eyewear:eyewear /data
COPY backend ./backend
USER eyewear
EXPOSE 8018
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8018/api/health',timeout=4)"
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8018", "--workers", "1", "--no-proxy-headers"]

FROM caddy:2 AS web
COPY --from=frontend /build/dist /srv
COPY deploy/Caddyfile /etc/caddy/Caddyfile
