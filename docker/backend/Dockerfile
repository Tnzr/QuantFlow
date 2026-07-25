# Backend API (Python FastAPI — fallback when Bun/Effect-TS not available)
FROM python:3.11-slim-bookworm AS python

WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic httpx

COPY docker/backend/server.py /app/server.py

HEALTHCHECK --interval=15s --timeout=5s --retries=5 \
    CMD curl -f http://localhost:3000/health || exit 1

EXPOSE 3000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3000"]

# ── Bun/Effect-TS target (uncomment when Bun is available)
# FROM oven/bun:latest AS bun
# WORKDIR /app
# COPY backend-effect/package.json backend-effect/bun.lockb ./
# RUN bun install
# COPY backend-effect/src ./src
# EXPOSE 3000
# CMD ["bun", "run", "src/main.ts"]
