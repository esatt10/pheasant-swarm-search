# The pheasant-swarm-lab image: the console (UI + HTTP API + MCP) and the CLI
# it launches runs with, in one container.
#
#   docker compose up -d --build          # lab + a bundled pheasant region
#   open http://127.0.0.1:8770            # key: PHEASANT_LAB_CONSOLE_TOKEN
#
# What makes it a Docker deployment rather than a laptop checkout in a box:
#
# * Runs are written to /app/runs, a volume, and every run records
#   `environment.deployment: docker` in its manifest. The console runs with
#   PHEASANT_LAB_RUN_SCOPE=docker, so it lists, reads and deletes only runs
#   made in this deployment - a host directory mounted over /app/runs with
#   laptop runs in it does not leak into the UI or the MCP tools.
# * What the console writes (topics, connections, prices: configs/*.local.yaml
#   on a laptop) goes to /data/local, a volume, because the image's own
#   configs/ is content an upgrade replaces. Stored connection tokens live in
#   /app/runs/.console/secrets.json, 0600.
# * The console binds 0.0.0.0 inside the container, so it refuses to start
#   without PHEASANT_LAB_CONSOLE_TOKEN; Compose publishes it on loopback.

# --------------------------------------------------------------------------
# Stage 1 - the console UI bundle.
# --------------------------------------------------------------------------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

# --------------------------------------------------------------------------
# Stage 2 - the runtime.
# --------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="pheasant-swarm-lab" \
      org.opencontainers.image.description="Hierarchical research swarms against a Pheasant knowledge region: configure, run and compare isolated arms" \
      org.opencontainers.image.source="https://github.com/esatt10/pheasant-swarm-search"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PHEASANT_LAB_DEPLOYMENT=docker \
    PHEASANT_LAB_RUN_SCOPE=docker \
    PHEASANT_LAB_LOCAL_DIR=/data/local \
    PHEASANT_LAB_UI_DIST=/app/ui \
    PHEASANT_LAB_PROMPTS=/app/prompts \
    PHEASANT_LAB_FIXTURES=/app/tests/fixtures/literature

WORKDIR /app

# The package first, so a change to configs or prompts does not reinstall it.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install ".[duckdb]"

COPY configs/ ./configs/
COPY prompts/ ./prompts/
COPY schemas/ ./schemas/
COPY scripts/run_fine_tuning_collection.py ./scripts/run_fine_tuning_collection.py
# The offline literature `fixtures` provider reads; the demo needs nothing else.
COPY tests/fixtures/literature/ ./tests/fixtures/literature/
COPY --from=ui /ui/dist/ ./ui/

RUN useradd --system --uid 10002 --home-dir /app lab \
    && mkdir -p /app/runs /data/local \
    && chown -R lab:lab /app/runs /data

USER lab
VOLUME ["/app/runs", "/data/local"]
EXPOSE 8770

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8770/api/auth', timeout=4).status == 200 else 1)"

ENTRYPOINT ["pheasant-lab"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8770", "--config", "configs/docker.yaml"]
