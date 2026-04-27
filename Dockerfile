# adaptmem daemon — production container.
#
# Build:
#     docker build -t adaptmem:latest .
#
# Run (no auth, localhost only):
#     docker run --rm -p 7800:7800 adaptmem:latest
#
# Run (with auth + persistent HF cache):
#     docker run --rm -p 7800:7800 \
#         -e ADAPTMEM_API_KEY=secret \
#         -v adaptmem_hf:/home/adaptmem/.cache/huggingface \
#         adaptmem:latest
#
# Run with mTLS (mount cert files):
#     docker run --rm -p 7800:7800 \
#         -v $(pwd)/certs:/certs:ro \
#         adaptmem:latest serve \
#         --ssl-keyfile /certs/server.key \
#         --ssl-certfile /certs/server.crt \
#         --ssl-ca-certs /certs/ca.crt
#
# Two-stage build keeps the runtime image small (no compiler / build deps).

# ---- Stage 1: deps ------------------------------------------------------
FROM python:3.12-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY adaptmem ./adaptmem

# Install with [server] extras (FastAPI + uvicorn + pydantic).
# `--target` puts site-packages somewhere we can copy verbatim.
RUN pip install --target=/install ".[server]"

# ---- Stage 2: runtime ---------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user for security (production deployments should never run as root).
RUN groupadd --system adaptmem && \
    useradd --system --gid adaptmem --create-home --home-dir /home/adaptmem adaptmem

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/usr/local/lib/python-packages \
    PATH="/usr/local/lib/python-packages/bin:${PATH}" \
    HOME=/home/adaptmem

# Copy installed packages from the deps stage.
COPY --from=deps /install /usr/local/lib/python-packages

USER adaptmem
WORKDIR /home/adaptmem

# Default: bind 0.0.0.0 so the container is reachable from outside.
# Override with explicit args at `docker run` time when you need TLS / auth.
EXPOSE 7800
ENTRYPOINT ["python", "-m", "adaptmem.cli"]
CMD ["serve", "--host", "0.0.0.0", "--port", "7800"]

# Healthcheck via /healthz (Docker auto-restarts on failure if --restart=always).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7800/healthz', timeout=3).status == 200 else 1)"
