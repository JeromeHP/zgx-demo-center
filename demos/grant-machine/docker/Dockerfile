FROM python:3.12-slim
ARG UV_VERSION=0.12.8
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir "uv==${UV_VERSION}" \
 && useradd --create-home --uid 10001 demo
WORKDIR /opt/grant-machine
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY app ./app
COPY web ./web
COPY config ./config
COPY data ./data
RUN mkdir -p runtime reports data/replays && chown -R demo:demo runtime reports data/replays
USER demo
ENV GM_SERVER__HOST=0.0.0.0 GM_SERVER__PORT=8095
HEALTHCHECK --interval=15s --timeout=3s --retries=5 CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8095/api/health', timeout=2)" || exit 1
CMD ["/opt/grant-machine/.venv/bin/python", "-m", "app.main"]
