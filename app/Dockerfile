FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	BROWSER_PATH=/usr/bin/chromium \
	CHROME_PATH=/usr/bin/chromium \
	UV_LINK_MODE=copy \
	UV_PYTHON_DOWNLOADS=never \
	UV_PROJECT_ENVIRONMENT=/opt/venv \
	PATH=/opt/venv/bin:$PATH

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

RUN apt-get update && \
	apt-get install -y --no-install-recommends chromium && \
	rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
