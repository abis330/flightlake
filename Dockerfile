FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 FLIGHTLAKE_ROOT=/data/lake
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ulog]"
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "flightlake.api:app", "--host", "0.0.0.0", "--port", "8000"]
