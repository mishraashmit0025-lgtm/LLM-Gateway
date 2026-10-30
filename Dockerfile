FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY gateway ./gateway
RUN pip install --no-cache-dir .
COPY config.example.yaml config.demo.yaml ./
EXPOSE 8000
CMD ["python", "-m", "gateway", "--config", "/app/config.yaml", "--host", "0.0.0.0", "--port", "8000"]
