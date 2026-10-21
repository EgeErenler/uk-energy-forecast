# The forecasting API. Build it, then run it with a trained model mounted:
#   docker build -t energyforecast .
#   docker run -p 8000:8000 -v "$PWD/artifacts:/app/artifacts" -v "$PWD/data/clean:/app/data/clean" energyforecast
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV MLFLOW_DISABLE_AGENT_HINT=1
EXPOSE 8000
CMD ["energyforecast", "serve", "--host", "0.0.0.0", "--port", "8000"]
