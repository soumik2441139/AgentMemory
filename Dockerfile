FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY pyproject.toml setup.py README.md ./
COPY agentmemory/ ./agentmemory/

RUN pip install --no-cache-dir uvicorn fastapi redis openai python-dotenv nltk scikit-learn
RUN pip install --no-cache-dir -e .

COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
