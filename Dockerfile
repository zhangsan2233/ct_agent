FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY chestct_agent ./chestct_agent
COPY demo ./demo
COPY scripts ./scripts
COPY README.md .env.example ./

EXPOSE 8080 8501
CMD ["uvicorn", "chestct_agent.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
