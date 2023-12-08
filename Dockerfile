# build
FROM python:3.11 as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    POETRY_HOME="/opt/poetry"
ENV PATH="${POETRY_HOME}/bin:${PATH}"

# Poetry
RUN python -c 'from urllib.request import urlopen; print(urlopen("https://install.python-poetry.org").read().decode())' | python -
RUN poetry config virtualenvs.create false

# Install dependencies
WORKDIR /workspace
COPY pyproject.toml /workspace/
RUN poetry install --no-interaction --no-ansi --no-root -vvv

# prod
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING="UTF-8"

# Copy binaries
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/chainlit /usr/local/bin/

# Install git
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
