FROM python:3.12-slim
WORKDIR /app

# Install apt dependencies
RUN apt-get update && apt-get install -y git curl

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Python dependencies
COPY pyproject.toml poetry.lock /app/
# GitHub token is required for private repositories
RUN --mount=type=secret,id=github_token,required=true \
    umask 077 && \
    trap 'rm -f /root/.netrc' EXIT && \
    printf 'machine github.com login git password %s\n' "$(cat /run/secrets/github_token)" > /root/.netrc && \
    poetry install --no-root --only main,deployment
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and install app
COPY . /app/
RUN poetry install --only-root

ENV DTC_WRAPPER_MODULE_NAME=dtc_super_resolution
ENTRYPOINT ["poetry", "run", "dtc-wrapper"]