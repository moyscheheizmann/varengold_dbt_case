# https://github.com/CrafterKolyan/tiny-python-docker-image/tree/main
FROM python:3.11.6-slim-bullseye

RUN apt update
RUN apt install -y --no-install-recommends git
RUN rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENTRYPOINT ["/bin/bash", "-c"]