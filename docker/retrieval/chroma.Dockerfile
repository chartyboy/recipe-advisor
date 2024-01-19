FROM python:3.10-slim-bookworm as base

RUN apt-get update --fix-missing && apt-get install -y --fix-missing \
    build-essential \
    gcc \
    g++ && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir chromadb
COPY ./echo_collections.py /