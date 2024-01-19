FROM python:3.10-slim-bookworm as base

RUN pip install --no-cache-dir --upgrade fastapi==0.104.1 pydantic uvicorn
RUN mkdir /retriever