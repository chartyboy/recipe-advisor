#syntax=docker/dockerfile:labs
FROM python:3.11-slim-bookworm as base

# RUN apt-get update --fix-missing && apt-get install -y --fix-missing \
#     build-essential \
#     gcc \
#     g++ && \
#     rm -rf /var/lib/apt/lists/*


FROM base as builder

RUN mkdir /install
WORKDIR /install

ADD https://github.com/chartyboy/recipe-advisor.git /repo
RUN cp /repo/docker/retrieval/requirements.txt requirements.txt

RUN pip install --no-cache-dir --prefix="/install" --ignore-requires-python --pre -r requirements.txt

FROM base as final

ENV PYTHONPATH = /:${PYTHONPATH}

COPY --link --from=builder /install /usr/local
COPY --link --from=builder /repo/src /src

# CMD uvicorn api:app --reload --host retriever --port 8000 --workers 1 --proxy-headers \
#     --app-dir /src/retriever
