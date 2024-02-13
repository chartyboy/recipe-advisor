#syntax=docker/dockerfile:labs
FROM python:3.11-slim-bookworm as base

FROM base as builder

RUN mkdir /install
WORKDIR /install

ADD https://github.com/chartyboy/recipe-advisor.git /repo
RUN cp /repo/docker/streamlit/requirements.txt requirements.txt

RUN pip install --no-cache-dir --prefix="/install" --ignore-requires-python --pre -r requirements.txt

FROM base as final

ENV PYTHONPATH = /:${PYTHONPATH}
ENV PYTHONPATH = /usr/local:${PYTHONPATH}

COPY --link --from=builder /install /usr/local
COPY --link --from=builder /repo/src /src

COPY secrets.toml /.streamlit/secrets.toml