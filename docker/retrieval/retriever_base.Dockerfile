FROM python:3.10-slim-bookworm as base

RUN apt-get update --fix-missing && apt-get install -y --fix-missing \
    build-essential \
    gcc \
    g++ && \
    rm -rf /var/lib/apt/lists/*


FROM base as builder

RUN mkdir /install
WORKDIR /install

ADD https://github.com/chartyboy/recipe-advisor.git /repo

RUN cp /repo/recipe-advisor/requirements.txt requirements.txt
RUN pip install --no-cache-dir --prefix="/install" -r requirements.txt

FROM base as final

ENV PYTHONPATH = /:${PYTHONPATH}

COPY --link --from=builder /install /usr/local
COPY --link --from=builder /repo/recipe-advisor/src /src

CMD uvicorn api:app --reload --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers \
    --app-dir /src/retriever
