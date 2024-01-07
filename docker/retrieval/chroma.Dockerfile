FROM chromadb/chroma
RUN touch /chroma-logs/log_config.yml
WORKDIR /chroma
RUN echo "Rebuilding hnsw to ensure architecture compatibility"
RUN pip install --force-reinstall --no-cache-dir chroma-hnswlib
RUN export IS_PERSISTENT=1
RUN export CHROMA_SERVER_NOFILE=65535
# CMD uvicorn chromadb.app:app --workers 1 --host 0.0.0.0 --port 8000 \
#     --proxy-headers --log-config chromadb/log_config.yml --timeout-keep-alive 30