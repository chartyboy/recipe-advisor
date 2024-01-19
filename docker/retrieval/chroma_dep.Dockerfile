FROM chromadb/chroma

# WORKDIR /chroma
# RUN echo "Rebuilding hnsw to ensure architecture compatibility"
# RUN pip install --force-reinstall --no-cache-dir chroma-hnswlib
# RUN export IS_PERSISTENT=1
# RUN export PERSIST_DIRECTORY='/chroma/chroma'
# RUN export CHROMA_SERVER_NOFILE=65535
# CMD uvicorn chromadb.app:app --workers 1 --host database --port 8000 \
#     --proxy-headers --log-config chromadb/log_config.yml --timeout-keep-alive 30