import chromadb
import os
import logging

from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Dict, Any, Annotated

from fastapi_simple_security import api_key_router, api_key_security
from src.features.interfaces import EmbeddingFunctionInterface

# Initialize env vars
CHROMA_HOST_ADDRESS = os.getenv("CHROMA_HOST_ADDRESS")
CHROMA_HOST_PORT = os.getenv("CHROMA_HOST_PORT")
EMBED_MODEL_CACHE = os.getenv("EMBED_MODEL_CACHE")

database = dict()

logger = logging.getLogger(__name__)


# Initialize model
@lru_cache
async def connect_database():
    # Connect to db
    logger.debug("Connecting to database.")
    chroma_conn = chromadb.HttpClient(host=CHROMA_HOST_ADDRESS, port=CHROMA_HOST_PORT)

    logger.info("Successful database connection.")
    logger.debug(chroma_conn.list_collections())
    return chroma_conn


@lru_cache
async def load_retriever_model():
    logger.debug("Loading retriever model.")
    model_name = "BAAI/bge-large-en"
    # model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=EMBED_MODEL_CACHE,
    )
    embed_func = EmbeddingFunctionInterface(hf.embed_documents)
    logger.info("Successfully loaded retriever model.")
    return embed_func


async def initialize_database_and_retriever(
    db: Annotated[Any, Depends(connect_database)],
    retriever: Annotated[Any, Depends(load_retriever_model)],
):
    database["conn"] = db
    database["retriever"] = retriever
    return database


# Initialize model on API startup to cache for actual API calls
@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    _ = await connect_database()
    _ = await load_retriever_model()

    logging.info("Event: Global variables initialized")
    yield
    logging.info("Event: API Shutdown")


# Init API
app = FastAPI(lifespan=lifespan)
app.include_router(api_key_router, prefix="/auth", tags=["_auth"])


class RetrieveRequest(BaseModel):
    query: str
    collection_name: str
    n_docs: int


class RetrieveResponse(BaseModel):
    input_query: str
    docs: List[str]


@app.get("/")
def homepage():
    return "RETRIEVER API HOMEPAGE"


@app.get(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(api_key_security)],
)
def retrieve(
    input: Annotated[RetrieveRequest, Depends(RetrieveRequest)],
    database: Annotated[Any, Depends(initialize_database_and_retriever)],
):
    chroma_conn = database["conn"]
    embed_func = database["retriever"]
    try:
        collection_handle = chroma_conn.get_collection(
            input.collection_name, embedding_function=embed_func
        )
        res = collection_handle.query(
            query_texts=input.query, n_results=input.n_docs, include=["documents"]
        )
    except Exception as e:
        logger.debug(e)
        raise HTTPException(status_code=404, detail="Collection or items not found\n")
    logger.debug(res["documents"])
    docs = res["documents"][0]
    return RetrieveResponse(input_query=input.query, docs=docs)
