import chromadb
import os
import logging
import sys
from dotenv import load_dotenv

# For non-container deployment
load_dotenv()

from contextlib import asynccontextmanager
from functools import lru_cache
from urllib.parse import urljoin, unquote_plus
from fastapi import FastAPI, Depends, HTTPException, Query
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Dict, Any, Annotated

from fastapi_simple_security import api_key_router, api_key_security
from src.features.interfaces import EmbeddingFunctionInterface

# For non-container deployment
load_dotenv()

# Initialize env vars
CHROMA_HOST_ADDRESS = os.getenv("CHROMA_HOST_ADDRESS")
CHROMA_HOST_PORT = os.getenv("CHROMA_HOST_PORT")

EMBED_MODEL_CACHE = os.getenv("EMBED_MODEL_CACHE")
if str(EMBED_MODEL_CACHE).lower() == "false":
    EMBED_MODEL_CACHE = None

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

database = dict()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Initialize model
@lru_cache
def connect_database():
    # Connect to db
    logger.debug("Connecting to database.")
    chroma_conn = chromadb.HttpClient(host=CHROMA_HOST_ADDRESS, port=CHROMA_HOST_PORT)

    logger.info("Successful database connection.")
    logger.debug(chroma_conn.list_collections())
    return chroma_conn


@lru_cache
def load_retriever_model():
    logger.debug("Loading retriever model.")
    model_name = EMBEDDING_MODEL_NAME
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        cache_folder=EMBED_MODEL_CACHE,
    )
    embed_func = EmbeddingFunctionInterface(hf.embed_documents)
    logger.info(f"Successfully loaded retriever model. Model name:{model_name}")
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
    _ = connect_database()
    _ = load_retriever_model()
    logger.info("Event: Global variables initialized")
    yield
    logger.info("Event: API Shutdown")


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
    ids: List[str]


class GetRequest(BaseModel):
    ids: List[str]
    collection_name: str


class GetResponse(BaseModel):
    ids: List[str]
    docs: List[str]


@app.get("/")
def homepage():
    return "RETRIEVER API HOMEPAGE"


@app.post(
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
            query_texts=unquote_plus(input.query),
            n_results=input.n_docs,
            include=["documents"],
        )
        logger.debug(res)
    except Exception as e:
        logger.debug(e)
        raise HTTPException(status_code=404, detail="Collection or items not found\n")
    docs = res["documents"][0]
    ids = res["ids"][0]
    return RetrieveResponse(input_query=input.query, docs=docs, ids=ids)


@app.post(
    "/documents", response_model=GetResponse, dependencies=[Depends(api_key_security)]
)
def get_by_id(
    ids: Annotated[list[str], Query()],
    collection_name: Annotated[str, Query()],
    database: Annotated[Any, Depends(initialize_database_and_retriever)],
):
    chroma_conn = database["conn"]
    embed_func = database["retriever"]
    # logger.debug(f"IDs:{input.ids}")
    try:
        collection_handle = chroma_conn.get_collection(
            collection_name, embedding_function=embed_func
        )
        res = collection_handle.get(
            ids=ids,
            include=["documents"],
        )
        logger.debug(res)
    except Exception as e:
        logger.debug(e)
        raise HTTPException(status_code=404, detail="Collection or items not found\n")
    docs = res["documents"]
    return GetResponse(ids=ids, docs=docs)
