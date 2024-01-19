import chromadb
import os
import logging

from contextlib import asynccontextmanager
from functools import lru_cache
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Dict, Any, Annotated

from .auth import verify_key
from src.features.interfaces import EmbeddingFunctionInterface

# Initialize env vars
CHROMA_HOST_ADDRESS = os.getenv("CHROMA_HOST_ADDRESS")


# Initialize model
@lru_cache
async def database():
    # Connect to db
    logging.debug("Connecting to database.")
    chroma_conn = chromadb.HttpClient(host=CHROMA_HOST_ADDRESS)

    logging.info("Successful database connection.")
    return chroma_conn


@lru_cache
async def retriever_model():
    logging.debug("Loading retriever model.")
    model_name = "BAAI/bge-large-en"
    # model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    embed_func = EmbeddingFunctionInterface(hf.embed_documents)
    logging.info("Successfully loaded retriever model.")
    return embed_func


# Initialize model on API startup to avoid delaying loading to first API call
@asynccontextmanager
async def lifespan(
    app: FastAPI,
    conn: Annotated[Any, Depends(database)],
    retriever: Annotated[Any, Depends(retriever_model)],
):
    # database["conn"] = conn
    # database["retriever"] = retriever

    logging.info("Event: Global variables initialized")
    yield
    logging.info("Event: API Shutdown")


# Init API
app = FastAPI(lifespan=lifespan)


class TokenData(BaseModel):
    username: str | None = None


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
    "/retrieve", response_model=RetrieveResponse, dependencies=[Depends(verify_key)]
)
def retrieve(
    input: Annotated[RetrieveRequest, Depends(RetrieveRequest)],
    conn: Annotated[Any, Depends(database)],
    retriever: Annotated[Any, Depends(retriever_model)],
):
    chroma_conn = conn
    embed_func = retriever
    try:
        collection_handle = chroma_conn.get_collection(
            input.collection_name, embedding_function=embed_func
        )
        res = collection_handle.query(
            query_texts=input.query, n_results=input.n_docs, include=["documents"]
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="Collection or items not found")
    docs = res["documents"]
    return RetrieveResponse(input_query=input.query, docs=docs)
