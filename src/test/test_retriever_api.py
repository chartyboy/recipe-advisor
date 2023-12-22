import pytest
import numpy as np
import chromadb
import json
import os

from fastapi.testclient import TestClient
from pytest_mock.plugin import MockerFixture
from src import embeddings

from src.retriever.api import app
from src.test.stub_classes import StubChromaClient, StubChromaCollection

client = TestClient(app)


def stub_embed_func(*args, **kwargs):
    return [[1, 2], [3, 4]]


async def override_retriever_dep():
    embed_func = stub_embed_func
    chroma_conn = StubChromaClient()
    return {"embed_func": embed_func, "chroma_conn": chroma_conn}


app.dependency_overrides["database_retriever"] = override_retriever_dep


# Test authentication scheme
# Should fail on incorrect key
# Should pass on correct key
# Token should expire at set time
# Access tokens can be revoked


# Test retrieval endpoint
# Returns descriptive error on: incorrect collection
# Returns number of documents requested
# Returns documents in the form of a list of strings

# Should be able to dynamically reconnect to chroma database
