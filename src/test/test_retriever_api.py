import pytest

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pytest_mock.plugin import MockerFixture

from src.retriever.api import app, database, retriever_model, verify_key
from src.test.stub_classes import (
    StubChromaClient,
    StubChromaCollection,
    StubErrorChromaClient,
)
from src.features.interfaces import EmbeddingFunctionInterface

client = TestClient(app)


def stub_embed_func(*args, **kwargs):
    return [[1, 2], [3, 4]]


async def override_database():
    return StubChromaClient()


async def override_retriever():
    embed_func = EmbeddingFunctionInterface(stub_embed_func)
    return embed_func


async def override_verify_key():
    return True


app.dependency_overrides[database] = override_database
app.dependency_overrides[retriever_model] = override_retriever
app.dependency_overrides[verify_key] = override_verify_key

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
class TestRetrieveEndpoint:
    def test_retrieve_document_quantity(self):
        expected_number_of_documents = 4
        response = client.get(
            f"/retrieve?query=test&collection_name=test&n_docs={expected_number_of_documents}"
        )
        content = response.json()
        assert len(content["docs"]) == expected_number_of_documents

    def test_retrieve_document_type(self):
        response = client.get(f"/retrieve?query=test&collection_name=test&n_docs=1")
        content = response.json()
        assert isinstance(content["docs"][0], str)

    def test_retrieve_nonexistent_collection_error(self):
        async def override_error_database():
            return StubErrorChromaClient()

        app.dependency_overrides[database] = override_error_database
        response = client.get(
            f"/retrieve?query=test&collection_name=error_collection&n_docs=1"
        )
        assert response.status_code == 404
        app.dependency_overrides[database] = override_database


class TestAuthSchema:
    def test_fail_on_invalid_key(self):
        pass

    def test_pass_on_valid_key(self):
        pass

    def test_token_expiry(self):
        pass

    def test_token_revoking(self):
        pass
