import pytest
import os
import logging

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pytest_mock.plugin import MockerFixture

from src.retriever.api import (
    app,
    connect_database,
    load_retriever_model,
    api_key_security,
    RetrieveResponse,
)
from src.test.stub_classes import (
    StubChromaClient,
    StubChromaCollection,
    StubErrorChromaClient,
)
from src.features.interfaces import EmbeddingFunctionInterface

client = TestClient(app)
auth_secret = "SECRET_TEST"
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module", autouse=True)
def set_env():
    os.environ["CHROMA_HOST_ADDRESS"] = ""
    os.environ["CHROMA_HOST_PORT"] = ""
    os.environ["EMBED_MODEL_CACHE"] = ""
    os.environ["FASTAPI_SIMPLE_SECURITY_SECRET"] = auth_secret
    os.environ["FASTAPI_SIMPLE_SECURITY_DB_LOCATION"] = "./sqlite.db"

    yield
    os.remove("./sqlite.db")


@pytest.fixture
def secret_header():
    return {"secret-key": auth_secret}


@pytest.fixture
def valid_key(secret_header):
    new_key_payload = {"name": "test_valid_api_key"}
    valid_key_response = client.get(
        "/auth/new", headers=secret_header, params=new_key_payload
    )
    key = valid_key_response.json()
    return key


@pytest.fixture
def query_params():
    return {
        "query": "test",
        "collection_name": "test_collection",
        "n_docs": 1,
    }


def stub_embed_func(*args, **kwargs):
    return [[1, 2], [3, 4]]


async def override_database():
    return StubChromaClient()


async def override_retriever():
    embed_func = EmbeddingFunctionInterface(stub_embed_func)
    return embed_func


async def override_api_key_security():
    return True


app.dependency_overrides[connect_database] = override_database
app.dependency_overrides[load_retriever_model] = override_retriever
app.dependency_overrides[api_key_security] = override_api_key_security

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

    def test_retrieve_document_type(self, query_params):
        response = client.get("/retrieve", params=query_params)
        content = response.json()
        assert isinstance(content["docs"][0], str)

    def test_retrieve_nonexistent_collection_error(self, query_params):
        async def override_error_database():
            return StubErrorChromaClient()

        app.dependency_overrides[connect_database] = override_error_database
        response = client.get("/retrieve", params=query_params)
        assert response.status_code == 404
        app.dependency_overrides[connect_database] = override_database


class TestAuthSchema:
    @classmethod
    def setup_class(cls):
        app.dependency_overrides[api_key_security] = api_key_security

    @classmethod
    def teardown_class(cls):
        app.dependency_overrides[api_key_security] = override_api_key_security

    def test_fail_on_no_key(self, query_params):
        response = client.get("/retrieve", params=query_params)
        assert response.status_code == 403
        assert "docs" not in response.json().keys()

    def test_fail_on_invalid_key(self, query_params):
        invalid_key_header = {"api-key": "invalid-key"}

        response = client.get(
            "/retrieve", headers=invalid_key_header, params=query_params
        )

        assert response.status_code == 403
        assert "docs" not in response.json().keys()

    def test_pass_on_valid_key(self, valid_key, query_params):
        query_header = {"api-key": valid_key}

        response = client.get("/retrieve", headers=query_header, params=query_params)
        content = response.json()

        assert response.status_code == 200
        assert len(content["docs"]) == 1
        assert isinstance(content["docs"][0], str)

    def test_token_expiry(self, valid_key, query_params, secret_header):
        # Setup expired token
        query_header = {"api-key": valid_key}
        renew_payload = {"api-key": valid_key, "expiration-date": "2022-07-15"}
        client.get("/auth/renew", headers=secret_header, params=renew_payload)

        response = client.get("/retrieve", headers=query_header, params=query_params)

        assert response.status_code == 403
        assert "docs" not in response.json().keys()

    def test_token_revoking(self, valid_key, query_params, secret_header):
        revoke_payload = {"api-key": valid_key}
        query_header = revoke_payload
        client.get("/auth/revoke", headers=secret_header, params=revoke_payload)

        response = client.get("/retrieve", headers=query_header, params=query_params)

        assert response.status_code == 403
        assert "docs" not in response.json().keys()
