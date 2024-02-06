import pytest
import os
import logging
import requests
import atexit

import streamlit as st

from datetime import datetime, timedelta
from streamlit.testing.v1 import AppTest

from .stub_classes import MockResponse, StubLLM

from src.streamlit import connections

# from src.streamlit import test_app

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def mock_response(monkeypatch):
    def _mock_response(json_response=dict(), status_code=200, method="get"):
        def mock_get(*args, **kwargs):
            return MockResponse(json_response, status_code)

        monkeypatch.setattr(requests, method, mock_get)

    return _mock_response


@pytest.fixture
def chroma_instance() -> connections.ChromaConnection:
    return connections.ChromaConnection(base_url="test", api_secret="test_secret")


@pytest.fixture
def streamlit_test_instance():
    at = AppTest.from_file("../src/streamlit/app.py")
    at.secrets["RETRIEVER_API_BASE"] = "test"
    at.secrets["RETRIEVER_API_SECRET"] = "test-secret"
    at.secrets["OPENAI_API_KEY"] = "test-key"
    at.secrets["TEST_ENV"] = True
    return at


@pytest.mark.usefixtures("chroma_instance")
class TestChromaConnection:
    # Test method to get new key from retriever auth
    def test_chroma_connection_get_api_key(
        self,
        chroma_instance: connections.ChromaConnection,
        mock_response,
    ):
        expected_api_response = "test-key"
        expected_status_code = 200
        mock_response(expected_api_response, expected_status_code, "get")
        key = chroma_instance.get_connection()

        assert isinstance(key, str)

    def test_retrieve_documents_dict_on_success(
        self, chroma_instance: connections.ChromaConnection, mock_response
    ):
        expected_api_response = "test-key"
        expected_status_code = 200
        mock_response(expected_api_response, expected_status_code, "get")
        key = chroma_instance.get_connection()

        test_query = "test"
        expected_post_response = {"ids": ["test-id"], "docs": [["test_doc"]]}
        expected_status_code = 200
        mock_response(expected_post_response, expected_status_code, "post")
        result = chroma_instance.retrieve_documents(key, query=test_query)

        assert isinstance(result, dict)

    def test_retrieve_documents_response_code_on_fail(
        self, chroma_instance: connections.ChromaConnection, mock_response
    ):
        expected_api_response = "test-key"
        expected_status_code = 200
        mock_response(expected_api_response, expected_status_code, "get")
        key = chroma_instance.get_connection()

        test_query = "test"
        fail_post_response = {"fail": "fail"}
        generic_fail_code = 400
        mock_response(fail_post_response, generic_fail_code, "post")
        result = chroma_instance.retrieve_documents(key, query=test_query)

        assert isinstance(result, int)

    def test_get_documents_by_id(
        self, chroma_instance: connections.ChromaConnection, mock_response
    ):
        expected_api_response = "test-key"
        expected_status_code = 200
        mock_response(expected_api_response, expected_status_code, "get")
        key = chroma_instance.get_connection()

        test_id = ["test-id"]
        expected_post_response = {"ids": ["test-id"], "docs": [["test_doc"]]}
        mock_response(expected_post_response, expected_status_code, "post")
        result = chroma_instance.get_documents(key, ids=test_id)

        assert isinstance(result, dict)

        fail_post_response = {"fail": "fail"}
        generic_fail_code = 400
        mock_response(fail_post_response, generic_fail_code, "post")
        result = chroma_instance.get_documents(key, ids=test_id)

        assert isinstance(result, int)


# class TestStreamlitApp:
#     # App asks auth server to generate new key only for new instances of app
#     def test_get_new_key(self, streamlit_test_instance, mock_response):
#         mock_response()
#         at = streamlit_test_instance
#         at.run()

#         assert "key" in at.session_state
#         assert isinstance(at.session_state.key, str)
#         assert at.session_state.key_expire_datetime > datetime.utcnow()

#     # App renews key if it has an expired one, and does not generate a new key
#     def test_renew_key_on_expiry(self, streamlit_test_instance, mock_response):
#         mock_response()
#         at = streamlit_test_instance
#         at.session_state.key = "already-generated-key"
#         at.session_state.key_expire_datetime = datetime.utcnow() - timedelta(days=1)
#         at.run()

#         assert "already-generated-key" == at.session_state.key
#         assert at.session_state.key_expire_datetime > datetime.utcnow()

#     # Method to connect to LLM, whether it is API or locally hosted model
#     def test_connect_llm(self, streamlit_test_instance, mock_response):
#         mock_response()
#         at = streamlit_test_instance
#         at.run()

#         assert "llm" in at.session_state
# App revokes key on shutdown


# Move to integration tests
# def test_revoke_on_shutdown(
#     self, chroma_instance, streamlit_test_instance, mock_response, monkeypatch
# ):
#     expected_api_response = "test-key"
#     expected_status_code = 200
#     mock_response(expected_api_response, expected_status_code)

#     mock_connection = Mock(spec=connections.TestConnection)

#     monkeypatch.setattr(
#         test_app.TestConnection, "revoke_key", mock_connection.revoke_key
#     )
#     # at = streamlit_test_instance
#     at = AppTest.from_file("../streamlit/test_app.py")
#     at.secrets["RETRIEVER_API_BASE"] = "test"
#     at.secrets["RETRIEVER_API_SECRET"] = "test-secret"
#     at.secrets["OPENAI_API_KEY"] = "test-key"
#     at.secrets["TEST_ENV"] = True
#     at.session_state.key = "existing-key"
#     at.session_state.key_expire_datetime = datetime.utcnow() + timedelta(days=1)

#     at.run()

#     assert mock_connection.revoke_key.assert_called_once()
