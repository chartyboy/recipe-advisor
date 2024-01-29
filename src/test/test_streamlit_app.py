import pytest
import os
import logging
import requests
import atexit

import streamlit as st

from datetime import datetime, timedelta
from unittest.mock import Mock
from dependency_injector import containers, providers
from streamlit.testing.v1 import AppTest

from stub_classes import MockResponse, StubLLM

from src.streamlit import connections
from src.streamlit.connections import ChromaConnection

# from src.streamlit import test_app

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.fixture
def mock_response(monkeypatch):
    def _mock_response(json_response=dict(), status_code=200):
        def mock_get(*args, **kwargs):
            return MockResponse(json_response, status_code)

        monkeypatch.setattr(requests, "get", mock_get)

    return _mock_response


@pytest.fixture
def chroma_instance() -> connections.ChromaConnection:
    return connections.ChromaConnection(base_url="test", api_secret="test_secret")


@pytest.fixture
def streamlit_test_instance():
    at = AppTest.from_file("../streamlit/test_app.py")
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
        mock_response(expected_api_response, expected_status_code)
        key = chroma_instance.get_connection()

        assert isinstance(key, str)

    def test_retrieve_documents(
        self, chroma_instance: connections.ChromaConnection, mock_response
    ):
        expected_api_response = {"docs": ["test1", "test2"]}
        expected_status_code = 200
        mock_response(expected_api_response, expected_status_code)
        key = chroma_instance.get_connection()

        test_query = "test"
        result = chroma_instance.retrieve_documents(key, query=test_query)

        assert isinstance(result, list)
        assert isinstance(result[0], str)


class TestStreamlitApp:
    # App asks auth server to generate new key only for new instances of app
    def test_get_new_key(self, streamlit_test_instance, mock_response):
        mock_response()
        at = streamlit_test_instance
        at.run()

        assert "key" in at.session_state
        assert isinstance(at.session_state.key, str)
        assert at.session_state.key_expire_datetime > datetime.utcnow()

    # App renews key if it has an expired one, and does not generate a new key
    def test_renew_key_on_expiry(self, streamlit_test_instance, mock_response):
        mock_response()
        at = streamlit_test_instance
        at.session_state.key = "already-generated-key"
        at.session_state.key_expire_datetime = datetime.utcnow() - timedelta(days=1)
        at.run()

        assert "already-generated-key" == at.session_state.key
        assert at.session_state.key_expire_datetime > datetime.utcnow()

    # Method to connect to LLM, whether it is API or locally hosted model
    def test_connect_llm(self, streamlit_test_instance, mock_response):
        mock_response()
        at = streamlit_test_instance
        at.run()

        assert "llm" in at.session_state
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
