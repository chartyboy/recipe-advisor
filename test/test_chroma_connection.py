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

    def test_get_connection_fail(
        self, chroma_instance: connections.ChromaConnection, mock_response
    ):
        error_api_response = ""
        error_status_code = 400
        mock_response(error_api_response, error_status_code, "get")
        key = chroma_instance.get_connection()

        assert key is None

    def test_renew_key(
        self, chroma_instance: connections.ChromaConnection, mock_response
    ):
        expected_status_code = 200
        mock_response(status_code=expected_status_code, method="get")
        response_code = chroma_instance.renew_key("test-key")

        assert isinstance(response_code, int)
        assert response_code == expected_status_code

    def test_revoke_key(
        self, chroma_instance: connections.ChromaConnection, mock_response
    ):
        expected_status_code = 200
        mock_response(status_code=expected_status_code, method="get")
        response_code = chroma_instance.revoke_key("test-key")

        assert isinstance(response_code, int)
        assert response_code == expected_status_code

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
        result = chroma_instance.retrieve_documents(
            key, query=test_query  # type:ignore
        )

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
        result = chroma_instance.retrieve_documents(
            key, query=test_query  # type:ignore
        )

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
        result = chroma_instance.get_documents(key, ids=test_id)  # type:ignore

        assert isinstance(result, dict)

        fail_post_response = {"fail": "fail"}
        generic_fail_code = 400
        mock_response(fail_post_response, generic_fail_code, "post")
        result = chroma_instance.get_documents(key, ids=test_id)  # type:ignore

        assert isinstance(result, int)
