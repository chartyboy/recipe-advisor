import logging
import uuid
import requests
import warnings

from datetime import datetime, timedelta
from urllib.parse import urljoin, unquote_plus
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


# RETRIEVER_API_BASE = st.secrets["RETRIEVER_API_BASE"]
# RETRIEVER_API_SECRET = st.secrets["RETRIEVER_API_SECRET"]
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


class StubLLM:
    @staticmethod
    def query(*args, **kwargs):
        return "test_reponse"


class BaseConnection(ABC):
    @abstractmethod
    def get_connection(self, *args, **kwargs):
        pass

    @abstractmethod
    def renew_key(self, *args, **kwargs):
        pass

    @abstractmethod
    def revoke_key(self, *args, **kwargs):
        pass

    @abstractmethod
    def retrieve_documents(self, *args, **kwargs):
        pass


class ChromaConnection(BaseConnection):
    def __init__(self, base_url, api_secret):
        self.base_url = base_url
        self.api_secret = api_secret
        self._verify = True

    def get_connection(self) -> dict | str | None:
        auth_payload = {
            "name": "client_" + " " + uuid.uuid4().hex + " ",
        }
        auth_header = {"secret-key": self.api_secret}
        token = requests.get(
            urljoin(self.base_url, "/auth/new"),
            params=auth_payload,
            headers=auth_header,
            verify=self._verify,
        )

        if token.status_code == 200:
            logger.info("Successfully received new API key")
            return token.json()
        else:
            logger.info("Failed to receive new API key")

    def renew_key(self, key: str) -> int:
        renew_payload = {"api-key": key}
        resp = requests.get(
            urljoin(self.base_url, "/auth/renew"),
            params=renew_payload,
            verify=self._verify,
        )
        if resp.status_code == 200:
            logger.info("Successfully renewed API key")
        else:
            logger.info("Failed to renew API key")
        return resp.status_code

    # Will only be called on shutdown
    def revoke_key(self, key: str) -> int:
        revoke_payload = {"api-key": key}
        revoke_header = {"secret-key": self.api_secret}
        resp = requests.get(
            urljoin(self.base_url, "/auth/revoke"),
            params=revoke_payload,
            headers=revoke_header,
            verify=self._verify,
        )
        if resp.status_code == 200:
            logger.info("Successfully revoked API key")
        return resp.status_code

    def retrieve_documents(self, key, query, n_docs=1) -> list[str] | int:
        headers = {"api-key": key}
        payload = {
            "query": query,
            "collection_name": "summed",
            "n_docs": n_docs,
        }
        docs = requests.get(
            urljoin(self.base_url, "retrieve"),
            headers=headers,
            params=payload,
            verify=self._verify,
        )
        if docs.status_code == 200:
            return docs.json()["docs"]
        else:  # Did not receive successful response
            logger.info(f"Status Code: {docs.status_code} \n Reason: {docs.text}")
            return docs.status_code


class TestConnection(BaseConnection):
    def get_connection(self):
        return "test-key"

    def renew_key(self, key, status_code=0):
        return status_code

    def revoke_key(self, key, status_code=0):
        return status_code

    def retrieve_documents(self):
        return ["test_documents"]
