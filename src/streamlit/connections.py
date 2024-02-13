"""
Contains abstract and concrete class representations of API wrappers used in the
Streamlit app.
"""

import logging
import uuid
import requests
import warnings

from datetime import datetime, timedelta
from urllib.parse import urljoin, unquote_plus
from abc import ABC, abstractmethod
from typing import List


logger = logging.getLogger(__name__)


class StubLLM:
    @staticmethod
    def query(*args, **kwargs):
        return "test_reponse"


class BaseConnection(ABC):
    """
    Base class for API wrappers used in the Streamlit app.
    """

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

    @abstractmethod
    def heartbeat(self):
        pass


class ChromaConnection(BaseConnection):
    """
    Implementation of wrapper for a Chroma database retrieval API.

    Attributes
    ----------
    base_url: str
        Domain name without paths.

    api-secret: str
        Secret key to pass into header for API auth.

    Methods
    -------
    get_connection()
        Initializes requirements needed to process API requests. Currently, this
        includes fetching a valid API key.

    renew_key(key)
        Passes an API key to the server for renewal.

    revoke_key()
        Passes an API key to the server for invalidation.

    retrieve_documents(key, query, n_docs=1, collection_name="summed")
        Packages and submits a query request to the server to find similar documents.

    get_documents()
        Packages and submits a fetch request to the server to get documents by
        their database id.

    heartbeat()
        Checks if the server is alive. Returns a boolean describing the state
        of the server connection.

    """

    def __init__(self, base_url, api_secret):
        self.base_url = base_url
        self.api_secret = api_secret

        # Toggle requests library checking validity of SSL/TLS certificates
        self._verify = True

    def get_connection(self) -> str | None:
        """
        Initialize the connection to the API server. This involves authenticating
        using the API secret and getting a valid API key from the server.

        Returns
        -------
        key : str
            API key for authenicating requests.

        None
            Returns None if authentication failed.
        """
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
            key = token.json()
            return key
        else:
            logger.info("Failed to receive new API key")

    def renew_key(self, key: str) -> int:
        """
        Submits a request using the client secret and a valid or invalid API key
        to extend the expiration date of the key by 24 hours.

        Parameters
        ----------
        key : str
            API key for authenicating requests.

        Returns
        -------
        status_code : int
            HTTP status code from the server response.
        """
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
        status_code = resp.status_code
        return status_code

    # Will only be called on shutdown
    def revoke_key(self, key: str) -> int:
        """
        Submits a request to invalidate an API key. This key will be unable
        to authenticate requests unless renewed.

        Parameters
        ----------
        key : str
            API key for authenicating requests.

        Returns
        -------
        status_code : int
            HTTP status code from the server response.

        See Also
        --------
        renew_key : Renew an invalid API key.
        """
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
        status_code = resp.status_code
        return status_code

    def retrieve_documents(
        self, key: str, query: str, n_docs: int = 1, collection_name: str = "summed"
    ) -> dict[str, str] | int:
        """
        Submits a request with an API key and a query to retrieve similar documents
        from the database.

        Parameters
        ----------
        key : str
            API key for authenicating requests.

        query : str
            Query string used to compare similarity.

        n_docs : int
            Number of the most relevant documents to retrieve.

        collection_name : str
            Name of the Chroma database collection.

        Returns
        -------
        documents : dict
            Dictionary of (document_id:str, document_contents:str) pairs. The keys
            are the document's database identifier, and the values are the text content
            inside the document.

        status_code : int
            HTTP status code returned on failed request.

        """
        headers = {"api-key": key}
        payload = {
            "query": query,
            "collection_name": collection_name,
            "n_docs": n_docs,
        }
        docs = requests.post(
            urljoin(self.base_url, "retrieve"),
            headers=headers,
            params=payload,
            verify=self._verify,
        )
        if docs.status_code == 200:
            doc_contents = docs.json()["docs"]
            ids = docs.json()["ids"]
            documents = dict(zip(ids, doc_contents))
            return documents
        else:  # Did not receive successful response
            logger.info(f"Status Code: {docs.status_code} \n Reason: {docs.text}")
            return docs.status_code

    def heartbeat(self) -> bool:
        """
        Checks if the server connection is alive.

        Returns
        -------
        is_alive : bool
            Boolean representing the connection state to the API server.
        """
        resp = requests.get(urljoin(self.base_url, "/"), verify=self._verify)
        if resp.status_code == 200:
            return True
        else:
            return False

    def get_documents(
        self, key: str, ids: List[str], collection_name: str = "summed"
    ) -> dict[str, str] | int:
        """
        Submits a request using the API key and a list of database ids to retrieve the
        documents associated with the submitted ids.

        Parameters
        ----------
        key : str
            API key for authenicating requests.

        ids : List[str]
            List of document ids to retrieve.

        collection_name : str
            Name of the Chroma database collection.

        Returns
        -------
        documents : dict
            Dictionary of (document_id:str, document_contents:str) pairs. The keys
            are the document's database identifier, and the values are the text content
            inside the document.

        status_code : int
            HTTP status code returned on failed request.
        """
        headers = {"api-key": key}
        payload = {"ids": ids, "collection_name": collection_name}
        docs = requests.post(
            urljoin(self.base_url, "/documents"),
            headers=headers,
            params=payload,
            verify=self._verify,
        )
        if docs.status_code == 200:
            doc_contents = docs.json()["docs"]
            docs_ids = docs.json()["ids"]
            documents = dict(zip(docs_ids, doc_contents))
            return documents
        else:  # Did not receive successful response
            logger.info(f"Status Code: {docs.status_code} \n Reason: {docs.text}")
            return docs.status_code


class TestConnection(BaseConnection):
    """
    Stub API wrapper used for testing Streamlit apps. Included in the module because
    of Streamlit's script-like behavior.

    """

    def get_connection(self, *args, **kwargs):
        return "test-key"

    def renew_key(self, key, *args, status_code=0, **kwargs):
        return status_code

    def revoke_key(self, key, *args, status_code=0, **kwargs):
        return status_code

    def retrieve_documents(self, *args, **kwargs):
        return ["test_documents"]

    def heartbeat(self):
        return True
