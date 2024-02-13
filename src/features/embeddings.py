"""
Module contains classes and methods for using the LangChain embeddings library to
calculate and store text embeddings into a Chroma database.

Classes
-------
RecipeEmbeddings
    Class with packaged methods to load text data, calculate text embeddings, and
    Chroma database management.
"""

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from sklearn.preprocessing import normalize
from typing import Callable, List, Any, Iterable, Optional, Sequence
from langchain.docstore.document import Document
from dataclasses import dataclass, field
from numpy.typing import ArrayLike
from src.features.interfaces import EmbeddingFunctionInterface
import chromadb
import numpy as np
import uuid
import os
import shutil


def metadata_factory(source_website: str) -> Callable[[dict, dict], dict]:
    """
    Factory to customize Langchain metadata function.

    Parameters
    ----------
    source_website : str
        String of metadata content

    Returns
    -------
    website_metadata : Callable[[dict, dict], dict]
        Metadata function that fits Langchain's metadata interface.

    See Also
    --------
    Langchain Chroma integrations_

    .. _Langchain Chroma integrations: https://python.langchain.com/docs/integrations/vectorstores/chroma
    """

    def website_metadata(record: dict, metadata: dict) -> dict[str, str]:
        metadata["website"] = source_website
        return metadata

    return website_metadata


def default_base_collections():
    return ["name", "ingredient", "instruction"]


@dataclass
class RecipeEmbeddings:
    """
    Collection of methods to handle embedding generation from JSON text data.

    Attributes
    ----------
    json_path: List[str]
        Paths to JSONLines files containing text data to embed.

    embedding_model: langchain.Embeddings
        Handle to one of Langchain's text embedding models.

    persist_path: str
        Path to store Chroma database files.

    base_collections: List[str], default = ["name", "ingredient", "instruction"]
        JSON names to use when generating text/embedding pairs.

    sources: List[str], optional
        Source websites to be passed to metadata factory. Must be same length
        as `json_path`.

    reset: bool, default = False
        Boolean to reset Chroma database if one already exists.

    shared_ids: bool, default = False
        Boolean to sync ids across all created collections

    Methods
    -------
    load_jsonlines(self)
        Loads JSONLines files with a jq schema.

    initialize_chroma(self)
        Main function to clean and output data.

    reset_chroma(self)

    create_collections(self)

    create_summed_collection(
        self,
        corpus_keys: List[str],
        collection_name: str = "summed",
        corpus_type: str = "content",
    )

    """

    json_path: List[str]
    embedding_model: Any
    persist_path: str
    base_collections: List[str] = field(default_factory=default_base_collections)
    sources: Optional[List[str]] = None
    reset: Optional[bool] = False
    shared_ids: Optional[bool] = False

    def __post_init__(self):
        self._load_embedding_model()
        self.initialize_chroma()
        self.embeddings: dict[str, ArrayLike] = dict()
        self.source_map: dict[str, str] = dict()

        if self.sources is None:
            self.source_map = dict(zip(self.json_path, [""] * len(self.json_path)))
        else:
            self.source_map = dict(zip(self.json_path, self.sources))
        self.load_jsonlines()

        if self.shared_ids:
            if self.reset:
                self.ids = self.generate_ids()
            else:
                self.ids = self.get_keys()

    def generate_ids(self) -> List[str]:
        keys = list(self.document_corpus.keys())
        return [str(uuid.uuid4()) for _ in self.document_corpus[keys[0]]]

    def get_keys(self) -> List[str]:
        collections = self.chroma_client.list_collections()
        if collections:
            coll_name = collections[0].name
            collection_handle = self.chroma_client.get_collection(name=coll_name)
            get_result = collection_handle.get()
            return get_result["ids"]
        else:
            return self.generate_ids()

    def _load_embedding_model(self):
        self.embedding_function = EmbeddingFunctionInterface(
            self.embedding_model.embed_documents
        )

    def process(self):
        """
        Main loop to load and process data.

        See Also
        --------
        load_jsonlines : Loads JSONLines files with a jq schema.
        create_collections : Create Chroma database collections from text data.
        """

        # self.load_jsonlines()
        self.create_collections()

    def load_jsonlines(self):
        """
        Loads multiple JSONLines files with a jq schema.
        """

        self.document_corpus = dict()
        collection_types = ["content", "name", "ingredient", "instruction", "step"]
        schemas = [
            ".whole_recipe",
            ".recipe_name",
            ".ingredients",
            ".instructions",
            ".step_instructions",
        ]

        for document_type in collection_types:
            self.document_corpus[document_type] = list()

        for fpath in self.json_path:
            meta_func = metadata_factory(self.source_map[fpath])
            json_loader = [
                self.create_loader(
                    fpath,
                    schema,
                    json_lines=True,
                    text_content=False,
                    metadata_func=meta_func,
                )
                for schema in schemas
            ]
            loaded_json = [loader.load() for loader in json_loader]
            loaded_documents = dict(
                zip(
                    collection_types,
                    loaded_json,
                )
            )

            for document_type in collection_types:
                self.document_corpus[document_type] += loaded_documents[document_type]
        self.corpus = self.document_corpus.values()

    def initialize_chroma(self):
        """
        Initializes a Chroma vector database

        See Also
        --------
        reset_chroma : Delete a Chroma database.
        """

        if self.reset:
            self.reset_chroma()
        self.chroma_client = chromadb.PersistentClient(path=self.persist_path)

    def reset_chroma(self):
        """
        Delete a Chroma database.
        """

        if os.path.exists(self.persist_path):
            shutil.rmtree(self.persist_path)

    def create_collections(self) -> dict[str, chromadb.Collection]:
        """
        Create Chroma database collections from text/embedding pairs.

        See Also
        --------
        create_chroma_collection : Calculate embeddings and insert into vector database.
        """

        handles = dict()
        for corpus_type in self.base_collections:
            documents = self.document_corpus[corpus_type]
            document_contents = self.extract_page_content(documents)
            embeddings = self.embedding_model.embed_documents(document_contents)

            collection_handle = self.create_chroma_collection(
                corpus_type, self.normalize_array(embeddings), document_contents  # type: ignore
            )
            handles[corpus_type] = collection_handle
            self.embeddings[corpus_type] = embeddings
        return handles

    def create_chroma_collection(
        self,
        collection_name: str,
        embeddings: chromadb.Embeddings,
        documents: List[str],
    ) -> chromadb.Collection:
        """
        Calculates embeddings for a set of Document and insert into a Chroma database.

        Parameters
        ----------
        collection_name : str
            Path to location of JSONLines file.

        embeddings : List[chromadb.Embeddings]
            Array of vector embeddings.

        documents : List[str]
            Array of text strings.

        Returns
        -------
        collection_handle : chromadb.Collection
            Reference to created Chroma database collection.
        """

        collection_handle = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )
        # Chroma max batch size for number of documents inserted in a single call
        CHROMA_MAX_BATCH_SIZE = 41666
        if len(documents) > CHROMA_MAX_BATCH_SIZE:
            n_splits = len(documents) // CHROMA_MAX_BATCH_SIZE + 1
            for i in range(n_splits):
                start = i * CHROMA_MAX_BATCH_SIZE
                end = min(len(documents), (i + 1) * CHROMA_MAX_BATCH_SIZE)

                if self.shared_ids:
                    ids = self.ids[start:end]
                else:
                    ids = [str(uuid.uuid4()) for _ in documents[start:end]]

                collection_handle.add(
                    ids=ids,
                    embeddings=embeddings[start:end],
                    documents=documents[start:end],
                )
        else:
            if self.shared_ids:
                ids = self.ids
            else:
                ids = [str(uuid.uuid4()) for _ in documents]

            collection_handle.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
            )
        return collection_handle

    def create_summed_collection(
        self,
        corpus_keys: List[str],
        collection_name: str = "summed",
        corpus_type: str = "content",
    ) -> chromadb.Collection:
        """
        Create a collection of vector embeddings from a summation of other vector embeddings.

        Parameters
        ----------
        corpus_keys: List[str]
            List of JSON names corresponding to sets of embeddings to sum.

        collection_name: str, default = "summed"
            Name of collection to be created.

        corpus_type: str, default = "content"
            JSON name of documents to pair with summed embeddings.

        Returns
        -------
        collection_handle : chromadb.Collection
            Reference to created Chroma database collection.
        """

        vectors_to_add = list()

        # If embeddings are not already loaded into memory
        for key in corpus_keys:
            if key not in self.embeddings.keys():
                stored_collection = self.get_embeddings(key)
                pass
                self.embeddings[key] = stored_collection["embeddings"]  # type: ignore
        vectors_to_add = [self.embeddings[key] for key in corpus_keys]
        summed_embeddings = self.sum_vertically(vectors_to_add)
        normalized_and_summed = self.normalize_array(summed_embeddings)
        collection_handle = self.create_chroma_collection(
            collection_name,
            normalized_and_summed,  # type: ignore
            self.extract_page_content(self.document_corpus[corpus_type]),
        )
        return collection_handle

    def get_embeddings(self, collection_name: str) -> chromadb.GetResult | dict:
        """
        Retrieve vector embeddings and documents from a Chroma collection

        Parameters
        ----------
        collection_name: str
            Name of collection to retrieve from.

        Returns
        -------
        chromadb.GetResult
            Dict-like with keys corresponding to retrieved vector embeddings and documents.
        """

        collection_handle = self.chroma_client.get_collection(name=collection_name)
        res = collection_handle.get(include=["embeddings", "documents"])
        return res

    @staticmethod
    def sum_vertically(embeddings: Iterable[ArrayLike]) -> np.ndarray:
        """
        Sum a tensor along its first axis.

        Parameters
        ----------
        embeddings: Iterable[ArrayLike]
            n-dimensional tensor of vector embeddings.

        Returns
        -------
        summed_embeddings : np.ndarray
            n-1 dimensional np.ndarray of vector embeddings
        """

        summed_embeddings = np.sum(
            embeddings,  # type: ignore
            axis=0,
        )  # type: ignore
        return summed_embeddings

    @staticmethod
    def normalize_array(array: ArrayLike) -> np.ndarray:
        """
        Normalize each row of a matrix

        Parameters
        ----------
        array: ArrayLike
            Matrix to normalize rows.

        Returns
        -------
        np.ndarray
            Normalized matrix.
        """

        return normalize(array, axis=1).tolist()  # type: ignore

    @staticmethod
    def create_loader(json_path: str, schema: str, **kwargs) -> JSONLoader:
        """
        Creates Langchain JSONLoader objects.

        Parameters
        ----------
        json_path: str
            Path to location of JSONLines file.

        schema: str
            jq schema to use when parsing JSONLines file.

        **kwargs
            Other keyword arguments are passed to the JSONLoader constructor.

        Returns
        -------
        JSONLoader
            Langchain JSONLoader object.

        See Also
        --------
        to_pandas : Convert loaded content into pandas Dataframes
        """
        return JSONLoader(file_path=json_path, jq_schema=schema, **kwargs)

    @staticmethod
    def extract_page_content(documents: List[Document]) -> List[str]:
        """
        Extracts text from Document objects.

        Parameters
        ----------
        documents: List[Document]
            List of Document.

        Returns
        -------
        res: List[str]
            List of extracted texts from input list of Document.
        """
        res = [doc.page_content for doc in documents]
        return res
