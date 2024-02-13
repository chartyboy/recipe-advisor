import pytest
import numpy as np
import chromadb
import json
import os

from typing import Callable

from src.features import embeddings
from src.features.interfaces import EmbeddingFunctionInterface
from .stub_classes import (
    StubDocument,
    StubJSONLoader,
    StubChromaCollection,
    StubChromaClient,
)


# Modify class to be tested by removing constructor calls to instance methods
class EmbeddingsTestObject(embeddings.RecipeEmbeddings):
    def __post_init__(self):
        return


# Stub function for stubbing EmbeddingFunctionInterface
@pytest.fixture
def function_to_wrap():
    def to_wrap(*args, **kwargs):
        return

    return to_wrap


# Initialize test class with type appropriate attributes
@pytest.fixture()
def embeddings_instance(tmp_path_factory, function_to_wrap):
    json_path = [""]
    embedding_model = EmbeddingFunctionInterface(function_to_wrap)
    persist_path = str(tmp_path_factory.mktemp("db"))
    instance = EmbeddingsTestObject(json_path, embedding_model, persist_path)
    instance.chroma_client = StubChromaClient()  # type: ignore
    return instance


@pytest.fixture(scope="session")
def jsonl_length():
    return 4


@pytest.fixture(scope="session")
def jsonl_file(tmp_path_factory, jsonl_length):
    content = [{"test": ["a", "b", "c"]}, {"test2": ["ab", "bb", "cb"]}]
    fn = tmp_path_factory.mktemp("dataset") / "test.jsonl"
    with open(fn, "w", encoding="utf-8") as f:
        for i in range(jsonl_length):
            f.write(json.dumps(content) + "\n")

    return str(fn)


@pytest.fixture
def dir_to_delete(tmp_path_factory):
    delete_path = tmp_path_factory.mktemp("dataset") / "delete_test.sqlite3"
    return delete_path


class TestEmbeddingsHelpers:
    def test_metadata_factory(self):
        website = "google.com"
        meta_func = embeddings.metadata_factory(website)
        returned_metadata = meta_func(dict(), dict())
        # Should return a function that takes two dicts as input, and modifies and returns a dict named "metadata"
        assert isinstance(meta_func, Callable)
        assert isinstance(returned_metadata, dict)
        assert "website" in returned_metadata.keys()


class TestEmbeddings:
    @pytest.mark.parametrize(
        "array",
        [
            np.ones((4, 5, 3, 2)),
            [[[1, 2], [2, 1]], [[3, 4], [4, 4]], [[5, 5], [7, 7]]],
        ],  # Shape (3,2,2)
    )
    def test_sum_vertically(self, array):
        # Should sum vectors together
        # Should only collapse the first axis in a n-dimensional tensor
        expected_shape = np.array(array).shape[1:]
        res = embeddings.RecipeEmbeddings.sum_vertically(array)

        assert res.shape == expected_shape
        pass

    def test_normalize_array(self):
        array = [[1, 2], [3, 4]]
        normalized_array = embeddings.RecipeEmbeddings.normalize_array(array)
        # Should normalize rows of a 2-dimensional matrix
        # Check for all row magnitudes = 1
        for mag in np.linalg.norm(normalized_array, axis=1):
            assert np.isclose(1, mag)
        pass

    def test_extract_page_content(self):
        # Should take a list of Document and return a list of strings from the Document's
        # page_content attribute
        num_docs = 5
        documents = [StubDocument() for i in range(num_docs)]
        texts = embeddings.RecipeEmbeddings.extract_page_content(documents)  # type: ignore

        assert isinstance(texts, list)
        assert isinstance(texts[0], str)
        assert len(texts) == num_docs

    def test_load_jsonlines(self, embeddings_instance, jsonl_file, monkeypatch):
        def mock_create_loader(*args, **kwargs):
            return StubJSONLoader()

        monkeypatch.setattr(
            embeddings.RecipeEmbeddings, "create_loader", mock_create_loader
        )
        embeddings_instance.json_path = [jsonl_file, jsonl_file]
        embeddings_instance.source_map = dict(
            zip(
                embeddings_instance.json_path, [""] * len(embeddings_instance.json_path)
            )
        )
        embeddings_instance.load_jsonlines()

        first_key = list(embeddings_instance.document_corpus.keys())[0]
        # Should be able to multiple read json/jsonLines files and assign to instance attribute
        # Assigned object should be of type dict with multiple keys according to input
        assert isinstance(embeddings_instance.document_corpus, dict)
        assert isinstance(embeddings_instance.document_corpus[first_key], list)

    def test_initialize_chroma(self, embeddings_instance):
        embeddings_instance.reset = False
        # Should take a path and return a chroma client handle
        embeddings_instance.initialize_chroma()
        assert isinstance(embeddings_instance.chroma_client, chromadb.api.client.Client)  # type: ignore

    def test_reset_chroma(self, embeddings_instance, dir_to_delete):
        embeddings_instance.persist_path = dir_to_delete
        embeddings_instance.reset = True
        embeddings_instance.reset_chroma()

        assert not os.path.exists(dir_to_delete)
        pass

    @pytest.mark.parametrize(
        "shared_ids",
        [True, False],
    )
    def test_create_chroma_collection(
        self, embeddings_instance, function_to_wrap, shared_ids
    ):
        exceeds_max_batch_size = 50000
        under_max_batch_size = 2
        collection_name = "test"
        large_documents = ["a"] * exceeds_max_batch_size
        large_embeddings = [0] * exceeds_max_batch_size
        small_documents = ["a"] * under_max_batch_size
        small_embeddings = [0] * under_max_batch_size

        embeddings_instance.shared_ids = shared_ids

        # Should create and return a single chroma collection handle
        # Should be able to handle creating collections with sizes larger than
        # CHROMA_MAX_BATCH_SIZE = 41666
        embeddings_instance.embedding_function = EmbeddingFunctionInterface(
            function_to_wrap
        )
        embeddings_instance.ids = [str(i) for i in range(len(small_documents))]
        returned_collection = embeddings_instance.create_chroma_collection(
            collection_name, small_embeddings, small_documents
        )

        assert isinstance(returned_collection, StubChromaCollection)
        assert len(small_documents) == returned_collection.n_elements
        collection_ids = sorted(returned_collection.database["ids"])
        if shared_ids:
            assert embeddings_instance.ids == collection_ids
        else:
            assert embeddings_instance.ids != collection_ids

        embeddings_instance.ids = [str(i) for i in range(len(large_documents))]
        returned_collection = embeddings_instance.create_chroma_collection(
            collection_name, large_embeddings, large_documents
        )
        assert isinstance(returned_collection, StubChromaCollection)
        assert len(large_documents) == returned_collection.n_elements
