import pytest
import numpy as np
import chromadb
import json
import os

from pytest_mock.plugin import MockerFixture
from src import embeddings


# Modify class to be tested by removing constructor calls to instance methods
class EmbeddingsTestObject(embeddings.RecipeEmbeddings):
    def __post_init__(self):
        return


class StubDocument:
    def __init__(self) -> None:
        self.page_content = "TEST"


class StubJSONLoader:
    @staticmethod
    def load(*args, **kwargs):
        return [StubDocument()]


class StubChromaCollection:
    @staticmethod
    def add(*args, **kwargs):
        pass

    @staticmethod
    def get(*args, **kwargs):
        pass


class StubChromaClient:
    @staticmethod
    def get_or_create_collection(*args, **kwargs):
        return StubChromaCollection()

    def get_collection(*args, **kwargs):
        return StubChromaCollection


@pytest.fixture
def function_to_wrap():
    def to_wrap(*args, **kwargs):
        return

    return to_wrap


# Initialize test class with type appropriate attributes
@pytest.fixture()
def embeddings_instance(tmp_path_factory, function_to_wrap):
    json_path = [""]
    embedding_model = function_to_wrap
    persist_path = tmp_path_factory.mktemp("db")
    instance = EmbeddingsTestObject(json_path, embedding_model, persist_path)
    instance.chroma_client = StubChromaClient()  # type: ignore
    return instance


@pytest.fixture(scope="session")
def jsonl_file(tmp_path_factory, jsonl_length):
    content = [{"test": ["a", "b", "c"]}, {"test2": ["ab", "bb", "cb"]}]
    fn = tmp_path_factory.mktemp("dataset") / "test.jsonl"
    with open(fn, "w", encoding="utf-8") as f:
        for i in range(jsonl_length):
            f.write(json.dumps(content) + "\n")

    return fn


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
        assert isinstance(meta_func, function)
        assert isinstance(returned_metadata, dict)
        assert "website" in returned_metadata.keys()

    # def test_EmbeddingFunctionInterface(self, function_to_wrap):
    #     chromadb
    #     # Should pass Chroma's EmbeddingFunction input validation checks
    #     # Calling the function should require a list of Document and return a list of vectors
    #     pass


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

    # def test_create_loader(self):
    #     # Should return a JSONLoader that conforms to Langchain's JSONLoader interface
    #     pass

    def test_extract_page_content(self):
        # Should take a list of Document and return a list of strings from the Document's
        # page_content attribute
        num_docs = 5
        documents = [StubDocument() for i in range(num_docs)]
        texts = embeddings.RecipeEmbeddings.extract_page_content(documents)

        assert isinstance(texts, list)
        assert isinstance(texts[0], str)
        assert len(texts) == num_docs

    def test_load_jsonlines(self, embeddings_instance, jsonl_file, monkeypatch):
        def mock_create_loader():
            return StubJSONLoader()

        monkeypatch.setattr(
            embeddings.RecipeEmbeddings, "create_loader", mock_create_loader
        )
        embeddings_instance.json_path = [jsonl_file, jsonl_file]
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
        assert isinstance(embeddings_instance.chroma_client, chromadb.Client.__class__)

    def test_reset_chroma(self, embeddings_instance, dir_to_delete):
        embeddings_instance.persist_path = dir_to_delete
        embeddings_instance.reset = True
        embeddings_instance.reset_chroma()

        assert not os.path.exists(dir_to_delete)
        pass

    def test_create_chroma_collection(self, embeddings_instance):
        exceeds_max_batch_size = 50000
        under_max_batch_size = 2
        collection_name = "test"
        large_documents = ["a"] * exceeds_max_batch_size
        large_embeddings = [0] * exceeds_max_batch_size
        small_documents = ["a"] * under_max_batch_size
        small_embeddings = [0] * under_max_batch_size

        # Should create and return a single chroma collection handle
        # Should be able to handle creating collections with sizes larger than
        # CHROMA_MAX_BATCH_SIZE = 41666

        returned_collection = embeddings_instance.create_chroma_collection(
            collection_name, small_embeddings, small_documents
        )

        assert isinstance(returned_collection, StubChromaCollection)

        returned_collection = embeddings_instance.create_chroma_collection(
            collection_name, large_embeddings, large_documents
        )
        assert isinstance(returned_collection, StubChromaCollection)

    # def test_create_collections(self):
    #     def mock_create_chroma_collection():
    #         return StubChromaCollection()

    #     def mock_extract_page_content():
    #         return ["a", "b", "c"]

    #     # Should return a list of chroma collection handles, as a side effect, create missing collections
    #     pass

    # def test_create_summed_collection(self):
    #     # Should take a list of collection names and return a single collection handle
    #     # Should sum and normalize embeddings from multiple collections
    #     # Should call get_embeddings when embeddings are not loaded into instance state
    #     # Should use another arbitrary set of documents when inserting into new collection
    #     pass

    # def test_get_embeddings(self, embeddings_instance, monkeypatch):
    #     class ExtendedStubCollection(StubChromaCollection):
    #         def __init__(self, return_empty) -> None:
    #             self.return_empty = return_empty

    #         def get(self):
    #             if self.return_empty:
    #                 return []
    #             else:
    #                 return {"documents": StubDocument()}

    #     def mock_get_collection(name):
    #         if name == "test":
    #             return ExtendedStubCollection(True)
    #         else:
    #             return ExtendedStubCollection(False)

    #     embeddings_instance.chroma_client = StubChromaClient()
    #     monkeypatch.setattr(
    #         embeddings_instance.chroma_client, "get_collection", mock_get_collection
    #     )

    #     # Get embeddings from collection that exists
    #     retrieved = embeddings_instance.get_embeddings("test")
    #     assert isinstance(retrieved, dict)

    #     with pytest.raises(ValueError):
    #         embeddings_instance.get_embeddings("does_not_exist")
    #     # Should retrieve documents from a chroma collection from the instance's chroma client
    #     # Should raise an error if collection is missing
    #     pass
