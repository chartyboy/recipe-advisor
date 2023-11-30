import pytest
import numpy as np
from pytest_mock.plugin import MockerFixture
from src import embeddings


@pytest.fixture
def mock_embeddings(mocker: MockerFixture):
    pass


class TestEmbeddingsHelpers:
    def test_metadata_factory(self):
        # Should return a function that takes two dicts as input, and modifies and returns a dict named "metadata"
        pass

    def test_EmbeddingFunctionInterface(self):
        # Should pass Chroma's EmbeddingFunction input validation checks
        # Calling the function should require a list of Document and return a list of vectors
        pass


class TestEmbeddings:
    def test_load_jsonlines(self):
        # Should be able to multiple read json/jsonLines files and assign to instance attribute
        # Assigned object should be of type dict with multiple keys according to input
        pass

    def test_initialize_chroma(self):
        # Should take a path and return a chroma client handle
        pass

    def test_reset_chroma(self):
        # Should call delete on the chroma path
        pass

    def test_create_collections(self):
        # Should return a list of chroma collection handles, as a side effect, create missing collections
        pass

    def test_create_chroma_collection(self):
        # Should create and return a single chroma collection handle
        # Should just return a handle if the collection already exists
        # Should be able to handle creating collections with sizes larger than
        # CHROMA_MAX_BATCH_SIZE = 41666
        pass

    def test_create_summed_collection(self):
        # Should take a list of collection names and return a single collection handle
        # Should sum and normalize embeddings from multiple collections
        # Should call get_embeddings when embeddings are not loaded into instance state
        # Should use another arbitrary set of documents when inserting into new collection
        pass

    def test_get_embeddings(self):
        # Should retrieve documents from a chroma collection from the instance's chroma client
        # Should raise an error if collection is missing
        pass

    def test_sum_vertically(self, mocker: MockerFixture):
        # Should sum vectors together
        # Should only collapse the first axis in a n-dimensional tensor
        pass

    def test_normalize_array(self):
        # Should normalize rows of a 2-dimensional matrix
        pass

    def test_create_loader(self):
        # Should return a JSONLoader that conforms to Langchain's JSONLoader interface
        pass

    def test_extract_page_content(self):
        # Should take a list of Document and return a list of strings from the Document's
        # page_content attribute
        pass
