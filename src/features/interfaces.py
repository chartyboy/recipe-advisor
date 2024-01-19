import chromadb
from typing import List, Callable


# Wrapper class for Chroma to pass validation checks with Langchain embedding interfaces
class EmbeddingFunctionInterface(chromadb.EmbeddingFunction):
    """
    Wrapper class to allow Langchain embedding interfaces to pass Chroma validation checks.

    Attributes
    ----------
    embedding_function : Callable[[List[str]], chromadb.Embeddings]
        Function with single input argument of list of strings

    See Also
    --------
    Chroma EmbeddingFunction_

    .. _EmbeddingFunction: https://docs.trychroma.com/embeddings

    """

    def __init__(
        self, embedding_function: Callable[[List[str]], chromadb.Embeddings]
    ) -> None:
        super().__init__()
        self.embedding_function = embedding_function

    def __call__(self, input: List[str]) -> chromadb.Embeddings:
        return self.embedding_function(input)
