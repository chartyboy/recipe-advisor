"""
Collection of helper functions to dynamically instantiate LangChain chains.

See Also
--------
LangChain Expression Language : https://python.langchain.com/docs/expression_language/
"""

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from prompts import (
    DEFAULT_DOCUMENT_PROMPT,
    strip_name_prompt,
    modified_name_prompt,
    cot_multi_prompt,
)


def combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def initialize_retriever_chain():
    retriever_conn = None
    return retriever_conn


def naming_chain(LLM_model):
    """
    Creates a LCEL chain for creating new recipe names from user-requestd changes.
    """
    output_parser = StrOutputParser()
    generate_new_name = {
        "new_name": {"resp": modified_name_prompt | LLM_model | output_parser}
        | strip_name_prompt
        | LLM_model
        | output_parser,
        "inputs": RunnablePassthrough(),
    }
    return RunnablePassthrough() | generate_new_name


def initialize_LLM_chain(LLM_model, retrieved_docs: str):
    """
    Creates a LCEL chain for creating new recipes from user-requestd changes.
    """
    output_parser = StrOutputParser()

    prompt_inputs = {
        "inputs": lambda x: x["inputs"]
        | {"new_name": x["new_name"], "retrieved_recipes": x["retrieved"]}
    }
    parallels = {
        "retrieved": lambda x: [retrieved_docs],
        "new_name": itemgetter("new_name"),
        "inputs": RunnablePassthrough(),
    }
    output_chain = RunnableParallel(
        {
            "output": prompt_inputs
            | RunnablePassthrough()
            | itemgetter("inputs")
            | cot_multi_prompt
            | LLM_model
            | output_parser,
            "context_recipes": itemgetter("retrieved"),
            "prompt": prompt_inputs
            | RunnablePassthrough()
            | itemgetter("inputs")
            | cot_multi_prompt,
        }
    )
    rag_cot_chain = RunnablePassthrough() | parallels | output_chain
    return rag_cot_chain
