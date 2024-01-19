from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain.schema import format_document
from langchain.globals import set_debug, set_verbose
from langchain.memory import ConversationBufferMemory
from langchain.schema.output_parser import StrOutputParser
from langchain.llms import OpenAI
from prompts import (
    beef_recipe,
    mushroom_recipe,
    DEFAULT_DOCUMENT_PROMPT,
    strip_name_prompt,
    modified_name_prompt,
    cot_multi_prompt,
)


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def initialize_retriever_chain():
    retriever_conn = None
    return retriever_conn


def naming_chain(LLM_model: OpenAI):
    output_parser = StrOutputParser()
    generate_new_name = {
        "new_name": {"resp": modified_name_prompt | LLM_model | output_parser}
        | strip_name_prompt
        | LLM_model
        | output_parser,
        "inputs": RunnablePassthrough(),
    }
    return generate_new_name


def initialize_LLM_chain(LLM_model, retrieved_docs):
    # RAG + COT
    output_parser = StrOutputParser()
    pass_inputs = RunnablePassthrough()
    retriever_branch = itemgetter("new_name") | retrieved_docs | _combine_documents
    generate_new_name = {
        "new_name": {"resp": modified_name_prompt | LLM_model | output_parser}
        | strip_name_prompt
        | LLM_model
        | output_parser,
        "inputs": RunnablePassthrough(),
    }
    prompt_inputs = {
        "inputs": lambda x: x["inputs"]
        | {"new_name": x["new_name"], "retrieved_recipes": x["retrieved"]}
    }
    parallels = {
        "retrieved": retriever_branch,
        "new_name": itemgetter("new_name"),
        "inputs": itemgetter("inputs"),
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
