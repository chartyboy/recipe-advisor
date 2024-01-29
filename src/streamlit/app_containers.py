import streamlit as st

from dependency_injector import containers, providers
from connections import ChromaConnection
from langchain.chat_models import ChatOpenAI as OpenAI


class LLMContainer(containers.DeclarativeContainer):
    chroma_connection = providers.Singleton(ChromaConnection)
    llm = providers.Object(OpenAI(model="gpt-3.5-turbo-1106"))
    test_obj = providers.Object("test")


class SecretsContainer(containers.DeclarativeContainer):
    RETRIEVER_API_BASE = st.secrets["RETRIEVER_API_BASE"]
    RETRIEVER_API_SECRET = st.secrets["RETRIEVER_API_SECRET"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
