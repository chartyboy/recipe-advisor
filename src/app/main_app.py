import streamlit as st
import requests

from datetime import datetime, timedelta
from urllib.parse import urljoin
from chain import initialize_retriever_chain, initialize_LLM_chain, naming_chain
from prompts import BASE_CONTEXT_INPUTS
from langchain.llms import OpenAI

st.title("Recipe Helper")

RETRIEVER_API_BASE = st.secrets["RETRIEVER_API_BASE"]
RETRIEVER_API_PASSWORD = st.secrets["RETRIEVER_API_PASSWORD"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


def model_handle_hash_func(model: OpenAI):
    return model.model_name


# Connect to retriever and authenticate
def get_chroma_connection():
    auth_payload = {"username": "streamlit", "password": RETRIEVER_API_PASSWORD}
    token = requests.post(
        urljoin(RETRIEVER_API_BASE, "token"), params=auth_payload
    ).json()
    st.session_state.token = token["access_token"]
    st.session_state.token_expire = token["exp"]


def refresh_token():
    if "token" not in st.session_state:
        get_chroma_connection()
    elif datetime.utcnow() > st.session_state.token_expire:
        get_chroma_connection()
    return


# Initialize LLM chain
@st.cache_data
def get_openai_model(model_name="gpt-3.5-turbo-1106"):
    return OpenAI(model_name=model_name)


naming_chain_cached = st.cache_data(
    naming_chain, hash_funcs={OpenAI: model_handle_hash_func}
)


# Functions to make API calls to LLM chain
def retrieve_documents(query, n_docs=1):
    headers = {"Authentication": st.session_state.token}
    payload = {"query": query, "collection_name": "summed", "n_docs": n_docs}
    docs = requests.post(
        urljoin(RETRIEVER_API_BASE, "retrieve"), headers=headers, params=payload
    ).json()
    return docs["docs"]


refresh_token()
model = get_openai_model()
naming = naming_chain_cached(model)
# Get user input
with st.form(""):
    form_cols = st.columns([])
    old_name = ""
    submit = st.form_submit_button("Modify my recipe!")
    user_inputs = dict()
    if submit:
        chain_inputs = BASE_CONTEXT_INPUTS | user_inputs
        new_name = naming.invoke(chain_inputs)
        context_docs = retrieve_documents(new_name)
        response_chain = initialize_LLM_chain(model, context_docs)
        res = response_chain.invoke(chain_inputs)
    # Update with results
