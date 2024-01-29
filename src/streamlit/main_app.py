import streamlit as st
import requests
import uuid
import atexit
import logging
import warnings

from datetime import datetime, timedelta
from urllib.parse import urljoin
from chain import initialize_retriever_chain, initialize_LLM_chain, naming_chain
from prompts import BASE_CONTEXT_INPUTS
from langchain.llms import OpenAI

logger = logging.getLogger(__name__)

st.title("Recipe Helper")

RETRIEVER_API_BASE = st.secrets["RETRIEVER_API_BASE"]
RETRIEVER_API_SECRET = st.secrets["RETRIEVER_API_SECRET"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]


def update_expire_date(n_days: int = 1):
    if "key" not in st.session_state:
        raise RuntimeError("API key not configured yet.")

    if n_days <= 0:
        warnings.warn(
            "Negative amount of days passed as an argument. The current \
            expire date will not be changed."
        )
        return

    st.session_state.key_expire_datetime = datetime.utcnow() + timedelta(days=n_days)
    st.session_state.key_expire = st.session_state.key_expire_datetime.timestamp()
    logger.info(f"API key expires at {st.session_state.key_expire}")


# Connect to retriever and authenticate
class ChromaConnection:
    def get_connection(self):
        auth_payload = {
            "name": "client_" + uuid.uuid4().hex,
        }
        auth_header = {"secret-key": RETRIEVER_API_SECRET}
        token = requests.get(
            urljoin(RETRIEVER_API_BASE, "/auth/new"),
            params=auth_payload,
            headers=auth_header,
        ).json()

        if token.status_code == 200:
            logger.info("Successfully received new API key")
            st.session_state.key = token["api_key"]
            update_expire_date(n_days=1)
        else:
            logger.debug("Failed to receive new API key")

    def renew_key(self):
        if "key" not in st.session_state:
            return
        else:
            renew_payload = {"api-key": st.session_state.key}
            resp = requests.get(
                urljoin(RETRIEVER_API_BASE, "/auth/renew"), params=renew_payload
            )
            if resp.status_code == 200:
                logger.info("Successfully renewed API key")
                update_expire_date(n_days=1)
            else:
                logger.debug("Failed to renew API key")

    def refresh_key(self):
        if "key" not in st.session_state:
            self.get_connection()
        elif datetime.utcnow() >= st.session_state.key_expire_datetime:
            self.renew_key()
        return

    # Will only be called on shutdown
    def revoke_key(self):
        if "key" not in st.session_state:
            return
        else:
            revoke_payload = {"api-key": st.session_state.key}
            resp = requests.get(
                urljoin(RETRIEVER_API_BASE, "/auth/revoke"), params=revoke_payload
            )
            if resp.status_code == 200:
                logger.info("Successfully revoked API key")


# Injector function for patching in testing
def generate_connection():
    return ChromaConnection()


def register_atexit_func(func, *args, **kwargs):
    if "atexit_registered" not in st.session_state:
        atexit.register(func, *args, **kwargs)
        logger.debug("Shutdown handler registered")
        st.session_state.atexit_registered = True
    else:
        logger.debug("Attempted to re-register shutdown handler")
    return


# Initialize LLM chain
@st.cache_data
def get_openai_model(model_name="gpt-3.5-turbo-1106"):
    return OpenAI(model_name=model_name)


# Define hashing of LLM API for serialization in cache
def model_handle_hash_func(model: OpenAI):
    return model.model_name


naming_chain_cached = st.cache_data(
    func=naming_chain, hash_funcs={OpenAI: model_handle_hash_func}
)


# Functions to make API calls to LLM chain
def retrieve_documents(query, n_docs=1):
    headers = {"api-key": st.session_state.key}
    payload = {"query": query, "collection_name": "summed", "n_docs": n_docs}
    docs = requests.get(
        urljoin(RETRIEVER_API_BASE, "retrieve"), headers=headers, params=payload
    )
    if docs.status_code == 200:
        return docs.json()["docs"]
    else:  # Did not receive successful response
        logger.info(f"Status Code: {docs.status_code} \n Reason: {docs.text}")
        return docs


retriever_api = generate_connection()
retriever_api.refresh_key()
register_atexit_func(retriever_api.revoke_key)

model = get_openai_model()
naming = naming_chain_cached(model)

# Get user input
with st.form(""):
    # Inputs
    form_cols = st.columns([])
    old_name = ""
    # Add name
    name_container = st.container()
    name_container.write("Recipe Name")
    name_container.text_input(
        label="Name", key="name_input", label_visibility="collapsed"
    )

    # Add ingredients
    ingredient_container = st.container()
    ingredient_container.write("Ingredients")
    for i in range(15):
        ingredient_container.text_input(
            label="Ingredient",
            key="ingredient_input_" + str(i),
            label_visibility="collapsed",
        )

    # Add instructions
    instruction_container = st.container()
    for i in range(15):
        instruction_container.text_input(
            label="Instruction",
            key="instruction_input_" + str(i),
            label_visibility="collapsed",
        )

    # Add modifications
    modify_container = st.container()
    for i in range(5):
        instruction_container.text_input(
            label="Instruction",
            key="instruction_input_" + str(i),
            label_visibility="collapsed",
        )

    output_container = st.container()
    # Submit event
    submit = st.form_submit_button("Modify my recipe!")

    user_inputs = dict()
    if submit:
        chain_inputs = BASE_CONTEXT_INPUTS | user_inputs
        new_name = naming.invoke(chain_inputs)
        context_docs = retrieve_documents(new_name)
        response_chain = initialize_LLM_chain(model, context_docs)
        res = response_chain.invoke(chain_inputs)

    # Update with results
