import streamlit as st
import requests
import uuid
import atexit
import logging
import warnings
import time
import logging

from datetime import datetime, timedelta

from langchain.chat_models import ChatOpenAI as OpenAI

from connections import ChromaConnection, TestConnection, BaseConnection, StubLLM

logger = logging.getLogger(__name__)

RETRIEVER_API_BASE = st.secrets["RETRIEVER_API_BASE"]
RETRIEVER_API_SECRET = st.secrets["RETRIEVER_API_SECRET"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if "TEST_ENV" in st.secrets:
    TEST_ENV = True
else:
    TEST_ENV = False


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


def refresh_key(conn: BaseConnection) -> None:
    if "key" not in st.session_state:
        st.session_state.key = conn.get_connection()
        update_expire_date(n_days=1)
    elif datetime.utcnow() >= st.session_state.key_expire_datetime:
        conn.renew_key(st.session_state.key)
        update_expire_date(n_days=1)
    else:
        pass


@st.cache_resource
def init_retriever() -> BaseConnection:
    if TEST_ENV:
        return TestConnection()
    else:
        return ChromaConnection(
            base_url=RETRIEVER_API_BASE, api_secret=RETRIEVER_API_SECRET
        )


@st.cache_resource
def init_llm():
    if TEST_ENV:
        return StubLLM()
    else:
        return OpenAI(model="gpt-3.5-turbo-1106")


@st.cache_data
def register_shutdown_handler(_conn: BaseConnection):
    atexit.register(_conn.revoke_key, key=st.session_state.key)
    atexit.register(logger.debug, "Running exit routine.")


@st.cache_data
def query_llm(query):
    return st.session_state.llm.query(OPENAI_API_KEY, query)


@st.cache_data
def query_retriever(query: str):
    return st.session_state.chroma_connection.retrieve_documents(
        st.session_state.key, query
    )


st.session_state.chroma_connection = init_retriever()
st.session_state.llm = init_llm()
refresh_key(st.session_state.chroma_connection)
register_shutdown_handler(st.session_state.chroma_connection)

# st.set_page_config(layout="wide")
modify_col_widths = [1.3, 2, 0.5, 2]
add_another_row_button_label = "Add Another"

ingredient_input_key = "ingredient_input_"
instruction_input_key = "instruction_input_"
selectbox_key_base = "modify_select_"
modify_button_key_base = "base_modify_button_"
modify_text_key_base = "base_modify_"
modify_end_text_key_base = "end_modify_"

st.title("Recipe Helper")
st.write("Instructions for app goes here")


def extend(key: str):
    st.session_state[key] += 1


def extend_button(key: str, calling_widget: str):
    extend(key)
    del st.session_state[calling_widget]


def shorten(key: str):
    st.session_state[key] += -1


def set_default_state(key: str, default_state):
    if key not in st.session_state:
        st.session_state[key] = default_state


def gather(key_count, key_base) -> list[str]:
    gathered = list()
    for i in range(st.session_state[key_count]):
        gathered.append(st.session_state[key_base + str(i)])
    return gathered


def gather_inputs() -> dict[str, list[str]]:
    ingredients = gather("n_ingredient", ingredient_input_key)
    instructions = gather("n_instruction", instruction_input_key)

    modifications = list()
    for i in range(st.session_state.n_modification):
        selectbox_key = selectbox_key_base + str(i)
        modify_type = st.session_state[selectbox_key]
        match modify_type:
            case "Replace":
                modify_input = " "
                modify_input = modify_input.join(
                    [
                        st.session_state[modify_text_key_base + str(i)],
                        "with",
                        st.session_state[modify_end_text_key_base + str(i)],
                    ]
                )
            case "Add":
                modify_input = st.session_state[modify_text_key_base + str(i)]
            case "Remove":
                modify_input = st.session_state[modify_text_key_base + str(i)]
            case "Vegetarian":
                modify_input = ""
            case "Keto":
                modify_input = ""
            case _:
                modify_input = ""
        modifications.append(modify_type + " " + modify_input)

    return {
        "Recipe Name": [st.session_state["name_input"]],
        "Ingredients": ingredients,
        "Instructions": instructions,
        "Modifications": modifications,
    }


def format_inputs(inputs: dict) -> str:
    input_strings = list()
    for key, value in inputs.items():
        sub_input = list()
        sub_input.append(key + "\n")
        sub_input.append("\n".join(value))
        input_strings.append("\n".join(sub_input))
    return "\n\n".join(input_strings)


def assemble_inputs(inputs, docs) -> str:
    return ""


set_default_state("n_ingredient", 1)
set_default_state("n_instruction", 1)
set_default_state("n_modification", 1)
set_default_state("modify_rows", list())

st.text_input(
    label="Recipe Name",
    value="Recipe Name",
    key="name_input",
    label_visibility="collapsed",
)

with st.container():
    st.subheader("Ingredients")
    for i in range(st.session_state.n_ingredient):
        if i < st.session_state.n_ingredient - 1:
            st.text_input(
                label="Ingredient",
                key="ingredient_input_" + str(i),
                label_visibility="collapsed",
            )
        else:
            st.text_input(
                label="Ingredient",
                key="ingredient_input_" + str(i),
                label_visibility="collapsed",
                on_change=extend,
                args=("n_ingredient",),
            )
    if st.session_state.n_ingredient > 1:
        st.button("Delete last ingredient", on_click=shorten, args=("n_ingredient",))

with st.container():
    st.subheader("Instructions")
    header = st.columns([1, 32])
    for i in range(st.session_state.n_instruction):
        header = st.columns([1, 16])
        header[0].markdown(
            '<span style="font-size:1.6em;">' + str(i + 1) + ". </span>",
            unsafe_allow_html=True,
        )

        if i < st.session_state.n_instruction - 1:
            header[1].text_input(
                label="instructions",
                key=instruction_input_key + str(i),
                label_visibility="collapsed",
            )
        else:
            header[1].text_input(
                label="instructions",
                key=instruction_input_key + str(i),
                label_visibility="collapsed",
                on_change=extend,
                args=("n_instruction",),
            )
    if st.session_state.n_instruction > 1:
        st.button("Delete last instruction", on_click=shorten, args=("n_instruction",))

with st.container():
    st.session_state.modify_rows = list()
    st.subheader("Modifications")
    modify_row = st.empty()
    modify = modify_row.columns(modify_col_widths)
    st.session_state.modify_rows.append(modify_row)

    modify[0].write("Customization")
    modify[1].write("")
    modify[2].write("")
    modify[3].write("")

    for i in range(st.session_state.n_modification):
        modify_row = st.empty()
        modify_menu_col = modify_row.columns(
            [modify_col_widths[0], sum(modify_col_widths[1:])]
        )

        selectbox_key = selectbox_key_base + str(i)
        modify_menu_col[0].selectbox(
            "Modification Type",
            ("Replace", "Add", "Remove", "Vegetarian", "Keto"),
            key=selectbox_key,
            label_visibility="collapsed",
        )

        modify_text_cols = modify_menu_col[1].empty()
        modify_text_cols.empty()

        text_inputs = modify_text_cols.columns(modify_col_widths[1:])

        button_key = modify_button_key_base + str(i)
        text_box_key = modify_text_key_base + str(i)
        text_box_end_key = modify_end_text_key_base + str(i)

        if i < st.session_state.n_modification - 1:
            match st.session_state[selectbox_key]:
                case "Replace":
                    text_inputs[0].text_input(
                        label="base modify",
                        key=text_box_key,
                        label_visibility="collapsed",
                    )
                    text_inputs[1].markdown(
                        '<div style="font-size:1.6em;text-align:center">with</div>',
                        unsafe_allow_html=True,
                    )

                    text_inputs[2].text_input(
                        label="end ingredient",
                        key=text_box_end_key,
                        label_visibility="collapsed",
                    )
                case "Add":
                    text_inputs[0].text_input(
                        label="base modify",
                        key=text_box_key,
                        label_visibility="collapsed",
                    )
                case "Remove":
                    pass
                    text_inputs[0].text_input(
                        label="base modify",
                        key=text_box_key,
                        label_visibility="collapsed",
                    )
                case "Vegetarian":
                    pass
                case "Keto":
                    pass
                case _:
                    raise NotImplementedError(
                        "Select choice is outside of accepted parameters."
                    )
        else:
            match st.session_state[selectbox_key]:
                case "Replace":
                    text_inputs[0].text_input(
                        label="base modify",
                        key=text_box_key,
                        label_visibility="collapsed",
                    )
                    text_inputs[1].markdown(
                        '<div style="font-size:1.6em;text-align:center">with</div>',
                        unsafe_allow_html=True,
                    )

                    text_inputs[2].text_input(
                        label="end ingredient",
                        key=text_box_end_key,
                        label_visibility="collapsed",
                        on_change=extend,
                        args=("n_modification",),
                    )
                case "Add":
                    text_inputs[0].text_input(
                        label="base modify",
                        key=text_box_key,
                        label_visibility="collapsed",
                        on_change=extend,
                        args=("n_modification",),
                    )
                case "Remove":
                    text_inputs[0].text_input(
                        label="base modify",
                        key=text_box_key,
                        label_visibility="collapsed",
                        on_change=extend,
                        args=("n_modification",),
                    )
                case "Vegetarian":
                    text_inputs[0].button(
                        add_another_row_button_label,
                        key=button_key,
                        on_click=extend_button,
                        args=("n_modification", button_key),
                    )
                case "Keto":
                    text_inputs[0].button(
                        add_another_row_button_label,
                        key=button_key,
                        on_click=extend_button,
                        args=("n_modification", button_key),
                    )
                case _:
                    raise NotImplementedError(
                        "Select choice is outside of accepted parameters."
                    )
    if st.session_state.n_modification > 1:
        st.button(
            "Delete last modification", on_click=shorten, args=("n_modification",)
        )

st.divider()
with st.form("Input", border=False):
    submit = st.form_submit_button("Modify my Recipe!")
    if submit:
        with st.empty():
            st.write("Submitted!")
            with st.spinner("Retrieving..."):
                time.sleep(1)
                gathered_inputs = format_inputs(gather_inputs())
                docs = query_retriever(gathered_inputs)
            if isinstance(docs, list):
                with st.spinner("Asking the LLM..."):
                    llm_query = assemble_inputs(gathered_inputs, docs)
                    llm_response = query_llm(llm_query)
            else:
                llm_response = "Error in querying retriever API. Try again later."
            # st.write(format_inputs(gather_inputs()))
            st.write(llm_response)
