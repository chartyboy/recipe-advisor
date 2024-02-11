"""
Streamlit app for receiving user input.
"""

import streamlit as st
import atexit
import logging
import warnings
import time
import logging
import sys
import warnings
import re

warnings.filterwarnings("ignore")

from datetime import datetime, timedelta

from langchain_community.chat_models import ChatOpenAI as OpenAI
from connections import ChromaConnection, BaseConnection, TestConnection

from chain import initialize_LLM_chain, naming_chain

# logging.basicConfig(level=logging.INFO)


@st.cache_resource()
def init_loggers():
    """
    Initialize Python loggers with specified formatting.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s\n")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = init_loggers()


RETRIEVER_API_BASE = st.secrets["RETRIEVER_API_BASE"]
RETRIEVER_API_SECRET = st.secrets["RETRIEVER_API_SECRET"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if "TEST_ENV" in st.secrets:
    TEST_ENV = True
else:
    TEST_ENV = False


def days_offset_from_current_time(n_days=1) -> datetime:
    """
    Generate a datetime object n days from the current time.
    """
    return datetime.utcnow() + timedelta(days=n_days)


def update_expire_date(n_days: int = 1):
    """
    Update the API key expiration date.
    """
    if "key" not in st.session_state:
        raise RuntimeError("API key not configured yet.")

    if n_days <= 0:
        warnings.warn(
            "Negative amount of days passed as an argument. The current \
            expire date will not be changed."
        )
        return

    st.session_state.key_expire_datetime = days_offset_from_current_time(n_days=1)
    # st.session_state.key_expire = st.session_state.key_expire_datetime.timestamp()
    logger.info(f"API key expires at {st.session_state.key_expire_datetime}")


def validate_time(cached_data: dict) -> bool:
    """
    Checks if the currently stored API key has expired.
    """
    expire_time = cached_data["expires"]
    if "key_expire_datetime" in st.session_state:
        if expire_time < st.session_state.key_expire_datetime:
            expire_time = st.session_state.key_expire_datetime

    # Key has expired
    if datetime.utcnow() >= expire_time:
        return False
    else:
        return True


@st.cache_resource(ttl="1d", validate=validate_time)
def get_key() -> dict:
    """
    Get a new key from the retriever API.
    """
    new_key = st.session_state.chroma_connection.get_connection()
    expire_time = days_offset_from_current_time(n_days=1)
    return {"key": new_key, "expires": expire_time}


def refresh_key(conn: BaseConnection) -> None:
    """
    Get a new key or renew an existing one from the API.
    """
    if "key" not in st.session_state:
        key_data = get_key()
        st.session_state.key = key_data["key"]
        st.session_state.key_expire_datetime = key_data["expires"]
    elif datetime.utcnow() >= st.session_state.key_expire_datetime:
        conn.renew_key(st.session_state.key)
        update_expire_date(n_days=1)
    else:
        pass


@st.cache_resource
def init_retriever() -> BaseConnection:
    """
    Initializes the retriever API connection.
    """
    logger.info("Creating new retriever connection")
    if TEST_ENV:
        # return TestConnection()
        conn = ChromaConnection(
            base_url=RETRIEVER_API_BASE, api_secret=RETRIEVER_API_SECRET
        )
        conn._verify = False
        return conn
    else:
        conn = ChromaConnection(
            base_url=RETRIEVER_API_BASE, api_secret=RETRIEVER_API_SECRET
        )
        conn._verify = False
        return conn


@st.cache_resource
def init_llm():
    """
    Initializes LLM objects.
    """
    import torch
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    if TEST_ENV:
        pipe = pipeline(
            "text-generation",
            model="GeneZC/MiniChat-2-3B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            # max_new_tokens=512,
            # do_sample=True,
            # temperature=0.7,
            # top_k=50,
            # top_p=0.95,
            # repetition_penalty=1.15
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info("Test env initialized")
        return llm
        # return OpenAI(model="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY)
    else:
        return OpenAI(model="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY)


@st.cache_resource
def register_shutdown_handler(_conn: BaseConnection, key):
    """
    Registers a function to revoke the API key on shutdown.
    """
    logger.info("Registering shutdown handler")
    atexit.register(_conn.revoke_key, key=key)
    atexit.register(logger.info, "Running exit routine.")


@st.cache_data
def query_llm(chain_inputs: dict, _docs):
    """
    Sends a query to the LLM and returns the response.
    """
    llm_chain = initialize_LLM_chain(st.session_state.llm, _docs)
    return llm_chain.invoke(chain_inputs)


@st.cache_data
def query_new_name(chain_inputs: dict):
    name_chain = naming_chain(st.session_state.llm)
    return name_chain.invoke(chain_inputs)


@st.cache_data(ttl=timedelta(minutes=1))
def query_retriever(query: str, n_docs=1, collection_name="summed") -> dict[str, str]:
    """
    Sends a query to the retriever API and returns the response.
    """
    result = st.session_state.chroma_connection.retrieve_documents(
        st.session_state.key, query, n_docs=n_docs, collection_name=collection_name
    )
    if isinstance(result, dict):
        return result
    else:
        logger.info(result)
        return dict()


@st.cache_data
def get_by_ids_retriever(ids: list[str], collection_name="summed"):
    """
    Get documents by their database IDs from the retriever API.
    """
    return st.session_state.chroma_connection.get_documents(
        st.session_state.key, ids, collection_name=collection_name
    )


st.session_state.chroma_connection = init_retriever()
st.session_state.llm = init_llm()
refresh_key(st.session_state.chroma_connection)
register_shutdown_handler(st.session_state.chroma_connection, st.session_state.key)

is_alive = st.session_state.chroma_connection.heartbeat()
if not is_alive:
    st.warning("Unable to connect to database server. Try again later.")
    st.stop()


# App consts
# st.set_page_config(layout="wide")
modify_col_widths = [1.3, 2, 0.5, 2]
add_another_row_button_label = "Add Another"

ingredient_input_key = "ingredient_input_"
instruction_input_key = "instruction_input_"
selectbox_key_base = "modify_select_"
modify_button_key_base = "base_modify_button_"
modify_text_key_base = "base_modify_"
modify_end_text_key_base = "end_modify_"


# UI helper functions
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


def gather_by_key(key_count, key_base) -> list[str]:
    gathered = list()
    for i in range(st.session_state[key_count]):
        gathered.append(st.session_state[key_base + str(i)])
    return gathered


def gather_inputs() -> dict[str, list[str] | str]:
    """
    Collect user inputs from text boxes.
    """
    ingredients = gather_by_key("n_ingredient", ingredient_input_key)
    instructions = gather_by_key("n_instruction", instruction_input_key)

    modifications = dict()
    for i in range(st.session_state.n_modification):
        selectbox_key = selectbox_key_base + str(i)
        modify_type = st.session_state[selectbox_key]
        match modify_type:
            case "Replace":
                modify_input = [
                    st.session_state[modify_text_key_base + str(i)].strip(),
                    st.session_state[modify_end_text_key_base + str(i)].strip(),
                ]
                if (
                    st.session_state[modify_text_key_base + str(i)].strip() == ""
                    or st.session_state[modify_end_text_key_base + str(i)].strip() == ""
                ):
                    modify_input = ""
            case "Add":
                modify_input = st.session_state[modify_text_key_base + str(i)].strip()
            case "Remove":
                modify_input = st.session_state[modify_text_key_base + str(i)].strip()
            case "Vegetarian":
                modify_input = ""
            case "Keto":
                modify_input = ""
            case _:
                modify_input = ""
        if modify_input == "" and modify_type not in {"Vegetarian", "Keto"}:
            pass
        else:
            if modify_type in modifications.keys():
                pass
            else:
                modifications[modify_type] = list()
            modifications[modify_type].append(modify_input)
    formatted_modifications = format_customization(modifications)
    if st.session_state.is_filled_recipe:
        recipe_name = st.session_state["select_recipe"]
    else:
        recipe_name = st.session_state["name_input"]
    return {
        "Recipe Name": [recipe_name],
        "Ingredients": ingredients,
        "Instructions": instructions,
        "Modifications": formatted_modifications,
    }


def format_recipe(inputs: dict) -> dict[str, str]:
    """
    Formats user input into a recipe format.
    """
    input_strings = list()
    recipe = dict()
    for key, value in inputs.items():
        sub_input = list()
        sub_input.append(key + ":")
        sub_input.append("\n".join(value))

        sub_string = "\n".join(sub_input)

        input_strings.append(sub_string)

        recipe[key] = sub_string
    recipe["recipe"] = "\n\n".join(input_strings)
    # logger.info(recipe["recipe"])
    return recipe


def format_customization(modifications: dict[str, str | list[str]]) -> str:
    """
    Formats the user requested modifications into a series of requests.
    """
    format_strings = list()
    has_added_vegetarian = False
    has_added_keto = False

    for modify_type in modifications.keys():
        for input in modifications[modify_type]:
            match modify_type:
                case "Replace":
                    to_remove = input[0]
                    to_include = input[1]
                    format_strings.append(
                        "replace the "
                        + to_remove
                        + " from the original recipe with "
                        + to_include
                    )
                case "Add":
                    format_strings.append("add " + input + " to the recipe")
                case "Remove":
                    format_strings.append("remove " + input + " from the recipe")
                case "Vegetarian":
                    if has_added_vegetarian:
                        pass
                    else:
                        has_added_vegetarian = True
                        format_strings.append(
                            "make the recipe suitable for vegetarians"
                        )
                case "Keto":
                    if has_added_keto:
                        pass
                    else:
                        has_added_keto = True
                        format_strings.append(
                            "make the recipe suitable for a ketogenic diet"
                        )

    if len(format_strings) > 1:
        format_strings[-1] = "and " + format_strings[-1]
    formatted_customization = ", ".join(format_strings)
    return formatted_customization


def find_recipes():
    """
    Query the retriever API to find recipe names similar to the user input.
    """
    query = st.session_state["name_input"]
    retrieved_names = query_retriever(query, n_docs=5, collection_name="name")
    st.session_state.names_ids = dict()

    if isinstance(retrieved_names, dict):
        logger.info(f"{query},{retrieved_names}")
    else:
        logger.info(retrieved_names)
        st.warning("Could not query the database. Try again later.")
        return

    st.session_state.show_search_results = True

    # Need to account for duplicate names
    names = dict()
    st.session_state.is_duplicate = {id: False for id in retrieved_names.keys()}
    for recipe_id, recipe_name in retrieved_names.items():
        if recipe_name in names.keys():
            old_name = recipe_name
            recipe_name = recipe_name + " " + str(names[recipe_name])
            names[old_name] += 1
            st.session_state.is_duplicate[recipe_id] = True
        else:
            names[recipe_name] = 1
        st.session_state.names_ids[recipe_name] = recipe_id


def get_recipe_data():
    """
    Get the recipe ingredients and instructions from a recipe's name and database ID.
    """
    recipe_name = st.session_state["select_recipe"]
    id = st.session_state.names_ids[recipe_name]
    st.session_state.ingredient = get_by_ids_retriever(
        [id], collection_name="ingredient"
    )
    st.session_state.instruction = get_by_ids_retriever(
        [id], collection_name="instruction"
    )


def reset_search_results():
    st.session_state.show_search_results = False


def reset_fill():
    st.session_state.current_ingredient = [""]
    st.session_state.current_instruction = [""]
    st.session_state.n_ingredient = 1
    st.session_state.n_instruction = 1
    st.session_state.n_modification = 1
    st.session_state.is_filled_recipe = False


# App layout
st.title("Recipe Helper")
st.write(
    "Describe your recipe in the sections below. Include the customizations you want for your recipe, then hit the 'Modify my Recipe!' button"
)

set_default_state("n_ingredient", 1)
set_default_state("n_instruction", 1)
set_default_state("n_modification", 1)
set_default_state("modify_rows", list())
set_default_state("show_search_results", False)
set_default_state("retrieved_recipes", dict())
set_default_state("ingredient", "")
set_default_state("instruction", "")
set_default_state("current_ingredient", [""])
set_default_state("current_instruction", [""])
set_default_state("is_filled_recipe", False)

st.subheader("Recipe Name")
name_cols = st.columns([3, 1])
name_cols[0].text_input(
    label="Recipe Name",
    value="Recipe Name",
    key="name_input",
    label_visibility="collapsed",
    on_change=reset_search_results,
)
name_cols[1].button("Find Recipe", on_click=find_recipes)
if st.session_state.show_search_results:
    with st.form(key="search"):
        st.selectbox(
            "Search Results",
            key="select_recipe",
            options=st.session_state.names_ids.keys(),
        )
        submit = st.form_submit_button(label="Fill Recipe")
        if submit:
            get_recipe_data()
            st.session_state.is_filled_recipe = True
            id = st.session_state.names_ids[st.session_state["select_recipe"]]
            ingredient = st.session_state.ingredient[id].split("\n")
            instruction = st.session_state.instruction[id].replace(".,", ".")
            instruction = instruction.split("\n")

            ingredient.append("")
            instruction.append("")

            st.session_state.current_ingredient = ingredient
            st.session_state.n_ingredient = len(ingredient)

            st.session_state.current_instruction = instruction
            st.session_state.n_instruction = len(instruction)

            recipe_id = st.session_state.names_ids[st.session_state["select_recipe"]]
            # if st.session_state.is_duplicate[recipe_id]:
            #     st.session_state["name_input"] = st.session_state["select_recipe"][:-2]
            # else:
            #     st.session_state["name_input"] = st.session_state["select_recipe"]

            # for item in ingredient:
            #     st.write(item)
            # st.write("")
            # for item in instruction:
            #     st.write(item)

opt_cols = st.columns([1, 1, 2])
opt_cols[0].checkbox(label="Use multiline input", key="text_checkbox")
opt_cols[1].button(label="Reset Recipe", on_click=reset_fill)

with st.container():
    st.subheader("Ingredients")
    if st.session_state["text_checkbox"]:
        st.text_area(
            label="Ingredient",
            value="\n".join(st.session_state.current_ingredient),
            key=ingredient_input_key + str(1),
            label_visibility="collapsed",
        )
    else:
        for i in range(st.session_state.n_ingredient):

            if i >= len(st.session_state.current_ingredient):
                text_value = ""
            else:
                text_value = st.session_state.current_ingredient[i]

            if i < st.session_state.n_ingredient - 1:
                st.text_input(
                    label="Ingredient",
                    value=text_value,
                    key="ingredient_input_" + str(i),
                    label_visibility="collapsed",
                )
            else:
                st.text_input(
                    label="Ingredient",
                    value=text_value,
                    key="ingredient_input_" + str(i),
                    label_visibility="collapsed",
                    on_change=extend,
                    args=("n_ingredient",),
                )
        if st.session_state.n_ingredient > 1:
            st.button(
                "Delete last ingredient", on_click=shorten, args=("n_ingredient",)
            )

with st.container():
    st.subheader("Instructions")
    header = st.columns([1, 32])
    if st.session_state["text_checkbox"]:
        st.text_area(
            label="Ingredient",
            value="\n".join(st.session_state.current_instruction),
            key=instruction_input_key + str(1),
            label_visibility="collapsed",
        )
    else:
        for i in range(st.session_state.n_instruction):
            if i >= len(st.session_state.current_instruction):
                text_value = ""
            else:
                text_value = st.session_state.current_instruction[i]

            header = st.columns([1, 16])
            header[0].markdown(
                '<span style="font-size:1.6em;">' + str(i + 1) + ". </span>",
                unsafe_allow_html=True,
            )

            if i < st.session_state.n_instruction - 1:
                header[1].text_input(
                    label="instructions",
                    value=text_value,
                    key=instruction_input_key + str(i),
                    label_visibility="collapsed",
                )
            else:
                header[1].text_input(
                    label="instructions",
                    value=text_value,
                    key=instruction_input_key + str(i),
                    label_visibility="collapsed",
                    on_change=extend,
                    args=("n_instruction",),
                )
        if st.session_state.n_instruction > 1:
            st.button(
                "Delete last instruction", on_click=shorten, args=("n_instruction",)
            )

with st.container():
    st.session_state.modify_rows = list()
    st.subheader("Modifications")
    modify_row = st.empty()
    modify = modify_row.columns(modify_col_widths)
    st.session_state.modify_rows.append(modify_row)

    # modify[0].write("Customization")
    # modify[1].write("")
    # modify[2].write("")
    # modify[3].write("")

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
    submit = st.form_submit_button("Modify my recipe!")
    if submit:
        with st.empty():
            st.write("Submitted!")

            with st.spinner("Retrieving..."):
                time.sleep(1)
                gathered = gather_inputs()
                # logger.info(gathered)

                recipe_input = {
                    k: gathered[k]
                    for k in ["Recipe Name", "Ingredients", "Instructions"]
                }
                gathered_inputs = format_recipe(recipe_input)
                # logger.info(gathered_inputs)
                docs = list(
                    query_retriever(gathered_inputs["recipe"], n_docs=2).values()
                )
                logger.info(docs)
            if isinstance(docs, list):
                if len(docs) >= 1 and st.session_state.is_filled_recipe:
                    docs = [docs[-1]]
                else:
                    docs = [docs[0]]
                docs_content = "\n\n".join(docs)

                with st.spinner("Asking the LLM..."):
                    chain_inputs = {
                        "recipe_name": gathered_inputs["Recipe Name"],
                        "customization": gathered["Modifications"],
                        "user_recipe": gathered_inputs["recipe"],
                    }

                    new_name = query_new_name(chain_inputs)
                    logger.info(new_name)
                    chain_inputs["new_name"] = new_name["new_name"]

                    result = query_llm(chain_inputs, docs_content)
                    llm_response = result["output"]
                    logger.info(result)
                    # logger.info(type(llm_response))
            else:
                llm_response = "Error in querying retriever API. Try again later."

            with st.expander(label="Result", expanded=True):
                st.write(llm_response)
