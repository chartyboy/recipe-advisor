import torch

from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import SequentialChain
from langchain.memory import SimpleMemory
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

names = Chroma(
    collection_name="names", persist_directory="./chroma_db", embedding_function=hf
)

# Load llm
model_path = "TheBloke/Llama-2-13B-chat-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="auto", revision="main"
)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15,
)
llm = HuggingFacePipeline(pipeline=pipe)
# Initialize chain components
names_retriever = names.as_retriever(search_kwargs={"k": 4})

# Question param is hardcoded in source
# Template used in Llama fine-tuning
template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{question}

### Input:
{context}

### Response:
\n
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)

find_base_recipe = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=SimpleMemory(
        memories={
            "question": "Identify the recipe or dish in the input sentence. Limit your response to just the name of the recipe or dish."
        }
    ),
)

analyze_and_modify = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=names_retriever,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context",
        "document_prompt": document_prompt,
    },
)

# For RetrivalQA chain, Query = question param in prompt. Don't ask me why, it's hardcoded in the source code.
input = "Why does steak have chamomile in it?"
res = find_base_recipe.run(context=input)
print(names_retriever.get_relevant_documents(input))
print(res)
pass
# Result:
# To add artichokes to your chicken noodle soup, you could try incorporating them into one of the existing recipes or adding them as a separate ingredient. Here's how you might do it:

# * In the "Coconut, Yam, and Leek Soup" recipe, you could add artichoke hearts to the pot along with the other vegetables. They may need a little extra cooking time, so keep an eye on them and adjust the cooking time accordingly.
# * In the "Instant Pot Shepherd's Pie with Potatoes and Yams" recipe, you could add artichoke hearts to the filling along with the other vegetables. They may also need a little extra cooking time, so keep an eye on them and adjust the cooking time accordingly.
# * In the "Powerhouse African Yam Stew" recipe, you could add artichoke hearts to the pot along with the other vegetables. They may need a little extra cooking time, so keep an eye on them and adjust the cooking time accordingly.

# Keep in mind that artichokes have a strong flavor, so you may want to start with a small amount and adjust to taste. You could also consider adding them towards the end of the cooking time, so they don't get overcooked.

# I hope this helps! Let me know if you have any other questions.
pass
