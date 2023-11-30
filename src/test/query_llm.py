import torch

from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load embeddings
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

db = Chroma(persist_directory="./chroma_db", embedding_function=hf)

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
db_retriever = db.as_retriever()

# Question param is hardcoded in source
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

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_retriever,
    chain_type_kwargs={
        "prompt": prompt,
        "document_variable_name": "context",
        "document_prompt": document_prompt,
    },
)

# Query param to be used for retrieval and insertion into question param in prompt
qa.run(query="Why is the sky blue?")
