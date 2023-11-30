from langchain.embeddings import HuggingFaceBgeEmbeddings
from pathlib import Path
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.vectorstores import Chroma

fpath = "./datasets/sample.jl"


def allrecipes_metadata(record: dict, metadata: dict):
    metadata["website"] = r"https://www.allrecipes.com/"
    return metadata

loader = JSONLoader(
    file_path=fpath,
    jq_schema=r"{recipe_name, ingredients, instructions: [.instructions[].text]}",
    json_lines=True,
    text_content=False,
    metadata_func=allrecipes_metadata,
)

data = loader.load()
model_name = "BAAI/bge-large-en"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": False}
# splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, model_name=model_name)
# chunks = splitter.split_documents(data)

hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# Create db, delete existing entries, then insert new ones
db = Chroma.from_documents(documents=data, embedding=hf, persist_directory="./chroma_db")
pass