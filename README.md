# Recipe Advisor

## Table of Contents
[Features](#features)  
[Get Started](#get-started)  
[Deployment](#deployment)
[Credits](#credits)

## Features
##### REST API with Chroma for Document Retrieval
+ Chroma database for storing text embeddings
+ REST API for document retrieval from database
+ HTTPS support
+ API key auth

##### Streamlit App UI
+ Collects user input and makes requests to the document retrieval API.
+ Queries locally-run HuggingFaceHub LLMs or OpenAI's API to create user-requested recipes.

Try the app out on the [Streamlit Community Cloud](https://recipe-advisor-hkdqtdh9jhyqmcmqjdckvr.streamlit.app/)!
![Streamlit App Example](/references/images/app_example.gif)

## Get Started
```git clone https://github.com/chartyboy/recipe-advisor.git```
### Text Embeddings
Download the Chroma vector database with precomputed embeddings or create a Chroma database with your
own embeddings from the raw text files.  
+ [Embeddings for BAAI/bge-small-en-v1.5](https://drive.google.com/drive/folders/12zdpaWa2vwjLVAEVsXTAix0xNpy0_dfL?usp=sharing)  
+ [Embeddings for BAAI/bge-large-en](https://drive.google.com/drive/folders/1Jkwr81F1z5nJ2FxIY72gv9wdQoBn9PpB?usp=sharing)  
+ [Raw recipe text (JSONLines)](https://drive.google.com/drive/folders/1tew7C_h2T0znZ5jUvN7jcQ9QWf0hxQ1l?usp=sharing)

### Embedding Models
Download the embeddings model according to the type of embeddings downloaded above.  
+ [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
+ [BAAI/bge-large-en](https://huggingface.co/BAAI/bge-large-en)

### Requirements
#### Installing Required Packages
From the root repo directory,
```
pip install --ignore-requires-python -r requirements.txt
``` 
This installs all the packages
necessary for development on this repo. Other ```requirements.txt``` files, such as those located in
```docker/retrieval``` or ```src/streamlit``` are minimal package lists for running the respective app.
The CPU-only version of ```pytorch``` is installed by default. To enable the usage of CUDA when running pytorch models, follow  [pytorch's CUDA installation insructions](https://pytorch.org/).  

## Deployment 
### Docker (Recommended)

#### Environment Variables

##### API
Edit sample.env in /docker/retrieval and rename it to .env.  
+ ```EMBEDDING_MODEL_NAME```: Path to HuggingFaceHub text embedding model. For example, 
the path to this [model](https://huggingface.co/Salesforce/SFR-Embedding-Mistral) would be Salesforce/SFR-Embedding-Mistral.
+ ```DB_DATA_PATH```: Path to Chroma vector database.
+ ```RETRIEVER_DOMAIN_NAME```: Domain name pointing to the API.
To run locally, set the domain name to ```localhost```.
+ ```API_SECRET_KEY```: Secret key for authenticating the generation of new API keys. Can be any valid string.
+ ```FASTAPI_SIMPLE_SECURITY_DB_PATH``` (optional): Path database for storing API keys. Defaults to ```/sqlite.db```
+ ```FAST_API_SIMPLE_SECURITY_AUTOMATIC_EXPIRATION_DAYS``` (optional): Number of days for an API key to expire after
generation. Defaults to 15 days.
+ ```EMBED_MODEL_SOURCE```: Path to local cache for HuggingFaceHub models. Typically, models are stored at
```~/.cache/torch/sentence_transformers```.
+ ```EMBED_MODEL_CACHE``` (optional): Path to store models in the container. Defaults to ```/models```.

##### Streamlit app
Edit example_secrets.toml in /docker/streamlit and rename it to secrets.toml.
+ ```RETRIEVER_API_BASE```: Domain name pointing to the API
+ ```RETRIEVER_API_SECRET```: Secret key for authenticating the generation of new API keys. This should
be identical to ```API_SECRET_KEY``` above.

In addition, at least one of the two variables need to be configured.
+ ```OPENAI_API_KEY```: Client API key for using OpenAI's LLM API.
+ ```HUGGINGFACE_MODEL_PATH```: Path to HuggingFaceHub text generation/conversational LLM. For example, the path to use this [model](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) in the app would be ```mistralai/Mixtral-8x7B-Instruct-v0.1```. If a model is specified, it will be used instead of the OpenAI API.

#### Retriever API and Chroma Database
From the root repo directory,  
```
cd docker/retrieval 
docker compose -f retriever.yml up
```
#### Streamlit App
From the root repo directory,
```
cd docker/streamlit
docker compose -f streamlit.yml up
```

Alternatively, the Streamlit app can be deployed on the [Streamlit Community Cloud](https://streamlit.io/cloud).

### Deploy Locally
Deploying the retriever API and embeddings database in this manner is not recommended, as it exposes the API secret
and other API keys over HTTP.

#### Requirements
Follow the instructions in the [Installing Required Packages](#installing-required-packages) section above.

#### Environment Variables
Edit sample.env in the root repo directory and rename it to .env. Edit example_secrets.toml in /src/streamlit/.streamlit and rename it to secrets.toml.

#### Retriever API
From the root repo directory,
```
uvicorn src.retriever.api:app --host localhost --port 8001
```
This launches a FastAPI instance at localhost:8001. Try out the endpoints at the /docs endpoint.

#### Chroma Database
From the root repo directory,
```
chroma run --path <path-to-database> --host localhost  --port 8000
```

This launches a Chroma server instance at localhost:8000.

#### Streamlit App
Try the app out on the [Streamlit Community Cloud](https://recipe-advisor-hkdqtdh9jhyqmcmqjdckvr.streamlit.app/).

##### Running the Streamlit App Locally
From the root repo directory,
```
cd src/streamlit
streamlit run app.py
```

This launches a Streamlit app instance at localhost:8501. For additional deployment options,
try ```streamlit run --help```.

## Credits
[https-portal](https://github.com/SteveLTN/https-portal) by [SteveLTN](https://github.com/SteveLTN)  
[fastapi_simple_security](https://github.com/mrtolkien/fastapi_simple_security) by [mrtolkien](https://github.com/mrtolkien)  
[Llama-2-13B-chat-GPTQ](https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ) by [Tom Jobbins](https://huggingface.co/TheBloke)  
[bge-large-en](https://huggingface.co/BAAI/bge-large-en) and [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) by [BAAI](https://www.baai.ac.cn/english.html)
