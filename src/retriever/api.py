import chromadb
import os

from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, status, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.docstore.document import Document
from typing import List, Dict, Any, Annotated
from src.features.embeddings import EmbeddingFunctionInterface
from passlib.context import CryptContext
from jose import JWTError, jwt

# Initialize env vars
CHROMA_HOST_ADDRESS = os.getenv("CHROMA_HOST_ADDRESS")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"

# Authentication
oauth2 = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Initialize model
async def database_retriever():
    model_name = "BAAI/bge-large-en"
    # model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    embed_func = EmbeddingFunctionInterface(hf.embed_documents)

    # Connect to db
    chroma_conn = chromadb.HttpClient(host=CHROMA_HOST_ADDRESS)
    return {"embed_func": embed_func, "chroma_conn": chroma_conn}


# Init API
app = FastAPI()

users_db = {
    "streamlit": {
        "username": "streamlit",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}


class TokenData(BaseModel):
    username: str | None = None


class RetrieveRequest(BaseModel):
    query: str
    collection_name: str
    n_docs: int


class RetrieveResponse(BaseModel):
    input_query: str
    docs: List[str]


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    input: Annotated[RetrieveRequest, Depends(RetrieveRequest)],
    retriever: Annotated[dict, Depends(database_retriever)],
    token: str = Depends(oauth2),
):
    chroma_conn = retriever["chroma_conn"]
    embed_func = retriever["embed_func"]
    collection_handle = chroma_conn.get_collection(
        input.collection_name, embedding_function=embed_func
    )
    res = collection_handle.query(
        query_texts=input.query, n_results=input.n_docs, include=["documents"]
    )
    docs = res["documents"][0]
    return RetrieveResponse(input_query=input.query, docs=docs)
