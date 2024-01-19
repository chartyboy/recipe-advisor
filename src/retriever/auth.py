import os

from abc import ABC, abstractmethod
from typing import Annotated, Any
from fastapi.security import APIKeyHeader
from fastapi import Depends, FastAPI
from fastapi import HTTPException

from .key_managers import KeyManager, EnvKeyManager, DBKeyManager, SQLiteKeyManager

header_auth = APIKeyHeader(name="api_header", auto_error=False)
cookie_auth = APIKeyHeader(name="api_cookie", auto_error=False)


async def setup_manager():
    KEY_MANAGER_SCHEMA = os.getenv("KEY_MANAGER")
    if KEY_MANAGER_SCHEMA is None:
        KEY_MANAGER_SCHEMA = "env"

    match KEY_MANAGER_SCHEMA:
        case "db":
            key_manager = DBKeyManager
        case "sqlite":
            key_manager = SQLiteKeyManager
        case "env":  # env lookup
            key_manager = EnvKeyManager
        case _:
            key_manager = KeyManager
    return key_manager


async def make_manager(key_manager: Annotated[Any, Depends(setup_manager)]):
    return key_manager()


async def verify_key(
    api_key_header: Annotated[Any, Depends(header_auth)],
    api_key_cookie: Annotated[Any, Depends(cookie_auth)],
    key_manager: Annotated[Any, Depends(make_manager)],
):
    if api_key_header and key_manager.check_key(api_key_header):
        return api_key_header
    elif api_key_cookie and key_manager.check_key(api_key_cookie):
        return api_key_cookie
    else:
        raise HTTPException(403, "Access Forbidden")
