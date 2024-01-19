import os
import logging
from abc import ABC, abstractmethod


class KeyManager(ABC):
    def __init__(self) -> None:
        self.key = None

    @abstractmethod
    async def initialize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    async def check_key(self, *args, **kwargs):
        raise NotImplementedError

    async def renew_key(self, *args, **kwargs):
        raise NotImplementedError

    async def revoke_key(self, *args, **kwargs):
        raise NotImplementedError

    async def add_key(self, *args, **kwargs):
        raise NotImplementedError


class EnvKeyManager(KeyManager):
    async def initialize(self, *args, **kwargs):
        pass

    async def check_key(self, key, key_env_name):
        if self.key is None:
            self.key = os.getenv(key_env_name)
        try:
            is_valid_key = self.key == key
        except:
            return False
        return is_valid_key


class DBKeyManager(KeyManager):
    async def initialize(self, *args, **kwargs):
        pass

    async def check_key(self, key, key_env_name):
        if self.key is None:
            self.key = os.getenv(key_env_name)
        try:
            is_valid_key = self.key == key
        except:
            return False
        return is_valid_key


class SQLiteKeyManager(KeyManager):
    async def get_key(self):
        pass
