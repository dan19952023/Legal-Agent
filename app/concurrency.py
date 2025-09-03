from typing import Callable, Generator, AsyncGenerator
import asyncio
from functools import partial, wraps
from starlette.concurrency import run_in_threadpool

def sync2async(sync_func: Callable):
    async def async_func(*args, **kwargs):
        res = run_in_threadpool(partial(sync_func, *args, **kwargs))

        if isinstance(res, (Generator, AsyncGenerator)):
            return res

        return await res

    return async_func if not asyncio.iscoroutinefunction(sync_func) else sync_func


def limit_asyncio_concurrency(num_of_concurrent_calls: int):
    semaphore = asyncio.Semaphore(num_of_concurrent_calls)

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                res = func(*args, **kwargs)

                if isinstance(res, (Generator, AsyncGenerator)):
                    return res

                return await res

        return wrapper
    return decorator

def batching(gen: Generator, batch_size: int = 1):
    batch = []

    for item in gen:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
        
async def batching_async(gen: AsyncGenerator, batch_size: int = 1):
    batch = []

    async for item in gen:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch
        
import asyncio
from contextlib import asynccontextmanager

class AsyncRWLock:
    def __init__(self):
        self._readers = 0
        self._writer = False
        self._lock = asyncio.Lock()
        self._read_ready = asyncio.Condition(self._lock)

    @asynccontextmanager
    async def read(self):
        await self.acquire_read()
        try:
            yield
        finally:
            await self.release_read()

    @asynccontextmanager
    async def write(self):
        await self.acquire_write()
        try:
            yield
        finally:
            await self.release_write()

    async def acquire_read(self):
        async with self._read_ready:
            while self._writer:
                await self._read_ready.wait()
            self._readers += 1

    async def release_read(self):
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    async def acquire_write(self):
        async with self._read_ready:
            while self._writer or self._readers > 0:
                await self._read_ready.wait()
            self._writer = True

    async def release_write(self):
        async with self._read_ready:
            self._writer = False
            self._read_ready.notify_all()
