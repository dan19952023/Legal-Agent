from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from .oai_models import (
    ChatCompletionRequest, 
    ChatCompletionStreamResponse,
    random_uuid
)
from typing import AsyncGenerator
import logging
import time
from app.engine import maintainance_loop
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI
from app.agent import handle_prompt, Executor
from app.engine import get_db_info

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state = app.state
    
    # Check if background jobs should be disabled
    import os
    disable_background_jobs = os.getenv("DISABLE_BACKGROUND_JOBS", "0").lower() in ("1", "true", "yes")
    
    if not disable_background_jobs:
        task = asyncio.create_task(maintainance_loop(app_state.db_path))
    else:
        task = None
        logger.info("Background jobs disabled via DISABLE_BACKGROUND_JOBS=1")

    try:
        yield
    finally:
        if task:
            task.cancel()

router = APIRouter(lifespan=lifespan)

@router.post("/prompt")
async def prompt(request: ChatCompletionRequest):
    enqueued = time.time()
    ttft, tps, n_tokens = float("inf"), None, 0
    req_id = request.request_id or f"req-{random_uuid()}"
    generator = handle_prompt(request.messages)

    if request.stream:

        async def to_bytes(gen: AsyncGenerator) -> AsyncGenerator[bytes, None]:
            nonlocal ttft, tps, n_tokens, req_id, enqueued

            async for chunk in gen:
                current_time = time.time()

                n_tokens += 1
                ttft = min(ttft, current_time - enqueued)
                tps = n_tokens / (current_time - enqueued)

                if isinstance(chunk, ChatCompletionStreamResponse):
                    data = chunk.model_dump_json()
                    yield "data: " + data + "\n\n"

            logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
            yield "data: [DONE]\n\n"

        return StreamingResponse(to_bytes(generator), media_type="text/event-stream")
    
    else:
        async for chunk in generator:
            current_time = time.time()

            n_tokens += 1
            ttft = min(ttft, current_time - enqueued)
            tps = n_tokens / (current_time - enqueued)

        logger.info(f"Request {req_id} - TTFT: {ttft:.2f}s, TPS: {tps:.2f} tokens/s")
        return JSONResponse(chunk.model_dump())
