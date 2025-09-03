from app.configs import settings
from app.concurrency import limit_asyncio_concurrency, batching, AsyncRWLock
import httpx
import logging
import asyncio
from pydantic import BaseModel
from typing import Any
import json
import os

engine_maintainance_lock = AsyncRWLock()
logger = logging.getLogger(__name__)

async def list_available_models() -> list[str]:
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.llm_base_url}/models",
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}"
            }
        )
        
        if response.status_code != 200:
            logger.warning(f"Failed to list available models: {response.status_code} {response.text}")
            return []

        response_json: dict = response.json()

        return [
            model['id']
            for model in response_json.get('data', [])
            if model.get('task') == 'chat'
        ]
        
class Metadata(BaseModel):
    volume: str | None = None
    chapter: str | None = None
    section: str | None = None
    reference_url: str | None = None

class DataItem(BaseModel):
    id: str
    content: str
    metadata: Metadata
    
class DatabaseMetadata(BaseModel):
    name: str
    version: str
    description: str

class Database(BaseModel):
    data: list[DataItem]
    metadata: DatabaseMetadata

class State:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.current_db: Database | None = None
        
    def assign_db(self, db: Database) -> Database:
        self.current_db = db
        return self.current_db
    
    def get_db(self) -> Database | None:
        return self.current_db

state_manager = State()

def load_db(db_path: str) -> Database | None:
    if not db_path or not os.path.exists(db_path):
        logger.warning(f"No database path provided or database path does not exist; Returning an empty database")
        return None

    with open(db_path, "r", encoding="utf-8") as f:
        data_json: dict = json.load(f)
        
    return Database(
        data=[
            DataItem(
                id=item['id'],
                content=item['content'],
                metadata=Metadata(**item.get('metadata', {}))
            )
            for item in data_json.get("data", [])
        ],
        metadata=DatabaseMetadata(**data_json.get("metadata", {}))
    )

def get_db_info() -> DatabaseMetadata | None:
    db = state_manager.get_db()
    return db.metadata if db else None