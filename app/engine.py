from app.configs import settings
from app.concurrency import limit_asyncio_concurrency, batching, AsyncRWLock
import httpx
import logging
import asyncio
from app.chroma_db_manager import list_collections, Collection, get_or_create_collection
from app.configs import settings
from pydantic import BaseModel
from typing import Any
from chromadb.api.models.Collection import QueryResult
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
            if model['id'].startswith('text-embedding-') or model.get('task') == 'chat'
        ]
        
class ActiveCollection:
    def __init__(self, collection: Collection, model_id: str):
        self.collection = collection
        self.model_id = model_id

class Metadata(BaseModel):
    volume: str | None = None
    chapter: str | None = None
    section: str | None = None
    reference_url: str | None = None

class DataItem(BaseModel):
    id: str
    content: str
    metadata: Metadata

class SearchResult(DataItem):
    distance: float
    
class DatabaseMetadata(BaseModel):
    name: str
    version: str
    description: str

class Database(BaseModel):
    data: list[DataItem]
    metadata: DatabaseMetadata
        
async def get_current_available_collection() -> ActiveCollection | None:
    models = await list_available_models()
        
    if not len(models):
        logger.warning("No available models")
        return None

    collections = list_collections()

    for collection_name in collections:
        if any(collection_name == model for model in models):
            return ActiveCollection( # priotize the current existing collection
                collection=get_or_create_collection(collection_name),
                model_id=collection_name
            )

    selected_model = models[0]
    logger.warning(f"No available collections; Seclecting the first one; Model: {selected_model}")
    return ActiveCollection(
        collection=get_or_create_collection(selected_model),
        model_id=selected_model
    )

@limit_asyncio_concurrency(num_of_concurrent_calls=16)
async def embed(texts: list[str], model_id: str) -> list[list[float]]:
    if not texts:
        logger.warning("No texts to embed; Returning empty results")
        return []

    for i in range(3):
        async with httpx.AsyncClient() as client:                
            try:
                response = await client.post(
                    f"{settings.llm_base_url}/embeddings",
                    json={
                        "input": texts,
                        "model": model_id
                    },
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    headers={
                        "Authorization": f"Bearer {settings.llm_api_key}"
                    }
                )

                status = response.status_code

                if status >= 500:
                    logger.error(f"[1] Error calling remote model: {status}; {response.text}")
                    await asyncio.sleep(2 ** i)
                    continue

            except Exception as err:
                logger.error(f"[2] Error calling remote model: {str(err)}")
                continue

            if response.status_code != 200:
                logger.error(f"[1] Error calling remote model: {response.status_code}; {response.text}")
                return None

            response_json = response.json()

            if response_json.get('model', 'idk') != model_id:
                logger.warning(f"[ALERT] Invalid response model id: {response_json.get('model', 'idk')}; Expected: {model_id}")

            return [e['embedding'] for e in response_json.get('data', [])]

@limit_asyncio_concurrency(num_of_concurrent_calls=8)
async def embed_bg(texts: list[str], model_id: str):
    return await embed(texts, model_id)

async def stupid_wrapper(offset: int, coro: asyncio.coroutines) -> tuple[int, Any | Exception]:
    try:
        return offset, await coro
    except Exception as e:
        logger.error(f"Error in task {offset}: {e}", exc_info=True)
        return offset, e

async def bulk_insert(data: list[DataItem], collection: ActiveCollection) -> bool:
    active_collection = collection.collection
    model_id = collection.model_id

    tasks = []
    batch_size = 16

    for i, batch in enumerate(batching(data, batch_size=batch_size)):
        batch: list[DataItem]
        tasks.append(
            stupid_wrapper(i * batch_size, embed_bg(
                [item.content for item in batch], 
                model_id=model_id,
            ))
        )

    for something in asyncio.as_completed(tasks):
        offset, embeddings = await something

        if isinstance(embeddings, Exception):
            logger.error(f"Error in task {offset}: {embeddings}")
            continue

        sub_data = data[offset:offset+batch_size]
        logger.info(f"Upserting {len(sub_data)} items")

        async with engine_maintainance_lock.write():
            active_collection.upsert(
                embeddings=embeddings,
                documents=[item.content for item in sub_data],
                ids=[item.id for item in sub_data],
                metadatas=[
                    item.metadata.model_dump(exclude_none=True)
                    for item in sub_data
                ]
            )

    return True

class State:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.current_active_collection: ActiveCollection | None = None
        self.current_db: Database | None = None
        
    def assign_db(self, db: Database) -> Database:
        self.current_db = db
        return self.current_db
    
    def get_db(self) -> Database | None:
        return self.current_db

    async def refresh(self) -> ActiveCollection | None:
        self.current_active_collection = await get_current_available_collection()
        return self.current_active_collection

    def get(self) -> ActiveCollection | None:
        return self.current_active_collection

state_manager = State()

async def search(
    queries: list[str] | str, 
    max_results: int = 10, 
    semantic_weight: float = 0.8 # TODO: add keyword search later
) -> list[SearchResult]:
    if isinstance(queries, str):
        queries = [queries]

    active_collection = state_manager.get()
    
    if not active_collection:
        logger.warning("No available collection at this time")
        return []
    
    collection = active_collection.collection
    model_id = active_collection.model_id

    embeddings = await embed(queries, model_id)

    if not embeddings:
        logger.warning("No embeddings generated; Returning empty results")
        return []

    async with engine_maintainance_lock.read():
        results: QueryResult = collection.query(
            query_embeddings=embeddings,
            n_results=max_results,
            include=["distances", "metadatas", "documents"]
        )

    ids, docs, metadatas, distances = [], [], [], []

    for _id, _doc, _metadata, _distance in zip(results["ids"], results["documents"], results["metadatas"], results["distances"]):
        ids.extend(_id)
        docs.extend(_doc)
        metadatas.extend(_metadata)
        distances.extend(_distance)

    collected = set([])
    final_results = []
    
    for _id, _doc, _metadata, _distance in zip(ids, docs, metadatas, distances):
        if _id in collected:
            continue
        
        collected.add(_id)
        final_results.append(SearchResult(id=_id, content=_doc, metadata=_metadata, distance=_distance))
        
    return final_results

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

async def maintainance_loop(db_path: str):
    if not db_path or not os.path.exists(db_path):
        logger.warning("No database to maintain; Exiting maintainance loop")
        return

    try:
        db = state_manager.assign_db(load_db(db_path))
    except Exception as e:
        logger.error(f"Error loading database: {e}", exc_info=True)
        return
    
    if not db:
        logger.warning("No database to maintain; Exiting maintainance loop")
        return
    
    logger.info(f"Loaded database {db.metadata.name} with {len(db.data)} items")
    target_ids = [item.id for item in db.data]

    while True:
        try:
            active_collection = await state_manager.refresh()

            if not active_collection:
                logger.warning("No available collection at this time; Sleeping for 10 seconds")
                continue

            collection = active_collection.collection

            async with engine_maintainance_lock.read():
                ids_in_collection = collection.get(ids=target_ids)

            ids_not_in_collection = [
                id for id in target_ids 
                if id not in ids_in_collection['ids']
            ]

            if not ids_not_in_collection:
                logger.info("All items are already in the collection; Sleeping for 60 seconds")
                continue

            if not await bulk_insert(db.data, active_collection):
                logger.error("Failed to insert items into the collection; Sleeping for 10 seconds")

        except Exception as e:
            logger.error(f"Error in maintainance loop: {e}", exc_info=True)

        finally:
            await asyncio.sleep(60)

def get_db_info() -> DatabaseMetadata | None:
    return state_manager.get_db().metadata

def get_active_collection() -> ActiveCollection | None:
    return state_manager.get()