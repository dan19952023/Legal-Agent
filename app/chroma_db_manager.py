import chromadb
from chromadb.api.models.Collection import Collection
import logging
from app.configs import settings

logger = logging.getLogger(__name__)

chroma_client = chromadb.PersistentClient(path=settings.persistent_storage_path)

def get_or_create_collection(name: str) -> Collection:
    return chroma_client.get_or_create_collection(name)

def list_collections() -> list[str]:
    return [
        c.name
        for c in chroma_client.list_collections()
    ]