import fastapi
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.configs import settings
from app.apis import router as api_router
import logging
from argparse import ArgumentParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

server_app = fastapi.FastAPI()

server_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server_app.include_router(api_router)
# server_app.mount("/", StaticFiles(directory="public"), name="static")

@server_app.get("/health")
async def healthcheck():
    return {"status": "ok", "message": "Yo, I am alive"}

def main():
    parser = ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    server_app.state.db_path = args.db_path
    uvicorn.run(server_app, host=settings.host, port=settings.port)

if __name__ == '__main__':
    main()
