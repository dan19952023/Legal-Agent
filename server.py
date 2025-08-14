import fastapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
from app.configs import settings
from app.apis import router as api_router
from app.engine import state_manager
import logging
from argparse import ArgumentParser
import time
from collections import defaultdict
import asyncio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Simple rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_ip]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < self.window_seconds]
        
        if len(client_requests) >= self.max_requests:
            return False
        
        client_requests.append(now)
        return True

rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

server_app = fastapi.FastAPI(
    title="Legal Agent API",
    description="AI-Powered USCIS Legal Research Assistant",
    version="1.0.0"
)

# Security middleware
server_app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this based on your deployment
)

# CORS middleware with restrictions
server_app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8080",  # Local development
        "http://127.0.0.1:8080",  # Local development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting middleware
@server_app.middleware("http")
async def rate_limit_middleware(request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return fastapi.responses.JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded. Please try again later."}
        )
    
    response = await call_next(request)
    return response

server_app.include_router(api_router)
# server_app.mount("/", StaticFiles(directory="public"), name="static")

@server_app.get("/health")
async def healthcheck():
    return {"status": "ok", "message": "Legal Agent is running", "timestamp": time.time()}

def main():
    parser = ArgumentParser()
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    server_app.state.db_path = args.db_path
    
    # Load database immediately at startup
    if args.db_path:
        logger.info(f"Loading database from: {args.db_path}")
        if state_manager.load_database_sync(args.db_path):
            logger.info("Database loaded successfully at startup")
        else:
            logger.error("Failed to load database at startup")
            logger.error("Server will start but database functionality may not work")
    else:
        logger.warning("No database path provided - server will start without database")
    
    logger.info(f"Starting Legal Agent server on {settings.host}:{settings.port}")
    uvicorn.run(server_app, host=settings.host, port=settings.port)

if __name__ == '__main__':
    main()
