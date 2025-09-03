import json
import os
import httpx
import logging
import sys
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Configure logging to write to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("regulations-mcp")

# Initialize FastMCP server
mcp = FastMCP("regulations")

# Constants
REGULATIONS_API_BASE = "https://api.regulations.gov/v4"


# Helper function to make API requests
async def make_request(url: str) -> dict[str, any] | None:
    """Make a request to the Regulations.gov API with proper error handling."""
    # Load environment variables from .env file
    load_dotenv()
    
    headers = {}
    if api_key := os.environ.get("REGULATIONS_API_KEY"):
        headers["X-Api-Key"] = api_key
    else:
        return {"Error": "REGULATIONS_API_KEY environment variable is required"}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"Error": str(e)}


@mcp.tool()
async def search_documents(
    search_term: str = None,
    agency_id: str = None,
    document_type: str = None,
    docket_id: str = None,
    posted_date: str = None,
    page_size: int = 20,
    page_number: int = 1,
) -> str:
    """Search for documents in the regulations.gov database.

    Args:
        search_term: Search term to filter documents (e.g. "water", "healthcare")
        agency_id: Agency acronym to filter by (e.g. "EPA", "FDA")
        document_type: Document type filter (Notice, Rule, Proposed Rule, Supporting & Related Material, Other)
        docket_id: Docket ID to filter by
        posted_date: Posted date filter (format: yyyy-MM-dd)
        page_size: Number of results per page (5-250, default: 20)
        page_number: Page number (1-20, default: 1)
    """
    # Build the URL with parameters
    params = []
    if search_term:
        params.append(f"filter[searchTerm]={search_term}")
    if agency_id:
        params.append(f"filter[agencyId]={agency_id}")
    if document_type:
        params.append(f"filter[documentType]={document_type}")
    if docket_id:
        params.append(f"filter[docketId]={docket_id}")
    if posted_date:
        params.append(f"filter[postedDate]={posted_date}")
    
    params.append(f"page[size]={page_size}")
    params.append(f"page[number]={page_number}")
    
    url = f"{REGULATIONS_API_BASE}/documents"
    if params:
        url += "?" + "&".join(params)
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch documents or no documents found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching documents: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_document_details(
    document_id: str,
    include_attachments: bool = False,
) -> str:
    """Get detailed information for a specific document.

    Args:
        document_id: The document ID (e.g. "FDA-2009-N-0501-0012")
        include_attachments: Whether to include attachments in the response
    """
    # Build the URL
    url = f"{REGULATIONS_API_BASE}/documents/{document_id}"
    if include_attachments:
        url += "?include=attachments"
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch document details or document not found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching document details: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    # Log server startup
    logger.info("Starting Regulations.gov MCP Server...")

    # Initialize and run the server
    mcp.run(transport="stdio")

    # This line won't be reached during normal operation
    logger.info("Server stopped")