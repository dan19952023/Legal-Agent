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


@mcp.tool()
async def search_comments(
    search_term: str = None,
    agency_id: str = None,
    comment_on_id: str = None,
    posted_date: str = None,
    last_modified_date: str = None,
    page_size: int = 20,
    page_number: int = 1,
    sort: str = None,
) -> str:
    """Search for comments in the regulations.gov database.

    Args:
        search_term: Search term to filter comments (e.g. "water", "healthcare")
        agency_id: Agency acronym to filter by (e.g. "EPA", "FDA")
        comment_on_id: Object ID to filter comments for a specific document
        posted_date: Posted date filter (format: yyyy-MM-dd)
        last_modified_date: Last modified date filter (format: yyyy-MM-dd HH:mm:ss)
        page_size: Number of results per page (5-250, default: 20)
        page_number: Page number (1-20, default: 1)
        sort: Sort field (postedDate, lastModifiedDate, documentId) with optional - prefix for desc
    """
    # Build the URL with parameters
    params = []
    if search_term:
        params.append(f"filter[searchTerm]={search_term}")
    if agency_id:
        params.append(f"filter[agencyId]={agency_id}")
    if comment_on_id:
        params.append(f"filter[commentOnId]={comment_on_id}")
    if posted_date:
        params.append(f"filter[postedDate]={posted_date}")
    if last_modified_date:
        params.append(f"filter[lastModifiedDate]={last_modified_date}")
    if sort:
        params.append(f"sort={sort}")
    
    params.append(f"page[size]={page_size}")
    params.append(f"page[number]={page_number}")
    
    url = f"{REGULATIONS_API_BASE}/comments"
    if params:
        url += "?" + "&".join(params)
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch comments or no comments found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching comments: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_comment_details(
    comment_id: str,
    include_attachments: bool = False,
) -> str:
    """Get detailed information for a specific comment.

    Args:
        comment_id: The comment ID (e.g. "HHS-OCR-2018-0002-5313")
        include_attachments: Whether to include attachments in the response
    """
    # Build the URL
    url = f"{REGULATIONS_API_BASE}/comments/{comment_id}"
    if include_attachments:
        url += "?include=attachments"
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch comment details or comment not found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching comment details: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


@mcp.tool()
async def search_dockets(
    search_term: str = None,
    agency_id: str = None,
    docket_type: str = None,
    last_modified_date: str = None,
    page_size: int = 20,
    page_number: int = 1,
    sort: str = None,
) -> str:
    """Search for dockets in the regulations.gov database.

    Args:
        search_term: Search term to filter dockets (e.g. "water", "healthcare")
        agency_id: Agency acronym to filter by (e.g. "EPA", "FDA")
        docket_type: Docket type filter (Rulemaking, Nonrulemaking)
        last_modified_date: Last modified date filter (format: yyyy-MM-dd HH:mm:ss)
        page_size: Number of results per page (5-250, default: 20)
        page_number: Page number (1-20, default: 1)
        sort: Sort field (title, docketId, lastModifiedDate) with optional - prefix for desc
    """
    # Build the URL with parameters
    params = []
    if search_term:
        params.append(f"filter[searchTerm]={search_term}")
    if agency_id:
        params.append(f"filter[agencyId]={agency_id}")
    if docket_type:
        params.append(f"filter[docketType]={docket_type}")
    if last_modified_date:
        params.append(f"filter[lastModifiedDate]={last_modified_date}")
    if sort:
        params.append(f"sort={sort}")
    
    params.append(f"page[size]={page_size}")
    params.append(f"page[number]={page_number}")
    
    url = f"{REGULATIONS_API_BASE}/dockets"
    if params:
        url += "?" + "&".join(params)
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch dockets or no dockets found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching dockets: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_docket_details(
    docket_id: str,
) -> str:
    """Get detailed information for a specific docket.

    Args:
        docket_id: The docket ID (e.g. "EPA-HQ-OAR-2003-0129")
    """
    # Build the URL
    url = f"{REGULATIONS_API_BASE}/dockets/{docket_id}"
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch docket details or docket not found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching docket details: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_agency_categories(
    agency_acronym: str,
) -> str:
    """Get available categories for a specific agency.

    Args:
        agency_acronym: Agency acronym (e.g. "EPA", "FDA")
    """
    # Build the URL
    url = f"{REGULATIONS_API_BASE}/agency-categories?filter[acronym]={agency_acronym}"
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch agency categories or agency not found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching agency categories: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)



@mcp.tool()
async def search_comments(
    search_term: str = None,
    agency_id: str = None,
    comment_on_id: str = None,
    posted_date: str = None,
    last_modified_date: str = None,
    page_size: int = 20,
    page_number: int = 1,
    sort: str = None,
) -> str:
    """Search for comments in the regulations.gov database.

    Args:
        search_term: Search term to filter comments (e.g. "water", "healthcare")
        agency_id: Agency acronym to filter by (e.g. "EPA", "FDA")
        comment_on_id: Object ID to filter comments for a specific document
        posted_date: Posted date filter (format: yyyy-MM-dd)
        last_modified_date: Last modified date filter (format: yyyy-MM-dd HH:mm:ss)
        page_size: Number of results per page (5-250, default: 20)
        page_number: Page number (1-20, default: 1)
        sort: Sort field (postedDate, lastModifiedDate, documentId) with optional - prefix for desc
    """
    # Build the URL with parameters
    params = []
    if search_term:
        params.append(f"filter[searchTerm]={search_term}")
    if agency_id:
        params.append(f"filter[agencyId]={agency_id}")
    if comment_on_id:
        params.append(f"filter[commentOnId]={comment_on_id}")
    if posted_date:
        params.append(f"filter[postedDate]={posted_date}")
    if last_modified_date:
        params.append(f"filter[lastModifiedDate]={last_modified_date}")
    if sort:
        params.append(f"sort={sort}")
    
    # Check if at least one filter is provided
    if not any([search_term, agency_id, comment_on_id, posted_date, last_modified_date]):
        return "Error: At least one filter parameter (search_term, agency_id, comment_on_id, posted_date, or last_modified_date) is required for comments search."
    
    params.append(f"page[size]={page_size}")
    params.append(f"page[number]={page_number}")
    
    url = f"{REGULATIONS_API_BASE}/comments"
    if params:
        url += "?" + "&".join(params)
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch comments or no comments found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching comments: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


@mcp.tool()
async def get_comment_details(
    comment_id: str,
    include_attachments: bool = False,
) -> str:
    """Get detailed information for a specific comment.

    Args:
        comment_id: The comment ID (e.g. "HHS-OCR-2018-0002-5313")
        include_attachments: Whether to include attachments in the response
    """
    # Build the URL
    url = f"{REGULATIONS_API_BASE}/comments/{comment_id}"
    if include_attachments:
        url += "?include=attachments"
    
    data = await make_request(url)

    # Check if data is found
    if not data:
        return "Unable to fetch comment details or comment not found."

    # Check for errors
    if "Error" in data:
        return f"Error fetching comment details: {data['Error']}"

    # Stringify the response
    return json.dumps(data, indent=2)


if __name__ == "__main__":
    # Log server startup
    logger.info("Starting Regulations.gov MCP Server...")

    # Initialize and run the server
    mcp.run(transport="stdio")

    # This line won't be reached during normal operation
    logger.info("Server stopped")