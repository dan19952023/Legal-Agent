# Legal Agent

An AI-powered immigration law assistant that provides comprehensive legal guidance using the USCIS Policy Manual database. The system combines semantic search, keyword extraction, and structured reasoning to deliver accurate immigration law information with proper citations.

## Summary

The Legal Agent is an intelligent system that leverages vector embeddings and large language models to provide detailed immigration law assistance. It searches through the USCIS Policy Manual database, extracts relevant information, and presents it in a structured format with proper legal citations and step-by-step guidance.

## Eligibility

- **Python 3.10+** with pip package management
- **ChromaDB** for vector storage and semantic search
- **FastAPI** for REST API endpoints
- **OpenAI-compatible LLM** for natural language processing
- **Vector embeddings** for semantic similarity matching

## Exceptions / Waivers

- Requires valid LLM API credentials and base URL
- Database path must be provided at startup
- Persistent storage directory must be writable
- Network access for LLM API calls

## Filing Checklist

### Prerequisites
- Python 3.10+ installed
- Valid LLM API credentials
- USCIS Policy Manual database in JSON format
- Sufficient disk space for ChromaDB storage

### Installation
```bash
# Clone repository
git clone <repository-url>
cd Legal-Agent

# Install dependencies
python -m pip install -r requirements.txt

# Set environment variables
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL_ID="gpt-4o-mini"
export PORT=8080
```

### Database Setup
Prepare your USCIS Policy Manual database in the required JSON format:
```json
{
  "metadata": {
    "name": "USCIS Policy Manual",
    "version": "1.0.0",
    "description": "Immigration law database"
  },
  "data": [
    {
      "id": "unique-id",
      "content": "Legal text content",
      "metadata": {
        "volume": "Volume A",
        "chapter": "Chapter 1",
        "section": "Section 1.1",
        "reference_url": "https://www.uscis.gov/policy-manual/..."
      }
    }
  ]
}
```

## Processing & RFEs

### Startup Process
1. **Server Initialization**: FastAPI server starts on specified host/port
2. **Database Loading**: USCIS database loaded and parsed into structured format
3. **Collection Management**: ChromaDB collections created/updated based on available embedding models
4. **Maintenance Loop**: Background process continuously syncs database with vector embeddings
5. **API Endpoints**: REST endpoints become available for legal queries

### Agent Workflow
1. **Query Reception**: User sends legal question via `/prompt` endpoint
2. **Context Analysis**: KeyBERT extracts relevant keywords from user message
3. **Semantic Search**: Vector embeddings search USCIS database for relevant content
4. **Chain-of-Thought Planning**: LLM breaks down complex legal requests into logical steps
5. **Information Retrieval**: Agent searches database and executes Python code as needed
6. **Response Generation**: Structured legal analysis with proper citations and formatting
7. **Streaming Output**: Real-time response delivery with proper formatting

### Tool Execution
- **Search Tool**: Semantic search across USCIS Policy Manual
- **Python Tool**: Data analysis and calculations with resource limits
- **Planning Tools**: Structured reasoning and step-by-step analysis

## Common Pitfalls

- **Missing API Credentials**: Ensure LLM_API_KEY and LLM_BASE_URL are set
- **Invalid Database Format**: JSON structure must match expected schema
- **Insufficient Resources**: Python tool execution has memory and CPU limits
- **Network Issues**: LLM API calls may fail due to connectivity problems
- **Storage Permissions**: ChromaDB requires writable persistent storage directory

## Sources & Links

### Core Components
- **Agent Engine** (`app/agent.py`): Main reasoning and tool execution logic
- **Search Engine** (`app/engine.py`): Vector database management and semantic search
- **API Layer** (`app/apis.py`): FastAPI endpoints and request handling
- **Database Manager** (`app/chroma_db_manager.py`): ChromaDB collection management
- **Keyword Extraction** (`app/lite_keybert.py`): BERT-based keyword analysis
- **Utility Functions** (`app/utils.py`): Helper functions and response formatting

### Configuration
- **Settings** (`app/configs.py`): Environment-based configuration management
- **Concurrency** (`app/concurrency.py`): Async operation management
- **Models** (`app/oai_models.py`): OpenAI-compatible response models

### Usage Examples
```bash
# Start server with custom database
PORT=8080 python server.py --db-path path/to/your/uscis_database.json

# Test agent functionality
python test_agent.py

# Docker deployment
docker build -t legal-agent .
docker run -p 8080:80 legal-agent
```

### API Endpoints
- `POST /prompt`: Submit legal questions for analysis
- `GET /health`: Server health check
- Streaming responses supported for real-time output

### Data Format Requirements

The system expects USCIS Policy Manual data in this specific JSON structure:

```json
{
  "metadata": {
    "name": "abc",
    "version": "1.0.0",
    "description": "a database for ..."
  },
  "data": [
    {
      "id": "d2575559-66b7-40da-b7c8-ec1869696db8",
      "content": "An official website of the United States government\nHere's how you know",
      "metadata": {
        "volume": "Search | None",
        "chapter": "https://www.uscis.gov/policy-manual/search | None",
        "section": "USCIS Policy Manual | None",
        "reference_url": "https://example.com | None"
      }
    }
  ]
}
```

The Legal Agent provides comprehensive immigration law assistance with proper citations, structured responses, and semantic search capabilities across the USCIS Policy Manual database.