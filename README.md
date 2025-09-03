# Legal Agent

An AI-powered legal research assistant that provides guidance across multiple areas of law using external regulatory APIs. The system combines structured reasoning and Model Context Protocol (MCP) integration to deliver accurate legal information with proper citations.

## Summary

The Legal Agent is an intelligent system that leverages large language models and MCP tools to provide detailed legal assistance across multiple areas of law. It integrates with external regulatory APIs, searches for relevant information, and presents it in a structured format with proper legal citations and step-by-step guidance.

### Legal Expertise Areas

The agent provides comprehensive assistance in:

- **Federal regulations and administrative law**
- **Constitutional law and civil rights**
- **Business and corporate law**
- **Employment law and labor relations**
- **Environmental law and regulations**
- **Healthcare law and compliance**
- **Tax law and regulations**
- **Intellectual property law**
- **Criminal law and procedure**
- **Family law and domestic relations**
- **Real estate and property law**
- **Contract law and commercial transactions**

## Eligibility

- **Python 3.10+** with pip package management
- **FastAPI** for REST API endpoints
- **OpenAI-compatible LLM** for natural language processing
- **MCP (Model Context Protocol)** for external API integration
- **Regulations.gov API** for federal regulatory data access

## Exceptions / Waivers

- Requires valid LLM API credentials and base URL
- Network access for LLM API calls
- Regulations.gov API key for federal regulatory data access

## Filing Checklist

### Prerequisites
- Python 3.10+ installed
- Valid LLM API credentials
- Regulations.gov API key for federal regulatory data access

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
export REGULATIONS_API_KEY="your-regulations-api-key"
```

### API Setup
The Legal Agent uses external APIs for legal research. No local database setup is required.

## Processing & RFEs

### Startup Process
1. **Server Initialization**: FastAPI server starts on specified host/port
2. **MCP Integration**: Model Context Protocol tools become available for external API access
3. **API Endpoints**: REST endpoints become available for legal queries

### Agent Workflow
1. **Query Reception**: User sends legal question via `/prompt` endpoint
2. **Chain-of-Thought Planning**: LLM breaks down complex legal requests into logical steps
3. **Information Retrieval**: Agent searches federal regulatory databases and executes Python code as needed
4. **Response Generation**: Structured legal analysis with proper citations and formatting
5. **Streaming Output**: Real-time response delivery with proper formatting

### Tool Execution
- **Search Documents**: Search federal regulatory databases for current regulations and documents
- **Get Document Details**: Retrieve detailed information for specific federal regulatory documents
- **Python Tool**: Data analysis and calculations with resource limits
- **Planning Tools**: Structured reasoning and step-by-step analysis

## Common Pitfalls

- **Missing API Credentials**: Ensure LLM_API_KEY, LLM_BASE_URL, and REGULATIONS_API_KEY are set
- **Insufficient Resources**: Python tool execution has memory and CPU limits
- **Network Issues**: LLM API calls may fail due to connectivity problems
- **API Rate Limits**: Regulations.gov API has rate limits that may affect performance

## Sources & Links

### Core Components
- **Agent Engine** (`app/agent.py`): Main reasoning and tool execution logic (imports MCP functions)
- **API Layer** (`app/apis.py`): FastAPI endpoints and request handling
- **Utility Functions** (`app/utils.py`): Helper functions and response formatting
- **MCP Functions** (`mcps/regulations/main.py`): Reusable API functions for federal regulatory data

### Configuration
- **Settings** (`app/configs.py`): Environment-based configuration management
- **Concurrency** (`app/concurrency.py`): Async operation management
- **Models** (`app/oai_models.py`): OpenAI-compatible response models

### Usage Examples
```bash
# Start the main Legal Agent server
PORT=8080 python server.py

# Start MCP server for external API integration (optional - can run standalone)
python mcps/regulations/main.py

# Test agent functionality
python test_agent.py

# Docker deployment
docker build -t legal-agent .
docker run -p 8080:80 legal-agent
```

### MCP Integration

The Legal Agent includes Model Context Protocol (MCP) integration for accessing external regulatory APIs:

#### Available MCP Tools

1. **`search_documents`**: Search the Regulations.gov database for federal regulatory documents
   - Parameters: `search_term`, `agency_id`, `document_type`, `docket_id`, `posted_date`
   - Returns: JSON-formatted search results from federal regulatory database

2. **`get_document_details`**: Get detailed information for specific regulatory documents
   - Parameters: `document_id`, `include_attachments`
   - Returns: Complete document details including metadata and attachments

#### MCP Server Setup

```bash
# Set up Regulations.gov API key (optional)
export REGULATIONS_API_KEY="your-api-key"

# Start the MCP server
python mcps/regulations/main.py
```

#### MCP Benefits

- **Standardized API Access**: Consistent interface for external API calls
- **Real-time Data**: Access to current federal regulatory information
- **Tool Discovery**: AI agents can automatically discover available API tools
- **Error Handling**: Built-in error handling and response formatting
- **Extensible**: Easy to add new external API integrations
- **Code Reuse**: MCP functions are imported and reused by the main agent
- **Maintainable**: Single source of truth for API logic

### API Endpoints
- `POST /prompt`: Submit legal questions for analysis
- `GET /health`: Server health check
- Streaming responses supported for real-time output

### API Integration

The Legal Agent integrates with external APIs through the Model Context Protocol (MCP). No local database setup is required - all legal research is performed through real-time API calls to federal regulatory databases.

The Legal Agent provides comprehensive legal assistance across multiple areas of law with proper citations, structured responses, and real-time access to federal regulatory data through MCP integration.