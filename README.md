# Legal Agent - AI-Powered USCIS Legal Research Assistant

An intelligent AI agent specialized in USCIS (U.S. Citizenship and Immigration Services) legal research, powered by advanced RAG (Retrieval-Augmented Generation) technology and Chain of Thought (CoT) reasoning.

## 🚀 Features

- **Legal Expertise**: Specialized in USCIS immigration law and policy
- **RAG Architecture**: Advanced document retrieval and semantic search
- **Chain of Thought (CoT)**: Transparent reasoning with step-by-step planning
- **Real-time Search**: Dynamic legal document search and analysis
- **Streaming Responses**: Interactive chat experience
- **Legal Accuracy**: Built-in validation and citation tracking
- **Flexible Planning**: Configurable step planning (3-15 steps)

## 🏗️ Architecture

### Core Components
- **Agent Engine**: Main AI reasoning and planning system
- **Vector Database**: ChromaDB for semantic document search
- **RAG Pipeline**: Retrieval-augmented generation for accurate responses
- **CoT Planner**: Chain of Thought reasoning with configurable steps
- **Legal Metadata**: Enhanced document indexing with legal context

### Technology Stack
- **Backend**: FastAPI + Python 3.10+
- **AI Models**: OpenAI-compatible LLMs
- **Vector DB**: ChromaDB for embeddings
- **Search**: Hybrid semantic + keyword search
- **Streaming**: Real-time response streaming

## 📋 Requirements

- Python 3.10 or higher
- pip package manager
- OpenAI API key (or compatible LLM endpoint)
- 4GB+ RAM recommended

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd legal-agent-master
   ```

2. **Install dependencies**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   export LLM_API_KEY="your-api-key"
   export LLM_BASE_URL="your-llm-endpoint"
   export LLM_MODEL_ID="your-model-id"
   ```

## 🚀 Quick Start

### 1. Start the Server
```bash
# Basic startup
python3 server.py --db-path new_db.json

# With CoT enabled (recommended)
USE_SIMPLE_COT_PLANNER=1 python3 server.py --db-path new_db.json

# With custom step count (3 steps for speed, 5 for accuracy)
USE_SIMPLE_COT_PLANNER=1 SIMPLE_COT_MAX_STEPS=3 python3 server.py --db-path new_db.json
```

### 2. Test the Agent
```bash
# Using the chat toolkit
python3 toolkit/chat.py --address http://localhost:8080/prompt

# Or via HTTP API
curl -X POST http://localhost:8080/prompt \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What are naturalization requirements?"}],"stream":false}'
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | Required | Your LLM API key |
| `LLM_BASE_URL` | Required | LLM endpoint URL |
| `LLM_MODEL_ID` | Required | Model identifier |
| `PORT` | 8080 | Server port |
| `HOST` | 0.0.0.0 | Server host |
| `DISABLE_BACKGROUND_JOBS` | 0 | Disable background maintenance |
| `USE_SIMPLE_COT_PLANNER` | 0 | Enable optimized CoT planner |
| `SIMPLE_COT_MAX_STEPS` | 5 | Number of planning steps (1-15) |

### Data Format

The agent expects JSON data in this format:

```json
{
  "metadata": {
    "name": "USCIS Policy Manual",
    "version": "1.0.0",
    "description": "USCIS legal database"
  },
  "data": [
    {
      "id": "unique-id",
      "content": "Legal document content...",
      "metadata": {
        "volume": "Policy Manual Volume",
        "chapter": "Chapter Title",
        "section": "Section Name",
        "reference_url": "https://uscis.gov/..."
      }
    }
  ]
}
```

## 🔧 Advanced Usage

### CoT Planning Modes

1. **Standard Mode** (default)
   - Basic planning with reasoning
   - Good balance of speed and accuracy

2. **Simple CoT Mode** (`USE_SIMPLE_COT_PLANNER=1`)
   - Optimized step-by-step planning
   - Configurable step count (3-15)
   - Enhanced legal accuracy

### Performance Tuning

- **Fast Mode**: `SIMPLE_COT_MAX_STEPS=3` for quick responses
- **Accurate Mode**: `SIMPLE_COT_MAX_STEPS=5` for comprehensive coverage
- **Background Jobs**: Set `DISABLE_BACKGROUND_JOBS=1` for development

### Custom Database

```bash
# Use your own legal database
python3 server.py --db-path /path/to/your/legal_docs.json

# Pre-process documents with test.py
python3 test.py --input raw_docs.json --output processed_db.json
```

## 📚 API Endpoints

- `GET /health` - Health check
- `POST /prompt` - Chat completion endpoint
- `GET /docs` - API documentation (Swagger UI)

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8080/health
```

### Chat Test
```bash
python3 toolkit/chat.py --address http://localhost:8080/prompt
```

### API Test
```bash
curl -X POST http://localhost:8080/prompt \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Test message"}],"stream":false}'
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   pkill -f "python3 server.py"
   # or change port in configs.py
   ```

2. **LLM Connection Issues**
   - Verify API key and endpoint URL
   - Check network connectivity
   - Ensure model ID is correct

3. **Memory Issues**
   - Use smaller database files
   - Set `DISABLE_BACKGROUND_JOBS=1`
   - Reduce `SIMPLE_COT_MAX_STEPS`

4. **Slow Performance**
   - Use `SIMPLE_COT_MAX_STEPS=3` for speed
   - Ensure background jobs are disabled during development
   - Check database size and chunking

## 🔒 Legal Disclaimer

This tool provides general legal information based on USCIS policies and procedures. It is not a substitute for professional legal advice. Always consult with a qualified immigration attorney for specific legal matters.

## 📄 License

[License information]

## 🤝 Contributing

[Contribution guidelines]

## 📞 Support

[Support information]