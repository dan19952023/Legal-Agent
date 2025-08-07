# legal-agent
lawyer

## Data format 

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

## Setup

Requires:
- python 3.10+
- pip installed

Executes:
```bash
python -m pip install -r requirements.txt
```

## Run

Executes:

```bash
PORT=8000 python server.py --db-path path/2/your-db.json
```