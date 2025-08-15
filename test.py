import os, json

with open('full_data.json', 'r') as f:
    data = json.load(f)

data = data['data']
data_length = [len(e['content']) for e in data]

print(data_length)
print(sum(data_length) / len(data_length))
max_length = max(data_length)
min_length = min(data_length)
print(max_length, min_length)

import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

def _divide_by_tokens(content: str):
    chunks = []
    tokens = tokenizer.encode(content)
    step = max(1, 512 - 128)

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + 512]
        text = tokenizer.decode(chunk_tokens).strip()

        if text:
            chunks.append(text)

    return chunks or [content]

def validate_legal_document(item: dict) -> bool:
    """Validate legal document quality"""
    required_fields = ['content', 'metadata']
    metadata_fields = ['volume', 'chapter', 'section']
    
    # Check required fields
    if not all(field in item for field in required_fields):
        return False
    
    # Check metadata quality
    if not all(field in item['metadata'] for field in metadata_fields):
        return False
    
    # Check content quality
    if len(item['content']) < 50:  # Too short
        return False
    
    if not any(keyword in item['content'].lower() for keyword in ['uscis', 'policy', 'law', 'regulation']):
        return False
    
    return True

new_db = {
    "metadata": {
    "name": "uscis_policy_manual",
    "version": "1.1.1",
    "description": "USCIS Policy Manual database containing immigration law and policy information",
    "crawl_date": "2025-08-08T10:58:37.481276",
    "total_chapters": 445,
    "total_records": 3084,
    "chunk_data": True,
    "chunk_length": 512,
    "overlap": 128,
    "chunk_by_sections": True
  },
  'data': []
}

for item in data:
    # Validate legal document quality
    if not validate_legal_document(item):
        print(f"Skipping invalid legal document: {item.get('id', 'unknown')}")
        continue
        
    id = item['id']
    content = item['content']
    meta = item['metadata']
    chunks = _divide_by_tokens(content)
    for i, chunk in enumerate(chunks):
        # Enhance metadata with legal-specific fields
        enhanced_meta = meta.copy()
        enhanced_meta['document_type'] = 'Policy Manual'
        enhanced_meta['jurisdiction'] = 'USCIS'
        
        new_db['data'].append({
            'id': f'{id}/{i}',
            'content': chunk,
            'metadata': enhanced_meta
        })

with open('new_db.json', 'w') as f:
    json.dump(new_db, f, indent=4)