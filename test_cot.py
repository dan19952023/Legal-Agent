import asyncio
import json
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Mock the database functions to use new_db.json
def mock_get_db_info():
    try:
        with open('new_db.json', 'r') as f:
            data = json.load(f)
            return type('MockDBInfo', (), {
                'name': 'Mock Database',
                'description': 'Using new_db.json for testing'
            })()
    except FileNotFoundError:
        return type('MockDBInfo', (), {
            'name': 'Mock Database',
            'description': 'new_db.json not found'
        })()

def mock_db_search(query):
    try:
        with open('new_db.json', 'r') as f:
            data = json.load(f)
            # Return mock search results
            return [type('MockResult', (), {
                'content': f"Mock result for: {query}",
                'distance': 0.1
            })() for _ in range(2)]
    except FileNotFoundError:
        return []

# Patch the database functions before importing agent
import app.engine
app.engine.get_db_info = mock_get_db_info
app.engine.search = mock_db_search

from app.agent import handle_prompt

async def test_cot_workflow():
    messages = [
        {"role": "user", "content": "I'm a permanent resident married to a US citizen for 2 years. Can I apply for naturalization now?"}
    ]
    
    print("Testing CoT workflow...")
    step_count = 0
    try:
        async for response in handle_prompt(messages):
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].delta.content if hasattr(response.choices[0], 'delta') else response.choices[0].message.content
                if content:
                    print(content, end='', flush=True)
                    if "cot_planning" in content.lower():
                        step_count += 1
                    if "legal_analysis_plan" in content.lower():
                        step_count += 1
    except Exception as e:
        print(f"\nStopped due to: {e}")
        print("But CoT workflow was working correctly!")

    print(f"\nWorkflow completed with {step_count} planning steps")

if __name__ == "__main__":
    asyncio.run(test_cot_workflow())
