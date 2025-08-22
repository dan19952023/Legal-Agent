#!/usr/bin/env python3

import asyncio
import sys
import os
import json

# Add the current directory to the path so app.engine resolves correctly
sys.path.insert(0, os.path.dirname(__file__))

async def test_agent():
    try:
        from app.agent import handle_prompt
        from app.engine import state_manager, load_db, get_db_info
        
        # Load the database first
        print("Loading database from new_db.json...")
        db = load_db('new_db.json')
        if db:
            state_manager.assign_db(db)
            print(f"Database loaded: {db.metadata.name} with {len(db.data)} items")
            
            # Test if get_db_info works now
            db_info = get_db_info()
            if db_info:
                print(f"Database info retrieved: {db_info.name}")
            else:
                print("Warning: get_db_info still returns None")
        else:
            print("Failed to load database")
            return
        
        # Test message
        messages = [{"role": "user", "content": "What are the basic requirements for naturalization?"}]
        
        print("Testing agent with message:", messages[0]['content'])
        print("=" * 50)
        
        # Test the agent
        async for response in handle_prompt(messages):
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].delta.content
                if content:
                    print(content, end='', flush=True)
        
        print("\n" + "=" * 50)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error testing agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent())
