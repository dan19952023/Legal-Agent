#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

async def test_models():
    try:
        from app.engine import list_available_models, get_current_available_collection
        
        print("Testing available models...")
        models = await list_available_models()
        print(f"Available models: {models}")
        
        print("\nTesting collection selection...")
        collection = await get_current_available_collection()
        if collection:
            print(f"Selected collection: {collection.model_id}")
        else:
            print("No collection selected")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_models())
