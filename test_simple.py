#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(__file__))

async def test_simple():
    try:
        from app.engine import state_manager, load_db
        
        # Load the database
        print("Loading database...")
        db = load_db('new_db.json')
        if db:
            state_manager.assign_db(db)
            print(f"Database loaded: {db.metadata.name} with {len(db.data)} items")
        else:
            print("Failed to load database")
            return
        
        # Initialize the collection
        print("Initializing collection...")
        await state_manager.refresh()
        
        # Test basic search functionality
        from app.engine import search
        print("Testing search functionality...")
        results = await search("naturalization requirements")
        print(f"Search returned {len(results)} results")
        if results:
            print(f"First result: {results[0].content[:100]}...")
        
        print("Basic functionality test completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_simple())
