#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

async def test_agent():
    try:
        from agent import handle_prompt
        
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
