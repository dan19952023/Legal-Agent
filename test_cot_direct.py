#!/usr/bin/env python3
"""
Direct test of CoT functionality without web server
"""

import asyncio
import os
import sys
sys.path.append('app')

from app.agent import handle_prompt, is_complex
from app.configs import settings

async def test_cot_directly():
    """Test CoT functionality directly"""
    
    # Set environment variables
    os.environ["USE_SIMPLE_COT_PLANNER"] = "1"
    os.environ["SIMPLE_COT_MAX_STEPS"] = "3"
    
    print("Testing CoT Functionality Directly")
    print("=" * 50)
    
    # Test query complexity detection
    simple_query = "What is naturalization?"
    complex_query = "I am a permanent resident married to a US citizen for 2 years and 8 months. I lived in the US for 3 years but traveled abroad for 6 months total during that time, including a 3-month trip to care for my sick mother. I want to apply for naturalization. What are my options and what documents do I need?"
    
    print(f"\nSimple Query: '{simple_query}'")
    print(f"Complexity: {is_complex(simple_query)}")
    
    print(f"\nComplex Query: '{complex_query[:100]}...'")
    print(f"Complexity: {is_complex(complex_query)}")
    
    # Test CoT execution
    print(f"\nTesting CoT Execution...")
    
    messages = [{"role": "user", "content": complex_query}]
    
    try:
        async for chunk in handle_prompt(messages):
            if hasattr(chunk, 'choices') and chunk.choices:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
    except Exception as e:
        print(f"\nError during CoT execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_cot_directly())
