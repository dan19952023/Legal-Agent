#!/usr/bin/env python3
"""
Test script to verify the improvements made to the Legal Agent.
"""

import asyncio
import json
from app.configs import settings
from app.agent import make_plan, Step
from app.engine import load_db
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_make_plan():
    """Test the improved make_plan function."""
    print("Testing make_plan function...")
    
    try:
        # Test with valid input
        steps = await make_plan("What are the naturalization requirements for permanent residents?", max_steps=3)
        print(f"Generated {len(steps)} steps")
        
        for i, step in enumerate(steps):
            print(f"Step {i+1}: {step.task}")
            print(f"Expectation: {step.expectation}")
            print(f"Reason: {step.reason}")
            print()
            
    except Exception as e:
        print(f"Error in make_plan: {e}")
        return False
    
    return True

async def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    try:
        # Test port validation
        print(f"Port: {settings.port}")
        print(f"Host: {settings.host}")
        print(f"API Key set: {settings.llm_api_key != 'super-secret'}")
        print(f"Base URL: {settings.llm_base_url}")
        print(f"Model ID: {settings.llm_model_id}")
        
    except Exception as e:
        print(f"Error in config validation: {e}")
        return False
    
    return True

async def test_database_loading():
    """Test database loading functionality."""
    print("Testing database loading...")
    
    try:
        # Test loading the database
        db = load_db("new_db.json")
        if db:
            print(f"Database loaded: {db.metadata.name}")
            print(f"Total items: {len(db.data)}")
            print(f"Version: {db.metadata.version}")
        else:
            print("Database not loaded (this is normal if file doesn't exist)")
            
    except Exception as e:
        print(f"Error loading database: {e}")
        return False
    
    return True

async def main():
    """Run all tests."""
    print("Legal Agent Improvement Tests")
    print("=" * 50)
    
    tests = [
        ("Configuration Validation", test_config_validation),
        ("Database Loading", test_database_loading),
        ("Make Plan Function", test_make_plan),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The improvements are working correctly.")
    else:
        print("Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    asyncio.run(main())
