#!/usr/bin/env python3
"""
Test CoT logic components directly without database dependencies
"""

import sys
import asyncio
sys.path.append('app')

from app.agent import is_complex, extract_legal_intent, validate_legal_step, build_legal_search_query, Step

def test_complexity_detection():
    """Test query complexity detection"""
    print("Testing Query Complexity Detection")
    print("=" * 40)
    
    test_queries = [
        ("What is naturalization?", False),
        ("How do I apply for citizenship?", False),
        ("I am a permanent resident married to a US citizen for 2 years and 8 months. I lived in the US for 3 years but traveled abroad for 6 months total during that time, including a 3-month trip to care for my sick mother. I want to apply for naturalization. What are my options and what documents do I need?", True),
        ("What are the requirements for naturalization if I have been married to a US citizen for 3 years and traveled outside the US for 6 months?", True),
        ("Tell me about naturalization", False),
        ("I need help with my naturalization case because I traveled abroad and my mother was sick", True)
    ]
    
    for query, expected in test_queries:
        result = is_complex(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status} | Expected: {expected}, Got: {result} | Query: {query[:60]}...")
    
    print()

async def test_legal_intent_extraction():
    """Test legal intent extraction"""
    print("Testing Legal Intent Extraction")
    print("=" * 40)
    
    test_query = "I am a permanent resident married to a US citizen for 2 years and 8 months. I lived in the US for 3 years but traveled abroad for 6 months total during that time, including a 3-month trip to care for my sick mother. I want to apply for naturalization. What are my options and what documents do I need?"
    
    intent = await extract_legal_intent(test_query)
    print(f"Query: {test_query[:80]}...")
    print(f"Extracted Intent: {intent}")
    print()

def test_legal_step_validation():
    """Test legal step validation"""
    print("Testing Legal Step Validation")
    print("=" * 40)
    
    test_steps = [
        Step(
            reason="Need to determine eligibility under INA 319(a) for spouses of US citizens",
            task="Search for 'INA 319(a) naturalization spouse US citizen 3 year rule requirements'",
            expectation="Find specific eligibility criteria for spouses of US citizens"
        ),
        Step(
            reason="Must verify continuous residence requirements and physical presence",
            task="Search for 'continuous residence physical presence naturalization INA 316 travel absence'",
            expectation="Find rules about travel and absence from US during residence period"
        ),
        Step(
            reason="Need to understand exceptions for caring for sick family members",
            task="Search for 'naturalization absence exception sick family member medical emergency'",
            expectation="Find USCIS policy on excusing absences for family care"
        )
    ]
    
    for i, step in enumerate(test_steps, 1):
        is_valid = validate_legal_step(step)
        status = "VALID" if is_valid else "INVALID"
        print(f"Step {i}: {status}")
        print(f"  Task: {step.task}")
        print(f"  Reason: {step.reason}")
        print()

def test_search_query_building():
    """Test legal search query building"""
    print("Testing Legal Search Query Building")
    print("=" * 40)
    
    step = Step(
        reason="Need to determine eligibility under INA 319(a) for spouses of US citizens",
        task="Search for 'INA 319(a) naturalization spouse US citizen 3 year rule requirements'",
        expectation="Find specific eligibility criteria for spouses of US citizens"
    )
    
    legal_context = {
        "step_1": "Found information about naturalization requirements",
        "step_2": "Identified continuous residence rules"
    }
    
    query = build_legal_search_query(step, legal_context)
    print(f"Original Task: {step.task}")
    print(f"Built Query: {query}")
    print()

async def main():
    """Main test function"""
    print("CoT Logic Component Testing")
    print("=" * 50)
    print()
    
    test_complexity_detection()
    await test_legal_intent_extraction()
    test_legal_step_validation()
    test_search_query_building()
    
    print("Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main())
