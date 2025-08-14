#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced Legal Agent improvements.
"""

import asyncio
import json
from app.configs import settings
from app.agent import (
    make_plan, Step, validate_legal_step, extract_legal_terms,
    build_legal_search_query, score_legal_relevance, check_step_dependencies,
    extract_legal_intent
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_legal_step_validation():
    """Test the legal step validation function."""
    print("Testing Legal Step Validation...")
    
    # Test valid legal step
    valid_step = Step(
        reason="Research naturalization requirements",
        task="Search for 'continuous residence requirements naturalization INA 316'",
        expectation="Find specific legal requirements for naturalization"
    )
    
    is_valid = validate_legal_step(valid_step)
    print(f"Valid legal step: {is_valid}")
    
    # Test invalid step
    invalid_step = Step(
        reason="General research",
        task="Look up information",
        expectation="Find some information"
    )
    
    is_invalid = validate_legal_step(invalid_step)
    print(f"Invalid step validation: {not is_invalid}")
    
    return is_valid and not is_invalid

def test_legal_term_extraction():
    """Test the legal term extraction function."""
    print("\nTesting Legal Term Extraction...")
    
    test_text = "Search for INA 316 continuous residence requirements and N-400 form for naturalization"
    
    legal_terms = extract_legal_terms(test_text)
    print(f"Extracted legal terms: {legal_terms}")
    
    expected_terms = ['INA 316', 'N-400', 'continuous residence', 'naturalization']
    all_found = all(term in legal_terms for term in expected_terms)
    
    print(f"All expected terms found: {all_found}")
    return all_found

def test_legal_search_query_building():
    """Test the legal search query building function."""
    print("\nTesting Legal Search Query Building...")
    
    step = Step(
        reason="Research spouse naturalization",
        task="Find INA 319 spouse naturalization requirements",
        expectation="Identify specific requirements for spouse naturalization"
    )
    
    context = {"step_1": "Found general naturalization requirements"}
    
    query = build_legal_search_query(step, context)
    print(f"Built search query: {query}")
    
    # Check if query contains legal terms
    has_legal_terms = any(term in query for term in ['INA 319', 'spouse', 'naturalization'])
    print(f"Query contains legal terms: {has_legal_terms}")
    
    return has_legal_terms

def test_legal_relevance_scoring():
    """Test the legal relevance scoring function."""
    print("\nTesting Legal Relevance Scoring...")
    
    content = "INA 316 specifies continuous residence requirements for naturalization. USCIS Policy Manual Chapter 3 outlines the process."
    query = "INA 316 continuous residence naturalization"
    
    score = score_legal_relevance(content, query)
    print(f"Legal relevance score: {score:.2f}")
    
    # Score should be high for relevant content
    is_high_score = score > 0.5
    print(f"High relevance score: {is_high_score}")
    
    return is_high_score

def test_step_dependencies():
    """Test the step dependency checking function."""
    print("\nTesting Step Dependencies...")
    
    steps = [
        Step(
            reason="Research eligibility",
            task="Find naturalization eligibility requirements",
            expectation="Identify basic eligibility criteria"
        ),
        Step(
            reason="Research specific requirements",
            task="Find continuous residence requirements based on eligibility findings",
            expectation="Identify specific residence requirements"
        )
    ]
    
    dependencies_valid = check_step_dependencies(steps)
    print(f"Step dependencies valid: {dependencies_valid}")
    
    return dependencies_valid

def test_legal_intent_extraction():
    """Test the legal intent extraction function."""
    print("\nTesting Legal Intent Extraction...")
    
    query = "How do I apply for naturalization? I need to know the requirements and process urgently."
    
    intent = asyncio.run(extract_legal_intent(query))
    print(f"Extracted legal intent: {intent}")
    
    # Check for expected intent patterns
    has_process = intent.get("process", False)
    has_eligibility = intent.get("eligibility", False)
    has_urgency = intent.get("urgency") == "urgent"
    
    print(f"Process intent: {has_process}")
    print(f"Eligibility intent: {has_eligibility}")
    print(f"Urgency intent: {has_urgency}")
    
    return has_process and has_eligibility and has_urgency

async def test_enhanced_make_plan():
    """Test the enhanced make_plan function."""
    print("\nTesting Enhanced Make Plan...")
    
    try:
        # Test with valid legal query
        steps = await make_plan("What are the naturalization requirements for permanent residents?", max_steps=3)
        print(f"Generated {len(steps)} steps")
        
        if steps:
            # Validate each step
            valid_steps = 0
            for i, step in enumerate(steps):
                is_valid = validate_legal_step(step)
                print(f"Step {i+1} validation: {is_valid}")
                if is_valid:
                    valid_steps += 1
            
            print(f"Valid steps: {valid_steps}/{len(steps)}")
            
            # Check dependencies
            dependencies_valid = check_step_dependencies(steps)
            print(f"Dependencies valid: {dependencies_valid}")
            
            return valid_steps > 0 and dependencies_valid
        else:
            print("No steps generated")
            return False
            
    except Exception as e:
        print(f"Error in make_plan: {e}")
        return False

async def main():
    """Run all tests."""
    print("Enhanced Legal Agent Improvement Tests")
    print("=" * 60)
    
    tests = [
        ("Legal Step Validation", test_legal_step_validation),
        ("Legal Term Extraction", test_legal_term_extraction),
        ("Legal Search Query Building", test_legal_search_query_building),
        ("Legal Relevance Scoring", test_legal_relevance_scoring),
        ("Step Dependencies", test_step_dependencies),
        ("Legal Intent Extraction", test_legal_intent_extraction),
        ("Enhanced Make Plan", test_enhanced_make_plan),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! The legal improvements are working correctly.")
    else:
        print("Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())
