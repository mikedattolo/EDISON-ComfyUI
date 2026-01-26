#!/usr/bin/env python3
"""
Test suite for fact extraction functionality
Run with: python3 test_fact_extraction.py
"""

import sys
sys.path.insert(0, '.')

from services.edison_core.app import test_extract_facts

if __name__ == "__main__":
    print("Running fact extraction tests...\n")
    test_extract_facts()
    print("\n✅ Fact extraction working correctly!")
