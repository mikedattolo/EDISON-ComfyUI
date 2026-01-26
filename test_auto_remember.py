#!/usr/bin/env python3
"""
Test suite for auto-remember scoring system
Run with: python3 test_auto_remember.py
"""

import sys
sys.path.insert(0, '.')

from services.edison_core.app import test_should_remember

if __name__ == "__main__":
    print("=" * 60)
    print("EDISON Auto-Remember Scoring System Tests")
    print("=" * 60)
    print()
    test_should_remember()
    print()
    print("=" * 60)
    print("✅ All auto-remember tests passed!")
    print("=" * 60)
