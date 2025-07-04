#!/usr/bin/env python3
"""
Test script to verify FLUX Redux integration
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_flux_redux_import():
    """Test if the FLUX Redux module can be imported."""
    try:
        from postprocessing.flux_redux import process_flux_redux
        print("✅ FLUX Redux import successful")
        return True
    except ImportError as e:
        print(f"❌ FLUX Redux import failed: {e}")
        return False

def test_database_redux_function():
    """Test if the database Redux save function exists."""
    try:
        from core.database import save_flux_redux_generation
        print("✅ Database FLUX Redux function available")
        return True
    except ImportError as e:
        print(f"❌ Database FLUX Redux function missing: {e}")
        return False

def test_flux_fill_visibility_function():
    """Test if the updated visibility function handles Redux."""
    try:
        from postprocessing.flux_fill import update_flux_fill_controls_visibility
        
        # Test with Redux option
        result = update_flux_fill_controls_visibility("FLUX Redux")
        if len(result) == 7:  # Should return 7 items now (including Redux group)
            print("✅ Visibility function updated for FLUX Redux")
            return True
        else:
            print(f"❌ Visibility function returns {len(result)} items, expected 7")
            return False
    except Exception as e:
        print(f"❌ Visibility function test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔄 Testing FLUX Redux integration...")
    print("=" * 50)
    
    tests = [
        test_flux_redux_import,
        test_database_redux_function,
        test_flux_fill_visibility_function
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! FLUX Redux integration ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)