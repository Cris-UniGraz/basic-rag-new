#!/usr/bin/env python3
"""
Script de prueba para verificar que el fix del import funciona correctamente.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test que los imports funcionen correctamente."""
    print("🔍 Testing imports after fix...")
    
    try:
        # Test import of utility
        from pymilvus import utility
        print("✅ pymilvus.utility import successful")
        
        # Test import of RAGService
        from app.services.rag_service import RAGService
        print("✅ RAGService import successful")
        
        # Check if the method exists
        if hasattr(RAGService, 'initialize_retrievers_parallel'):
            print("✅ initialize_retrievers_parallel method exists")
        else:
            print("❌ initialize_retrievers_parallel method not found")
            return False
            
        # Test method signature
        import inspect
        method = getattr(RAGService, 'initialize_retrievers_parallel')
        sig = inspect.signature(method)
        print(f"✅ Method signature: {sig}")
        
        # Check if it's an async method
        if inspect.iscoroutinefunction(method):
            print("✅ Method is properly async")
        else:
            print("❌ Method is not async")
            return False
            
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_utility_usage():
    """Test que utility se pueda usar correctamente."""
    print("\n🔍 Testing utility usage...")
    
    try:
        from pymilvus import utility
        
        # Test basic utility functions (these should work even without connection)
        available_functions = [func for func in dir(utility) if not func.startswith('_')]
        print(f"✅ Available utility functions: {len(available_functions)}")
        print(f"   Key functions: {[f for f in available_functions if 'collection' in f.lower()]}")
        
        # Verify has_collection exists
        if hasattr(utility, 'has_collection'):
            print("✅ utility.has_collection function exists")
            return True
        else:
            print("❌ utility.has_collection function not found")
            return False
            
    except Exception as e:
        print(f"❌ Error testing utility: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing import fix for parallel retriever initialization\n")
    
    import_test_passed = test_imports()
    utility_test_passed = test_utility_usage()
    
    print(f"\n📊 Test Results:")
    print(f"   Import test: {'✅ PASSED' if import_test_passed else '❌ FAILED'}")
    print(f"   Utility test: {'✅ PASSED' if utility_test_passed else '❌ FAILED'}")
    
    if import_test_passed and utility_test_passed:
        print(f"\n🎉 All tests passed! The import fix should resolve the error.")
        print(f"   The error 'name utility is not defined' should no longer occur.")
        return True
    else:
        print(f"\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)