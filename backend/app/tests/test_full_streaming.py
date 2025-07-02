#!/usr/bin/env python3
"""
Script de prueba para verificar el streaming completo (backend + frontend).
"""

import asyncio
import os
import sys
import requests
import json
import time
from pathlib import Path

# Agregar el directorio backend al path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_streaming_endpoint():
    """Test the streaming endpoint directly."""
    print("=== Testing Backend Streaming Endpoint ===\n")
    
    # Test data
    test_messages = [
        {"role": "user", "content": "What is the University of Graz?"}
    ]
    
    # API URL
    api_url = os.getenv("API_URL", "http://localhost:8000")
    endpoint = f"{api_url}/api/chat/stream"
    
    print(f"Testing endpoint: {endpoint}")
    print(f"Test query: {test_messages[0]['content']}")
    print("-" * 50)
    
    try:
        # Make streaming request
        response = requests.post(
            endpoint,
            json=test_messages,
            stream=True,
            timeout=30.0
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        # Process streaming response
        print("\n📡 Streaming response:")
        print("-" * 30)
        
        full_response = ""
        chunk_count = 0
        sources = []
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    
                    if "error" in data:
                        print(f"❌ Streaming error: {data['error']}")
                        return False
                    
                    if "chunk" in data:
                        chunk = data["chunk"]
                        full_response += chunk
                        chunk_count += 1
                        
                        # Print chunk (with visual indicator)
                        print(chunk, end="", flush=True)
                        
                        # Extract metadata
                        if "sources" in data:
                            sources = data["sources"]
                    
                    elif data.get("done"):
                        print(f"\n\n✅ Streaming complete!")
                        break
                    
                    elif "status" in data:
                        print(f"ℹ️ Status: {data.get('message', 'Processing...')}")
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing streaming data: {e}")
                    continue
        
        print(f"\n📊 Streaming Statistics:")
        print(f"   Total chunks received: {chunk_count}")
        print(f"   Full response length: {len(full_response)} characters")
        print(f"   Sources found: {len(sources)}")
        
        if sources:
            print(f"\n📚 Sources:")
            for i, source in enumerate(sources[:3]):  # Show first 3 sources
                print(f"   {i+1}. {source.get('source', 'Unknown')}")
        
        return len(full_response) > 0
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the backend is running.")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_regular_endpoint():
    """Test the regular (non-streaming) endpoint for comparison."""
    print("\n=== Testing Regular Endpoint (for comparison) ===\n")
    
    # Test data
    test_messages = [
        {"role": "user", "content": "What is the University of Graz?"}
    ]
    
    # API URL
    api_url = os.getenv("API_URL", "http://localhost:8000")
    endpoint = f"{api_url}/api/chat/chat"
    
    print(f"Testing endpoint: {endpoint}")
    
    try:
        start_time = time.time()
        
        response = requests.post(
            endpoint,
            json=test_messages,
            timeout=30.0
        )
        
        end_time = time.time()
        
        if response.status_code != 200:
            print(f"❌ Request failed with status {response.status_code}")
            return False
        
        result = response.json()
        processing_time = end_time - start_time
        
        print(f"✅ Regular response received")
        print(f"   Response time: {processing_time:.2f} seconds")
        print(f"   Response length: {len(result.get('response', ''))}")
        print(f"   Sources: {len(result.get('sources', []))}")
        print(f"   From cache: {result.get('from_cache', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def check_environment():
    """Check environment configuration."""
    print("=== Environment Configuration ===\n")
    
    # Check required variables
    required_vars = [
        "AZURE_LLM_MODEL",
        "STREAMING_RESPONSE",
    ]
    
    missing_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            missing_vars.append(var)
    
    # Check optional but important variables
    optional_vars = [
        "ENABLE_FRONTEND_STREAMING",
        "API_URL",
    ]
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"⚠️ {var}: Not set (using default)")
    
    print()
    
    if missing_vars:
        print(f"❌ Missing required variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment.")
        return False
    
    return True


def main():
    """Main test function."""
    print("🧪 Full Streaming Test Suite")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Exiting.")
        sys.exit(1)
    
    print("\n🔍 Starting tests...\n")
    
    # Test streaming endpoint
    streaming_success = test_streaming_endpoint()
    
    # Test regular endpoint for comparison
    regular_success = test_regular_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Streaming endpoint: {'✅ PASS' if streaming_success else '❌ FAIL'}")
    print(f"   Regular endpoint: {'✅ PASS' if regular_success else '❌ FAIL'}")
    
    if streaming_success and regular_success:
        print("\n🎉 All tests passed! Streaming is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Start the frontend: streamlit run frontend/app.py")
        print("   2. Test streaming in the web interface")
        print("   3. Toggle ENABLE_FRONTEND_STREAMING=False to test non-streaming mode")
    else:
        print("\n⚠️ Some tests failed. Please check the backend configuration.")
        
        if not streaming_success:
            print("   - Streaming endpoint issue: Check backend logs")
            print("   - Verify STREAMING_RESPONSE=True in backend")
            
        if not regular_success:
            print("   - Regular endpoint issue: Check backend setup")
            print("   - Verify API is running and accessible")


if __name__ == "__main__":
    main()