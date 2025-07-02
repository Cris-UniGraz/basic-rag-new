#!/usr/bin/env python3
"""
Script de prueba para verificar el streaming de respuestas en los proveedores LLM.
"""

import asyncio
import os
import sys
from pathlib import Path

# Agregar el directorio backend al path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core.config import settings
from app.services.llm_providers import LLMProviderFactory


async def test_streaming():
    """Test streaming functionality for both providers."""
    
    print("=== Testing LLM Streaming Functionality ===\n")
    
    # Test prompt
    test_prompt = "What is the University of Graz?"
    
    try:
        # Create provider
        provider = LLMProviderFactory.create_provider()
        provider_info = provider.get_provider_info()
        
        print(f"Testing provider: {provider_info['provider']}")
        print(f"Model: {provider_info['model']}")
        print(f"STREAMING_RESPONSE setting: {settings.STREAMING_RESPONSE}")
        print("-" * 50)
        
        # Test non-streaming response
        print("Testing NON-STREAMING response:")
        try:
            response = await provider.generate_response(test_prompt)
            print(f"Response: {response[:100]}...")
            print("✓ Non-streaming response successful")
        except Exception as e:
            print(f"✗ Non-streaming response failed: {e}")
        
        print("\n" + "-" * 50)
        
        # Test streaming response
        print("Testing STREAMING response:")
        try:
            chunks = []
            async for chunk in provider.generate_response_stream(test_prompt):
                chunks.append(chunk)
                print(chunk, end="", flush=True)
            
            print(f"\n\n✓ Streaming response successful with {len(chunks)} chunks")
            
            # Compare responses
            full_response = "".join(chunks)
            print(f"Full response length: {len(full_response)} characters")
            
        except Exception as e:
            print(f"✗ Streaming response failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"✗ Provider initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test completed ===")


if __name__ == "__main__":
    # Set some basic environment variables if they don't exist
    if not os.getenv("AZURE_LLM_MODEL"):
        print("⚠️ Warning: AZURE_LLM_MODEL not set. Set it to 'openai' or 'meta' to test.")
        print("Example: export AZURE_LLM_MODEL=openai")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_streaming())