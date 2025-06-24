"""
Test script to verify LLM provider functionality for both OpenAI and Meta.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from app.services.llm_providers import LLMProviderFactory, get_llm_provider, reset_llm_provider
from app.core.config import settings


async def test_provider(provider_type: str):
    """Test a specific LLM provider."""
    print(f"\n{'='*60}")
    print(f"Testing {provider_type} Provider")
    print(f"{'='*60}")
    
    # Set the environment variable for testing
    original_value = os.environ.get('AZURE_LLM_MODEL', '')
    os.environ['AZURE_LLM_MODEL'] = provider_type
    
    # Reset the provider instance to pick up the new configuration
    reset_llm_provider()
    
    try:
        # Create provider
        provider = LLMProviderFactory.create_provider()
        print(f"✅ Successfully created {provider_type} provider")
        
        # Get provider info
        info = provider.get_provider_info()
        print(f"📋 Provider Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Test synchronous call
        print(f"\n🔄 Testing synchronous call...")
        test_prompt = "What is 2 + 2? Give a very brief answer."
        try:
            sync_response = provider(test_prompt)
            print(f"✅ Sync response: {sync_response[:100]}...")
        except Exception as e:
            print(f"❌ Sync call failed: {str(e)}")
        
        # Test asynchronous call
        print(f"\n🔄 Testing asynchronous call...")
        try:
            async_response = await provider.generate_response(
                test_prompt, 
                "You are a helpful math assistant."
            )
            print(f"✅ Async response: {async_response[:100]}...")
        except Exception as e:
            print(f"❌ Async call failed: {str(e)}")
        
        print(f"✅ {provider_type} provider test completed successfully")
        
    except Exception as e:
        print(f"❌ {provider_type} provider test failed: {str(e)}")
    
    finally:
        # Restore original environment variable
        if original_value:
            os.environ['AZURE_LLM_MODEL'] = original_value
        else:
            os.environ.pop('AZURE_LLM_MODEL', None)


async def test_provider_factory():
    """Test the provider factory functionality."""
    print(f"\n{'='*60}")
    print(f"Testing Provider Factory")
    print(f"{'='*60}")
    
    # Test invalid provider
    print("🔄 Testing invalid provider...")
    original_value = os.environ.get('AZURE_LLM_MODEL', '')
    os.environ['AZURE_LLM_MODEL'] = 'InvalidProvider'
    reset_llm_provider()
    
    try:
        provider = LLMProviderFactory.create_provider()
        print("❌ Expected error for invalid provider, but succeeded")
    except ValueError as e:
        print(f"✅ Correctly caught ValueError: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    finally:
        # Restore original environment variable
        if original_value:
            os.environ['AZURE_LLM_MODEL'] = original_value
        else:
            os.environ.pop('AZURE_LLM_MODEL', None)
    
    # Test factory info
    print("\n🔄 Testing factory info...")
    try:
        info = LLMProviderFactory.get_provider_info()
        print(f"✅ Factory info: {info}")
    except Exception as e:
        print(f"❌ Factory info failed: {str(e)}")


def check_environment_variables():
    """Check if required environment variables are set."""
    print(f"\n{'='*60}")
    print(f"Environment Variables Check")
    print(f"{'='*60}")
    
    # Check AZURE_LLM_MODEL
    llm_model = os.getenv('AZURE_LLM_MODEL', '')
    print(f"AZURE_LLM_MODEL: {llm_model}")
    
    if llm_model == 'OpenAI':
        openai_vars = [
            'AZURE_OPENAI_API_KEY',
            'AZURE_OPENAI_ENDPOINT',
            'AZURE_OPENAI_API_VERSION',
            'AZURE_OPENAI_API_LLM_DEPLOYMENT_ID',
            'AZURE_OPENAI_LLM_MODEL'
        ]
        print(f"\nRequired OpenAI variables:")
        for var in openai_vars:
            value = os.getenv(var, '')
            status = "✅" if value else "❌"
            # Mask sensitive data
            display_value = value[:10] + "..." if len(value) > 10 and 'KEY' in var else value
            print(f"   {status} {var}: {display_value}")
    
    elif llm_model == 'Meta':
        meta_vars = [
            'AZURE_META_API_KEY',
            'AZURE_META_ENDPOINT',
            'AZURE_META_API_VERSION',
            'AZURE_META_API_LLM_DEPLOYMENT_ID',
            'AZURE_META_LLM_MODEL'
        ]
        print(f"\nRequired Meta variables:")
        for var in meta_vars:
            value = os.getenv(var, '')
            status = "✅" if value else "❌"
            # Mask sensitive data
            display_value = value[:10] + "..." if len(value) > 10 and 'KEY' in var else value
            print(f"   {status} {var}: {display_value}")
    
    else:
        print(f"❌ Unsupported or missing AZURE_LLM_MODEL: '{llm_model}'")


async def main():
    """Main test function."""
    print("🚀 LLM Providers Testing Suite")
    print("This script tests the LLM provider abstraction layer")
    
    # Check environment variables
    check_environment_variables()
    
    # Test factory functionality
    await test_provider_factory()
    
    # Get current LLM model from environment
    current_model = os.getenv('AZURE_LLM_MODEL', '').strip()
    
    if current_model in ['OpenAI', 'Meta']:
        # Test current provider
        await test_provider(current_model)
        
        # Optionally test the other provider if configured
        other_model = 'Meta' if current_model == 'OpenAI' else 'OpenAI'
        
        print(f"\n{'='*60}")
        print(f"Would you like to test {other_model} provider as well?")
        print(f"Note: This requires {other_model} environment variables to be set.")
        print(f"{'='*60}")
        
        # For automated testing, we'll skip the interactive part
        # await test_provider(other_model)
        
    else:
        print(f"\n❌ No valid AZURE_LLM_MODEL configured for testing")
        print(f"Please set AZURE_LLM_MODEL to either 'OpenAI' or 'Meta'")
    
    print(f"\n{'='*60}")
    print(f"Testing Complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())