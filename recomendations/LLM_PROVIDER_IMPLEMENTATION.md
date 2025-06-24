# LLM Provider Implementation

This document describes the implementation of the LLM provider abstraction layer that allows the RAG system to work with both OpenAI and Meta LLM services based on configuration.

## Overview

The implementation adds a configurable LLM provider system that automatically selects between OpenAI and Meta LLM services based on the `AZURE_LLM_MODEL` environment variable.

## Architecture

### 1. Provider Abstraction (`app/services/llm_providers.py`)

#### Abstract Base Class
- `LLMProvider`: Abstract base class defining the interface for all LLM providers
- Methods:
  - `generate_response()`: Async method for generating responses
  - `__call__()`: Sync method for LangChain compatibility
  - `get_provider_info()`: Returns provider metadata

#### Concrete Implementations
- `OpenAIProvider`: Implementation for Azure OpenAI services
- `MetaProvider`: Implementation for Azure Meta LLM services

#### Factory Pattern
- `LLMProviderFactory`: Creates the appropriate provider based on configuration
- `get_llm_provider()`: Global function to get the configured provider

### 2. Modified LLM Service (`app/services/llm_service.py`)

The existing `LLMService` class has been updated to use the provider abstraction:
- Maintains backward compatibility
- Automatically delegates to the configured provider
- Preserves existing interface for seamless integration

### 3. Environment Validation (`app/core/environment_manager.py`)

Enhanced environment validation to:
- Validate `AZURE_LLM_MODEL` value (must be 'OpenAI' or 'Meta')
- Check provider-specific required variables
- Provide clear error messages for missing configuration

## Configuration

### Environment Variables

#### Required for All Configurations
```bash
AZURE_LLM_MODEL=OpenAI  # or 'Meta'
```

#### OpenAI Configuration (when AZURE_LLM_MODEL=OpenAI)
```bash
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_API_LLM_DEPLOYMENT_ID=your_deployment_id
AZURE_OPENAI_LLM_MODEL=gpt-4
AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_ID=your_embeddings_deployment_id
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

#### Meta Configuration (when AZURE_LLM_MODEL=Meta)
```bash
AZURE_META_API_KEY=your_meta_api_key
AZURE_META_ENDPOINT=your_meta_endpoint
AZURE_META_API_VERSION=2024-01-01
AZURE_META_API_LLM_DEPLOYMENT_ID=your_meta_deployment_id
AZURE_META_LLM_MODEL=llama-3.1-405b-instruct
```

## Implementation Details

### Provider Selection Logic

1. **Startup**: The system reads `AZURE_LLM_MODEL` environment variable
2. **Validation**: Ensures the value is either 'OpenAI' or 'Meta'
3. **Provider Creation**: Factory creates the appropriate provider instance
4. **Configuration Check**: Validates provider-specific environment variables
5. **Initialization**: Provider initializes the appropriate client (OpenAI or Meta)

### OpenAI Provider Features

- Uses `azure-openai` library with LangSmith integration
- Supports token usage tracking
- Implements retry logic with exponential backoff
- Thread pool for async execution
- Full compatibility with existing OpenAI-based features

### Meta Provider Features

- Uses `azure-ai-inference` library
- Simple, minimalistic implementation based on `test_llama_llm.py`
- Supports basic text generation
- Thread pool for async execution
- Error handling and logging

### Error Handling

- **Configuration Errors**: Clear messages for missing or invalid environment variables
- **Runtime Errors**: Proper exception handling with fallbacks
- **Metrics Integration**: Error tracking through existing metrics system
- **Logging**: Detailed logging for debugging and monitoring

## Testing

### Test Script (`app/tests/test_llm_providers.py`)

A comprehensive test script that:
- Validates environment variable configuration
- Tests provider creation and initialization
- Verifies both sync and async API calls
- Tests error handling for invalid configurations
- Provides detailed output for debugging

### Running Tests

```bash
# Set environment variables for the provider you want to test
export AZURE_LLM_MODEL=OpenAI
# ... set OpenAI-specific variables ...

# Run the test script
python backend/app/tests/test_llm_providers.py
```

## Usage Examples

### Basic Usage (Automatic Provider Selection)

```python
from app.services.llm_service import llm_service

# The service automatically uses the configured provider
response = await llm_service.generate_response(
    "What is the capital of France?",
    "You are a helpful geography assistant."
)

# Or use the sync interface
response = llm_service("What is 2 + 2?")
```

### Direct Provider Usage

```python
from app.services.llm_providers import get_llm_provider

# Get the configured provider
provider = get_llm_provider()

# Check provider info
info = provider.get_provider_info()
print(f"Using {info['provider']} with model {info['model']}")

# Use the provider
response = await provider.generate_response("Hello, world!")
```

### Provider Factory Usage

```python
from app.services.llm_providers import LLMProviderFactory

# Create provider based on configuration
provider = LLMProviderFactory.create_provider()

# Get provider information
info = LLMProviderFactory.get_provider_info()
```

## Migration Guide

### For Existing Code

No changes needed! The existing `llm_service` interface remains unchanged:
- `llm_service.generate_response()` works as before
- `llm_service()` call interface unchanged
- All existing RAG pipeline code continues to work

### For New Code

Use the provider abstraction for more flexibility:
```python
# Old way (still works)
from app.services.llm_service import llm_service
response = await llm_service.generate_response(prompt)

# New way (more flexible)
from app.services.llm_providers import get_llm_provider
provider = get_llm_provider()
response = await provider.generate_response(prompt)
```

## Deployment Considerations

### Environment-Specific Configuration

1. **Development**: Set `AZURE_LLM_MODEL=OpenAI` with appropriate OpenAI credentials
2. **Testing**: Can switch between providers for testing compatibility
3. **Production**: Set to the desired provider with production credentials

### Monitoring

- Provider selection is logged at startup
- Provider-specific errors are tracked in metrics
- Environment validation runs at startup with detailed reporting

### Security

- API keys are masked in logs
- Environment variables are validated but not logged
- Provider information excludes sensitive data

## Troubleshooting

### Common Issues

1. **Invalid AZURE_LLM_MODEL**: 
   - Error: "Unsupported AZURE_LLM_MODEL"
   - Solution: Set to either 'OpenAI' or 'Meta'

2. **Missing Provider Variables**:
   - Error: "Missing [Provider]-specific variable"
   - Solution: Set all required variables for the chosen provider

3. **Provider Initialization Failed**:
   - Error: "Failed to initialize [Provider] provider"
   - Solution: Check API credentials and endpoints

### Debug Steps

1. Run the test script: `python backend/app/tests/test_llm_providers.py`
2. Check environment variables
3. Verify API credentials and endpoints
4. Check application logs for detailed error messages

## Future Enhancements

### Potential Improvements

1. **Additional Providers**: Support for other LLM services (Claude, etc.)
2. **Dynamic Switching**: Runtime provider switching without restart
3. **Load Balancing**: Round-robin between multiple providers
4. **Fallback Mechanism**: Automatic fallback to secondary provider on failure
5. **Provider-Specific Optimizations**: Tailored parameters for each provider

### Extension Points

The current architecture supports easy extension:
- Add new provider classes implementing `LLMProvider`
- Update factory to recognize new provider types
- Add provider-specific configuration validation
- Implement provider-specific optimizations

## Conclusion

This implementation provides a flexible, maintainable solution for supporting multiple LLM providers while maintaining backward compatibility and adding comprehensive validation and error handling. The system can now seamlessly switch between OpenAI and Meta LLM services based on simple environment variable configuration.