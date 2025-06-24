"""
LLM Providers abstraction layer for supporting multiple LLM services.

This module provides a factory pattern for creating LLM providers based on
configuration, supporting both OpenAI and Meta LLM services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from loguru import logger
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.metrics import record_llm_tokens, ERROR_COUNTER


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response asynchronously."""
        pass
    
    @abstractmethod
    def __call__(self, prompt: Any) -> str:
        """Call method for compatibility with LangChain."""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider implementation."""
    
    def __init__(self):
        """Initialize the OpenAI provider."""
        self._client = None
        self._thread_pool = ThreadPoolExecutor(max_workers=5)
    
    @property
    def client(self):
        """Get the Azure OpenAI client, initializing if necessary."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
                from langsmith.wrappers import wrap_openai
                
                # Configure Azure OpenAI client
                client = AzureOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                    azure_deployment=settings.AZURE_OPENAI_API_LLM_DEPLOYMENT_ID
                )
                
                # Wrap with LangSmith for metrics
                self._client = wrap_openai(client)
                
                logger.info(f"Initialized Azure OpenAI client with model {settings.AZURE_OPENAI_LLM_MODEL}")
                
            except Exception as e:
                ERROR_COUNTER.labels(
                    error_type="ConnectionError",
                    component="azure_openai"
                ).inc()
                logger.error(f"Failed to initialize Azure OpenAI client: {e}")
                raise RuntimeError(f"Failed to initialize Azure OpenAI client: {str(e)}")
        
        return self._client
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3)
    )
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the OpenAI LLM with retry logic."""
        system_content = system_prompt or "You are a helpful assistant."
        
        try:
            response = self.client.chat.completions.create(
                model=settings.AZURE_OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Track token usage
            usage = getattr(response, 'usage', None)
            if usage:
                record_llm_tokens(
                    settings.AZURE_OPENAI_LLM_MODEL,
                    "completion",
                    usage.total_tokens
                )
            
            return response.choices[0].message.content
            
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="APIError",
                component="azure_openai"
            ).inc()
            logger.error(f"Error calling Azure OpenAI: {e}")
            raise
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response asynchronously."""
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._thread_pool,
            lambda: self._call_llm(prompt, system_prompt)
        )
    
    def __call__(self, prompt: Any) -> str:
        """Call method for compatibility with LangChain."""
        # Extract prompt content
        if hasattr(prompt, 'content'):
            prompt_content = prompt.content
        else:
            prompt_content = str(prompt)
        
        return self._call_llm(prompt_content)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "OpenAI",
            "model": settings.AZURE_OPENAI_LLM_MODEL,
            "endpoint": settings.AZURE_OPENAI_ENDPOINT,
            "api_version": settings.AZURE_OPENAI_API_VERSION,
            "deployment_id": settings.AZURE_OPENAI_API_LLM_DEPLOYMENT_ID
        }


class MetaProvider(LLMProvider):
    """Meta LLM provider implementation."""
    
    def __init__(self):
        """Initialize the Meta provider."""
        self._client = None
        self._thread_pool = ThreadPoolExecutor(max_workers=5)
    
    @property
    def client(self):
        """Get the Azure Meta LLM client, initializing if necessary."""
        if self._client is None:
            try:
                from azure.ai.inference import ChatCompletionsClient
                from azure.core.credentials import AzureKeyCredential
                
                # Configure Azure Meta client
                self._client = ChatCompletionsClient(
                    endpoint=settings.AZURE_META_ENDPOINT,
                    credential=AzureKeyCredential(settings.AZURE_META_API_KEY),
                    api_version=settings.AZURE_META_API_VERSION
                )
                
                logger.info(f"Initialized Azure Meta client with model {settings.AZURE_META_LLM_MODEL}")
                
            except Exception as e:
                ERROR_COUNTER.labels(
                    error_type="ConnectionError",
                    component="azure_meta"
                ).inc()
                logger.error(f"Failed to initialize Azure Meta client: {e}")
                raise RuntimeError(f"Failed to initialize Azure Meta client: {str(e)}")
        
        return self._client
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(3)
    )
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Call the Meta LLM with retry logic."""
        try:
            from azure.ai.inference.models import SystemMessage, UserMessage
            
            # Prepare messages
            messages = []
            
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            else:
                messages.append(SystemMessage(content="You are a helpful AI assistant."))
            
            messages.append(UserMessage(content=prompt))
            
            # Default parameters
            params = {
                "messages": messages,
                "model": settings.AZURE_META_LLM_MODEL,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            }
            
            # Make the API call
            response = self.client.complete(**params)
            
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                # Track token usage (if available)
                usage = getattr(response, 'usage', None)
                if usage:
                    record_llm_tokens(
                        settings.AZURE_META_LLM_MODEL,
                        "completion",
                        getattr(usage, 'total_tokens', 0)
                    )
                
                return content
            else:
                raise RuntimeError("No response received from the Meta model")
                
        except Exception as e:
            ERROR_COUNTER.labels(
                error_type="APIError",
                component="azure_meta"
            ).inc()
            logger.error(f"Error calling Azure Meta: {e}")
            raise
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response asynchronously."""
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._thread_pool,
            lambda: self._call_llm(prompt, system_prompt)
        )
    
    def __call__(self, prompt: Any) -> str:
        """Call method for compatibility with LangChain."""
        # Extract prompt content
        if hasattr(prompt, 'content'):
            prompt_content = prompt.content
        else:
            prompt_content = str(prompt)
        
        return self._call_llm(prompt_content)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": "Meta",
            "model": settings.AZURE_META_LLM_MODEL,
            "endpoint": settings.AZURE_META_ENDPOINT,
            "api_version": settings.AZURE_META_API_VERSION,
            "deployment_id": settings.AZURE_META_API_LLM_DEPLOYMENT_ID
        }


class LLMProviderFactory:
    """Factory for creating LLM providers based on configuration."""
    
    @staticmethod
    def create_provider() -> LLMProvider:
        """
        Create and return the appropriate LLM provider based on configuration.
        
        Returns:
            Configured LLM provider instance
            
        Raises:
            ValueError: If AZURE_LLM_MODEL is not supported
            RuntimeError: If provider initialization fails
        """
        llm_model = settings.AZURE_LLM_MODEL.strip().lower()
        
        logger.info(f"Creating LLM provider for model type: {settings.AZURE_LLM_MODEL}")
        
        if llm_model == 'openai':
            try:
                provider = OpenAIProvider()
                logger.info("Successfully created OpenAI provider")
                return provider
            except Exception as e:
                logger.error(f"Failed to create OpenAI provider: {e}")
                raise RuntimeError(f"Failed to initialize OpenAI provider: {str(e)}")
                
        elif llm_model == 'meta':
            try:
                provider = MetaProvider()
                logger.info("Successfully created Meta provider")
                return provider
            except Exception as e:
                logger.error(f"Failed to create Meta provider: {e}")
                raise RuntimeError(f"Failed to initialize Meta provider: {str(e)}")
                
        else:
            supported_models = ['OpenAI', 'Meta']
            raise ValueError(
                f"Unsupported AZURE_LLM_MODEL: '{settings.AZURE_LLM_MODEL}'. "
                f"Supported models: {supported_models}"
            )
    
    @staticmethod
    def get_provider_info() -> Dict[str, Any]:
        """Get information about the current provider configuration."""
        try:
            provider = LLMProviderFactory.create_provider()
            info = provider.get_provider_info()
            info['status'] = 'available'
            return info
        except Exception as e:
            return {
                "provider": settings.AZURE_LLM_MODEL,
                "status": "error",
                "error": str(e)
            }


# Global provider instance
_provider_instance = None


def get_llm_provider() -> LLMProvider:
    """
    Get the global LLM provider instance.
    
    Returns:
        Configured LLM provider instance
    """
    global _provider_instance
    
    if _provider_instance is None:
        _provider_instance = LLMProviderFactory.create_provider()
        logger.info(f"Initialized global LLM provider: {_provider_instance.get_provider_info()['provider']}")
    
    return _provider_instance


def reset_llm_provider():
    """Reset the global LLM provider instance."""
    global _provider_instance
    _provider_instance = None
    logger.info("Reset global LLM provider instance")