from typing import Any, Dict, List, Optional, AsyncGenerator
from loguru import logger

from app.services.llm_providers import get_llm_provider, LLMProviderFactory


class LLMService:
    """
    Service for LLM operations with provider abstraction.
    
    Features:
    - Support for multiple LLM providers (OpenAI, Meta)
    - Automatic provider selection based on configuration
    - Token usage tracking
    - Retry logic for robustness
    - Asynchronous execution
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self._provider = None
    
    @property
    def provider(self):
        """Get the LLM provider, initializing if necessary."""
        if self._provider is None:
            self._provider = get_llm_provider()
            logger.info(f"Initialized LLM service with provider: {self._provider.get_provider_info()['provider']}")
        return self._provider
    
    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response asynchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        return await self.provider.generate_response(prompt, system_prompt)
    
    async def generate_response_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response asynchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Yields:
            Response chunks as they are generated
        """
        async for chunk in self.provider.generate_response_stream(prompt, system_prompt):
            yield chunk
    
    def __call__(self, prompt: Any) -> str:
        """
        Call method for compatibility with LangChain.
        
        Args:
            prompt: Prompt string or HumanMessage
            
        Returns:
            Model response
        """
        return self.provider(prompt)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current LLM provider."""
        return self.provider.get_provider_info()


# Global instance
llm_service = LLMService()