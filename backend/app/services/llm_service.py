from typing import Any, Dict, List, Optional
from loguru import logger
from openai import AzureOpenAI
from langsmith.wrappers import wrap_openai
import os
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.core.metrics import record_llm_tokens, ERROR_COUNTER


class LLMService:
    """
    Service for LLM operations using Azure OpenAI.
    
    Features:
    - Token usage tracking
    - Retry logic for robustness
    - Asynchronous execution
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        self._client = None
        self._thread_pool = ThreadPoolExecutor(max_workers=5)
    
    @property
    def client(self) -> AzureOpenAI:
        """
        Get the Azure OpenAI client, initializing if necessary.
        
        Returns:
            Azure OpenAI client
        """
        if self._client is None:
            try:
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
        """
        Call the LLM with retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
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
        """
        Generate a response asynchronously.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Model response
        """
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._thread_pool,
            lambda: self._call_llm(prompt, system_prompt)
        )
    
    def __call__(self, prompt: Any) -> str:
        """
        Call method for compatibility with LangChain.
        
        Args:
            prompt: Prompt string or HumanMessage
            
        Returns:
            Model response
        """
        # Extract prompt content
        if hasattr(prompt, 'content'):
            prompt_content = prompt.content
        else:
            prompt_content = str(prompt)
        
        return self._call_llm(prompt_content)


# Global instance
llm_service = LLMService()