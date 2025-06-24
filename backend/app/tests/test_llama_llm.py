"""
Test script for Azure Meta LLM API interaction
This script allows interactive testing of the Azure Meta LLM endpoint
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import from app
sys.path.append(str(Path(__file__).parent.parent))

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from app.core.config import settings


class AzureMetaLLMTester:
    """Test class for Azure Meta LLM API interactions"""
    
    def __init__(self):
        """Initialize the Azure Meta LLM client"""
        
        self.endpoint = AZURE_META_ENDPOINT
        self.api_key = AZURE_META_API_KEY
        self.api_version = AZURE_META_API_VERSION
        self.deployment_id = AZURE_META_API_LLM_DEPLOYMENT_ID
        self.model_name = AZURE_META_LLM_MODEL
        
        # Validate required environment variables
        self._validate_config()
        
        # Initialize the client
        self.client = ChatCompletionsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.api_key),
            api_version=self.api_version
        )
        
        print(f"✅ Azure Meta LLM Client initialized successfully")
        print(f"📍 Endpoint: {self.endpoint}")
        print(f"🤖 Model: {self.model_name}")
        print(f"📋 Deployment ID: {self.deployment_id}")
        print(f"🔢 API Version: {self.api_version}")
        print("-" * 60)
    
    def _validate_config(self):
        """Validate that all required configuration variables are set"""
        required_vars = [
            ("AZURE_META_ENDPOINT", self.endpoint),
            ("AZURE_META_API_KEY", self.api_key),
            ("AZURE_META_API_VERSION", self.api_version),
            ("AZURE_META_API_LLM_DEPLOYMENT_ID", self.deployment_id),
            ("AZURE_META_LLM_MODEL", self.model_name),
        ]
        
        missing_vars = []
        for var_name, var_value in required_vars:
            if not var_value:
                missing_vars.append(var_name)
        
        if missing_vars:
            print("❌ Missing required environment variables:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nPlease set these variables in your .env file or environment")
            sys.exit(1)
    
    def ask_question(self, question: str, system_message: str = None, **kwargs) -> str:
        """
        Send a question to the Azure Meta LLM and get a response
        
        Args:
            question: The user question
            system_message: Optional system message to set context
            **kwargs: Additional parameters for the API call
        
        Returns:
            The LLM response content
        """
        try:
            # Prepare messages
            messages = []
            
            if system_message:
                messages.append(SystemMessage(content=system_message))
            else:
                messages.append(SystemMessage(content="You are a helpful AI assistant."))
            
            messages.append(UserMessage(content=question))
            
            # Default parameters
            params = {
                "messages": messages,
                "model": self.model_name,
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            }
            
            # Override with any provided kwargs
            params.update(kwargs)
            
            print(f"🤔 Asking: {question}")
            print("⏳ Waiting for response...")
            
            # Make the API call
            response = self.client.complete(**params)
            
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content
                print(f"🤖 Response: {answer}")
                return answer
            else:
                print("❌ No response received from the model")
                return "No response received"
                
        except HttpResponseError as e:
            print(f"❌ HTTP Error: {e}")
            print(f"Status Code: {e.status_code}")
            print(f"Error Details: {e.error}")
            return f"Error: {str(e)}"
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_interactive_session(self):
        """Run an interactive question-answer session"""
        print("🚀 Starting interactive session with Azure Meta LLM")
        print("💡 Type 'quit', 'exit', or 'bye' to end the session")
        print("💡 Type 'help' for available commands")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if not question:
                    continue
                
                # Handle special commands
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                elif question.lower() == 'help':
                    self._show_help()
                    continue
                elif question.lower() == 'info':
                    self._show_info()
                    continue
                elif question.lower().startswith('system:'):
                    # Allow setting custom system message
                    system_msg = question[7:].strip()
                    question = input("❓ Your question with custom system message: ").strip()
                    if question:
                        self.ask_question(question, system_message=system_msg)
                    continue
                
                # Ask the question
                self.ask_question(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error in interactive session: {str(e)}")
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 Available commands:")
        print("  help                    - Show this help message")
        print("  info                    - Show API configuration info")
        print("  system: <message>       - Set custom system message for next question")
        print("  quit/exit/bye          - End the session")
        print("  <your question>        - Ask a question to the LLM")
    
    def _show_info(self):
        """Show API configuration information"""
        print(f"\n📊 Current Configuration:")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Model: {self.model_name}")
        print(f"  Deployment ID: {self.deployment_id}")
        print(f"  API Version: {self.api_version}")
        print(f"  Max Tokens: 2048")
        print(f"  Temperature: 0.7")
        print(f"  Top P: 0.9")
    
    def run_predefined_tests(self):
        """Run a set of predefined test questions"""
        print("🧪 Running predefined test questions...")
        print("=" * 60)
        
        test_questions = [
            {
                "question": "What is the capital of France?",
                "system": "You are a geography expert."
            },
            {
                "question": "Explain quantum computing in simple terms.",
                "system": "You are a science educator who explains complex topics simply."
            },
            {
                "question": "Write a short poem about artificial intelligence.",
                "system": "You are a creative poet."
            },
            {
                "question": "What are the benefits of renewable energy?",
                "system": "You are an environmental expert."
            }
        ]
        
        for i, test in enumerate(test_questions, 1):
            print(f"\n🧪 Test {i}/4")
            print("-" * 30)
            self.ask_question(test["question"], test["system"])
            
            # Ask user if they want to continue
            if i < len(test_questions):
                cont = input("\n⏭️  Continue to next test? (y/n): ").strip().lower()
                if cont != 'y':
                    break
        
        print("\n✅ Predefined tests completed!")


def main():
    """Main function to run the test script"""
    print("🚀 Azure Meta LLM Test Script")
    print("=" * 60)
    
    try:
        # Initialize the tester
        tester = AzureMetaLLMTester()
        
        # Show menu
        while True:
            print("\n📋 Choose an option:")
            print("1. Interactive question session")
            print("2. Run predefined tests")
            print("3. Single question test")
            print("4. Exit")
            
            choice = input("\n🔢 Enter your choice (1-4): ").strip()
            
            if choice == '1':
                tester.run_interactive_session()
            elif choice == '2':
                tester.run_predefined_tests()
            elif choice == '3':
                question = input("\n❓ Enter your question: ").strip()
                if question:
                    tester.ask_question(question)
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    except Exception as e:
        print(f"❌ Error initializing tester: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
