import streamlit as st
import httpx
import os
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv
import aiohttp
import requests

# Forzar tema oscuro antes de cualquier otra operación de Streamlit
os.environ['STREAMLIT_THEME'] = 'dark'

# Load environment variables
load_dotenv()

# Debug configuration - safe version that won't error if already running
import debugpy
try:
    # Check if debugpy is already listening - the hasattr check helps avoid errors
    if not hasattr(debugpy, "_listening"):
        debugpy.listen(("0.0.0.0", 5679))
        print("Debug server started on port 5679")
        # Uncomment these if you want to pause until the debugger connects
        # print("Waiting for debugger attach")
        # debugpy.wait_for_client()
        # print("Debugger attached")
except Exception as e:
    print(f"Debug setup error (this is not a problem): {e}")

# API configuration
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")
print(f"Using API URL: {DEFAULT_API_URL}")

# Frontend configuration
SHOW_FULL_FRONTEND = os.getenv("SHOW_FULL_FRONTEND", "True").lower() == "true"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")
ENABLE_FRONTEND_STREAMING = os.getenv("ENABLE_FRONTEND_STREAMING", "True").lower() == "true"
print(f"SHOW_FULL_FRONTEND: {SHOW_FULL_FRONTEND}")
print(f"Default COLLECTION_NAME: {COLLECTION_NAME}")
print(f"ENABLE_FRONTEND_STREAMING: {ENABLE_FRONTEND_STREAMING}")

# Try several API URLs to find one that works
ALL_API_URLS = [
    "http://localhost:8000",
    "http://backend:8000",
    "http://127.0.0.1:8000",
    "http://host.docker.internal:8000"
]


def get_api_url(endpoint: str) -> str:
    """
    Get the full API URL for an endpoint.
    
    Args:
        endpoint: API endpoint path
        
    Returns:
        Full API URL
    """
    # Use session state API URL if available, otherwise use default
    api_url = st.session_state.get("api_url", DEFAULT_API_URL)
    full_url = f"{api_url}/api{endpoint}"
    print(f"[DEBUG] Constructed API URL: {full_url} (api_url={api_url}, endpoint={endpoint})")
    return full_url


# Page configuration
st.set_page_config(
    page_title="Uni AI Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add minimal custom CSS for better styling
st.markdown("""
<style>
/* Minimal styling for chat interface */
.stButton > button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 0.5rem;
}
.stButton > button:hover {
    background-color: #FF6B6B;
}
/* Specific styling for clear button */
div[data-testid="column"]:first-child .stButton > button {
    background-color: #DC3545 !important;
    color: white !important;
    border: none !important;
    border-radius: 0.5rem !important;
    height: 2.5rem !important;
    margin-top: 1.75rem !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    background-color: #C82333 !important;
}
/* Disabled button styling */
div[data-testid="column"]:first-child .stButton > button:disabled {
    background-color: #6C757D !important;
    color: #ADB5BD !important;
    cursor: not-allowed !important;
}
/* Align the clear button with chat input */
.stChatInput {
    margin-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state efficiently
session_defaults = {
    "messages": [],
    "sources": [],
    "processing_time": None,
    "collections": [],
    "api_url": DEFAULT_API_URL,
    "selected_collection": COLLECTION_NAME if not SHOW_FULL_FRONTEND and COLLECTION_NAME else ""
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# Functions for API interaction
@st.cache_data(ttl=300)
def get_collections() -> List[Dict[str, Any]]:
    """
    Get available collections from the API.
    
    Returns:
        List of collections
    """
    try:
        print(f"Fetching collections from {get_api_url('/documents/collections')}")
        with httpx.Client() as client:
            # Add debug logs
            print(f"API URL: {get_api_url('/documents/collections')}")
            print(f"Using default API URL: {DEFAULT_API_URL}")
            print(f"Session state API URL: {st.session_state.get('api_url', 'not set')}")
            
            # Try all possible API URLs
            urls_to_try = []
            
            # First try the session state URL if available
            if "api_url" in st.session_state:
                urls_to_try.append(f"{st.session_state.api_url}/api/documents/collections")
            
            # Then try all the default URLs
            for base_url in ALL_API_URLS:
                urls_to_try.append(f"{base_url}/api/documents/collections")
                
            # Remove duplicates
            urls_to_try = list(set(urls_to_try))
            
            # Debug network information
            try:
                import socket
                print(f"Hostname: {socket.gethostname()}")
                print(f"Local IP: {socket.gethostbyname(socket.gethostname())}")
            except Exception as net_err:
                print(f"Error getting network info: {net_err}")
                
            for url in urls_to_try:
                try:
                    print(f"Trying URL: {url}")
                    # Create client with more verbose logging
                    response = client.get(
                        url, 
                        timeout=30.0,  # Increased timeout
                        headers={"X-Debug": "true", "User-Agent": "Streamlit/Frontend"}
                    )
                    
                    # Print response details before checking status
                    print(f"Response status: {response.status_code}")
                    print(f"Response headers: {response.headers}")
                    
                    # Only now check status and try to parse
                    response.raise_for_status()
                    print(f"Success with URL: {url}")
                    
                    result = response.json()
                    print(f"Got result: {result}")
                    
                    # Save the working URL for future use
                    if url != get_api_url("/documents/collections"):
                        working_base_url = url.replace("/api/documents/collections", "")
                        print(f"Updating API URL to working URL: {working_base_url}")
                        st.session_state.api_url = working_base_url
                        
                    return result
                except Exception as url_error:
                    print(f"Failed with URL {url}: {url_error}")
                    print(f"Error type: {type(url_error).__name__}")
                    
                    # Try to get more details on connection errors
                    if isinstance(url_error, httpx.ConnectError):
                        print(f"Connection error details: {url_error}")
                        
                        # Test basic connectivity to the host
                        try:
                            host = url.split("://")[1].split("/")[0]  # Extract host from URL
                            port = 8000  # Default port
                            
                            if ":" in host:
                                host, port_str = host.split(":")
                                port = int(port_str)
                                
                            print(f"Testing raw socket connection to {host}:{port}")
                            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s.settimeout(3)
                            result = s.connect_ex((host, port))
                            s.close()
                            
                            if result == 0:
                                print(f"Socket connection to {host}:{port} SUCCEEDED")
                            else:
                                print(f"Socket connection to {host}:{port} FAILED with code {result}")
                        except Exception as sock_err:
                            print(f"Error during socket test: {sock_err}")
                    
                    continue
            
            # If we reach here, all URLs failed
            raise Exception("All connection attempts failed")
    except Exception as e:
        print(f"Error fetching collections: {e}")
        st.error(f"Error fetching collections: {e}")
        return []


def get_streaming_response_live(streaming_placeholder) -> Dict[str, Any]:
    """
    Get streaming response and display it live in the existing chat interface.
    
    Args:
        streaming_placeholder: Streamlit placeholder to update with streaming content
    
    Returns:
        Final API response data
    """
    try:
        # Prepare messages for API - exclude empty assistant messages (placeholders)
        messages = []
        for msg in st.session_state.messages:
            # Skip empty assistant messages (placeholders)
            if msg["role"] == "assistant" and not msg["content"]:
                continue
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Prepare parameters
        params = {}
        if "selected_collection" in st.session_state and st.session_state.selected_collection:
            params["collection_name"] = st.session_state.selected_collection
        
        # Make streaming request
        current_response = ""
        sources = []
        processing_time = 0
        from_cache = False
        first_chunk_received = False
        
        response = requests.post(
            get_api_url("/chat/stream"),
            json=messages,
            params=params,
            stream=True,
            timeout=180.0
        )
        
        response.raise_for_status()
        
        # Process streaming response
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                try:
                    data = json.loads(line[6:])  # Remove "data: " prefix
                    
                    if "error" in data:
                        streaming_placeholder.error(f"❌ Error: {data['error']}")
                        return {
                            "response": f"Error: {data['error']}",
                            "sources": [],
                            "from_cache": False,
                            "processing_time": 0
                        }
                    
                    if "chunk" in data:
                        # Add chunk to current response
                        chunk = data["chunk"]
                        current_response += chunk
                        
                        # Update the streaming display immediately with each chunk
                        # The spinner will automatically hide when we start updating the placeholder
                        streaming_placeholder.markdown(current_response)
                        first_chunk_received = True
                        
                        # Extract metadata if present
                        if "sources" in data:
                            sources = data["sources"]
                        if "processing_time" in data:
                            processing_time = data["processing_time"]
                        if "from_cache" in data:
                            from_cache = data["from_cache"]
                    
                    elif data.get("done"):
                        # Extract final metadata from done signal
                        if "sources" in data:
                            sources = data["sources"]
                        if "processing_time" in data:
                            processing_time = data["processing_time"]
                        if "from_cache" in data:
                            from_cache = data["from_cache"]
                        # Streaming complete
                        break
                        
                except json.JSONDecodeError as e:
                    print(f"Error parsing streaming data: {e}")
                    continue
        
        return {
            "response": current_response,
            "sources": sources,
            "from_cache": from_cache,
            "processing_time": processing_time
        }
        
    except requests.exceptions.Timeout:
        streaming_placeholder.error("⏰ Die Anfrage ist abgelaufen. Bitte versuchen Sie es erneut.")
        return {
            "response": "Die Anfrage ist abgelaufen. Bitte versuchen Sie es erneut.",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }
    except requests.exceptions.RequestException as e:
        streaming_placeholder.error(f"🌐 Netzwerkfehler: {str(e)}")
        return {
            "response": f"Netzwerkfehler: {str(e)}",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }
    except Exception as e:
        streaming_placeholder.error(f"Fehler beim Streaming: {e}")
        return {
            "response": f"Entschuldigung, ein Fehler ist aufgetreten: {str(e)}",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }


def get_streaming_response(message: str, message_placeholder) -> Dict[str, Any]:
    """
    Get streaming response using Streamlit's write_stream functionality.
    
    Args:
        message: User message
        message_placeholder: Streamlit placeholder for response
        
    Returns:
        Final API response data
    """
    try:
        # Prepare messages for API
        messages = []
        for msg in st.session_state.messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message (already added to session state)
        # Don't add again to avoid duplication
        
        # Prepare parameters
        params = {}
        if "selected_collection" in st.session_state and st.session_state.selected_collection:
            params["collection_name"] = st.session_state.selected_collection
        
        # Make streaming request
        current_response = ""
        sources = []
        processing_time = 0
        from_cache = False
        
        def response_generator():
            """Generator function for streaming response."""
            nonlocal current_response, sources, processing_time, from_cache
            
            try:
                response = requests.post(
                    get_api_url("/chat/stream"),
                    json=messages,
                    params=params,
                    stream=True,
                    timeout=180.0
                )
                
                response.raise_for_status()
                
                # Process streaming response
                for line in response.iter_lines(decode_unicode=True):
                    if line and line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Remove "data: " prefix
                            
                            if "error" in data:
                                yield f"\n❌ Error: {data['error']}"
                                return
                            
                            if "chunk" in data:
                                # Yield the chunk for streaming display
                                chunk = data["chunk"]
                                current_response += chunk
                                yield chunk
                                
                                # Extract metadata if present
                                if "sources" in data:
                                    sources = data["sources"]
                                if "processing_time" in data:
                                    processing_time = data["processing_time"]
                                if "from_cache" in data:
                                    from_cache = data["from_cache"]
                            
                            elif data.get("done"):
                                # Streaming complete
                                break
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parsing streaming data: {e}")
                            continue
                            
            except requests.exceptions.Timeout:
                yield "\n⏰ Die Anfrage ist abgelaufen. Bitte versuchen Sie es erneut."
                current_response = "Die Anfrage ist abgelaufen. Bitte versuchen Sie es erneut."
            except requests.exceptions.RequestException as e:
                yield f"\n🌐 Netzwerkfehler: {str(e)}"
                current_response = f"Netzwerkfehler: {str(e)}"
        
        # Clear the thinking indicator and stream the response
        message_placeholder.empty()
        streamed_text = st.write_stream(response_generator())
        
        return {
            "response": current_response or streamed_text,
            "sources": sources,
            "from_cache": from_cache,
            "processing_time": processing_time
        }
        
    except Exception as e:
        st.error(f"Fehler beim Streaming: {e}")
        return {
            "response": f"Entschuldigung, ein Fehler ist aufgetreten: {str(e)}",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }


def send_message_stream_integrated(message: str, response_placeholder) -> Dict[str, Any]:
    """
    Send a message to the streaming chat API with integrated chat display.
    
    Args:
        message: User message
        response_placeholder: Streamlit placeholder for response
        
    Returns:
        Final API response data
    """
    try:
        # Prepare messages for API
        messages = []
        for msg in st.session_state.messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Prepare parameters
        params = {}
        if "selected_collection" in st.session_state and st.session_state.selected_collection:
            params["collection_name"] = st.session_state.selected_collection
            print(f"Using selected collection for streaming chat: {st.session_state.selected_collection}")
        
        # Make streaming request
        current_response = ""
        sources = []
        processing_time = 0
        from_cache = False
        
        try:
            response = requests.post(
                get_api_url("/chat/stream"),
                json=messages,
                params=params,
                stream=True,
                timeout=180.0
            )
            
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "error" in data:
                            st.error(f"Streaming error: {data['error']}")
                            break
                        
                        if "chunk" in data:
                            # Add chunk to current response
                            current_response += data["chunk"]
                            
                            # Update the response placeholder with streaming content
                            # Use the same HTML structure as display_chat()
                            with response_placeholder.container():
                                st.markdown(f"""
                                <div class="chat-message assistant">
                                    <div class="message">
                                        <p>{current_response}</p>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Extract metadata if present
                            if "sources" in data:
                                sources = data["sources"]
                            if "processing_time" in data:
                                processing_time = data["processing_time"]
                            if "from_cache" in data:
                                from_cache = data["from_cache"]
                        
                        elif data.get("done"):
                            # Streaming complete
                            break
                            
                        elif "status" in data:
                            # Status update
                            print(f"Status: {data.get('message', 'Processing...')}")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing streaming data: {e}")
                        continue
            
            return {
                "response": current_response,
                "sources": sources,
                "from_cache": from_cache,
                "processing_time": processing_time
            }
            
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return {
                "response": "I'm sorry, the request timed out. Please try again.",
                "sources": [],
                "from_cache": False,
                "processing_time": 0
            }
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            return {
                "response": f"I'm sorry, I encountered a network error: {str(e)}",
                "sources": [],
                "from_cache": False,
                "processing_time": 0
            }
            
    except Exception as e:
        st.error(f"Error sending streaming message: {e}")
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }


def send_message_stream_improved(message: str) -> Dict[str, Any]:
    """
    Send a message to the streaming chat API with improved UI handling.
    
    Args:
        message: User message
        
    Returns:
        Final API response data
    """
    # Create a single container for the entire streaming process
    assistant_container = st.empty()
    
    # Show initial "thinking" state
    with assistant_container.container():
        st.markdown("**Assistant:** 🤔 Denken...")
    
    try:
        # Prepare messages for API
        messages = []
        for msg in st.session_state.messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Prepare parameters
        params = {}
        if "selected_collection" in st.session_state and st.session_state.selected_collection:
            params["collection_name"] = st.session_state.selected_collection
            print(f"Using selected collection for streaming chat: {st.session_state.selected_collection}")
        
        # Make streaming request
        current_response = ""
        sources = []
        processing_time = 0
        from_cache = False
        
        try:
            response = requests.post(
                get_api_url("/chat/stream"),
                json=messages,
                params=params,
                stream=True,
                timeout=180.0
            )
            
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "error" in data:
                            st.error(f"Streaming error: {data['error']}")
                            break
                        
                        if "chunk" in data:
                            # Add chunk to current response
                            current_response += data["chunk"]
                            
                            # Update the container with streaming response
                            with assistant_container.container():
                                # Use markdown for better text rendering
                                st.markdown(f"**Assistant:** {current_response}")
                            
                            # Extract metadata if present
                            if "sources" in data:
                                sources = data["sources"]
                            if "processing_time" in data:
                                processing_time = data["processing_time"]
                            if "from_cache" in data:
                                from_cache = data["from_cache"]
                        
                        elif data.get("done"):
                            # Streaming complete
                            break
                            
                        elif "status" in data:
                            # Status update - update the thinking message
                            status_message = data.get('message', 'Processing...')
                            with assistant_container.container():
                                st.markdown(f"**Assistant:** 🤔 {status_message}")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing streaming data: {e}")
                        continue
            
            return {
                "response": current_response,
                "sources": sources,
                "from_cache": from_cache,
                "processing_time": processing_time
            }
            
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return {
                "response": "I'm sorry, the request timed out. Please try again.",
                "sources": [],
                "from_cache": False,
                "processing_time": 0
            }
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            return {
                "response": f"I'm sorry, I encountered a network error: {str(e)}",
                "sources": [],
                "from_cache": False,
                "processing_time": 0
            }
            
    except Exception as e:
        st.error(f"Error sending streaming message: {e}")
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }


def send_message_stream(message: str, response_container) -> Dict[str, Any]:
    """
    Send a message to the streaming chat API.
    
    Args:
        message: User message
        response_container: Streamlit container to update with streaming response
        
    Returns:
        Final API response data
    """
    try:
        # Prepare messages for API
        messages = []
        for msg in st.session_state.messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Prepare parameters
        params = {}
        if "selected_collection" in st.session_state and st.session_state.selected_collection:
            params["collection_name"] = st.session_state.selected_collection
            print(f"Using selected collection for streaming chat: {st.session_state.selected_collection}")
        
        # Make streaming request
        current_response = ""
        sources = []
        processing_time = 0
        from_cache = False
        
        try:
            response = requests.post(
                get_api_url("/chat/stream"),
                json=messages,
                params=params,
                stream=True,
                timeout=180.0
            )
            
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])  # Remove "data: " prefix
                        
                        if "error" in data:
                            st.error(f"Streaming error: {data['error']}")
                            break
                        
                        if "chunk" in data:
                            # Add chunk to current response
                            current_response += data["chunk"]
                            
                            # Update the response container in real-time
                            response_container.markdown(current_response)
                            
                            # Extract metadata if present
                            if "sources" in data:
                                sources = data["sources"]
                            if "processing_time" in data:
                                processing_time = data["processing_time"]
                            if "from_cache" in data:
                                from_cache = data["from_cache"]
                        
                        elif data.get("done"):
                            # Streaming complete
                            break
                            
                        elif "status" in data:
                            # Status update
                            print(f"Status: {data.get('message', 'Processing...')}")
                            
                    except json.JSONDecodeError as e:
                        print(f"Error parsing streaming data: {e}")
                        continue
            
            return {
                "response": current_response,
                "sources": sources,
                "from_cache": from_cache,
                "processing_time": processing_time
            }
            
        except requests.exceptions.Timeout:
            st.error("Request timed out. Please try again.")
            return {
                "response": "I'm sorry, the request timed out. Please try again.",
                "sources": [],
                "from_cache": False,
                "processing_time": 0
            }
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {e}")
            return {
                "response": f"I'm sorry, I encountered a network error: {str(e)}",
                "sources": [],
                "from_cache": False,
                "processing_time": 0
            }
            
    except Exception as e:
        st.error(f"Error sending streaming message: {e}")
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }


def send_message(message: str) -> Dict[str, Any]:
    """
    Send a message to the chat API.
    
    Args:
        message: User message
        
    Returns:
        API response
    """
    try:
        # Prepare messages for API
        messages = []
        for msg in st.session_state.messages:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        # Call API
        with httpx.Client() as client:
            # Add debug logging
            print(f"Sending request to {get_api_url('/chat')}")
            print(f"Request payload: {messages}")
            print(f"return_documents: {False}")
            
            # Create query parameters for return_documents
            params = {
                "return_documents": str(False).lower()
            }
            
            # Add collection parameter if selected
            if "selected_collection" in st.session_state and st.session_state.selected_collection:
                params["collection_name"] = st.session_state.selected_collection
                print(f"Using selected collection for chat: {st.session_state.selected_collection}")
            
            # Send request with query parameters instead of in JSON body
            print(f"Sending messages: {messages}")
            response = client.post(
                get_api_url("/chat"),
                json=messages,  # Send messages directly as the JSON body
                params=params,  # Send return_documents as query parameters
                timeout=180.0  # Longer timeout for RAG
            )
            
            # Add more debug info
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error sending message: {e}")
        return {
            "response": f"I'm sorry, I encountered an error: {str(e)}",
            "sources": [],
            "from_cache": False,
            "processing_time": 0
        }


def upload_documents(files, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload documents to the API.
    
    Args:
        files: Files to upload
        collection_name: Optional collection name
        
    Returns:
        API response
    """
    try:
        # Prepare form data
        form_data = {}
        if collection_name:
            form_data["collection_name"] = collection_name
        
        # Prepare files
        files_dict = {f"files": (file.name, file.getvalue()) for file in files}
        
        # Call API
        with httpx.Client() as client:
            response = client.post(
                get_api_url("/documents/upload"),
                data=form_data,
                files=files_dict,
                timeout=60.0  # Longer timeout for uploads
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error uploading documents: {e}")
        return {
            "message": f"Error uploading documents: {str(e)}",
            "uploaded_files": []
        }


def get_upload_progress(task_id: str) -> Dict[str, Any]:
    """
    Get upload progress from the API.
    
    Args:
        task_id: The unique ID of the upload task
        
    Returns:
        Progress data
    """
    try:
        with httpx.Client() as client:
            response = client.get(
                get_api_url(f"/documents/progress/{task_id}"),
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error fetching progress: {e}")
        return {
            "progress": 0,
            "status": "unknown",
            "message": f"Error fetching progress: {str(e)}",
            "timestamp": time.time()
        }


def search_documents(query: str, collection_name: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    """
    Search for documents using the API.
    
    Args:
        query: Search query
        collection_name: Optional collection name
        top_k: Number of results to return
        
    Returns:
        API response
    """
    try:
        # Prepare query parameters
        params = {
            "query": query,
            "top_k": top_k
        }
        if collection_name:
            params["collection_name"] = collection_name
        
        # Call API
        with httpx.Client() as client:
            response = client.get(
                get_api_url("/documents/search"),
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return {
            "documents": [],
            "total": 0,
            "reranked": False,
            "from_cache": False,
            "search_time": 0
        }


# UI components
def display_chat():
    """Display the chat interface using Streamlit's native chat elements."""
    # Display all messages in the chat history
    for i, message in enumerate(st.session_state.messages):
        # INICIO: Modificación para mensajes de usuario
        if message['role'] == 'user':
            # Usar HTML/CSS para alinear el mensaje del usuario a la derecha
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
                    <div style="background-color: #0072C6; color: white; border-radius: 0.5rem; padding: 0.5rem 1rem; max-width: 70%; text-align: left;">
                        {message['content']}
                    </div>
                    <div style="font-size: 1.75rem; margin-left: 0.5rem; align-self: flex-end;">
                        👤
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        # FIN: Modificación para mensajes de usuario
        else: # Mantener la lógica original para el asistente
            with st.chat_message(message["role"]):
                # Check if this is an empty assistant message (placeholder) and we're pending response
                is_streaming_placeholder = (message["role"] == "assistant" and 
                                          not message["content"] and
                                          st.session_state.get("pending_response", False))
                
                if is_streaming_placeholder:
                    # This is a live streaming message - show current content
                    streaming_placeholder = st.empty()
                    
                    # Show spinner while processing
                    with st.spinner("Denken..."):
                        # Start streaming - this will update the placeholder in real-time
                        response_data = get_streaming_response_live(streaming_placeholder)
                    
                    # Update the message content with final response
                    st.session_state.messages[i]["content"] = response_data["response"]
                    st.session_state.messages[i]["sources"] = response_data["sources"]
                    st.session_state.messages[i]["processing_time"] = response_data["processing_time"]
                    
                    # Clear pending response flag
                    st.session_state.pending_response = False
                    
                    # Rerun to show final content with sources and processing time
                    st.rerun()
                else:
                    # Regular message display
                    st.markdown(message["content"])
                    
                    # Show sources for completed assistant messages
                    if message['role'] == 'assistant' and 'sources' in message and message['sources']:
                        sources = message['sources']
                        with st.expander("📚 Quellen", expanded=False):
                            for j, source in enumerate(sources, 1):
                                st.markdown(f"**{j}. {source['source']}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"📄 Seite: {source.get('page_number', 'N/A')}")
                                    st.write(f"📁 Typ: {source.get('file_type', 'Unknown')}")
                                with col2:
                                    if source.get('sheet_name'):
                                        st.write(f"📊 Blatt: {source['sheet_name']}")
                                    if 'reranking_score' in source:
                                        st.write(f"⭐ Relevanz: {source.get('reranking_score', 0):.3f}")
                                if j < len(sources):
                                    st.divider()
                    
                    # Show processing time for completed assistant messages
                    if message['role'] == 'assistant' and 'processing_time' in message and message['processing_time'] > 0:
                        st.caption(f"⏱️ Antwort generiert in {message['processing_time']:.2f} Sekunden")


def handle_user_input():
    """Handle user input with streaming support using Streamlit chat elements."""
    # Create input area at the bottom with clear button inline
    col1, col2 = st.columns([0.35, 9.65])
    
    with col1:
        # Red clear button - only show if there are messages to clear
        if st.session_state.messages:
            if st.button("🗑️", key="clear_chat", help="Chat löschen"):
                # Clear all chat data efficiently
                st.session_state.messages.clear()
                st.session_state.sources.clear() if hasattr(st.session_state, 'sources') else None
                st.session_state.processing_time = None
                # Clear the pending response flag
                if "pending_response" in st.session_state:
                    del st.session_state["pending_response"]
                # Clear any streaming state
                if "streaming_content" in st.session_state:
                    del st.session_state["streaming_content"]
                st.rerun()
        else:
            # Show disabled button when no messages
            st.button("🗑️", key="clear_chat_disabled", help="Kein Chat zum Löschen", disabled=True)
        
        st.markdown(
            """
            <style>
            [data-testid="stButton"] button {
                background-color: #E53935 !important;
                color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        user_input = st.chat_input("Stelle eine Frage...")
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        # Mark that we're waiting for a response
        st.session_state.pending_response = True
        
        # Rerun to show the user message and start processing
        st.rerun()


def documents_tab():
    """Display the documents management tab."""
    st.header("Document Management")
    
    # Refresh collections
    if st.button("Refresh Collections", key="refresh_documents_collections"):
        with st.spinner("Refreshing collections..."):
            st.session_state.collections = get_collections()
            st.success("Collections refreshed!")
    
    # Display collections
    if not st.session_state.collections:
        with st.spinner("Loading collections..."):
            st.session_state.collections = get_collections()
    
    if st.session_state.collections:
        st.subheader("Available Collections")
        
        # Process collections - simplified for unified processing
        processed_collections = {}
        for collection in st.session_state.collections:
            name = collection.get("name", "")
            count = collection.get("count", 0)
            exists = collection.get("exists", False)
            
            processed_collections[name] = {
                "name": name,
                "total_docs": count,
                "exists": exists
            }
        
        # Display collections with delete buttons
        for collection_name, collection_info in processed_collections.items():
            col1, col2, col3 = st.columns([4, 2, 1])
            
            with col1:
                st.write(f"**{collection_info['name']}**")
            
            with col2:
                st.write(f"Documents: {collection_info['total_docs']}")
            
            with col3:
                if st.button("Delete", key=f"delete_{collection_name}"):
                    with st.spinner(f"Deleting collection '{collection_name}'..."):
                        try:
                            with httpx.Client() as client:
                                response = client.delete(
                                    get_api_url(f"/documents/collections/{collection_name}"),
                                    timeout=10.0
                                )
                                response.raise_for_status()
                                st.success(f"Collection '{collection_name}' deleted successfully!")
                                # Refresh collections after delete
                                st.session_state.collections = get_collections()
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting collection: {e}")
            
            st.markdown("---")
    else:
        st.info("No collections available.")
    
    # Upload documents
    st.subheader("Upload Documents")
    
    with st.form("upload_form"):
        # Upload files
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["pdf", "docx", "xlsx", "txt", "csv", "json", "md"]
        )
        
        # Language selection removed - using unified processing
        
        # Collection name - now required
        collection_name = st.text_input(
            "Collection Name (required)",
            placeholder="Enter collection name"
        )
        
        # Submit button
        submit_button = st.form_submit_button("Upload Documents")
        
        if submit_button:
            # Validate inputs
            if not uploaded_files:
                st.error("Please select at least one file to upload.")
            elif not collection_name:
                st.error("Collection name is required. Please enter a collection name.")
            else:
                with st.spinner("Uploading documents..."):
                    response = upload_documents(
                        uploaded_files,
                        collection_name
                    )
                    
                    if "message" in response:
                        st.success(response["message"])
                        
                        # Display uploaded files
                        if response.get("uploaded_files"):
                            st.write("Uploaded files:")
                            for file in response["uploaded_files"]:
                                st.write(f"- {file}")
                        
                        # Get task_id for progress tracking
                        task_id = response.get("task_id")
                        if task_id:
                            # Create a progress bar container - it will be updated in the background
                            progress_container = st.empty()
                            with progress_container.container():
                                st.write("**Processing Documents:**")
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Initialize the progress
                                progress_bar.progress(0)
                                
                                # Poll for progress updates
                                progress_complete = False
                                while not progress_complete:
                                    # Get current progress
                                    progress_data = get_upload_progress(task_id)
                                    progress = progress_data.get("progress", 0)
                                    status = progress_data.get("status", "processing")
                                    message = progress_data.get("message", "Processing documents...")
                                    
                                    # Update UI
                                    if progress >= 0:
                                        progress_bar.progress(progress / 100)
                                    status_text.text(message)
                                    
                                    # Check if processing is complete or failed
                                    if status == "completed":
                                        progress_bar.progress(1.0)
                                        status_text.text("Processing complete!")
                                        progress_complete = True
                                    elif status == "error":
                                        progress_bar.progress(1.0)
                                        status_text.text(f"Error: {message}")
                                        progress_complete = True
                                    
                                    # Wait before next poll
                                    time.sleep(1)
                        else:
                            # Fallback to simulated progress if no task_id was returned
                            progress_container = st.empty()
                            with progress_container.container():
                                st.write("**Processing Documents:**")
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # Initialize the progress
                                progress_bar.progress(0)
                                
                                # Simulate progress
                                for i in range(0, 101, 5):
                                    # Update status text based on progress
                                    if i < 20:
                                        status_text.text("Loading and parsing documents...")
                                    elif i < 40:
                                        status_text.text("Processing and splitting documents...")
                                    elif i < 70:
                                        status_text.text("Creating vector embeddings...")
                                    elif i < 90:
                                        status_text.text("Finalizing document storage...")
                                    else:
                                        status_text.text("Completing processing...")
                                    
                                    # Update progress bar
                                    progress_bar.progress(i / 100)
                                    
                                    # Artificial delay
                                    time.sleep(0.3)
                                
                                # Final update
                                progress_bar.progress(1.0)
                                status_text.text("Processing complete!")
                        
                        # Refresh collections
                        st.session_state.collections = get_collections()
    
    # Search documents
    st.subheader("Search Documents")
    
    search_query = st.text_input("Search Query")
    
    search_col1, search_col2, search_col3 = st.columns(3)
    
    with search_col1:
        search_language = st.selectbox(
            "Search Language",
            options=["german", "english"],
            index=0
        )
    
    with search_col2:
        search_collection = st.selectbox(
            "Collection",
            options=[""] + [col["name"] for col in st.session_state.collections],
            index=0
        )
    
    with search_col3:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=5)
    
    if st.button("Search") and search_query:
        with st.spinner("Searching..."):
            results = search_documents(
                search_query,
                search_collection or None,
                search_language,
                top_k
            )
            
            if results.get("documents"):
                st.success(f"Found {results['total']} results in {results['search_time']:.2f} seconds")
                
                for i, doc in enumerate(results["documents"]):
                    with st.expander(f"Result {i+1}: {doc['metadata'].get('source', 'Unknown')}"):
                        # Display metadata
                        st.markdown("**Metadata:**")
                        st.json(doc["metadata"])
                        
                        # Display content preview
                        st.markdown("**Content:**")
                        st.text(doc["content"][:500] + ("..." if len(doc["content"]) > 500 else ""))
            else:
                st.info("No results found.")


def settings_tab():
    """Display the settings tab."""
    st.header("Settings")
    
    # Make sure collections are loaded
    if not st.session_state.collections:
        st.session_state.collections = get_collections()
    
    # Language and Collection selection
    st.subheader("Chat Settings")
    
    # Language selection removed - using unified processing
    
    # Initialize collection selection in session state if not present
    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = ""
        
        # Get available collections for dropdown
        collections = []
        
        # Use collections from session state
        if st.session_state.collections:
            # Extract base collection names without language suffixes
            raw_collections = [col["name"] for col in st.session_state.collections]
            collections = []
            
            # Process collection names to extract root names (remove _de and _en suffixes)
            for name in raw_collections:
                root_name = name
                if name.endswith("_de") or name.endswith("_en"):
                    root_name = name[:-3]  # Remove the suffix
                if root_name not in collections:
                    collections.append(root_name)
            
        # Add empty option
        collections = [""] + collections
        
        # Refresh button for collections
        if st.button("Refresh Collections", key="refresh_settings_collections"):
            with st.spinner("Refreshing collections..."):
                st.session_state.collections = get_collections()
                # Update collections list
                if st.session_state.collections:
                    collections = [""] + [col["name"] for col in st.session_state.collections if col.get("exists", True)]
                st.success("Collections refreshed!")
                # Force rerun to update the dropdown
                st.rerun()
        
        # Find the index of the currently selected collection
        try:
            index = collections.index(st.session_state.selected_collection) if st.session_state.selected_collection in collections else 0
        except ValueError:
            index = 0  # Default to empty if not found
            
        # Collection selection dropdown
        selected_collection = st.selectbox(
            "Collection for Chat",
            options=collections,
            index=index,
            help="Select the collection to use for chat. If empty, the default collection will be used."
        )
        
        if selected_collection != st.session_state.selected_collection:
            st.session_state.selected_collection = selected_collection
            st.success(f"Collection for chat changed to {selected_collection or 'default'}.")
    
    # Clear chat
    st.subheader("Chat History")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.session_state.processing_time = None
        st.success("Chat history cleared.")
    
    # Streaming Configuration
    st.subheader("Streaming Settings")
    streaming_status = "Enabled" if ENABLE_FRONTEND_STREAMING else "Disabled"
    st.write(f"Frontend Streaming: {streaming_status}")
    
    if ENABLE_FRONTEND_STREAMING:
        st.info("🚀 Streaming is enabled! You'll see responses appear in real-time.")
    else:
        st.info("⚡ Streaming is disabled. Responses will appear all at once.")
    
    st.caption("Note: Streaming can be controlled via the ENABLE_FRONTEND_STREAMING environment variable.")
    
    # API connection
    st.subheader("API Connection")
    api_url = st.session_state.get("api_url", DEFAULT_API_URL)
    st.write(f"API URL: {api_url}")
    
    # Allow manual API URL change
    new_api_url = st.text_input("Change API URL", value=api_url)
    if st.button("Update API URL"):
        st.session_state.api_url = new_api_url
        st.success(f"API URL updated to: {new_api_url}")
    
    if st.button("Test Connection"):
        # Get current API URL from session state
        current_api_url = st.session_state.get("api_url", DEFAULT_API_URL)
        
        try:
            with httpx.Client() as client:
                # Try with current API URL
                health_url = f"{current_api_url}/health"
                st.info(f"Testing connection to {health_url}...")
                response = client.get(health_url, timeout=5.0)
                response.raise_for_status()
                st.success(f"Connection successful to {health_url}!")
        except Exception as e:
            st.error(f"Primary connection failed: {e}")
            
            # Try alternate URLs as fallback
            alternate_urls = [
                "http://localhost:8000/health", 
                "http://backend:8000/health",
                "http://host.docker.internal:8000/health",
                "http://127.0.0.1:8000/health"
            ]
            
            st.info("Trying alternate URLs...")
            for url in alternate_urls:
                try:
                    st.info(f"Testing {url}...")
                    with httpx.Client() as client:
                        response = client.get(url, timeout=3.0)
                        if response.status_code == 200:
                            st.success(f"Connection successful via {url}!")
                            # Update the URL for future calls
                            new_url = url.replace("/health", "")
                            st.session_state["api_url"] = new_url
                            st.info(f"Updated API URL to {new_url} for future calls")
                            break
                except Exception as e2:
                    st.warning(f"Failed with {url}: {e2}")

st.markdown("""
<iframe src="https://static.uni-graz.at/dist/unigraz/images/animatelogo.svg" 
style="border:none; width:100px; height:85px; margin-bottom:5px;"></iframe>
""", unsafe_allow_html=True)

# Main UI
def main():
    """Main application."""
    st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
    st.title("Uni AI Chatbot")
    
    # Check if we need to add an assistant response placeholder
    if st.session_state.get("pending_response", False):
        # Check if we already have an assistant placeholder
        if (not st.session_state.messages or 
            st.session_state.messages[-1]["role"] != "assistant"):
            # Add assistant placeholder message
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",  # Will be filled during streaming
                "timestamp": datetime.now().isoformat()
            })
    
    # Si SHOW_FULL_FRONTEND es True, mostrar todas las pestañas
    if SHOW_FULL_FRONTEND:
        tab1, tab2, tab3 = st.tabs(["Chat", "Documents", "Settings"])
        
        with tab1:
            # Create main container for better layout control
            chat_container = st.container()
            with chat_container:
                display_chat()
            
            # Input stays at bottom
            handle_user_input()
        
        with tab2:
            documents_tab()
        
        with tab3:
            settings_tab()
    else:
        # Si SHOW_FULL_FRONTEND es False, mostrar solo la pestaña de Chat
        # y usar la colección definida por defecto
        if COLLECTION_NAME:
            st.session_state.selected_collection = COLLECTION_NAME
            print(f"Using default collection: {COLLECTION_NAME}")
        
        # Create main container for better layout control
        chat_container = st.container()
        with chat_container:
            display_chat()
        
        # Input stays at bottom
        handle_user_input()


if __name__ == "__main__":
    main()