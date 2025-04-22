import streamlit as st
import httpx
import os
import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv

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
    page_title="RAG Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.assistant {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
}
.source-item {
    margin-bottom: 0.5rem;
    padding: 0.5rem;
    border-radius: 0.25rem;
    background-color: rgba(151, 166, 195, 0.15);
}
.source-label {
    font-weight: bold;
    color: #9CBBE9;
}
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "language" not in st.session_state:
    st.session_state.language = "german"
if "sources" not in st.session_state:
    st.session_state.sources = []
if "processing_time" not in st.session_state:
    st.session_state.processing_time = None
if "collections" not in st.session_state:
    st.session_state.collections = []
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


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
            print(f"Sending request to {get_api_url('/chat/chat')}")
            print(f"Request payload: {messages}")
            print(f"Language: {st.session_state.language}, return_documents: {False}")
            
            # Create query parameters for language and return_documents
            params = {
                "language": st.session_state.language,
                "return_documents": str(False).lower()
            }
            
            # Add collection parameter if selected
            if "selected_collection" in st.session_state and st.session_state.selected_collection:
                params["collection_name"] = st.session_state.selected_collection
                print(f"Using selected collection for chat: {st.session_state.selected_collection}")
            
            # Send request with query parameters instead of in JSON body
            print(f"Sending messages: {messages}")
            response = client.post(
                get_api_url("/chat/chat"),
                json=messages,  # Send messages directly as the JSON body
                params=params,  # Send language and return_documents as query parameters
                timeout=60.0  # Longer timeout for RAG
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


def upload_documents(files, language: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Upload documents to the API.
    
    Args:
        files: Files to upload
        language: Document language
        collection_name: Optional collection name
        
    Returns:
        API response
    """
    try:
        # Prepare form data
        form_data = {
            "language": language
        }
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


def search_documents(query: str, collection_name: Optional[str] = None, language: str = "german", top_k: int = 5) -> Dict[str, Any]:
    """
    Search for documents using the API.
    
    Args:
        query: Search query
        collection_name: Optional collection name
        language: Document language
        top_k: Number of results to return
        
    Returns:
        API response
    """
    try:
        # Prepare query parameters
        params = {
            "query": query,
            "language": language,
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
    """Display the chat interface."""
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="message">
                    <p>{message['content']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Display sources from last assistant message
    if st.session_state.sources:
        with st.expander("Sources", expanded=False):
            for source in st.session_state.sources:
                st.markdown("---")
                st.write(f"**Source:** {source['source']}")
                st.write(f"**Page:** {source.get('page_number', 'N/A')}")
                st.write(f"**Type:** {source.get('file_type', 'Unknown')}")
                
                if source.get('sheet_name'):
                    st.write(f"**Sheet:** {source['sheet_name']}")
                
                if 'reranking_score' in source:
                    st.write(f"**Score:** {source.get('reranking_score', 0):.4f}")
    
    # Display processing time
    if st.session_state.processing_time:
        st.caption(f"Response generated in {st.session_state.processing_time:.2f} seconds")


def handle_user_input():
    """Handle user input."""
    # Get user input
    user_input = st.chat_input("Ask a question...")
    if not user_input:
        return
    
    # Add user message to state
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })
      
    # Show spinner while processing
    with st.spinner("Thinking..."):
        # Send message to API
        response = send_message(user_input)
        
        # Extract response
        assistant_message = response.get("response", "I'm sorry, I couldn't generate a response.")
        sources = response.get("sources", [])
        processing_time = response.get("processing_time", 0)
        
        # Update state
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": datetime.now().isoformat()
        })
        st.session_state.sources = sources
        st.session_state.processing_time = processing_time
    
    # Display messages (including new user message)
    display_chat()
      
    # Rerun to update UI
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
        
        # Process collections to extract root collection names
        processed_collections = {}
        for collection in st.session_state.collections:
            name = collection.get("name", "")
            count = collection.get("count", 0)
            exists = collection.get("exists", False)
            
            # Determine if this is a base collection or has language suffix
            if name.endswith("_de") or name.endswith("_en"):
                # Extract the root name (without suffix)
                root_name = name[:-3]
                suffix = name[-3:]
                
                # Initialize the root collection entry if not exists
                if root_name not in processed_collections:
                    processed_collections[root_name] = {
                        "name": root_name,
                        "total_docs": 0,
                        "languages": [],
                        "exists": False
                    }
                
                # Update the root collection info
                processed_collections[root_name]["total_docs"] += count
                processed_collections[root_name]["exists"] |= exists
                processed_collections[root_name]["languages"].append(suffix[1:])  # Add without underscore
            else:
                # Regular collection without language suffix
                if name not in processed_collections:
                    processed_collections[name] = {
                        "name": name,
                        "total_docs": count,
                        "languages": [],
                        "exists": exists
                    }
        
        # Display collections with delete buttons
        for collection_name, collection_info in processed_collections.items():
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                st.write(f"**{collection_info['name']}**")
            
            with col2:
                st.write(f"Documents: {collection_info['total_docs']}")
            
            with col3:
                languages = ", ".join(collection_info["languages"]) if collection_info["languages"] else "N/A"
                st.write(f"Languages: {languages}")
            
            with col4:
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
        
        # Language selection
        upload_language = st.selectbox(
            "Document Language",
            options=["german", "english"],
            index=0
        )
        
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
                        upload_language,
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
    col1, col2 = st.columns(2)
    
    with col1:
        language = st.selectbox(
            "Response Language",
            options=["german", "english"],
            index=0 if st.session_state.language == "german" else 1
        )
        
        if language != st.session_state.language:
            st.session_state.language = language
            st.success(f"Language changed to {language}.")
    
    with col2:
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


# Main UI
def main():
    """Main application."""
    st.title("RAG Assistant")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Documents", "Settings"])
    
    with tab1:
        display_chat()
        handle_user_input()
    
    with tab2:
        documents_tab()
    
    with tab3:
        settings_tab()


if __name__ == "__main__":
    main()