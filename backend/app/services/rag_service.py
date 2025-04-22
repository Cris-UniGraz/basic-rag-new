import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_milvus import Milvus
from langchain.chains import create_history_aware_retriever
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
from tenacity import retry, wait_exponential, stop_after_attempt

from app.core.config import settings
from app.core.embedding_manager import embedding_manager
from app.core.coroutine_manager import coroutine_manager
from app.core.metrics import measure_time, EMBEDDING_RETRIEVAL_DURATION
from app.core.cache import cache_result
from app.utils.loaders import load_documents
from app.models.vector_store import vector_store_manager
from app.models.document_store import document_store_manager


class RAGService:
    """
    Service for Retrieval-Augmented Generation operations.
    
    Features:
    - Multi-language support (German and English)
    - Ensemble retrieval for improved recall
    - Reranking for improved precision
    - Caching for performance
    - Memory and batch processing optimizations
    """
    
    def __init__(self, llm_provider: Any):
        """
        Initialize the RAG service.
        
        Args:
            llm_provider: LLM service or callable for generating responses
        """
        self.llm_provider = llm_provider
        self._retrievers = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the service components."""
        if self._initialized:
            return
        
        try:
            # Initialize embedding models
            embedding_manager.initialize_models(
                settings.GERMAN_EMBEDDING_MODEL_NAME,
                settings.ENGLISH_EMBEDDING_MODEL_NAME
            )
            
            # Connect to vector store
            vector_store_manager.connect()
            
            self._initialized = True
            logger.info("RAG service initialized")
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {e}")
            raise RuntimeError(f"Failed to initialize RAG service: {str(e)}")
    
    async def ensure_initialized(self) -> None:
        """Ensure the service is initialized."""
        if not self._initialized:
            await self.initialize()
    
    async def split_documents(
        self, 
        documents: List[Document], 
        parent_chunk_size: int = None, 
        parent_chunk_overlap: int = None
    ) -> List[Document]:
        """
        Split documents into chunks for indexing.
        
        Args:
            documents: Documents to split
            parent_chunk_size: Size of parent chunks
            parent_chunk_overlap: Overlap between parent chunks
            
        Returns:
            List of document chunks
        """
        # Use default settings if not provided
        parent_chunk_size = parent_chunk_size or settings.PARENT_CHUNK_SIZE
        parent_chunk_overlap = parent_chunk_overlap or settings.PARENT_CHUNK_OVERLAP
        
        # Create text splitter
        doc_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        split_docs = []
        
        # Process each document
        for doc in documents:
            # Split the document
            splits = doc_splitter.split_text(doc.page_content)
            
            # Create new documents for each split
            for i, split in enumerate(splits):
                # Copy metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # Add split information
                metadata['doc_chunk'] = i
                metadata['total_chunks'] = len(splits)
                
                # Create document
                split_docs.append(Document(page_content=split, metadata=metadata))
        
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs
    
    @measure_time(EMBEDDING_RETRIEVAL_DURATION, {"collection": "ensemble"})
    async def get_retriever(
        self,
        folder_path: str,
        embedding_model: Any,
        collection_name: str,
        top_k: int = 3,
        language: str = "german",
        max_concurrency: int = 5
    ) -> EnsembleRetriever:
        """
        Initialize an ensemble retriever for document retrieval.
        
        Args:
            folder_path: Path to the document folder
            embedding_model: Embedding model to use
            collection_name: Name of the vector collection
            top_k: Number of top documents to retrieve
            language: Language of the documents ("german" or "english")
            max_concurrency: Maximum number of concurrent operations
            
        Returns:
            Configured ensemble retriever
        """
        # Ensure service is initialized
        await self.ensure_initialized()
        
        # Check cache
        cache_key = f"{collection_name}_{language}_{top_k}"
        if cache_key in self._retrievers:
            return self._retrievers[cache_key]
        
        # Validate parameters
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        try:
            # Get vector store if it exists
            vector_store = vector_store_manager.get_collection(collection_name, embedding_model)
            
            if vector_store:
                # Collection exists
                children_vector_store = vector_store
                parent_collection_name = f"{collection_name}_parents"
                parent_retriever = await self.create_parent_retriever(
                    children_vector_store, 
                    parent_collection_name, 
                    top_k
                )
            else:
                # Need to create collection
                logger.info(f"Creating new collection '{collection_name}'")
                
                # Load documents
                docs = load_documents(paths=folder_path)
                logger.info(f"Loaded {len(docs)} documents from {folder_path}")
                
                # Split documents
                split_docs = await self.split_documents(docs)
                
                # Create vector store - use DONT_KEEP_COLLECTIONS from env
                vector_store = vector_store_manager.create_collection(
                    split_docs, 
                    embedding_model, 
                    collection_name
                )
                
                # Create parent retriever
                parent_collection_name = f"{collection_name}_parents"
                parent_retriever = await self.create_parent_retriever(
                    vector_store, 
                    parent_collection_name, 
                    top_k, 
                    docs=docs
                )
            
            # Configure base retriever
            base_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    base_retriever,
                    parent_retriever
                ],
                weights=[0.5, 0.5],
                c=60,
                batch_config={
                    "max_concurrency": max_concurrency
                }
            )
            
            # Configure prompt for history-aware retrieval
            contextualize_q_system_prompt = (
                f"Given a chat history and the latest user question which might reference "
                f"context in the chat history, formulate a standalone question which can "
                f"be understood without the chat history. Do NOT answer the question, just "
                f"reformulate it if needed and otherwise return it as is. Give the question in {language}."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                self.llm_provider,
                ensemble_retriever,
                contextualize_q_prompt
            )
            
            # Cache the retriever
            self._retrievers[cache_key] = history_aware_retriever
            
            logger.info(f"Created retriever for collection '{collection_name}' in {language}")
            return history_aware_retriever
            
        except Exception as e:
            logger.error(f"Error initializing retriever: {e}")
            raise RuntimeError(f"Failed to initialize retriever: {str(e)}")
    
    async def create_parent_retriever(
        self,
        vectorstore: VectorStore,
        collection_name: str, 
        top_k: int = 5,
        docs: Optional[List[Document]] = None
    ) -> ParentDocumentRetriever:
        """
        Create a parent document retriever.
        
        Args:
            vectorstore: Vector store for document embeddings
            collection_name: Name of the parent document collection
            top_k: Number of documents to retrieve
            docs: Optional list of documents to index
            
        Returns:
            Configured parent document retriever
        """
        # Create text splitters
        parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n\n", "\n\n", "\n", ".", ""],
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
            model_name="gpt-4",
            is_separator_regex=False,
        )
        
        child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators=["\n\n\n", "\n\n", "\n", ".", ""],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            model_name="gpt-4",
            is_separator_regex=False,
        )
        
        # Get document store
        store = document_store_manager.get_mongo_store(collection_name)
        
        # Check if collection exists
        collection_stats = document_store_manager.get_collection_stats(collection_name)
        
        if collection_stats.get("exists", False) and collection_stats.get("count", 0) > 0:
            logger.info(f"Using existing parent document collection: '{collection_name}'")
        else:
            if docs is None:
                raise ValueError("Documents are required when creating a new collection")
            
            logger.info(f"Creating parent document collection: '{collection_name}'")
            
            # Create temporary retriever for adding documents
            temp_retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                k=top_k,
            )
            
            # Process and add documents
            logger.info(f"Adding {len(docs)} documents to parent collection")
            for i, doc in enumerate(docs):
                # Ensure metadata exists
                if doc.metadata is None:
                    doc.metadata = {}
                
                # Add metadata
                doc.metadata['start_index'] = i
                
                # Add split information
                doc.metadata['doc_chunk'] = i
                doc.metadata['total_chunks'] = len(docs)
                
                doc.metadata['doc_id'] = str(i)
                
                # Skip empty documents
                if not doc.page_content.strip():
                    logger.warning(f"Skipping empty document {i}")
                    continue
                
                try:
                    # Add document to retriever
                    temp_retriever.add_documents([doc])
                except Exception as e:
                    logger.error(f"Error processing document {i}: {e}")
        
        # Create the final retriever
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            k=top_k,
        )
        
        return retriever
    
    @coroutine_manager.coroutine_handler(timeout=30, task_type="translation")
    async def translate_query(
        self, 
        query: str, 
        source_language: str, 
        target_language: str
    ) -> str:
        """
        Translate a query from one language to another.
        
        Args:
            query: Query to translate
            source_language: Source language
            target_language: Target language
            
        Returns:
            Translated query
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"Translate the following text from {source_language} to {target_language}. "
             "Only provide the translation, no explanations."),
            ("human", "{query}")
        ])
         
        chain = prompt | self.llm_provider | StrOutputParser()
        translated_query = await chain.ainvoke({"query": query})
        
        if settings.SHOW_INTERNAL_MESSAGES:
            logger.debug(f"Translated query:\nOriginal ({source_language}): {query}\n"
                        f"Translated ({target_language}): {translated_query}")
        
        return translated_query
    
    @cache_result(prefix="rag_results", ttl=3600)
    async def process_queries_and_combine_results(
        self,
        query: str,
        retriever_de: Any,
        retriever_en: Any,
        chat_history: List[Tuple[str, str]] = [],
        language: str = "german",
    ) -> Dict[str, Any]:
        """
        Process queries in multiple languages and combine results.
        
        This method handles cases where one or both retrievers might be None,
        which happens when collections for specific languages don't exist.
        
        Args:
            query: User query
            retriever_de: German retriever (can be None if collection doesn't exist)
            retriever_en: English retriever (can be None if collection doesn't exist)
            chat_history: Chat history
            language: Primary language for the response
        """
        """
        Process queries in multiple languages and combine results.
        
        Args:
            query: User query
            retriever_de: German retriever
            retriever_en: English retriever
            chat_history: Chat history
            language: Primary language for the response
            
        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()
            sources_for_cache = []
            
            # Process queries in appropriate languages
            if language.lower() == "german":
                query_de = query
                query_en = await self.translate_query(query, "German", "English")
            else:
                query_en = query
                query_de = await self.translate_query(query, "English", "German")
            
            # Retrieve and rerank in parallel, but only for retrievers that exist
            retrieval_tasks = []
            
            # Add German retriever task if it exists
            if retriever_de is not None:
                logger.info("Adding German retriever task")
                retrieval_tasks.append(
                    self.retrieve_context_reranked(
                        query_de, 
                        retriever_de, 
                        settings.GERMAN_COHERE_RERANKING_MODEL, 
                        chat_history, 
                        "german"
                    )
                )
            else:
                logger.warning("German retriever is None, skipping German retrieval")
                
            # Add English retriever task if it exists
            if retriever_en is not None:
                logger.info("Adding English retriever task")
                retrieval_tasks.append(
                    self.retrieve_context_reranked(
                        query_en, 
                        retriever_en, 
                        settings.ENGLISH_COHERE_RERANKING_MODEL, 
                        chat_history, 
                        "english"
                    )
                )
            else:
                logger.warning("English retriever is None, skipping English retrieval")
            
            results = await coroutine_manager.gather_coroutines(*retrieval_tasks)
            
            # Process results
            all_reranked_docs = []
            for result in results:
                if result:
                    for document in result:
                        if not isinstance(document, Document):
                            continue
                        if not hasattr(document, 'metadata') or document.metadata is None:
                            document.metadata = {}
                        all_reranked_docs.append(document)
            
            # Handle case where no relevant documents were found
            if not all_reranked_docs:
                logger.warning("No relevant documents found for the query")
                no_docs_response = "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden."
                if language.lower() == "english":
                    no_docs_response = "I'm sorry, I couldn't find relevant information about your question in the available documents."
                    
                return {
                    'response': no_docs_response, 
                    'sources': [], 
                    'from_cache': False,
                    'processing_time': time.time() - start_time
                }
            
            # Prepare context for LLM
            filtered_context = []
            sources = []
            
            # Sort documents by reranking score
            all_reranked_docs.sort(
                key=lambda x: x.metadata.get('reranking_score', 0), 
                reverse=True
            )
            
            # Take top documents up to limit
            for document in all_reranked_docs[:settings.MAX_CHUNKS_LLM]:
                source = {
                    'source': document.metadata.get('source', 'Unknown'),
                    'page_number': document.metadata.get('page_number', 'N/A'),
                    'file_type': document.metadata.get('file_type', 'Unknown'),
                    'sheet_name': document.metadata.get('sheet_name', ''),
                    'reranking_score': document.metadata.get('reranking_score', 0)
                }
                
                if source not in sources:
                    sources.append(source)
                
                filtered_context.append(document)
                sources_for_cache.append(document.metadata)
            
            # Create prompt template
            prompt_template = ChatPromptTemplate.from_template(
                """
                You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
                Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
                Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
                If the CONTEXT does not contain information to answer the QUESTION, try to answer the question with your knowledge, but only if the answer is appropriate.
                Give detailed answers in {language}.

                QUERY: ```{question}```

                CONTEXT: ```{context}```
                """
            )
            
            # Create processing chain
            chain = prompt_template | self.llm_provider | StrOutputParser()
            
            # Generate response
            response = await chain.ainvoke({
                "context": filtered_context,
                "language": language,
                "question": query
            })
            
            processing_time = time.time() - start_time
            
            return {
                'response': response,
                'sources': sources,
                'from_cache': False,
                'processing_time': processing_time,
                'documents': filtered_context,
                'sources_metadata': sources_for_cache
            }
            
        except Exception as e:
            logger.error(f"Error processing queries: {e}")
            return {
                'response': f"I'm sorry, I encountered an error while processing your request. Please try again.",
                'sources': [],
                'from_cache': False,
                'processing_time': time.time() - start_time
            }
        finally:
            await coroutine_manager.cleanup()
    
    @coroutine_manager.coroutine_handler(task_type="retrieval")
    async def retrieve_context_reranked(
        self,
        query: str,
        retriever: Any,
        reranker_model: str,
        chat_history: List[Tuple[str, str]] = [],
        language: str = "german"
    ) -> List[Document]:
        """
        Retrieve and rerank documents for a query.
        
        Args:
            query: Query string
            retriever: Document retriever
            reranker_model: Reranker model name
            chat_history: Chat history
            language: Language of the query
            
        Returns:
            List of reranked documents
        """
        try:
            # Format chat history
            formatted_history = []
            for human_msg, ai_msg in chat_history:
                formatted_history.extend([
                    HumanMessage(content=human_msg),
                    AIMessage(content=ai_msg)
                ])
            
            # Retrieve documents
            retrieved_docs = await retriever.ainvoke({
                "input": query,
                "chat_history": formatted_history,
                "language": language
            })
            
            # Skip reranking if no results
            if not retrieved_docs:
                return []
            
            # Rerank documents
            reranked_docs = await self.rerank_docs(query, retrieved_docs, reranker_model)
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Error in retrieve_context_reranked: {e}")
            return []
    
    @coroutine_manager.coroutine_handler(timeout=30, task_type="reranking")
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def rerank_docs(
        self,
        query: str,
        retrieved_docs: List[Document],
        model: str
    ) -> List[Document]:
        """
        Rerank documents using Cohere.
        
        Args:
            query: Query string
            retrieved_docs: Documents to rerank
            model: Reranking model name
            
        Returns:
            List of reranked documents
        """
        try:
            # Return original docs if no reranking model specified
            if not model or not settings.RERANKING_TYPE == "cohere":
                return retrieved_docs
            
            # Initialize Cohere client
            co = cohere.Client(settings.COHERE_API_KEY)
            
            # Extract document text
            documents = [doc.page_content for doc in retrieved_docs]
            
            # Run reranking in a thread
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: co.rerank(
                    query=query, 
                    documents=documents, 
                    model=model, 
                    return_documents=True
                )
            )
            
            # Create reranked documents
            reranked_documents = []
            for r in results.results:
                # Filter low-scoring documents
                if r.relevance_score < settings.MIN_RERANKING_SCORE:
                    continue
                    
                # Create document with original metadata and reranking score
                reranked_documents.append(
                    Document(
                        page_content=r.document.text, 
                        metadata={
                            **retrieved_docs[r.index].metadata, 
                            "reranking_score": r.relevance_score
                        }
                    )
                )
            
            logger.debug(f"Reranked {len(documents)} documents, kept {len(reranked_documents)} with scores above threshold")
            return reranked_documents
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return retrieved_docs


# Factory function to create RAG service
def create_rag_service(llm_provider: Any) -> RAGService:
    """
    Create and initialize a RAG service.
    
    Args:
        llm_provider: LLM service or callable
        
    Returns:
        Initialized RAG service
    """
    service = RAGService(llm_provider)
    
    # Initialize in background
    asyncio.create_task(service.initialize())
    
    return service