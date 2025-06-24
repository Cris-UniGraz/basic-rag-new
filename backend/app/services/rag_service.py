import os
import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from loguru import logger
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_milvus import Milvus
from langchain.chains import create_history_aware_retriever, HypotheticalDocumentEmbedder
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cohere
from tenacity import retry, wait_exponential, stop_after_attempt
from pymilvus import utility

from app.core.config import settings
from app.core.embedding_manager import embedding_manager
from app.core.coroutine_manager import coroutine_manager
from app.core.metrics import measure_time, EMBEDDING_RETRIEVAL_DURATION
from app.core.cache import cache_result
from app.core.query_optimizer import QueryOptimizer
from app.core.metrics_manager import MetricsManager
from app.core.async_metadata_processor import async_metadata_processor, MetadataType
from app.utils.loaders import load_documents
from app.utils.glossary import find_glossary_terms, find_glossary_terms_with_explanation
from app.utils.output_parsers import LineListOutputParser
from app.models.vector_store import vector_store_manager
from app.models.document_store import document_store_manager


class RAGService:
    """
    Service for Retrieval-Augmented Generation operations.

    Features:
    - Multi-language support (German and English)
    - Ensemble retrieval for improved recall
    - MultiQuery retrieval for generating multiple query variations
    - Glossary-aware query processing
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
        self.query_optimizer = QueryOptimizer()
        self.metrics_manager = MetricsManager()
    
    async def initialize(self) -> None:
        """Initialize the service components."""
        if self._initialized:
            return
        
        try:
            # Initialize unified embedding model
            embedding_manager.initialize_model(settings.EMBEDDING_MODEL_NAME)
            
            # Connect to vector store
            vector_store_manager.connect()
            
            self._initialized = True
            logger.info("RAG service initialized with unified processing")
            
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
        collection_name: str,
        top_k: int = 3,
        max_concurrency: int = 5
    ) -> EnsembleRetriever:
        """
        Initialize an ensemble retriever for document retrieval.
        
        Args:
            collection_name: Name of the vector collection
            top_k: Number of top documents to retrieve
            max_concurrency: Maximum number of concurrent operations
            
        Returns:
            Configured ensemble retriever
        """
        # Ensure service is initialized
        await self.ensure_initialized()
        
        # Check cache
        cache_key = f"{collection_name}_{top_k}"
        if cache_key in self._retrievers:
            return self._retrievers[cache_key]
        
        # Validate parameters
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        try:
            # Get unified embedding model
            embedding_model = embedding_manager.model
            
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
                # Collection doesn't exist
                logger.warning(f"Collection '{collection_name}' not found")
                return None
            
            # Configure base retriever
            base_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

            # Create multi-query retriever
            multi_query_retriever = await self.get_multi_query_retriever(parent_retriever)

            # Create HyDE retriever
            hyde_retriever = await self.get_hyde_retriever(
                embedding_model,
                collection_name,
                top_k
            )

            # Create BM25 retriever for keyword search
            bm25_retriever = self.get_bm25_retriever(collection_name, top_k)

            # Setup retrievers with config-based weights
            retrievers = [base_retriever, parent_retriever, multi_query_retriever]
            weights = [
                settings.RETRIEVER_WEIGHTS_BASE,
                settings.RETRIEVER_WEIGHTS_PARENT,
                settings.RETRIEVER_WEIGHTS_MULTI_QUERY
            ]
            
            # Normalizar pesos iniciales para que sumen 1.0
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            
            logger.info(f"Initial retriever weights: {[round(w, 3) for w in weights]}")

            # Add HyDE retriever if available
            if hyde_retriever:
                retrievers.append(hyde_retriever)
                # Recalcular pesos con HyDE
                total_weight_with_hyde = total_weight + settings.RETRIEVER_WEIGHTS_HYDE
                weights = [
                    settings.RETRIEVER_WEIGHTS_BASE / total_weight_with_hyde,
                    settings.RETRIEVER_WEIGHTS_PARENT / total_weight_with_hyde,
                    settings.RETRIEVER_WEIGHTS_MULTI_QUERY / total_weight_with_hyde,
                    settings.RETRIEVER_WEIGHTS_HYDE / total_weight_with_hyde
                ]
                logger.info(f"Added HyDE retriever to ensemble for {collection_name}")
                logger.info(f"Updated weights with HyDE: {[round(w, 3) for w in weights]}")
            else:
                logger.warning(f"HyDE retriever not available for {collection_name}")

            # Add BM25 retriever if available
            if bm25_retriever:
                retrievers.append(bm25_retriever)
                # Recalcular pesos con o sin HyDE
                if hyde_retriever:
                    # Con HyDE y BM25
                    total_weight_all = total_weight_with_hyde + settings.RETRIEVER_WEIGHTS_BM25
                    weights = [
                        settings.RETRIEVER_WEIGHTS_BASE / total_weight_all,
                        settings.RETRIEVER_WEIGHTS_PARENT / total_weight_all,
                        settings.RETRIEVER_WEIGHTS_MULTI_QUERY / total_weight_all,
                        settings.RETRIEVER_WEIGHTS_HYDE / total_weight_all,
                        settings.RETRIEVER_WEIGHTS_BM25 / total_weight_all
                    ]
                else:
                    # Sin HyDE pero con BM25
                    total_weight_with_bm25 = total_weight + settings.RETRIEVER_WEIGHTS_BM25
                    weights = [
                        settings.RETRIEVER_WEIGHTS_BASE / total_weight_with_bm25,
                        settings.RETRIEVER_WEIGHTS_PARENT / total_weight_with_bm25,
                        settings.RETRIEVER_WEIGHTS_MULTI_QUERY / total_weight_with_bm25,
                        settings.RETRIEVER_WEIGHTS_BM25 / total_weight_with_bm25
                    ]
                logger.info(f"Added BM25 retriever to ensemble for {collection_name}")
                logger.info(f"Final weights: {[round(w, 3) for w in weights]}")
            else:
                logger.warning(f"BM25 retriever not available for {collection_name}")

            # Create ensemble retriever with all retrievers
            ensemble_retriever = EnsembleRetriever(
                retrievers=retrievers,
                weights=weights,
                c=60,
                batch_config={
                    "max_concurrency": max_concurrency
                }
            )
            
            # Registrar métricas sobre retrievers utilizados y sus pesos
            retriever_info = {
                "num_retrievers": len(retrievers),
                "types": [r.__class__.__name__ for r in retrievers],
                "weights": [round(w, 3) for w in weights],
                "collection": collection_name
            }
            self.metrics_manager.log_operation(
                operation_type="ensemble_retriever_setup", 
                duration=0, 
                success=True, 
                details=retriever_info
            )
            
            # Configure prompt for history-aware retrieval
            contextualize_q_system_prompt = (
                f"Given a chat history and the latest user question which might reference "
                f"context in the chat history, formulate a standalone question which can "
                f"be understood without the chat history. Do NOT answer the question, just "
                f"reformulate it if needed and otherwise return it as is."
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
            
            logger.info(f"Created retriever for collection '{collection_name}'")
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

    def get_bm25_retriever(
        self,
        collection_name: str,
        top_k: int = 3
    ) -> BM25Retriever:
        """
        Create a BM25 retriever for keyword-based search.

        BM25 is a ranking function used to rank documents based on keyword matches
        without relying on embeddings, making it complementary to vector search.

        Args:
            collection_name: The name of the parent document collection
            top_k: Number of documents to retrieve

        Returns:
            A configured BM25Retriever or None if no documents are available
        """
        try:
            # Get documents from the parent collection in document store
            parent_collection_name = f"{collection_name}_parents"
            docstore = document_store_manager.get_mongo_store(parent_collection_name)

            # Get document IDs from the store using yield_keys() instead of list_doc_ids()
            try:
                keys = list(docstore.yield_keys())

                if not keys:
                    logger.warning(f"No documents found in collection '{parent_collection_name}' for BM25")
                    return None

                # Retrieve documents
                docs = docstore.mget(keys)
            except Exception as e:
                logger.error(f"Error accessing document keys: {e}")
                return None

            # Filter valid documents
            valid_docs = [
                doc for doc in docs
                if doc and hasattr(doc, 'page_content') and hasattr(doc, 'metadata')
            ]

            if not valid_docs:
                logger.warning("No valid documents found for BM25Retriever")
                return None

            # Create BM25Retriever
            retriever = BM25Retriever.from_documents(valid_docs)
            retriever.k = top_k

            logger.info(f"Created BM25Retriever for collection '{collection_name}' with {len(valid_docs)} documents")
            return retriever

        except Exception as e:
            logger.error(f"Error creating BM25Retriever: {e}")
            return None

    @coroutine_manager.coroutine_handler(timeout=30, task_type="query_generation")
    async def generate_step_back_query(
        self,
        query: str
    ) -> str:
        """
        Generate a more generic step-back query from the original query.

        A step-back query generalizes the question to help retrieve broader context
        that might be helpful to answer specific questions.

        Args:
            query: Original user query

        Returns:
            A more generic version of the query
        """
        # Define few-shot examples
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "What can the members of The Police do?",
            },
            {
                "input": "When was the UNIGRAzcard system introduced?",
                "output": "What is the history of the UNIGRAzcard system?",
            },
        ]

        # Create few-shot prompt template
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}"),
        ])

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        # Check if query contains glossary terms
        matching_terms = find_glossary_terms_with_explanation(query)

        if not matching_terms:
            # Standard prompt if no glossary terms
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are an expert at world knowledge. Your task is to step back and paraphrase
                    a question to a more generic step-back question, which is easier to answer.
                    Please note that the question has been asked in the context of the University of Graz.
                    Give the generic step-back question. Here are a few examples:""",
                ),
                few_shot_prompt,
                ("user", "{question}"),
            ])
        else:
            # Include glossary terms in prompt
            relevant_glossary = "\n".join([f"{term}: {explanation}"
                                       for term, explanation in matching_terms])

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are an expert at world knowledge. Your task is to step back and paraphrase
                    a question to a more generic step-back question, which is easier to answer.

                    The following terms from the question have specific meanings:
                    {relevant_glossary}

                    Please consider these specific meanings when generating the step-back question.
                    Please note that the question has been asked in the context of the University of Graz.
                    Give the generic step-back question. Here are a few examples:""",
                ),
                few_shot_prompt,
                ("user", "{question}"),
            ])

        # Create and execute chain
        chain = prompt | self.llm_provider | StrOutputParser()
        step_back_query = await chain.ainvoke({"question": query})

        if settings.SHOW_INTERNAL_MESSAGES:
            logger.debug(f"Step-back query generation:\nOriginal: {query}\n"
                       f"Step-back: {step_back_query}")

        return step_back_query

    @coroutine_manager.coroutine_handler(timeout=30, task_type="multi_query")
    async def generate_all_queries_in_one_call(
        self,
        query: str
    ) -> Dict[str, str]:
        """
        Generate all necessary query variations in a single LLM call for efficiency.

        This method generates:
        - The original query
        - A step-back version of the original query
        - Alternative formulations of the query

        Args:
            query: Original user query

        Returns:
            Dictionary containing all generated queries
        """
        # Get glossary terms
        matching_terms = find_glossary_terms_with_explanation(query)

        if not matching_terms:
            # Standard prompt if no glossary terms found - CORREGIDO: Escapar las llaves JSON
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are an AI assistant. Generate a step-back question for the given query.
                    A step-back question is more generic and broader than the original question.
                    
                    For example:
                    - Original: "Could the members of The Police perform lawful arrests?"
                    - Step-back: "What can the members of The Police do?"

                    Respond in JSON format:
                    {{{{
                        "original": "The original question",
                        "step_back": "The step-back version of the question"
                    }}}}
                    """
                ),
                ("human", "{question}")
            ])
        else:
            # Include glossary terms in prompt - CORREGIDO: Escapar las llaves JSON
            relevant_glossary = "\n".join([f"{term}: {explanation}"
                                        for term, explanation in matching_terms])

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are an AI assistant. Generate a step-back question for the given query.
                    
                    The following terms from the question have specific meanings:
                    {relevant_glossary}

                    Consider these specific meanings when generating the step-back question.
                    A step-back question is more generic and broader than the original question.

                    Respond in JSON format:
                    {{{{
                        "original": "The original question",
                        "step_back": "The step-back version of the question"
                    }}}}
                    """
                ),
                ("human", "{question}")
            ])

        # Define simplified Pydantic model for JSON validation
        class QueryOutput(BaseModel):
            original: str = Field(...)
            step_back: str = Field(...)

        # Create JSON parser
        parser = JsonOutputParser(pydantic_object=QueryOutput)

        # Create chain
        chain = prompt | self.llm_provider | parser

        try:
            # Invoke chain and get results
            result = await chain.ainvoke({"question": query})

            if settings.SHOW_INTERNAL_MESSAGES:
                logger.debug(f"Multi-query generation results for: {query}")
                # Handle both Pydantic models and regular dictionaries for debugging
                if hasattr(result, 'model_dump'):
                    result_items = result.model_dump().items()
                else:
                    result_items = result.items()

                for key, value in result_items:
                    if value:
                        logger.debug(f"- {key}: {value}")

            # Handle both Pydantic models and regular dictionaries
            if hasattr(result, 'model_dump'):  # It's a Pydantic object
                result_dict = result.model_dump()
            else:  # It's a dictionary
                result_dict = result

            logger.debug(f"Query generation result type: {type(result)}")

            # Return simplified result structure
            return {
                "original_query": result_dict.get("original", query),
                "step_back_query": result_dict.get("step_back", query),
                "multi_queries": []  # Could be extended in the future
            }

        except Exception as e:
            logger.error(f"Error generating multiple queries: {e}")
            # Fallback to basic step-back generation
            try:
                step_back = await self.generate_step_back_query(query)
                return {
                    "original_query": query,
                    "step_back_query": step_back,
                    "multi_queries": []
                }
            except Exception as fallback_error:
                logger.error(f"Fallback query generation also failed: {fallback_error}")
                return {
                    "original_query": query,
                    "step_back_query": query,
                    "multi_queries": []
                }


    async def get_hyde_retriever(
        self,
        embedding_model: Any,
        collection_name: str,
        top_k: int = 3
    ) -> Any:
        """
        Create a HyDE retriever with glossary-aware document generation.

        The Hypothetical Document Embedder (HyDE) generates a synthetic document that could
        hypothetically answer the query, then embeds that document to find similar real documents.

        Args:
            embedding_model: Base embedding model to use
            collection_name: Name of the collection to search in
            top_k: Number of documents to retrieve

        Returns:
            A configured retriever using HyDE embeddings
        """
        def create_hyde_chain(query: str):
            # Check if query contains glossary terms
            matching_terms = find_glossary_terms_with_explanation(query)

            if not matching_terms:
                # Basic prompt for document generation
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Please write a passage to answer the question."),
                    ("human", "{question}")
                ])
            else:
                # Include glossary terms in the prompt
                relevant_glossary = "\n".join([f"{term}: {explanation}"
                                          for term, explanation in matching_terms])

                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""Please write a passage to answer the question.
                    The following terms from the question have specific meanings:

                    {relevant_glossary}

                    Consider these meanings when writing your passage."""),
                    ("human", "{question}")
                ])

            # Create chain to generate hypothetical document
            chain = prompt | self.llm_provider | StrOutputParser()
            return chain

        # Custom HyDE embedder that's aware of glossary terms
        class GlossaryAwareHyDEEmbedder(HypotheticalDocumentEmbedder):
            def embed_query(self, query: str, *args, **kwargs):
                # Update the chain for each query to include potential glossary terms
                self.llm_chain = create_hyde_chain(query)

                if settings.SHOW_INTERNAL_MESSAGES:
                    # Log the prompt and response for debugging
                    try:
                        hypothetical_doc = self.llm_chain.invoke({"question": query})
                        logger.debug(f"HyDE original query: {query}")
                        logger.debug(f"HyDE generated document: {hypothetical_doc[:200]}...")
                    except Exception as e:
                        logger.error(f"Error generating HyDE document: {e}")

                # Call the parent class implementation with the updated chain
                return super().embed_query(query, *args, **kwargs)

        # Initialize with a placeholder chain that will be updated for each query
        hyde_embeddings = GlossaryAwareHyDEEmbedder(
            llm_chain=create_hyde_chain(""),  # Placeholder
            base_embeddings=embedding_model
        )

        # Get vector store with HyDE embeddings
        vector_store = vector_store_manager.get_collection(collection_name, hyde_embeddings)
        if not vector_store:
            logger.warning(f"Collection {collection_name} not found for HyDE retriever")
            return None

        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        return retriever

    async def get_multi_query_retriever(
        self,
        base_retriever: Any,
    ) -> MultiQueryRetriever:
        """
        Create a glossary-aware multi-query retriever.

        This retriever generates multiple variations of the user query by prompting the LLM,
        taking into account specialized glossary terms if present in the query.

        Args:
            base_retriever: The base retriever to use for document retrieval

        Returns:
            A configured MultiQueryRetriever
        """
        output_parser = LineListOutputParser()

        def create_multi_query_chain(query: str):
            # Check if query contains glossary terms
            matching_terms = find_glossary_terms_with_explanation(query)

            if not matching_terms:
                # Standard prompt if no glossary terms found
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an AI language model assistant. Your task is to generate
                    five different versions of the given user question to retrieve
                    relevant documents. By generating multiple perspectives on the user question,
                    your goal is to help overcome some limitations of distance-based similarity search.
                    Provide these alternative questions separated by newlines."""),
                    ("human", "{question}")
                ])
            else:
                # Include glossary terms in prompt if found
                relevant_glossary = "\n".join([f"{term}: {explanation}"
                                           for term, explanation in matching_terms])

                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are an AI language model assistant. Your task is to generate
                    five different versions of the given user question to retrieve
                    relevant documents. The following terms from the question have specific meanings:

                    {relevant_glossary}

                    Generate questions that incorporate these specific meanings. Provide these
                    alternative questions separated by newlines."""),
                    ("human", "{question}")
                ])

            # Create processing chain
            chain = prompt | self.llm_provider | output_parser
            return chain

        # Create custom MultiQueryRetriever that's aware of glossary terms
        class GlossaryAwareMultiQueryRetriever(MultiQueryRetriever):
            async def _aget_relevant_documents(self, query: str, *, run_manager=None):
                # Update the chain for each query to include potential glossary terms
                self.llm_chain = create_multi_query_chain(query)

                if settings.SHOW_INTERNAL_MESSAGES:
                    # Log the prompt and response for debugging
                    logger.debug(f"MultiQueryRetriever original query: {query}")
                    generated_queries = await self.llm_chain.ainvoke({"question": query})
                    logger.debug(f"MultiQueryRetriever generated queries: {generated_queries}")

                # Call the parent class implementation with the updated chain
                return await super()._aget_relevant_documents(query, run_manager=run_manager)

        # Initialize with a placeholder chain that will be updated for each query
        retriever = GlossaryAwareMultiQueryRetriever(
            retriever=base_retriever,
            llm_chain=create_multi_query_chain("")
        )

        return retriever

    async def _get_reformulated_query(
        self,
        query: str,
        chat_history: List[Tuple[str, str]] = []
    ) -> str:
        """
        Get the reformulated query that would be used by the history-aware retriever.
        
        Args:
            query: Original query string
            chat_history: Chat history
            
        Returns:
            Reformulated query considering chat history context
        """
        try:
            # If no chat history, return original query
            if not chat_history:
                return query
            
            # Format chat history
            formatted_history = []
            for human_msg, ai_msg in chat_history:
                formatted_history.extend([
                    HumanMessage(content=human_msg),
                    AIMessage(content=ai_msg)
                ])
            
            # Use the same prompt as the history-aware retriever
            contextualize_q_system_prompt = (
                f"Given a chat history and the latest user question which might reference "
                f"context in the chat history, formulate a standalone question which can "
                f"be understood without the chat history. Do NOT answer the question, just "
                f"reformulate it if needed and otherwise return it as is."
            )
            
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
            
            # Create chain and get reformulated query
            chain = contextualize_q_prompt | self.llm_provider | StrOutputParser()
            reformulated_query = await chain.ainvoke({
                "input": query,
                "chat_history": formatted_history
            })
            
            if settings.SHOW_INTERNAL_MESSAGES:
                logger.debug(f"Query reformulation:\nOriginal: {query}\nReformulated: {reformulated_query}")
            
            return reformulated_query.strip() if reformulated_query else query
            
        except Exception as e:
            logger.error(f"Error reformulating query: {e}")
            return query

    @coroutine_manager.coroutine_handler(task_type="retrieval")
    async def retrieve_context_without_reranking(
        self,
        query: str,
        retriever: Any,
        chat_history: List[Tuple[str, str]] = []
    ) -> List[Document]:
        """
        Retrieve documents for a query without reranking.
        
        This function is used to retrieve documents for all queries in parallel,
        which will later be combined and reranked in a single operation.
        
        Args:
            query: Query string
            retriever: Document retriever
            chat_history: Chat history
            
        Returns:
            List of retrieved documents
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
                "chat_history": formatted_history
            })
            
            # Log retrieval metrics asíncronamente
            async_metadata_processor.record_performance_async(
                "document_retrieval",
                0.0,  # Duration is measured by coroutine_handler decorator
                True,
                {
                    "query": query[:100],  # Solo los primeros 100 caracteres
                    "num_docs": len(retrieved_docs),
                    "sources": [doc.metadata.get('source', 'unknown') for doc in retrieved_docs if hasattr(doc, 'metadata')][:5]  # Máximo 5 fuentes
                }
            )
            
            return retrieved_docs
            
        except Exception as e:
            async_metadata_processor.log_async("ERROR", f"Error in retrieve_context_without_reranking: {e}", {
                "error": str(e),
                "query": query[:100]
            }, priority=3)
            return []
    
    
    @coroutine_manager.coroutine_handler(task_type="retrieval")
    async def retrieve_context_reranked(
        self,
        query: str,
        retriever: Any,
        reranker_model: str,
        chat_history: List[Tuple[str, str]] = []
    ) -> List[Document]:
        """
        Retrieve and rerank documents for a query.
        
        Args:
            query: Query string
            retriever: Document retriever
            reranker_model: Reranker model name
            chat_history: Chat history
            
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
                "chat_history": formatted_history
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
        Rerank documents using Cohere through Azure endpoint.
        
        Args:
            query: Query string
            retrieved_docs: Documents to rerank
            model: Reranking model name
            
        Returns:
            List of reranked documents
        """
        start_time = time.time()
        original_count = len(retrieved_docs)
        
        try:
            # Return original docs if no reranking model specified
            if not model or not settings.RERANKING_TYPE == "cohere":
                self.metrics_manager.log_operation(
                    operation_type="reranking_skipped",
                    duration=0,
                    success=True,
                    details={"reason": "no_model_specified"}
                )
                return retrieved_docs

            # Use Azure Cohere endpoint for reranking
            reranked_docs = await self._rerank_with_azure_cohere(query, retrieved_docs, model)
            
            # Registrar métricas de reranking
            duration = time.time() - start_time
            filtered_count = len(reranked_docs)
            
            # Extraer puntuaciones si existen
            scores = []
            for doc in reranked_docs:
                if hasattr(doc, 'metadata') and doc.metadata and 'reranking_score' in doc.metadata:
                    scores.append(doc.metadata['reranking_score'])
            
            # Registrar la operación de reranking
            self.metrics_manager.log_reranking(
                model=model,
                original_count=original_count,
                filtered_count=filtered_count,
                duration=duration,
                scores=scores
            )
            
            return reranked_docs
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error during reranking: {e}")
            
            # Registrar el error en las métricas
            self.metrics_manager.log_error(
                error_type="reranking_error",
                details=str(e),
                component="rerank_docs"
            )
            
            # Registrar la operación fallida
            self.metrics_manager.log_operation(
                operation_type="reranking",
                duration=duration,
                success=False,
                details={"error": str(e), "model": model}
            )
            
            return retrieved_docs
            
    async def _rerank_with_azure_cohere(
        self,
        query: str,
        retrieved_docs: List[Document],
        model: str
    ) -> List[Document]:
        """
        Rerank documents using Cohere through Azure endpoint.
        
        Args:
            query: Query string
            retrieved_docs: Documents to rerank
            model: Reranking model name
            
        Returns:
            List of reranked documents
        """
        import requests
        
        start_time = time.time()
        documents = [doc.page_content for doc in retrieved_docs]
        azure_cohere_endpoint = os.getenv("AZURE_COHERE_ENDPOINT")
        azure_cohere_api_key = os.getenv("AZURE_COHERE_API_KEY")
        
        if not documents:
            logger.warning("No documents to rerank")
            return retrieved_docs
        
        # Ensure endpoint doesn't end with a trailing slash
        azure_cohere_endpoint = azure_cohere_endpoint.rstrip('/')
        
        # Prepare request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": azure_cohere_api_key,
            "X-Client-Name": "RAG-Application"
        }
        
        # Prepare request payload
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": len(documents)  # Return all documents reranked
        }
        
        # Get current event loop
        loop = asyncio.get_event_loop()
        
        # Make the request in a non-blocking way
        try:
            endpoint_url = f"{azure_cohere_endpoint}/v2/rerank"
            logger.debug(f"Making reranking request to Azure Cohere endpoint")
            
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
            )
            
            if response.status_code != 200:
                logger.error(f"Error in Azure Cohere reranking: {response.status_code} - {response.text}")
                return retrieved_docs
                
            results_json = response.json()
            
            # Process results
            reranked_documents = []
            
            for result in results_json.get("results", []):
                idx = result.get("index")
                score = result.get("relevance_score")
                
                # Filter low-scoring documents
                if score < settings.MIN_RERANKING_SCORE:
                    continue
                
                # Ensure the index is valid
                if 0 <= idx < len(retrieved_docs):
                    doc = Document(
                        page_content=retrieved_docs[idx].page_content,
                        metadata={
                            **retrieved_docs[idx].metadata,
                            "reranking_score": score
                        }
                    )
                    reranked_documents.append(doc)
            
            processing_time = time.time() - start_time
            logger.debug(f"Azure Cohere reranking took {processing_time:.2f} seconds. "
                       f"Reranked {len(documents)} documents, kept {len(reranked_documents)} above threshold.")
            
            # Registrar llamada a API asíncronamente
            async_metadata_processor.record_api_call_async(
                "azure_cohere",
                "/v2/rerank",
                "POST",
                200,
                processing_time,
                len(json.dumps(payload).encode('utf-8')),
                len(response.content) if hasattr(response, 'content') else 0
            )
            
            # Registrar puntuaciones si hay resultados
            if results_json and 'results' in results_json:
                scores = [r.get('relevance_score', 0) for r in results_json['results']]
                for score in scores:
                    # Registrar puntuación mediante la función de prometheus
                    from app.core.metrics import record_reranking_score
                    record_reranking_score(model, score)
            
            return reranked_documents
            
        except Exception as e:
            logger.error(f"Exception during Azure Cohere reranking: {str(e)}")
            
            # Registrar error en API asíncronamente
            error_duration = time.time() - start_time
            async_metadata_processor.record_api_call_async(
                "azure_cohere",
                "/v2/rerank",
                "POST",
                500,
                error_duration
            )
            
            # Registrar el error
            self.metrics_manager.log_error(
                error_type="azure_cohere_reranking",
                details=str(e),
                component="rerank_with_azure_cohere"
            )
            
            return retrieved_docs
    
    async def process_query(
        self,
        query: str,
        retriever: Any,
        chat_history: List[Tuple[str, str]] = [],
    ) -> Dict[str, Any]:
        """
        Advanced asynchronous pipeline for query processing with maximum parallelization.
        
        This pipeline optimizes performance by:
        1. Running cache check, query optimization, and initial setup in parallel
        2. Overlapping query generation with embedding computation
        3. Parallelizing retrieval operations across all query variations
        4. Combining results efficiently with concurrent reranking
        
        Args:
            query: User query
            retriever: Unified retriever for the collection
            chat_history: Chat history
            
        Returns:
            Dictionary with response and metadata
        """
        def _extract_valid_results(results: List[Any], task_descriptions: List[str] = None) -> List[Any]:
            """Extract valid results from asyncio.gather results, filtering out exceptions and cancelled tasks."""
            valid_results = []
            task_descriptions = task_descriptions or [f"task_{i}" for i in range(len(results))]
            
            for i, result in enumerate(results):
                task_name = task_descriptions[i] if i < len(task_descriptions) else f"task_{i}"
                
                if isinstance(result, Exception):
                    if isinstance(result, asyncio.CancelledError):
                        logger.warning(f"Task '{task_name}' was cancelled")
                    elif isinstance(result, asyncio.TimeoutError):
                        logger.warning(f"Task '{task_name}' timed out")
                    else:
                        logger.error(f"Task '{task_name}' failed: {result}")
                    continue
                    
                # Additional check for None results
                if result is None:
                    logger.warning(f"Task '{task_name}' returned None result")
                    continue
                    
                valid_results.append(result)
            
            logger.debug(f"Extracted {len(valid_results)} valid results from {len(results)} total results")
            return valid_results

        try:
            pipeline_start_time = time.time()
            sources_for_cache = []
            
            logger.info(f"Starting async pipeline for query: '{query[:50]}...'")
            
            # === PHASE 1: PARALLEL INITIALIZATION AND CACHE CHECK ===
            phase1_start = time.time()
            
            async def cache_check_task():
                """Check cache and query optimization in parallel."""
                cached_result = self.query_optimizer.get_llm_response(query)
                if cached_result:
                    logger.info(f"Cache hit in async pipeline - Response length: {len(cached_result.get('response', ''))}")
                    self.metrics_manager.log_rag_query(
                        query=query,
                        processing_time=0.0,
                        num_sources=len(cached_result.get('sources', [])),
                        from_cache=True
                    )
                    return {
                        'response': cached_result['response'],
                        'sources': cached_result['sources'],
                        'from_cache': True,
                        'processing_time': 0.0
                    }
                return None
            
            async def embedding_generation_task():
                """Generate embedding for query optimization in parallel."""
                embedding_model = embedding_manager.model
                return await self.query_optimizer.optimize_query(query, embedding_model)
            
            async def glossary_check_task():
                """Check for glossary terms in parallel."""
                from app.utils.glossary import find_glossary_terms_with_explanation
                return find_glossary_terms_with_explanation(query)
            
            # Execute Phase 1 tasks in parallel
            phase1_results = await asyncio.gather(
                cache_check_task(),
                embedding_generation_task(),
                glossary_check_task(),
                return_exceptions=True
            )
            
            # Safely extract results
            cache_result = phase1_results[0]
            optimized_query = phase1_results[1]  
            matching_terms = phase1_results[2]
            
            phase1_time = time.time() - phase1_start
            logger.debug(f"Phase 1 (cache/optimization) completed in {phase1_time:.2f}s")
            
            # Handle cache hit
            if cache_result and not isinstance(cache_result, Exception):
                logger.info("Early return from cache in async pipeline")
                # Add pipeline metrics for cache hit (all phases are 0 except phase 1)
                cache_result['pipeline_metrics'] = {
                    'phase1_time': phase1_time,
                    'phase2_time': 0.0,
                    'phase3_time': 0.0, 
                    'phase4_time': 0.0,
                    'phase5_time': 0.0,
                    'phase6_time': 0.0,
                    'total_time': phase1_time
                }
                return cache_result
            
            # Handle optimization result
            if isinstance(optimized_query, Exception):
                logger.warning(f"Query optimization failed: {optimized_query}")
                optimized_query = {'result': {'original_query': query}, 'source': 'new'}
            
            # Handle glossary terms result
            if isinstance(matching_terms, Exception):
                logger.warning(f"Glossary check failed: {matching_terms}")
                matching_terms = []
            
            # Determine if this is a semantic cache hit
            is_semantic_cache_hit = optimized_query['source'] == 'semantic_cache'
            
            # Get reformulated query (only if not semantic cache hit)
            if is_semantic_cache_hit:
                # For semantic cache, use original query for reranking and LLM
                query_for_reranking_and_llm = query
                logger.info("Semantic cache detected - will use original query for reranking and LLM")
            else:
                # For normal processing, get reformulated query
                try:
                    query_for_reranking_and_llm = await self._get_reformulated_query(query, chat_history)
                    if query_for_reranking_and_llm != query:
                        logger.info(f"Query reformulated for reranking/LLM: '{query[:50]}...' -> '{query_for_reranking_and_llm[:50]}...'")
                    else:
                        logger.info("Query did not need reformulation (no chat history or no changes)")
                except Exception as e:
                    logger.error(f"Error getting reformulated query: {e}")
                    query_for_reranking_and_llm = query
            
            # Handle semantic cache result
            if optimized_query['source'] == 'cache':
                logger.info("Exact cache hit from optimizer")
                cache_response = optimized_query['result']
                # Add pipeline metrics for exact cache hit (all phases are 0 except phase 1)
                return {
                    'response': cache_response.get('response', ''),
                    'sources': cache_response.get('sources', []),
                    'from_cache': True,
                    'processing_time': phase1_time,
                    'documents': [],
                    'pipeline_metrics': {
                        'phase1_time': phase1_time,
                        'phase2_time': 0.0,
                        'phase3_time': 0.0,
                        'phase4_time': 0.0,
                        'phase5_time': 0.0,
                        'phase6_time': 0.0,
                        'total_time': phase1_time
                    }
                }
            elif optimized_query['source'] == 'semantic_cache':
                logger.info("Semantic cache hit from optimizer")
                semantic_result = await self._handle_semantic_cache_result(optimized_query, query)
                # Add pipeline metrics for semantic cache hit (phases 2 and 3 are 0, others may have time)
                if 'pipeline_metrics' not in semantic_result:
                    semantic_result['pipeline_metrics'] = {
                        'phase1_time': phase1_time,
                        'phase2_time': 0.0,
                        'phase3_time': 0.0,
                        'phase4_time': semantic_result.get('reranking_time', 0.0),
                        'phase5_time': 0.0,
                        'phase6_time': 0.0,
                        'total_time': phase1_time + semantic_result.get('reranking_time', 0.0)
                    }
                return semantic_result
            
            # === PHASE 2: PARALLEL QUERY GENERATION AND PREPARATION ===
            phase2_start = time.time()
            
            async def query_variations_task():
                """Generate all query variations in parallel."""
                return await self.generate_all_queries_in_one_call(query)
            
            async def retriever_validation_task():
                """Validate retriever in parallel."""
                return {
                    'retriever_available': retriever is not None
                }
            
            # Execute Phase 2 tasks in parallel
            phase2_results = await asyncio.gather(
                query_variations_task(),
                retriever_validation_task(),
                return_exceptions=True
            )
            
            # Safely extract results
            queries_result = phase2_results[0]
            retriever_status = phase2_results[1]
            
            if isinstance(queries_result, Exception):
                logger.error(f"Query generation failed: {queries_result}")
                raise queries_result
            
            if isinstance(retriever_status, Exception):
                logger.warning(f"Retriever validation failed: {retriever_status}")
                # Use fallback
                retriever_status = {
                    'retriever_available': retriever is not None
                }
            
            phase2_time = time.time() - phase2_start
            logger.debug(f"Phase 2 (query generation) completed in {phase2_time:.2f}s")
            
            # Extract query variations
            original_query = queries_result["original_query"]
            step_back_query = queries_result["step_back_query"]
            multi_queries = queries_result.get("multi_queries", [])
            
            logger.debug(f"Generated queries - Original: '{original_query[:30]}...', "
                        f"Step-back: '{step_back_query[:30]}...', Multi-queries: {len(multi_queries)}")
            
            # === PHASE 3: PARALLEL DOCUMENT RETRIEVAL ===
            phase3_start = time.time()
            
            retrieval_tasks = []
            task_descriptions = []
            
            # Add retrieval tasks if retriever is available
            if retriever_status.get('retriever_available', False):
                # Original query
                retrieval_tasks.append(
                    self.retrieve_context_without_reranking(original_query, retriever, chat_history)
                )
                task_descriptions.append("Original query")
                
                # Step-back query
                if step_back_query:
                    retrieval_tasks.append(
                        self.retrieve_context_without_reranking(step_back_query, retriever, chat_history)
                    )
                    task_descriptions.append("Step-back query")
                
                # Multi-queries
                for i, multi_query in enumerate(multi_queries[:3]):  # Limit to 3 for performance
                    retrieval_tasks.append(
                        self.retrieve_context_without_reranking(multi_query, retriever, chat_history)
                    )
                    task_descriptions.append(f"Multi-query {i+1}")
            
            if not retrieval_tasks:
                logger.warning("No retrieval tasks available - no valid retrievers")
                return {
                    'response': "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/",
                    'sources': [], 
                    'from_cache': False,
                    'processing_time': time.time() - pipeline_start_time
                }
            
            logger.info(f"Executing {len(retrieval_tasks)} retrieval tasks in parallel: {task_descriptions}")
            
            # Execute all retrieval tasks in parallel with timeout
            try:
                retrieval_results = await asyncio.wait_for(
                    asyncio.gather(*retrieval_tasks, return_exceptions=True),
                    timeout=settings.RETRIEVAL_TASK_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(f"Retrieval tasks timed out after {settings.RETRIEVAL_TASK_TIMEOUT} seconds")
                # Create fallback results with timeouts
                retrieval_results = [asyncio.TimeoutError("Retrieval timeout") for _ in retrieval_tasks]
            
            phase3_time = time.time() - phase3_start
            logger.debug(f"Phase 3 (retrieval) completed in {phase3_time:.2f}s with {len(retrieval_results)} results")
            
            # === PHASE 4: PARALLEL RESULT PROCESSING AND RERANKING ===
            phase4_start = time.time()
            
            async def document_consolidation_task():
                """Consolidate and deduplicate documents using improved result extraction."""
                all_retrieved_docs = []
                seen_contents = set()
                
                # Filter valid results from retrieval tasks
                valid_retrieval_results = _extract_valid_results(retrieval_results, task_descriptions)
                
                for result in valid_retrieval_results:
                    # Ensure result is iterable (list of documents)
                    try:
                        if not hasattr(result, '__iter__'):
                            logger.warning(f"Retrieval result is not iterable: {type(result)}")
                            continue
                            
                        if result:
                            for document in result:
                                if not isinstance(document, Document):
                                    continue
                                
                                # Ensure metadata exists
                                if not hasattr(document, 'metadata') or document.metadata is None:
                                    document.metadata = {}
                                
                                # Deduplicate by content hash
                                content_hash = hash(document.page_content)
                                if content_hash not in seen_contents:
                                    seen_contents.add(content_hash)
                                    all_retrieved_docs.append(document)
                                    
                    except Exception as e:
                        logger.error(f"Error processing retrieval result: {e}")
                        continue
                
                logger.info(f"Consolidated {len(all_retrieved_docs)} unique documents from {len(valid_retrieval_results)} valid retrieval results")
                return all_retrieved_docs
            
            async def reranking_preparation_task():
                """Prepare reranking model selection."""
                reranker_model = settings.COHERE_RERANKING_MODEL
                return reranker_model
            
            # Execute Phase 4 tasks in parallel
            phase4_results = await asyncio.gather(
                document_consolidation_task(),
                reranking_preparation_task(),
                return_exceptions=True
            )
            
            # Safely extract results
            consolidated_docs = phase4_results[0]
            reranker_model = phase4_results[1]
            
            if isinstance(consolidated_docs, Exception):
                logger.error(f"Document consolidation failed: {consolidated_docs}")
                raise consolidated_docs
            
            if isinstance(reranker_model, Exception):
                logger.warning(f"Reranker model preparation failed: {reranker_model}")
                # Use fallback
                reranker_model = settings.COHERE_RERANKING_MODEL
            
            if not consolidated_docs:
                logger.warning("No documents retrieved after consolidation")
                return {
                    'response': "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/",
                    'sources': [], 
                    'from_cache': False,
                    'processing_time': time.time() - pipeline_start_time
                }
            
            # Perform reranking using the appropriate query
            reranked_docs = await self.rerank_docs(query_for_reranking_and_llm, consolidated_docs, reranker_model)
            
            phase4_time = time.time() - phase4_start
            logger.debug(f"Phase 4 (processing/reranking) completed in {phase4_time:.2f}s")
            
            # === PHASE 5: PARALLEL RESPONSE GENERATION PREPARATION ===
            phase5_start = time.time()
            
            async def context_preparation_task():
                """Prepare context for LLM."""
                filtered_context = []
                sources = []
                
                # Sort documents by reranking score
                reranked_docs.sort(key=lambda x: x.metadata.get('reranking_score', 0), reverse=True)
                
                # Select top documents
                for document in reranked_docs[:settings.MAX_CHUNKS_LLM]:
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
                
                return filtered_context, sources
            
            def create_prompt_preparation_task(terms):
                """Create prompt preparation task with captured terms."""
                async def prompt_preparation_task():
                    """Prepare prompt template."""
                    # Use terms passed as parameter
                    current_matching_terms = terms if not isinstance(terms, Exception) else []
                    
                    if not current_matching_terms:
                        prompt_template = ChatPromptTemplate.from_template(
                            f"""
                            You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
                            Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
                            Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
                            Give always detailed answers including the essential and important points and that it does not exceed {settings.ANSWER_MAX_LENGTH} characters in length.
                            If the CONTEXT does not contain information to answer the QUESTION, do not try to answer the question with your knowledge, just say following:
                            "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/"

                            QUERY: ```{{question}}```

                            CONTEXT: ```{{context}}```
                            """
                        )
                        return prompt_template, None
                    else:
                        relevant_glossary = "\n".join([f"{term}: {explanation}"
                                                     for term, explanation in current_matching_terms])
                        prompt_template = ChatPromptTemplate.from_template(
                            f"""
                            You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
                            Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
                            Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
                            Give always detailed answers including the essential and important points and that it does not exceed {settings.ANSWER_MAX_LENGTH} characters in length.
                            If the CONTEXT does not contain information to answer the QUESTION, do not try to answer the question with your knowledge, just say following: 
                            "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/"

                            The following terms from the query have specific meanings:
                            {{glossary}}

                            Please consider these specific meanings when responding.

                            QUERY: ```{{question}}```

                            CONTEXT: ```{{context}}```
                            """
                        )
                        return prompt_template, relevant_glossary
                
                return prompt_preparation_task
            
            # Execute Phase 5 tasks in parallel
            phase5_results = await asyncio.gather(
                context_preparation_task(),
                create_prompt_preparation_task(matching_terms)(),
                return_exceptions=True
            )
            
            # Safely extract results with proper exception handling
            context_sources_result = phase5_results[0]
            prompt_result = phase5_results[1]
            
            # Check for exceptions in context preparation
            if isinstance(context_sources_result, Exception):
                logger.error(f"Context preparation failed: {context_sources_result}")
                raise context_sources_result
            
            # Check for exceptions in prompt preparation
            if isinstance(prompt_result, Exception):
                logger.error(f"Prompt preparation failed: {prompt_result}")
                # Use fallback for prompt preparation
                prompt_result = (ChatPromptTemplate.from_template(
                    f"""
                    You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
                    Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
                    Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
                    Give always detailed answers including the essential and important points and that it does not exceed {settings.ANSWER_MAX_LENGTH} characters in length.
                    If the CONTEXT does not contain information to answer the QUESTION, do not try to answer the question with your knowledge, just say following: 
                    "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/"

                    QUERY: ```{{question}}```

                    CONTEXT: ```{{context}}```
                    """), None)
            
            # Safely extract values
            try:
                filtered_context, sources = context_sources_result
                prompt_template, relevant_glossary = prompt_result
            except (ValueError, TypeError) as e:
                logger.error(f"Error unpacking Phase 5 results: {e}")
                raise RuntimeError(f"Failed to process Phase 5 results: {str(e)}")
            
            phase5_time = time.time() - phase5_start
            logger.debug(f"Phase 5 (response preparation) completed in {phase5_time:.2f}s")
            
            # === PHASE 6: LLM RESPONSE GENERATION ===
            phase6_start = time.time()
            
            # Create processing chain
            chain = prompt_template | self.llm_provider | StrOutputParser()
            
            # Generate response with timeout using the appropriate query
            try:
                if relevant_glossary:
                    response = await asyncio.wait_for(
                        chain.ainvoke({
                            "context": filtered_context,
                            "question": query_for_reranking_and_llm,
                            "glossary": relevant_glossary
                        }),
                        timeout=settings.LLM_GENERATION_TIMEOUT
                    )
                else:
                    response = await asyncio.wait_for(
                        chain.ainvoke({
                            "context": filtered_context,
                            "question": query_for_reranking_and_llm
                        }),
                        timeout=settings.LLM_GENERATION_TIMEOUT
                    )
            except asyncio.TimeoutError:
                logger.error(f"LLM generation timed out after {settings.LLM_GENERATION_TIMEOUT} seconds")
                response = "Leider konnte ich aufgrund einer Zeitüberschreitung keine Antwort generieren. Bitte versuchen Sie es mit einer einfacheren Anfrage erneut."
            
            phase6_time = time.time() - phase6_start
            logger.debug(f"Phase 6 (LLM generation) completed in {phase6_time:.2f}s")
            
            # === FINAL PROCESSING ===
            total_processing_time = time.time() - pipeline_start_time
            
            # Log detailed phase timings
            async_metadata_processor.log_async("INFO", 
                "Async pipeline phase timings",
                {
                    "total_time": total_processing_time,
                    "phase1_cache_optimization": phase1_time,
                    "phase2_query_generation": phase2_time,
                    "phase3_retrieval": phase3_time,
                    "phase4_processing_reranking": phase4_time,
                    "phase5_response_preparation": phase5_time,
                    "phase6_llm_generation": phase6_time,
                    "query": query[:50] + "..." if len(query) > 50 else query
                })
            
            # Register metrics
            self.metrics_manager.log_rag_query(
                query=query,
                processing_time=total_processing_time,
                num_sources=len(sources),
                from_cache=False
            )
            
            # Store in cache if valid documents
            has_relevant_docs = any(
                doc.metadata.get('reranking_score', 0) >= settings.MIN_RERANKING_SCORE 
                for doc in filtered_context if hasattr(doc, 'metadata') and doc.metadata
            )
            
            if has_relevant_docs:
                enhanced_sources_for_cache = []
                for doc in filtered_context:
                    if hasattr(doc, 'metadata') and doc.metadata and doc.metadata.get('reranking_score', 0) >= settings.MIN_RERANKING_SCORE:
                        enhanced_metadata = doc.metadata.copy()
                        enhanced_metadata['chunk_content'] = doc.page_content
                        enhanced_sources_for_cache.append(enhanced_metadata)
                
                self.query_optimizer._store_llm_response(query_for_reranking_and_llm, response, enhanced_sources_for_cache)
                logger.info(f"Response cached for query: '{query_for_reranking_and_llm[:50]}...' (relevant documents found)")
            
            logger.info(f"Async pipeline completed in {total_processing_time:.2f}s "
                       f"(phases: {phase1_time:.2f}+{phase2_time:.2f}+{phase3_time:.2f}+{phase4_time:.2f}+{phase5_time:.2f}+{phase6_time:.2f})")
            
            return {
                'response': response,
                'sources': sources,
                'from_cache': False,
                'processing_time': total_processing_time,
                'documents': filtered_context,
                'sources_metadata': sources_for_cache,
                'pipeline_metrics': {
                    'phase1_time': phase1_time,
                    'phase2_time': phase2_time, 
                    'phase3_time': phase3_time,
                    'phase4_time': phase4_time,
                    'phase5_time': phase5_time,
                    'phase6_time': phase6_time,
                    'total_time': total_processing_time
                }
            }
            
        except Exception as e:
            error_time = time.time() - pipeline_start_time
            logger.error(f"Error in async pipeline: {e}")
            
            # Register error metrics
            self.metrics_manager.log_error(
                error_type="async_pipeline_error",
                details=str(e),
                component="async_pipeline"
            )
            
            self.metrics_manager.log_rag_query(
                query=query,
                processing_time=error_time,
                num_sources=0,
                from_cache=False
            )
            
            return {
                'response': f"Es tut mir leid, bei der Bearbeitung Ihrer Anfrage ist ein Fehler aufgetreten. Bitte versuchen Sie es erneut.",
                'sources': [],
                'from_cache': False,
                'processing_time': error_time
            }
        finally:
            await coroutine_manager.cleanup()
    
    async def _handle_semantic_cache_result(self, optimized_query: Dict, query: str) -> Dict[str, Any]:
        """
        Handle semantic cache results with enhanced processing.
        
        Args:
            optimized_query: Optimized query result from cache
            query: Original query
            
        Returns:
            Processed cache result
        """
        match_info = optimized_query['result'].get('semantic_match', {})
        similarity = match_info.get('similarity', 0)
        logger.info(f"Semantic cache hit with similarity: {similarity:.4f}")
        
        # Register metrics
        self.metrics_manager.log_query_optimization(
            processing_time=0.0,
            was_cached=True,
            cache_type="semantic"
        )
        
        # Also register similarity score
        self.metrics_manager.metrics['query_similarity_scores'].append(similarity)
        
        # Get stored sources
        sources = optimized_query['result'].get('sources', [])
        original_response = optimized_query['result'].get('response', '')
        
        # Enhanced processing for semantic cache results
        if original_response and sources:
            logger.info(f"Found valid semantic cache with response (length: {len(original_response)}) and {len(sources)} sources")
            
            try:
                reranking_start = time.time()
                
                # Convert stored sources to documents for reranking
                cached_documents = []
                for source_metadata in sources:
                    if isinstance(source_metadata, dict):
                        chunk_content = source_metadata.get('chunk_content', source_metadata.get('source', ''))
                        cached_doc = Document(
                            page_content=chunk_content,
                            metadata=source_metadata
                        )
                        cached_documents.append(cached_doc)
                
                if cached_documents:
                    logger.info(f"Using {len(cached_documents)} cached chunks for reranking with new query")
                    
                    # Perform reranking with stored chunks and new query
                    reranker_model = settings.COHERE_RERANKING_MODEL
                    
                    reranked_docs = await self.rerank_docs(query, cached_documents, reranker_model)
                    
                    # Check for relevant documents
                    relevant_docs = [doc for doc in reranked_docs if doc.metadata.get('reranking_score', 0) >= settings.MIN_RERANKING_SCORE]
                    
                    if relevant_docs:
                        logger.info(f"Found {len(relevant_docs)} relevant chunks after reranking (score >= {settings.MIN_RERANKING_SCORE})")
                        
                        # Generate new response using reranked cached chunks
                        from app.utils.glossary import find_glossary_terms_with_explanation
                        matching_terms = find_glossary_terms_with_explanation(query)
                        
                        # Prepare context and sources
                        filtered_context = relevant_docs[:settings.MAX_CHUNKS_LLM]
                        response_sources = []
                        
                        for document in filtered_context:
                            source = {
                                'source': document.metadata.get('source', 'Unknown'),
                                'page_number': document.metadata.get('page_number', 'N/A'),
                                'file_type': document.metadata.get('file_type', 'Unknown'),
                                'sheet_name': document.metadata.get('sheet_name', ''),
                                'reranking_score': document.metadata.get('reranking_score', 0)
                            }
                            if source not in response_sources:
                                response_sources.append(source)
                        
                        # Generate new response
                        if not matching_terms:
                            prompt_template = ChatPromptTemplate.from_template(
                                f"""
                                You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
                                Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
                                Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
                                Give always detailed answers including the essential and important points and that it does not exceed {settings.ANSWER_MAX_LENGTH} characters in length.
                                If the CONTEXT does not contain information to answer the QUESTION, do not try to answer the question with your knowledge, just say following:
                                "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/"

                                QUERY: ```{{question}}```

                                CONTEXT: ```{{context}}```
                                """
                            )
                        else:
                            relevant_glossary = "\n".join([f"{term}: {explanation}"
                                                         for term, explanation in matching_terms])
                            prompt_template = ChatPromptTemplate.from_template(
                                f"""
                                You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
                                Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
                                Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
                                Give always detailed answers including the essential and important points and that it does not exceed {settings.ANSWER_MAX_LENGTH} characters in length.
                                If the CONTEXT does not contain information to answer the QUESTION, do not try to answer the question with your knowledge, just say following:
                                "Leider konnte ich in den verfügbaren Dokumenten keine relevanten Informationen zu Ihrer Frage finden.\n\nFür weitere Informationen siehe: https://intranet.uni-graz.at/"

                                The following terms from the query have specific meanings:
                                {{glossary}}

                                Please consider these specific meanings when responding.

                                QUERY: ```{{question}}```

                                CONTEXT: ```{{context}}```
                                """
                            )
                        
                        # Create processing chain and generate response
                        chain = prompt_template | self.llm_provider | StrOutputParser()
                        
                        if matching_terms:
                            new_response = await chain.ainvoke({
                                "context": filtered_context,
                                "question": query,
                                "glossary": relevant_glossary
                            })
                        else:
                            new_response = await chain.ainvoke({
                                "context": filtered_context,
                                "question": query
                            })
                        
                        # Store new response in cache
                        enhanced_sources_for_cache = []
                        for doc in filtered_context:
                            enhanced_metadata = doc.metadata.copy()
                            enhanced_metadata['chunk_content'] = doc.page_content
                            enhanced_sources_for_cache.append(enhanced_metadata)
                        
                        self.query_optimizer._store_llm_response(query, new_response, enhanced_sources_for_cache)
                        
                        reranking_time = time.time() - reranking_start
                        logger.info(f"Generated new response using {len(filtered_context)} reranked cached chunks from semantic match")
                        return {
                            'response': new_response,
                            'sources': response_sources,
                            'from_cache': False,
                            'semantic_match': match_info,
                            'processing_time': reranking_time,
                            'used_cached_chunks': True,
                            'reranking_time': reranking_time,
                            'documents': filtered_context
                        }
                
            except Exception as e:
                logger.error(f"Error processing cached chunks for semantic match: {e}")
        
        # Fallback to original cached response
        logger.info("Using original cached response from semantic match")
        cached_response = optimized_query['result']
        return {
            'response': cached_response.get('response', ''),
            'sources': cached_response.get('sources', []),
            'from_cache': True,
            'semantic_match': match_info,
            'processing_time': 0.0,
            'documents': []
        }
    
    async def initialize_retrievers_parallel(
        self,
        collection_name: str,
        top_k: int = None,
        max_concurrency: int = None
    ) -> Dict[str, Any]:
        """
        Initialize unified retriever for the specified collection.
        
        Args:
            collection_name: Collection name to use for retrieval
            top_k: Number of top documents to retrieve (defaults to settings)
            max_concurrency: Maximum concurrent operations (defaults to settings)
            
        Returns:
            Dictionary with initialized retriever and metadata
        """
        await self.ensure_initialized()
        
        # Use defaults from settings if not provided
        top_k = top_k or settings.MAX_CHUNKS_CONSIDERED
        max_concurrency = max_concurrency or settings.MAX_CONCURRENT_TASKS
        
        logger.info(f"Starting unified retriever initialization for collection: {collection_name}")
        
        initialization_metadata = {
            "total_tasks": 1,
            "successful_retrievers": 0,
            "failed_retrievers": 0,
            "initialization_time": 0,
            "collection_name": collection_name
        }
        
        start_time = time.time()
        retriever = None
        
        try:
            # Check if collection exists
            if utility.has_collection(collection_name):
                # Initialize unified retriever
                retriever = await self.get_retriever(
                    collection_name,
                    top_k=top_k,
                    max_concurrency=max_concurrency
                )
                
                if retriever:
                    initialization_metadata["successful_retrievers"] = 1
                    logger.info(f"Successfully initialized unified retriever for collection: {collection_name}")
                    
                    # Log async for detailed tracking
                    async_metadata_processor.log_async("INFO",
                        f"Unified retriever initialization successful",
                        {
                            "collection": collection_name,
                            "initialization_time": initialization_metadata["initialization_time"]
                        })
                else:
                    initialization_metadata["failed_retrievers"] = 1
                    logger.error(f"Failed to initialize retriever for collection: {collection_name}")
            else:
                initialization_metadata["failed_retrievers"] = 1
                logger.warning(f"Collection '{collection_name}' does not exist")
                
        except Exception as e:
            initialization_metadata["failed_retrievers"] = 1
            initialization_metadata["error"] = str(e)
            logger.error(f"Error initializing unified retriever for '{collection_name}': {e}")
            
            # Log async for detailed tracking
            async_metadata_processor.log_async("ERROR",
                f"Unified retriever initialization failed",
                {
                    "collection": collection_name,
                    "error": str(e)
                }, priority=3)
        
        initialization_metadata["initialization_time"] = time.time() - start_time
        logger.info(f"Unified retriever initialization completed in {initialization_metadata['initialization_time']:.2f}s")
        
        # Log final metrics
        async_metadata_processor.record_performance_async(
            "unified_retriever_initialization",
            initialization_metadata["initialization_time"],
            initialization_metadata["successful_retrievers"] > 0,
            {
                "successful_retrievers": initialization_metadata["successful_retrievers"],
                "failed_retrievers": initialization_metadata["failed_retrievers"],
                "collection_name": collection_name
            }
        )
        
        return {
            "retriever": retriever,
            "metadata": initialization_metadata
        }


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