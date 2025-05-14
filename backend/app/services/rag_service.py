import os
import asyncio
import time
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

from app.core.config import settings
from app.core.embedding_manager import embedding_manager
from app.core.coroutine_manager import coroutine_manager
from app.core.metrics import measure_time, EMBEDDING_RETRIEVAL_DURATION
from app.core.cache import cache_result
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

            # Create multi-query retriever
            multi_query_retriever = await self.get_multi_query_retriever(parent_retriever, language)

            # Create HyDE retriever
            hyde_retriever = await self.get_hyde_retriever(
                embedding_model,
                collection_name,
                language,
                top_k
            )

            # Create BM25 retriever for keyword search
            bm25_retriever = self.get_bm25_retriever(collection_name, top_k)

            # Setup retrievers and weights
            retrievers = [base_retriever, parent_retriever, multi_query_retriever]
            weights = [0.15, 0.15, 0.4]

            # Add HyDE retriever if available
            if hyde_retriever:
                retrievers.append(hyde_retriever)
                weights.append(0.15)
                logger.info(f"Added HyDE retriever to ensemble for {collection_name}")
            else:
                # Redistribute weights if HyDE is not available
                weights = [0.2, 0.2, 0.45]
                logger.warning(f"HyDE retriever not available for {collection_name}")

            # Add BM25 retriever if available
            if bm25_retriever:
                retrievers.append(bm25_retriever)
                # Adjust weights based on which retrievers are available
                if hyde_retriever:
                    weights.append(0.15)  # Final weights: 0.15, 0.15, 0.4, 0.15, 0.15
                else:
                    weights.append(0.15)  # Final weights: 0.2, 0.2, 0.45, 0.15
                logger.info(f"Added BM25 retriever to ensemble for {collection_name}")
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
        query: str,
        language: str = "german"
    ) -> str:
        """
        Generate a more generic step-back query from the original query.

        A step-back query generalizes the question to help retrieve broader context
        that might be helpful to answer specific questions.

        Args:
            query: Original user query
            language: Language of the query and response

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
        matching_terms = find_glossary_terms_with_explanation(query, language)

        if not matching_terms:
            # Standard prompt if no glossary terms
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    """You are an expert at world knowledge. Your task is to step back and paraphrase
                    a question to a more generic step-back question, which is easier to answer.
                    Please note that the question has been asked in the context of the University of Graz.
                    Give the generic step-back question in {language}. Here are a few examples:""",
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
                    Give the generic step-back question in {language}. Here are a few examples:""",
                ),
                few_shot_prompt,
                ("user", "{question}"),
            ])

        # Create and execute chain
        chain = prompt | self.llm_provider | StrOutputParser()
        step_back_query = await chain.ainvoke({"language": language, "question": query})

        if settings.SHOW_INTERNAL_MESSAGES:
            logger.debug(f"Step-back query generation:\nOriginal: {query}\n"
                       f"Step-back: {step_back_query}")

        return step_back_query

    @coroutine_manager.coroutine_handler(timeout=30, task_type="multi_query")
    async def generate_all_queries_in_one_call(
        self,
        query: str,
        language: str = "german"
    ) -> Dict[str, str]:
        """
        Generate all necessary queries in a single LLM call for efficiency.

        This method generates:
        - The original query
        - A translated version of the query (to the other language)
        - A step-back version of the original query
        - A step-back version of the translated query

        Args:
            query: Original user query
            language: Language of the original query

        Returns:
            Dictionary containing all generated queries
        """
        # Determine source and target languages
        source_lang = "German" if language.lower() == "german" else "English"
        target_lang = "English" if language.lower() == "german" else "German"
        source_lang_lower = source_lang.lower()
        target_lang_lower = target_lang.lower()

        # Get glossary terms
        matching_terms = find_glossary_terms_with_explanation(query, language)

        if not matching_terms:
            # Standard prompt if no glossary terms found
            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are a multilingual language expert specialized in German and English.
                    I will give you a question in {source_lang}. Perform the following steps in one go:

                    1. If the question is in {source_lang}, translate it to {target_lang} accurately.
                    2. Create a more generic "step-back" version of the original {source_lang} question.
                    3. Create a more generic "step-back" version of the translated {target_lang} question.

                    A "step-back" question is more generic and broader than the original question, making it easier to answer.
                    For example:
                    - Original: "Could the members of The Police perform lawful arrests?"
                    - Step-back: "What can the members of The Police do?"

                    Respond in JSON format with these exact keys:
                    {{{{
                        "original_{source_lang_lower}": "The original question",
                        "translated_{target_lang_lower}": "The translated question",
                        "step_back_{source_lang_lower}": "The step-back version of the original question",
                        "step_back_{target_lang_lower}": "The step-back version of the translated question"
                    }}}}
                    """
                ),
                ("human", "{question}")
            ])
        else:
            # Include glossary terms in prompt
            relevant_glossary = "\n".join([f"{term}: {explanation}"
                                         for term, explanation in matching_terms])

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    f"""You are a multilingual language expert specialized in German and English.
                    I will give you a question in {source_lang}. Perform the following steps in one go:

                    1. If the question is in {source_lang}, translate it to {target_lang} accurately.
                    2. Create a more generic "step-back" version of the original {source_lang} question.
                    3. Create a more generic "step-back" version of the translated {target_lang} question.

                    The following terms from the question have specific meanings in the context of the University of Graz:
                    {relevant_glossary}

                    IMPORTANT: When translating, preserve these specific terms exactly as they appear. Do not translate them.
                    Consider these specific meanings when generating step-back questions.

                    A "step-back" question is more generic and broader than the original question, making it easier to answer.
                    For example:
                    - Original: "Could the members of The Police perform lawful arrests?"
                    - Step-back: "What can the members of The Police do?"

                    Respond in JSON format with these exact keys:
                    {{{{
                        "original_{source_lang_lower}": "The original question",
                        "translated_{target_lang_lower}": "The translated question",
                        "step_back_{source_lang_lower}": "The step-back version of the original question",
                        "step_back_{target_lang_lower}": "The step-back version of the translated question"
                    }}}}
                    """
                ),
                ("human", "{question}")
            ])

        # Define Pydantic model for JSON validation
        class QueryOutput(BaseModel):
            original_german: Optional[str] = Field(default=None)
            original_english: Optional[str] = Field(default=None)
            translated_english: Optional[str] = Field(default=None)
            translated_german: Optional[str] = Field(default=None)
            step_back_german: str = Field(...)
            step_back_english: str = Field(...)

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

            # Format the result based on original language using dictionary access
            if language.lower() == "german":
                return {
                    "query_de": result_dict.get("original_german", query),
                    "query_en": result_dict.get("translated_english", ""),
                    "step_back_query_de": result_dict.get("step_back_german", ""),
                    "step_back_query_en": result_dict.get("step_back_english", "")
                }
            else:
                return {
                    "query_en": result_dict.get("original_english", query),
                    "query_de": result_dict.get("translated_german", ""),
                    "step_back_query_en": result_dict.get("step_back_english", ""),
                    "step_back_query_de": result_dict.get("step_back_german", "")
                }

        except Exception as e:
            logger.error(f"Error generating multiple queries: {e}")
            # Fallback to individual methods if combined approach fails
            if language.lower() == "german":
                translated = await self.translate_query(query, "German", "English")
                step_back_de = await self.generate_step_back_query(query, "german")
                step_back_en = await self.generate_step_back_query(translated, "english")
                return {
                    "query_de": query,
                    "query_en": translated,
                    "step_back_query_de": step_back_de,
                    "step_back_query_en": step_back_en
                }
            else:
                translated = await self.translate_query(query, "English", "German")
                step_back_en = await self.generate_step_back_query(query, "english")
                step_back_de = await self.generate_step_back_query(translated, "german")
                return {
                    "query_en": query,
                    "query_de": translated,
                    "step_back_query_en": step_back_en,
                    "step_back_query_de": step_back_de
                }

    async def get_hyde_retriever(
        self,
        embedding_model: Any,
        collection_name: str,
        language: str = "german",
        top_k: int = 3
    ) -> Any:
        """
        Create a HyDE retriever with glossary-aware document generation.

        The Hypothetical Document Embedder (HyDE) generates a synthetic document that could
        hypothetically answer the query, then embeds that document to find similar real documents.

        Args:
            embedding_model: Base embedding model to use
            collection_name: Name of the collection to search in
            language: Language for document generation
            top_k: Number of documents to retrieve

        Returns:
            A configured retriever using HyDE embeddings
        """
        def create_hyde_chain(query: str):
            # Check if query contains glossary terms
            matching_terms = find_glossary_terms_with_explanation(query, language)

            if not matching_terms:
                # Basic prompt for document generation
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"Please write a passage in {language} to answer the question."),
                    ("human", "{question}")
                ])
            else:
                # Include glossary terms in the prompt
                relevant_glossary = "\n".join([f"{term}: {explanation}"
                                          for term, explanation in matching_terms])

                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""Please write a passage in {language} to answer the question.
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
        language: str = "german",
    ) -> MultiQueryRetriever:
        """
        Create a glossary-aware multi-query retriever.

        This retriever generates multiple variations of the user query by prompting the LLM,
        taking into account specialized glossary terms if present in the query.

        Args:
            base_retriever: The base retriever to use for document retrieval
            language: The language to use for generating query variations

        Returns:
            A configured MultiQueryRetriever
        """
        output_parser = LineListOutputParser()

        def create_multi_query_chain(query: str):
            # Check if query contains glossary terms
            matching_terms = find_glossary_terms_with_explanation(query, language)

            if not matching_terms:
                # Standard prompt if no glossary terms found
                prompt = ChatPromptTemplate.from_messages([
                    ("system", f"""You are an AI language model assistant. Your task is to generate
                    five different versions of the given user question in {language} to retrieve
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
                    five different versions of the given user question in {language} to retrieve
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
        # Check if query contains glossary terms
        matching_terms = find_glossary_terms(query, source_language)

        if not matching_terms:
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Translate the following text from {source_language} to {target_language}. "
                "Only provide the translation, no explanations."),
                ("human", "{query}")
            ])
        else:
            # If glossary terms were found, instruct not to translate them
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"Translate the following text from {source_language} to {target_language}. "
                f"Only provide the translation, no explanations. If these terms {matching_terms} appear "
                f"in the text to be translated, do not translate them but use them as they are written."),
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

        This method handles various query enhancement techniques:
        - Multi-language processing (German and English)
        - Step-back queries for broader context retrieval
        - Glossary-aware query processing

        Args:
            query: User query
            retriever_de: German retriever (can be None if collection doesn't exist)
            retriever_en: English retriever (can be None if collection doesn't exist)
            chat_history: Chat history
            language: Primary language for the response

        Returns:
            Dictionary with response and metadata
        """
        try:
            start_time = time.time()
            sources_for_cache = []

            # Generate all query variations in one LLM call (original, translated, step-back in both languages)
            logger.info(f"Generating query variations for: '{query}'")
            queries = await self.generate_all_queries_in_one_call(query, language)

            # Extract all query variations
            query_de = queries["query_de"]
            query_en = queries["query_en"]
            step_back_query_de = queries["step_back_query_de"]
            step_back_query_en = queries["step_back_query_en"]

            logger.debug(f"Generated queries: Original DE: '{query_de}', Original EN: '{query_en}', "
                       f"Step-back DE: '{step_back_query_de}', Step-back EN: '{step_back_query_en}'")

            # Retrieve and rerank in parallel, but only for retrievers that exist
            retrieval_tasks = []

            # Add German retriever tasks if it exists
            if retriever_de is not None:
                logger.info("Adding German retriever tasks (original + step-back)")
                # Original German query
                retrieval_tasks.append(
                    self.retrieve_context_reranked(
                        query_de,
                        retriever_de,
                        settings.GERMAN_COHERE_RERANKING_MODEL,
                        chat_history,
                        "german"
                    )
                )
                # Step-back German query
                if step_back_query_de:
                    retrieval_tasks.append(
                        self.retrieve_context_reranked(
                            step_back_query_de,
                            retriever_de,
                            settings.GERMAN_COHERE_RERANKING_MODEL,
                            chat_history,
                            "german"
                        )
                    )
            else:
                logger.warning("German retriever is None, skipping German retrieval")

            # Add English retriever tasks if it exists
            if retriever_en is not None:
                logger.info("Adding English retriever tasks (original + step-back)")
                # Original English query
                retrieval_tasks.append(
                    self.retrieve_context_reranked(
                        query_en,
                        retriever_en,
                        settings.ENGLISH_COHERE_RERANKING_MODEL,
                        chat_history,
                        "english"
                    )
                )
                # Step-back English query
                if step_back_query_en:
                    retrieval_tasks.append(
                        self.retrieve_context_reranked(
                            step_back_query_en,
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
            
            # Check if query contains glossary terms
            matching_terms = find_glossary_terms_with_explanation(query, language)

            # Create prompt template
            if not matching_terms:
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
            else:
                # Include glossary terms and their explanations in the prompt
                relevant_glossary = "\n".join([f"{term}: {explanation}"
                                           for term, explanation in matching_terms])

                prompt_template = ChatPromptTemplate.from_template(
                    """
                    You are an experienced virtual assistant at the University of Graz and know all the information about the University of Graz.
                    Your main task is to extract information from the provided CONTEXT based on the user's QUERY.
                    Think step by step and only use the information from the CONTEXT that is relevant to the user's QUERY.
                    If the CONTEXT does not contain information to answer the QUESTION, try to answer the question with your knowledge, but only if the answer is appropriate.

                    The following terms from the query have specific meanings:
                    {glossary}

                    Please consider these specific meanings when responding. Give detailed answers in {language}.

                    QUERY: ```{question}```

                    CONTEXT: ```{context}```
                    """
                )
            
            # Create processing chain
            chain = prompt_template | self.llm_provider | StrOutputParser()

            # Generate response
            if matching_terms:
                response = await chain.ainvoke({
                    "context": filtered_context,
                    "language": language,
                    "question": query,
                    "glossary": relevant_glossary
                })
            else:
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
        Rerank documents using Cohere through Azure endpoint.
        
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

            # Use Azure Cohere endpoint for reranking
            return await self._rerank_with_azure_cohere(query, retrieved_docs, model)

        
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
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
            
            return reranked_documents
            
        except Exception as e:
            logger.error(f"Exception during Azure Cohere reranking: {str(e)}")
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