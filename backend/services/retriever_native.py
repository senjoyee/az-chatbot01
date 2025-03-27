"""
Document retriever implementation using native Azure AI Search SDK.
This replaces the langchain-based RetrieverTool with direct SDK calls.
"""

import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from config.azure_search_native import azure_search_client
from config.logging_config import setup_logging

logger = setup_logging()

class DocumentRetrieverNative:
    """
    Document retriever using native Azure AI Search SDK.
    Provides methods for retrieving documents using semantic hybrid search.
    """
    
    def __init__(self):
        self.azure_search_client = azure_search_client
    
    def run(self, query_params: Dict[str, Any]) -> List[Document]:
        """
        Run the document retriever with the given parameters.
        
        Args:
            query_params: Dictionary containing query parameters:
                - query: The search query string
                - k: Number of documents to retrieve (default: 10)
                - filters: Optional filter expression to apply
                
        Returns:
            List[Document]: A list of Document objects containing the search results
        """
        query = query_params.get("query", "")
        k = query_params.get("k", 10)
        filters = query_params.get("filters", None)
        
        try:
            logger.info(f"Executing retriever with query: '{query}', k: {k}, filters: '{filters}'")
            
            # Debug the filter format
            if filters:
                logger.info(f"Filter type: {type(filters)}, value: {filters}")
                
                # Check if filter is too long and truncate if necessary
                if len(filters) > 20000:
                    logger.warning(f"Filter expression is too long ({len(filters)} chars). Azure Search may have issues with it.")
                    logger.warning("Consider reducing the number of selected files.")
            
            # Execute the search with filters using semantic hybrid search
            # This leverages Azure AI Search's semantic ranking capabilities
            documents = self.azure_search_client.semantic_hybrid_search(
                query=query,
                k=k,
                filters=filters
            )
            
            # Log the results
            logger.info(f"Retrieved {len(documents)} documents")
            for i, doc in enumerate(documents):
                source = doc.metadata.get('source', 'unknown')
                logger.info(f"Document {i+1}: source={source}")
            
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise RuntimeError(f"Error retrieving documents: {str(e)}")
