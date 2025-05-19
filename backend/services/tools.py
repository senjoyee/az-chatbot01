from typing import List, Optional, Dict, Any, ClassVar
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import logging

from config.azure_search import vector_store

logger = logging.getLogger(__name__)

class RetrieverInput(BaseModel):
    """Input for the retriever tool."""
    query: str = Field(..., description="The query to search for")
    k: int = Field(default=10, description="Number of documents to retrieve")
    filters: Optional[str] = Field(default=None, description="Optional filter expression to apply")

class RetrieverTool(BaseTool):
    """Tool for retrieving documents using hybrid search."""
    name: ClassVar[str] = "document_retriever"
    description: ClassVar[str] = """Use this tool when you need to search for relevant documents.
    It performs a hybrid search combining keyword and semantic search.
    Input should be a natural language query."""
    args_schema: ClassVar[type[BaseModel]] = RetrieverInput

    def _run(self, query: str, k: int = 40, filters: Optional[str] = None) -> List[Document]:
        """Run the tool."""
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
            documents = vector_store.semantic_hybrid_search(
                query,
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