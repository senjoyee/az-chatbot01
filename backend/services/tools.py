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

    def _run(self, query: str, k: int = 10, filters: Optional[str] = None) -> List[Document]:
        """Run the tool."""
        try:
            logger.info(f"Executing retriever with query: '{query}', k: {k}, filters: '{filters}'")
            
            # Debug the filter format
            if filters:
                logger.info(f"Filter type: {type(filters)}, value: {filters}")
            
            # Execute the search with filters
            documents = vector_store.hybrid_search(
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


from langchain_community.tools import DuckDuckGoSearchResults

class DuckDuckGoSearchTool:
    """Web search using DuckDuckGo (no API key required)"""
    
    def __init__(self, max_results: int = 5):
        self.max_results = max(1, min(max_results, 10))
        self.tool = DuckDuckGoSearchResults(max_results=self.max_results)

    def run(self, query: str) -> List[Dict]:
        """Returns structured results (title, url, snippet)"""
        raw_results = self.tool.run(query)
        return self._parse_results(raw_results)

    def _parse_results(self, raw_results: str) -> List[Dict]:
        processed = []
        for result in raw_results.split('\n\n'):  # Split individual results
            parts = result.split('\n')
            if len(parts) >= 2:
                processed.append({
                    "title": parts[0].strip(),
                    "url": parts[1].strip(),
                    "snippet": parts[2].strip() if len(parts) > 2 else ""
                })
        return processed[:self.max_results]