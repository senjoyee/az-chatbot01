from typing import List, Optional, Dict, Any, ClassVar
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config.azure_search import vector_store

class RetrieverInput(BaseModel):
    """Input for the retriever tool."""
    query: str = Field(..., description="The query to search for")
    k: int = Field(default=10, description="Number of documents to retrieve")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters to apply")

class RetrieverTool(BaseTool):
    """Tool for retrieving documents using hybrid search."""
    name: ClassVar[str] = "document_retriever"
    description: ClassVar[str] = """Use this tool when you need to search for relevant documents.
    It performs a hybrid search combining keyword and semantic search.
    Input should be a natural language query."""
    args_schema: ClassVar[type[BaseModel]] = RetrieverInput

    def _run(self, query: str, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Run the tool."""
        try:
            documents = vector_store.hybrid_search(
                query,
                k=k,
                filters=filters
            )
            return documents
        except Exception as e:
            raise RuntimeError(f"Error retrieving documents: {str(e)}")