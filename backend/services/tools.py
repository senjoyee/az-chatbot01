from typing import List, Optional, Dict, Any
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
    name = "document_retriever"
    description = """Use this tool when you need to search for relevant documents.
    It performs a hybrid search combining keyword and semantic search.
    Input should be a natural language query."""
    args_schema = RetrieverInput

    def _run(self, tool_input: dict) -> List[Document]:
        """Run the tool."""
        try:
            documents = vector_store.hybrid_search(
                tool_input["query"],
                k=tool_input.get("k", 10),
                filter=tool_input.get("filter")
            )
            return documents
        except Exception as e:
            raise RuntimeError(f"Error retrieving documents: {str(e)}")