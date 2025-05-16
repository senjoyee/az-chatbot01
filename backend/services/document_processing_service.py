import logging
from typing import List, Optional, Dict, Any

from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate # Ensure PromptTemplate is imported
from langchain.schema import StrOutputParser
from langchain_core.documents import Document

from config.settings import AZURE_OPENAI_API_KEY_SC, AZURE_OPENAI_ENDPOINT_SC
from config.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# Initialize LLM for summarization
llm_summarizer = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",  # Or your preferred model for summarization
    openai_api_version="2024-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT_SC,
    api_key=AZURE_OPENAI_API_KEY_SC,
    temperature=0.3
)

async def generate_single_document_summary(document_content: str, file_name_for_logging: str = "Unknown File") -> Dict[str, Any]:
    """
    Generates a summary for the provided document content.

    Args:
        document_content: The actual text content of the document to summarize.
        file_name_for_logging: The name of the file, used for logging purposes.

    Returns:
        A dictionary with 'summary' (str) or 'error' (str).
    """
    if not document_content or not document_content.strip():
        logger.warning(f"Document content for {file_name_for_logging} is empty or not provided.")
        return {"error": f"Document content for {file_name_for_logging} is empty."}

    # SUMMARY_PROMPT is already a PromptTemplate object from config.prompts
    summary_prompt_template = SUMMARY_PROMPT
    
    summary_chain = (
        summary_prompt_template
        | llm_summarizer 
        | StrOutputParser()
    )
    
    try:
        summary = await summary_chain.ainvoke({"context": document_content})
        logger.info(f"Successfully generated summary for document: {file_name_for_logging}")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error generating summary for document '{file_name_for_logging}': {e}", exc_info=True)
        return {"error": f"Failed to generate summary for document '{file_name_for_logging}'. An LLM or processing error occurred."}

