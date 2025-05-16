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
        raw_summary = await summary_chain.ainvoke({"context": document_content})

        # Strip <answer> tags
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_idx = raw_summary.find(start_tag)
        end_idx = raw_summary.rfind(end_tag) # Use rfind for the last occurrence of end_tag

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            # Adjust start_idx to be after the tag itself
            clean_summary = raw_summary[start_idx + len(start_tag):end_idx].strip()
        else:
            # If tags are not found as expected, return the raw summary but log a warning
            logger.warning(f"<answer> tags not found in expected format for document '{file_name_for_logging}'. Returning raw summary.")
            clean_summary = raw_summary.strip()

        logger.info(f"Successfully generated summary for document: {file_name_for_logging}")
        return {"summary": clean_summary}
    except Exception as e:
        logger.error(f"Error generating summary for document '{file_name_for_logging}': {e}", exc_info=True)
        return {"error": f"Failed to generate summary for document '{file_name_for_logging}'. An LLM or processing error occurred."}

