import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT
)
from models.grader_models import GradeDocuments

logger = logging.getLogger(__name__)

# Initialize the LLM for grading
grader_llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",  # Adjust deployment as necessary
    openai_api_version="2024-08-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0  # Lower temperature for consistency
)

# Prompt for grading
grader_system_prompt = """You are a grader assessing the relevance of a retrieved document to a user question. \
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# Structured output parser
structured_grader = grader_llm.with_structured_output(GradeDocuments)

# Define the retrieval grader runnable
retrieval_grader = grade_prompt | structured_grader

def grade_document(question: str, document: str) -> str:
    """Grade a single document for relevance."""
    try:
        grading_result = retrieval_grader.invoke({
            "question": question,
            "document": document
        })
        return grading_result.binary_score.lower()
    except Exception as e:
        logger.error(f"Error grading document: {e}")
        return "no"  # Default to 'no' in case of error