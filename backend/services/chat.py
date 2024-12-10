"""
Chat service handling conversation logic and document retrieval.
Maintains the exact same conversation functionality as the original implementation.
"""

import logging
from typing import List, Optional, Dict, Any
from operator import itemgetter
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser, format_document
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT
)
from config.azure_search import vector_store
from models.schemas import Message, Conversation, ConversationRequest

# Get logger
logger = logging.getLogger(__name__)

# Initialize the language model (exactly as in original implementation)
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2023-03-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3
)

# Define prompt templates (exactly as in original implementation)
condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

answer_template = """Please provide an answer based strictly on the following context:
<context>
{context}
</context>

Guidelines for answering:
1. I will analyze the question and context thoroughly
2. Present information in a clear, structured format
3. Use numbered points for main ideas
4. Use a, b for sub-points instead of dashes 
5. Ensure all points are supported by the context
6. Clearly indicate if information is incomplete
7. Maintain consistency throughout the response

Question: {question}

Answer:
[If sufficient information exists in context]
1. [First main point]
    a. [Supporting detail]
    b. [Supporting detail]
2. [Second main point]
    a. [Supporting detail]
    etc.

[If information is incomplete or missing]
Based on the provided context, here is the relevant information that is available:
1. [Available information point]
    a. [Available detail]
2. [Available information point]
    a. [Available detail]

[If no relevant information exists]
The provided context does not contain information to answer this question."""

ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)

def _format_chat_history(chat_history: List[Message]) -> str:
    """Format chat history for the model (exactly as in original implementation)."""
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)

def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    """Combine multiple documents into a single string (exactly as in original implementation)."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    logger.info(f"Docstrings: {doc_strings}")
    return document_separator.join(doc_strings)

# Set up the conversation chain components (exactly as in original implementation)
_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": lambda x: _combine_documents(
        vector_store.hybrid_search(
            x["standalone_question"],
            k=10
        )
    ),
    "question": lambda x: x["standalone_question"],
}

# Create the final conversation chain
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()

async def get_conversation_response(question: str, chat_history: List[Message]) -> str:
    """
    Get a response from the conversation chain.
    This wraps the original implementation's functionality in a clean async interface.
    """
    # Debug logging
    logger.info(f"Using chat history: {chat_history}")
    
    return conversational_qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
