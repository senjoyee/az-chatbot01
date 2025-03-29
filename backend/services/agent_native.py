"""
Agent implementation using native Azure AI Search SDK.
This replaces the langchain-based agent with direct SDK calls.
"""

import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.documents import Document

from .retriever_native import DocumentRetrieverNative
from utils.helpers import is_casual_conversation

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY_SC,
    AZURE_OPENAI_ENDPOINT_SC,
)

from config.prompts import (
    CONDENSE_QUESTION_PROMPT,
    ANSWER_PROMPT,
    CONVERSATION_PROMPT,
    DECISION_PROMPT,
    SUMMARY_PROMPT
)

from models.schemas import Message, AgentState

logger = logging.getLogger(__name__)

# Initialize the native retriever
retriever_native = DocumentRetrieverNative()

# Initialize the language models (same as original implementation)
llm_4o_mini = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2023-03-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.0,
)

llm_o3_mini = AzureChatOpenAI(
    azure_deployment="o3-mini",
    openai_api_version="2024-12-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT_SC,    
    api_key=AZURE_OPENAI_API_KEY_SC,
    reasoning_effort="low"
)

def format_chat_history(chat_history: List[Message]) -> str:
    """Format chat history for the LLM."""
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)

# Define customer names for detection
CUSTOMER_NAMES = [
    "bsw",
    "tbs",
    "npac",
    "asahi"
    # Add more customer names here
]

def detect_customers(query: str) -> List[str]:
    """
    Detect customer names in the query string.
    Returns a list of detected customer names (case-insensitive).
    """
    query_lower = query.lower()
    return [name for name in CUSTOMER_NAMES if name.lower() in query_lower]

def is_summary_request(query: str) -> bool:
    """
    Detect if the user is asking for a document summary.
    Returns True if the query appears to be requesting a summary.
    """
    query_lower = query.lower()
    summary_keywords = [
        "summarize", "summary", "summarize this document", 
        "give me a summary", "can you summarize", 
        "provide a summary", "overview of this document",
        "key points", "main points", "tldr", "tl;dr"
    ]
    
    for keyword in summary_keywords:
        if keyword in query_lower:
            return True
    
    return False

def condense_question(state: AgentState) -> AgentState:
    """Condense the question based on chat history."""
    logger.info(f"Condensing question with state: {state}")
    if not state.chat_history:  # No history to condense from
        return state

    _input = (
        RunnableLambda(lambda x: {
            "chat_history": format_chat_history(x.chat_history),
            "question": x.question
        })
        | CONDENSE_QUESTION_PROMPT
        | llm_4o_mini
        | StrOutputParser()
    )
    result = _input.invoke(state)
    state.question = result
    return state

def check_customer_specification(state: AgentState) -> AgentState:
    """
    Inspects the (condensed) question for customer names.
    - If a known customer is mentioned, state.customers will be set accordingly.
    - Otherwise, proceed without customer filter.
    """
    detected = detect_customers(state.question)
    if detected:
        logger.info(f"Detected customers: {detected}")
        state.customers = detected
    else:
        logger.info("No customer specified; proceeding without customer filter.")
        state.customers = []
    return state

def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve documents using the native retriever."""
    logger.info(f"Retrieving documents for question: {state.question}")
    try:
        # Build filters based on available criteria
        filter_expression = None
        filter_parts = []
        
        # Add customer filter if customers are detected in the query
        if state.customers and len(state.customers) > 0:
            logger.info(f"Detected customers: {state.customers}")
            customer_filters = " or ".join([f"customer eq '{customer}'" for customer in state.customers])
            filter_parts.append(f"({customer_filters})")
            logger.info(f"Adding customer filter: {customer_filters}")
            
        # Add file filter if files are selected
        if state.selected_files and len(state.selected_files) > 0:
            # Check if this is effectively a "select all" scenario
            # We'll use a threshold to determine if the user has selected a large number of files
            # which would be equivalent to searching the entire database
            SELECT_ALL_THRESHOLD = 100  # If more than this many files are selected, treat as "select all"
            MAX_FILES_IN_FILTER = 50    # Allow up to 50 files in filter (up from 15)
            
            if len(state.selected_files) > SELECT_ALL_THRESHOLD:
                logger.info(f"Large number of files selected ({len(state.selected_files)}). Treating as 'select all' - no file filter will be applied.")
                # Don't add file filter - but customer filter may still be applied if detected
            else:
                selected_files = state.selected_files
                
                if len(selected_files) > MAX_FILES_IN_FILTER:
                    logger.warning(f"Too many files selected ({len(selected_files)}). Limiting to {MAX_FILES_IN_FILTER} files.")
                    selected_files = selected_files[:MAX_FILES_IN_FILTER]
                
                # Escape single quotes by doubling them
                file_filters = " or ".join([f"source eq '{file.replace(chr(39), chr(39)*2)}'" for file in selected_files])
                filter_parts.append(f"({file_filters})")
                logger.info(f"Adding file filter for {len(selected_files)} files")
            
        # Combine filters if present
        if filter_parts:
            filter_expression = " and ".join(filter_parts)
            logger.info(f"Combined filter expression: {filter_expression}")
            
            # Log the length of the filter expression to check if it's too long
            logger.info(f"Filter expression length: {len(filter_expression)}")
            
            # If filter expression is very long, log a warning
            if len(filter_expression) > 10000:
                logger.warning("Filter expression is very long, which might cause issues with Azure Search")
        else:
            logger.info("No filters applied - searching across all documents")
        
        try:
            # Use the native retriever instead of the langchain RetrieverTool
            state.documents = retriever_native.run({
                "query": state.question,
                "k": 15,
                "filters": filter_expression
            })
            
            logger.info(f"Retrieved {len(state.documents)} documents")
            
            # Log the first few document sources for debugging
            if state.documents:
                sources = [doc.metadata.get('source', 'unknown') for doc in state.documents[:5]]
                logger.info(f"First few document sources: {sources}")
            
            if not state.documents:
                state.response = "I could not find any relevant documents in the database."
                state.should_stop = True
                return state
        except Exception as e:
            logger.error(f"Error calling retriever: {str(e)}")
            state.documents = []
            state.response = "An error occurred while retrieving documents. For best results, either select up to 50 specific files or use 'Select All' to search across the entire database."
            state.should_stop = True
            return state
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        state.documents = []
        state.response = "An error occurred while retrieving documents."
        state.should_stop = True
        return state
    return state

def rerank_documents(state: AgentState) -> AgentState:
    """
    This function is kept for compatibility, but since we're using
    Azure AI Search's semantic ranking, we don't need to rerank locally.
    """
    logger.info("Reranking documents")
    documents = state.documents
    if not documents:
        logger.info("No documents to rerank, skipping reranking step")
        return state
    
    # The documents are already ranked by Azure AI Search's semantic ranking
    # We can skip the local reranking step
    logger.info("Documents already ranked by Azure AI Search semantic ranking")
    logger.info(f"Using {len(documents)} documents")
    
    return state

def generate_response(state: AgentState) -> AgentState:
    """Generate a response based on the retrieved documents."""
    logger.info("Generating response")
    
    if state.should_stop:
        logger.info(f"Stopping early with response: {state.response}")
        return state
    
    if not state.documents:
        state.response = "I couldn't find any relevant information to answer your question."
        return state
    
    # Format the documents for the prompt
    formatted_docs = []
    for i, doc in enumerate(state.documents[:10]):  # Limit to top 10 docs
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.strip()
        formatted_docs.append(f"Document {i+1} (Source: {source}):\n{content}\n")
    
    documents_str = "\n".join(formatted_docs)
    
    # Generate the answer
    _input = (
        RunnableLambda(lambda x: {
            "question": x.question,
            "documents": documents_str
        })
        | ANSWER_PROMPT
        | llm_4o_mini
        | StrOutputParser()
    )
    
    try:
        result = _input.invoke(state)
        state.response = result
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        state.response = "I encountered an error while generating a response. Please try again or rephrase your question."
    
    return state

def update_history(state: AgentState) -> AgentState:
    """Update the chat history with the new exchange."""
    logger.info("Updating chat history")
    
    # Add the user's question to the history
    state.chat_history.append(
        Message(role="user", content=state.question)
    )
    
    # Add the assistant's response to the history
    state.chat_history.append(
        Message(role="assistant", content=state.response)
    )
    
    return state

def detect_summary_intent(state: AgentState) -> AgentState:
    """
    Detect if the user is requesting a document summary.
    This is placed early in the workflow to enable proper routing.
    """
    state.is_summary_request = is_summary_request(state.question)
    if state.is_summary_request:
        logger.info("Summary request detected")
    return state

def detect_casual_talk(state: AgentState) -> AgentState:
    """
    Detect if the question is casual conversation.
    """
    state.needs_casual_response = is_casual_conversation(state.question)
    return state

def respond_to_casual(state: AgentState) -> AgentState:
    """Generate a response for casual conversation."""
    logger.info("Generating response for casual conversation")
    
    _input = (
        RunnableLambda(lambda x: {
            "chat_history": format_chat_history(x.chat_history),
            "question": x.question
        })
        | CONVERSATION_PROMPT
        | llm_o3_mini
        | StrOutputParser()
    )
    
    result = _input.invoke(state)
    state.response = result
    return state

def retrieve_documents_for_summary(state: AgentState) -> AgentState:
    """
    Retrieve documents specifically for summarization.
    This function is optimized for comprehensive coverage rather than precision.
    """
    logger.info(f"Retrieving documents for summarization: {state.question}")
    
    try:
        # Check if files are selected
        if not state.selected_files or len(state.selected_files) == 0:
            state.response = "Please select at least one file to summarize."
            state.should_stop = True
            return state
        
        # Check if too many files are selected
        if len(state.selected_files) > 1:
            state.response = "For summarization, please select only one document at a time. This helps ensure a focused and coherent summary."
            state.should_stop = True
            return state
        
        # Get the selected file
        file_to_summarize = state.selected_files[0]
        logger.info(f"Summarizing file: {file_to_summarize}")
        
        # Escape single quotes by doubling them
        escaped_file = file_to_summarize.replace("'", "''")
        filter_expression = f"source eq '{escaped_file}'"
        
        try:
            # Use the native retriever with a higher k value to get more chunks
            # from the document for a comprehensive summary
            state.documents = retriever_native.run({
                "query": "document summary overview main points",  # Generic query to retrieve document chunks
                "k": 30,  # Retrieve more chunks for comprehensive coverage
                "filters": filter_expression
            })
            
            logger.info(f"Retrieved {len(state.documents)} document chunks for summarization")
            
            if not state.documents:
                state.response = f"I could not find any content in the selected document: {file_to_summarize}"
                state.should_stop = True
                return state
                
        except Exception as e:
            logger.error(f"Error retrieving document chunks for summarization: {str(e)}")
            state.documents = []
            state.response = f"An error occurred while retrieving content from the selected document: {file_to_summarize}"
            state.should_stop = True
            return state
            
    except Exception as e:
        logger.error(f"Error in document retrieval for summarization: {str(e)}")
        state.documents = []
        state.response = "An error occurred while preparing to summarize the document."
        state.should_stop = True
        return state
        
    return state

def process_documents_for_summary(state: AgentState) -> AgentState:
    """
    Process and organize documents specifically for summarization.
    This function groups documents by source and prepares them for effective summarization.
    """
    logger.info("Processing documents for summarization")
    
    if state.should_stop or not state.documents:
        return state
    
    try:
        # Sort documents by their position in the original document
        # This assumes document chunks have metadata with position information
        # If not available, we'll use the order they were retrieved
        
        # First, try to extract page numbers or positions if available
        for doc in state.documents:
            if "page" in doc.metadata:
                doc.metadata["position"] = int(doc.metadata["page"])
            elif "position" in doc.metadata:
                doc.metadata["position"] = int(doc.metadata["position"])
            else:
                # If no position info, use a default
                doc.metadata["position"] = 0
        
        # Sort by position
        sorted_docs = sorted(state.documents, key=lambda x: x.metadata.get("position", 0))
        
        # Combine the content from all chunks
        combined_content = "\n\n".join([doc.page_content for doc in sorted_docs])
        
        # Store the combined content and document info for summarization
        state.document_content = combined_content
        state.document_name = state.selected_files[0]
        
        logger.info(f"Prepared document for summarization: {state.document_name}")
        logger.info(f"Combined content length: {len(combined_content)} characters")
        
    except Exception as e:
        logger.error(f"Error processing documents for summarization: {str(e)}")
        state.response = "An error occurred while processing the document for summarization."
        state.should_stop = True
        
    return state

def generate_summary(state: AgentState) -> AgentState:
    """
    Generate a comprehensive summary from the processed documents.
    This function is specifically designed for creating structured, informative summaries.
    """
    logger.info("Generating document summary")
    
    if state.should_stop or not hasattr(state, "document_content") or not state.document_content:
        return state
    
    try:
        # Prepare the input for the summary prompt
        _input = (
            RunnableLambda(lambda x: {
                "document_name": x.document_name,
                "document_content": x.document_content
            })
            | SUMMARY_PROMPT
            | llm_4o_mini
            | StrOutputParser()
        )
        
        # Generate the summary
        result = _input.invoke(state)
        
        # Format the final response with a header
        state.response = f"# Summary of {state.document_name}\n\n{result}"
        
        logger.info(f"Generated summary for {state.document_name}")
        
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        state.response = "I encountered an error while generating the document summary. The document might be too large or complex to summarize effectively."
    
    return state

# Build the Langgraph following the original agent.py pattern
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("detect_summary", detect_summary_intent)
builder.add_node("detect_casual", detect_casual_talk)
builder.add_node("respond_casual", respond_to_casual)
builder.add_node("condense", condense_question)
builder.add_node("check_customer", check_customer_specification)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("rerank", rerank_documents)
builder.add_node("generate", generate_response)
builder.add_node("update_history", update_history)
builder.add_node("retrieve_for_summary", retrieve_documents_for_summary)
builder.add_node("process_for_summary", process_documents_for_summary)
builder.add_node("generate_summary", generate_summary)

# Add edge from START to first node
builder.add_edge(START, "detect_summary")

# Follow the original working pattern for conditional edges
builder.add_conditional_edges(
    "detect_summary",
    lambda s: "detect_casual" if not s.is_summary_request else "condense"
)
builder.add_conditional_edges(
    "detect_casual",
    lambda s: "respond_casual" if s.needs_casual_response else "condense"
)
builder.add_edge("respond_casual", "update_history")
builder.add_edge("condense", "check_customer")
builder.add_conditional_edges(
    "check_customer",
    lambda s: "update_history" if s.should_stop else (
        "retrieve_for_summary" if s.is_summary_request else "retrieve"
    )
)
builder.add_conditional_edges(
    "retrieve",
    lambda s: "update_history" if s.should_stop else "rerank"
)
builder.add_edge("rerank", "generate")
builder.add_edge("generate", "update_history")
builder.add_edge("retrieve_for_summary", "process_for_summary")
builder.add_edge("process_for_summary", "generate_summary")
builder.add_edge("generate_summary", "update_history")
builder.add_edge("update_history", END)

# Compile the graph
graph = builder.compile()

async def run_agent_native(question: str, chat_history: List[Message], selected_files: List[str] = None):
    """
    Run the agent with the given question and chat history.
    
    Args:
        question: The user's question
        chat_history: The chat history
        selected_files: Optional list of selected files to search within
        
    Returns:
        dict: The agent's response and updated chat history
    """
    logger.info(f"Running agent with question: {question}")
    
    # Initialize the agent state
    state = AgentState(
        question=question,
        chat_history=chat_history.copy() if chat_history else [],
        selected_files=selected_files or [],
        documents=[],
        response="",
        should_stop=False,
        customers=[],
        is_summary_request=False,
        needs_casual_response=False
    )
    
    # Run the graph
    result = await graph.ainvoke(state)
    
    # Return the result
    return {
        "response": result["response"],
        "chat_history": result["chat_history"]
    }
