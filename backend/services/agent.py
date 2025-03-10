import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.documents import Document

from .tools import RetrieverTool
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

from config.azure_search import vector_store
from models.schemas import Message, AgentState

logger = logging.getLogger(__name__)

# Initialize the tool
retriever_tool = RetrieverTool()

# Initialize the language models
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
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)

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

CUSTOMER_NAMES = [
    "bsw",
    "tbs",
    "npac",
    "asahi"
    # Add more customer names here
]

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
    logger.info(f"Retrieving documents for question: {state.question}")
    try:
        # Build filters based on available criteria
        filter_expression = None
        filter_parts = []
        
        # Add customer filter if customers are detected in the query
        if state.customers:
            logger.info(f"Detected customers: {state.customers}")
            customer_filters = " or ".join([f"customer eq '{customer}'" for customer in state.customers])
            filter_parts.append(f"({customer_filters})")
            logger.info(f"Adding customer filter: {customer_filters}")
            
        # Add file filter if files are selected
        if state.selected_files:
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
            state.documents = retriever_tool.run({
                "query": state.question,
                "k": 25,
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
            logger.error(f"Error calling retriever tool: {str(e)}")
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
    logger.info("Reranking documents")
    documents = state.documents
    if not documents:
        logger.info("No documents to rerank, skipping reranking step")
        return state
    
    # The documents are already ranked by Azure AI Search's semantic ranking
    # We can skip the local reranking step
    logger.info("Documents already ranked by Azure AI Search semantic ranking")
    logger.info(f"Using {len(documents)} documents")
    
    # If you want to apply additional custom ranking logic, you can do it here
    # For now, we'll just return the documents in the order provided by Azure AI Search
    
    return state

def generate_response(state: AgentState) -> AgentState:
    logger.info("Generating response")
    TOP_K_DOCUMENTS = 10
    top_documents = state.documents[:TOP_K_DOCUMENTS]
    
    # Check if this is a summary request
    is_summary = is_summary_request(state.question)
    logger.info(f"Request identified as summary request: {is_summary}")
    
    # Prepare context differently based on request type
    if is_summary:
        # For summaries, include document metadata to provide better structure
        context_parts = []
        for i, doc in enumerate(top_documents):
            # Include document metadata if available
            source = doc.metadata.get('source', 'Unknown document')
            title = doc.metadata.get('title', f'Document {i+1}')
            
            # Format with metadata
            context_parts.append(f"## {title}\nSource: {source}\n\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
    else:
        # For regular questions, use the standard approach
        context = "\n\n".join(doc.page_content for doc in top_documents)
    
    logger.info(f"Using {len(top_documents)} documents with total context length: {len(context)}")

    # Use different prompts based on request type
    if is_summary:
        _input = (
            RunnableLambda(lambda x: {
                "context": context,
                "question": x.question
            })
            | SUMMARY_PROMPT  # Use the specialized summary prompt
            | llm_o3_mini
            | StrOutputParser()
        )
    else:
        _input = (
            RunnableLambda(lambda x: {
                "context": context,
                "question": x.question
            })
            | ANSWER_PROMPT
            | llm_o3_mini
            | StrOutputParser()
        )
    
    response = _input.invoke(state)
    cleaned_response = response.replace("<answer>", "").replace("</answer>", "").strip()
    
    # Check if the response indicates insufficient information
    if "I don't have enough information" in cleaned_response:
        state.response = "I couldn't find enough information in my knowledge base to answer your question properly. Could you please rephrase or ask something else?"
    else:
        state.response = cleaned_response
    
    return state

def update_history(state: AgentState) -> AgentState:
    logger.info(f"Updating history with state: {state}")
    if not state.chat_history:
        state.chat_history = []
    state.chat_history.extend([
        Message(role="user", content=state.question),
        Message(role="assistant", content=state.response)
    ])
    state.should_stop = True
    return state

def detect_casual_talk(state: AgentState) -> AgentState:
    state.needs_casual_response = is_casual_conversation(state.question, llm_4o_mini)
    return state

def detect_summary_intent(state: AgentState) -> AgentState:
    """
    Detect if the user is requesting a document summary.
    This is placed early in the workflow to enable proper routing.
    """
    state.is_summary_request = is_summary_request(state.question)
    logger.info(f"Request identified as summary request: {state.is_summary_request}")
    return state

def respond_to_casual(state: AgentState) -> AgentState:
    state.response = llm_4o_mini.invoke(
        CONVERSATION_PROMPT.format(
            message=state.question,
            history=format_chat_history(state.chat_history)
        )
    ).content
    state.should_stop = True
    return state

def retrieve_documents_for_summary(state: AgentState) -> AgentState:
    """
    Retrieve documents specifically for summarization.
    This function is optimized for comprehensive coverage rather than precision.
    """
    logger.info(f"Retrieving documents for summary: {state.question}")
    try:
        # Build filters based on available criteria - similar to retrieve_documents
        filter_expression = None
        filter_parts = []
        
        # Add customer filter if customers are detected in the query
        if state.customers:
            logger.info(f"Detected customers for summary: {state.customers}")
            customer_filters = " or ".join([f"customer eq '{customer}'" for customer in state.customers])
            filter_parts.append(f"({customer_filters})")
            logger.info(f"Adding customer filter for summary: {customer_filters}")
            
        # Add file filter if files are selected
        if state.selected_files:
            # Check if this is effectively a "select all" scenario
            SELECT_ALL_THRESHOLD = 100  # If more than this many files are selected, treat as "select all"
            MAX_FILES_IN_FILTER = 50    # Allow up to 50 files in filter
            
            if len(state.selected_files) > SELECT_ALL_THRESHOLD:
                logger.info(f"Large number of files selected for summary ({len(state.selected_files)}). Treating as 'select all' - no file filter will be applied.")
                # Don't add file filter - but customer filter may still be applied if detected
            else:
                selected_files = state.selected_files
                
                if len(selected_files) > MAX_FILES_IN_FILTER:
                    logger.warning(f"Too many files selected for summary ({len(selected_files)}). Limiting to {MAX_FILES_IN_FILTER} files.")
                    selected_files = selected_files[:MAX_FILES_IN_FILTER]
                
                # Escape single quotes by doubling them
                file_filters = " or ".join([f"source eq '{file.replace(chr(39), chr(39)*2)}'" for file in selected_files])
                filter_parts.append(f"({file_filters})")
                logger.info(f"Adding file filter for summary: {len(selected_files)} files")
            
        # Combine filters if present
        if filter_parts:
            filter_expression = " and ".join(filter_parts)
            logger.info(f"Combined filter expression for summary: {filter_expression}")
            
            # Log the length of the filter expression to check if it's too long
            logger.info(f"Filter expression length for summary: {len(filter_expression)}")
            
            # If filter expression is very long, log a warning
            if len(filter_expression) > 10000:
                logger.warning("Filter expression for summary is very long, which might cause issues with Azure Search")
        else:
            logger.info("No filters applied for summary - searching across all documents")
        
        try:
            # For summaries, retrieve more documents (100 instead of 25)
            # and configure search for broader coverage
            state.documents = retriever_tool.run({
                "query": state.question,
                "k": 100,  # Retrieve more documents for comprehensive summaries
                "filters": filter_expression,
                "queryType": "simple"  # Use simple query type for broader coverage
            })
            
            logger.info(f"Retrieved {len(state.documents)} documents for summary")
            
            # Log the first few document sources for debugging
            if state.documents:
                sources = [doc.metadata.get('source', 'unknown') for doc in state.documents[:5]]
                logger.info(f"First few document sources for summary: {sources}")
            
            if not state.documents:
                state.response = "I could not find any relevant documents in the database to summarize."
                state.should_stop = True
                return state
        except Exception as e:
            logger.error(f"Error calling retriever tool for summary: {str(e)}")
            state.documents = []
            state.response = "An error occurred while retrieving documents for summarization. For best results, either select up to 50 specific files or use 'Select All' to search across the entire database."
            state.should_stop = True
            return state
    except Exception as e:
        logger.error(f"Error retrieving documents for summary: {str(e)}")
        state.documents = []
        state.response = "An error occurred while retrieving documents for summarization."
        state.should_stop = True
        return state
    return state

def process_documents_for_summary(state: AgentState) -> AgentState:
    """
    Process and organize documents specifically for summarization.
    This function groups documents by source and prepares them for effective summarization.
    """
    logger.info("Processing documents for summary")
    documents = state.documents
    if not documents:
        logger.info("No documents to process for summary, skipping processing step")
        return state
    
    try:
        # Group documents by source
        documents_by_source = {}
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in documents_by_source:
                documents_by_source[source] = []
            documents_by_source[source].append(doc)
        
        logger.info(f"Grouped documents into {len(documents_by_source)} sources")
        
        # Prepare processed documents with structure
        processed_documents = []
        for source, docs in documents_by_source.items():
            # Sort documents by page/chunk number if available
            sorted_docs = sorted(docs, key=lambda d: d.metadata.get('chunk', 0))
            
            # Create a structured document with source information
            source_title = source.split('/')[-1] if '/' in source else source
            source_content = "\n\n".join([doc.page_content for doc in sorted_docs])
            
            # Add structured document to processed list
            processed_documents.append(Document(
                page_content=source_content,
                metadata={
                    'source': source,
                    'title': source_title,
                    'document_count': len(sorted_docs)
                }
            ))
        
        # Replace original documents with processed ones
        state.documents = processed_documents
        logger.info(f"Processed {len(processed_documents)} document groups for summary")
        
        return state
    except Exception as e:
        logger.error(f"Error processing documents for summary: {str(e)}")
        # If processing fails, continue with original documents
        logger.info("Continuing with original documents")
        return state

def generate_summary(state: AgentState) -> AgentState:
    """
    Generate a comprehensive summary from the processed documents.
    This function is specifically designed for creating structured, informative summaries.
    """
    logger.info("Generating document summary")
    
    # Prepare context with document metadata for better structure
    context_parts = []
    for i, doc in enumerate(state.documents):
        # Include document metadata if available
        source = doc.metadata.get('source', 'Unknown document')
        title = doc.metadata.get('title', f'Document {i+1}')
        doc_count = doc.metadata.get('document_count', 1)
        
        # Format with metadata
        context_parts.append(f"## {title}\nSource: {source}\nContains content from {doc_count} document sections\n\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    logger.info(f"Using {len(state.documents)} document groups with total context length: {len(context)}")
    
    # Use the specialized summary prompt
    _input = (
        RunnableLambda(lambda x: {
            "context": context,
            "question": x.question
        })
        | SUMMARY_PROMPT
        | llm_o3_mini
        | StrOutputParser()
    )
    
    response = _input.invoke(state)
    cleaned_response = response.replace("<answer>", "").replace("</answer>", "").strip()
    
    # Check if the response indicates insufficient information
    if "I don't have enough information" in cleaned_response:
        state.response = "I couldn't find enough information in my knowledge base to create a comprehensive summary. Could you please specify which documents you'd like me to summarize?"
    else:
        state.response = cleaned_response
    
    return state

# Build the Langgraph
builder = StateGraph(AgentState)

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
builder.add_conditional_edges(
    "retrieve_for_summary",
    lambda s: "update_history" if s.should_stop else "process_for_summary"
)
builder.add_edge("process_for_summary", "generate_summary")
builder.add_edge("generate_summary", "update_history")
builder.add_conditional_edges(
    "update_history",
    lambda s: END if s.should_stop else "detect_summary"
)

builder.set_entry_point("detect_summary")
agent = builder.compile()

async def run_agent(question: str, chat_history: List[Message], selected_files: List[str] = None) -> Dict[str, Any]:
    # Note: We no longer rely solely on the in-memory flag since it is reset each turn.
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "documents": None,
        "response": None,
        "should_stop": False,
        "customer_reminder_sent": False,
        "selected_files": selected_files or [],
        "is_summary_request": False  # Initialize the new field
    }
    try:
        result = await agent.ainvoke(inputs)
        return {
            "response": result["response"],
            "chat_history": result["chat_history"]
        }
    except Exception as e:
        logger.error(f"Agent execution error: {str(e)}")
        return {"response": "An error occurred while processing your request.", "chat_history": chat_history}