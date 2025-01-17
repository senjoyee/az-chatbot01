# services/agent.py

import logging
import json
from typing import List, Dict, Any
import torch
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.documents import Document

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    SERPER_API_KEY
)
from config.azure_search import vector_store
from models.schemas import Message, AgentState

logger = logging.getLogger(__name__)

# Initialize the search wrapper
search = GoogleSerperAPIWrapper()

# Constants for reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Initialize reranking model and tokenizer globally
logger.info(f"Initializing reranking model: {RERANKER_MODEL_NAME}")
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

# Initialize the language model
llm_4o_mini = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2023-03-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3,
    top_p=0.7
)

llm_4o = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.1,
    top_p=0.9
)

# Prompt templates

# Add new prompt for search type determination
determine_search_type_template = """Analyze the following question and determine if it requires internal document search or web search.

Question: {question}

Consider these guidelines:
1. Use internal search for:
   - Questions about specific customers, contracts, or company documents
   - Internal business processes or policies
   - Customer-specific information
2. Use web search for:
   - General knowledge questions
   - Industry trends or news
   - Technical information not specific to customers
   - Public information or current events

Respond in JSON format:
{
    "use_web_search": boolean,
    "reasoning": "brief explanation"
}
"""

DETERMINE_SEARCH_TYPE_PROMPT = PromptTemplate.from_template(determine_search_type_template)

query_reasoning_template = """
You are tasked with rewriting a user's query to make it more likely to match relevant documents in a retrieval system. The goal is to transform the query into a more assertive and focused form that will improve search results.

Follow these guidelines when rewriting the query:
1. Use an assertive tone
2. Be more specific and detailed
3. Include relevant keywords
4. Remove unnecessary words or phrases
5. Structure the query as a statement rather than a question, if applicable
6. Maintain the original intent of the query

Here is the user's original query:
<user_query>
{question}
</user_query>

Rewrite the query following the guidelines above. Think carefully about how to improve the query's effectiveness in retrieving relevant documents.
"""

QUERY_REASONING_PROMPT = PromptTemplate.from_template(query_reasoning_template)

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

answer_template = """
You are an AI assistant designed to answer questions based on provided documents. These documents may include service operating manuals, contract documents, or other relevant information. Your task is to analyze the given documents and use them to answer user questions accurately and helpfully.

First, carefully read and analyze the following documents:

<documents>
{context}
</documents>

As you read through the documents, pay attention to key information, important details, and any specific instructions or clauses that might be relevant to potential user questions. Create a mental index of the main topics and sections within the documents for quick reference.

Now, a user has asked the following question:

<user_question>
{question}
</user_question>

To answer the user's question, follow these steps:

1. Identify the main topic(s) of the question and search for relevant information within the provided documents.

2. If you find information directly related to the question, use it to formulate your answer. Be sure to paraphrase the information rather than quoting it verbatim, unless a direct quote is necessary for accuracy or clarity.

3. If the question is not directly addressed in the documents, use your understanding of the overall content to provide the best possible answer. In this case, make it clear that your response is based on your interpretation of the available information.

4. If the question cannot be answered using the provided documents, politely inform the user that the information is not available in the current documentation.

5. If appropriate, provide additional context or related information that might be helpful to the user, even if it doesn't directly answer their question.

6. If the user's question is unclear or too broad, ask for clarification to ensure you provide the most accurate and helpful response.

When formulating your answer, keep the following in mind:

- Be concise and to the point, while still providing comprehensive information.
- Use clear and simple language, avoiding jargon unless it's specifically relevant to the topic.
- If discussing technical procedures or contract terms, be precise and accurate.
- Maintain a professional and helpful tone throughout your response.

Provide your answer within <answer> tags. If you need to ask for clarification, do so before providing your answer. If you're unsure about any part of your response, indicate this clearly to the user.

Remember, your goal is to provide accurate, helpful information based on the documents provided, while maintaining a friendly and professional demeanor.
"""

ANSWER_PROMPT = PromptTemplate.from_template(answer_template)

def format_chat_history(chat_history: List[Message]) -> str:
    """Format chat history for the model."""
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)

def detect_customer_name(state: AgentState) -> AgentState:
    # Create a prompt template for customer name detection
    customer_detection_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the following question and determine if it mentions a specific customer name. Output JSON format with fields: has_customer_name (boolean) and customer_name (string or null)"),
        ("user", "{question}")
    ])
    
    # Chain for customer detection
    chain = (
        customer_detection_prompt 
        | llm_4o_mini 
        | StrOutputParser() 
        | json.loads
    )
    
    # Run detection
    result = chain.invoke({"question": state.question})
    state.customer_name = result["customer_name"]
    state.needs_customer_prompt = not result["has_customer_name"]
    
    return state

def generate_customer_prompt(state: AgentState) -> AgentState:
    if state.needs_customer_prompt:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a polite prompt asking the user if they want to specify a customer name for their query."),
            ("user", "Generate prompt for question: {question}")
        ])
        
        response = prompt | llm_4o_mini | StrOutputParser()
        state.awaiting_customer_response = True
        state.response = response.invoke({"question": state.question})
        return state
    return state

def condense_question(state: AgentState) -> AgentState:
    logger.info(f"Condensing question with state: {state}")
    if not state.chat_history:  # Empty history
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

def reason_about_query(state: AgentState) -> AgentState:
    logger.info(f"Reasoning about query with state: {state}")
    _input = (
        RunnableLambda(lambda x: {"question": x.question})
        | QUERY_REASONING_PROMPT
        | llm_4o_mini   
        | StrOutputParser()
    )
    rewritten_query = _input.invoke(state)
    state.original_question = state.question  # Keep the original question
    state.question = rewritten_query
    logger.info(f"Rewritten query: {rewritten_query}")
    return state

def retrieve_documents(state: AgentState) -> AgentState:
    logger.info(f"Retrieving documents for question: {state.question}")
    state.documents = vector_store.hybrid_search(
        state.question,
        k=10,
    )
    return state

def rerank_documents(state: AgentState) -> AgentState:
    logger.info("Reranking documents")
    
    query = state.question
    documents = state.documents
    
    # Prepare inputs for reranking
    text_pairs = [(query, doc.page_content) for doc in documents]
    inputs = reranker_tokenizer.batch_encode_plus(
        text_pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    
    # Get scores
    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze()
    
    # Convert scores to list if it's a tensor
    if torch.is_tensor(scores):
        scores = scores.tolist()
    
    # Sort documents by score
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Update state with reranked documents
    state.documents = [doc for doc, _ in scored_docs]
    return state

def determine_search_type(state: AgentState) -> AgentState:
    """Determine if the question needs web search or internal search."""
    chain = DETERMINE_SEARCH_TYPE_PROMPT | llm_4o_mini | StrOutputParser() | json.loads
    
    try:
        result = chain.invoke({"question": state.question})
        if not isinstance(result, dict) or "use_web_search" not in result:
            logger.error("Invalid response format from search type determination")
            state.use_web_search = False
            return state
            
        state.use_web_search = result["use_web_search"]
        logger.info(f"Search type determination: {result.get('reasoning', 'No reasoning provided')}")
    except json.JSONDecodeError:
        logger.error("Failed to parse search type determination response")
        state.use_web_search = False
    except Exception as e:
        logger.error(f"Error determining search type: {str(e)}")
        state.use_web_search = False
    
    return state

def perform_web_search(state: AgentState) -> AgentState:
    """Perform web search using GoogleSerperAPIWrapper."""
    if not state.use_web_search:
        return state
        
    if not SERPER_API_KEY:
        logger.error("SERPER_API_KEY not configured")
        state.use_web_search = False
        return state
        
    try:
        # Use the wrapper to perform the search
        results = search.results(state.question, num_results=3)
        
        if not results:
            logger.warning("No search results found")
            state.web_results = "No relevant information found from web search."
            return state
            
        # Format the results
        formatted_results = []
        for item in results:
            result = f"Title: {item.get('title', 'No title')}\n"
            result += f"Snippet: {item.get('snippet', 'No snippet')}\n"
            result += f"Link: {item.get('link', 'No link')}\n"
            formatted_results.append(result)
        
        state.web_results = "\n\n".join(formatted_results)
        logger.info(f"Web search completed successfully with {len(formatted_results)} results")
            
    except Exception as e:
        logger.error(f"Error in web search: {str(e)}")
        state.web_results = None
        state.use_web_search = False
    
    return state


def generate_response(state: AgentState) -> AgentState:
    """Generate response using either web search results or internal documents."""
    logger.info("Generating response")
    
    # Prepare context based on search type
    if state.use_web_search:
        if not state.web_results:
            state.response = "I apologize, but I couldn't retrieve any information from the web search. Please try rephrasing your question or asking something else."
            return state
        context = f"Web Search Results:\n{state.web_results}"
    else:
        if not state.documents:
            state.response = "I couldn't find any relevant information in our internal documents to answer your question."
            return state
        # Use top K documents for internal search
        TOP_K_DOCUMENTS = 3
        top_documents = state.documents[:TOP_K_DOCUMENTS]
        context = "\n\n".join(doc.page_content for doc in top_documents)
        logger.info(f"Using {len(top_documents)} internal documents")
    
    try:
        # Generate response using the appropriate context
        _input = (
            RunnableLambda(lambda x: {
                "context": context,
                "question": x.question
            })
            | ANSWER_PROMPT
            | llm_4o
            | StrOutputParser()
        )
        
        response = _input.invoke(state)
        response = response.replace("<answer>", "").replace("</answer>", "").strip()
        
        if not response:
            state.response = "I apologize, but I couldn't generate a meaningful response. Please try rephrasing your question."
        else:
            state.response = response
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        state.response = "I encountered an error while generating the response. Please try again."
    
    return state

def update_history(state: AgentState) -> AgentState:
    logger.info(f"Updating history with state: {state}")
    if not state.chat_history:
        state.chat_history = []
    state.chat_history.extend([
        Message(role="user", content=state.question),
        Message(role="assistant", content=state.response)
    ])
    return state

# Build the Langgraph
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("detect_customer", detect_customer_name)
builder.add_node("customer_prompt", generate_customer_prompt)
builder.add_node("determine_search", determine_search_type)
builder.add_node("web_search", perform_web_search)
builder.add_node("condense", condense_question)
builder.add_node("reason", reason_about_query)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("rerank", rerank_documents)
builder.add_node("generate", generate_response)
builder.add_node("update_history", update_history)

# Add edges
builder.add_edge(START, "detect_customer")

# From customer detection
builder.add_conditional_edges(
    "detect_customer",
    lambda x: "customer_prompt" if x.needs_customer_prompt else "determine_search",
    {
        "customer_prompt": "customer_prompt",
        "determine_search": "determine_search"
    }
)

# From customer prompt
builder.add_conditional_edges(
    "customer_prompt",
    lambda x: END if x.awaiting_customer_response else "determine_search",
    {
        END: END,
        "determine_search": "determine_search"
    }
)

# From search type determination
builder.add_conditional_edges(
    "determine_search",
    lambda x: "web_search" if x.use_web_search else "condense",
    {
        "web_search": "web_search",
        "condense": "condense"
    }
)

# Rest of the flow
builder.add_edge("web_search", "generate")
builder.add_edge("condense", "reason")
builder.add_edge("reason", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", "update_history")
builder.add_edge("update_history", END)

# Compile the graph
agent = builder.compile()

async def run_agent(question: str, chat_history: List[Message]) -> Dict[str, Any]:
    """Runs the Langgraph agent."""
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "documents": None,
        "response": None
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