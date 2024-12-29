# services/agent.py

import logging
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.documents import Document

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT
)
from config.azure_search import vector_store
from models.schemas import Message, AgentState

logger = logging.getLogger(__name__)

# Constants for reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Initialize reranking model and tokenizer globally
logger.info(f"Initializing reranking model: {RERANKER_MODEL_NAME}")
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

# Initialize the language model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2023-03-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3,
    top_p=0.7
)

# Prompt templates

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(condense_question_template)

reasoning_template = """You are an expert query rewriter for a retrieval-augmented generation (RAG) chatbot. Your goal is to take a user's original query and rewrite it to be more effective at retrieving relevant information from a knowledge base. The rewritten query should be clear, specific, and include relevant keywords to maximize the chances of finding the most helpful documents.

Here's the original user query:

Original Query: {question}

Rewrite the above query, keeping the following in mind:

Clarity and Specificity: Make the query as clear and specific as possible. Avoid vague language or ambiguous terms.
Keywords: Identify and include the most important keywords related to the user's intent. Consider synonyms and related terms.
Context (if needed): If the original query lacks necessary context, add relevant context to help the retrieval process. However, avoid adding unnecessary information.
Focus on the Core Need: Ensure the rewritten query directly addresses the user's underlying information need.
Question Format (if appropriate): If the original query is not in the form of a question, consider rephrasing it as a clear question.
Avoid Jargon (unless necessary): Use common language unless the topic specifically requires technical terms.
Maintain User Intent: Do not change the fundamental meaning or goal of the original query.
Rewritten Query:"""
REASONING_PROMPT = ChatPromptTemplate.from_template(reasoning_template)

answer_template = """
Please provide an answer based strictly on the following context:
<context>
{context}
</context>

IMPORTANT GUIDELINES:
1. **Content Reliance**
   - Do not invent or infer information not explicitly found in the context.
   - If context is insufficient, clearly state what is missing.

2. **Structure & Hierarchy**
   a. Use clear hierarchical numbering (1., 2., 3. for main points).
   b. Use indented sub-points (a., b., c.).
   c. Group related information logically.
   d. Maintain consistent indentation for visual hierarchy.

3. **Visual Formatting**
   a. Use Markdown for emphasis:
    - **Bold** for headers
    - *Italic* for emphasis
    - `code` for technical terms
    - > blockquotes for important quotes
   b. Use tables for structured data where relevant.
   c. Insert single line breaks between sections.
   d. Avoid extra spacing between bullet points.

4. **Content Organization**
   a. Begin with a concise *Summary*.
   b. Present information in order of importance.
   c. Use transition sentences between major sections.
   d. End with a conclusion or next steps, if applicable.
   e. Write succinctly; avoid redundant details and keep explanations clear.

5. **Question & Answer Structure**
   Question: {question}

   Answer:
   - **If sufficient information exists**
    **Summary:**
    [1 sentence overview]

    **Detailed Response:**
    1. [First main point]
    a. [Supporting detail]
    b. [Supporting detail]
    2. [Second main point]
    a. [Supporting detail]
    b. [Supporting detail]

   - **If information is incomplete**
    **Available Information:**
    1. [Available information point]
    a. [Supporting detail]

    **Information Gaps:**
    1. [Missing elements]
    a. [Specific missing details]
    b. [Impact on completeness]

   - **If no relevant information**
    **Notice:** The provided context does not contain information to answer this question.
    **Suggested Alternative:** [If applicable, suggest related topics]

6. **Quality Checks**
   ✓ Ensure points are supported by the provided context only.
   ✓ Identify and highlight any information gaps.
   ✓ Provide consistent formatting.
   ✓ Include direct citations or references from the context where relevant.
   ✓ Keep the response concise by avoiding unnecessary or repetitive details.
"""

ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)

def format_chat_history(chat_history: List[Message]) -> str:
    """Format chat history for the model."""
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)

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
        | llm
        | StrOutputParser()
    )
    result = _input.invoke(state)
    state.question = result
    return state

def reason_about_query(state: AgentState) -> AgentState:
    logger.info(f"Reasoning about query with state: {state}")
    _input = (
        RunnableLambda(lambda x: {"question": x.question})
        | REASONING_PROMPT
        | llm
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

def generate_response(state: AgentState) -> AgentState:
    logger.info("Generating response")
    
    if not state.documents:
        state.response = "I couldn't find any relevant information to answer your question."
        return state
    
    # Use only top K documents
    TOP_K_DOCUMENTS = 3
    top_documents = state.documents[:TOP_K_DOCUMENTS]
    
    context = "\n\n".join(doc.page_content for doc in top_documents)
    logger.info(f"Using {len(top_documents)} documents with total context length: {len(context)}")
    
    _input = (
        RunnableLambda(lambda x: {
            "context": context,
            "question": x.question
        })
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    
    state.response = _input.invoke(state)
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
builder.add_node("condense_question", condense_question)
builder.add_node("reason_about_query", reason_about_query) # Add the new reasoning node
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("rerank_documents", rerank_documents)
builder.add_node("generate_response", generate_response)
builder.add_node("update_history", update_history)

# Add edges
builder.set_entry_point("condense_question")
builder.add_edge("condense_question", "reason_about_query") # Connect condense_question to reasoning
builder.add_edge("reason_about_query", "retrieve_documents") # Connect reasoning to retrieval
builder.add_edge("retrieve_documents", "rerank_documents")
builder.add_edge("rerank_documents", "generate_response")
builder.add_edge("generate_response", "update_history")
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