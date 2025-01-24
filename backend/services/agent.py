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
from .tools import RetrieverTool

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT
)
from config.azure_search import vector_store
from models.schemas import Message, AgentState

logger = logging.getLogger(__name__)

# Initialize the tool
retriever_tool = RetrieverTool()

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
    try:
        state.documents = retriever_tool.run({
            "query": state.question,
            "k": 10
        })
        logger.info(f"Retrieved {len(state.documents)} documents")
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        state.documents = []  # Ensure we have a valid state even on failure
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
        | llm_4o
        | StrOutputParser()
    )
    
    response = _input.invoke(state)
    
    # Strip <answer> tags from the response
    response = response.replace("<answer>", "").replace("</answer>", "").strip()
    
    state.response = response
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
builder.add_node("condense", condense_question)
builder.add_node("reason", reason_about_query)  # Add the reasoning node
builder.add_node("retrieve", retrieve_documents)
builder.add_node("rerank", rerank_documents)
builder.add_node("generate", generate_response)
builder.add_node("update_history", update_history)

# Add edges
builder.add_edge("condense", "reason")  # Connect condense to reason
builder.add_edge("reason", "retrieve")  # Connect reason to retrieve
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", "update_history")
builder.add_edge("update_history", END)

# Set entry point
builder.set_entry_point("condense")

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