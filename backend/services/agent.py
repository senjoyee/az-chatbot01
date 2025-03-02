import logging
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
from config.azure_search import vector_store
from models.schemas import Message, AgentState

logger = logging.getLogger(__name__)

# Initialize the tool
retriever_tool = RetrieverTool()

# Constants for reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

logger.info(f"Initializing reranking model: {RERANKER_MODEL_NAME}")
reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)

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
)

# List of customer names for filtering
CUSTOMER_NAMES = [
    "bsw",
    "tbs",
    "npac",
    "asahi",
    # Add more customer names here
]

# Prompt templates
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

Now, a user has asked the following question:

<user_question>
{question}
</user_question>

Provide your answer within <answer> tags.
"""
ANSWER_PROMPT = PromptTemplate.from_template(answer_template)

CONVERSATION_PROMPT = PromptTemplate.from_template(
    """You are a friendly and helpful AI assistant. Respond to the following message in a natural, conversational way.
If there is chat history, maintain a consistent and contextual conversation.

Chat History:
{history}

User Message:
{message}

Your response should be brief and friendly."""
)

DECISION_PROMPT = PromptTemplate.from_template(
    """Given the following question and document excerpts, determine if a reasonable answer can be generated.

<question>
{question}
</question>

<documents>
{context}
</documents>

Respond with 'yes' if a reasonable answer can be generated, or 'no' if not.
"""
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
        filters = None
        filter_expressions = []
        
        # Add customer filter if customers are detected
        if state.customers:
            customer_filters = " or ".join([f"customer eq '{customer}'" for customer in state.customers])
            filter_expressions.append(f"({customer_filters})")
            logger.info(f"Adding customer filter: {state.customers}")
            
        # Add file filter if files are selected
        if state.selected_files:
            file_filters = " or ".join([f"source eq '{file}'" for file in state.selected_files])
            filter_expressions.append(f"({file_filters})")
            logger.info(f"Adding file filter: {state.selected_files}")
            
        # Combine filters with AND if both are present
        if filter_expressions:
            filters = " and ".join(filter_expressions)
            logger.info(f"Combined filter expression: {filters}")
        
        state.documents = retriever_tool.run({
            "query": state.question,
            "k": 25,
            "filters": filters
        })
        
        logger.info(f"Retrieved {len(state.documents)} documents with filters: {filters}")
        
        if not state.documents:
            state.response = "I could not find any relevant documents in the database."
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

    query = state.question
    text_pairs = [(query, doc.page_content) for doc in documents]
    inputs = reranker_tokenizer.batch_encode_plus(
        text_pairs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    with torch.no_grad():
        scores = reranker_model(**inputs).logits.squeeze()

    if torch.is_tensor(scores):
        scores = scores.tolist()

    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    state.documents = [doc for doc, _ in scored_docs]
    return state

def decide_to_generate(state: AgentState) -> AgentState:
    logger.info("Deciding whether to generate a response")
    TOP_K_DOCUMENTS = 10
    top_documents = state.documents[:TOP_K_DOCUMENTS]
    context = "\n\n".join(doc.page_content for doc in top_documents)

    _input = (
        RunnableLambda(lambda x: {
            "context": context,
            "question": x.question
        })
        | DECISION_PROMPT
        | llm_o3_mini
        | StrOutputParser()
    )
    response = _input.invoke(state).strip().lower()
    state.can_generate_answer = "yes" in response

    if state.can_generate_answer:
        state.answer_generated_from_document_store = "pass"
    else:
        state.answer_generated_from_document_store = "fail"
        state.response = "The answer to your query cannot be generated from the documents provided."
        logger.info("Answer generation decision: cannot generate answer")
    logger.info(f"Decision: {state.can_generate_answer}, Document Store Answer: {state.answer_generated_from_document_store}")
    return state

def generate_response(state: AgentState) -> AgentState:
    logger.info("Generating response")
    if state.answer_generated_from_document_store != "pass":
        return state
    TOP_K_DOCUMENTS = 10
    top_documents = state.documents[:TOP_K_DOCUMENTS]
    context = "\n\n".join(doc.page_content for doc in top_documents)
    logger.info(f"Using {len(top_documents)} documents with total context length: {len(context)}")

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
    state.response = response.replace("<answer>", "").replace("</answer>", "").strip()
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

def respond_to_casual(state: AgentState) -> AgentState:
    state.response = llm_4o_mini.invoke(
        CONVERSATION_PROMPT.format(
            message=state.question,
            history=format_chat_history(state.chat_history)
        )
    ).content
    state.should_stop = True
    return state

# Build the Langgraph
builder = StateGraph(AgentState)

builder.add_node("detect_casual", detect_casual_talk)
builder.add_node("respond_casual", respond_to_casual)
builder.add_node("condense", condense_question)
builder.add_node("check_customer", check_customer_specification)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("rerank", rerank_documents)
builder.add_node("decide_to_generate", decide_to_generate)
builder.add_node("generate", generate_response)
builder.add_node("update_history", update_history)

builder.add_conditional_edges(
    "detect_casual",
    lambda s: "respond_casual" if s.needs_casual_response else "condense"
)
builder.add_edge("respond_casual", "update_history")
builder.add_edge("condense", "check_customer")
builder.add_conditional_edges(
    "check_customer",
    lambda s: "update_history" if s.should_stop else "retrieve"
)
builder.add_conditional_edges(
    "retrieve",
    lambda s: "update_history" if s.should_stop else "rerank"
)
builder.add_edge("rerank", "decide_to_generate")
builder.add_conditional_edges(
    "decide_to_generate",
    lambda s: "generate" if s.can_generate_answer else "update_history"
)
builder.add_edge("generate", "update_history")
builder.add_conditional_edges(
    "update_history",
    lambda s: END if s.should_stop else "detect_casual"
)

builder.set_entry_point("detect_casual")
agent = builder.compile()

async def run_agent(question: str, chat_history: List[Message], selected_files: List[str] = None) -> Dict[str, Any]:
    # Note: We no longer rely solely on the in-memory flag since it is reset each turn.
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "documents": None,
        "response": None,
        "can_generate_answer": True,
        "should_stop": False,
        "answer_generated_from_document_store": None,
        "customer_reminder_sent": False,
        "selected_files": selected_files or []
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