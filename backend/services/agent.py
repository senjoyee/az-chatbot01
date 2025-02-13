import logging
from typing import List, Dict, Any
import torch
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
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
from models.schemas import Message, AgentState  # Import AgentState

logger = logging.getLogger(__name__)

# Initialize the tool
retriever_tool = RetrieverTool()

# Constants for reranking
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Initialize reranking model and tokenizer globally
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
"""
ANSWER_PROMPT = PromptTemplate.from_template(answer_template)

# Prompt for deciding if an answer can be generated
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
    """Format chat history for the model."""
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)


def detect_customers_and_operator(query: str) -> tuple[List[str], bool]:
    """
    Detect customer names in the query string and determine if user wants ALL customers (AND) or ANY customer (OR).
    Returns a tuple of (detected_customers, use_and_operator).
    """
    query_lower = query.lower()
    detected = [name for name in CUSTOMER_NAMES if name.lower() in query_lower]

    # Check if query implies ALL customers should be included
    use_and = False
    if len(detected) > 1:  # Only check for AND if multiple customers detected
        and_indicators = ['&', ' and ', ' both ', ' all ']
        use_and = any(indicator in query_lower for indicator in and_indicators)

    return detected, use_and

def detect_customers(query: str) -> List[str]:
    """
    Detect customer names in the query string.
    Returns a list of detected customer names (case-insensitive).
    """
    query_lower = query.lower()
    return [name for name in CUSTOMER_NAMES if name.lower() in query_lower]

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

def check_customer_specification(state: AgentState) -> AgentState:
    # 1. Check for explicit customer mentions
    if detected := detect_customers(state.question):
        logger.info(f"Detected customers: {detected}")
        return state

    # 2. Check chat history (last 3 messages)
    chat_context = "\n".join([msg.content for msg in state.chat_history[-3:]])
    if history_customers := detect_customers(chat_context):
        logger.info(f"Using historical customer context: {history_customers}")
        state.customer = history_customers[0]  # Take first match
        return state

    # 3. Check for contact/keyword triggers
    contact_triggers = {"contacts", "point of contact", "escalation"}
    if any(trigger in state.question.lower() for trigger in contact_triggers):
        state.response = "Which customer's contact information are you requesting?"
        state.should_stop = True
        return state

    # 4. Final LLM verification
    prompt = f"""Should this query be handled with customer-specific documents?
    Query: {state.question}
    Chat History: {chat_context}

    Answer ONLY yes/no:"""

    customer_response = llm_4o_mini.invoke(prompt)
    customer_intent = customer_response.content.strip().lower()

    if customer_intent.startswith("yes"):
        state.response = "Please specify which customer this request pertains to."
        state.should_stop = True

    return state

def retrieve_documents(state: AgentState) -> AgentState:
    logger.info(f"Retrieving documents for question: {state.question}")
    try:
        # Detect customers and operator type in the query
        detected_customers, use_and_operator = detect_customers_and_operator(state.question)
        filters = None

        if detected_customers:
            filter_conditions = [f"customer eq '{c}'" for c in detected_customers]
            operator = " and " if use_and_operator else " or "
            filters = operator.join(filter_conditions)
            logger.info(f"Applying customer filters with {operator.strip()} operator: {filters}")

        state.documents = retriever_tool.run({
            "query": state.question,
            "k": 20,
            "filters": filters
        })
        logger.info(f"Retrieved {len(state.documents)} documents")

        # --- SHORT-CIRCUIT IF NO DOCUMENTS ---
        if not state.documents:
            state.response = "I could not find any relevant documents in the database."
            state.should_stop = True
            return state  # Exit early

    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        state.documents = []  # Ensure we have a valid state even on failure
        state.response = "An error occurred while retrieving documents."
        state.should_stop = True  # Also stop on error
        return state

    return state

def rerank_documents(state: AgentState) -> AgentState:
    logger.info("Reranking documents")
    documents = state.documents

    # Check if there are no documents to rerank
    if not documents:
        logger.info("No documents to rerank, skipping reranking step")
        return state  # No need to raise an exception

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

    TOP_K_DOCUMENTS = 12
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

    # Set answer_generated_from_document_store and response accordingly
    if state.can_generate_answer:
        state.answer_generated_from_document_store = "pass"
    else:
        state.answer_generated_from_document_store = "fail"
        state.response = "The answer to your query cannot be generated from the documents provided."

    logger.info(f"Decision: {state.can_generate_answer}, Document Store Answer: {state.answer_generated_from_document_store}")
    return state


def generate_response(state: AgentState) -> AgentState:
    logger.info("Generating response")

    # Only generate if answer was successfully generated from documents
    if state.answer_generated_from_document_store != "pass":
        return state  #  Exit if not

    TOP_K_DOCUMENTS = 12
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

def detect_casual_talk(state: AgentState) -> AgentState:
    """Determines if message requires casual response."""
    state.needs_casual_response = is_casual_conversation(state.question, llm_4o_mini)
    return state

def respond_to_casual(state: AgentState) -> AgentState:
    """Generates conversational response using LLM."""
    state.response = llm_4o_mini.invoke(
        conversation_prompt.format(
            message=state.question,
            history=format_chat_history(state.chat_history)
        )
    ).content
    state.should_stop = True  # End conversation after responding
    return state

# Build the Langgraph
builder = StateGraph(AgentState)

# Nodes
builder.add_node("detect_casual", detect_casual_talk)
builder.add_node("respond_casual", respond_to_casual)
builder.add_node("condense", condense_question)
builder.add_node("check_customer", check_customer_specification)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("rerank", rerank_documents)
builder.add_node("decide_to_generate", decide_to_generate)
builder.add_node("generate", generate_response)
builder.add_node("update_history", update_history)

# Edges
# Handle casual conversation and main flow branching
builder.add_conditional_edges(
    "detect_casual",
    lambda s: "respond_casual" if s.needs_casual_response else "condense"
)

# Main conversation flow
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

# Final edge - either continue or end
builder.add_conditional_edges(
    "update_history",
    lambda s: END if s.should_stop else "detect_casual"
)

# Set entry point
builder.set_entry_point("detect_casual")

# Compile the graph
agent = builder.compile()

async def run_agent(question: str, chat_history: List[Message]) -> Dict[str, Any]:
    """Runs the Langgraph agent."""
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "documents": None,
        "response": None,
        "can_generate_answer": True,
        "should_stop": False,
        "answer_generated_from_document_store": None,  # Initialize correctly
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