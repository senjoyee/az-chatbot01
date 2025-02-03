import logging
import torch
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Optional, AsyncIterator

import asyncio
from langchain_core.callbacks import AsyncCallbackHandler

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from .tools import RetrieverTool

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT
)

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

# Initialize the language models
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
    top_p=0.9,
    streaming=True
)

# Define a callback handler for async LLM calls

class TokenStreamHandler(AsyncCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # Push each new token into the async queue
        await self.queue.put(token)

# List of customer names for filtering
CUSTOMER_NAMES = [
    "bsw",
    "tbs",
    "npac",
    "asahi",
    # Add more customer names here
]


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

def check_greeting_and_customer(state: AgentState) -> AgentState:
    """Handles greetings and conversation flow management."""
    logger.debug(f"Entering check_greeting_and_customer with question: {state.question}")
    logger.debug(f"Chat history type: {type(state.chat_history)}")
    if state.chat_history:
        logger.debug(f"First message type: {type(state.chat_history[0]) if state.chat_history else 'No messages'}")
        logger.debug(f"Chat history sample: {str(state.chat_history)[:200]}")
    
    if state.response:
        logger.debug("State already has response, returning early")
        return state

    state.conversation_turns += 1
    lower_question = state.question.lower()
    logger.debug(f"Processing turn {state.conversation_turns} with lower_question: {lower_question}")
    # Greeting pattern definitions
    greeting_patterns = {
        'initial': [r'\bhello\b', r'\bhi\b', r'\bhey\b', r'good morning', 
                   r'good afternoon', r'good evening', r'how are you'],
        'response': [r'\bi\'m good\b', r'\bdoing great\b', r'\bnot bad\b',
                    r'\bfine thanks\b', r'\bpretty good\b', r'\ball good\b']
    }

    # Initial greeting handling
    if state.conversation_turns == 1:
        for pattern in greeting_patterns['initial']:
            if re.search(pattern, lower_question):
                _input = (
                    RunnableLambda(lambda x: f"Respond to greeting and direct to docs: '{x.question}'")
                    | llm_4o
                    | StrOutputParser()
                )
                state.response = _input.invoke(state)
                state.should_stop = True
                return state

    # Follow-up casual conversation handling
    for pattern in greeting_patterns['response']:
        if re.search(pattern, lower_question):
            state.response = (
                "I'm here to help with document-related questions. "
                "Please ask about specific documents or policies."
            )
            state.should_stop = True
            return state

    state.conversation_turns = 0
    return state

def condense_question(state: AgentState) -> AgentState:
    """Condenses the conversation history and current question."""
    logger.debug(f"Entering condense_question with state: {state.__dict__}")
    logger.debug(f"Chat history format: {format_chat_history(state.chat_history)[:200]}")
    
    try:
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
        logger.debug(f"Condensed question result: {state.question}")
    except Exception as e:
        logger.exception("Error in condense_question")
        raise
    return state

def check_customer_specification(state: AgentState) -> AgentState:
    """Checks for customer specifications in the query."""
    logger.debug(f"Entering check_customer_specification with question: {state.question}")
    
    try:
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
        
        logger.debug(f"Detected customers: {detected}, use_and: {False}")
    except Exception as e:
        logger.exception("Error in check_customer_specification")
        raise
    return state

def reason_about_query(state: AgentState) -> AgentState:
    """Reasons about the query to improve retrieval."""
    logger.debug(f"Entering reason_about_query with question: {state.question}")
    
    try:
        _input = (
            RunnableLambda(lambda x: {"question": x.question})
            | QUERY_REASONING_PROMPT
            | llm_4o_mini
            | StrOutputParser()
        )
        rewritten_query = _input.invoke(state)
        state.original_question = state.question  # Keep the original question
        state.question = rewritten_query
        logger.debug(f"Reasoned query: {state.question}")
    except Exception as e:
        logger.exception("Error in reason_about_query")
        raise
    return state

def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieves relevant documents."""
    logger.debug(f"Entering retrieve_documents with question: {state.question}")
    
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
            "k": 10,
            "filters": filters
        })
        logger.debug(f"Retrieved documents count: {len(state.documents) if state.documents else 0}")
        if state.documents:
            logger.debug(f"First document sample: {str(state.documents[0])[:200]}")
    except Exception as e:
        logger.exception("Error in retrieve_documents")
        raise
    return state

def rerank_documents(state: AgentState) -> AgentState:
    """Reranks retrieved documents."""
    logger.debug(f"Entering rerank_documents")
    logger.debug(f"Documents before reranking: {len(state.documents) if state.documents else 0}")
    
    try:
        query = state.question
        documents = state.documents

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
        logger.debug(f"Documents after reranking: {len(state.documents) if state.documents else 0}")
    except Exception as e:
        logger.exception("Error in rerank_documents")
        raise
    return state

async def generate_response_stream(state: AgentState) -> AsyncIterator[str]:
    """Generates streaming response."""
    logger.debug(f"Entering generate_response_stream")
    logger.debug(f"Final question: {state.question}")
    logger.debug(f"Documents for response: {len(state.documents) if state.documents else 0}")
    
    try:
        if not state.documents:
            yield "I couldn't find any relevant information to answer your question."
            return

        TOP_K_DOCUMENTS = 3
        top_documents = state.documents[:TOP_K_DOCUMENTS]
        context = "\n\n".join(doc.page_content for doc in top_documents)

        # Prepare the prompt (similar to your ANSWER_PROMPT) for streaming.
        answer_chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | ANSWER_PROMPT
            | llm_4o
            | StrOutputParser()
        )

        # Create an instance of your token callback handler.
        token_handler = TokenStreamHandler()

        stream = answer_chain.astream(
            {"question": state.question}, 
            config={"callbacks": [token_handler]}
        )

        # As tokens arrive, yield them immediately.
        final_response = ""
        try:
            async for token_info in stream:
                final_response += token_info.get("token", "")
                yield token_info.get("token", "")
        except GeneratorExit:
            logger.info("Stream generator exited by client (GeneratorExit)")
            raise

        # At the end, update the state (if you want to keep the final response in the agent state).
        state.response = final_response
        logger.debug("Starting response generation stream")
    except Exception as e:
        logger.exception("Error in generate_response_stream")
        raise

async def generate_response_stream_sse(state: AgentState) -> AsyncIterator[str]:
    """Generates streaming response in Server-Sent Events format."""
    async for token in generate_response_stream(state):
        yield f"data: {token}\n\n"
