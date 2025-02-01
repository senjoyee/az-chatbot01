# (Imports remain unchanged)
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ... existing imports ...

# NEW NODE: Handle Greetings
def handle_greetings(state: AgentState) -> AgentState:
    # If a response is already set, skip further processing.
    if state.response:
        return state

    greeting_keywords = ["hello", "hi", "hey", "how are you", "greetings"]
    question_lower = state.question.lower() if state.question else ""
    if any(greet in question_lower for greet in greeting_keywords):
        # Generate a friendly greeting using the LLM.
        greeting_prompt = f"Generate a friendly greeting response for the message: '{state.question}'."
        immediate_response = llm_4o_mini(greeting_prompt)
        state.response = immediate_response.strip()
    return state

# NEW NODE: Check if the question is customer-specific
def check_customer_question(state: AgentState) -> AgentState:
    if state.response:
        return state

    # Check if a customer is mentioned explicitly.
    customers = detect_customers(state.question)
    if not customers:
        # Use LLM to determine if the question is customer-related.
        prompt = f"Does the following question require filtering documents by a specific customer? Answer yes or no.\n\nQuestion: {state.question}"
        answer = llm_4o_mini(prompt)
        if answer.strip().lower().startswith("yes"):
            state.response = "It appears your question might be related to a specific customer. Could you please specify the customer name?"
    return state

# Modify existing nodes to include a guard clause so that if a response already exists, they simply pass through.
def condense_question(state: AgentState) -> AgentState:
    if state.response:
        return state
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
    if state.response:
        return state
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
    if state.response:
        return state
    logger.info(f"Retrieving documents for question: {state.question}")
    try:
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
        logger.info(f"Retrieved {len(state.documents)} documents")
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        state.documents = []
    return state

def rerank_documents(state: AgentState) -> AgentState:
    if state.response:
        return state
    logger.info("Reranking documents")
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
    return state

def generate_response(state: AgentState) -> AgentState:
    if state.response:
        return state
    logger.info("Generating response")
    if not state.documents:
        state.response = "I couldn't find any relevant information to answer your question."
        return state
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
    response = response.replace("<answer>", "").replace("</answer>", "").strip()
    state.response = response
    return state

# Update the graph to include the new nodes.
builder = StateGraph(AgentState)
builder.add_node("handle_greetings", handle_greetings)
builder.add_node("condense", condense_question)
builder.add_node("reason", reason_about_query)
builder.add_node("check_customer", check_customer_question)
builder.add_node("retrieve", retrieve_documents)
builder.add_node("rerank", rerank_documents)
builder.add_node("generate", generate_response)
builder.add_node("update_history", update_history)

# Set up the new flow by updating the edges.
builder.add_edge("handle_greetings", "condense")
builder.add_edge("condense", "reason")
builder.add_edge("reason", "check_customer")
builder.add_edge("check_customer", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_edge("rerank", "generate")
builder.add_edge("generate", "update_history")
builder.add_edge("update_history", END)

builder.set_entry_point("handle_greetings")
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