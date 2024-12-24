import logging
import re
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import StrOutputParser

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT
)
from config.azure_search import vector_store
from models.schemas import Message
from services.grader import grade_document

logger = logging.getLogger(__name__)

# Initialize the language model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2023-03-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3
)

# Prompt templates
condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

# ----
# ENHANCED ANSWER TEMPLATE
# ----
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
    [2-3 sentence overview]

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

def condense_question(state: dict) -> dict:
    logger.info(f"Condensing question with state: {state}")
    chat_history_formatted = format_chat_history(state.get("chat_history", []))

    _input = (
        RunnablePassthrough()
        .assign(chat_history=lambda x: chat_history_formatted)
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser()
    )
    result = _input.invoke({
        "question": state.get("question"),
        "chat_history": chat_history_formatted
    })
    state["question"] = result  # Assign to 'question' directly
    return state

def retrieve_documents(state: dict) -> dict:
    logger.info(f"Retrieving documents for question: {state.get('question')}")
    state["documents"] = vector_store.hybrid_search(state["question"], k=5)
    return state

def grade_documents(state: dict) -> dict:
    logger.info(f"Grading documents for question: {state.get('question')}")

    graded_documents = []
    for doc in state.get("documents", []):
        score = grade_document(state.get("question"), doc.page_content)
        logger.info(f"Document ID {doc.metadata.get('id')}: Grade = {score}")
        if score == "yes":
            graded_documents.append(doc)

    state["graded_documents"] = graded_documents
    logger.info(f"Number of documents after grading: {len(graded_documents)}")
    return state

def generate_response(state: dict) -> dict:
    logger.info(f"Generating response with state: {state}")
    if "graded_documents" not in state or not state["graded_documents"]:
        logger.warning("No graded documents available for generating response.")
        state["response"] = "I'm sorry, but I couldn't find relevant information to answer your question."
        return state

    # Combine the content of graded documents into context
    context = "\n\n".join([doc.page_content for doc in state["graded_documents"]])
    logger.debug(f"Formatted context: {context}")

    _input = (
        RunnableLambda(lambda s: {"context": s["context"], "question": s["question"]})
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    state["response"] = _input.invoke({
        "context": context,
        "question": state.get("question")
    })
    logger.debug(f"Generated response: {state['response']}")
    return state

def update_history(state: dict) -> dict:
    logger.info(f"Updating history with state: {state}")
    state.setdefault("chat_history", []).extend([
        HumanMessage(content=state.get("question")),
        AIMessage(content=state.get("response"))
    ])
    return state

# Build the Langgraph
builder = StateGraph("agent_state")

# Add nodes
builder.add_node("condense_question", condense_question)
builder.add_node("retrieve_documents", retrieve_documents)
builder.add_node("grade_documents", grade_documents)
builder.add_node("generate_response", generate_response)
builder.add_node("update_history", update_history)

# Add edges
builder.set_entry_point("condense_question")
builder.add_edge("condense_question", "retrieve_documents")
builder.add_edge("retrieve_documents", "grade_documents")
builder.add_edge("grade_documents", "generate_response")
builder.add_edge("generate_response", "update_history")
builder.add_edge("update_history", END)

# Compile the graph
agent = builder.compile()

async def run_agent(question: str, chat_history: List[Message]) -> Dict[str, Any]:
    """Runs the Langgraph agent with grading step."""
    inputs = {
        "question": question,
        "chat_history": chat_history,
        "documents": None,
        "graded_documents": None,
        "response": None
    }
    result = await agent.ainvoke(inputs)
    return result  # Already a dict