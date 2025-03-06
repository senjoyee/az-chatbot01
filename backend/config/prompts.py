"""
Prompt templates for the chatbot application.
"""
from langchain.prompts import PromptTemplate

# Prompt for condensing a follow-up question with chat history into a standalone question
CONDENSE_QUESTION_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

# Prompt for generating an answer based on retrieved documents
ANSWER_TEMPLATE = """
You are a helpful assistant that answers questions based solely on the documents provided.

<documents>
{context}
</documents>

Question: {question}

Instructions:
- Read the documents and answer the question using only the provided information.
- Be direct and concise.
- If the documents lack sufficient information, respond with "I don't have enough information in my knowledge base to answer this question properly."
- Output your answer in MARKDOWN format, enclosed within <answer> tags, using proper markdown formatting without extra blank lines.

<answer>
"""

# Prompt for casual conversation without document context
CONVERSATION_TEMPLATE = """You are a friendly and helpful AI assistant. Respond to the following message in a natural, conversational way.
If there is chat history, maintain a consistent and contextual conversation.

Chat History:
{history}

User Message:
{message}

Your response should be brief and friendly."""

# Prompt for deciding if documents are sufficient to answer a question
DECISION_TEMPLATE = """Given the following question and document excerpts, determine if a reasonable answer can be generated.

<question>
{question}
</question>

<documents>
{context}
</documents>

Respond with 'yes' if a reasonable answer can be generated, or 'no' if not.
"""

# Create PromptTemplates from the templates
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
ANSWER_PROMPT = PromptTemplate.from_template(ANSWER_TEMPLATE)
CONVERSATION_PROMPT = PromptTemplate.from_template(CONVERSATION_TEMPLATE)
DECISION_PROMPT = PromptTemplate.from_template(DECISION_TEMPLATE)
