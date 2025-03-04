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
You are a helpful assistant providing information based on the documents provided. Please answer the question using only the information in the documents.

<documents>
{context}
</documents>

Question: {question}

Instructions:
1. Take a moment to understand the core of what is being asked.
2. Identify the key points that need to be addressed in your response.
3. Organize your thoughts to provide a clear and logical answer.
4. Ensure your response directly answers the question without unnecessary elaboration.
5. Answer based only on the information in the documents above
6. If you cannot provide a reasonable answer based solely on the documents provided, respond with: "I don't have enough information in my knowledge base to answer this question properly."
7. Format your answer within <answer> tags

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
