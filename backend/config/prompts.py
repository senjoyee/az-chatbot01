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
You are an AI assistant designed for a retrieval augmented generation application. Your task is to answer questions based on the provided context. Your answers should be precise, to the point, and use limited jargon. When appropriate, provide answers in bullet points.

Here is the context you should use to answer the question:

<documents>
{context}
</documents>

Now, please answer the following question:

<user_question>
{question}
</user_question>

When formulating your answer:
1. Carefully analyze the context and the question.
2. Provide a clear and concise answer based solely on the information given in the context.
3. If the answer requires multiple points, use bullet points for clarity.
4. Avoid using technical jargon unless it's absolutely necessary for accuracy.
5. If the question cannot be answered based on the given context, state this clearly.

Before providing any specific outputs (such as yes/no answers or numerical values), always explain your reasoning first.

Your final output should be structured as follows:
<answer>
[Your answer here, following the guidelines above]
</answer>

Remember, your final output should only include the content within the <answer> tags. Do not include any of your thought process or additional commentary outside of these tags.
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
