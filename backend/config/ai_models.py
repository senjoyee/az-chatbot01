import os
from langchain_openai import AzureChatOpenAI

# gpt-4.1-mini for conversation agent
llm_conversation_agent = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    openai_api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SC"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_SC"),
    temperature=0.3
)

# gpt-4.1-nano for summarization
llm_summarizer = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SC"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_SC"),
    temperature=0.3
)

# gpt-4.1-nano for contextualizer
llm_contextualizer = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SC"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_SC"),
    temperature=0.3
)

# gpt-4.1-nano for mindmap
llm_mindmap = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SC"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_SC"),
    temperature=0.3
)
