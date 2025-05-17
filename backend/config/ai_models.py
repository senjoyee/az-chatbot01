import os
from langchain_openai import AzureChatOpenAI

# Base model definitions (no temperature)
llm_41_mini = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    openai_api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SC"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_SC")
)

llm_41_nano = AzureChatOpenAI(
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_SC"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY_SC")
)
