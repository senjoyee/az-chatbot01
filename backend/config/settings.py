"""
Application settings and environment variables configuration.
Maintains the exact same environment variables as the original implementation.
"""

import os
from dotenv import find_dotenv, load_dotenv

# Load environment variables (exactly as in original implementation)
load_dotenv(find_dotenv())

# Azure Search settings
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

# Azure OpenAI settings
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Azure Blob Storage settings
BLOB_CONN_STRING = os.getenv("BLOB_CONN_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER")

# Construct Azure Search endpoint (exactly as in original implementation)
AZURE_SEARCH_SERVICE_ENDPOINT = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"

# Web Search API key
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
