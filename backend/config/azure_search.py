"""
Azure Search configuration and initialization.
Maintains the exact same configuration as the original implementation.
"""

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings

from .settings import (
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_KEY,
    AZURE_SEARCH_INDEX,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)

# Initialize embeddings (exactly as in original implementation)
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1536
)

# Define the embedding function
embedding_function = embeddings.embed_query

# Define search fields (exactly as in original implementation)
fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        analyzer_name="standard.lucene",
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        hidden=False,
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_configuration="default",
    ),
    SimpleField(
        name="metadata",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
    SearchableField(
        name="source",
        type=SearchFieldDataType.String,
        analyzer_name="standard.lucene",
    ),
    SimpleField(
        name="last_update",
        type=SearchFieldDataType.DateTimeOffset,
        filterable=True,
        sortable=True,
    ),
]

# Define scoring profile (exactly as in original implementation)
scoring_profile = ScoringProfile(
    name="content_source_freshness_profile",
    text_weights=TextWeights(weights={
        "content": 5,  # Highest weight for content
        "source": 3    # Lower weight for source
    }),
    function_aggregation="sum",
    functions=[
        FreshnessScoringFunction(
            field_name="last_update",
            boost=100,
            parameters=FreshnessScoringParameters(boosting_duration="P15D"),
            interpolation="linear"
        )
    ]
)

def init_vector_store() -> AzureSearch:
    """Initialize and return the Azure Search vector store with the original configuration."""
    return AzureSearch(
        azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        azure_search_key=AZURE_SEARCH_KEY,
        index_name=AZURE_SEARCH_INDEX,
        embedding_function=embedding_function,
        fields=fields,
        scoring_profiles=[scoring_profile],
        default_scoring_profile="content_source_freshness_profile"
    )

def init_search_client() -> SearchClient:
    """Initialize and return the Azure Search client."""
    return SearchClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

def init_index_client() -> SearchIndexClient:
    """Initialize and return the Azure Search index client."""
    return SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )
