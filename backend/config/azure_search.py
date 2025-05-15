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
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSearch,
    SemanticPrioritizedFields,
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
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=True,
        filterable=True,
    ),
    SearchableField(
        name="source",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        filterable=True,
        facetable=True,
        searchable=True,
    ),
    SearchableField(
        name="customer",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        filterable=True,
        facetable=True,
        searchable=True,
    ),
    SimpleField(  
        name="last_update",  
        type=SearchFieldDataType.DateTimeOffset,  
        filterable=True,  
        sortable=True,  
    ), 
    SimpleField(
        name="contextualized",
        type=SearchFieldDataType.Boolean,
        filterable=True,
        sortable=False,
    )
]

# Define scoring profile
scoring_profile = ScoringProfile(  
    name="content_source_freshness_profile",  
    text_weights=TextWeights(weights={  
        "content": 5,
        "customer": 10
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

# Initialize the vector store instance
vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=AZURE_SEARCH_INDEX,
    embedding_function=embedding_function,
    fields=fields,
    scoring_profiles=[scoring_profile],
    default_scoring_profile="content_source_freshness_profile",
    semantic_configuration_name="default-semantic-config"
)

# Initialize the search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Initialize the index client
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Check if index exists and create if it doesn't
try:
    index_client.get_index(AZURE_SEARCH_INDEX)
except Exception:
    # Create index if it doesn't exist
    
    # Define semantic configuration
    semantic_config = SemanticConfiguration(
        name="default-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="source"),
            content_fields=[
                SemanticField(field_name="content")
            ],
            keywords_fields=[
                SemanticField(field_name="metadata"),
                SemanticField(field_name="customer")
            ]
        )
    )
    
    # Create semantic settings with the configuration
    semantic_search = SemanticSearch(
        configurations=[semantic_config],
        default_configuration="default-semantic-config"
    )
    
    # Create index with semantic settings
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX,
        fields=fields,
        scoring_profiles=[scoring_profile],
        semantic_settings=semantic_search
    )
    index_client.create_index(index)
    print(f"Created index {AZURE_SEARCH_INDEX} with semantic configuration")

# Export these instances
__all__ = ['vector_store', 'search_client', 'index_client', 'embeddings']
