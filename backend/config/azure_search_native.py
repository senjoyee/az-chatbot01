"""
Azure Search configuration using the native Azure AI Search SDK.
This replaces the langchain-based implementation with direct SDK calls.
"""

import logging
from azure.core.credentials import AzureKeyCredential
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
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType

from openai import AzureOpenAI

from .settings import (
    AZURE_SEARCH_SERVICE_ENDPOINT,
    AZURE_SEARCH_KEY,
    AZURE_SEARCH_INDEX,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
)

logger = logging.getLogger(__name__)

# Initialize the Azure OpenAI client for embeddings
openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2023-05-15",
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# Initialize the search clients
search_client_native = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

index_client_native = SearchIndexClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Define search fields (same as original implementation)
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
        type=SearchFieldDataType.String,
        filterable=True,
        facetable=True,
        searchable=True,
    ),
    SearchableField(
        name="customer",
        type=SearchFieldDataType.String,
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

# Define vector search configuration
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="myHnsw",
            parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 500,
                "metric": "cosine"
            }
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="myHnswProfile",
            algorithm_configuration_name="myHnsw",
        )
    ]
)

class AzureSearchEmbedding:
    """
    A class to generate embeddings using Azure OpenAI.
    This replaces the langchain embeddings with direct API calls.
    """
    
    def __init__(self, model="text-embedding-3-large", dimensions=1536):
        self.model = model
        self.dimensions = dimensions
        self.client = openai_client
    
    def embed_query(self, text):
        """Generate an embedding for a single text query."""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model,
                dimensions=self.dimensions
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def embed_documents(self, documents):
        """Generate embeddings for a list of documents."""
        try:
            response = self.client.embeddings.create(
                input=documents,
                model=self.model,
                dimensions=self.dimensions
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings for documents: {str(e)}")
            raise

class AzureSearchClient:
    """
    A class to handle Azure AI Search operations.
    This replaces the langchain vector store with direct SDK calls.
    """
    
    def __init__(self, search_client, index_client, embedding_service):
        self.search_client = search_client
        self.index_client = index_client
        self.embedding_service = embedding_service
        
    def ensure_index_exists(self):
        """Check if index exists and create if it doesn't."""
        try:
            self.index_client.get_index(AZURE_SEARCH_INDEX)
            logger.info(f"Index {AZURE_SEARCH_INDEX} already exists")
        except Exception:
            # Create index if it doesn't exist
            index = SearchIndex(
                name=AZURE_SEARCH_INDEX,
                fields=fields,
                scoring_profiles=[scoring_profile],
                semantic_settings=semantic_search,
                vector_search=vector_search
            )
            self.index_client.create_index(index)
            logger.info(f"Created index {AZURE_SEARCH_INDEX} with semantic configuration")
    
    def semantic_hybrid_search(self, query, k=10, filters=None):
        """
        Perform a semantic hybrid search using Azure AI Search.
        This replaces the langchain vector store's semantic_hybrid_search method.
        """
        try:
            # Generate embedding for the query
            query_vector = self.embedding_service.embed_query(query)
            
            # Set up search options
            vector_queries = [{"vector": query_vector, "k": k, "fields": "content_vector"}]
            
            # Perform the search
            results = self.search_client.search(
                search_text=query,
                filter=filters,
                query_type=QueryType.SEMANTIC,
                semantic_configuration_name="default-semantic-config",
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                top=k,
                vector_queries=vector_queries,
                scoring_profile="content_source_freshness_profile"
            )
            
            # Convert results to Document objects for compatibility
            from langchain_core.documents import Document
            documents = []
            
            for result in results:
                # Parse the metadata from JSON string
                import json
                metadata = json.loads(result["metadata"])
                
                # Create a Document object
                doc = Document(
                    page_content=result["content"],
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic hybrid search: {str(e)}")
            raise

# Initialize the embedding service
embeddings_native = AzureSearchEmbedding()

# Initialize the Azure Search client
azure_search_client = AzureSearchClient(
    search_client=search_client_native,
    index_client=index_client_native,
    embedding_service=embeddings_native
)

# Ensure the index exists
azure_search_client.ensure_index_exists()

# Export these instances for use in other modules
__all__ = [
    'azure_search_client', 
    'search_client_native', 
    'index_client_native', 
    'embeddings_native'
]
