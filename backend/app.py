import os
import logging
from operator import itemgetter
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document, StrOutputParser, format_document
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AzureBlobStorageContainerLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import AzureChatOpenAI
from azure.storage.blob import BlobServiceClient
from dotenv import find_dotenv, load_dotenv
import json
import sys

# Load environment variables
load_dotenv(find_dotenv())

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Set logging levels for Azure SDK components
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)
logging.getLogger("azure.search").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Environment variables
AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
BLOB_CONN_STRING = os.getenv("BLOB_CONN_STRING")
BLOB_CONTAINER = os.getenv("BLOB_CONTAINER")

# Construct Azure Search endpoint
AZURE_SEARCH_SERVICE_ENDPOINT = f"https://{AZURE_SEARCH_SERVICE}.search.windows.net"

# Constants
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 200

# Initialize embeddings
embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large",dimensions=1536)

embedding_function = embeddings.embed_query


# Define fields for Azure Search
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
    ),
    # Additional field to store the title
    SearchableField(
        name="title",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
    ),
]

index_name: str = os.getenv("AZURE_SEARCH_INDEX")

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    azure_search_key=AZURE_SEARCH_KEY,
    index_name=AZURE_SEARCH_INDEX,
    embedding_function=embedding_function,
    fields=fields,
)

# Initialize search client for direct operations if needed
search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)

# Check if index exists and create if it doesn't
try:
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )
    index_client.get_index(AZURE_SEARCH_INDEX)
    logger.info(f"Index {AZURE_SEARCH_INDEX} already exists")
except Exception as e:
    logger.info(f"Index {AZURE_SEARCH_INDEX} does not exist. Creating...")
    logger.info("Index will be created automatically by Langchain")

# Pydantic Models
class Message(BaseModel):
    role: str
    content: str
    id: Optional[int] = None

class Conversation(BaseModel):
    conversation: list[Message]

class ConversationRequest(BaseModel):
    question: str
    conversation: Conversation

class DocumentIn(BaseModel):
    page_content: str
    metadata: dict = Field(default_factory=dict)

# Initialize the chat model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    openai_api_version="2023-03-15-preview",
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=0.3
)

# Initialize the retriever from vector store
retriever = vector_store.as_retriever(
    search_type="hybrid",
    search_kwargs={"k": 4}  # Fetch top 4 most relevant documents
)

condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_question_template)

answer_template = """Please provide an answer based strictly on the following context:
<context>
{context}
</context>

Guidelines for answering:
1. I will analyze the question and context thoroughly
2. Present information in a clear, structured format
3. Use numbered points for main ideas
4. Use a, b for sub-points instead of dashes 
5. Ensure all points are supported by the context
6. Clearly indicate if information is incomplete
7. Maintain consistency throughout the response

Question: {question}

Answer:
[If sufficient information exists in context]
1. [First main point]
    a. [Supporting detail]
    b. [Supporting detail]
2. [Second main point]
    a. [Supporting detail]
    etc.

[If information is incomplete or missing]
Based on the provided context, here is the relevant information that is available:
1. [Available information point]
    a. [Available detail]
2. [Available information point]
    a. [Available detail]

[If no relevant information exists]
The provided context does not contain information to answer this question.
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(answer_template)


DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    logger.info(f"Docstrings: {doc_strings}")
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history):
    buffer = []
    for message in chat_history:
        if message.role == "user":
            buffer.append(f"Human: {message.content}")
        elif message.role == "assistant":
            buffer.append(f"Assistant: {message.content}")
    return "\n".join(buffer)


_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

# Combine the chain components
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://documentchatbot01.azurewebsites.net",  # Frontend URL
        "https://jsragfunc01.azurewebsites.net",        # Function App URL
    ],
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE"],  # Specify allowed methods
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize Azure Blob Storage client
blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

@app.get("/test")
async def test():
    return {"test": "works"}


@app.post("/conversation")
async def ask_question(request: ConversationRequest) -> dict:
  try:
      question = request.question
      conversation = request.conversation
      
      # Debug logging
      logger.info(f"Received question: {question}")
      logger.info(f"Received conversation: {conversation}")

      # Use the messages directly without transformation
      chat_history = conversation.conversation
      
      logger.info(f"Using chat history: {chat_history}")
      
      answer = conversational_qa_chain.invoke({
          "question": question,
          "chat_history": chat_history
      })
      
      return {
          "answer": answer,
          "status": "success"
      }
  except Exception as e:
      logger.error(f"Conversation error: {str(e)}")
      raise HTTPException(status_code=500, detail=str(e))


@app.get("/listfiles")
async def list_files(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    try:
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
        blob_list = list(container_client.list_blobs())
        
        # Calculate pagination
        start = (page - 1) * page_size
        end = start + page_size
        
        files = [{
            "name": blob.name,
            "size": blob.size,
            "lastModified": blob.last_modified.isoformat(),
            "contentType": blob.content_settings.content_type
        } for blob in blob_list]
        
        return {
            "total_files": len(files),
            "files": files[start:end],
            "page": page,
            "total_pages": (len(files) - 1) // page_size + 1,
        }
    except Exception as e:
        logger.error(f"Error in list_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/deletefile/{filename}")
async def delete_file(filename: str):
  logger.info(f"Attempting to delete file: {filename}")
  container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
  blob_client = container_client.get_blob_client(blob=filename)

  try:
      # Check if blob exists first
      if not blob_client.exists():
          logger.error(f"File {filename} not found")
          raise HTTPException(status_code=404, detail=f"File {filename} not found")
          
      blob_client.delete_blob()
      logger.info(f"Successfully deleted file: {filename}")
      return {"message": f"File {filename} deleted successfully"}
  except Exception as e:
      logger.error(f"Error deleting file {filename}: {str(e)}")
      raise HTTPException(status_code=500, detail=str(e))


@app.post("/uploadfiles")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        # Add size validation
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
        
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER)
        uploaded_files = []

        for file in files:
            # Validate file size
            contents = await file.read()
            if len(contents) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE/(1024*1024)}MB"
                )
                
            try:
                blob_client = container_client.get_blob_client(blob=file.filename)
                blob_client.upload_blob(contents, overwrite=True)
                uploaded_files.append(file.filename)
            except Exception as e:
                logger.error(f"Error uploading file {file.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error uploading file {file.filename}: {str(e)}")

        return {"uploaded_files": uploaded_files}
    except Exception as e:
        logger.error(f"Error in upload_files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_uploaded_files")
async def process_uploaded_files():
    try:
        loader = AzureBlobStorageContainerLoader(
            conn_str=BLOB_CONN_STRING,
            container=BLOB_CONTAINER
        )
        
        data = loader.load()
        logger.info(f"Loaded {len(data)} documents from blob storage")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        split_documents = text_splitter.split_documents(data)
        logger.info(f"Split into {len(split_documents)} chunks")
        
        batch_size = 100
        total_processed = 0
        
        if len(split_documents) > 0:
            for i in range(0, len(split_documents), batch_size):
                batch = split_documents[i:i + batch_size]
                documents_in = [
                    DocumentIn(
                        page_content=str(doc.page_content),
                        metadata={
                            "source": os.path.basename(str(doc.metadata.get("source", "")))
                        }
                    )
                    for doc in batch
                ]
                
                # Index with full cleanup mode
                await index_documents(documents_in)
                total_processed += len(batch)
                logger.info(f"Processed {total_processed}/{len(split_documents)} documents")
        else:
            # If no documents, call index_documents with empty list to trigger cleanup
            logger.info("No documents found in blob storage, cleaning up indexes")
            await index_documents([])

        return {
            "message": "Documents processed successfully",
            "document_count": len(split_documents),
            "total_processed": total_processed
        }
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        logger.error(f"Document data: {data if 'data' in locals() else 'Not loaded'}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index_documents/")
async def index_documents(documents_in: list[DocumentIn]):
    logging.debug('Starting document indexing process')
    try:
        for document in documents_in:
            logging.debug('Processing document: %s', document.metadata['source'])
            # Assuming `get_document_embedding` is a function that generates embeddings
            try:
                embedding = embeddings.embed_query(document.page_content)
                logging.debug('Generated embedding for document: %s', document.metadata['source'])
            except Exception as e:
                logging.error('Error generating embedding for document: %s', str(e))
                continue
            
            # Create a unique ID for each document
            doc_id = f"doc_{hash(document.page_content)}"
            
            # Create the document object
            document_obj = {
                "id": doc_id,
                "content": document.page_content,
                "content_vector": embedding,
                "metadata": json.dumps(document.metadata),
                "source": document.metadata.get("source", "unknown")
            }
            
            # Index document in Azure Search
            logging.debug('Indexing document into Azure Search: %s', document.metadata['source'])
            try:
                result = search_client.upload_documents(documents=[document_obj])
                logging.debug('Indexed document into Azure Search: %s', document.metadata['source'])
            except Exception as e:
                logging.error('Error indexing document: %s', str(e))
                raise
            logging.debug('Successfully indexed document: %s', document.metadata['source'])
    except Exception as e:
        logging.error('Error during document indexing: %s', str(e))
        raise
    logging.debug('Completed document indexing process')
    return {"message": f"Successfully indexed {len(documents_in)} documents"}
