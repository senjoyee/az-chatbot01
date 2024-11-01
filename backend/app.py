import logging
import os
from operator import itemgetter

from azure.storage.blob import BlobServiceClient
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document, StrOutputParser, format_document
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain_community.document_loaders import Docx2txtLoader
from langchain_postgres import PGVector
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import Optional

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

conn_str = os.getenv("BLOB_CONN_STRING")
container_name = os.getenv("BLOB_CONTAINER")
blob_service_client = BlobServiceClient.from_connection_string(conn_str=conn_str)

host = os.getenv("PG_VECTOR_HOST")
user = os.getenv("PG_VECTOR_USER")
password = os.getenv("PG_VECTOR_PASSWORD")
COLLECTION_NAME = os.getenv("PGDATABASE")
CONNECTION_STRING = (
    f"postgresql+psycopg2://{user}:{password}@{host}:5432/{COLLECTION_NAME}"
)

namespace = f"pgvector/{COLLECTION_NAME}"
record_manager = SQLRecordManager(namespace, db_url=CONNECTION_STRING)
record_manager.create_schema()

embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large",dimensions="1536")

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 20})


class Message(BaseModel):
  role: str  # 'user' or 'bot'
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


def _format_chat_history(conversation: list[Message]) -> str:
    formatted_history = ""
    for message in conversation:
        formatted_history += f"{message.role}: {message.content}\n"
    return formatted_history.rstrip()


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version="2023-03-15-preview",
    temperature=0)

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
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | llm | StrOutputParser()

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["https://documentchatbot01.azurewebsites.net"],  # Frontend URL
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
  expose_headers=["*"],
)


@app.get("/test")
async def test():
    return {"test": "works"}


def get_row_count():
    engine = create_engine(CONNECTION_STRING)
    with engine.connect() as connection:
        result = connection.execute(text("SELECT COUNT(*) FROM langchain_pg_embedding"))
        row_count = result.scalar()
    return row_count


@app.get("/row_count")
async def row_count():
    try:
        count = get_row_count()
        return {"row_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
async def list_files(page: int = 1, page_size: int = 10):
  try:
      container_client = blob_service_client.get_container_client(container_name)
      blob_list = list(container_client.list_blobs())  # Convert to list to materialize it
      
      files = [{
          "name": blob.name,
          "size": blob.size,
          "lastModified": blob.last_modified.isoformat() if blob.last_modified else None,
          "contentType": blob.content_settings.content_type if blob.content_settings else None
      } for blob in blob_list]
      
      total_files = len(files)
      start = (page - 1) * page_size
      end = start + page_size
      
      response_data = {
          "total_files": total_files,
          "files": files[start:end],
          "page": page,
          "total_pages": (total_files - 1) // page_size + 1,
      }
      
      logger.info(f"Returning files response: {response_data}")
      return response_data
      
  except Exception as e:
      logger.error(f"Error in list_files: {str(e)}")
      raise HTTPException(status_code=500, detail=str(e))

@app.delete("/deletefile/{filename}")
async def delete_file(filename: str):
  logger.info(f"Attempting to delete file: {filename}")
  container_client = blob_service_client.get_container_client(container_name)
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
  # Add size validation
  MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
  
  container_client = blob_service_client.get_container_client(container_name)
  uploaded_files = []

  for file in files:
      # Validate file size
      contents = await file.read()
      if len(contents) > MAX_FILE_SIZE:
          raise HTTPException(
              status_code=413,
              detail=f"File {file.filename} exceeds maximum size of {MAX_FILE_SIZE/(1024*1024)}MB"
          )
          
      blob_client = container_client.get_blob_client(blob=file.filename)
      blob_client.upload_blob(contents, overwrite=True)
      uploaded_files.append(file.filename)

  return {"uploaded_files": uploaded_files}

@app.post("/index_documents/")
async def index_documents(documents_in: list[DocumentIn]):
    print(f"Received documents for indexing: {documents_in}")  # Debug logging
    try:
        documents = [
            Document(
                page_content=doc.page_content,
                metadata={
                    "customer_name": doc.metadata.get("customer_name", "Unknown")
                }
            )
            for doc in documents_in
        ]
        
        logging.info(f"Processing {len(documents)} documents for indexing")
        logging.info(f"First document metadata example: {documents[0].metadata if documents else 'No documents'}")
        
        result = index(
            documents,
            record_manager,
            vector_store,
            cleanup="full",
            source_id_key="customer_name",  # Changed to use customer_name
        )
        return result
    except Exception as e:
        logging.error(f"Error during document indexing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
