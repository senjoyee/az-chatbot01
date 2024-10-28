import logging
import os
import requests
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import AzureBlobStorageContainerLoader, Docx2txtLoader
from langchain_openai import AzureChatOpenAI
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import TextSplitter

app = func.FunctionApp()

# Environment variables
FASTAPI_ENDPOINT = os.getenv("FASTAPI_INDEX_DOCUMENTS_ENDPOINT")
BLOB_CONN_STRING = os.getenv("BLOB_CONN_STRING")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER")

if not FASTAPI_ENDPOINT:
  raise ValueError("FASTAPI_INDEX_DOCUMENTS_ENDPOINT environment variable is not set.")

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

class GPTSplitter(TextSplitter):
  def __init__(self, model_name: str = "gpt-4o-mini", **kwargs: Any) -> None:
      super().__init__(**kwargs)
      self.model = AzureChatOpenAI(model=model_name,api_version="2023-03-15-preview",temperature=0)

      self.prompt = ChatPromptTemplate.from_template(
          """You are an expert in analyzing and structuring various types of documents. Your task is to divide the given text into coherent, meaningful chunks that preserve both the document's structure and its detailed content. Each chunk should encapsulate a complete section or logical unit, including all relevant details, explanations, and supporting information.

          Follow these guidelines:
          1. Preserve the document's structure (e.g., main sections, subsections) and include all content under each header.
          2. Keep related information together (e.g., lists, tables, explanations, examples).
          3. Ensure each chunk is self-contained and includes all necessary context and details.
          4. Aim for chunks that cover complete sections or logical groupings of information.
          5. Include all text content, not just headers or main points.
          6. Maintain the original formatting and structure within each chunk.

          Wrap each chunk in <<<>>> markers.

          Example:
          <<<2. Product Overview
          2.1. Key Features
          • Feature A: Description of Feature A
          • Feature B: Description of Feature B
          [Include all features and their descriptions]

          2.2. Benefits
          • Benefit 1: Explanation of Benefit 1
          • Benefit 2: Explanation of Benefit 2
          [Include all benefits and their explanations]

          2.3. Use Cases
          • Use Case 1: Description and example of Use Case 1
          • Use Case 2: Description and example of Use Case 2
          [Include all use cases with their descriptions and examples]>>>

          Now, process the following text, ensuring to include ALL content and details:

          {text}
          """
      )
      self.output_parser = StrOutputParser()
      self.chain = (
          {"text": RunnablePassthrough()}
          | self.prompt
          | self.model
          | self.output_parser
      )

  def split_text(self, text: str) -> List[Document]:
      chunks = self.chain.invoke(text).split("<<<")[1:]
      return [Document(page_content=chunk.strip(">>>").strip()) for chunk in chunks]

def process_blob_document(blob_client):
  # Download blob content
  download_stream = blob_client.download_blob()
  content = download_stream.readall()
  
  # Create a temporary file to store the content
  temp_file_path = f"/tmp/{blob_client.blob_name}"
  os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
  
  with open(temp_file_path, "wb") as temp_file:
      temp_file.write(content)
  
  # Process the document
  loader = Docx2txtLoader(temp_file_path)
  data = loader.load()
  
  # Split using GPT
  gpt_splitter = GPTSplitter()
  split_documents = gpt_splitter.split_text(data[0].page_content)
  
  # Add metadata
  for doc in split_documents:
      doc.metadata["source"] = blob_client.blob_name
  
  # Clean up temporary file
  os.remove(temp_file_path)
  
  return split_documents

@app.function_name(name="myblobtrigger")
@app.event_grid_trigger(arg_name="event")
def eventGridTest(event: func.EventGridEvent):
  try:
      # Get blob information from the event
      data = event.get_json()
      blob_url = data['url']
      container_client = blob_service_client.get_container_client(CONTAINER_NAME)
      blob_name = blob_url.split('/')[-1]
      blob_client = container_client.get_blob_client(blob_name)

      # Process the document
      split_documents = process_blob_document(blob_client)

      logging.info(f"Document count: {len(split_documents)}")
      
      # Prepare documents for the API
      documents_in = [
          {
              "page_content": doc.page_content,
              "metadata": {"source": doc.metadata["source"]},
          }
          for doc in split_documents
      ]

      # Send to FastAPI endpoint
      response = requests.post(FASTAPI_ENDPOINT, json=documents_in)

      if response.status_code == 200:
          logging.info("Documents sent successfully to FastAPI endpoint.")
      else:
          logging.error(
              f"Failed to send documents. Status Code: {response.status_code} Response: {response.text}"
          )

  except Exception as e:
      logging.error(f"Error processing document: {str(e)}")
      raise