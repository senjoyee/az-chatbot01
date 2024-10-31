import logging
import os
import requests
import tempfile
import hashlib
import re
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import azure.functions as func
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import AzureChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters.base import TextSplitter
from langchain_core.document_loaders import BaseLoader
import pymupdf4llm

app = func.FunctionApp()

FASTAPI_ENDPOINT = os.getenv("FASTAPI_INDEX_DOCUMENTS_ENDPOINT")
BLOB_CONN_STRING = os.getenv("BLOB_CONN_STRING")
CONTAINER_NAME = os.getenv("BLOB_CONTAINER")

if not FASTAPI_ENDPOINT:
    raise ValueError("FASTAPI_INDEX_DOCUMENTS_ENDPOINT environment variable is not set.")

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONN_STRING)

class PyMuPDF4LLMMarkdownLoader(BaseLoader):
    """Custom document loader for extracting markdown-formatted text from PDF files."""
    
    def __init__(
        self,
        file_path: Union[str, Path],
        pages: Optional[List[int]] = None,
        page_chunks: bool = True,
        extract_images: bool = False,
        image_path: Optional[str] = None,
        image_format: str = "png",
        dpi: int = 300,
        extract_words: bool = False
    ):
        self.file_path = Path(file_path)
        self.pages = pages
        self.page_chunks = page_chunks
        self.extract_images = extract_images
        self.image_path = image_path
        self.image_format = image_format
        self.dpi = dpi
        self.extract_words = extract_words
        
        if self.extract_images and not self.image_path:
            raise ValueError("image_path must be provided when extract_images=True")

    def load(self) -> List[Document]:
        """Load and process the PDF file."""
        content = pymupdf4llm.to_markdown(
            doc=str(self.file_path),
            pages=self.pages,
            page_chunks=self.page_chunks,
            write_images=self.extract_images,
            image_path=self.image_path,
            image_format=self.image_format,
            dpi=self.dpi,
            extract_words=self.extract_words
        )
        
        documents = []
        if self.page_chunks:
            for page_num, page_content in enumerate(content):
                doc = Document(
                    page_content=page_content["text"],
                    metadata={}  # Empty metadata, will be set by GPTSplitter
                )
                documents.append(doc)
        else:
            doc = Document(
                page_content=content,
                metadata={}  # Empty metadata, will be set by GPTSplitter
            )
            documents.append(doc)
                
        return documents

class GPTSplitter(TextSplitter):
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini", 
        temperature: float = 0.0,
        cache: Optional[Dict[str, List[str]]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.model = AzureChatOpenAI(
            model=model_name,
            api_version="2023-03-15-preview",
            temperature=temperature
        )
        self._cache = cache or {}
        
        self.split_prompt = ChatPromptTemplate.from_template(
            """You are an expert document splitter. Split the following text into logical chunks, preserving complete sections and context.
            Each chunk should be a complete logical unit.
            Use consistent splitting criteria like headers or major topic changes.
            Mark chunk boundaries with <<<>>> markers.
            Maintain original formatting.

            Text to split:
            {text}
            """
        )

        self.customer_prompt = ChatPromptTemplate.from_template(
            """Extract the customer company name from the text. If multiple variations exist, use the formal name.
            Common patterns:
            - "Customer: [Name]"
            - "[Name] has engaged"
            - "On behalf of [Name]"
            - References to "the Customer [Name]"

            Return ONLY the company name, nothing else.
            If no customer is found, return "Unknown".

            Text:
            {text}
            """
        )
        
        self.output_parser = StrOutputParser()
        
        self.split_chain = (
            {"text": RunnablePassthrough()}
            | self.split_prompt
            | self.model
            | self.output_parser
        )
        
        self.customer_chain = (
            {"text": RunnablePassthrough()}
            | self.customer_prompt
            | self.model
            | self.output_parser
        )

    def _detect_customer(self, text: str) -> str:
        try:
            customer_name = self.customer_chain.invoke(text[:2000]).strip()
            return customer_name if customer_name else "Unknown"
        except Exception as e:
            logging.error(f"Customer detection failed: {e}")
            return "Unknown"

    def _normalize_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\n+', '\n', text)
        return text

    def _get_cache_key(self, text: str) -> str:
        normalized = self._normalize_text(text)
        return hashlib.md5(normalized.encode()).hexdigest()

    def split_text(self, text: str) -> List[Document]:
        cache_key = self._get_cache_key(text)
        customer_name = self._detect_customer(text)
        
        if cache_key in self._cache:
            return [
                Document(
                    page_content=chunk, 
                    metadata={"customer_name": customer_name}
                ) for chunk in self._cache[cache_key]
            ]
        
        normalized_text = self._normalize_text(text)
        
        try:
            chunks = self.split_chain.invoke(normalized_text).split("<<<")[1:]
            chunks = [chunk.strip(">>>").strip() for chunk in chunks]
            self._cache[cache_key] = chunks
            
            return [
                Document(
                    page_content=chunk, 
                    metadata={"customer_name": customer_name}
                ) for chunk in chunks
            ]
            
        except Exception as e:
            logging.error(f"GPT splitting failed: {e}")
            chunks = self._fallback_split(normalized_text)
            self._cache[cache_key] = chunks
            return [
                Document(
                    page_content=chunk, 
                    metadata={"customer_name": customer_name}
                ) for chunk in chunks
            ]
    
    def _fallback_split(self, text: str) -> List[str]:
        header_pattern = r'\n\d+\.\s+[A-Z]|\n[A-Z][^a-z\n]+\n'
        chunks = re.split(header_pattern, text)
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        return chunks

@app.function_name(name="blobtrigger")
@app.event_grid_trigger(arg_name="event")
def eventGridTest(event: func.EventGridEvent):
    try:
        blob_url = event.get_json().get('url')
        if not blob_url:
            raise ValueError("No blob URL found in event")

        blob_name = blob_url.split('/')[-1]
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        # Download blob content
        download_stream = blob_client.download_blob()
        content = download_stream.readall()
        
        # Process based on file type
        file_extension = os.path.splitext(blob_name)[1].lower()
        documents = []
        
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
            
            try:
                if file_extension == '.pdf':
                    loader = PyMuPDF4LLMMarkdownLoader(temp_path)
                    docs = loader.load()
                elif file_extension in ['.docx', '.doc']:
                    loader = Docx2txtLoader(temp_path)
                    docs = loader.load()
                else:
                    logging.warning(f"Unsupported file type: {file_extension}")
                    return
                
                gpt_splitter = GPTSplitter()
                for doc in docs:
                    split_docs = gpt_splitter.split_text(doc.page_content)
                    documents.extend(split_docs)
                
            finally:
                os.unlink(temp_path)

        if not documents:
            logging.warning("No documents were processed")
            return

        logging.info(f"Processed {len(documents)} document chunks")
        
        # Prepare documents for API
        documents_in = [
            {
                "page_content": doc.page_content,
                "metadata": {
                    "customer_name": doc.metadata.get("customer_name", "Unknown")
                }
            }
            for doc in documents
        ]

        # Send to FastAPI endpoint
        response = requests.post(FASTAPI_ENDPOINT, json=documents_in)
        
        if response.status_code == 200:
            logging.info("Documents sent successfully to FastAPI endpoint")
        else:
            logging.error(f"Failed to send documents. Status: {response.status_code}, Response: {response.text}")

    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise