from fastapi import HTTPException
from datetime import datetime
import asyncio
import logging
import json
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.pptx import partition_pptx

from models.schemas import BlobEvent, DocumentIn, FileProcessingStatus, Message, Conversation
from models.enums import ProcessingStatus
from services.document_service import DocumentService
from services.azure_storage import AzureStorageManager
from services.file_processor import FileProcessor
from services.agent import run_agent # Keep if used elsewhere, otherwise can be removed if generate_mindmap also changes
from services.document_processing_service import generate_single_document_summary # Import new service
from config.prompts import MINDMAP_TEMPLATE
from langchain_openai import AzureChatOpenAI # Added for mindmap generation
from config.settings import AZURE_OPENAI_API_KEY_SC, AZURE_OPENAI_ENDPOINT_SC # Added for mindmap generation # SUMMARY_TEMPLATE is now used within the service
# langchain_openai and settings for LLM are no longer directly used here for summary


logger = logging.getLogger(__name__)

class DocumentController:
    def __init__(self):
        self.storage_manager = AzureStorageManager()
        self.file_processor = FileProcessor()
        self.document_service = DocumentService()
    
    async def process_uploaded_files(self, event: BlobEvent):
        file_name = event.file_name
        can_process, error_message = await self.storage_manager.check_and_acquire_processing_lock(
            self.storage_manager.container_name, file_name)
        if not can_process:
            return FileProcessingStatus(
                status=ProcessingStatus.IN_PROGRESS,
                file_name=file_name,
                error_message=error_message
            )
        try:
            # Start the processing task
            asyncio.create_task(self.file_processor.process_file_async(event))
            return FileProcessingStatus(
                status=ProcessingStatus.IN_PROGRESS,
                file_name=file_name,
                start_time=datetime.utcnow()
            )
        except Exception as e:
            await self.storage_manager.update_status(file_name, ProcessingStatus.FAILED, str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def index_documents(self, documents_in):
        return await self.document_service.index_documents(documents_in)
        
    async def summarize_document(self, filename: str):
        """
        Generate a summary for the specified document using the agent's summarization capabilities
        """
        try:
            logger.info(f"Generating summary for document: {filename}\n")
            
            # Download and extract document text
            file_path = await self.storage_manager.download_file(filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.pdf':
                elements = partition_pdf(filename=file_path, strategy='fast', include_metadata=True)
            elif ext in ['.doc', '.docx']:
                elements = partition_docx(filename=file_path)
            elif ext in ['.xlsx', '.xls']:
                elements = partition_xlsx(filename=file_path)
            elif ext in ['.ppt', '.pptx']:
                elements = partition_pptx(filename=file_path)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
            document_content = ' '.join([el.text for el in elements if hasattr(el, 'text') and el.text.strip()])
            
            # Call the document processing service to generate the summary
            summary_result = await generate_single_document_summary(
                document_content=document_content, 
                file_name_for_logging=filename
            )

            if "error" in summary_result:
                logger.error(f"Error from summarization service for {filename}: {summary_result['error']}")
                raise HTTPException(status_code=500, detail=summary_result['error'])
            
            logger.info(f"Successfully generated summary for {filename} via service.")
            return {"summary": summary_result.get("summary")}
        except Exception as e:
            logger.error(f"Error generating summary for {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")
            
    async def generate_mindmap(self, filename: str):
        """
        Generate a mind map visualization data for the specified document
        """
        try:
            logger.info(f"Generating mind map for document: {filename}")
            
            # Download the document to local file and extract text
            file_path = await self.storage_manager.download_file(filename)
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.pdf':
                elements = partition_pdf(filename=file_path, strategy='fast', include_metadata=True)
            elif ext in ['.doc', '.docx']:
                elements = partition_docx(filename=file_path)
            elif ext in ['.xlsx', '.xls']:
                elements = partition_xlsx(filename=file_path)
            elif ext in ['.ppt', '.pptx']:
                elements = partition_pptx(filename=file_path)
            else:
                raise HTTPException(status_code=400, detail=f'Unsupported file type: {ext}')
            document_content = ' '.join([el.text for el in elements if hasattr(el, 'text') and el.text.strip()])
            
            # Create a mind map generation request using the template
            mindmap_prompt = MINDMAP_TEMPLATE.format(document_content=document_content)
            
            # Generate the mind map via direct AzureChatOpenAI call
            llm = AzureChatOpenAI(
                azure_deployment="gpt-4.1-nano",
                openai_api_version="2024-12-01-preview",
                azure_endpoint=AZURE_OPENAI_ENDPOINT_SC,
                api_key=AZURE_OPENAI_API_KEY_SC,
                temperature=0.3
            )
            mindmap_response = await llm.apredict(mindmap_prompt)
            
            # Extract the JSON from the response
            try:
                # Try to find JSON in the response (it might be wrapped in ```json ... ``` blocks)
                json_start = mindmap_response.find('{')
                json_end = mindmap_response.rfind('}')
                
                if json_start >= 0 and json_end >= 0:
                    json_str = mindmap_response[json_start:json_end+1]
                    mindmap_data = json.loads(json_str)
                else:
                    # If no JSON found, try to parse the whole response
                    mindmap_data = json.loads(mindmap_response)
                    
                logger.info(f"Generated mind map for {filename}")
                return {"mindmap": mindmap_data}
            except json.JSONDecodeError as je:
                logger.error(f"Error parsing mind map JSON: {str(je)}")
                raise HTTPException(status_code=500, detail=f"Failed to parse mind map data: {str(je)}")
                
        except Exception as e:
            logger.error(f"Error generating mind map for {filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate mind map: {str(e)}")