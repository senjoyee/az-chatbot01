"""
Conversation controller using native Azure AI Search SDK.
This replaces the langchain-based controller with direct SDK calls.
"""

import logging
from fastapi import HTTPException

from models.schemas import ConversationRequest
from services.agent_native import run_agent_native
from config.logging_config import setup_logging

logger = setup_logging()

class ConversationControllerNative:
    """
    Conversation controller using native Azure AI Search SDK.
    Handles conversation requests and routes them to the native agent.
    """
    
    def __init__(self):
        pass
        
    async def ask_question(self, request: ConversationRequest) -> dict:
        """
        Process a question from the user and return an answer.
        
        Args:
            request: The conversation request containing the question, 
                    conversation history, and selected files
                    
        Returns:
            dict: The answer and status
        """
        try:
            question = request.question
            conversation = request.conversation
            selected_files = request.files or []
            
            logger.info(f"Received question: {question}")
            logger.info(f"Received conversation: {conversation}")
            logger.info(f"Received selected files: {selected_files}")
            
            # Use the native agent implementation
            result = await run_agent_native(question, conversation.conversation, selected_files)
            answer = result.get("response")
            return {"answer": answer, "status": "success"}
        except Exception as e:
            logger.error(f"Conversation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
