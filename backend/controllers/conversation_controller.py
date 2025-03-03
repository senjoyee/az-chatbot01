import logging
from fastapi import HTTPException

from models.schemas import ConversationRequest
from services.agent import run_agent
from config.logging_config import setup_logging

logger = setup_logging()

class ConversationController:
    def __init__(self):
        pass
        
    async def ask_question(self, request: ConversationRequest) -> dict:
        try:
            question = request.question
            conversation = request.conversation
            selected_files = request.files or []
            
            logger.info(f"Received question: {question}")
            logger.info(f"Received conversation: {conversation}")
            logger.info(f"Received selected files: {selected_files}")
            
            result = await run_agent(question, conversation.conversation, selected_files)
            answer = result.get("response")
            return {"answer": answer, "status": "success"}
        except Exception as e:
            logger.error(f"Conversation error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))