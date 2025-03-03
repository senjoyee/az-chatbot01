from fastapi import APIRouter, Depends

from models.schemas import ConversationRequest
from controllers.conversation_controller import ConversationController

router = APIRouter(tags=["Conversation"])

@router.post("/conversation")
async def ask_question(request: ConversationRequest, controller: ConversationController = Depends()):
    return await controller.ask_question(request)