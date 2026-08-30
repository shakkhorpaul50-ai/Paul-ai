from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


class Message(BaseModel):
    role: str
    content: str
    created_at: Optional[datetime] = None


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Message]
