import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from model import ChatModel
from database import Database
from schemas import ChatRequest, ChatResponse, ConversationHistory, Message

model: ChatModel | None = None
db: Database | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, db
    model = ChatModel()
    db = Database()
    yield


app = FastAPI(title="AI Chat", lifespan=lifespan)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    conv_id = req.conversation_id
    if not conv_id:
        conv_id = db.create_conversation()

    history = db.get_history(conv_id)
    history_dicts = [{"role": m["role"], "content": m["content"]} for m in history]

    response_text = model.generate(req.message, history_dicts)

    db.save_message(conv_id, "user", req.message)
    db.save_message(conv_id, "assistant", response_text)

    return ChatResponse(response=response_text, conversation_id=conv_id)


@app.get("/api/history/{conversation_id}", response_model=ConversationHistory)
def history(conversation_id: str):
    rows = db.get_history(conversation_id)
    messages = [Message(role=r["role"], content=r["content"]) for r in rows]
    return ConversationHistory(conversation_id=conversation_id, messages=messages)


@app.delete("/api/conversation/{conversation_id}")
def delete_conversation(conversation_id: str):
    db.delete_conversation(conversation_id)
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse("../frontend/index.html")


app.mount("/static", StaticFiles(directory="../frontend"), name="static")
