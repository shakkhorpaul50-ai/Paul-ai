import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from model import ChatModel, MODEL_PATH as MODEL_FILE
from database import Database
from schemas import ChatRequest, ChatResponse, ConversationHistory, Message

model: ChatModel | None = None
db: Database | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, db
    try:
        print("[lifespan] Loading ChatModel...")
        model = ChatModel()
        print(f"[lifespan] ChatModel loaded: {model.llm}")
    except Exception as e:
        print(f"[lifespan] ChatModel failed: {e}")
        import traceback

        traceback.print_exc()
        # Keep model as None to allow health checks, chat will error gracefully
        model = None
    try:
        print("[lifespan] Connecting Database...")
        db = Database()
        print("[lifespan] Database ready")
    except Exception as e:
        print(f"[lifespan] Database failed: {e}")
        import traceback

        traceback.print_exc()
        db = None
    yield


app = FastAPI(title="AI Chat", lifespan=lifespan)


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded - check MODEL_PATH and GGUF download (see build logs for curl)")
    if db is None:
        raise HTTPException(status_code=503, detail="Database not ready - check DATABASE_URL")
    conv_id = req.conversation_id
    if not conv_id:
        conv_id = db.create_conversation()

    history = db.get_history(conv_id)
    history_dicts = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        response_text = model.generate(req.message, history_dicts)
    except Exception as e:
        print(f"[chat] generate failed: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model generate failed: {e}")

    try:
        db.save_message(conv_id, "user", req.message)
        db.save_message(conv_id, "assistant", response_text)
    except Exception as e:
        print(f"[chat] db save failed: {e}")

    return ChatResponse(response=response_text, conversation_id=conv_id)


@app.get("/health")
def health():
    mpath = os.environ.get("MODEL_PATH", MODEL_FILE)
    ok = model is not None and os.path.exists(mpath) and os.path.getsize(mpath) > 10 * 1024 * 1024
    return {"ok": ok, "model": mpath, "model_exists": os.path.exists(mpath), "model_size_mb": round(os.path.getsize(mpath)/1e6,1) if os.path.exists(mpath) else 0, "db": db is not None and getattr(db, 'url', None) is not None}


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
